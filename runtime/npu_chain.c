/*
 * npu_chain: several networks (.nb, VIPLite) run one after the other on the A733 NPU for a batch of jobs, every
 * network loaded once, the tensors between them never written to files nor turned into floats.
 *
 *   npu_chain a.nb b.nb ... @jobs
 *
 * Each line of jobs has one group per network, in order, separated by "|": "in... -- out...". An input is a file of
 * raw little-endian float32 values, or "@k.j": output j of network k (0-based) of this same job. An output is a file
 * to write (float32) or "-" (kept for the networks after it only).
 *
 * By default it goes network by network: one network in memory at a time (the largest one's memory, not the sum of
 * all), every job through it, then the next; the outputs a later network reads wait in RAM as the network wrote them
 * (int16). Prints "step <network> <job> <ms>" for each run.
 * NPU_CHAIN_RESIDENT=1 loads all the networks at once and takes each job through all of them, handing an output's
 * own buffer to the next network: about the same speed, about twice the memory. Prints "run <ms>..." per job.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vip_lite.h"

#define MAXNET 8
#define MAXT 32

typedef struct {
    vip_buffer_create_params_t p;
    vip_buffer buf;
    size_t count;
} tensor_t;

typedef struct {
    vip_network net;
    vip_uint32_t nin, nout;
    tensor_t in[MAXT], out[MAXT];
    double ms;
} net_t;

static float half_to_float(unsigned short h)
{
    unsigned int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}

static unsigned short float_to_half(float f)
{
    unsigned int x;
    memcpy(&x, &f, 4);
    unsigned int s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e == 255) return (unsigned short)(s | 0x7c00 | (m ? 0x200 : 0));
    int ne = (int)e - 127 + 15;
    if (ne >= 31) return (unsigned short)(s | 0x7c00);
    if (ne <= 0) {
        if (ne < -10) return (unsigned short)s;
        m |= 0x800000;
        return (unsigned short)(s | ((m >> (14 - ne)) + ((m >> (13 - ne)) & 1)));
    }
    return (unsigned short)(s | (ne << 10) | ((m >> 13) + ((m >> 12) & 1)));
}

/* float → the tensor's own format */
static void put(const tensor_t *t, void *dst, const float *src)
{
    const vip_buffer_create_params_t *p = &t->p;
    for (size_t i = 0; i < t->count; i++) {
        float v = src[i];
        switch (p->data_format) {
        case VIP_BUFFER_FORMAT_FP32: ((float *)dst)[i] = v; break;
        case VIP_BUFFER_FORMAT_FP16: ((unsigned short *)dst)[i] = float_to_half(v); break;
        case VIP_BUFFER_FORMAT_INT16: {
            float q = p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT ? v * ldexpf(1.0f, p->quant_data.dfp.fixed_point_pos) : v;
            q = q > 32767.0f ? 32767.0f : q < -32768.0f ? -32768.0f : q;
            ((short *)dst)[i] = (short)lrintf(q);
            break;
        }
        case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_UINT8: {
            float q;
            if (p->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) q = roundf(v / p->quant_data.affine.scale) + (float)p->quant_data.affine.zeroPoint;
            else if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) q = roundf(v * ldexpf(1.0f, p->quant_data.dfp.fixed_point_pos));
            else q = roundf(v);
            if (p->data_format == VIP_BUFFER_FORMAT_UINT8) ((unsigned char *)dst)[i] = (unsigned char)(q > 255 ? 255 : q < 0 ? 0 : q);
            else ((signed char *)dst)[i] = (signed char)(q > 127 ? 127 : q < -128 ? -128 : q);
            break;
        }
        default: fprintf(stderr, "unsupported input format %d\n", p->data_format); exit(2);
        }
    }
}

/* the tensor's own format → float */
static void get(const tensor_t *t, float *dst, const void *src)
{
    const vip_buffer_create_params_t *p = &t->p;
    for (size_t i = 0; i < t->count; i++) {
        float v;
        switch (p->data_format) {
        case VIP_BUFFER_FORMAT_FP32: v = ((const float *)src)[i]; break;
        case VIP_BUFFER_FORMAT_FP16: v = half_to_float(((const unsigned short *)src)[i]); break;
        case VIP_BUFFER_FORMAT_INT16:
            v = (float)((const short *)src)[i];
            if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) v = ldexpf(v, -p->quant_data.dfp.fixed_point_pos);
            break;
        case VIP_BUFFER_FORMAT_INT8: case VIP_BUFFER_FORMAT_UINT8: {
            float q = p->data_format == VIP_BUFFER_FORMAT_UINT8 ? (float)((const unsigned char *)src)[i] : (float)((const signed char *)src)[i];
            if (p->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) v = (q - (float)p->quant_data.affine.zeroPoint) * p->quant_data.affine.scale;
            else if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) v = ldexpf(q, -p->quant_data.dfp.fixed_point_pos);
            else v = q;
            break;
        }
        default: fprintf(stderr, "unsupported output format %d\n", p->data_format); exit(2);
        }
        dst[i] = v;
    }
}

static int describe(vip_network net, int is_input, vip_uint32_t i, tensor_t *t)
{
    vip_buffer_create_params_t *p = &t->p;
    memset(p, 0, sizeof(*p));
    vip_status_e (*q)(vip_network, vip_uint32_t, vip_enum, void *) = is_input ? vip_query_input : vip_query_output;
    q(net, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &p->num_of_dims);
    q(net, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p->sizes);
    q(net, i, VIP_BUFFER_PROP_DATA_FORMAT, &p->data_format);
    q(net, i, VIP_BUFFER_PROP_QUANT_FORMAT, &p->quant_format);
    if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) q(net, i, VIP_BUFFER_PROP_FIXED_POINT_POS, &p->quant_data.dfp.fixed_point_pos);
    if (p->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        q(net, i, VIP_BUFFER_PROP_TF_SCALE, &p->quant_data.affine.scale);
        q(net, i, VIP_BUFFER_PROP_TF_ZERO_POINT, &p->quant_data.affine.zeroPoint);
    }
    t->count = 1;
    for (vip_uint32_t d = 0; d < p->num_of_dims; d++) t->count *= p->sizes[d];
    p->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    return vip_create_buffer(p, sizeof(*p), &t->buf) == VIP_SUCCESS ? 0 : -1;
}

/* Same bytes mean the same values: same format, same quantization. */
static int same(const tensor_t *a, const tensor_t *b)
{
    const vip_buffer_create_params_t *p = &a->p, *q = &b->p;
    if (a->count != b->count || p->data_format != q->data_format || p->quant_format != q->quant_format) return 0;
    if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) return p->quant_data.dfp.fixed_point_pos == q->quant_data.dfp.fixed_point_pos;
    if (p->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM)
        return p->quant_data.affine.scale == q->quant_data.affine.scale && p->quant_data.affine.zeroPoint == q->quant_data.affine.zeroPoint;
    return 1;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static float *scratch;
static size_t scratch_n;

static float *floats(size_t n)
{
    if (n > scratch_n) {
        free(scratch);
        scratch = malloc(n * sizeof(float));
        scratch_n = n;
        if (!scratch) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    return scratch;
}

/* One job: tok[0..n) is "in... -- out... | in... -- out... | ...". 0 on success. */
static int run_job(net_t *nets, int nn, char **tok, int n)
{
    int k = 0, at = 0;
    while (k < nn) {
        int end = at;
        while (end < n && strcmp(tok[end], "|")) end++;
        int sep = -1;
        for (int i = at; i < end; i++) if (!strcmp(tok[i], "--")) sep = i;
        net_t *N = &nets[k];
        if (sep - at != (int)N->nin || end - sep - 1 != (int)N->nout) {
            fprintf(stderr, "network %d: need %u inputs and %u outputs\n", k, N->nin, N->nout);
            return 1;
        }
        for (vip_uint32_t i = 0; i < N->nin; i++) {
            const char *a = tok[at + i];
            tensor_t *T = &N->in[i];
            if (a[0] == '@') {
                int pk, pj;
                if (sscanf(a + 1, "%d.%d", &pk, &pj) != 2 || pk < 0 || pk >= k || pj < 0 || pj >= (int)nets[pk].nout) {
                    fprintf(stderr, "bad reference %s\n", a);
                    return 1;
                }
                tensor_t *P = &nets[pk].out[pj];
                if (same(P, T)) {
                    vip_set_input(N->net, i, P->buf);  /* the NPU reads what it wrote: no copy */
                    continue;
                }
                if (P->count != T->count) { fprintf(stderr, "%s: %zu values for an input of %zu\n", a, P->count, T->count); return 1; }
                float *v = floats(P->count);
                vip_flush_buffer(P->buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
                get(P, v, vip_map_buffer(P->buf));
                vip_unmap_buffer(P->buf);
                put(T, vip_map_buffer(T->buf), v);
            } else {
                FILE *f = fopen(a, "rb");
                if (!f) { fprintf(stderr, "cannot read %s\n", a); return 1; }
                float *v = floats(T->count);
                size_t got = fread(v, sizeof(float), T->count, f);
                fclose(f);
                if (got != T->count) { fprintf(stderr, "%s: want %zu floats\n", a, T->count); return 1; }
                put(T, vip_map_buffer(T->buf), v);
            }
            vip_unmap_buffer(T->buf);
            vip_flush_buffer(T->buf, VIP_BUFFER_OPER_TYPE_FLUSH);
            vip_set_input(N->net, i, T->buf);
        }
        for (vip_uint32_t i = 0; i < N->nout; i++) vip_set_output(N->net, i, N->out[i].buf);
        double t0 = now();
        if (vip_run_network(N->net) != VIP_SUCCESS) { fprintf(stderr, "network %d: run failed\n", k); return 1; }
        N->ms = (now() - t0) * 1000;
        for (vip_uint32_t i = 0; i < N->nout; i++) {
            const char *a = tok[sep + 1 + i];
            if (!strcmp(a, "-")) continue;
            tensor_t *T = &N->out[i];
            float *v = floats(T->count);
            vip_flush_buffer(T->buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
            get(T, v, vip_map_buffer(T->buf));
            vip_unmap_buffer(T->buf);
            FILE *f = fopen(a, "wb");
            if (!f) { fprintf(stderr, "cannot write %s\n", a); return 1; }
            fwrite(v, sizeof(float), T->count, f);
            fclose(f);
        }
        k++;
        at = end + 1;
    }
    fprintf(stderr, "run");
    for (int i = 0; i < nn; i++) fprintf(stderr, " %.0f", nets[i].ms);
    fprintf(stderr, " ms\n");
    return 0;
}

/* ---- stage by stage: one network in memory at a time ------------------------------------------------------------
 * The default. Each network is loaded once and every job goes through it before the next one is loaded, so memory
 * holds the largest network only (not the five). What a later network reads is kept between them in RAM as the
 * network wrote it (int16, half the size of float, no conversion when the next one reads the same format). */

typedef struct {
    void *data;              /* the raw tensor (NULL = not kept) */
    size_t bytes, count;
    vip_buffer_create_params_t p;
} kept_t;

static size_t elem_bytes(vip_enum f)
{
    switch (f) {
    case VIP_BUFFER_FORMAT_FP32: case VIP_BUFFER_FORMAT_INT32: case VIP_BUFFER_FORMAT_UINT32: return 4;
    case VIP_BUFFER_FORMAT_FP16: case VIP_BUFFER_FORMAT_INT16: case VIP_BUFFER_FORMAT_UINT16: case VIP_BUFFER_FORMAT_BFP16: return 2;
    default: return 1;
    }
}

typedef struct {
    char *tok[1024];
    int n;
    int at[MAXNET], sep[MAXNET], end[MAXNET]; /* each network's group: tok[at..sep) inputs, tok[sep+1..end) outputs */
} job_t;

static int stage_major(int nn, char **files, const char *jobs_path)
{
    /* the jobs, all read first */
    FILE *f = fopen(jobs_path, "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", jobs_path); return 1; }
    int nj = 0, cap = 16;
    job_t *jobs = calloc(cap, sizeof(job_t));
    static char line[65536];
    while (fgets(line, sizeof line, f)) {
        job_t J;
        memset(&J, 0, sizeof J);
        for (char *s = strtok(line, " \t\r\n"); s && J.n < 1024; s = strtok(NULL, " \t\r\n")) J.tok[J.n++] = strdup(s);
        if (!J.n) continue;
        int k = 0, at = 0;
        while (k < nn && at <= J.n) {
            int end = at;
            while (end < J.n && strcmp(J.tok[end], "|")) end++;
            J.at[k] = at; J.end[k] = end; J.sep[k] = -1;
            for (int i = at; i < end; i++) if (!strcmp(J.tok[i], "--")) J.sep[k] = i;
            if (J.sep[k] < 0) { fprintf(stderr, "job %d, network %d: no --\n", nj, k); return 1; }
            k++;
            at = end + 1;
        }
        if (k != nn) { fprintf(stderr, "job %d: %d groups for %d networks\n", nj, k, nn); return 1; }
        if (nj == cap) jobs = realloc(jobs, (cap *= 2) * sizeof(job_t));
        jobs[nj++] = J;
    }
    fclose(f);
    /* the last network that reads each kept output: freed after it */
    int last[MAXNET][MAXT];
    for (int k = 0; k < nn; k++) for (int o = 0; o < MAXT; o++) last[k][o] = -1;
    for (int j = 0; j < nj; j++)
        for (int k = 0; k < nn; k++)
            for (int i = jobs[j].at[k]; i < jobs[j].sep[k]; i++) {
                int a, b;
                if (jobs[j].tok[i][0] == '@' && sscanf(jobs[j].tok[i] + 1, "%d.%d", &a, &b) == 2 && a >= 0 && a < k && b >= 0 && b < MAXT && k > last[a][b]) last[a][b] = k;
            }
    kept_t *kept = calloc((size_t)nj * nn * MAXT, sizeof(kept_t));
#define KEPT(j, k, o) kept[((size_t)(j) * nn + (k)) * MAXT + (o)]
    int rc = 0;
    for (int k = 0; k < nn && !rc; k++) {
        double t0 = now();
        net_t N;
        memset(&N, 0, sizeof N);
        if (vip_create_network(files[k], 0, VIP_CREATE_NETWORK_FROM_FILE, &N.net) != VIP_SUCCESS) { fprintf(stderr, "cannot load %s\n", files[k]); return 1; }
        vip_query_network(N.net, VIP_NETWORK_PROP_INPUT_COUNT, &N.nin);
        vip_query_network(N.net, VIP_NETWORK_PROP_OUTPUT_COUNT, &N.nout);
        if (N.nin > MAXT || N.nout > MAXT) { fprintf(stderr, "%s: too many tensors\n", files[k]); return 1; }
        for (vip_uint32_t i = 0; i < N.nin; i++) if (describe(N.net, 1, i, &N.in[i])) { fprintf(stderr, "input buffer failed\n"); return 1; }
        for (vip_uint32_t i = 0; i < N.nout; i++) if (describe(N.net, 0, i, &N.out[i])) { fprintf(stderr, "output buffer failed\n"); return 1; }
        if (vip_prepare_network(N.net) != VIP_SUCCESS) { fprintf(stderr, "prepare %s failed\n", files[k]); return 1; }
        double load = (now() - t0) * 1000;
        for (int j = 0; j < nj && !rc; j++) {
            job_t *J = &jobs[j];
            if (J->sep[k] - J->at[k] != (int)N.nin || J->end[k] - J->sep[k] - 1 != (int)N.nout) {
                fprintf(stderr, "network %d: need %u inputs and %u outputs\n", k, N.nin, N.nout);
                rc = 1;
                break;
            }
            for (vip_uint32_t i = 0; i < N.nin && !rc; i++) {
                const char *a = J->tok[J->at[k] + i];
                tensor_t *T = &N.in[i];
                void *dst = vip_map_buffer(T->buf);
                if (a[0] == '@') {
                    int pk, pj;
                    sscanf(a + 1, "%d.%d", &pk, &pj);
                    kept_t *K = (pk >= 0 && pk < k && pj >= 0 && pj < MAXT) ? &KEPT(j, pk, pj) : NULL;
                    if (!K || !K->data) { fprintf(stderr, "%s: not kept\n", a); rc = 1; break; }
                    tensor_t P = { K->p, NULL, K->count };
                    if (same(&P, T)) memcpy(dst, K->data, K->bytes); /* as the NPU wrote it: no conversion */
                    else {
                        if (P.count != T->count) { fprintf(stderr, "%s: %zu values for an input of %zu\n", a, P.count, T->count); rc = 1; break; }
                        float *v = floats(P.count);
                        get(&P, v, K->data);
                        put(T, dst, v);
                    }
                } else {
                    FILE *in = fopen(a, "rb");
                    if (!in) { fprintf(stderr, "cannot read %s\n", a); rc = 1; break; }
                    float *v = floats(T->count);
                    size_t got = fread(v, sizeof(float), T->count, in);
                    fclose(in);
                    if (got != T->count) { fprintf(stderr, "%s: want %zu floats\n", a, T->count); rc = 1; break; }
                    put(T, dst, v);
                }
                vip_unmap_buffer(T->buf);
                vip_flush_buffer(T->buf, VIP_BUFFER_OPER_TYPE_FLUSH);
                vip_set_input(N.net, i, T->buf);
            }
            if (rc) break;
            for (vip_uint32_t i = 0; i < N.nout; i++) vip_set_output(N.net, i, N.out[i].buf);
            double t1 = now();
            if (vip_run_network(N.net) != VIP_SUCCESS) { fprintf(stderr, "network %d: run failed\n", k); rc = 1; break; }
            double ms = (now() - t1) * 1000;
            for (vip_uint32_t i = 0; i < N.nout && !rc; i++) {
                const char *a = J->tok[J->sep[k] + 1 + i];
                tensor_t *T = &N.out[i];
                vip_flush_buffer(T->buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
                void *src = vip_map_buffer(T->buf);
                if (!strcmp(a, "-")) {
                    if (last[k][i] > k) { /* a later network reads it: kept, raw */
                        kept_t *K = &KEPT(j, k, i);
                        K->bytes = T->count * elem_bytes(T->p.data_format);
                        K->count = T->count;
                        K->p = T->p;
                        K->data = malloc(K->bytes);
                        if (!K->data) { fprintf(stderr, "out of memory\n"); rc = 1; }
                        else memcpy(K->data, src, K->bytes);
                    }
                } else {
                    float *v = floats(T->count);
                    get(T, v, src);
                    FILE *o = fopen(a, "wb");
                    if (!o) { fprintf(stderr, "cannot write %s\n", a); rc = 1; }
                    else { fwrite(v, sizeof(float), T->count, o); fclose(o); }
                }
                vip_unmap_buffer(T->buf);
            }
            fprintf(stderr, "step %d %d %.0f ms (load %.0f ms)\n", k, j, ms, j == 0 ? load : 0.0);
        }
        for (vip_uint32_t i = 0; i < N.nin; i++) vip_destroy_buffer(N.in[i].buf);
        for (vip_uint32_t i = 0; i < N.nout; i++) vip_destroy_buffer(N.out[i].buf);
        vip_finish_network(N.net);
        vip_destroy_network(N.net);
        /* what no later network reads any more */
        for (int j = 0; j < nj; j++)
            for (int a = 0; a <= k; a++)
                for (int o = 0; o < MAXT; o++)
                    if (KEPT(j, a, o).data && last[a][o] <= k) { free(KEPT(j, a, o).data); KEPT(j, a, o).data = NULL; }
    }
    for (size_t i = 0; i < (size_t)nj * nn * MAXT; i++) free(kept[i].data);
    free(kept);
    for (int j = 0; j < nj; j++) for (int i = 0; i < jobs[j].n; i++) free(jobs[j].tok[i]);
    free(jobs);
#undef KEPT
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argv[argc - 1][0] != '@' || argc - 2 > MAXNET) {
        fprintf(stderr, "usage: %s a.nb b.nb ... @jobs\n", argv[0]);
        return 1;
    }
    int nn = argc - 2;
    if (vip_init() != VIP_SUCCESS) { fprintf(stderr, "vip_init failed (is /dev/vipcore there?)\n"); return 1; }
    if (!getenv("NPU_CHAIN_RESIDENT")) {
        /* the default: one network in memory at a time */
        int rc = stage_major(nn, argv + 1, argv[argc - 1] + 1);
        vip_destroy();
        return rc;
    }
    /* NPU_CHAIN_RESIDENT=1: all networks in memory at once, each job through all of them (less memory-friendly) */
    double t0 = now();
    net_t *nets = calloc(nn, sizeof(net_t));
    vip_uint32_t pool = 0;
    for (int k = 0; k < nn; k++) {
        net_t *N = &nets[k];
        if (vip_create_network(argv[1 + k], 0, VIP_CREATE_NETWORK_FROM_FILE, &N->net) != VIP_SUCCESS) { fprintf(stderr, "cannot load %s\n", argv[1 + k]); return 1; }
        vip_uint32_t ps = 0;
        vip_query_network(N->net, VIP_NETWORK_PROP_INPUT_COUNT, &N->nin);
        vip_query_network(N->net, VIP_NETWORK_PROP_OUTPUT_COUNT, &N->nout);
        vip_query_network(N->net, VIP_NETWORK_PROP_MEMORY_POOL_SIZE, &ps);
        if (N->nin > MAXT || N->nout > MAXT) { fprintf(stderr, "%s: too many tensors\n", argv[1 + k]); return 1; }
        if (ps > pool) pool = ps;
        for (vip_uint32_t i = 0; i < N->nin; i++) if (describe(N->net, 1, i, &N->in[i])) { fprintf(stderr, "input buffer failed\n"); return 1; }
        for (vip_uint32_t i = 0; i < N->nout; i++) if (describe(N->net, 0, i, &N->out[i])) { fprintf(stderr, "output buffer failed\n"); return 1; }
    }
    /* One pool for all: they never run at the same time. NPU_OWN_POOLS=1 gives each its own (more memory). */
    vip_buffer shared = NULL;
    if (!getenv("NPU_OWN_POOLS") && pool) {
        vip_buffer_create_params_t p;
        memset(&p, 0, sizeof(p));
        p.num_of_dims = 1;
        p.sizes[0] = pool;
        p.data_format = VIP_BUFFER_FORMAT_UINT8;
        p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        if (vip_create_buffer(&p, sizeof(p), &shared) != VIP_SUCCESS) shared = NULL;
    }
    for (int k = 0; k < nn; k++) {
        if (shared && vip_set_network(nets[k].net, VIP_NETWORK_PROP_SET_MEMORY_POOL, shared) != VIP_SUCCESS) {
            fprintf(stderr, "shared memory pool refused\n");
            return 1;
        }
        if (vip_prepare_network(nets[k].net) != VIP_SUCCESS) { fprintf(stderr, "prepare %s failed\n", argv[1 + k]); return 1; }
    }
    fprintf(stderr, "load+prepare %.0f ms, %s pool %u bytes\n", (now() - t0) * 1000, shared ? "shared" : "own", pool);
    FILE *jobs = fopen(argv[argc - 1] + 1, "r");
    if (!jobs) { fprintf(stderr, "cannot read %s\n", argv[argc - 1] + 1); return 1; }
    static char line[65536];
    int rc = 0;
    while (!rc && fgets(line, sizeof line, jobs)) {
        char *tok[1024];
        int n = 0;
        for (char *s = strtok(line, " \t\r\n"); s && n < 1024; s = strtok(NULL, " \t\r\n")) tok[n++] = s;
        if (n) rc = run_job(nets, nn, tok, n);
    }
    fclose(jobs);
    for (int k = 0; k < nn; k++) {
        for (vip_uint32_t i = 0; i < nets[k].nin; i++) vip_destroy_buffer(nets[k].in[i].buf);
        for (vip_uint32_t i = 0; i < nets[k].nout; i++) vip_destroy_buffer(nets[k].out[i].buf);
        vip_finish_network(nets[k].net);
        vip_destroy_network(nets[k].net);
    }
    if (shared) vip_destroy_buffer(shared);
    vip_destroy();
    return rc;
}
