/*
 * npu_run: one run of a network (.nb, VIPLite) on the A733 NPU, float32 in and out.
 *
 *   npu_run model.nb in0.f32 in1.f32 ... -- out0.f32 ...
 *   npu_run model.nb @jobs        (one such "in... -- out..." per line; the network is loaded once)
 *
 * Each input file is the tensor as raw little-endian float32 (in the network's own order); each output is written
 * the same way. Quantized tensors (int16 dynamic fixed point, int8/uint8 affine) are converted here, so the caller
 * only ever sees floats. Prints the tensors' shapes and formats, and the run's time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vip_lite.h"

typedef struct {
    vip_buffer_create_params_t p;
    vip_buffer buf;
    size_t count;
} tensor_t;

static size_t elem_bytes(vip_enum f)
{
    switch (f) {
    case VIP_BUFFER_FORMAT_FP32: case VIP_BUFFER_FORMAT_INT32: case VIP_BUFFER_FORMAT_UINT32: return 4;
    case VIP_BUFFER_FORMAT_FP16: case VIP_BUFFER_FORMAT_INT16: case VIP_BUFFER_FORMAT_UINT16: case VIP_BUFFER_FORMAT_BFP16: return 2;
    default: return 1;
    }
}

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
    if (p->data_format == VIP_BUFFER_FORMAT_INT16) {
        /* the common case, in one tight loop: a multiply and a clamp per value */
        const float s = p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT ? ldexpf(1.0f, p->quant_data.dfp.fixed_point_pos) : 1.0f;
        short *d = dst;
        for (size_t i = 0; i < t->count; i++) {
            float q = src[i] * s;
            q = q > 32767.0f ? 32767.0f : q < -32768.0f ? -32768.0f : q;
            d[i] = (short)lrintf(q);
        }
        return;
    }
    for (size_t i = 0; i < t->count; i++) {
        float v = src[i];
        switch (p->data_format) {
        case VIP_BUFFER_FORMAT_FP32: ((float *)dst)[i] = v; break;
        case VIP_BUFFER_FORMAT_FP16: ((unsigned short *)dst)[i] = float_to_half(v); break;
        case VIP_BUFFER_FORMAT_INT16: {
            float q = p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT ? v * ldexpf(1.0f, p->quant_data.dfp.fixed_point_pos) : v;
            q = roundf(q);
            ((short *)dst)[i] = (short)(q > 32767 ? 32767 : q < -32768 ? -32768 : q);
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
    if (p->data_format == VIP_BUFFER_FORMAT_INT16) {
        const float s = p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT ? ldexpf(1.0f, -p->quant_data.dfp.fixed_point_pos) : 1.0f;
        const short *q = src;
        for (size_t i = 0; i < t->count; i++) dst[i] = (float)q[i] * s;
        return;
    }
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
    char name[64] = {0};
    q(net, i, VIP_BUFFER_PROP_NAME, name);
    t->count = 1;
    for (vip_uint32_t d = 0; d < p->num_of_dims; d++) t->count *= p->sizes[d];
    fprintf(stderr, "%s %u %-12s dims", is_input ? "in " : "out", i, name);
    for (vip_uint32_t d = 0; d < p->num_of_dims; d++) fprintf(stderr, " %u", p->sizes[d]);
    fprintf(stderr, " format %d quant %d fl %d\n", p->data_format, p->quant_format, p->quant_data.dfp.fixed_point_pos);
    p->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    return vip_create_buffer(p, sizeof(*p), &t->buf) == VIP_SUCCESS ? 0 : -1;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* One run: the files named in args[0..n) as "in... -- out...". 0 on success. */
static int run_once(vip_network net, tensor_t *in, vip_uint32_t nin, tensor_t *out, vip_uint32_t nout, char **args, int n)
{
    int sep = -1;
    for (int i = 0; i < n; i++) if (!strcmp(args[i], "--")) sep = i;
    if (sep != (int)nin || n - sep - 1 != (int)nout) { fprintf(stderr, "need %u inputs and %u outputs\n", nin, nout); return 1; }
    double t0 = now();
    for (vip_uint32_t i = 0; i < nin; i++) {
        FILE *f = fopen(args[i], "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", args[i]); return 1; }
        float *v = malloc(in[i].count * sizeof(float));
        size_t got = fread(v, sizeof(float), in[i].count, f);
        fclose(f);
        if (got != in[i].count) { fprintf(stderr, "%s: want %zu floats\n", args[i], in[i].count); free(v); return 1; }
        put(&in[i], vip_map_buffer(in[i].buf), v);
        vip_unmap_buffer(in[i].buf);
        vip_flush_buffer(in[i].buf, VIP_BUFFER_OPER_TYPE_FLUSH);
        free(v);
        vip_set_input(net, i, in[i].buf);
    }
    for (vip_uint32_t i = 0; i < nout; i++) vip_set_output(net, i, out[i].buf);
    double t1 = now();
    if (vip_run_network(net) != VIP_SUCCESS) { fprintf(stderr, "run failed\n"); return 1; }
    double t2 = now();
    for (vip_uint32_t i = 0; i < nout; i++) {
        vip_flush_buffer(out[i].buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
        float *v = malloc(out[i].count * sizeof(float));
        get(&out[i], v, vip_map_buffer(out[i].buf));
        vip_unmap_buffer(out[i].buf);
        FILE *f = fopen(args[sep + 1 + i], "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", args[sep + 1 + i]); free(v); return 1; }
        fwrite(v, sizeof(float), out[i].count, f);
        fclose(f);
        free(v);
    }
    fprintf(stderr, "inputs %.0f ms, run %.0f ms, outputs %.0f ms\n", (t1 - t0) * 1000, (t2 - t1) * 1000, (now() - t2) * 1000);
    return 0;
}

int main(int argc, char **argv)
{
    /* npu_run model.nb in... -- out...   one run
     * npu_run model.nb @jobs              one run per line of jobs ("in... -- out..."), loading the network once */
    if (argc < 3) { fprintf(stderr, "usage: %s model.nb in... -- out... | %s model.nb @jobs\n", argv[0], argv[0]); return 1; }
    int batch = argv[2][0] == '@';
    if (vip_init() != VIP_SUCCESS) { fprintf(stderr, "vip_init failed (is /dev/vipcore there?)\n"); return 1; }
    double t0 = now();
    vip_network net;
    if (vip_create_network(argv[1], 0, VIP_CREATE_NETWORK_FROM_FILE, &net) != VIP_SUCCESS) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    vip_uint32_t nin = 0, nout = 0, pool = 0;
    vip_query_network(net, VIP_NETWORK_PROP_INPUT_COUNT, &nin);
    vip_query_network(net, VIP_NETWORK_PROP_OUTPUT_COUNT, &nout);
    vip_query_network(net, VIP_NETWORK_PROP_MEMORY_POOL_SIZE, &pool);
    fprintf(stderr, "network: %u inputs, %u outputs, memory pool %u bytes\n", nin, nout, pool);
    tensor_t *in = calloc(nin, sizeof(tensor_t)), *out = calloc(nout, sizeof(tensor_t));
    for (vip_uint32_t i = 0; i < nin; i++) if (describe(net, 1, i, &in[i])) { fprintf(stderr, "input %u buffer failed\n", i); return 1; }
    for (vip_uint32_t i = 0; i < nout; i++) if (describe(net, 0, i, &out[i])) { fprintf(stderr, "output %u buffer failed\n", i); return 1; }
    if (vip_prepare_network(net) != VIP_SUCCESS) { fprintf(stderr, "prepare failed\n"); return 1; }
    fprintf(stderr, "load+prepare %.0f ms\n", (now() - t0) * 1000);
    int rc = 0;
    if (!batch) {
        rc = run_once(net, in, nin, out, nout, argv + 2, argc - 2);
    } else {
        FILE *jobs = fopen(argv[2] + 1, "r");
        if (!jobs) { fprintf(stderr, "cannot read %s\n", argv[2] + 1); return 1; }
        char line[16384];
        while (!rc && fgets(line, sizeof line, jobs)) {
            char *tok[512];
            int n = 0;
            for (char *s = strtok(line, " \t\r\n"); s && n < 512; s = strtok(NULL, " \t\r\n")) tok[n++] = s;
            if (n) rc = run_once(net, in, nin, out, nout, tok, n);
        }
        fclose(jobs);
    }
    for (vip_uint32_t i = 0; i < nin; i++) vip_destroy_buffer(in[i].buf);
    for (vip_uint32_t i = 0; i < nout; i++) vip_destroy_buffer(out[i].buf);
    vip_finish_network(net);
    vip_destroy_network(net);
    vip_destroy();
    return rc;
}
