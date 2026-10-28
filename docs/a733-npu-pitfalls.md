# Allwinner A733 NPU: pitfalls found while porting Kokoro

Board: Radxa Cubie A7Z (Allwinner A733, VIP9000NANODI_PLUS, 1 GB RAM), Radxa OS (Debian 11), kernel 5.15.147,
VIPLite driver 2.0.3, ACUITY toolkit 6.30.22, int16 dynamic fixed point unless noted.

**The ACUITY simulator does not reproduce any of the accuracy problems below.** For the same compiled stage it
reported 58.6 dB against the float model while the chip returned 23.4 dB. Always compare the chip's output against
ONNX Runtime, and over the valid region only (masked padding holds values nobody reads).

Every number here was measured on the chip with a one- or few-operation network built by the scripts in
[`probes/`](../probes).

## 1. Long 1-D sequences hang the chip or run slowly

The generator works on 128 channels x 24,001 samples. As `[N, C, L]`:

| Operation | `[1,128,24001]` | `[1,128,1,24001]` (height 1) | `[1,128,47,512]` |
|---|---|---|---|
| Add of two tensors | 8 ms | **3 ms** | 4 ms |
| Snake activation `x + sin²(a·x)/a` | **hangs** | **18 ms** | 17 ms |
| ReduceSum over the 24,001 | **hangs** | **hangs** | 2 ms |

A hang shows up as `fail to wait network ... timeout` (the driver gives up after 20 s; the NPU recovers by itself a
minute later). It is not data dependent. **Fix:** rewrite the whole model with every rank-3 tensor as
`[N, C, 1, L]` and convolutions as 2-D with `[1, k]` kernels ([`convert/rewrites/to4d.py`](../convert/rewrites/to4d.py)).
That alone made the generator stages 36-43 % faster and the compiled files 3-4x smaller.

Also, one compiled network should finish well under 20 s: the full Kokoro decoder in one network never finished,
split in five stages each runs in 0.08-1.2 s.

## 2. `ReduceSum` is imprecise; `GlobalAveragePool` is not

Summing 128 x 24,064 values per channel:

| Method | Values ~ N(0, 1) | Squares (like a variance) | Time |
|---|---|---|---|
| `ReduceSum` (512, then the rows) | 41 dB | 74 dB | 2 ms |
| Depthwise conv with all-ones kernels | broken | broken | 27 ms |
| **`GlobalAveragePool` x count** | **83 dB** | **96 dB** | 2 ms |

On the real normalizations (tiny values) `ReduceSum` fell to -10 dB. **Fix:**
[`poolfix.py`](../convert/rewrites/poolfix.py). Make the length a multiple of 512 (pad once, at the input) so the
pooled view is a plain reshape: [`len64.py`](../convert/rewrites/len64.py).

## 3. `Sin` is wrong for large arguments

`sin(t)` for t on a ramp over [-A, A]:

| A | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| Plain `Sin` | 46 dB | 53 dB | **31 dB** (max error 1.75) | **34 dB** |
| `sin²` with t reduced to [-π/2, π/2] | 58 dB | 59 dB | 60 dB | 59 dB |

An error of 1.75 means a wrong sign. **Fix:** Kokoro only needs `sin²`, which repeats every π:
`r = t - π·round(t/π)`, then `sin(r)·sin(r)` ([`sinfix.py`](../convert/rewrites/sinfix.py)). `Round` works on the NPU.

## 4. A small constant added to small values loses precision

Instance norm computes `1/sqrt(var + 1e-5)`. For var in [1e-4, 1e-3]:

| How epsilon is added | Accuracy | Max relative error |
|---|---|---|
| `var + eps` (constant or input) | **32 dB** | 4.7 % |
| epsilon already in the input | 71 dB | 0.07 % |
| `(sum + eps·N) · (1/N)`, added while the numbers are large | **71 dB** | 0.07 % |

`Sqrt` alone (74 dB), `Reciprocal` alone (72 dB) and a long multiply by a per-channel factor (78 dB) are fine.
**Fix:** [`epsfix.py`](../convert/rewrites/epsfix.py), with `eps·N` passed as an input (N is the sentence's real
length, like `1/N`).

## 5. 8-bit quantization ruins a vocoder

| Stage | int16 | int8 (dynamic fixed point) | uint8 (asymmetric affine) |
|---|---|---|---|
| decoder | 42 dB | 1 dB | - |
| g1 | 56 dB | 17 dB | 21 dB |
| g3 | - | 18 dB | - |

int8 is ~2x faster, but the audio is unusable. Stay with int16.

## 6. Speed depends on free memory

The driver maps its memory pool in 64 KB pages when it can get them. With memory fragmented (a running process,
the page cache), it falls back to small pages and the same network runs about 2x slower (g4: 1.16 s -> 1.98 s).
Setting `vm.compaction_proactiveness = 90` (default 20) keeps large free blocks and restores the full speed. Releasing
the CPU models before the NPU stages and deleting intermediate files as soon as they are used also helps.

## 7. Smaller things

- Conversion of 3-D Resize, Pad (reflect), ConvTranspose, LeakyRelu, Gemm all work.
- The export (`pegasus export ... --pack-nbg-unify`) runs the network once in a software simulator; its time grows
  with the network size (one hour for the full decoder, 1-3 minutes per stage after the rewrites).
- Small per-channel tensors (`[1, C, 1, 1]`) are given float16 inputs and outputs by the compiler.
- `vpmdHARDWARE_TIMEOUT` is 20 s and compiled into the kernel module.
