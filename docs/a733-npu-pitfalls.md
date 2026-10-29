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

## 5. 8-bit quantization ruins a vocoder (before sections 7-9's fixes)

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

## 7. A tensor whose channels have very different sizes

Per-tensor int16 means one scale for all of a tensor's channels. Kokoro's decoder normalizes (instance norm) a
tensor that holds text features (std ~0.02) next to F0 and energy (std ~7): the small channels' squared deviations
round to zero, their variance is lost and the norm multiplies their rounding noise by up to 50. On real sentences
the decoder's first block came out at **2-32 dB** depending on the sentence (better calibration changed nothing).

**Fix:** scale each channel by 1/std (a constant measured on the calibration set) just before the norm, and the
norm's epsilon by s² per channel. A norm does not care about a channel's scale, so the float result is identical
([`decnorm.py`](../convert/rewrites/decnorm.py)). The first norm went from 10 dB (worst sentence -4 dB) to 48 dB,
the whole decoder from 25 dB to 55 dB.

## 8. A per-channel constant with one outlier

The same happens to constants. Snake is `x + sin²(a·x)/a` with `a` learned per channel; a few channels have
`a ≈ 0` (one has 1e-4), so the `1/a` constant goes from 0.8 to -10552. Its single int16 scale then rounds every other
channel's `1/a` to steps of 0.5: that multiply came out at **17.7 dB** while everything before it was at 53 dB.

**Fix:** bound `|1/a|` to 32 ([`snakeclamp.py`](../convert/rewrites/snakeclamp.py)). For those channels
`sin²(a·x)/a ≈ a·x²` is almost nothing; the stage's float output changes by 80-96 dB below the signal. Look at the
range of every constant the compiler quantizes (the `.quantize` file lists min/max per tensor): a max far above the
typical values is a warning.

## 9. Convolution + Add on a long sequence: wrong after ~8022 samples

On `[1, 128, 1, 24064]`, a residual `Add` that takes a `Conv`'s output was correct up to column 8021 and wrong from
there on (**17.5 dB**), while the `Conv` alone was at 68 dB and the `Add` alone is fine. The compiler fuses them and
splits the length in three tiles; the fused kernel is wrong past the first one. Shorter sequences (4000, under the
8191 limit) are not affected. Swapping the Add's inputs does not help.

**Fix:** a multiply by the mask between the Conv and the Add ([`resmask.py`](../convert/rewrites/resmask.py)): it
keeps them apart, costs one cheap pass, and is exact over the real samples. The residual block went to 73 dB, the
whole g3 stage from 25 dB to 64 dB, g4 from 27 dB to 42 dB.

**How these three were found:** every stage was measured on the chip against ONNX Runtime with held-out sentences,
then cut into sub-graphs (the stage up to one tensor, compiled alone) and measured again, halving the distance until
one operation was left. Calibration settings (KL divergence, `--MLE`, 3x more and shorter sentences) changed none of
these numbers: when every sentence gives the same SNR, look for a broken operation, not for the quantization.

## 10. Smaller things

- Conversion of 3-D Resize, Pad (reflect), ConvTranspose, LeakyRelu, Gemm all work.
- The export (`pegasus export ... --pack-nbg-unify`) runs the network once in a software simulator; its time grows
  with the network size (one hour for the full decoder, 1-3 minutes per stage after the rewrites).
- Small per-channel tensors (`[1, C, 1, 1]`) are given float16 inputs and outputs by the compiler.
- `vpmdHARDWARE_TIMEOUT` is 20 s and compiled into the kernel module.
- `Round` and `Sin` on a reduced argument are accurate (68 dB on a toy network).
- Several networks can share one memory pool (`VIP_NETWORK_PROP_SET_MEMORY_POOL`) and an output buffer can be
  handed to the next network as its input when both have the same format and fixed-point position: no copy, no CPU.
  [`runtime/npu_chain.c`](../runtime/npu_chain.c) does this for the five stages; per sentence it saves the loading,
  the files and the int16/float conversions between stages (3.9 s -> 0.8 s for two sentences).
- An export that rewrites many elementwise operations can make the NBG generator (`gen_nbg`) take 5 GB of memory and
  be killed ("Fatal model generation error: 35072"). Give the export machine plenty of RAM or fewer parallel exports.
