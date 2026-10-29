# Rebuilding the NPU model (advanced)

You only need this to change the model or reproduce the conversion; `install.sh` downloads a ready-made package.

## What you need

- An **x86-64 Linux PC** (or VM / container) with about 30 GB free and 16 GB of RAM.
- **Allwinner's ACUITY toolkit** for the A733 (version 6.30.x), with the Vivante SDK command tools. Board makers
  hand it out to developers (Radxa: see its NPU development docs). It cannot be redistributed here.
- **Python 3.10+** with `onnx`, `onnxruntime`, `onnxsim` and `kokoro-onnx==0.6.1`. ACUITY ships its own older
  Python; the ONNX steps below run in this newer one.
- `kokoro-v1.0.onnx` (the float model) and `voices-v1.0.bin` from the
  [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).

## Steps

Run them from one work directory; paths below are relative to it. `C` = the `convert/` folder of this repository.

```bash
# 1. Run Kokoro on sample sentences and cut it: front / head / tail for the CPU, the middle for the NPU (200 frames)
python C/split.py kokoro-v1.0.onnx voices-v1.0.bin cut 200
python C/static.py cut 200
python C/mask.py cut 200            # averages over the real frames only (inputs mask_* and inv_*)
python C/quantize_front.py cut      # optional: the front in int8 for the board's CPU

# 2. The middle's calibration set, then its decoder and the generator's four stages
python C/prep.py cut 200 npu
python C/parts.py                   # -> parts/dec, parts/gen
python C/gensplit.py                # -> parts/g1 .. parts/g4 (with test inputs and exact outputs)

# 3. The graph rewrites the chip needs (each one checks itself against ONNX Runtime)
python C/stage.py parts/dec  ready/dec epsfix decpool decnorm
python C/stage.py parts/g1   ready/g1  to4d sinfix poolfix epsfix snakeclamp
python C/stage.py parts/g2   ready/g2  to4d sinfix poolfix epsfix snakeclamp
python C/stage.py parts/g3   ready/g3  to4d len64 sinfix poolfix epsfix snakeclamp resmask
python C/stage.py parts/g4   ready/g4  to4d len64 sinfix poolfix epsfix snakeclamp resmask

# 4. Compile each stage (inside the ACUITY environment)
for s in dec g1 g2 g3 g4; do C/build_stage.sh ready/$s; done

# 5. The package for the board
python C/package.py cut package dec=ready/dec g1=ready/g1 g2=ready/g2 g3=ready/g3 g4=ready/g4
cp voices-v1.0.bin package/ && tar czf kokoro-a733-npu-model.tar.gz -C package .
```

Then on the board: `bash install.sh --package kokoro-a733-npu-model.tar.gz`.

## Checking a stage on the chip

Each `ready/<stage>` has the first sentence's inputs (`in00.f32`, ...) and the exact outputs (`ref_0.npy`). Copy
the stage's `wksp/p_nbg_unify/network_binary.nb` and the inputs to the board and run

```bash
/opt/kokoro-npu/bin/npu_run network_binary.nb in00.f32 in01.f32 ... -- out.f32
```

then compare `out.f32` with `ref_0.npy` (over the real samples only). The stages shipped in v1.1.0 reach, on
sentences not used for calibration: dec 55 dB, g1 56, g2 48, g3 64, g4 42. Check a few sentences of different
lengths: a stage that gives the same poor SNR on every sentence has a broken operation (see the pitfalls, 7-9).

The calibration set should look like what the board will speak: sentences of every length (short ones too), and
several voices. `prep.py` uses the sentences `split.py` ran; add yours there if your texts differ a lot.

## If you convert another model

Read [a733-npu-pitfalls.md](a733-npu-pitfalls.md), and test each operation type of your model on the chip with
small networks first (see [`probes/`](../probes)): the simulator will not show the problems.
