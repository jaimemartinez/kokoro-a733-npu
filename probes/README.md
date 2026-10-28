# Probes: small networks that test one operation on the chip

Each script writes one or more tiny ONNX models under `toys/<name>/` (with a calibration sample) and test inputs in
`/tmp`, to compile with `convert/build_stage.sh toys/<name>` and run on the board with `npu_run`. Comparing the
output with NumPy shows what the chip really does; the toolkit's simulator does not.

| Script | What it measures | Result on the A733 |
|---|---|---|
| `layouttoys.py` | add, snake, long sum in four tensor layouts, int16 and int8 | 1-D long sequences hang; `[N,C,1,L]` is fastest |
| `snaketoys.py` | the snake activation as 1-D at 24,001 / 12,000 and as 47 x 512 | 1-D hangs at both lengths |
| `sintoy.py` | `Sin` by argument size; `sin²` with the argument reduced | plain 31-53 dB, reduced 58-60 dB |
| `sumtoys.py` | four ways to sum 24,064 values | `GlobalAveragePool` 83-96 dB, `ReduceSum` 41-74 dB |
| `epstoys.py` | where to add a normalization's epsilon | small + small 32 dB, large + large 71 dB |
| `smalltoys.py` | `Sqrt`, `Reciprocal` on 128 per-channel values | 72-75 dB |
| `multoys.py` | long tensor times a per-channel factor | 78 dB |
| `divtoys.py` | `x / sqrt(v + eps)` three ways | 32 dB each: the epsilon, not the division |

Details and numbers: [../docs/a733-npu-pitfalls.md](../docs/a733-npu-pitfalls.md).
