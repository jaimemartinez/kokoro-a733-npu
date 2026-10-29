"""The Snake activations' 1/alpha constants bounded to |1/alpha| <= K (default 32, sign kept).

Snake is x + sin^2(alpha*x)/alpha, with alpha learned per channel. A few channels have alpha close to zero (one in
g3 has 1e-4, so 1/alpha = -10552). The whole [1, C, 1, 1] constant gets one int16 scale, set by that outlier: every
other channel's 1/alpha is then rounded to steps of 0.5 and the Snake's output falls to 18 dB on the chip. For those
few channels sin^2(alpha*x)/alpha ~ alpha*x^2 is nearly nothing, and bounding 1/alpha changes the stage's output by
60-100 dB below the signal (measured on real sentences).

  snakeclamp.py <in.onnx> <out.onnx> [K]"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
K = float(sys.argv[3]) if len(sys.argv) > 3 else 32.0
m = onnx.load(src)
n = 0
for i, t in enumerate(m.graph.initializer):
    if "Reciprocal" in t.name:
        v = numpy_helper.to_array(t)
        c = np.clip(v, -K, K).astype(np.float32)
        n += int((c != v).sum())
        m.graph.initializer[i].CopyFrom(numpy_helper.from_array(c, t.name))
data = m.SerializeToString()

rng = np.random.default_rng(0)
feed = {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1e-3, np.float32) if i.name.startswith(("inv", "epsn")) else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 0.5).astype(np.float32))
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
snr = min(10 * np.log10(np.sum(x.astype(np.float64) ** 2) / max(np.sum((x - y).astype(np.float64) ** 2), 1e-30)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {n} channels bounded at |1/alpha| <= {K:g}; output {snr:.1f} dB from the original")
if snr < 50:
    sys.exit("the bound changes the result too much; not saved")
open(dst, "wb").write(data)
