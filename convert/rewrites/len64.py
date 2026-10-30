"""The generator's last stages (4-D, from to4d.py) at a sample length rounded up to a multiple of 512 (24001 -> 24064 =
47 * 512 at 200 frames, 12001 -> 12288 at 100), so every long sum is a plain reshape to [.., rows, 512] with no
copy-and-pad before it. The extra samples are zero in the inputs (har, mask_s) and masked like any padding, so the real
samples come out the same; the caller pads and cuts.

  len64.py <in.onnx> <out.onnx>"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
OLD = next(i.type.tensor_type.shape.dim[-1].dim_value for i in g.input if i.name == "mask_s")  # 120 * frames + 1
NEW = -(-OLD // 512) * 512
FILL = OLD * 5 // 6  # the test sentence's real samples
inits = {t.name: t for t in g.initializer}
for i in g.input:
    d = i.type.tensor_type.shape.dim
    if d and d[-1].dim_value == OLD:
        d[-1].dim_value = NEW
for o in g.output:
    o.type.tensor_type.ClearField("shape")
del g.value_info[:]

keep, rewire = [], {}
for n in g.node:
    n.input[:] = [rewire.get(i, i) for i in n.input]
    if n.op_type == "Pad" and n.name.endswith("_rows/Pad"):
        rewire[n.output[0]] = n.input[0]  # the sum's padding: now nothing to pad
        continue
    if n.op_type == "Pad" and n.input[1] in inits:
        pads = numpy_helper.to_array(inits[n.input[1]]).copy()
        half = len(pads) // 2
        if pads[half - 1] + pads[-1] == 1 and pads.sum() == 1:  # the reflect pad after the last upsampling (one sample)
            pads[-1] += NEW - OLD  # ... now also out to NEW at the end
            inits[n.input[1]].CopyFrom(numpy_helper.from_array(pads, n.input[1]))
    keep.append(n)
for n in keep:
    n.input[:] = [rewire.get(i, i) for i in n.input]
del g.node[:]
g.node.extend(keep)
used = {i for n in g.node for i in n.input}
for t in list(g.initializer):
    if t.name not in used:
        g.initializer.remove(t)
data = m.SerializeToString()

# The same real samples as before, for a sentence filling FILL of them.
rng = np.random.default_rng(0)
old_in = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in onnx.load(src).graph.input}
feed, feed64 = {}, {}
for name, s in old_in.items():
    if name.startswith("inv"):
        v = np.full(s, 1.0 / FILL, np.float32)
    elif name.startswith("mask"):
        v = np.zeros(s, np.float32)
        v[..., : (FILL if s[-1] == OLD else s[-1] * FILL // OLD)] = 1
    else:
        v = (rng.standard_normal(s) * 0.5).astype(np.float32)
        if s[-1] == OLD:
            v[..., FILL:] = 0
    feed[name] = v
    feed64[name] = np.pad(v, [(0, 0)] * (len(s) - 1) + [(0, NEW - OLD)]) if s[-1] == OLD else v
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed64)[0]
va, vb = a[..., :FILL], b[..., :FILL]
err = float(np.abs(va - vb).max() / (np.abs(va).max() + 1e-12))
print(f"{src} -> {dst}: output {list(b.shape)}, {len(g.node)} nodes, max relative difference on the real samples {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
