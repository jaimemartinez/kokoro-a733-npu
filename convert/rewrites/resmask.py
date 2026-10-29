"""Residual adds after a long convolution: the convolution's output times the mask first (exact over the real samples).

On the chip a Conv followed by an Add on a sequence longer than 8191 is fused and split in tiles of ~8022 samples;
past the first tile the fused result is wrong (a residual block came out at 17.5 dB, exact up to column 8021 and off
from there on). Putting the mask multiply between them keeps them apart (73 dB). Swapping the add's inputs does not
help. The mask only zeroes the padding, which every later sum and convolution masks anyway.

  resmask.py <in.onnx> <out.onnx>      (needs the mask_s input: run it after to4d.py and len64.py)"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inf = shape_inference.infer_shapes(m)
sh = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(inf.graph.value_info) + list(inf.graph.output)}
prod = {o: n for n in g.node for o in n.output}
if "mask_s" not in {i.name for i in g.input}:
    sys.exit("no mask_s input")
new, fixed = [], 0
for n in list(g.node):
    s = sh.get(n.output[0], [])
    if n.op_type != "Add" or len(n.input) != 2 or len(s) != 4 or s[-1] <= 8191:
        continue
    convs = [i for i in n.input if i in prod and prod[i].op_type == "Conv"]
    for c in convs:
        mo = c + "_resmask"
        new.append(helper.make_node("Mul", [c, "mask_s"], [mo], name=f"{n.name}/resmask{len(new)}"))
        n.input[list(n.input).index(c)] = mo
    fixed += bool(convs)
g.node.extend(new)
prod = {o: x for x in g.node for o in x.output}
have = {i.name for i in g.input} | {t.name for t in g.initializer}
out, seen = [], set()


def visit(x):
    if id(x) in seen:
        return
    seen.add(id(x))
    for i in x.input:
        if i and i in prod and i not in have:
            visit(prod[i])
    out.append(x)


for x in list(g.node):
    visit(x)
del g.node[:]
g.node.extend(out)
data = m.SerializeToString()

# all-ones masks: nothing is padding, so the result must not change at all
rng = np.random.default_rng(0)
feed = {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1e-3, np.float32) if i.name.startswith(("inv", "epsn")) else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 0.5).astype(np.float32))
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {fixed} residual adds kept apart from their convolution, max relative difference {err:.2e}")
if err > 1e-5:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
