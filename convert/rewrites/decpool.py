"""The decoder's sums over time (ReduceSum over the last axis of a [1, C, L] tensor) as GlobalAveragePool times L,
like poolfix.py does for the generator: the chip's ReduceSum is imprecise.

  decpool.py <in.onnx> <out.onnx>      (checks the result against the original with ONNX Runtime)"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inf = shape_inference.infer_shapes(m)
sh = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(inf.graph.value_info) + list(inf.graph.input)}
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
nodes, fixed = [], 0
for n in g.node:
    axes = init.get(n.input[1]) if n.op_type == "ReduceSum" and len(n.input) > 1 else None
    s = sh.get(n.input[0])
    if axes is not None and s and len(s) == 3 and list(np.atleast_1d(axes)) in ([2], [-1]):
        C, L = s[1], s[2]
        p = n.name
        g.initializer.extend([
            numpy_helper.from_array(np.array([1, C, 1, L], np.int64), p + "_4d"),
            numpy_helper.from_array(np.array([1, C, 1], np.int64), p + "_3d"),
            numpy_helper.from_array(np.array(L, np.float32), p + "_count"),
        ])
        nodes += [helper.make_node("Reshape", [n.input[0], p + "_4d"], [p + "_x4"], name=p + "/To4d"),
                  helper.make_node("GlobalAveragePool", [p + "_x4"], [p + "_avg"], name=p + "/Pool"),
                  helper.make_node("Reshape", [p + "_avg", p + "_3d"], [p + "_avg3"], name=p + "/To3d"),
                  helper.make_node("Mul", [p + "_avg3", p + "_count"], list(n.output), name=p + "/Count")]
        fixed += 1
        continue
    nodes.append(n)
del g.node[:]
g.node.extend(nodes)
used = {i for n in g.node for i in n.input}
for t in list(g.initializer):
    if t.name not in used:
        g.initializer.remove(t)
data = m.SerializeToString()

rng = np.random.default_rng(0)
feed = {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1e-3, np.float32) if i.name.startswith(("inv", "epsn")) else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 0.5).astype(np.float32))
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {fixed} sums as pooling, max relative difference {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
