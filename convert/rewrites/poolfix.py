"""The A733 NPU's ReduceSum adds with too little precision (a normalization's variance came out at -10 dB on real
data), while its global average pooling is exact to 80-95 dB at the same speed. Every long sum from to4d.py (a row
reshape, then ReduceSum over the 512 and over the rows) becomes GlobalAveragePool times the element count.

  poolfix.py <in.onnx> <out.onnx>      (checks the result against the original with ONNX Runtime)"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
sh = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in shape_inference.infer_shapes(m).graph.value_info}
first = {n.output[0]: n for n in g.node if n.op_type == "ReduceSum" and n.name.endswith("_rows/Sum512")}
nodes, extra, fixed = [], [], 0
for n in g.node:
    if n.name.endswith("_rows/Sum512"):
        continue
    if n.op_type == "ReduceSum" and n.name.endswith("_rows/SumRows") and n.input[0] in first:
        x = first[n.input[0]].input[0]
        count = int(np.prod(sh[x][2:]))
        p = n.name[: -len("/SumRows")]
        extra.append(numpy_helper.from_array(np.array(count, np.float32), p + "_count"))
        nodes += [helper.make_node("GlobalAveragePool", [x], [p + "_avg"], name=p + "/Pool"),
                  helper.make_node("Mul", [p + "_avg", p + "_count"], list(n.output), name=p + "/Count")]
        fixed += 1
        continue
    nodes.append(n)
del g.node[:]
g.node.extend(nodes)
g.initializer.extend(extra)
used = {i for n in g.node for i in n.input}
for t in list(g.initializer):
    if t.name not in used:
        g.initializer.remove(t)
data = m.SerializeToString()

rng = np.random.default_rng(0)
feed = {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1e-3, np.float32) if i.name.startswith("inv") else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 0.5).astype(np.float32))
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {fixed} sums as pooling, max relative difference {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
