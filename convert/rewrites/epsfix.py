"""The A733 NPU keeps a small constant added to small values with only a few bits: a normalization's
var + 1e-5 came out up to 5 % off (32 dB), and the voice got rough. Added while the numbers are still large it is
exact (71 dB): every  sum * inv_k + eps  (the variance, then its epsilon) here becomes  (sum + epsn_k) * inv_k,
with epsn_k = eps / inv_k = eps * (real length) a new input the caller computes, like inv_k itself.

  epsfix.py <in.onnx> <out.onnx>      (checks the result against the original with ONNX Runtime)"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
maker = {o: n for n in g.node for o in n.output}
inputs = {i.name: i for i in g.input}
drop, replace, eps_of, new_inputs = set(), {}, {}, []
for n in g.node:
    if n.op_type != "Add":
        continue
    c = [i for i in n.input if i in inits and inits[i].size == 1]
    x = [i for i in n.input if i not in inits]
    if len(c) != 1 or len(x) != 1 or not (0 < float(inits[c[0]]) < 1e-3):
        continue
    mul = maker.get(x[0])
    if mul is None or mul.op_type != "Mul":
        continue
    inv = [i for i in mul.input if i.startswith("inv_")]
    if len(inv) != 1:
        continue
    total = [i for i in mul.input if i != inv[0]][0]
    eps = float(inits[c[0]])
    k = inv[0][len("inv_"):]
    if k in eps_of and abs(eps_of[k] - eps) > 1e-12:
        sys.exit(f"different epsilons for {k}")
    if k not in eps_of:
        eps_of[k] = eps
        s = [d.dim_value for d in inputs[inv[0]].type.tensor_type.shape.dim]
        new_inputs.append(helper.make_tensor_value_info("epsn_" + k, onnx.TensorProto.FLOAT, s))
    p = n.name + "_epsn"
    replace[n.name] = [helper.make_node("Add", [total, "epsn_" + k], [p + "_sum"], name=p + "/Add"),
                       helper.make_node("Mul", [p + "_sum", inv[0]], list(n.output), name=p + "/Mul")]
nodes = []
for n in g.node:
    nodes.extend(replace.get(n.name, [n]))
del g.node[:]
g.node.extend(nodes)
g.input.extend(new_inputs)
used = {i for n in g.node for i in n.input}
keep = [n for n in g.node if any(o in used or o in {x.name for x in g.output} for o in n.output)]
while len(keep) != len(g.node):  # the old variance Mul is unused now
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input}
    keep = [n for n in g.node if any(o in used or o in {x.name for x in g.output} for o in n.output)]
for t in list(g.initializer):
    if t.name not in used:
        g.initializer.remove(t)
data = m.SerializeToString()

rng = np.random.default_rng(0)
feed, feed2 = {}, {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1.0 / 3000, np.float32) if i.name.startswith("inv") else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 0.5).astype(np.float32))
feed2 = dict(feed)
for k, e in eps_of.items():
    feed2["epsn_" + k] = (e / feed["inv_" + k]).astype(np.float32)
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed2)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {len(replace)} epsilons moved, new inputs {[(k, e) for k, e in eps_of.items()]}, max relative difference {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
