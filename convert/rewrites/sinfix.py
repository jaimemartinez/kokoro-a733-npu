"""The A733 NPU's sine goes wrong for large arguments (errors up to 1.7 past |t| ~ 8: the voice gets rough), while on
[-pi/2, pi/2] it is exact to ~60 dB. Kokoro only needs sin(t)^2, which repeats every pi, so every Sin -> Pow(2) here
becomes: r = t - pi * round(t / pi); sin(r) * sin(r). The same value, with the argument the NPU gets right.

  sinfix.py <in.onnx> <out.onnx>      (checks the result against the original with ONNX Runtime)"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
users = {}
for n in g.node:
    for i in n.input:
        users.setdefault(i, []).append(n)
consts = [numpy_helper.from_array(np.array(1 / np.pi, np.float32), "sinfix_inv_pi"),
          numpy_helper.from_array(np.array(np.pi, np.float32), "sinfix_pi")]
nodes, drop, fixed = [], set(), 0
for n in g.node:
    if n.name in drop:
        continue
    if n.op_type == "Sin":
        pw = users.get(n.output[0], [])
        if len(pw) == 1 and pw[0].op_type == "Pow" and pw[0].input[1] in inits and float(inits[pw[0].input[1]]) == 2.0:
            t, p = n.input[0], n.name + "_red"
            nodes += [
                helper.make_node("Mul", [t, "sinfix_inv_pi"], [p + "_u"], name=p + "/Div"),
                helper.make_node("Round", [p + "_u"], [p + "_k"], name=p + "/Round"),
                helper.make_node("Mul", [p + "_k", "sinfix_pi"], [p + "_kp"], name=p + "/Mul"),
                helper.make_node("Sub", [t, p + "_kp"], [p + "_r"], name=p + "/Sub"),
                helper.make_node("Sin", [p + "_r"], [p + "_s"], name=n.name),
                helper.make_node("Mul", [p + "_s", p + "_s"], list(pw[0].output), name=pw[0].name),
            ]
            drop.add(pw[0].name)
            fixed += 1
            continue
    nodes.append(n)
del g.node[:]
g.node.extend(nodes)
g.initializer.extend(consts)
data = m.SerializeToString()

rng = np.random.default_rng(0)
feed = {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1e-3, np.float32) if i.name.startswith("inv") else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 2).astype(np.float32))
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {fixed} sines reduced, max relative difference {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
