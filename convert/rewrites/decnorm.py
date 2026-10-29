"""Each decoder instance norm's input channels brought to about the same size before the norm (exact in float).

The decoder's norms get text features (std ~0.02) next to F0 and energy (std ~7) in one tensor. With one int16 scale
for the whole tensor, the small channels' squared deviations round to zero, their variance is lost, and the norm
blows them up: on the chip the decoder's first block came out at 2-32 dB depending on the sentence. A norm does not
care about the scale of a channel, so x*s (s = 1/std per channel, measured on the calibration set) gives the same
result once its epsilon becomes s^2*eps per channel. Run it after epsfix.py (it moves the epsilon it scales here).

  decnorm.py <in.onnx> <out.onnx>      (the calibration set: the stage's dataset.txt, see stage.py)"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
cal = os.environ.get("KOKORO_STAGE_SRC") or os.path.dirname(os.path.abspath(src))
m = onnx.load(src)
g = m.graph
byname = {n.name: n for n in g.node}
sites = []
for n in g.node:
    if n.name.endswith("/ReduceMean_mask") and "/norm" in n.name:
        p = n.name[: -len("/ReduceMean_mask")]
        sub, epsadd = byname.get(p + "/Sub"), byname.get(p + "/Add_2_epsn/Add")
        if sub is not None and epsadd is not None and sub.input[0] == n.input[0]:
            sites.append((p, n, sub, epsadd))
if not sites:
    sys.exit("no instance norm found (run epsfix.py first)")

# The largest variance each channel shows over the calibration rows (valid frames only): no channel ends up larger
# than about 1 after the scaling.
probe = onnx.load(src)
for _, n, _, _ in sites:
    probe.graph.output.append(helper.make_tensor_value_info(n.input[0], onnx.TensorProto.FLOAT, None))
se = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
# The calibration rows are in the ORIGINAL stage's inputs (stage.py's source); this model may already have more
# (epsfix's epsn_*, from inv_*) or longer ones: built here the way stage.py carries them.
cal_inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx.load(os.path.join(cal, "middle.onnx"), load_external_data=False).graph.input]
cal_names = [nm for nm, _ in cal_inputs]
now_inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]
rows = [r.split() for r in open(os.path.join(cal, "dataset.txt")).read().split("\n") if r.strip()]


def row_feed(r):
    get = lambda nm: np.load(os.path.join(cal, r[cal_names.index(nm)]))
    feed = {}
    for nm, shape in now_inputs:
        if nm.startswith("epsn_"):
            feed[nm] = (1e-5 / get("inv_" + nm[len("epsn_"):])).reshape(shape).astype(np.float32)
            continue
        v = get(nm)
        if v.ndim == 3 and len(shape) == 4:
            v = v.reshape(v.shape[0], v.shape[1], 1, v.shape[2])
        if v.shape[-1] < shape[-1]:
            v = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, shape[-1] - v.shape[-1])])
        feed[nm] = np.ascontiguousarray(v[..., : shape[-1]]).reshape(shape).astype(np.float32)
    return feed


var = [None] * len(sites)
for r in rows:
    feed = row_feed(r)
    fr = int(feed["mask_f1"].sum())
    for k, x in enumerate(se.run([n.input[0] for _, n, _, _ in sites], feed)):
        v = x[0][:, : fr * x.shape[-1] // feed["mask_f1"].shape[-1]]
        cur = ((v - v.mean(1, keepdims=True)) ** 2).mean(1)
        var[k] = cur if var[k] is None else np.maximum(var[k], cur)

for k, (p, n, sub, epsadd) in enumerate(sites):
    s = (1.0 / np.maximum(np.sqrt(var[k]), 1e-3)).astype(np.float32)
    C = s.size
    g.initializer.append(numpy_helper.from_array(s.reshape(1, C, 1), p + "_chs"))
    g.initializer.append(numpy_helper.from_array((s * s).reshape(1, C, 1), p + "_chs2"))
    g.node.append(helper.make_node("Mul", [n.input[0], p + "_chs"], [p + "_xs"], name=p + "/chscale"))
    n.input[0] = p + "_xs"
    sub.input[0] = p + "_xs"
    eps = [i for i in epsadd.input if i.startswith("epsn")][0]
    g.node.append(helper.make_node("Mul", [eps, p + "_chs2"], [p + "_epsc"], name=p + "/epsscale"))
    epsadd.input[list(epsadd.input).index(eps)] = p + "_epsc"

# keep the nodes in topological order
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

feed = row_feed(rows[0])
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {len(sites)} norms with scaled channels, max relative difference {err:.2e}")
if err > 1e-3:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
