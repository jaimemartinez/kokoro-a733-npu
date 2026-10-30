"""Fewer full-length elementwise passes in the generator's last stage (exact in float; g4 runs about 3 % faster).

g4 is limited by memory traffic, not arithmetic: each elementwise op reads and writes the whole 24064-sample tensor.
Its AdaIN + Snake + convolution blocks,
  y = (1+g)*(x-m)/s + b ; out = y + sin^2(a*y)/a ; conv(out*mask)
become
  t = xn*(a*(1+g)) + a*b        with xn = (x-m)/s, kept: the parallel resblocks share it
  z = t + sin^2(t)              (= a*out)
  conv(z*mask)                  with the convolution's weights divided by a (per input channel).
The Sin's range reduction (sinfix.py) is kept as it is. Where the normalized input is not shared, only the Snake and
the convolution are folded (folding that norm too made the compiler exceed the NPU's stream-processor limit). Sites
with some |a| < amin are left alone: dividing the weights by a tiny alpha gives them one huge channel, and one int16
scale for the whole weight tensor would round the others away (snakeclamp.py's trap). Run it last.

  fold.py <in.onnx> <out.onnx> [amin=0.1]"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
AMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
m = onnx.load(src)
g = m.graph
init = {t.name: t for t in g.initializer}
arr = lambda n: numpy_helper.to_array(init[n])


def index():
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            if n not in cons.setdefault(i, []):
                cons[i].append(n)
    return prod, cons


def other(node, name):
    return [i for i in node.input if i != name][0]


done = skipped = 0
for S in [n for n in g.node if n.op_type == "Sin"]:
    prod, cons = index()
    R = prod.get(S.input[0])
    T = prod.get(R.input[0]) if R is not None and R.op_type == "Sub" else None
    if T is None or T.op_type != "Mul" or not any(i in init for i in T.input):
        skipped += 1
        continue
    t = T.output[0]
    a_name = next(i for i in T.input if i in init)
    a = arr(a_name)
    if np.abs(a).min() < AMIN:
        # dividing the conv's weights by a tiny alpha would give them one huge channel: one int16 scale for the
        # whole weight tensor would then round the others away (the same trap as snakeclamp's 1/alpha)
        skipped += 1
        continue
    y = other(T, a_name)
    pw = [c for c in cons[S.output[0]] if c.op_type == "Mul" and c.input[0] == c.input[1] == S.output[0]]
    if len(pw) != 1:
        skipped += 1
        continue
    P = pw[0]
    m1 = cons.get(P.output[0], [])
    if len(m1) != 1 or m1[0].op_type != "Mul":
        skipped += 1
        continue
    M1 = m1[0]
    add = cons.get(M1.output[0], [])
    if len(add) != 1 or add[0].op_type != "Add" or y not in add[0].input:
        skipped += 1
        continue
    A = add[0]
    zs = cons.get(A.output[0], [])
    convs = []
    ok = True
    for zm in zs:
        if zm.op_type != "Mul":
            ok = False
            break
        cv = cons.get(zm.output[0], [])
        if len(cv) != 1 or cv[0].op_type != "Conv" or cv[0].input[1] not in init:
            ok = False
            break
        convs.append(cv[0])
    if not ok or not convs:
        skipped += 1
        continue
    if set(c.name for c in cons[y]) != {T.name, A.name}:
        skipped += 1
        continue
    tag = S.name.replace("/", "_")
    new = []
    # the AdaIN in front, when it is one: y = Add(Mul(G, Div(D, Sq)), B)
    Yp = prod.get(y)
    if Yp is not None and Yp.op_type == "Add" and len(cons[y]) == 2:
        M3 = next((prod.get(i) for i in Yp.input if prod.get(i) is not None and prod[i].op_type == "Mul"), None)
        if M3 is not None and len(cons[M3.output[0]]) == 1:
            B = other(Yp, M3.output[0])
            D1 = next((prod.get(i) for i in M3.input if prod.get(i) is not None and prod[i].op_type == "Div"), None)
            if D1 is not None and len(cons[D1.output[0]]) > 1:
                # the normalized input shared with the parallel resblocks: keep it, fold the rest
                G = other(M3, D1.output[0])
                aa = tag + "_a"
                g.initializer.append(numpy_helper.from_array(a.astype(np.float32), aa))
                new += [
                    helper.make_node("Mul", [G, aa], [tag + "_Aa"], name=tag + "/fold/Aa"),
                    helper.make_node("Mul", [B, aa], [tag + "_Ba"], name=tag + "/fold/Ba"),
                    helper.make_node("Mul", [D1.output[0], tag + "_Aa"], [tag + "_xa"], name=tag + "/fold/xa"),
                    helper.make_node("Add", [tag + "_xa", tag + "_Ba"], [t], name=tag + "/fold/t"),
                ]
                for n in (Yp, M3, T):
                    g.node.remove(n)
    # z = t + s^2 in place of y + s^2/a, under the old output name (the masks and convs read it)
    new.append(helper.make_node("Add", [t, P.output[0]], [A.output[0]], name=tag + "/fold/z"))
    g.node.remove(M1)
    g.node.remove(A)
    # the convs now get a times their input: their weights divided by a, per input channel
    for cv in convs:
        w = arr(cv.input[1])
        nw = (w / a.reshape(1, -1, *([1] * (w.ndim - 2)))).astype(np.float32)
        wn = cv.input[1] + "_fold"
        g.initializer.append(numpy_helper.from_array(nw, wn))
        cv.input[1] = wn
    # place the new nodes where the removed ones were: just before the Sin's reduction chain (topological order is
    # restored below)
    g.node.extend(new)
    done += 1

# topological sort
prod = {o: n for n in g.node for o in n.output}
have = {i.name for i in g.input} | set(init) | {t.name for t in g.initializer}
order, seen = [], set()


def visit(n):
    if id(n) in seen:
        return
    seen.add(id(n))
    for i in n.input:
        if i and i in prod and i not in have:
            visit(prod[i])
    order.append(n)


for n in list(g.node):
    visit(n)
del g.node[:]
g.node.extend(order)
# drop initializers nothing reads any more
used = {i for n in g.node for i in n.input}
for t in list(g.initializer):
    if t.name not in used:
        g.initializer.remove(t)
data = m.SerializeToString()

# all-ones masks: nothing is padding, so the result must not change
rng = np.random.default_rng(0)
feed = {}
for i in onnx.load(src).graph.input:
    s = [d.dim_value for d in i.type.tensor_type.shape.dim]
    feed[i.name] = (np.full(s, 1e-3, np.float32) if i.name.startswith(("inv", "epsn")) else
                    np.ones(s, np.float32) if i.name.startswith("mask") else (rng.standard_normal(s) * 0.5).astype(np.float32))
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed)
err = max(float(np.abs(x - y).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: folded {done} snake sites, left {skipped}, max relative difference {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
