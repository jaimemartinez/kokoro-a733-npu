"""The A733 NPU runs Kokoro's long 1-D sequences badly as [N, C, L]: the snake activation hangs, elementwise ops are
~2.5x slower. As [N, C, 1, L] (a 2-D image one row high) they run, so this rewrites a part of the model in that layout:
every rank-3 tensor gets a unit axis at 2, convolutions become 2-D with [1, k] kernels, axes move, per-channel
constants are reshaped. Sums over a long row still hang in any layout but [.., A, 512], so those are cut that way.

  to4d.py <in.onnx> <out.onnx>      (checks the result against the original with ONNX Runtime)"""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

SUM_LIMIT, CHUNK = 1024, 512
src, dst = sys.argv[1], sys.argv[2]
m = shape_inference.infer_shapes(onnx.load(src))
g = m.graph
shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
         for v in list(g.value_info) + list(g.input) + list(g.output)}
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
rank = lambda n: len(inits[n].shape) if n in inits else len(shape.get(n, []))
ax4 = lambda a, r=3: (a + r if a < 0 else a) + (1 if (a + r if a < 0 else a) >= 2 else 0)  # rank-3 axis -> rank-4

new_inits, used_as = {}, {}
for n in g.node:
    for j, i in enumerate(n.input):
        used_as.setdefault(i, set()).add((n.op_type, j))


def init4(name):
    """A rank-3 constant used elementwise, as rank 4 (unit axis at 2)."""
    v = inits[name]
    if v.ndim == 3 and name not in new_inits:
        new_inits[name] = v.reshape(v.shape[0], v.shape[1], 1, v.shape[2])
    return name


def const(name, value):
    new_inits[name] = np.asarray(value)
    return name


nodes = []
for n in g.node:
    op, ins = n.op_type, list(n.input)
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    if op in ("Conv", "ConvTranspose"):
        w = inits[ins[1]]
        if w.ndim == 3:
            new_inits[ins[1]] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
            attrs["kernel_shape"] = [1, attrs.get("kernel_shape", [w.shape[2]])[0]]
            p = attrs.get("pads", [0, 0])
            attrs["pads"] = [0, p[0], 0, p[1]]
            attrs["strides"] = [1, attrs.get("strides", [1])[0]]
            attrs["dilations"] = [1, attrs.get("dilations", [1])[0]]
            if "output_padding" in attrs:
                attrs["output_padding"] = [0, attrs["output_padding"][0]]
    elif op in ("Add", "Sub", "Mul", "Div", "Pow"):
        ins = [init4(i) if i in inits else i for i in ins]
    elif op == "ReduceSum" and rank(ins[0]) == 3:
        axes = [ax4(int(a)) for a in inits[ins[1]].ravel()]
        L = shape[ins[0]][2]
        if axes == [3] and L >= SUM_LIMIT:
            C, A = shape[ins[0]][1], -(-L // CHUNK)
            p = n.name + "_rows"
            nodes += [
                helper.make_node("Pad", [ins[0], const(p + "_pads", np.array([0, 0, 0, 0, 0, 0, 0, A * CHUNK - L], np.int64))], [p + "_x"], name=p + "/Pad", mode="constant"),
                helper.make_node("Reshape", [p + "_x", const(p + "_shape", np.array([1, C, A, CHUNK], np.int64))], [p + "_r"], name=p + "/Reshape"),
                helper.make_node("ReduceSum", [p + "_r", const(p + "_a3", np.array([3], np.int64))], [p + "_s"], name=p + "/Sum512", keepdims=1),
                helper.make_node("ReduceSum", [p + "_s", const(p + "_a2", np.array([2], np.int64))], list(n.output), name=p + "/SumRows", keepdims=1),
            ]
            continue
        ins[1] = const(n.name + "_axes4", np.array(axes, np.int64))
    elif op == "Slice" and rank(ins[0]) == 3:
        if len(ins) > 3 and ins[3]:
            ins[3] = const(n.name + "_axes4", np.array([ax4(int(a)) for a in inits[ins[3]].ravel()], np.int64))
    elif op == "Concat" and rank(ins[0]) == 3:
        attrs["axis"] = ax4(attrs["axis"])
    elif op == "Unsqueeze" and rank(n.output[0]) == 3:
        # The rank-3 result's axes moved to rank 4, plus the new unit axis at 2.
        ins[1] = const(n.name + "_axes4", np.array(sorted({ax4(int(a)) for a in inits[ins[1]].ravel()} | {2}), np.int64))
    elif op == "Pad" and rank(ins[0]) == 3:
        pv = inits[ins[1]].ravel()
        ins[1] = const(n.name + "_pads4", np.array([pv[0], pv[1], 0, pv[2], pv[3], pv[4], 0, pv[5]], np.int64))
    elif op == "Resize" and rank(ins[0]) == 3:
        for j in (2, 3):
            if len(ins) > j and ins[j] and ins[j] in inits:
                v = inits[ins[j]].ravel()
                ins[j] = const(f"{n.name}_in{j}_4d", np.insert(v, 2, 1).astype(v.dtype))
    elif op not in ("Gemm", "Sqrt", "Sin", "LeakyRelu", "Slice", "Concat", "Unsqueeze", "Pad", "Resize", "ReduceSum"):
        sys.exit(f"to4d: no rule for {op} ({n.name})")
    node = helper.make_node(op, ins, list(n.output), name=n.name, **attrs)
    nodes.append(node)

for k, v in inits.items():
    new_inits.setdefault(k, v)
graph = helper.make_graph(
    nodes, g.name,
    [helper.make_tensor_value_info(i.name, onnx.TensorProto.FLOAT, (lambda s: s[:2] + [1] + s[2:] if len(s) == 3 else s)(shape[i.name])) for i in g.input],
    [helper.make_tensor_value_info(o.name, onnx.TensorProto.FLOAT, None) for o in g.output],
    [numpy_helper.from_array(v, k) for k, v in new_inits.items() if any(k in nd.input for nd in nodes)],
)
out = helper.make_model(graph, opset_imports=m.opset_import)
out.ir_version = m.ir_version
data = out.SerializeToString()

rng = np.random.default_rng(0)
feed, feed4 = {}, {}
for i in g.input:
    s = shape[i.name]
    v = np.ones(s, np.float32) if len(s) == 3 and s[1] == 1 and not i.name.startswith("inv") else (rng.standard_normal(s) * 0.5).astype(np.float32)
    if i.name.startswith("inv"):
        v = np.full(s, 1.0 / 1000, np.float32)
    feed[i.name] = v
    feed4[i.name] = v.reshape(s[:2] + [1] + s[2:]) if len(s) == 3 else v
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)
b = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(None, feed4)
err = max(float(np.abs(x.ravel() - y.ravel()).max() / (np.abs(x).max() + 1e-12)) for x, y in zip(a, b))
print(f"{src} -> {dst}: {len(nodes)} nodes (was {len(g.node)}), max relative difference {err:.2e}")
if err > 1e-4:
    sys.exit("result changed; not saved")
open(dst, "wb").write(data)
