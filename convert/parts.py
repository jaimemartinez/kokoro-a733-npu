"""Step 4 (run in the work directory, after prep.py wrote npu/): splits the middle in two, parts/dec (decode blocks)
and parts/gen (generator), each with its own model and calibration set captured from the full middle on the same
sentences. The whole middle is too big for one NPU network (the driver gives up after 20 s); gensplit.py cuts the
generator further.

  python parts.py"""
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils

SRC = "npu/middle.onnx"
CUT = ["/decoder/decoder/decode.3/Div_4_output_0", "/decoder/Slice_output_0"]
m = onnx.load(SRC)
order = [i.name for i in m.graph.input]
final = [o.name for o in m.graph.output]

producer = {o: n for n in m.graph.node for o in n.output}
inits = {i.name for i in m.graph.initializer}


def needs(outputs, stop=()):
    """Graph inputs (or cut tensors in `stop`) the given outputs depend on, in a stable order."""
    seen, found, todo = set(), set(), list(outputs)
    while todo:
        t = todo.pop()
        if t in seen or t in inits or not t:
            continue
        seen.add(t)
        if t in stop or t in order:
            found.add(t)
            continue
        todo.extend(producer[t].input)
    return [t for t in list(stop) + order if t in found]


parts = {"dec": (needs(CUT), CUT), "gen": (needs(final, stop=CUT), final)}
for name, (ins, outs) in parts.items():
    d = os.path.join("parts", name)
    os.makedirs(os.path.join(d, "calib"), exist_ok=True)
    utils.extract_model(SRC, os.path.join(d, "middle.onnx"), ins, outs, check_model=False)
    print(name, "inputs", ins, "outputs", outs)

# Calibration: run the full middle with the cut tensors exposed, on every sentence of the original set.
probe = onnx.load(SRC)
for t in CUT:
    probe.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
lines = {k: [] for k in parts}
for row in open("npu/dataset.txt").read().split("\n"):
    files = row.split()
    if not files:
        continue
    tag = os.path.basename(files[0]).rsplit("_asr.npy", 1)[0]
    feed = {k: np.load(os.path.join("npu", f)) for k, f in zip(order, files)}
    res = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
    vals = {**feed, **{t: res[t] for t in CUT}}
    for name, (ins, _) in parts.items():
        row_files = []
        for t in ins:
            fn = f"calib/{tag}_{t.strip('/').replace('/', '_')}.npy"
            np.save(os.path.join("parts", name, fn), vals[t].astype(np.float32))
            row_files.append(fn)
        lines[name].append(" ".join(row_files))
for name in parts:
    open(os.path.join("parts", name, "dataset.txt"), "w").write("\n".join(lines[name]) + "\n")
    print(name, len(lines[name]), "calibration rows")
