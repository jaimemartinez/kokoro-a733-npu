"""Step 5 (run in the work directory): cuts the generator (parts/gen) into its four stages - first upsampling + noise
branch, resblocks 0-2, second upsampling + noise branch, resblocks 3-5 + conv_post - as parts/g1..g4, each with the
tensors crossing its borders as inputs/outputs, a calibration set captured from the whole generator, the first
sentence's inputs (in*.f32) and exact outputs (ref_*.npy) for checking the NPU result.

  python gensplit.py"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils

SRC = "parts/gen/middle.onnx"
ROWS = None  # calibration sentences per stage (None: all of them)

m = onnx.load(SRC)
nodes = list(m.graph.node)
order = [i.name for i in m.graph.input]
inits = {i.name for i in m.graph.initializer}
final = [o.name for o in m.graph.output]

STAGES = [("/ups.0/", "/noise_convs.0/", "/noise_res.0/"), ("/resblocks.0/", "/resblocks.1/", "/resblocks.2/"),
          ("/ups.1/", "/noise_convs.1/", "/noise_res.1/"), ("/resblocks.3/", "/resblocks.4/", "/resblocks.5/", "/conv_post/")]
# Each node goes to its module's stage; unnamed glue goes to the latest stage that feeds it.
maker = {o: k for k, n in enumerate(nodes) for o in n.output}
stage = {}
for k, n in enumerate(nodes):
    named = next((s for s, marks in enumerate(STAGES) if any(mk in n.name for mk in marks)), None)
    fed = max((stage[maker[i]] for i in n.input if i in maker), default=0)
    stage[k] = max(fed, named if named is not None else 0)
    if named is not None and named < fed:
        print("note:", n.name, "named stage", named, "fed from", fed)
spans = [[n for k, n in enumerate(nodes) if stage[k] == s] for s in range(len(STAGES))]

pieces = []
for k, span in enumerate(spans):
    own = {o for n in span for o in n.output}
    ins = []
    for n in span:
        for i in n.input:
            if i and i not in own and i not in inits and i not in ins:
                ins.append(i)
    later = {i for s in spans[k + 1:] for n in s for i in n.input}
    outs = [o for n in span for o in n.output if o in later or o in final]
    pieces.append((ins, outs))
    print(f"g{k + 1}: {len(span)} nodes, ins {len(ins)}, outs {len(outs)}: {outs[:4]}")

cross = {t for ins, _ in pieces for t in ins if t not in order}
probe = onnx.load(SRC)
for t in cross:
    probe.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]

rows = [r.split() for r in open("parts/gen/dataset.txt").read().split("\n") if r.strip()][:ROWS]
lines = [[] for _ in pieces]
for r, files in enumerate(rows):
    feed = {k: np.load(os.path.join("parts/gen", f)) for k, f in zip(order, files)}
    vals = {**feed, **dict(zip(names, sess.run(None, feed)))}
    for k, (ins, _) in enumerate(pieces):
        d = f"parts/g{k + 1}"
        os.makedirs(os.path.join(d, "calib"), exist_ok=True)
        row = []
        for j, t in enumerate(ins):
            fn = f"calib/r{r}_{j}.npy"
            np.save(os.path.join(d, fn), vals[t].astype(np.float32))
            row.append(fn)
        lines[k].append(" ".join(row))
        if r == 0:
            # Reference outputs of sentence 0, to check the NPU result of each piece.
            for j, t in enumerate(pieces[k][1]):
                ref = vals[t] if t in vals else None
                if ref is not None:
                    np.save(os.path.join(d, f"ref_{j}.npy"), ref.astype(np.float32))
            for j, t in enumerate(ins):
                vals[t].astype(np.float32).tofile(os.path.join(d, f"in{j:02d}.f32"))

for k, (ins, outs) in enumerate(pieces):
    d = f"parts/g{k + 1}"
    utils.extract_model(SRC, os.path.join(d, "middle.onnx"), ins, outs, check_model=False)
    open(os.path.join(d, "dataset.txt"), "w").write("\n".join(lines[k]) + "\n")
