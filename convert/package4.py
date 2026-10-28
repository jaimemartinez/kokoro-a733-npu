"""The device's speech package from built parts: package4.py <out> <stage>=<dir> ... (in order, e.g.
dec=parts4/dec g1=parts4/g1_i8 ...). Each <dir> holds the stage's middle.onnx and its compiled network; the CPU
models front, head and tail come from cut/. manifest.json lists the stages with their inputs and outputs."""
import json
import os
import shutil
import sys
import onnx
from onnx import shape_inference

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for f in ("front", "head", "tail"):
    shutil.copy(f"cut/{f}.onnx", f"{out}/{f}.onnx")
stages = []
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    m = shape_inference.infer_shapes(onnx.load(f"{d}/middle.onnx"))
    sh = {v.name: [x.dim_value for x in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
    nb = f"{d}/wksp/p_nbg_unify/network_binary.nb"
    if not os.path.exists(nb):
        sys.exit(f"{name}: no network in {d}")
    shutil.copy(nb, f"{out}/{name}.nb")
    stages.append({"name": name, "engine": "npu", "file": f"{name}.nb", "from": d,
                   "inputs": [(i.name, sh[i.name]) for i in m.graph.input], "outputs": [(o.name, sh[o.name]) for o in m.graph.output]})
json.dump({"frames": 200, "stages": stages}, open(f"{out}/manifest.json", "w"), indent=1)
for s in stages:
    print(s["name"], s["from"], os.path.getsize(f"{out}/{s['file']}") // 1024, "KB", s["outputs"])
