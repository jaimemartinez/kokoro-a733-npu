"""Step 8: the board's speech package from the built stages, for runtime/kokoro_npu.py:

  python package.py <cut dir> <out dir> dec=<dir> g1=<dir> g2=<dir> g3=<dir> g4=<dir>

Each stage <dir> holds its middle.onnx and its compiled wksp/p_nbg_unify/network_binary.nb. The CPU models come from
<cut dir>: front (front.int8.onnx when quantize_front.py made it, else front.onnx), head and tail. manifest.json lists
the stages in order with their inputs and outputs (names and shapes)."""
import json
import os
import shutil
import sys
import onnx
from onnx import shape_inference

cut, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
front = os.path.join(cut, "front.int8.onnx")
shutil.copy(front if os.path.exists(front) else os.path.join(cut, "front.onnx"), os.path.join(out, "front.onnx"))
for f in ("head", "tail"):
    shutil.copy(os.path.join(cut, f"{f}.onnx"), os.path.join(out, f"{f}.onnx"))
stages = []
for arg in sys.argv[3:]:
    name, d = arg.split("=", 1)
    m = shape_inference.infer_shapes(onnx.load(os.path.join(d, "middle.onnx")))
    sh = {v.name: [x.dim_value for x in v.type.tensor_type.shape.dim]
          for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
    nb = os.path.join(d, "wksp", "p_nbg_unify", "network_binary.nb")
    if not os.path.exists(nb):
        sys.exit(f"{name}: no network in {d}")
    shutil.copy(nb, os.path.join(out, f"{name}.nb"))
    stages.append({"name": name, "engine": "npu", "file": f"{name}.nb",
                   "inputs": [(i.name, sh[i.name]) for i in m.graph.input],
                   "outputs": [(o.name, sh[o.name]) for o in m.graph.output]})
json.dump({"frames": 200, "stages": stages}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
for s in stages:
    print(s["name"], os.path.getsize(os.path.join(out, s["file"])) // 1024, "KB", s["outputs"])
