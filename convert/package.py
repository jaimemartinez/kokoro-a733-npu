"""Step 8: the board's speech package from the built stages, for runtime/kokoro_npu.py:

  python package.py <cut dir> <out dir> <ready dir> [<ready dir> ...]

Each <ready dir> holds one compiled length's stages (dec, g1, g2, g3, g4), each with its middle.onnx and its compiled
wksp/p_nbg_unify/network_binary.nb; the length is read from the decoder's input. The longest is the main one (the
longest piece one pass takes); the shorter ones ("sets") are there so a short piece does not pay for a full pass of
padding: the runtime runs each piece at the shortest length it fits. The networks are written as <stage>_<frames>.nb.

The CPU models come from <cut dir>: front (front.int8.onnx when quantize_front.py made it, else front.onnx), head
and tail. manifest.json lists each length's stages in order with their inputs and outputs (names and shapes)."""
import json
import os
import shutil
import sys
import onnx
from onnx import shape_inference

STAGES = ("dec", "g1", "g2", "g3", "g4")
cut, out, readys = sys.argv[1], sys.argv[2], sys.argv[3:]
if not readys:
    sys.exit(__doc__)
os.makedirs(out, exist_ok=True)
front = os.path.join(cut, "front.int8.onnx")
shutil.copy(front if os.path.exists(front) else os.path.join(cut, "front.onnx"), os.path.join(out, "front.onnx"))
for f in ("head", "tail"):
    shutil.copy(os.path.join(cut, f"{f}.onnx"), os.path.join(out, f"{f}.onnx"))


def length(ready):
    """The stages of one compiled length: (frames, [stage])."""
    stages, frames = [], None
    for name in STAGES:
        d = os.path.join(ready, name)
        m = shape_inference.infer_shapes(onnx.load(os.path.join(d, "middle.onnx")))
        sh = {v.name: [x.dim_value for x in v.type.tensor_type.shape.dim]
              for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
        if name == "dec":
            frames = sh["asr"][-1]
        nb = os.path.join(d, "wksp", "p_nbg_unify", "network_binary.nb")
        if not os.path.exists(nb):
            sys.exit(f"{name}: no network in {d}")
        file = f"{name}_{frames}.nb"
        shutil.copy(nb, os.path.join(out, file))
        stages.append({"name": name, "engine": "npu", "file": file,
                       "inputs": [(i.name, sh[i.name]) for i in m.graph.input],
                       "outputs": [(o.name, sh[o.name]) for o in m.graph.output]})
    return frames, stages


built = sorted((length(r) for r in readys), key=lambda fs: -fs[0])
if len({f for f, _ in built}) != len(built):
    sys.exit("two ready dirs of the same length")
(frames, stages), sets = built[0], built[1:]
man = {"frames": frames, "stages": stages}
if sets:
    man["sets"] = [{"frames": f, "stages": st} for f, st in sets]
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
for f, st in built:
    print(f, "frames:", ", ".join(f"{s['name']} {os.path.getsize(os.path.join(out, s['file'])) // 1024} KB" for s in st))
