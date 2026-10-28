"""Kokoro for the NPU, step 4 (modern Python): the masked middle at an opset ACUITY reads, and its calibration set
(every captured sentence, padded to the fixed length, one .npy per input, listed in dataset.txt).

  python prep.py CUTDIR FRAMES OUTDIR"""
import glob, os, sys
import numpy as np
import onnx
from onnx import version_converter
import onnxruntime as ort

cut, FR, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(os.path.join(out, "calib"), exist_ok=True)
m = onnx.load(os.path.join(cut, f"middle_{FR}_masked.onnx"))
try:
    m17 = version_converter.convert_version(m, 17)
except Exception as e:  # the converter cannot always go down; then keep 20 and let ACUITY say
    print("opset conversion failed:", e)
    m17 = m
m17.ir_version = 8
onnx.checker.check_model(m17)
# Short, plain input names (ACUITY uses them in its files).
rename = {"/encoder/MatMul_1_output_0": "asr", "/encoder/Reshape_output_0": "f0", "/encoder/Reshape_1_output_0": "n",
          "/decoder/decoder/generator/Concat_3_output_0": "har", "/decoder/decoder/generator/conv_post/Conv_output_0": "post"}
for node in m17.graph.node:
    for i, x in enumerate(node.input):
        if x in rename:
            node.input[i] = rename[x]
    for i, x in enumerate(node.output):
        if x in rename:
            node.output[i] = rename[x]
for v in list(m17.graph.input) + list(m17.graph.output):
    if v.name in rename:
        v.name = rename[v.name]
path = os.path.join(out, "middle.onnx")
onnx.save(m17, path)
order = [i.name for i in m17.graph.input]
print("opset", next(o.version for o in m17.opset_import if o.domain in ("", "ai.onnx")), "inputs", order)

head = ort.InferenceSession(os.path.join(cut, "head.onnx"), providers=["CPUExecutionProvider"])
lines = []
for f in sorted(glob.glob(os.path.join(cut, "run_*.npz"))):
    r = np.load(f)
    fr = r["MatMul_1_output_0"].shape[-1]
    if fr > FR:
        continue
    pad = FR - fr
    feed = {
        "asr": np.pad(r["MatMul_1_output_0"], ((0, 0), (0, 0), (0, pad))),
        "f0": np.pad(r["Reshape_output_0"], ((0, 0), (0, 2 * pad))),
        "n": np.pad(r["Reshape_1_output_0"], ((0, 0), (0, 2 * pad))),
        "style": r["style"],
    }
    feed["har"] = np.pad(head.run(None, {"/encoder/Reshape_output_0": r["Reshape_output_0"]})[0], ((0, 0), (0, 0), (0, 120 * pad)))
    for L, (t, valid) in {FR: ("f1", fr), 2 * FR: ("f2", 2 * fr), 20 * FR: ("f20", 20 * fr), 120 * FR + 1: ("s", 120 * fr + 1)}.items():
        mk = np.zeros((1, 1, L), np.float32)
        mk[..., :valid] = 1
        feed[f"mask_{t}"], feed[f"inv_{t}"] = mk, np.full((1, 1, 1), 1.0 / valid, np.float32)
    tag = os.path.basename(f)[4:-4]
    files = []
    for name in order:
        p = os.path.join("calib", f"{tag}_{name}.npy")
        np.save(os.path.join(out, p), feed[name].astype(np.float32))
        files.append(p)
    lines.append(" ".join(files))
open(os.path.join(out, "dataset.txt"), "w").write("\n".join(lines) + "\n")
print("calibration sentences:", len(lines))
