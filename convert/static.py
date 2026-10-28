"""Kokoro for the NPU, step 2: the four pieces (front: text → features, F0, N; head: F0 → source STFT; middle: the
conv stack, FIXED frames; tail: spectrum → audio), and how much padding a sentence to the fixed length changes its
audio (the middle normalises over time). Writes front/head/tail/middle_<FR>.onnx and prints SNRs.

  python static.py OUTDIR FRAMES"""
import glob, os, sys
import numpy as np
import onnx, onnxruntime as ort
from onnx import helper
from onnxsim import simplify

out, FR = sys.argv[1], int(sys.argv[2])
probe = onnx.load(os.path.join(out, "probe.onnx"))
ASR, F0, N = "/encoder/MatMul_1_output_0", "/encoder/Reshape_output_0", "/encoder/Reshape_1_output_0"
HAR, POST = "/decoder/decoder/generator/Concat_3_output_0", "/decoder/decoder/generator/conv_post/Conv_output_0"

ext = onnx.utils.Extractor(probe)
pieces = {
    "front": ext.extract_model(["tokens", "style", "speed"], [ASR, F0, N]),
    "head": ext.extract_model([F0], [HAR]),
    "tail": ext.extract_model([POST], ["audio"]),
}
for name, mm in pieces.items():
    onnx.save(mm, os.path.join(out, f"{name}.onnx"))
    print(name, len(mm.graph.node), "nodes", [i.name for i in mm.graph.input])

# The middle with fixed shapes, simplified (shape arithmetic folded away).
mid = onnx.load(os.path.join(out, "middle_dynamic.onnx"))
shapes = {ASR: [1, 512, FR], F0: [1, 2 * FR], N: [1, 2 * FR], HAR: [1, 22, 120 * FR + 1], "style": [1, 256]}
shapes[POST] = [1, 22, 120 * FR + 1]
for i in list(mid.graph.input) + list(mid.graph.output):
    dims = i.type.tensor_type.shape.dim
    del dims[:]
    for d in shapes[i.name]:
        dims.add().dim_value = d
# Intermediate annotations carried over without a shape would stop the simplifier: shape inference redoes them.
del mid.graph.value_info[:]
mid_s, ok = simplify(mid)
print("middle simplified:", ok, len(mid.graph.node), "->", len(mid_s.graph.node), "nodes")
# ACUITY's ONNX importer reads IR 8 at most.
mid_s.ir_version = min(mid_s.ir_version, 8)
mpath = os.path.join(out, f"middle_{FR}.onnx")
onnx.save(mid_s, mpath)
ops = {}
for n in mid_s.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("middle ops:", dict(sorted(ops.items(), key=lambda x: -x[1])))

sess = {n: ort.InferenceSession(os.path.join(out, f"{n}.onnx"), providers=["CPUExecutionProvider"]) for n in ("head", "tail")}
mid_sess = ort.InferenceSession(mpath, providers=["CPUExecutionProvider"])


def snr(ref, x):
    n = min(len(ref), len(x))
    ref, x = ref[:n], x[:n]
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-12))


rows = []
for f in sorted(glob.glob(os.path.join(out, "run_*.npz"))):
    r = np.load(f)
    fr = r["MatMul_1_output_0"].shape[-1]
    if fr > FR:
        continue
    pad = FR - fr
    asr = np.pad(r["MatMul_1_output_0"], ((0, 0), (0, 0), (0, pad)), mode="edge")
    f0 = np.pad(r["Reshape_output_0"], ((0, 0), (0, 2 * pad)), mode="edge")
    nn = np.pad(r["Reshape_1_output_0"], ((0, 0), (0, 2 * pad)), mode="edge")
    har = sess["head"].run(None, {F0: f0})[0]
    post = mid_sess.run(None, {ASR: asr, F0: f0, N: nn, HAR: har, "style": r["style"]})[0]
    audio = sess["tail"].run(None, {POST: post[:, :, : 120 * fr + 1]})[0].reshape(-1)
    rows.append((os.path.basename(f), fr, round(float(snr(r["audio"].reshape(-1), audio)), 1)))
for row in rows:
    print("padded to", FR, "vs whole model:", row, "dB")
print("median SNR dB:", float(np.median([r[2] for r in rows])))
