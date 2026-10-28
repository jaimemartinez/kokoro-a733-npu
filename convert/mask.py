"""Kokoro for the NPU, step 3: the fixed middle's instance norms averaged over the REAL frames only. Each
ReduceMean over time (lengths 200 / 400 / 4000 / 24001 for 200 frames) becomes ReduceSum(x · mask_L) · inv_L, with
mask_L [1,1,L] and inv_L [1,1,1] as inputs (made on the CPU from the sentence's own length). Then: how close a padded
sentence gets to the whole model.

  python mask.py OUTDIR FRAMES"""
import glob, os, sys
import numpy as np
import onnx, onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference, TensorProto

out, FR = sys.argv[1], int(sys.argv[2])
src = os.path.join(out, f"middle_{FR}.onnx")
m = shape_inference.infer_shapes(onnx.load(src))
g = m.graph
opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.input) + list(g.output)}
LENGTHS = {FR: "f1", 2 * FR: "f2", 20 * FR: "f20", 120 * FR + 1: "s"}
for L, tag in LENGTHS.items():
    g.input.append(helper.make_tensor_value_info(f"mask_{tag}", TensorProto.FLOAT, [1, 1, L]))
    g.input.append(helper.make_tensor_value_info(f"inv_{tag}", TensorProto.FLOAT, [1, 1, 1]))
axes2 = numpy_helper.from_array(np.array([2], dtype=np.int64), "masked_axes_2")
g.initializer.append(axes2)

new_nodes, replaced, zeroed = [], 0, 0
for n in g.node:
    if n.op_type in ("Conv", "ConvTranspose"):
        # What the whole model sees past a sentence's end is the convolution's own zero padding: keep the padded
        # frames at zero before every convolution, so the real frames near the end get exactly that.
        L = shape.get(n.input[0], [0, 0, 0])[-1]
        tag = LENGTHS.get(L)
        if tag is not None:
            z = n.input[0] + f"_zeroed_{zeroed}"
            new_nodes.append(helper.make_node("Mul", [n.input[0], f"mask_{tag}"], [z], name=n.name + "_zero"))
            n.input[0] = z
            zeroed += 1
        new_nodes.append(n)
        continue
    if n.op_type == "ReduceMean":
        L = shape.get(n.input[0], [0, 0, 0])[-1]
        tag = LENGTHS.get(L)
        if tag is None:
            raise SystemExit(f"unexpected length {L} at {n.name}")
        x, y = n.input[0], n.output[0]
        masked, summed = y + "_masked", y + "_sum"
        new_nodes.append(helper.make_node("Mul", [x, f"mask_{tag}"], [masked], name=n.name + "_mask"))
        if opset >= 13:
            new_nodes.append(helper.make_node("ReduceSum", [masked, "masked_axes_2"], [summed], keepdims=1, name=n.name + "_sum"))
        else:
            new_nodes.append(helper.make_node("ReduceSum", [masked], [summed], axes=[2], keepdims=1, name=n.name + "_sum"))
        new_nodes.append(helper.make_node("Mul", [summed, f"inv_{tag}"], [y], name=n.name + "_inv"))
        replaced += 1
    else:
        new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)
del g.value_info[:]
m.ir_version = min(m.ir_version, 8)
onnx.checker.check_model(m)
dst = os.path.join(out, f"middle_{FR}_masked.onnx")
onnx.save(m, dst)
print("opset", opset, "replaced", replaced, "ReduceMean; zeroed", zeroed, "conv inputs ->", dst)

ASR, F0, N = "/encoder/MatMul_1_output_0", "/encoder/Reshape_output_0", "/encoder/Reshape_1_output_0"
HAR, POST = "/decoder/decoder/generator/Concat_3_output_0", "/decoder/decoder/generator/conv_post/Conv_output_0"
S = {n: ort.InferenceSession(os.path.join(out, f"{n}.onnx"), providers=["CPUExecutionProvider"]) for n in ("head", "tail")}
mid = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])


def masks(fr):
    feed = {}
    for L, tag in LENGTHS.items():
        valid = {"f1": fr, "f2": 2 * fr, "f20": 20 * fr, "s": 120 * fr + 1}[tag]
        mk = np.zeros((1, 1, L), np.float32)
        mk[..., :valid] = 1
        feed[f"mask_{tag}"], feed[f"inv_{tag}"] = mk, np.full((1, 1, 1), 1.0 / valid, np.float32)
    return feed


def snr(ref, x):
    n = min(len(ref), len(x))
    return 10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum((ref[:n] - x[:n]) ** 2), 1e-12))


def spec_db(ref, x):
    """Log-spectral distance (dB), closer to what one hears than the waveform SNR."""
    n = min(len(ref), len(x)) // 512 * 512
    A = np.abs(np.fft.rfft(ref[:n].reshape(-1, 512) * np.hanning(512), axis=1)) + 1e-6
    B = np.abs(np.fft.rfft(x[:n].reshape(-1, 512) * np.hanning(512), axis=1)) + 1e-6
    return float(np.mean(np.sqrt(np.mean((20 * np.log10(A / B)) ** 2, axis=1))))


rows = []
for f in sorted(glob.glob(os.path.join(out, "run_*.npz"))):
    r = np.load(f)
    fr = r["MatMul_1_output_0"].shape[-1]
    if fr > FR:
        continue
    pad = FR - fr
    asr = np.pad(r["MatMul_1_output_0"], ((0, 0), (0, 0), (0, pad)), mode="constant")
    f0 = np.pad(r["Reshape_output_0"], ((0, 0), (0, 2 * pad)), mode="constant")
    nn = np.pad(r["Reshape_1_output_0"], ((0, 0), (0, 2 * pad)), mode="constant")
    har = S["head"].run(None, {F0: f0})[0]
    post = mid.run(None, {ASR: asr, F0: f0, N: nn, HAR: har, "style": r["style"], **masks(fr)})[0]
    audio = S["tail"].run(None, {POST: post[:, :, : 120 * fr + 1]})[0].reshape(-1)
    ref = r["audio"].reshape(-1)
    rows.append((os.path.basename(f), fr, round(float(snr(ref, audio)), 1), round(spec_db(ref, audio), 2)))
for row in rows:
    print("masked, padded to", FR, ":", row, "(SNR dB, spectral distance dB)")
print("median SNR dB:", float(np.median([r[2] for r in rows])), " median spectral distance dB:", float(np.median([r[3] for r in rows])))
