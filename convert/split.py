"""Kokoro for the NPU, step 1: run the whole model on Spanish sentences keeping what crosses the generator's two cuts,
and cut out the middle (text features + F0 + N + style + the source's STFT → spectrum before the iSTFT) with a
fixed number of frames. Checks that the cut model alone gives what the whole model gave.

  python split.py kokoro-v1.0.onnx voices-v1.0.bin OUTDIR FRAMES"""
import json, os, sys
import numpy as np
import onnx, onnxruntime as ort
from onnx import helper, numpy_helper
from kokoro_onnx import Kokoro

model_path, voices_path, out, FR = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(out, exist_ok=True)
ASR, F0, N = "/encoder/MatMul_1_output_0", "/encoder/Reshape_output_0", "/encoder/Reshape_1_output_0"
HAR, POST = "/decoder/decoder/generator/Concat_3_output_0", "/decoder/decoder/generator/conv_post/Conv_output_0"
CUTS = [ASR, F0, N, HAR, POST]

m = onnx.load(model_path)
g = m.graph
for name in CUTS:
    g.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
probe_path = os.path.join(out, "probe.onnx")
onnx.save(m, probe_path)
probe = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
k = Kokoro(model_path, voices_path)

SENTENCES = [
    "Hola, gracias por llamar. En este momento no podemos atenderte.",
    "Por favor, deja tu nombre y tu mensaje después del tono.",
    "Te devolveremos la llamada lo antes posible.",
    "El horario de atención es de lunes a viernes, de ocho a cinco.",
    "Que tengas un excelente día.",
    "La reunión es el sábado a las siete de la noche en el salón principal.",
    "¿Necesitas ayuda con algo en especial?",
    "Recuerda que puedes escribirnos en cualquier momento.",
]
VOICES = ["ef_dora", "em_alex", "em_santa"]
runs = []
for vi, voice in enumerate(VOICES):
    for si, text in enumerate(SENTENCES):
        phonemes = k.tokenizer.phonemize(text, lang="es")
        tokens = k.tokenizer.tokenize(phonemes)
        style = np.asarray(k.get_voice_style(voice)[len(tokens)], dtype=np.float32).reshape(1, 256)
        ids = np.array([[0, *tokens, 0]], dtype=np.int64)
        res = dict(zip([o.name for o in probe.get_outputs()],
                       probe.run(None, {"tokens": ids, "style": style, "speed": np.array([1.0], dtype=np.float32)})))
        tag = f"{voice}_{si}"
        np.savez(os.path.join(out, f"run_{tag}.npz"), style=style, audio=res["audio"], **{k2.split("/")[-1]: res[k2] for k2 in CUTS})
        runs.append({"tag": tag, "frames": int(res[ASR].shape[-1]), "audio": int(res["audio"].shape[-1]),
                     "shapes": {k2.split("/")[-1]: list(res[k2].shape) for k2 in CUTS}})
json.dump(runs, open(os.path.join(out, "runs.json"), "w"), indent=1)
print("frames per sentence:", [r["frames"] for r in runs[:len(SENTENCES)]])
print("shapes of the first:", runs[0]["shapes"], "audio", runs[0]["audio"])

# The middle, cut out: what it needs besides the cut tensors.
ext = onnx.utils.Extractor(onnx.load(probe_path))
sub = ext.extract_model([ASR, F0, N, HAR, "style"], [POST])
print("cut model:", len(sub.graph.node), "nodes; inputs", [i.name for i in sub.graph.input])
onnx.save(sub, os.path.join(out, "middle_dynamic.onnx"))
