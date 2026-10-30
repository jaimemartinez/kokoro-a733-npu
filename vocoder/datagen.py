"""Teacher data for the student vocoder: Kokoro v1.0 (float, GPU) on sentences from public-domain Spanish books,
keeping what the student will see (the decoder's output, the generator's style, F0 and energy) and what it must
produce (the audio). One .npz shard per SHARD sentences.

  python datagen.py OUTDIR COUNT [seed]"""
import glob
import os
import random
import re
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from kokoro_onnx import Kokoro

out_dir, COUNT = sys.argv[1], int(sys.argv[2])
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SHARD = 250
DEC, STYLE = "/decoder/decoder/decode.3/Div_4_output_0", "/decoder/Slice_output_0"
F0, N = "/encoder/Reshape_output_0", "/encoder/Reshape_1_output_0"
os.makedirs(out_dir, exist_ok=True)

m = onnx.load("kokoro-v1.0.onnx")
for name in (DEC, STYLE, F0, N):
    m.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
onnx.save(m, "teacher.onnx")
ort.preload_dlls()  # the CUDA libraries that came with torch
sess = ort.InferenceSession("teacher.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("providers", sess.get_providers(), flush=True)
outs = [o.name for o in sess.get_outputs()]
k = Kokoro.from_session(sess, "voices-v1.0.bin")


def sentences():
    out = []
    for f in sorted(glob.glob("texts/*.txt")):
        t = open(f, encoding="utf-8", errors="ignore").read()
        a, b = t.find("*** START"), t.find("*** END")
        t = t[t.find("\n", a) + 1 if a >= 0 else 0: b if b > 0 else None]
        t = re.sub(r"\s+", " ", t.replace("_", ""))
        for s in re.split(r"(?<=[.!?¡¿;:])\s+", t):
            s = s.strip(" -—«»\"'")
            if 15 <= len(s) <= 220 and not re.search(r"[0-9\[\]{}<>@#*=/\\|]", s) and sum(c.isupper() for c in s) < len(s) / 3:
                out.append(s)
    return out


rng = random.Random(SEED)
pool = sentences()
rng.shuffle(pool)
print(f"{len(pool)} candidate sentences", flush=True)
voices = k.get_voices()
spanish = ["ef_dora", "em_alex", "em_santa"]
items, shard, done, t0, frames = [], 0, 0, time.time(), 0
for text in pool:
    if done >= COUNT:
        break
    voice = rng.choice(spanish) if rng.random() < 0.7 else rng.choice(voices)
    speed = round(rng.uniform(0.85, 1.15), 2)
    try:
        style_of, phonemes, batches = k._prepare(text, voice, speed, "es", False, 0.25, 0.1)
    except Exception:
        continue
    for ph, _ in batches:
        tokens = k.tokenizer.tokenize(ph)
        if not tokens:
            continue
        style = np.asarray(k._style_for(style_of, len(tokens)), np.float32).reshape(1, 256)
        r = dict(zip(outs, sess.run(None, {"tokens": np.array([[0, *tokens, 0]], np.int64), "style": style,
                                            "speed": np.array([speed], np.float32)})))
        audio = np.asarray(r[outs[0]], np.float32).ravel()
        dec, f0, n = r[DEC][0], r[F0].reshape(-1), r[N].reshape(-1)
        if dec.shape[-1] != f0.shape[-1] or len(audio) != dec.shape[-1] * 300:
            print("shape mismatch", dec.shape, f0.shape, len(audio), flush=True)
            continue
        items.append({"dec": dec.astype(np.float16), "style": r[STYLE].reshape(-1).astype(np.float32),
                      "f0": f0.astype(np.float32), "n": n.astype(np.float32),
                      "audio": np.clip(audio * 32767, -32768, 32767).astype(np.int16), "voice": voice, "speed": speed})
        frames += dec.shape[-1]
        done += 1
    if len(items) >= SHARD:
        np.savez(os.path.join(out_dir, f"shard{SEED:02d}_{shard:04d}.npz"), items=np.array(items, dtype=object))
        shard += 1
        items = []
        hours = frames * 300 / 24000 / 3600
        print(f"{done} pieces, {hours:.2f} h of audio, {time.time() - t0:.0f} s", flush=True)
if items:
    np.savez(os.path.join(out_dir, f"shard{SEED:02d}_{shard:04d}.npz"), items=np.array(items, dtype=object))
print(f"DONE {done} pieces, {frames * 300 / 24000 / 3600:.2f} h", flush=True)
