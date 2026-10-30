"""Shards from datagen.py -> one memory-mapped set: feats [frames, 514] float16 (decoder output, log F0, energy),
audio int16 (300 samples per frame), styles [items, 128], and per item its first frame and frame count.

  python pack.py DATADIR PACKDIR"""
import glob
import json
import os
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "*.npz")))
total = 0
counts = []
for f in files:
    for it in np.load(f, allow_pickle=True)["items"]:
        counts.append(it["dec"].shape[-1])
total = sum(counts)
feats = np.lib.format.open_memmap(os.path.join(dst, "feats.npy"), "w+", np.float16, (total, 514))
audio = np.lib.format.open_memmap(os.path.join(dst, "audio.npy"), "w+", np.int16, (total * 300,))
styles, index, voices = [], [], []
at = 0
for f in files:
    for it in np.load(f, allow_pickle=True)["items"]:
        n = it["dec"].shape[-1]
        feats[at:at + n, :512] = it["dec"].T
        feats[at:at + n, 512] = np.log1p(np.maximum(it["f0"], 0)) / 6
        feats[at:at + n, 513] = it["n"] / 10
        audio[at * 300:(at + n) * 300] = it["audio"]
        styles.append(it["style"])
        index.append((at, n))
        voices.append(str(it["voice"]))
        at += n
feats.flush()
audio.flush()
np.save(os.path.join(dst, "styles.npy"), np.array(styles, np.float32))
np.save(os.path.join(dst, "index.npy"), np.array(index, np.int64))
json.dump(voices, open(os.path.join(dst, "voices.json"), "w"))
print(f"{len(index)} items, {total} frames, {total * 300 / 24000 / 3600:.2f} h")
