"""Kokoro with its heavy middle on the board's NPU. The same voice as the CPU model, cut in stages (manifest.json):
front (text -> features, F0, energy) and head (F0 -> harmonic source) on the CPU, then the decoder and generator
stages, each on the NPU (a compiled network, run by npu_run) or the CPU (ONNX), then tail (spectrum -> audio).

Stage by stage over every piece of every voice: each NPU network is loaded once per request and only one is in
memory at a time; the tensors between stages wait on disk and each is deleted once no later stage needs it, so memory
does not grow with the text. The NPU networks have fixed lengths (manifest "frames", about 5 s of speech, and the
shapes of their inputs): shorter pieces are zero-padded and masked (the result is the same as at their own length),
longer ones are cut at a word boundary first. The NPU slows down when memory runs short (its driver then gets small
pages), so the CPU models are released before the NPU stages run."""
import gc
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from kokoro_onnx.chunker import split_phonemes
from kokoro_onnx.config import SAMPLE_RATE
from kokoro_onnx.trim import trim as trim_audio

ASR, F0, N = "/encoder/MatMul_1_output_0", "/encoder/Reshape_output_0", "/encoder/Reshape_1_output_0"
POST = "/decoder/decoder/generator/conv_post/Conv_output_0"
CUT_PAUSE = 0.1  # seconds between the halves of a piece too long for one pass, as at a comma
NORM_EPS = 1e-5  # Kokoro's instance normalizations' epsilon


def fit(v, shape):
    """v zero-padded (or cut) on its last axis to the length `shape` asks for, then given that shape."""
    v = np.asarray(v, np.float32)
    n = shape[-1]
    if v.shape[-1] < n:
        v = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, n - v.shape[-1])])
    return np.ascontiguousarray(v[..., :n]).reshape(shape)


class NpuKokoro:
    def __init__(self, pkg, runner, voices, threads=2):
        self.pkg, self.runner, self.voices_path = pkg, runner, voices
        man = json.load(open(os.path.join(pkg, "manifest.json")))
        self.frames, self.stages = man["frames"], man["stages"]
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = threads
        self.front = self.k = None
        self._load_front()
        # The longest each named tensor must be for any stage (the generator's last stages run on whole rows of 512).
        self.shape = {}
        for s in self.stages:
            for name, shape in s["inputs"]:
                if name not in self.shape or shape[-1] > self.shape[name][-1]:
                    self.shape[name] = shape

    def _session(self, f):
        return ort.InferenceSession(os.path.join(self.pkg, f), sess_options=self.opts, providers=["CPUExecutionProvider"])

    def _load_front(self):
        if self.front is None:
            self.front = self._session("front.onnx")
            # Kokoro's own text side (phonemes, tokens, voice styles, pauses); its model is never run here.
            self.k = Kokoro.from_session(self.front, self.voices_path)

    def _release_front(self):
        self.front = None
        if self.k is not None:
            self.k.sess = None
        gc.collect()

    def get_voices(self):
        self._load_front()
        return self.k.get_voices()

    def _units(self, text, voice, speed, lang):
        """The text's batches as units that fit the NPU length: [(style, asr, f0, n, pause after)]."""
        style_of, phonemes, batches = self.k._prepare(text, voice, speed, lang, False, 0.25, 0.1)
        out, todo = [], list(batches)
        while todo:
            ph, pause = todo.pop(0)
            tokens = self.k.tokenizer.tokenize(ph)
            if not tokens:
                continue
            style = np.asarray(self.k._style_for(style_of, len(tokens)), np.float32).reshape(1, 256)
            asr, f0, n = self.front.run([ASR, F0, N], {"tokens": np.array([[0, *tokens, 0]], np.int64), "style": style,
                                                      "speed": np.array([speed], np.float32)})
            if asr.shape[-1] > self.frames:
                halves = split_phonemes(ph, max(1, len(ph) // 2 + 1))
                if len(halves) < 2:
                    raise ValueError(f"cannot shorten a piece of {asr.shape[-1]} frames")
                todo[:0] = [(h, CUT_PAUSE) for h in halves[:-1]] + [(halves[-1], pause)]
                continue
            out.append((style, asr, f0, n, pause))
        return out

    def speak(self, pieces, voices, speed=1.0, lang="es", work=None, progress=None):
        """{voice: [audio of each piece]} (float32, 24 kHz), as Kokoro.create would make each one.
        progress(stage_index, stage_count) is called as the stages go."""
        work = tempfile.mkdtemp(prefix="wa-npu-", dir=work)
        try:
            return self._speak(pieces, list(voices), speed, lang, work, progress or (lambda i, n: None))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _speak(self, pieces, voices, speed, lang, work, progress):
        FR, steps = self.frames, len(self.stages) + 2
        path = lambda u, name: os.path.join(work, f"u{u}", name.strip("/").replace("/", "_") + ".f32")
        self._load_front()
        head = self._session("head.onnx")
        units = []  # (voice, piece index, frames, pause)
        for v in voices:
            for pi, text in enumerate(pieces):
                for style, asr, f0, n, pause in self._units(text, v, speed, lang):
                    u = len(units)
                    os.makedirs(os.path.join(work, f"u{u}"))
                    fr = asr.shape[-1]
                    har = head.run(None, {F0: f0})[0]
                    t = {"asr": asr, "f0": f0, "n": n, "style": style, "har": har}
                    for k, valid in (("f1", fr), ("f2", 2 * fr), ("f20", 20 * fr), ("s", 120 * fr + 1)):
                        mk = np.zeros((1, 1, valid), np.float32) + 1
                        t["mask_" + k], t["inv_" + k] = mk, np.full((1, 1, 1), 1.0 / valid, np.float32)
                        # The normalizations' epsilon times the length: added to the sums while they are large (the
                        # NPU loses a small constant added to small values).
                        t["epsn_" + k] = np.full((1, 1, 1), NORM_EPS * valid, np.float32)
                    for name, val in t.items():
                        if name in self.shape:
                            fit(val, self.shape[name]).tofile(path(u, name))
                    units.append((v, pi, fr, pause))
        del head
        self._release_front()
        progress(1, steps)
        for si, s in enumerate(self.stages):
            if s["engine"] == "npu":
                jobs = os.path.join(work, s["name"] + ".jobs")
                with open(jobs, "w") as f:
                    for u in range(len(units)):
                        f.write(" ".join([path(u, n) for n, _ in s["inputs"]] + ["--"] + [path(u, n) for n, _ in s["outputs"]]) + "\n")
                r = subprocess.run([self.runner, os.path.join(self.pkg, s["file"]), "@" + jobs], capture_output=True, text=True, timeout=900)
                if r.returncode != 0:
                    raise RuntimeError(f"npu {s['name']}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
            else:
                sess = self._session(s["file"])
                for u in range(len(units)):
                    feed = {n: np.fromfile(path(u, n), np.float32).reshape(shape) for n, shape in s["inputs"]}
                    for (n, _), val in zip(s["outputs"], sess.run(None, feed)):
                        np.ascontiguousarray(val, np.float32).tofile(path(u, n))
                del sess
            # What no later stage reads is deleted now: its pages leave the cache, and memory stays free for the NPU.
            later = {n for t in self.stages[si + 1:] for n, _ in t["inputs"]} | {"post"}
            for n, _ in s["inputs"]:
                if n not in later:
                    for u in range(len(units)):
                        try:
                            os.remove(path(u, n))
                        except FileNotFoundError:
                            pass
            progress(si + 2, steps)
        tail = self._session("tail.onnx")
        out = {v: [[] for _ in pieces] for v in voices}
        for u, (v, pi, fr, pause) in enumerate(units):
            post = np.fromfile(path(u, "post"), np.float32).reshape(1, 22, -1)[..., :120 * fr + 1]
            a = np.asarray(tail.run(None, {POST: np.ascontiguousarray(post)})[0], np.float32).ravel()
            a = trim_audio(a)[0]
            if pause:
                a = np.concatenate([a, np.zeros(int(pause * SAMPLE_RATE), np.float32)])
            out[v][pi].append(a)
        progress(steps, steps)
        return {v: [np.concatenate(a) if a else np.zeros(0, np.float32) for a in out[v]] for v in voices}
