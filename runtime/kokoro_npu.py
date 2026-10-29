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
from kokoro_onnx.config import SAMPLE_RATE
from kokoro_onnx.trim import trim as trim_audio

ASR, F0, N = "/encoder/MatMul_1_output_0", "/encoder/Reshape_output_0", "/encoder/Reshape_1_output_0"
POST = "/decoder/decoder/generator/conv_post/Conv_output_0"
WIN_OVERLAP = 8  # frames two windows of one long piece share (0.2 s), crossfaded
HOP = 600  # audio samples per frame (24 kHz)
CHAIN_BATCH = int(os.environ.get("KOKORO_NPU_BATCH", "4"))  # units per npu_chain run: bounds the RAM between networks
NORM_EPS = 1e-5  # Kokoro's instance normalizations' epsilon


def fast_cpus():
    """The board's fastest cores (the A733's two A76 next to its six A55): the CPU side of a request runs there."""
    best = {}
    for c in range(os.cpu_count() or 1):
        try:
            best[c] = int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read())
        except (OSError, ValueError):
            return None
    top = max(best.values())
    fast = {c for c, f in best.items() if f == top}
    return fast if 0 < len(fast) < len(best) else None


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
        fast = fast_cpus()
        if fast:
            try:
                os.sched_setaffinity(0, fast)
                threads = len(fast)
            except OSError:
                pass
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = threads
        self.front = self.k = None
        self._seen = {}
        self._load_front()
        # Every shape each named tensor is read at (the generator's last stages run on whole rows of 512, the first
        # ones at the audio's own length): one file per shape, since npu_run reads a file's first values and a row
        # padded for a longer stage would shift every later row of a shorter one.
        self.shapes = {}
        for s in self.stages:
            for name, shape in s["inputs"]:
                self.shapes.setdefault(name, [])
                if list(shape) not in self.shapes[name]:
                    self.shapes[name].append(list(shape))

    def _session(self, f):
        return ort.InferenceSession(os.path.join(self.pkg, f), sess_options=self.opts, providers=["CPUExecutionProvider"])

    def _load_front(self):
        if self.front is None:
            self.front = self._session("front.onnx")
            # Kokoro's own text side (phonemes, tokens, voice styles, pauses); its model is never run here.
            self.k = Kokoro.from_session(self.front, self.voices_path)

    def _release_front(self):
        self.front = None
        self._seen = {}
        if self.k is not None:
            self.k.sess = None
        gc.collect()

    def get_voices(self):
        self._load_front()
        return self.k.get_voices()

    def _units(self, text, voice, speed, lang):
        """The text's batches as units that fit the NPU length: [(style, asr, f0, n, pause after)]. Kept for the
        request (packing tries a text before speaking it)."""
        key = (text, voice, speed, lang)
        if key not in self._seen:
            self._seen[key] = self._units_of(text, voice, speed, lang)
        return self._seen[key]

    def _units_of(self, text, voice, speed, lang, probe=False):
        """probe: only whether the text fits as it is (None when a batch is too long), without cutting it."""
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
            F = asr.shape[-1]
            if F > self.frames:
                if probe:
                    return None
                # Longer than one pass: the text side (durations, intonation, the harmonic source) stays the whole
                # piece's; only the decoder runs in overlapping windows, crossfaded back together (no pause and no
                # new intonation in the middle of a sentence).
                O = WIN_OVERLAP
                count = -(-(F - O) // (self.frames - O))
                L = -(-(F + (count - 1) * O) // count)
                gid = object()
                for k in range(count):
                    a, b = k * (L - O), min(k * (L - O) + L, F)
                    if k == count - 1:
                        b = F
                    win = {"gid": gid, "k": k, "of": count, "start": a, "end": b, "f0": f0}
                    out.append((style, asr[..., a:b], f0[..., 2 * a:2 * b], n[..., 2 * a:2 * b], pause if k == count - 1 else 0, win))
                continue
            out.append((style, asr, f0, n, pause, None))
        return out

    def _packed(self, pieces, voice, speed, lang):
        """[(index of the first piece, text)]: consecutive pieces joined while they still fit one pass (Kokoro speaks
        the full stops between them itself). The pieces folded into an earlier one come back empty from speak."""
        out = []
        for pi, text in enumerate(pieces):
            if out:
                joined = out[-1][1] + " " + text
                key = (joined, voice, speed, lang)
                fit = self._seen[key] if key in self._seen else self._units_of(joined, voice, speed, lang, probe=True)
                if fit is not None and len(fit) == 1:
                    self._seen[key] = fit  # speak reuses it
                    out[-1] = (out[-1][0], joined)
                    continue
            out.append((pi, text))
        return out

    def speak(self, pieces, voices, speed=1.0, lang="es", work=None, progress=None):
        """{voice: [audio of each piece]} (float32, 24 kHz), as Kokoro.create would make each one; a piece spoken
        together with the one before it (see _packed) is empty, its audio in that one's.
        progress(stage_index, stage_count) is called as the stages go."""
        work = tempfile.mkdtemp(prefix="kokoro-npu-", dir=work)
        try:
            return self._speak(pieces, list(voices), speed, lang, work, progress or (lambda i, n: None))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _speak(self, pieces, voices, speed, lang, work, progress):
        FR, steps = self.frames, len(self.stages) + 2
        path = lambda u, name: os.path.join(work, f"u{u}", name.strip("/").replace("/", "_") + ".f32")
        # A tensor read at more than one shape: the file for the shape a stage reads it at.
        at = lambda u, name, shape: path(u, name) if len(self.shapes.get(name, [])) < 2 else path(u, f"{name}@{shape[-1]}")
        self._load_front()
        head = self._session("head.onnx")
        units = []  # (voice, piece index, frames, pause, window)
        whole = {}  # a long piece's harmonic source, for its windows
        for v in voices:
            for pi, text in self._packed(pieces, v, speed, lang):
                for style, asr, f0, n, pause, win in self._units(text, v, speed, lang):
                    u = len(units)
                    os.makedirs(os.path.join(work, f"u{u}"))
                    fr = asr.shape[-1]
                    if win is None:
                        har = head.run(None, {F0: f0})[0]
                    else:
                        # the harmonic source of the WHOLE piece, cut: its phase runs on across the windows
                        if win["gid"] not in whole:
                            whole[win["gid"]] = head.run(None, {F0: win["f0"]})[0]
                        har = whole[win["gid"]][..., 120 * win["start"]:120 * win["end"] + 1]
                    t = {"asr": asr, "f0": f0, "n": n, "style": style, "har": har}
                    for k, valid in (("f1", fr), ("f2", 2 * fr), ("f20", 20 * fr), ("s", 120 * fr + 1)):
                        mk = np.zeros((1, 1, valid), np.float32) + 1
                        t["mask_" + k], t["inv_" + k] = mk, np.full((1, 1, 1), 1.0 / valid, np.float32)
                        # The normalizations' epsilon times the length: added to the sums while they are large (the
                        # NPU loses a small constant added to small values).
                        t["epsn_" + k] = np.full((1, 1, 1), NORM_EPS * valid, np.float32)
                    for name, val in t.items():
                        for shape in self.shapes.get(name, []):
                            fit(val, shape).tofile(at(u, name, shape))
                    units.append((v, pi, fr, pause, None if win is None else {k: win[k] for k in ("gid", "k", "of")}))
        del head
        self._release_front()
        progress(1, steps)
        chain = os.path.join(os.path.dirname(self.runner), "npu_chain")
        if all(s["engine"] == "npu" for s in self.stages) and os.path.exists(chain) and not os.environ.get("KOKORO_NPU_NO_CHAIN"):
            self._chain(chain, units, work, path, at, lambda done: progress(1 + (steps - 2) * done / len(units), steps))
            progress(steps - 1, steps)
            return self._tail(pieces, voices, units, path, progress, steps)
        for si, s in enumerate(self.stages):
            if s["engine"] == "npu":
                jobs = os.path.join(work, s["name"] + ".jobs")
                with open(jobs, "w") as f:
                    for u in range(len(units)):
                        f.write(" ".join([at(u, n, sh) for n, sh in s["inputs"]] + ["--"] + [path(u, n) for n, _ in s["outputs"]]) + "\n")
                r = subprocess.run([self.runner, os.path.join(self.pkg, s["file"]), "@" + jobs], capture_output=True, text=True, timeout=900)
                if r.returncode != 0:
                    raise RuntimeError(f"npu {s['name']}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
            else:
                sess = self._session(s["file"])
                for u in range(len(units)):
                    feed = {n: np.fromfile(at(u, n, shape), np.float32).reshape(shape) for n, shape in s["inputs"]}
                    for (n, _), val in zip(s["outputs"], sess.run(None, feed)):
                        np.ascontiguousarray(val, np.float32).tofile(path(u, n))
                del sess
            # What no later stage reads is deleted now: its pages leave the cache, and memory stays free for the NPU.
            later = {(n, sh[-1]) for t in self.stages[si + 1:] for n, sh in t["inputs"]} | {("post", None)}
            for n, sh in s["inputs"]:
                if (n, sh[-1]) not in later:
                    for u in range(len(units)):
                        try:
                            os.remove(at(u, n, sh))
                        except FileNotFoundError:
                            pass
            progress(si + 2, steps)
        return self._tail(pieces, voices, units, path, progress, steps)

    def _chain(self, chain, units, work, path, at, done=lambda d: None):
        """Every NPU stage of every unit through npu_chain: each network loaded once per batch, the tensors between them
        never written to files nor turned into floats; only the last one's output comes back. npu_chain keeps one
        network in memory at a time and a batch's tensors between networks in RAM: batches of CHAIN_BATCH units keep
        that bounded (a few MB per unit) whatever the length of the text."""
        made = {}
        for k, s in enumerate(self.stages):
            for j, (n, _) in enumerate(s["outputs"]):
                made.setdefault(n, (k, j))
        per = 1 if os.environ.get("NPU_CHAIN_RESIDENT") else len(self.stages)  # progress lines per unit
        finished = 0
        for b0 in range(0, len(units), CHAIN_BATCH):
            batch = range(b0, min(b0 + CHAIN_BATCH, len(units)))
            jobs = os.path.join(work, f"chain{b0}.jobs")
            with open(jobs, "w") as f:
                for u in batch:
                    groups = []
                    for k, s in enumerate(self.stages):
                        ins = ["@%d.%d" % made[n] if n in made and made[n][0] < k else at(u, n, sh) for n, sh in s["inputs"]]
                        outs = [path(u, n) if n == "post" else "-" for n, _ in s["outputs"]]
                        groups.append(" ".join(ins + ["--"] + outs))
                    f.write(" | ".join(groups) + "\n")
            # npu_chain prints a line per finished run ("step" per unit and network, "run" per unit when resident).
            p = subprocess.Popen([chain] + [os.path.join(self.pkg, s["file"]) for s in self.stages] + ["@" + jobs],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            lines = []
            for line in p.stderr:
                lines.append(line.rstrip())
                if line.startswith(("run ", "step ")):
                    finished += 1
                    done(finished / per)
            if p.wait(timeout=900) != 0:
                raise RuntimeError(f"npu chain: {lines[-1:]}")

    def _tail(self, pieces, voices, units, path, progress, steps):
        tail = self._session("tail.onnx")
        out = {v: [[] for _ in pieces] for v in voices}
        pending = None  # the windows of a long piece so far, joined
        for u, (v, pi, fr, pause, win) in enumerate(units):
            post = np.fromfile(path(u, "post"), np.float32).reshape(1, 22, -1)[..., :120 * fr + 1]
            a = np.asarray(tail.run(None, {POST: np.ascontiguousarray(post)})[0], np.float32).ravel()
            if win is not None:
                if win["k"] == 0:
                    pending = a
                else:
                    # the windows share WIN_OVERLAP frames: an equal-power crossfade over them
                    ov = min(WIN_OVERLAP * HOP, len(pending), len(a))
                    w = np.sin(np.linspace(0, np.pi / 2, ov, dtype=np.float32)) ** 2
                    pending = np.concatenate([pending[:-ov], pending[-ov:] * (1 - w) + a[:ov] * w, a[ov:]])
                if win["k"] < win["of"] - 1:
                    continue
                a, pending = pending, None
            a = trim_audio(a)[0]
            if pause:
                a = np.concatenate([a, np.zeros(int(pause * SAMPLE_RATE), np.float32)])
            out[v][pi].append(a)
        progress(steps, steps)
        return {v: [np.concatenate(a) if a else np.zeros(0, np.float32) for a in out[v]] for v in voices}
