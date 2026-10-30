"""The student vocoder as one NPU stage: a fixed window of T frames in, the 601-bin log magnitude and phase out.

Rewritten for the A733 toolchain while staying exact in float (checked against train.py's model below):
  - 4-D tensors [1, C, 1, T] and 1x7 / 1x1 convolutions only (the toolchain's fast path, as for Kokoro's g stages);
  - each input channel divided by its std on the CPU and the std folded into the first convolution, so the 512
    decoder channels (~0.1) and F0 / energy (~1) share one int16 scale without losing either;
  - layer norm written out (mean, variance, 1/sqrt) and GELU in its tanh form; layer scale folded into pw2;
  - the residual stream carried K times smaller (it reaches +-2500, and its squares millions: past what one int16
    scale holds). Every norm is blind to scale, so only its epsilon changes (eps / K^2);
  - GELU's cubic term on x clamped to +-6 (tanh is already +-1 there): unclamped, x^3 reached 10^6 and ate the
    small values' precision.
The model is fully convolutional with a 27-frame reach each side, so the runtime runs any length as overlapping
windows of T and keeps each window's middle.

  python export.py PACKDIR CKPT OUTDIR [T]   -> OUTDIR/middle.onnx, scale.npy, calib/, dataset.txt, in*.f32, ref_*.npy"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

pack, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
T = int(sys.argv[4]) if len(sys.argv) > 4 else 256
K = float(os.environ.get("VOC_K", "64"))
EPS = 1e-5 / K ** 2
os.makedirs(os.path.join(out, "calib"), exist_ok=True)
sd = torch.load(ckpt, map_location="cpu")
d = sd["embed.weight"].shape[0]
nb = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})

feats = np.load(os.path.join(pack, "feats.npy"), mmap_mode="r")
styles = np.load(os.path.join(pack, "styles.npy"))
index = np.load(os.path.join(pack, "index.npy"))
rng = np.random.default_rng(1)
probe = np.concatenate([np.asarray(feats[a:a + n], np.float32) for a, n in index[rng.choice(len(index), 200)]])
scale = probe.std(0).clip(min=1e-3).astype(np.float32)  # per input channel
np.save(os.path.join(out, "scale.npy"), scale)


class NpuStudent(nn.Module):
    def __init__(self):
        super().__init__()
        w = sd["embed.weight"] * torch.from_numpy(scale)[None, :, None]  # input arrives divided by scale
        self.embed = nn.Conv2d(514, d, (1, 7), padding=(0, 3))
        self.embed.weight.data = w[:, :, None, :] / K
        self.embed.bias.data = sd["embed.bias"] / K
        self.dw, self.film, self.pw1, self.pw2 = (nn.ModuleList() for _ in range(4))
        for i in range(nb):
            p = f"blocks.{i}."
            c = nn.Conv2d(d, d, (1, 7), padding=(0, 3), groups=d)
            c.weight.data, c.bias.data = sd[p + "dw.weight"][:, :, None, :], sd[p + "dw.bias"] / K
            self.dw.append(c)
            for name, lst, gain in (("film", self.film, None), ("pw1", self.pw1, None), ("pw2", self.pw2, sd[p + "gamma"])):
                wt, b = sd[p + name + ".weight"], sd[p + name + ".bias"]
                if gain is not None:
                    wt, b = wt * gain[:, None] / K, b * gain / K
                c = nn.Conv2d(wt.shape[1], wt.shape[0], 1)
                c.weight.data, c.bias.data = wt[:, :, None, None], b
                lst.append(c)
        self.nw, self.nb_ = sd["norm.weight"][None, :, None, None], sd["norm.bias"][None, :, None, None]
        self.out = nn.Conv2d(d, sd["out.weight"].shape[0], 1)
        self.out.weight.data, self.out.bias.data = sd["out.weight"][:, :, None, None], sd["out.bias"]

    @staticmethod
    def norm(x):
        x = x - x.mean(1, keepdim=True)
        return x * torch.rsqrt((x * x).mean(1, keepdim=True) + EPS)

    @staticmethod
    def gelu(x):
        c = x.clamp(-6.0, 6.0)
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (c + 0.044715 * c * c * c)))

    def forward(self, x, style):  # x [1, 514, 1, T] already divided by scale; style [1, 128, 1, 1]
        x = self.embed(x)
        for dw, film, pw1, pw2 in zip(self.dw, self.film, self.pw1, self.pw2):
            sc, sh = film(style).chunk(2, 1)
            h = self.norm(dw(x)) * (1 + sc) + sh
            x = x + pw2(self.gelu(pw1(h)))
        return self.out(self.norm(x) * self.nw + self.nb_)  # [1, 1202, 1, T]: log magnitude (601) then phase (601)


m = NpuStudent().eval()

# Same numbers as the trained model (train.py's Student, exact GELU), on a real window.
sys.argv = ["train.py", pack, "/tmp/unused", "--dim", str(d), "--blocks", str(nb)]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")).read()
src = src[:src.index("# ---------------------------------------------------------------- discriminators")]
ns = {}
exec(compile(src.replace('dev = "cuda"', 'dev = "cpu"'), "train.py", "exec"), ns)
ref = ns["Student"](d, nb).eval()
ref.load_state_dict(sd)
a, n = index[0]
x = torch.from_numpy(np.asarray(feats[a:a + T], np.float32).T.copy())[None]
st = torch.from_numpy(styles[0])[None]
with torch.no_grad():
    mag, ph = ref.spec(x, st)
    h = m((x / torch.from_numpy(scale)[None, :, None])[:, :, None, :], st[:, :, None, None])[:, :, 0]
    want = torch.cat([torch.log(mag), ph], 1)
    snr = 10 * torch.log10((want ** 2).sum() / ((want - h) ** 2).sum())
print(f"dim {d}, {nb} blocks, T {T}: rewritten vs trained {snr:.1f} dB")

torch.onnx.export(m, (torch.zeros(1, 514, 1, T), torch.zeros(1, 128, 1, 1)), os.path.join(out, "middle.onnx"),
                  input_names=["feats", "style"], output_names=["spec"], opset_version=13, dynamo=False)

# Calibration windows (real speech, every voice) and two test windows with their float references.
ids = rng.choice([i for i in range(len(index)) if index[i][1] >= T], 40, replace=False)
lines = []
for r, i in enumerate(ids):
    a, n = index[i]
    o = rng.integers(0, n - T + 1)
    xi = (np.asarray(feats[a + o:a + o + T], np.float32) / scale).T[None, :, None, :]
    si = styles[i].reshape(1, 128, 1, 1).astype(np.float32)
    np.save(os.path.join(out, f"calib/r{r}_feats.npy"), np.ascontiguousarray(xi))
    np.save(os.path.join(out, f"calib/r{r}_style.npy"), si)
    lines.append(f"calib/r{r}_feats.npy calib/r{r}_style.npy")
    if r < 2:
        with torch.no_grad():
            y = m(torch.from_numpy(xi), torch.from_numpy(si)).numpy()
        np.ascontiguousarray(xi).tofile(os.path.join(out, f"t{r}_in00.f32"))
        si.tofile(os.path.join(out, f"t{r}_in01.f32"))
        np.save(os.path.join(out, f"ref_t{r}.npy"), y)
open(os.path.join(out, "dataset.txt"), "w").write("\n".join(lines) + "\n")
print(out, "ready:", len(lines), "calibration windows")
