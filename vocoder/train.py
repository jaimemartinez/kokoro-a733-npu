"""Student vocoder for Kokoro (Vocos-style): from the decoder's output (512 ch), log F0, energy and the generator's
style, at Kokoro's 80 frames/s, predict an STFT (n_fft 1200, hop 300) whose inverse is the audio. Everything the NPU
would run is at frame rate (ConvNeXt blocks); the iSTFT is a few ms on the CPU.

Trained on the teacher's own audio (datagen.py + pack.py): mel L1 + multi-resolution STFT, then with the Vocos
discriminators (multi-period + multi-resolution) once the mel loss has settled.

  python train.py PACKDIR RUNDIR [--gan-after STEPS] [--steps N]"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("run")
ap.add_argument("--steps", type=int, default=400000)
ap.add_argument("--gan-after", type=int, default=20000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seg", type=int, default=64)  # frames per training segment (0.8 s)
ap.add_argument("--gan-seg", type=int, default=32)  # shorter segments once the discriminators run (they dominate the step)
ap.add_argument("--amp", action="store_true")  # fp16 discriminators (was slower on the GTX 1650)
ap.add_argument("--dim", type=int, default=384)
ap.add_argument("--blocks", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.run, exist_ok=True)
dev = "cuda"
SR, HOP, NFFT = 24000, 300, 1200

# ---------------------------------------------------------------- data
feats = np.load(os.path.join(args.pack, "feats.npy"), mmap_mode="r")
audio = np.load(os.path.join(args.pack, "audio.npy"), mmap_mode="r")
styles = np.load(os.path.join(args.pack, "styles.npy"))
index = np.load(os.path.join(args.pack, "index.npy"))
rng = np.random.default_rng(0)
order = rng.permutation(len(index))
held = set(order[:64].tolist())  # never trained on: for the samples and the validation loss
train_ids = np.array([i for i in range(len(index)) if i not in held and index[i][1] > args.seg])


def batch(seg):
    ids = rng.choice(train_ids, args.batch)
    x = np.empty((args.batch, 514, seg), np.float32)
    y = np.empty((args.batch, seg * HOP), np.float32)
    s = styles[ids]
    for b, i in enumerate(ids):
        a, n = index[i]
        o = a + rng.integers(0, n - seg + 1)
        x[b] = feats[o:o + seg].T
        y[b] = audio[o * HOP:(o + seg) * HOP] / 32768.0
    return torch.from_numpy(x).to(dev), torch.from_numpy(s).to(dev), torch.from_numpy(y).to(dev)


# ---------------------------------------------------------------- student
class Block(nn.Module):
    def __init__(self, d, cond, scale):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.film = nn.Linear(cond, 2 * d)  # the style scales and shifts each block (AdaLN)
        self.pw1, self.pw2 = nn.Linear(d, 3 * d), nn.Linear(3 * d, d)
        self.gamma = nn.Parameter(torch.full((d,), scale))  # layer scale, 1 / blocks as in Vocos

    def forward(self, x, c):
        r = x
        x = self.dw(x).transpose(1, 2)
        sc, sh = self.film(c).unsqueeze(1).chunk(2, -1)
        x = self.norm(x) * (1 + sc) + sh
        x = self.pw2(F.gelu(self.pw1(x)))
        return r + (self.gamma * x).transpose(1, 2)


class Student(nn.Module):
    def __init__(self, d=384, blocks=8):
        super().__init__()
        self.embed = nn.Conv1d(514, d, 7, padding=3)
        self.blocks = nn.ModuleList(Block(d, 128, 1 / blocks) for _ in range(blocks))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, NFFT + 2)  # log magnitude and phase of 601 bins

    def spec(self, x, style):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x, style)
        h = self.out(self.norm(x.transpose(1, 2))).transpose(1, 2)
        mag, ph = h.chunk(2, 1)
        return torch.exp(mag.clamp(max=7.0)), ph

    def forward(self, x, style):
        mag, ph = self.spec(x, style)
        z = torch.complex(mag * torch.cos(ph), mag * torch.sin(ph))
        return torch.istft(z, NFFT, HOP, NFFT, window=torch.hann_window(NFFT, device=x.device), center=True,
                           length=x.shape[-1] * HOP)


# ---------------------------------------------------------------- discriminators (Vocos / HiFi-GAN)
class DiscP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        ch = [1, 32, 128, 256, 512, 512]  # half of HiFi-GAN's widest layers: the 4 GB card holds both models
        self.convs = nn.ModuleList(nn.utils.parametrizations.weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3 if i < 4 else 1, 1), (2, 0))) for i in range(5))
        self.post = nn.utils.parametrizations.weight_norm(nn.Conv2d(512, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        b, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), "reflect")
        x = x.view(b, 1, -1, self.p)
        fm = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class DiscR(nn.Module):
    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        wn = nn.utils.parametrizations.weight_norm
        self.convs = nn.ModuleList([wn(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                                    *[wn(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))) for _ in range(3)],
                                    wn(nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))])
        self.post = wn(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        s = torch.stft(x, self.n_fft, self.hop, self.n_fft, torch.hann_window(self.n_fft, device=x.device), return_complex=True).abs()
        x = s.unsqueeze(1)
        fm = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class Discs(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([*(DiscP(p) for p in (2, 3, 5, 7, 11)), *(DiscR(n, h) for n, h in ((2048, 512), (1024, 256), (512, 128)))])

    def forward(self, x):
        return [d(x) for d in self.ds]


# ---------------------------------------------------------------- losses
mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=100, power=1).to(dev)


def mel_loss(a, b):
    return F.l1_loss(torch.log(mel(a).clamp(min=1e-5)), torch.log(mel(b).clamp(min=1e-5)))


def stft_loss(a, b):
    tot = 0
    for n, h in ((512, 128), (1024, 256), (2048, 512)):
        w = torch.hann_window(n, device=a.device)
        A, B = torch.stft(a, n, h, n, w, return_complex=True).abs(), torch.stft(b, n, h, n, w, return_complex=True).abs()
        tot = tot + (B - A).norm() / B.norm().clamp(min=1e-5) + F.l1_loss(torch.log(A.clamp(min=1e-5)), torch.log(B.clamp(min=1e-5)))
    return tot / 3


# ---------------------------------------------------------------- training
G = Student(args.dim, args.blocks).to(dev)
D = Discs().to(dev)
oG = torch.optim.AdamW(G.parameters(), 5e-4, betas=(0.8, 0.9))
oD = torch.optim.AdamW(D.parameters(), 2e-4, betas=(0.8, 0.9))
scG, scD = torch.amp.GradScaler("cuda", enabled=args.amp), torch.amp.GradScaler("cuda", enabled=args.amp)
step = 0
ck = os.path.join(args.run, "last.pt")
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev)
    G.load_state_dict(st["G"]); D.load_state_dict(st["D"]); oG.load_state_dict(st["oG"]); oD.load_state_dict(st["oD"])
    step = st["step"]
print(f"student {sum(p.numel() for p in G.parameters()) / 1e6:.2f} M parameters, {len(train_ids)} training items, from step {step}", flush=True)
json.dump(vars(args), open(os.path.join(args.run, "args.json"), "w"))


def lr_at(s):  # warm-up, then cosine
    return min(1.0, s / 1000) * (0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)) * 0.95 + 0.05)


t0 = time.time()
log = open(os.path.join(args.run, "log.txt"), "a")
while step < args.steps:
    for o, base in ((oG, 5e-4), (oD, 2e-4)):
        for gr in o.param_groups:
            gr["lr"] = base * lr_at(step)
    gan = step >= args.gan_after
    x, s, y = batch(args.gan_seg if gan else args.seg)
    yh = G(x, s)
    if gan:
        # The discriminators run in fp16 (the card's fast path); the student and the losses stay in fp32.
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            real = D(y)
            fake = D(yh.detach())
            dl = 0
            for (r, _), (f, _) in zip(real, fake):
                dl = dl + F.relu(1 - r.float()).mean() + F.relu(1 + f.float()).mean()
        oD.zero_grad(set_to_none=True)
        scD.scale(dl).backward()
        scD.step(oD)
        scD.update()
        dr = [[t.detach() for t in fmr] for _, fmr in real]  # the real side's features, reused for feature matching
    lm, ls = mel_loss(yh, y), stft_loss(yh, y)
    loss = 45 * lm + 2 * ls
    if gan:
        adv = fm = 0
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            gen = D(yh)
        for (f, fmf), fmr in zip(gen, dr):
            adv = adv + F.relu(1 - f.float()).mean()
            fm = fm + sum(F.l1_loss(a.float(), b.float()) / (b.float().abs().mean() + 1e-5) for a, b in zip(fmf, fmr)) / len(fmf)
        loss = loss + adv + 2 * fm
    oG.zero_grad(set_to_none=True)
    scG.scale(loss).backward()
    scG.unscale_(oG)
    torch.nn.utils.clip_grad_norm_(G.parameters(), 100)
    scG.step(oG)
    scG.update()
    step += 1
    if step % 200 == 0:
        msg = f"{step} mel {lm.item():.4f} stft {ls.item():.4f}" + (f" adv {adv.item():.3f} fm {fm.item():.3f} d {dl.item():.3f}" if gan else "") + f" {200 / (time.time() - t0):.2f} it/s"
        t0 = time.time()
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
    if step % 5000 == 0:
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(), "step": step}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        torch.save(G.state_dict(), os.path.join(args.run, f"G_{step:07d}.pt"))
