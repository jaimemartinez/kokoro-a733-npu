# The experimental vocoder (advanced)

`kokoro-npu --fast` replaces the second half of Kokoro (its generator: g1-g4 on the NPU, plus head and tail on the
CPU) with a much smaller network trained to imitate it, in the style of [Vocos](https://arxiv.org/abs/2306.00814):
eight ConvNeXt blocks at the decoder's frame rate (80 per second) predict a spectrum (log magnitude and phase,
n_fft 1200, hop 300), and an inverse STFT on the CPU turns it into audio. The decoder stays as it is.

```
text -> [front, CPU] -> [dec, NPU] -> [vocoder, NPU] -> [inverse STFT, CPU] -> audio
```

It is **experimental**: much faster, but it still sounds somewhat robotic (its phase is the hardest part to learn).
Install it with `install.sh --experimental-vocoder`; it is used only when asked for (`--fast`, or `vocoder=True` /
`KOKORO_NPU_VOCODER=1` from Python).

## What it learned from

Kokoro itself, in float on a GPU, speaking sentences from ten Spanish public-domain books of Project Gutenberg
(numbers 1619, 2000, 15532, 16625, 17013, 24536, 25640, 36805, 42727, 49836), 70 % in the Spanish voices and the
rest in random ones, at speeds 0.85-1.15: 4,500 pieces, 6.2 hours. For each piece it keeps what the vocoder sees (the
decoder's output, F0, energy and the generator's style) and what it must produce (Kokoro's audio).

## Training it yourself

On a PC with an NVIDIA GPU (a 4 GB GTX 1650 is enough; about 2 steps per second), Python 3.10+ with `torch`,
`torchaudio`, `onnx`, `onnxruntime-gpu` and `kokoro-onnx`, `kokoro-v1.0.onnx` and `voices-v1.0.bin` in the work
directory and the books as `texts/*.txt`. `V` = the `vocoder/` folder of this repository.

```bash
python V/datagen.py data 4500             # Kokoro's own output, in shards of 250 pieces
python V/pack.py data pack                # one memory-mapped training set
python V/train.py pack run --gan-after 30000 --batch 8    # mel + STFT losses, then the Vocos discriminators
```

`train.py` saves `run/G_<step>.pt` every 5,000 steps and resumes from `run/last.pt`. The released one is step
85,000 of 400,000.

## For the NPU

```bash
python V/export.py pack run/G_0085000.pt npu 256      # -> npu/middle.onnx and its calibration set
C/build_stage.sh npu                                  # inside the ACUITY environment (C = convert/), int16
```

`export.py` rewrites the network for the chip without changing its output (checked against the trained model):

- 4-D tensors and 1x7 / 1x1 convolutions only.
- Each input channel divided by its spread, with the factor folded into the first convolution: the decoder's 512
  channels and F0 / energy then share one int16 scale.
- **The residual stream carried 64 times smaller.** It reaches ±2500 and its squares millions, past what one int16
  scale holds; every norm ignores scale (only its epsilon changes). Without this the chip gave 21-24 dB; with it,
  53-60 dB on the spectrum.
- **GELU's cubic term computed on x clamped to ±6** (tanh is already ±1 there): unclamped, x³ reached a million and
  the small values lost their precision.

The package is `voc_256.nb` (the compiled network), `voc_scale.npy` (the input's per-channel scale) and
`vocoder.json` (`{"file", "frames": 256, "context": 28, "scale"}`), next to the model's `manifest.json`.

The runtime runs a piece of any length in windows of 256 frames that overlap by 28 frames on each side (the network
sees 27 frames each way) and keeps each window's middle. **No window reaches past the end of the piece:** frames of
zeros are input the network never saw in training, and a window padded with them came out at 2-7 dB. The last window
is aligned to the end instead, and a piece shorter than a window is padded with copies of its own last frame.
