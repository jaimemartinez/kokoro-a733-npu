# Kokoro text-to-speech on the Allwinner A733 NPU

Turn text into natural speech **on the board itself**, offline, using the A733's built-in AI accelerator (NPU)
instead of the CPU. Works on the **Radxa Cubie A7A / A7Z** and other boards with the Allwinner A733.

```bash
kokoro-npu "Hola, ¿cómo estás? Esto lo dice la placa, sin internet." -o hola.wav
```

It uses [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a small open voice model with 54 voices in English,
Spanish, French, Italian, Portuguese, Hindi, Japanese and Chinese.

| Same greeting, 3 voices, Radxa Cubie A7Z (1 GB) | Total time | Board memory used (peak) | Distance to the original model |
|---|---|---|---|
| Kokoro on the CPU (int8) | 80–160 s | 393 MB | 2.8–3.3 dB |
| Kokoro on the NPU, v1.0.0 | 85 s | 271 MB | 3.5–4.2 dB |
| **Kokoro on the NPU, v1.1.1** | **57–81 s** | **287 MB** | **0.6–0.7 dB** |

*Board memory used*: how far the board's available memory fell during the run (the whole job: text, CPU parts,
NPU), measured the same way for the three. Times vary with what else the board is doing; each row's range is from
several runs. *Distance to the original model*: how far the audio is from the full-precision (float) Kokoro, as the
average difference of the mel spectrum after aligning the two in time (lower is better; under ~1 dB is very hard to
hear).
Since v1.1.0 the NPU is both **faster than the CPU** and **closer to the original voice than the CPU's int8 model**,
and the CPU stays free for everything else the board is doing (calls, web servers, a home assistant...).

---

## What you need

- A board with the **Allwinner A733**: Radxa Cubie A7A or A7Z (tested on the A7Z with 1 GB of RAM).
- The board maker's **official system image** (for Radxa: Radxa OS). It already includes the NPU driver.
- **Internet** on the board during installation (about 450 MB is downloaded once).
- **2 GB of free space** on the board.
- A way to type commands on the board: a keyboard and screen, or SSH from your computer
  (`ssh radxa@<the board's IP>`).

## Install (one command, a few minutes)

On the board, copy and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/jaimemartinez/kokoro-a733-npu/main/install.sh | bash
```

It will ask for your password once (to install into `/opt`). It takes 1 to 10 minutes depending on your internet,
and shows each step as it goes:

1. Checks that the board has the A733 NPU and enough space.
2. Installs the compiler `gcc` if it is missing.
3. Installs its own Python 3.12 and libraries in `/opt/kokoro-npu` (it does not touch your system's Python).
4. Downloads the NPU runtime from Radxa's official model zoo (keeps only a few small files).
5. Builds the small program that talks to the NPU.
6. Downloads the voice model compiled for the NPU (from this project's release) and the voices.
7. Sets one memory setting the NPU needs, adds the `kokoro-npu` command, and speaks a test sentence into
   `~/kokoro-npu-prueba.wav`.

When you see **Done!**, it works.

## Use it

```bash
# Spanish (default voice ef_dora)
kokoro-npu "Buenos días. Hoy va a hacer sol." -o dia.wav

# Another voice and language
kokoro-npu "Good morning! It's going to be sunny today." -v af_heart -l en-us -o morning.wav

# Slower or faster (0.5 to 2.0)
kokoro-npu "Despacio, por favor." -s 0.8 -o despacio.wav

# A long text from a file
kokoro-npu - -o libro.wav < capitulo1.txt

# All the voices
kokoro-npu --voices
```

**Listen** on the board with headphones or a speaker plugged in: `aplay dia.wav`.
**Or copy it to your computer** (run this on your computer): `scp radxa@<the board's IP>:dia.wav .`

### Voices

| Language | `-l` | Voices (`-v`) |
|---|---|---|
| Spanish | `es` | `ef_dora` (woman), `em_alex`, `em_santa` (men) |
| English (US) | `en-us` | `af_heart`, `af_bella`, `af_nicole`, `am_michael`, `am_adam`, ... (20) |
| English (UK) | `en-gb` | `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`, ... (8) |
| French | `fr-fr` | `ff_siwis` |
| Italian | `it` | `if_sara`, `im_nicola` |
| Portuguese (BR) | `pt-br` | `pf_dora`, `pm_alex`, `pm_santa` |
| Hindi, Japanese, Chinese | `hi`, `ja`, `cmn` | `hf_*`, `hm_*`, `jf_*`, `jm_*`, `zf_*`, `zm_*` |

Any voice can read any language (it will have that voice's accent).

### From Python

```python
import sys
sys.path.insert(0, "/opt/kokoro-npu/runtime")
from kokoro_npu import NpuKokoro
import numpy as np
import soundfile as sf

tts = NpuKokoro("/opt/kokoro-npu/model", "/opt/kokoro-npu/bin/npu_run", "/opt/kokoro-npu/model/voices-v1.0.bin")
pieces = tts.speak(["Hola.", "¿Qué tal tu día?"], ["ef_dora"], speed=1.0, lang="es")["ef_dora"]
sf.write("hola.wav", np.concatenate(pieces), 24000)
```

Run it with `/opt/kokoro-npu/venv/bin/python`. `speak` returns one array per sentence you gave it; a sentence spoken
in the same NPU pass as the one before it comes back empty (its audio is in that one), so join them in order.
Asking for several sentences and voices in one `speak` call is much faster than many calls: the NPU networks are
loaded once for the whole batch.

## If something goes wrong

| Message | What to do |
|---|---|
| `the NPU driver is not loaded (/dev/vipcore is missing)` | Flash the official Radxa OS image for your board; custom kernels often leave the NPU out. |
| `not enough free disk space` | Free 2 GB (`sudo apt clean`, delete old files) or use a bigger SD card / eMMC. |
| `npu dec: ... fail to ...` or `run failed` | Another program may be using the NPU, or the board is very short of memory. Close it and try again; a reboot clears it. |
| It works but is slow | Check `cat /proc/sys/vm/compaction_proactiveness` says `90` (the installer sets it; it needs to stay). |
| A word is pronounced wrong | Check `-l` matches the language of the text. |

To remove everything: `curl -fsSL https://raw.githubusercontent.com/jaimemartinez/kokoro-a733-npu/main/uninstall.sh | bash`

---

## How it works

Kokoro is split in pieces: what is light or text-related runs on the CPU, and the heavy audio generator runs on the
NPU as five compiled networks (int16):

```
text -> phonemes -> [front, CPU int8] -> [head, CPU] -> [dec -> g1 -> g2 -> g3 -> g4, NPU] -> [tail, CPU] -> audio
```

Each NPU pass handles about 5 seconds of speech: consecutive sentences share one pass while they fit, and a shorter
piece is padded and masked (same result). A longer sentence keeps its own intonation and timing: the CPU side runs on
the whole sentence, and only the NPU part runs in overlapping windows that are crossfaded back together, so there is
no pause or new intonation in the middle of it. The five networks run in one process (`npu_chain`), network by
network: each one is loaded once per batch of four pieces, so only one is in memory at a time, and what the next one
reads waits in RAM as int16 (no files, no conversions). The CPU side runs on the A733's two fast cores (Cortex-A76).

## What we learned about the A733 NPU

Getting Kokoro to run *and sound right* on this NPU needed several workarounds. The toolkit's simulator says the
compiled model is accurate (58 dB), while the real chip gave rough audio (23 dB): **several operations behave
differently on the chip than in the simulator.** If you convert your own models, read
[docs/a733-npu-pitfalls.md](docs/a733-npu-pitfalls.md) first. In short:

- Long 1-D sequences (`[N, C, L]`, L in the tens of thousands) hang the NPU or run slowly: use `[N, C, 1, L]`.
- `ReduceSum` is imprecise: use `GlobalAveragePool` times the count.
- `Sin` is wrong for arguments beyond about ±8: reduce the argument first.
- A small constant added to small values loses ~5 %: add it while the numbers are still large.
- int8 / uint8 ruins the audio of a vocoder; int16 works.
- The NPU runs about 2× slower when free memory is fragmented: `vm.compaction_proactiveness = 90`.
- One int16 scale per tensor: channels of very different sizes (or a constant with one outlier) lose the small ones.
  Rescale per channel where the math allows it (before a norm), bound the outliers.
- A convolution followed by an add on a sequence longer than 8191 is wrong after ~8022 samples: keep them apart.

v1.1.0 fixed the last three in the model; the audio went from "sounds like the CPU" to measurably closer to the
original than the CPU's int8 model.

## Rebuilding the model yourself (advanced)

Only needed to change the model; the installer downloads a ready-made one. It needs Allwinner's ACUITY toolkit
(from your board maker) on an x86-64 Linux PC. See [docs/rebuild.md](docs/rebuild.md).

## License and credits

- This project's code: Apache License 2.0 ([LICENSE](LICENSE)).
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad (Apache 2.0) and the ONNX export and voices of
  [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). The model package in the releases is derived from them.
- The NPU runtime (`libVIPhal`, `libNBGlinker`, `vip_lite.h`) belongs to Allwinner / VeriSilicon and is downloaded
  from [Radxa's model zoo](https://docs.radxa.com/cubie/a7z/app-dev/npu-dev/model-zoo/model-zoo-download) during
  installation; it is not part of this repository.
