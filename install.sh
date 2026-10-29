#!/bin/bash
# kokoro-a733-npu installer: Kokoro text-to-speech on the Allwinner A733 NPU (Radxa Cubie A7A / A7Z and other A733
# boards). Run it on the board:
#
#   curl -fsSL https://raw.githubusercontent.com/jaimemartinez/kokoro-a733-npu/main/install.sh | bash
#
# It installs into /opt/kokoro-npu and adds the `kokoro-npu` command. Nothing else on the system is changed except
# one kernel setting (see below), the gcc compiler if missing, and the command in /usr/local/bin.
# Options: --package <file.tar.gz> uses a model package already downloaded (offline install).
set -euo pipefail

VERSION=v1.1.1
MODEL_VERSION=v1.1.0 # the compiled model package (a release asset); the code above comes from $VERSION
REPO=jaimemartinez/kokoro-a733-npu
PREFIX=/opt/kokoro-npu
MODEL_URL=https://github.com/$REPO/releases/download/$MODEL_VERSION/kokoro-a733-npu-model-$MODEL_VERSION.tar.gz
VOICES_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
ZOO_URL=https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz
RAW=https://raw.githubusercontent.com/$REPO/$VERSION
PACKAGE=""
[ "${1:-}" = "--package" ] && PACKAGE=$(readlink -f "$2")

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
# A big download that survives a dropped connection: it resumes where it stopped, up to 8 times.
fetch() {
  local url=$1 out=$2 i
  for i in 1 2 3 4 5 6 7 8; do
    curl -fL --progress-bar --connect-timeout 30 -C - -o "$out" "$url" && return 0
    echo "The download stopped; resuming ($i/8)..."
    sleep 3
  done
  echo "Could not download $url. Check the board's internet connection and run the installer again." >&2
  return 1
}
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo

step "1/7 Checking the board"
[ "$(uname -m)" = aarch64 ] || fail "this is not a 64-bit ARM board (found $(uname -m)). It needs an Allwinner A733 board."
[ -e /dev/vipcore ] || fail "the NPU driver is not loaded (/dev/vipcore is missing). Use the board maker's official system image (for Radxa: Radxa OS for Cubie A7A/A7Z), which includes it."
free_mb=$(df -Pm /opt | awk 'NR==2 {print $4}')
[ "$free_mb" -gt 1500 ] || fail "not enough free disk space: ${free_mb} MB, needs about 1.5 GB."
echo "OK: A733 NPU found, ${free_mb} MB free."

step "2/7 Installing the tools it needs (gcc, curl)"
if ! command -v gcc >/dev/null || ! command -v curl >/dev/null; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq gcc curl ca-certificates >/dev/null
fi
echo "OK."

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
$SUDO mkdir -p $PREFIX/{bin,lib,include,runtime,model}
$SUDO chown -R "$(id -u):$(id -g)" $PREFIX

step "3/7 Python and its libraries (about 200 MB, a few minutes)"
export UV_INSTALL_DIR=$PREFIX/uv UV_PYTHON_INSTALL_DIR=$PREFIX/python UV_NO_MODIFY_PATH=1
[ -x $PREFIX/uv/uv ] || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null
$PREFIX/uv/uv venv -q --python 3.12 $PREFIX/venv
$PREFIX/uv/uv pip install -q --python $PREFIX/venv/bin/python kokoro-onnx==0.6.1 onnxruntime==1.31.1 soundfile numpy
echo "OK."

step "4/7 The NPU runtime from Radxa's model zoo (downloads 200 MB once, keeps a few small files)"
fetch "$ZOO_URL" "$TMP/zoo.tar.gz"
tar -xzf "$TMP/zoo.tar.gz" -C "$TMP" --wildcards '*/common/npuruntime/lib_linux_aarch64/A733/*' '*/common/npuruntime/include/*'
cp "$TMP"/*/common/npuruntime/lib_linux_aarch64/A733/*.so $PREFIX/lib/
cp "$TMP"/*/common/npuruntime/include/*.h $PREFIX/include/
rm -rf "$TMP"/*
echo "OK."

step "5/7 Building the NPU runner"
if [ -f runtime/npu_run.c ] && [ -f runtime/npu_chain.c ] && [ -f runtime/kokoro_npu.py ]; then SRC=runtime; else
  SRC="$TMP/src"; mkdir -p "$SRC"
  for f in npu_run.c npu_chain.c kokoro_npu.py kokoro-npu; do curl -fsSL -o "$SRC/$f" "$RAW/runtime/$f"; done
fi
# npu_run runs one network; npu_chain runs all five per sentence in one process, the tensors between them kept on
# the NPU (kokoro_npu.py uses it when it is there).
for prog in npu_run npu_chain; do
  gcc -O2 -o $PREFIX/bin/$prog "$SRC/$prog.c" -I$PREFIX/include -L$PREFIX/lib -lNBGlinker -lVIPhal -lm \
    -Wl,--disable-new-dtags,-rpath,$PREFIX/lib
done
cp "$SRC/kokoro_npu.py" $PREFIX/runtime/
sed "1s|.*|#!$PREFIX/venv/bin/python|" "$SRC/kokoro-npu" > $PREFIX/bin/kokoro-npu && chmod +x $PREFIX/bin/kokoro-npu
echo "OK."

step "6/7 The voice model for the NPU (about 110 MB) and the voices (28 MB)"
if [ -n "$PACKAGE" ]; then cp "$PACKAGE" "$TMP/model.tar.gz"; else fetch "$MODEL_URL" "$TMP/model.tar.gz"; fi
tar -xzf "$TMP/model.tar.gz" -C $PREFIX/model
[ -f $PREFIX/model/voices-v1.0.bin ] || fetch "$VOICES_URL" $PREFIX/model/voices-v1.0.bin
echo "OK."

step "7/7 Last settings and a test"
# The NPU needs large free blocks of memory; without this setting it runs about 2x slower once memory fragments.
echo "vm.compaction_proactiveness = 90" | $SUDO tee /etc/sysctl.d/90-kokoro-npu.conf >/dev/null
$SUDO sysctl -q -p /etc/sysctl.d/90-kokoro-npu.conf || true
$SUDO ln -sf $PREFIX/bin/kokoro-npu /usr/local/bin/kokoro-npu
cd "$HOME"
kokoro-npu "Hola. La instalación terminó y la voz funciona en la unidad neuronal." -o kokoro-npu-prueba.wav

printf '\n\033[1;32mDone!\033[0m Try it:\n\n  kokoro-npu "Hola, ¿cómo estás?" -o hola.wav\n  aplay hola.wav\n\n'
printf 'Listen to the test: %s/kokoro-npu-prueba.wav   More: kokoro-npu --help\n' "$HOME"
