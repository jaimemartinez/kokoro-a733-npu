#!/bin/bash
# Step 7 (inside the ACUITY toolkit environment): import, int16-quantize on the stage's calibration set and export
# one stage as an A733 network binary (NBG): <stage dir>/wksp/p_nbg_unify/network_binary.nb.
#
#   build_stage.sh <stage dir> [int16]
#
# PEGASUS: the toolkit's pegasus.py (default /usr/local/acuity_command_line_tools/pegasus.py)
# VIV_SDK: the Vivante SDK command tools (default /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools)
# int8 is not offered: on Kokoro's generator it ruins the audio (1-21 dB against the float model).
set -u
cd "$1"
P="python3 ${PEGASUS:-/usr/local/acuity_command_line_tools/pegasus.py}"
SDK=${VIV_SDK:-/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools}
Q=${2:-int16}
log() { echo "$(date +%T) $(basename "$1"): $2"; }

$P import onnx --model middle.onnx --output-model middle.json --output-data middle.data > import.log 2>&1 \
  || { log "$1" "import failed"; tail -5 import.log; exit 1; }
$P generate inputmeta --model middle.json --input-meta-output middle_inputmeta.yml > meta.log 2>&1
# Plain tensors, not images: no channel flip, nothing scaled.
sed -i -e 's/category: image/category: undefined/' -e 's/reverse_channel: true/reverse_channel: false/' middle_inputmeta.yml

$P quantize --model middle.json --model-data middle.data --device CPU --with-input-meta middle_inputmeta.yml \
  --iterations "$(wc -l < dataset.txt)" --rebuild --model-quantize middle_int16.quantize \
  --quantizer dynamic_fixed_point --qtype "$Q" > quant.log 2>&1
[ -s middle_int16.quantize ] || { log "$1" "quantize failed"; tail -8 quant.log; exit 1; }

rm -rf wksp
$P export ovxlib --pack-nbg-unify --optimize VIP9000NANODI_PLUS_PID0X1000003B --viv-sdk "$SDK" \
  --model middle.json --model-data middle.data --dtype quantized --model-quantize middle_int16.quantize \
  --target-ide-project linux64 --with-input-meta middle_inputmeta.yml --output-path wksp/p/p > export.log 2>&1
nb=wksp/p_nbg_unify/network_binary.nb
[ -s $nb ] || { log "$1" "export failed"; grep -iE "error|fail" export.log | grep -v "Error(0)" | head -5; exit 1; }
log "$1" "network $(stat -c %s $nb) bytes"
