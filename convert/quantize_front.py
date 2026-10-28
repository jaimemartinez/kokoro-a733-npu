"""Optional: the front (text -> features, F0, energy) as dynamic int8 for the board's CPU, like the int8 Kokoro build:
2.4x faster and 27 MB instead of 106 MB, durations within one frame of the float model.

  python quantize_front.py <cut dir>      -> <cut dir>/front.int8.onnx"""
import os
import sys
from onnxruntime.quantization import QuantType, quantize_dynamic

cut = sys.argv[1]
quantize_dynamic(os.path.join(cut, "front.onnx"), os.path.join(cut, "front.int8.onnx"), weight_type=QuantType.QInt8)
print("front:", os.path.getsize(os.path.join(cut, "front.onnx")) >> 20, "->",
      os.path.getsize(os.path.join(cut, "front.int8.onnx")) >> 20, "MB")
