"""Kokoro's snake activation alone, x + sin(a x)^2 / a, on 128 channels, to see which layout the NPU runs:
snake24k (1-D, 24001: like g3, expected to hang), snake12k (1-D, 12000) and snake24k4d (padded to 47 x 512, 4-D)."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

rng = np.random.default_rng(0)
C = 128
for name, shape in (("snake24k", [1, C, 24001]), ("snake12k", [1, C, 12000]), ("snake24k4d", [1, C, 47, 512])):
    d = os.path.join("toys", name)
    os.makedirs(os.path.join(d, "calib"), exist_ok=True)
    ash = [1, C] + [1] * (len(shape) - 2)
    a = rng.uniform(0.5, 3, ash).astype(np.float32)
    inits = [numpy_helper.from_array(a, "a"), numpy_helper.from_array((1 / a).astype(np.float32), "ia"),
             numpy_helper.from_array(np.array(2, np.float32), "two")]
    nodes = [
        helper.make_node("Mul", ["a", "x"], ["ax"]),
        helper.make_node("Sin", ["ax"], ["s"]),
        helper.make_node("Pow", ["s", "two"], ["s2"]),
        helper.make_node("Mul", ["ia", "s2"], ["q"]),
        helper.make_node("Add", ["x", "q"], ["y"]),
    ]
    g = helper.make_graph(nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, os.path.join(d, "middle.onnx"))
    np.save(os.path.join(d, "calib", "x0.npy"), (rng.standard_normal(shape) * 2).astype(np.float32))
    open(os.path.join(d, "dataset.txt"), "w").write("calib/x0.npy\n")
    print(name, shape)
