"""The NPU on small per-channel tensors (128 values, like a normalization's statistics), v in [1e-4, 1e-3]:
toys/small_<op>: add (v + 1e-5), mul (v * 1000), sqrt (sqrt(v)), recip (1 / v), rsqrt (1 / sqrt(v + 1e-5)).
Test input /tmp/smallv.f32."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

shape = [1, 128, 1, 1]
v = np.random.default_rng(0).uniform(1e-4, 1e-3, shape).astype(np.float32)
v.tofile("/tmp/smallv.f32")
k = lambda n, val: numpy_helper.from_array(np.array(val, np.float32), n)
OPS = {
    "add": ([helper.make_node("Add", ["v", "eps"], ["y"])], [k("eps", 1e-5)]),
    "mul": ([helper.make_node("Mul", ["v", "big"], ["y"])], [k("big", 1000.0)]),
    "sqrt": ([helper.make_node("Sqrt", ["v"], ["y"])], []),
    "recip": ([helper.make_node("Reciprocal", ["v"], ["y"])], []),
    "rsqrt": ([helper.make_node("Add", ["v", "eps"], ["a"]), helper.make_node("Sqrt", ["a"], ["s"]), helper.make_node("Reciprocal", ["s"], ["y"])], [k("eps", 1e-5)]),
}
for op, (nodes, inits) in OPS.items():
    d = f"toys/small_{op}"
    os.makedirs(f"{d}/calib", exist_ok=True)
    g = helper.make_graph(nodes, d, [helper.make_tensor_value_info("v", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, f"{d}/middle.onnx")
    np.save(f"{d}/calib/x0.npy", v)
    open(f"{d}/dataset.txt", "w").write("calib/x0.npy\n")
print("toys:", len(OPS))
