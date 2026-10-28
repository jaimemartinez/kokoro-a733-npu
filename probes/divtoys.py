"""How exact the NPU's normalizing division is: x / sqrt(v + eps) for x [1,128,1,24064] ~ N(0, 0.02) and per-channel
variances v [1,128,1,1] in [1e-4, 1e-3] (like Kokoro's noise branch). toys/div_<way>: plain (Div by Sqrt, as now),
recip (times Reciprocal of Sqrt), newton (inverse square root refined by two Newton steps, on the 128 values only).
Test inputs /tmp/divx.f32, /tmp/divv.f32 and the exact result /tmp/divref.npy."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

C, L = 128, 24064
rng = np.random.default_rng(0)
x = (rng.standard_normal((1, C, 1, L)) * 0.02).astype(np.float32)
v = rng.uniform(1e-4, 1e-3, (1, C, 1, 1)).astype(np.float32)
eps = 1e-5
x.tofile("/tmp/divx.f32"); v.tofile("/tmp/divv.f32")
np.save("/tmp/divref.npy", x.astype(np.float64) / np.sqrt(v.astype(np.float64) + eps))
k = lambda n, val: numpy_helper.from_array(np.array(val, np.float32), n)
common = [helper.make_node("Add", ["v", "eps"], ["ve"]), helper.make_node("Sqrt", ["ve"], ["sd"])]
WAYS = {
    "plain": (common + [helper.make_node("Div", ["x", "sd"], ["y"])], [k("eps", eps)]),
    "recip": (common + [helper.make_node("Reciprocal", ["sd"], ["r"]), helper.make_node("Mul", ["x", "r"], ["y"])], [k("eps", eps)]),
    "newton": (common + [
        helper.make_node("Reciprocal", ["sd"], ["r0"]),
        # y <- y * (1.5 - 0.5 * ve * y^2), twice: the inverse square root to near full precision
        helper.make_node("Mul", ["r0", "r0"], ["a0"]), helper.make_node("Mul", ["a0", "ve"], ["b0"]), helper.make_node("Mul", ["b0", "half"], ["c0"]),
        helper.make_node("Sub", ["threehalf", "c0"], ["d0"]), helper.make_node("Mul", ["r0", "d0"], ["r1"]),
        helper.make_node("Mul", ["r1", "r1"], ["a1"]), helper.make_node("Mul", ["a1", "ve"], ["b1"]), helper.make_node("Mul", ["b1", "half"], ["c1"]),
        helper.make_node("Sub", ["threehalf", "c1"], ["d1"]), helper.make_node("Mul", ["r1", "d1"], ["r2"]),
        helper.make_node("Mul", ["x", "r2"], ["y"])], [k("eps", eps), k("half", 0.5), k("threehalf", 1.5)]),
}
for way, (nodes, inits) in WAYS.items():
    d = f"toys/div_{way}"
    os.makedirs(f"{d}/calib", exist_ok=True)
    g = helper.make_graph(nodes, d, [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape), helper.make_tensor_value_info("v", TensorProto.FLOAT, v.shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, x.shape)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, f"{d}/middle.onnx")
    np.save(f"{d}/calib/x0.npy", x); np.save(f"{d}/calib/x1.npy", v)
    open(f"{d}/dataset.txt", "w").write("calib/x0.npy calib/x1.npy\n")
print("toys:", len(WAYS))
