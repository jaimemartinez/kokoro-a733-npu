"""How exact the NPU's multiply is on a long tensor: x [1,128,1,24064] ~ N(0, 0.02) times
  bcast_in: a per-channel input s [1,128,1,1] in [30, 100] (as 1/std),
  bcast_const: the same s as a constant,
  full: an input of x's own shape (s repeated along the row),
  dwconv: a depthwise 1x1 convolution with s as its weights (the NN core's own per-channel multiply).
Test inputs /tmp/mulx.f32, /tmp/muls.f32, /tmp/mulsfull.f32; exact result /tmp/mulref.npy."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

C, L = 128, 24064
rng = np.random.default_rng(0)
x = (rng.standard_normal((1, C, 1, L)) * 0.02).astype(np.float32)
s = rng.uniform(30, 100, (1, C, 1, 1)).astype(np.float32)
sf = np.broadcast_to(s, x.shape).astype(np.float32)
x.tofile("/tmp/mulx.f32"); s.tofile("/tmp/muls.f32"); sf.tofile("/tmp/mulsfull.f32")
np.save("/tmp/mulref.npy", x.astype(np.float64) * s.astype(np.float64))
vi = lambda n, a: helper.make_tensor_value_info(n, TensorProto.FLOAT, a.shape)
WAYS = {
    "bcast_in": ([helper.make_node("Mul", ["x", "s"], ["y"])], [vi("x", x), vi("s", s)], [], [x, s]),
    "bcast_const": ([helper.make_node("Mul", ["x", "sc"], ["y"])], [vi("x", x)], [numpy_helper.from_array(s, "sc")], [x]),
    "full": ([helper.make_node("Mul", ["x", "s"], ["y"])], [vi("x", x), vi("s", sf)], [], [x, sf]),
    "dwconv": ([helper.make_node("Conv", ["x", "w"], ["y"], group=C, kernel_shape=[1, 1])], [vi("x", x)],
               [numpy_helper.from_array(s.reshape(C, 1, 1, 1), "w")], [x]),
}
for way, (nodes, ins, inits, calib) in WAYS.items():
    d = f"toys/mul_{way}"
    os.makedirs(f"{d}/calib", exist_ok=True)
    g = helper.make_graph(nodes, d, ins, [vi("y", x)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, f"{d}/middle.onnx")
    for j, a in enumerate(calib):
        np.save(f"{d}/calib/x{j}.npy", a)
    open(f"{d}/dataset.txt", "w").write(" ".join(f"calib/x{j}.npy" for j in range(len(calib))) + "\n")
print("toys:", len(WAYS))
