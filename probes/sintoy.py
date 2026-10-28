"""How exact the NPU's sine is by argument size. toys/sinacc: one Sin; toys/sin2red: sin(t)^2 with the argument first
reduced to [-pi/2, pi/2] (t - pi * round(t / pi)), as Kokoro's snake needs it. Both on [1,128,1,24064], calibrated to
+-64; test inputs /tmp/sin_A.f32 with every channel a ramp over [-A, A] for A in 1, 4, 16, 64."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

shape = [1, 128, 1, 24064]
c = lambda name, v: numpy_helper.from_array(np.array(v, np.float32), name)
TOYS = {
    "sinacc": ([helper.make_node("Sin", ["x"], ["y"])], []),
    "sin2red": ([
        helper.make_node("Mul", ["x", "inv_pi"], ["u"]),
        helper.make_node("Round", ["u"], ["k"]),
        helper.make_node("Mul", ["k", "pi"], ["kp"]),
        helper.make_node("Sub", ["x", "kp"], ["r"]),
        helper.make_node("Sin", ["r"], ["s"]),
        helper.make_node("Mul", ["s", "s"], ["y"]),
    ], [c("inv_pi", 1 / np.pi), c("pi", np.pi)]),
}
for name, (nodes, inits) in TOYS.items():
    d = f"toys/{name}"
    os.makedirs(f"{d}/calib", exist_ok=True)
    g = helper.make_graph(nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, f"{d}/middle.onnx")
    np.save(f"{d}/calib/x0.npy", np.linspace(-64, 64, np.prod(shape), dtype=np.float32).reshape(shape))
    open(f"{d}/dataset.txt", "w").write("calib/x0.npy\n")
for A in (1, 4, 16, 64):
    np.linspace(-A, A, np.prod(shape), dtype=np.float32).tofile(f"/tmp/sin_{A}.f32")
print("ok")
