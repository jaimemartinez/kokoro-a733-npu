"""Where to add a normalization's epsilon so the NPU keeps it: the inverse deviation 1/sqrt(var + 1e-5) for per-channel
variances var in [1e-4, 1e-3] (sums over N = 24064 values), three ways:
  pre: var + eps arrives already added (the chip only takes sqrt and reciprocal);
  epsin: var + e, with e = 1e-5 as an input tensor instead of a constant;
  scaled: (sum + eps * N) * inv, adding while the numbers are large (sum = var * N, inv = 1 / N) - the proposed fix.
toys/eps_<way>; test inputs /tmp/eps_*.f32; exact /tmp/epsref.npy."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

shape, one, N, eps = [1, 128, 1, 1], [1, 1, 1, 1], 24064, 1e-5
var = np.random.default_rng(0).uniform(1e-4, 1e-3, shape).astype(np.float32)
arrays = {"var": var, "vpe": (var + eps).astype(np.float32), "e": np.full(one, eps, np.float32),
          "sum": (var * N).astype(np.float32), "en": np.full(one, eps * N, np.float32), "inv": np.full(one, 1.0 / N, np.float32)}
for k, a in arrays.items():
    a.tofile(f"/tmp/eps_{k}.f32")
np.save("/tmp/epsref.npy", 1 / np.sqrt(var.astype(np.float64) + eps))
tail = lambda x: [helper.make_node("Sqrt", [x], ["s"]), helper.make_node("Reciprocal", ["s"], ["y"])]
WAYS = {
    "pre": (["vpe"], tail("vpe")),
    "epsin": (["var", "e"], [helper.make_node("Add", ["var", "e"], ["a"])] + tail("a")),
    "scaled": (["sum", "en", "inv"], [helper.make_node("Add", ["sum", "en"], ["b"]), helper.make_node("Mul", ["b", "inv"], ["a"])] + tail("a")),
}
for way, (ins, nodes) in WAYS.items():
    d = f"toys/eps_{way}"
    os.makedirs(f"{d}/calib", exist_ok=True)
    g = helper.make_graph(nodes, d, [helper.make_tensor_value_info(i, TensorProto.FLOAT, arrays[i].shape) for i in ins],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, f"{d}/middle.onnx")
    for j, i in enumerate(ins):
        np.save(f"{d}/calib/x{j}.npy", arrays[i])
    open(f"{d}/dataset.txt", "w").write(" ".join(f"calib/x{j}.npy" for j in range(len(ins))) + "\n")
    open(f"{d}/inputs.txt", "w").write(" ".join(f"/tmp/eps_{i}.f32" for i in ins) + "\n")
print("toys:", len(WAYS))
