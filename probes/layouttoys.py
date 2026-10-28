"""Which tensor layout the A733 NPU runs long sequences best in: the snake activation, a long sum and a plain add,
each on 128 x 24001 values as [1,C,L], [1,C,1,L], [1,C,L,1] and [1,C,47,512] (padded): toys/lay_<op>_<layout>."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

rng = np.random.default_rng(0)
C, L = 128, 24001
LAYOUTS = {"3d": [1, C, L], "h1": [1, C, 1, L], "w1": [1, C, L, 1], "47x512": [1, C, 47, 512]}


def chan(shape, v):
    """A per-channel constant broadcast against `shape`."""
    return v.reshape([1, C] + [1] * (len(shape) - 2)).astype(np.float32)


for lname, shape in LAYOUTS.items():
    last = len(shape) - 1
    a = rng.uniform(0.5, 3, C)
    for op in ("snake", "sum", "add"):
        d = os.path.join("toys", f"lay_{op}_{lname}")
        os.makedirs(os.path.join(d, "calib"), exist_ok=True)
        inits, ins = [], [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
        if op == "snake":
            inits = [numpy_helper.from_array(chan(shape, a), "a"), numpy_helper.from_array(chan(shape, 1 / a), "ia"),
                     numpy_helper.from_array(np.array(2, np.float32), "two")]
            nodes = [helper.make_node("Mul", ["a", "x"], ["ax"]), helper.make_node("Sin", ["ax"], ["s"]),
                     helper.make_node("Pow", ["s", "two"], ["s2"]), helper.make_node("Mul", ["ia", "s2"], ["q"]),
                     helper.make_node("Add", ["x", "q"], ["y"])]
            oshape = shape
        elif op == "sum":
            axes = [2, 3] if lname == "47x512" else [2] if lname in ("3d", "w1") else [3]
            inits = [numpy_helper.from_array(np.array(axes, np.int64), "axes")]
            nodes = [helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)]
            oshape = [1, C] + [1] * (len(shape) - 2)
        else:
            ins.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, shape))
            nodes = [helper.make_node("Add", ["x", "z"], ["y"])]
            oshape = shape
        g = helper.make_graph(nodes, d, ins, [helper.make_tensor_value_info("y", TensorProto.FLOAT, oshape)], inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, os.path.join(d, "middle.onnx"))
        files = []
        for j, i in enumerate(ins):
            np.save(os.path.join(d, "calib", f"x{j}.npy"), (rng.standard_normal(shape) * 2).astype(np.float32))
            files.append(f"calib/x{j}.npy")
        open(os.path.join(d, "dataset.txt"), "w").write(" ".join(files) + "\n")
print("toys:", len(LAYOUTS) * 3)
