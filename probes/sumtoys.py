"""Which way of summing 24064 values per channel the A733 NPU gets right (its ReduceSum is far off on real data):
toys/sum_<way> over x [1,128,47,512] -> [1,128,1,1]. Ways: reduce (ReduceSum 512 then 47, as now), mean (ReduceMean,
times the count), conv (depthwise convolutions with all-ones kernels, [1,512] stride 512 then [47,1]), pool (average
pooling, times the count). Calibrated and tested on squares of N(0, 0.2) (like a variance sum) and on N(0, 1);
test inputs /tmp/sumin_<kind>.f32 and exact sums /tmp/sumref_<kind>.npy."""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

C, A, W = 128, 47, 512
shape, oshape = [1, C, A, W], [1, C, 1, 1]
rng = np.random.default_rng(0)
data = {"sq": (rng.standard_normal(shape) * 0.2) ** 2, "signed": rng.standard_normal(shape)}
for kind, v in data.items():
    v = v.astype(np.float32)
    v.tofile(f"/tmp/sumin_{kind}.f32")
    np.save(f"/tmp/sumref_{kind}.npy", v.astype(np.float64).sum(axis=(2, 3)).reshape(oshape))
c = lambda n, v: numpy_helper.from_array(np.asarray(v), n)
WAYS = {
    "reduce": ([helper.make_node("ReduceSum", ["x", "a3"], ["s"], keepdims=1), helper.make_node("ReduceSum", ["s", "a2"], ["y"], keepdims=1)],
               [c("a3", np.array([3], np.int64)), c("a2", np.array([2], np.int64))]),
    "mean": ([helper.make_node("ReduceMean", ["x", "a23"], ["m"], keepdims=1), helper.make_node("Mul", ["m", "n"], ["y"])],
             [c("a23", np.array([2, 3], np.int64)), c("n", np.array(A * W, np.float32))]),
    "conv": ([helper.make_node("Conv", ["x", "w1"], ["s"], group=C, kernel_shape=[1, W], strides=[1, W]),
              helper.make_node("Conv", ["s", "w2"], ["y"], group=C, kernel_shape=[A, 1])],
             [c("w1", np.ones((C, 1, 1, W), np.float32)), c("w2", np.ones((C, 1, A, 1), np.float32))]),
    "pool": ([helper.make_node("GlobalAveragePool", ["x"], ["m"]), helper.make_node("Mul", ["m", "n"], ["y"])],
             [c("n", np.array(A * W, np.float32))]),
}
for kind in data:
    for way, (nodes, inits) in WAYS.items():
        d = f"toys/sum_{way}_{kind}"
        os.makedirs(f"{d}/calib", exist_ok=True)
        g = helper.make_graph(nodes, d, [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, oshape)], inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, f"{d}/middle.onnx")
        np.save(f"{d}/calib/x0.npy", data[kind].astype(np.float32))
        open(f"{d}/dataset.txt", "w").write("calib/x0.npy\n")
print("toys:", len(WAYS) * len(data))
