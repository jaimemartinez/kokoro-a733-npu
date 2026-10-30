"""Step 6: one stage made ready for the A733 NPU. Applies the graph rewrites in order (each one checks itself against
ONNX Runtime and refuses to save a changed result), then carries the stage's calibration set and test inputs over to
the new model's inputs: rank-3 tensors get their unit axis, sequences are zero-padded to the new length, and the
epsn_* inputs added by epsfix.py are computed from the matching inv_* (eps / inv).

  python stage.py <src dir> <dst dir> <rewrite> [<rewrite> ...]
  rewrites: to4d len64 sinfix poolfix epsfix decpool decnorm snakeclamp resmask fold
  e.g. dec: epsfix decpool decnorm; g1/g2: to4d sinfix poolfix epsfix snakeclamp;
       g3: to4d len64 sinfix poolfix epsfix snakeclamp resmask; g4: the same, then fold"""
import os
import shutil
import subprocess
import sys
import numpy as np
import onnx

EPS = 1e-5  # Kokoro's instance normalizations
HERE = os.path.dirname(os.path.abspath(__file__))
src, dst, steps = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(os.path.join(dst, "calib"), exist_ok=True)

# The rewrites, one after the other.
model = os.path.join(src, "middle.onnx")
for i, step in enumerate(steps):
    out = os.path.join(dst, f".step{i}.onnx")
    # KOKORO_STAGE_SRC: where a rewrite that measures the data (decnorm.py) finds the calibration set
    subprocess.run([sys.executable, os.path.join(HERE, "rewrites", f"{step}.py"), model, out], check=True,
                   env={**os.environ, "KOKORO_STAGE_SRC": os.path.abspath(src)})
    model = out
os.replace(model, os.path.join(dst, "middle.onnx"))
for f in os.listdir(dst):
    if f.startswith(".step"):
        os.remove(os.path.join(dst, f))


def inputs(path):
    return [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
            for i in onnx.load(path, load_external_data=False).graph.input]


old, new = inputs(os.path.join(src, "middle.onnx")), inputs(os.path.join(dst, "middle.onnx"))
old_names = [n for n, _ in old]


def carry(get, name, shape):
    """The value of new input `name` (shape `shape`) from the old inputs: get(old index) loads one."""
    if name.startswith("epsn_"):
        return (EPS / get(old_names.index("inv_" + name[len("epsn_"):]))).reshape(shape).astype(np.float32)
    v = get(old_names.index(name))
    if v.ndim == 3 and len(shape) == 4:
        v = v.reshape(v.shape[0], v.shape[1], 1, v.shape[2])
    if v.shape[-1] < shape[-1]:
        v = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, shape[-1] - v.shape[-1])])
    return np.ascontiguousarray(v[..., : shape[-1]]).reshape(shape).astype(np.float32)


rows = [r.split() for r in open(os.path.join(src, "dataset.txt")).read().split("\n") if r.strip()]
lines = []
for ri, row in enumerate(rows):
    files = []
    for name, shape in new:
        f = f"calib/r{ri}_{name.strip('/').replace('/', '_')}.npy"
        np.save(os.path.join(dst, f), carry(lambda j: np.load(os.path.join(src, row[j])), name, shape))
        files.append(f)
    lines.append(" ".join(files))
open(os.path.join(dst, "dataset.txt"), "w").write("\n".join(lines) + "\n")

if os.path.exists(os.path.join(src, "in00.f32")):
    def test_input(j):
        return np.fromfile(os.path.join(src, f"in{j:02d}.f32"), np.float32).reshape(old[j][1])
    for j, (name, shape) in enumerate(new):
        carry(test_input, name, shape).tofile(os.path.join(dst, f"in{j:02d}.f32"))
    for f in os.listdir(src):
        if f.startswith("ref_") and f.endswith(".npy"):
            shutil.copy(os.path.join(src, f), dst)
print(f"{dst}: {', '.join(steps) or 'no rewrites'}; {len(rows)} calibration rows; inputs {[n for n, _ in new]}")
