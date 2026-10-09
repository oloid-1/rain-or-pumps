"""
Kaggle kernel: the two follow-ups to the first BiLSTM run.

    1. transformer, 5 district folds   the transformer's unseen-district number, never run
                                       before, so the BiLSTM's CV has something to compare to
    2. bilstm without rain_anom_mm     the anomaly channel carries the well's normal rain,
                                       which can act as a well identifier; this shows how much
                                       of the sequence-only R2 depends on it

Pushed with:  python kaggle/push.py --kernel-only --script run_followup.py
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

hits = glob.glob("/kaggle/input/**/seq_channels.npz", recursive=True)
assert hits, "attach the rain-or-pumps-bilstm-data dataset"
data = os.path.dirname(hits[0])
code = os.path.dirname(glob.glob("/kaggle/input/**/train_bilstm.py", recursive=True)[0])
work = Path("/kaggle/working")
print("data", data, "code", code, flush=True)

# ---- 1. transformer CV. It reads data/training/{tabular.parquet, feature_spec.csv,
# rain_seq.npz}; rain_seq is channel 0 of seq_channels, byte for byte.
tdir = work / "training"
tdir.mkdir(exist_ok=True)
for f in ("tabular.parquet", "feature_spec.csv"):
    os.symlink(os.path.join(data, f), tdir / f)
z = np.load(hits[0])
np.savez(tdir / "rain_seq.npz", weeks=z["x"][:, :, 0], row_index=np.arange(len(z["x"])))
del z

runner = (
    "import sys, pathlib; sys.path.insert(0, %r); import train_transformer as tt; "
    "tt.T = pathlib.Path(%r); sys.argv = ['train_transformer.py'] + sys.argv[1:]; tt.main()"
) % (code, str(tdir))
cmd = [sys.executable, "-u", "-c", runner, "--cv", "--epochs", "20",
       "--out", str(work / "transformer_cv.json")]
print("\n$ transformer --cv --epochs 20", flush=True)
subprocess.run(cmd, check=True, cwd=code)

# ---- 2. bilstm without the anomaly channel
cmd = [sys.executable, "-u", os.path.join(code, "train_bilstm.py"), "--data", data,
       "--out", str(work / "no_anom"), "--epochs", "25", "--no-baselines",
       "--channels", "rain_mm,in_interval,doy_sin,doy_cos,wet_frac"]
print("\n$", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=code)
