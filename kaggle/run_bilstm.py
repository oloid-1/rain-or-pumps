"""
Kaggle kernel entry point. Pushed by kaggle/push.py; not meant to run locally.

Finds the attached dataset (bilstm-data/out plus the model code), then runs
    1. the year-split comparison: baselines, bilstm, bilstm 1ch, bilstm seq only
    2. the 5 district folds
Everything lands in /kaggle/working/run/, which is the kernel's output.
"""

import glob
import os
import subprocess
import sys

hits = glob.glob("/kaggle/input/**/seq_channels.npz", recursive=True)
assert hits, "attach the rain-or-pumps-bilstm-data dataset"
data = os.path.dirname(hits[0])
code = os.path.dirname(glob.glob("/kaggle/input/**/train_bilstm.py", recursive=True)[0])
out = "/kaggle/working/run"
print("data", data, "code", code, flush=True)

subprocess.run(["nvidia-smi"], check=False)


def run(*args):
    cmd = [sys.executable, "-u", os.path.join(code, "train_bilstm.py"),
           "--data", data, "--out", out, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=code)


run("--epochs", "25")
run("--cv", "--epochs", "20", "--no-baselines")
