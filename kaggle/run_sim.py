"""
Kaggle kernel: trains the simulator (simulator/train_sim.py) on the 6-channel data,
year split, then the 5 district folds. Output in /kaggle/working/sim_6ch/.
ONNX export runs locally from sim.pt (python simulator/export_onnx.py).

Pushed with:  python kaggle/push.py --script run_sim.py
"""

import glob
import os
import subprocess
import sys

data = os.path.dirname(glob.glob("/kaggle/input/**/seq_channels.npz", recursive=True)[0])
code = os.path.dirname(glob.glob("/kaggle/input/**/train_sim.py", recursive=True)[0])
out = "/kaggle/working/sim_6ch"
print("data", data, "\ncode", code, flush=True)


def run(*args):
    cmd = [sys.executable, "-u", os.path.join(code, "train_sim.py"), "--data", data, "--out", out, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=code)


run("--no-export")
run("--cv", "--epochs", "20")
