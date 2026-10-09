"""
Kaggle kernel: the BiLSTM on the main data, and the simulator model on both datasets.

    main_bilstm/        models/train_bilstm.py on data/training/ (rain only), same
                        settings as the bilstm-data run: year split + 5 district folds
    sim_6ch/            simulator/train_sim.py on bilstm-data, rain-response penalty
    sim_6ch_nomono/     the same with the penalty off, to show what the penalty buys
    sim_1ch/            simulator on data/training/, penalty on
    sim_1ch_nomono/     simulator on data/training/, penalty off
    sim_6ch/results_cv.json   the simulator on unseen districts

ONNX export runs locally from sim.pt (python simulator/export_onnx.py), so the
kernel does not depend on the onnx package.

Pushed with:  python kaggle/push.py --main-data --script run_sim.py
"""

import glob
import os
import subprocess
import sys

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"no {name} under /kaggle/input"
    return os.path.dirname(hits[0])

bilstm_data = find("seq_channels.npz")
main_data = find("rain_seq.npz")
code = find("train_sim.py")
W = "/kaggle/working"
# version 1 ran the main-data BiLSTM and then failed on the simulator (cuDNN refuses an
# RNN backward in eval mode); its outputs are kept, so later versions skip it
RUN_MAIN_BILSTM = False
print("bilstm-data", bilstm_data, "\nmain data", main_data, "\ncode", code, flush=True)


def run(script, *args):
    cmd = [sys.executable, "-u", os.path.join(code, script), *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=code)


if RUN_MAIN_BILSTM:
    run("train_bilstm.py", "--data", main_data, "--out", f"{W}/main_bilstm", "--epochs", "25", "--no-baselines")
    run("train_bilstm.py", "--data", main_data, "--out", f"{W}/main_bilstm", "--cv", "--epochs", "20")

for name, data, mono in (("sim_6ch", bilstm_data, "1.0"), ("sim_6ch_nomono", bilstm_data, "0"),
                         ("sim_1ch", main_data, "1.0"), ("sim_1ch_nomono", main_data, "0")):
    run("train_sim.py", "--data", data, "--out", f"{W}/{name}", "--mono-weight", mono, "--no-export")

run("train_sim.py", "--data", bilstm_data, "--out", f"{W}/sim_6ch", "--cv", "--epochs", "20")
