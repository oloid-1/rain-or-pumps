"""
Uploads data/training and the model code to Kaggle as a private dataset, then
pushes a kernel from kaggle/ as a private GPU run against it.

    python kaggle/push.py --script run_sim.py             dataset + simulator kernel
    python kaggle/push.py --script run_bilstm.py          dataset + comparison BiLSTM kernel
    python kaggle/push.py --kernel-only --script ...      dataset unchanged
    python kaggle/push.py --fetch --script run_sim.py     download the output to data/kaggle/

Needs the kaggle CLI logged in (~/.kaggle) and python models/build_sequences.py run first.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "training"
STAGE = REPO / "data" / "kaggle_stage"
DATA_FILES = ["tabular.parquet", "feature_spec.csv", "seq_channels.npz", "build_report.txt"]
CODE_FILES = ["models/transformer.py", "models/train_transformer.py", "models/bilstm.py",
              "models/train_bilstm.py", "models/build_training_data.py", "models/features.py",
              "simulator/train_sim.py"]
DATASET = "rain-or-pumps-bilstm-data"
KERNEL = "rain-or-pumps-bilstm"


def kaggle(*args):
    print("$ kaggle", " ".join(args))
    return subprocess.run(["kaggle", *args], check=True, capture_output=True, text=True).stdout


def user():
    out = kaggle("config", "view")
    return next(l.split(":", 1)[1].strip() for l in out.splitlines() if "username" in l)


def push_dataset(owner):
    d = STAGE / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for f in DATA_FILES:
        shutil.copy2(OUT / f, d / f)
    for f in CODE_FILES:
        shutil.copy2(REPO / f, d / Path(f).name)
    upload(d, owner, DATASET, "rain-or-pumps bilstm data")


def upload(d, owner, slug, title):
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }, indent=2))
    exists = slug in kaggle("datasets", "list", "--mine", "-s", slug)
    if exists:
        print(kaggle("datasets", "version", "-p", str(d), "-m", "rebuild", "-r", "skip"))
    else:
        print(kaggle("datasets", "create", "-p", str(d), "-r", "skip"))


def kernel_name(script):
    stem = Path(script).stem.removeprefix("run_").replace("_", "-")
    return KERNEL if stem == "bilstm" else f"{KERNEL}-{stem}"


def push_kernel(owner, script):
    name = kernel_name(script)
    k = STAGE / name
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir(parents=True)
    shutil.copy2(REPO / "kaggle" / script, k / script)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{name}",
        "title": name,
        "code_file": script,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "dataset_sources": [f"{owner}/{DATASET}"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))
    print(kaggle("kernels", "push", "-p", str(k)))
    print(f"status:  kaggle kernels status {owner}/{name}")


def fetch(owner, script):
    name = kernel_name(script)
    dest = REPO / "data" / "kaggle" / name
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "status", f"{owner}/{name}"))
    print(kaggle("kernels", "output", f"{owner}/{name}", "-p", str(dest)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-only", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--script", default="run_sim.py", help="kernel entry point in kaggle/")
    a = ap.parse_args()
    owner = user()
    if a.fetch:
        return fetch(owner, a.script)
    if not a.kernel_only:
        push_dataset(owner)
    push_kernel(owner, a.script)


if __name__ == "__main__":
    main()
