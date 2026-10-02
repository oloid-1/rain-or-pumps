"""
Uploads bilstm-data/out and the model code to Kaggle as a private dataset, then
pushes kaggle/run_bilstm.py as a private GPU kernel against it.

    python kaggle/push.py                 dataset (create or new version) + kernel
    python kaggle/push.py --kernel-only   code unchanged on the dataset side
    python kaggle/push.py --kernel-only --script run_followup.py   a second kernel, same dataset
    python kaggle/push.py --fetch         download the kernel's output to bilstm-data/out/kaggle_run/

Needs the kaggle CLI logged in (~/.kaggle). Build the data first:
    python bilstm-data/build_bilstm_data.py
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "bilstm-data" / "out"
STAGE = REPO / "bilstm-data" / "kaggle_stage"
DATA_FILES = ["tabular.parquet", "feature_spec.csv", "seq_channels.npz", "build_report.txt"]
CODE_FILES = ["transformer.py", "train_transformer.py", "bilstm.py", "train_bilstm.py"]
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
        shutil.copy2(REPO / "models" / f, d / f)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "rain-or-pumps bilstm data",
        "id": f"{owner}/{DATASET}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }, indent=2))
    exists = DATASET in kaggle("datasets", "list", "--mine", "-s", DATASET)
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
    dest = OUT / ("kaggle_run" if name == KERNEL else f"kaggle_{name.removeprefix(KERNEL + '-')}")
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "status", f"{owner}/{name}"))
    print(kaggle("kernels", "output", f"{owner}/{name}", "-p", str(dest)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-only", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--script", default="run_bilstm.py", help="kernel entry point in kaggle/")
    a = ap.parse_args()
    owner = user()
    if a.fetch:
        return fetch(owner, a.script)
    if not a.kernel_only:
        push_dataset(owner)
    push_kernel(owner, a.script)


if __name__ == "__main__":
    main()
