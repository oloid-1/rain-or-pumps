"""
Exports a simulator run trained elsewhere (Kaggle) to ONNX for the browser.

    python simulator/export_onnx.py --run data/kaggle/rain-or-pumps-bilstm-sim/sim_6ch

Rebuilds the scaling statistics from the same training rows, loads sim.pt, and
writes sim.onnx and sim_meta.json into the run folder. Then it checks that ONNX
and torch agree on 500 real rows.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "models")]
from bilstm import RainBiLSTM  # noqa: E402
from train_bilstm import load, channel_stats, scale  # noqa: E402
from train_sim import encode_static, export, STATIC_NUMERIC  # noqa: E402
from build_training_data import OUT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=str(OUT_DIR))
    a = ap.parse_args()
    run = Path(a.run)

    tab, x, channels, _, _ = load(a.data)
    mk = (tab.split == "train").to_numpy()
    cs = channel_stats(x, channels, mk)
    S = scale(x, channels, cs)
    Xn, C, sizes, tstats = encode_static(tab, mk)
    m = RainBiLSTM(len(STATIC_NUMERIC), sizes, len(channels))
    m.load_state_dict(torch.load(run / "sim.pt", map_location="cpu"))
    res = json.loads((run / "results.json").read_text())
    export(m, S, Xn, C, channels, cs, tstats, res["valid"], res["response_valid"], run)

    import onnxruntime as ort
    k = np.random.default_rng(0).choice(len(tab), 500, replace=False)
    po = ort.InferenceSession(str(run / "sim.onnx")).run(None, {"seq": S[k], "num": Xn[k], "cat": C[k]})[0]
    with torch.no_grad():
        pt = m(torch.from_numpy(S[k]), torch.from_numpy(Xn[k]), torch.from_numpy(C[k])).numpy()
    print(f"exported {run / 'sim.onnx'}; onnx vs torch max diff {np.abs(po - pt).max():.2e}")


if __name__ == "__main__":
    main()
