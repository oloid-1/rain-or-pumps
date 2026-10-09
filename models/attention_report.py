"""
Reads the trained transformer back and produces the two figures that a boosted
tree cannot produce.

    python3 ml/attention_report.py

Writes to data/training/
    attention_by_season.png   which of the 26 four-week patches the model reads
    residual_by_district.csv  mean residual per district, the attribution hook
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer import RainTransformer
from train_transformer import load, encode, scale_seq, TARGET, metrics

REPO = Path(__file__).resolve().parents[1]
from build_training_data import OUT_DIR as T  # noqa: E402


def main():
    tab, seq, nums, cats = load()
    tr = (tab.split == "train").to_numpy()
    Xn, C, sizes = encode(tab, nums, cats, tr)
    S, _ = scale_seq(seq, tr)
    y = tab[TARGET].to_numpy("float32")

    m = RainTransformer(len(nums), sizes, seq.shape[1])
    m.load_state_dict(torch.load(T / "transformer.pt", map_location="cpu"))
    m.eval()

    # ---- predictions on everything, for the residual table
    with torch.no_grad():
        p = []
        for a in range(0, len(tab), 4096):
            k = slice(a, a + 4096)
            p.append(m(torch.from_numpy(S[k]), torch.from_numpy(Xn[k]),
                       torch.from_numpy(C[k])).numpy())
    pred = np.concatenate(p)
    tab = tab.assign(pred=pred, resid=y - pred)

    for s in ("train", "valid", "test"):
        k = tab.split == s
        print(s, metrics(y[k.to_numpy()], pred[k.to_numpy()]))

    # Residual sign: delta_h_m is positive when the level fell, so a positive
    # residual means the level fell MORE than the rain accounts for. Averaged
    # over a district and a run of years, that is the extraction-pressure
    # signal the project is after. This is the hook, not the finished
    # attribution: no district effects are fitted here.
    res = (tab.groupby("district")
           .agg(resid_mean=("resid", "mean"), n=("resid", "size"),
                wells=("well_uid", "nunique"))
           .query("wells >= 3")
           .sort_values("resid_mean", ascending=False))
    res.to_csv(T / "residual_by_district.csv")
    print("\ntop 10 districts by unexplained decline")
    print(res.head(10).to_string())

    # ---- attention, averaged per season over valid rows
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    order = ["JAN", "MAY", "AUG", "NOV"]
    for ax, s in zip(axes, order):
        k = np.flatnonzero(((tab.season == s) & (tab.split == "valid")).to_numpy())[:2000]
        with torch.no_grad():
            maps = m.seq.attention_maps(torch.from_numpy(S[k]))
        # how much each patch is attended TO, summed over queries, last layer
        w = maps[-1].mean(dim=0).mean(dim=0).numpy()
        weeks_ago = np.arange(m.seq.n_patch)[::-1] * 4 + 2
        ax.bar(weeks_ago, w, width=3.2)
        ax.invert_xaxis()
        ax.set_title(f"{s} campaign  (n={len(k)})")
        ax.set_xlabel("weeks before reading")
    axes[0].set_ylabel("attention received")
    fig.suptitle("Which weeks of rainfall the encoder reads, by campaign season",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(T / "attention_by_season.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {T/'attention_by_season.png'}")


if __name__ == "__main__":
    main()
