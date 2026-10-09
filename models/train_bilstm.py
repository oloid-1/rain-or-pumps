"""
Trains the rain BiLSTM on the bilstm-data copy and prints the same comparison
table as train_transformer.py, on the same rows and the same split.

    python models/train_bilstm.py                         6 channels, ablations, baselines
    python models/train_bilstm.py --epochs 2 --no-baselines --limit 20000   smoke test
    python models/train_bilstm.py --channels rain_mm      the transformer's exact input
    python models/train_bilstm.py --cv                    5 district folds
    python models/train_bilstm.py --data ../data/training the main build, rain only, same settings

Runs, in order:
    baselines         zero, season mean, ridge, lightgbm (unchanged from the transformer script)
    bilstm            the chosen channels + tabular arm
    bilstm 1ch        rain_mm only + tabular arm: same input as the transformer, so
                      bilstm vs bilstm 1ch isolates what the extra channels add, and
                      bilstm 1ch vs transformer isolates the architecture
    bilstm seq only   the chosen channels, no tabular arm

Writes to --out (default bilstm-data/out/run/):
    results.json, bilstm.pt, pooling_by_season.png/.csv, residual_by_district.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bilstm import RainBiLSTM, SeqOnlyBiLSTM
from train_transformer import encode, metrics, baselines, run_nn, TARGET, SEED
from transformer import count_params

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "bilstm-data" / "out"

# how each channel is scaled; statistics from train rows only
LOG_CHANNELS = {"rain_mm"}            # log1p, then standardise
SIGNED_LOG_CHANNELS = {"rain_anom_mm"}  # sign * log1p(|x|), then standardise
# in_interval, doy_sin, doy_cos, wet_frac are already in [-1, 1] and are left as is


def load(data_dir):
    data_dir = Path(data_dir)
    tab = pd.read_parquet(data_dir / "tabular.parquet")
    spec = pd.read_csv(data_dir / "feature_spec.csv")
    if (data_dir / "seq_channels.npz").exists():
        z = np.load(data_dir / "seq_channels.npz")
        x, channels = z["x"], [str(c) for c in z["channels"]]
    else:
        # the main build (data/training/): one rain channel, the transformer's input
        x, channels = np.load(data_dir / "rain_seq.npz")["weeks"][:, :, None], ["rain_mm"]
    assert len(tab) == len(x), (len(tab), len(x))

    feats = spec.loc[spec.role == "feature", "column"].tolist()
    cats = spec.loc[(spec.role == "feature") & (spec.kind == "categorical"), "column"].tolist()
    nums = [c for c in feats if c not in cats]

    forbidden = set(spec.loc[spec.role == "meta", "column"]) | {TARGET}
    leak = [c for c in feats if c in forbidden]
    assert not leak, f"leakage: {leak}"
    return tab, x, channels, nums, cats


def channel_stats(x, channels, train_mask):
    stats = {}
    for i, c in enumerate(channels):
        if c in LOG_CHANNELS or c in SIGNED_LOG_CHANNELS:
            v = _squash(x[train_mask, :, i].astype("float32"), c)
            stats[c] = (float(np.nanmean(v)), max(float(np.nanstd(v)), 1e-6))
    return stats


def _squash(v, c):
    if c in LOG_CHANNELS:
        return np.log1p(np.clip(v, 0, None))
    if c in SIGNED_LOG_CHANNELS:
        return np.sign(v) * np.log1p(np.abs(v))
    return v


def scale(x, channels, stats):
    out = np.empty(x.shape, "float32")
    for i, c in enumerate(channels):
        v = _squash(x[:, :, i].astype("float32"), c)
        if c in stats:
            mu, sd = stats[c]
            v = (v - mu) / sd
        out[:, :, i] = np.where(np.isfinite(v), v, 0.0)
    return out


def rain_scaled(x, channels, factor):
    """The same sequences with every week's rain multiplied by factor.

    The anomaly moves with it (anom + (factor-1) * rain); in_interval and the
    calendar do not change; wet_frac is left as is, a small understatement.
    """
    y = x.astype("float32").copy()
    r = channels.index("rain_mm")
    if "rain_anom_mm" in channels:
        a = channels.index("rain_anom_mm")
        y[:, :, a] += (factor - 1.0) * y[:, :, r]
    y[:, :, r] *= factor
    return y


@torch.no_grad()
def predict_all(model, S, Xn, C, idx, device):
    model.eval()
    p = []
    for a in range(0, len(idx), 4096):
        k = idx[a:a + 4096]
        p.append(model(torch.from_numpy(S[k]).to(device),
                       torch.from_numpy(Xn[k]).to(device),
                       torch.from_numpy(C[k]).to(device)).cpu().numpy())
    return np.concatenate(p)


def build_model(kind, a, n_ch, n_num, sizes):
    kw = dict(d_in=a.d_in, hidden=a.hidden, n_layers=a.layers, d_model=a.d_model,
              dropout=a.dropout, pool=a.pool)
    if kind == "seq":
        return SeqOnlyBiLSTM(n_ch, **kw)
    return RainBiLSTM(n_num, sizes, n_ch, **kw)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(DATA / "run"))
    ap.add_argument("--channels", default="all",
                    help="'all' or a comma list, e.g. rain_mm,in_interval")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--d-in", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", default="attn", choices=["attn", "mean", "last"])
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--cv", action="store_true", help="5 district folds instead of the year split")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="subsample rows, for smoke tests")
    a = ap.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tab, x, channels, nums, cats = load(a.data)
    if a.limit:
        keep = np.sort(np.random.default_rng(SEED).choice(len(tab), a.limit, replace=False))
        tab, x = tab.iloc[keep].reset_index(drop=True), x[keep]
    x = x[:, -a.weeks:]
    use = channels if a.channels == "all" else a.channels.split(",")
    missing = [c for c in use if c not in channels]
    assert not missing, f"unknown channels {missing}; have {channels}"
    x = x[:, :, [channels.index(c) for c in use]]
    y = tab[TARGET].to_numpy("float32")
    print(f"rows {len(tab):,}  numeric {len(nums)}  categorical {len(cats)}  "
          f"weeks {x.shape[1]}  channels {use}  device {device}")

    results = {"config": vars(a), "rows": int(len(tab)), "channels": use}

    def fit(kind, chans, tr, va, mk, label):
        xi = x[:, :, [use.index(c) for c in chans]]
        S = scale(xi, chans, channel_stats(xi, chans, mk))
        Xn, C, sizes = encode(tab, nums, cats, mk)
        m = build_model(kind, a, len(chans), len(nums), sizes)
        print(f"[{label}] params {count_params(m):,}")
        t0 = time.time()
        m, vm, hist, _ = run_nn(m, S, None, Xn, C, y, tr, va, a.epochs, a.bs, a.lr,
                                a.patience, device=device, label=label)
        vm["minutes"] = round((time.time() - t0) / 60, 1)
        return m, vm, hist, S, Xn, C, xi, chans

    if a.cv:
        agg = []
        for f in sorted(tab.district_fold.unique()):
            tr = np.flatnonzero((tab.district_fold != f).to_numpy())
            va = np.flatnonzero((tab.district_fold == f).to_numpy())
            mk = np.zeros(len(tab), bool); mk[tr] = True
            _, vm, _, *_ = fit("full", use, tr, va, mk, f"fold{f}")
            agg.append(vm)
        results["cv_folds"] = agg
        results["cv_mean"] = {k: float(np.mean([d[k] for d in agg])) for k in agg[0]}
        print("\ndistrict-fold CV mean:", results["cv_mean"])
        (out / "results_cv.json").write_text(json.dumps(results, indent=2))
        return

    tr = np.flatnonzero((tab.split == "train").to_numpy())
    va = np.flatnonzero((tab.split == "valid").to_numpy())
    te = np.flatnonzero((tab.split == "test").to_numpy())
    mk = np.zeros(len(tab), bool); mk[tr] = True

    if not a.no_baselines:
        t0 = time.time()
        results["baselines"] = baselines(tab, nums, cats, tr, va)
        print(f"baselines done in {time.time()-t0:.0f}s")

    m, vm, hist, S, Xn, C, xi, chans = fit("full", use, tr, va, mk, "bilstm")
    results["bilstm"] = vm
    results["history"] = hist
    pred = predict_all(m, S, Xn, C, np.arange(len(tab)), device)
    results["bilstm_test"] = metrics(y[te], pred[te])
    results["bilstm_train"] = metrics(y[tr], pred[tr])
    torch.save(m.state_dict(), out / "bilstm.pt")

    # more rain should never predict a deeper fall; check it on valid rows
    stats = channel_stats(xi, chans, mk)
    sens = {}
    if "rain_mm" in chans:
        for fct in (0.8, 1.2):
            Sf = scale(rain_scaled(xi[va], chans, fct), chans, stats)
            pf = predict_all(m, Sf, Xn[va], C[va], np.arange(len(va)), device)
            sens[f"x{fct}"] = float((pf - pred[va]).mean())
        print(f"rain sensitivity on valid, mean change in predicted delta_h_m: {sens} "
              "(want x0.8 > 0 > x1.2)")
    results["rain_sensitivity"] = sens

    # residual hook, same sign convention as attention_report.py
    res = (tab.assign(resid=y - pred).groupby("district")
           .agg(resid_mean=("resid", "mean"), n=("resid", "size"),
                wells=("well_uid", "nunique"))
           .query("wells >= 3").sort_values("resid_mean", ascending=False))
    res.to_csv(out / "residual_by_district.csv")

    if a.pool == "attn":
        pooling_figure(m, S, tab, device, out)

    rows = [(k, v) for k, v in (results.get("baselines") or {}).items() if not k.startswith("_")]
    rows.append(("bilstm", vm))

    if not a.no_ablations:
        if use != ["rain_mm"] and "rain_mm" in use:
            _, v1, *_ = fit("full", ["rain_mm"], tr, va, mk, "bilstm 1ch")
            results["bilstm_1ch"] = v1
            rows.append(("bilstm 1ch", v1))
        _, vs, *_ = fit("seq", use, tr, va, mk, "bilstm seq only")
        results["bilstm_seq_only"] = vs
        rows.append(("bilstm seq only", vs))

    print("\nvalid 2015-2017")
    print(f"{'model':<22}{'mae':>9}{'rmse':>9}{'r2':>9}")
    for k, v in rows:
        print(f"{k:<22}{v['mae']:>9.4f}{v['rmse']:>9.4f}{v['r2']:>9.4f}")
    print(f"\nbilstm on held-out test 2018-2022: {results['bilstm_test']}")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


def pooling_figure(m, S, tab, device, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["JAN", "MAY", "AUG", "NOV"]
    T = S.shape[1]
    weeks_ago = np.arange(T)[::-1] + 1
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    table = {"weeks_before_reading": weeks_ago}
    m.eval()
    for ax, s in zip(axes, order):
        k = np.flatnonzero(((tab.season == s) & (tab.split == "valid")).to_numpy())[:2000]
        if len(k) == 0:
            continue
        w = m.seq.pooling_weights(torch.from_numpy(S[k]).to(device)).mean(0).cpu().numpy()
        table[s] = w
        ax.bar(weeks_ago, w, width=0.9)
        ax.axhline(1 / T, color="grey", lw=0.8, ls="--")
        ax.invert_xaxis()
        ax.set_title(f"{s} campaign  (n={len(k)})")
        ax.set_xlabel("weeks before reading")
    axes[0].set_ylabel("pooling weight")
    fig.suptitle("Which weeks of rainfall the BiLSTM pools, by campaign season "
                 "(dashed: uniform)", y=1.04)
    fig.tight_layout()
    fig.savefig(out / "pooling_by_season.png", dpi=150, bbox_inches="tight")
    pd.DataFrame(table).to_csv(out / "pooling_by_season.csv", index=False)


if __name__ == "__main__":
    main()
