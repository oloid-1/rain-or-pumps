"""
Loading and scaling the training data for the BiLSTM models. No torch here, so the
app's data build and the API run without it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_training_data import TARGET

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


STATIC_NUMERIC = ["well_depth_m", "sy", "days_since_prev", "rain_normal_annual_mm",
                  "rain_normal_season_mm", "rain_normal_monsoon_mm"]
STATIC_CATEGORICAL = ["season", "aquifer", "well_type", "transition"]


def encode_static(tab, train_mask, stats=None):
    """Standardise the static numerics and index the categoricals with statistics
    from TRAIN rows, or with `stats` from a trained model's sim_meta.json, and
    return the statistics so the browser can repeat it exactly."""
    X = tab[STATIC_NUMERIC].to_numpy("float32")
    if stats:
        med, mu, sd = (np.asarray(stats[k], "float32") for k in ("median", "mean", "std"))
        X = np.where(np.isfinite(X), X, med)
    else:
        med = np.nanmedian(X[train_mask], axis=0)
        X = np.where(np.isfinite(X), X, med)
        mu, sd = X[train_mask].mean(0), X[train_mask].std(0)
        sd[sd < 1e-6] = 1.0
    Xn = ((X - mu) / sd).astype("float32")

    codes, levels_out = [], {}
    for c in STATIC_CATEGORICAL:
        v = tab[c].astype(str).fillna("unknown")
        levels = stats["categorical"][c] if stats else sorted(v[train_mask].unique())
        idx = pd.Index(levels).get_indexer(v)
        codes.append(np.where(idx < 0, len(levels), idx).astype("int64"))
        levels_out[c] = levels
    stats = dict(numeric=STATIC_NUMERIC, median=med.tolist(), mean=mu.tolist(),
                 std=sd.tolist(), categorical=levels_out)
    return Xn, np.stack(codes, 1), [len(levels_out[c]) + 1 for c in STATIC_CATEGORICAL], stats
