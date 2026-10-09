"""
The mentor-requested comparison: random (well-grouped) validation split,
two hyperparameter configurations each for BiLSTM, LightGBM and the
transformer, results written to one file.

This reuses, unchanged, everything from train_transformer.py that the split
doesn't touch: load(), encode(), scale_seq(), metrics(), run_nn(), and the
ridge/LightGBM code inside baselines(). Only the split itself is new
(random_split.py) and the configs are new. Nothing about the leakage guard,
the Huber loss, AdamW, OneCycle, or the early-stopping rule has changed.

Usage:
    python3 train_random.py --model transformer --config a
    python3 train_random.py --model transformer --config b
    python3 train_random.py --model bilstm --config a
    python3 train_random.py --model bilstm --config b
    python3 train_random.py --model lightgbm --config a
    python3 train_random.py --model lightgbm --config b
    python3 train_random.py --all      # all six, in order

Each call appends/overwrites its own key in random_split_results.json so
the six runs can be done as separate processes (or separate notebooks)
without clobbering each other.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformer import RainTransformer, count_params
from bilstm import BiLSTMModel
from train_transformer import (encode, scale_seq, metrics, run_nn, baselines,
                                TARGET)
from random_split import make_random_split

REPO = Path(__file__).resolve().parents[1]
T = REPO / "data" / "training"
OUT = T / "random_split_results.json"
SEED = 42

# -----------------------------------------------------------------------
# Hyperparameter configs. Two per model, chosen to answer a real question
# rather than two arbitrary points:
#
#  transformer:
#    a - the repo's existing default (d_model 64, 3 layers, 4 heads, lr 1e-3)
#    b - half the width, 2 layers, higher lr: tests whether the result is
#        sensitive to capacity, same question BILSTM_PARAMETERS.md section 7
#        raises for the BiLSTM ("hidden size 32 and 128... to show the result
#        is not sensitive to capacity, not to beat it")
#
#  bilstm:
#    a - the documented default (hidden 64, 2 layers, lr 2e-3)
#    b - hidden 32, lr 1e-3: directly answers two of the open items in
#        BILSTM_PARAMETERS.md section 7 ("try 1e-3 ... on the rain-only
#        BiLSTM" and "hidden size 32 and 128") in one run
#
#  lightgbm:
#    a - the repo's existing default (800 trees, lr 0.05, 63 leaves)
#    b - fewer, shallower trees, higher lr: a more regularised fit, checking
#        whether the headline number depends on a large leaf count
# -----------------------------------------------------------------------
CONFIGS = {
    "transformer": {
        "a": dict(d_model=64, layers=3, heads=4, patch=4, lr=1e-3, bs=512),
        "b": dict(d_model=32, layers=2, heads=4, patch=4, lr=2e-3, bs=512),
    },
    "bilstm": {
        "a": dict(hidden=64, layers=2, lr=2e-3, bs=512),
        "b": dict(hidden=32, layers=2, lr=1e-3, bs=512),
    },
    "lightgbm": {
        "a": dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
                   min_child_samples=40),
        "b": dict(n_estimators=300, learning_rate=0.1, num_leaves=31,
                   min_child_samples=60),
    },
}

EPOCHS = 12       # capped for wall-clock reasons on a 2-core CPU box; see
PATIENCE = 4      # the note in random_split_results.json / the write-up


def _load():
    tab = pd.read_parquet(T / "tabular.parquet")
    seq = np.load(T / "rain_seq.npz")["weeks"].astype("float32")
    spec = pd.read_csv(T / "feature_spec.csv")
    feats = spec.loc[spec.role == "feature", "column"].tolist()
    cats = spec.loc[(spec.role == "feature") & (spec.kind == "categorical"), "column"].tolist()
    nums = [c for c in feats if c not in cats]

    if "split_random" not in tab.columns:
        tab["split_random"] = make_random_split(tab)

    tr = np.flatnonzero((tab.split_random == "train").to_numpy())
    va = np.flatnonzero((tab.split_random == "valid").to_numpy())
    te = np.flatnonzero((tab.split_random == "test").to_numpy())
    return tab, seq, nums, cats, tr, va, te


def _save(key, payload):
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[key] = payload
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {key} -> {OUT}")
    return payload


def _cached(key):
    if OUT.exists():
        results = json.loads(OUT.read_text())
        if key in results:
            print(f"[{key}] already in {OUT.name} - reusing it instead of retraining. "
                  f"Pass force=True to redo the run.")
            return results[key]
    return None


def run_transformer(cfg_name, force=False):
    label = f"transformer-{cfg_name}"
    if not force:
        cached = _cached(label)
        if cached is not None:
            return cached
    cfg = CONFIGS["transformer"][cfg_name]
    tab, seq, nums, cats, tr, va, te = _load()
    mk = np.zeros(len(tab), bool); mk[tr] = True
    Xn, C, sizes = encode(tab, nums, cats, mk)
    S, _ = scale_seq(seq, mk)
    y = tab[TARGET].to_numpy("float32")

    torch.manual_seed(SEED)
    m = RainTransformer(len(nums), sizes, seq.shape[1], cfg["patch"],
                        cfg["d_model"], cfg["heads"], cfg["layers"])
    print(f"[{label}] params {count_params(m):,}  config {cfg}")
    t0 = time.time()
    m, vm, hist, predict = run_nn(m, S, None, Xn, C, y, tr, va, EPOCHS,
                                  cfg["bs"], cfg["lr"], patience=PATIENCE,
                                  label=label)
    te_m = metrics(y[te], predict(te))
    return _save(label, dict(config=cfg, params=count_params(m),
                       epochs_run=len(hist), valid=vm, test=te_m,
                       seconds=round(time.time() - t0, 1), history=hist))


def run_bilstm(cfg_name, force=False):
    label = f"bilstm-{cfg_name}"
    if not force:
        cached = _cached(label)
        if cached is not None:
            return cached
    cfg = CONFIGS["bilstm"][cfg_name]
    tab, seq, nums, cats, tr, va, te = _load()
    mk = np.zeros(len(tab), bool); mk[tr] = True
    Xn, C, sizes = encode(tab, nums, cats, mk)
    S, _ = scale_seq(seq, mk)
    y = tab[TARGET].to_numpy("float32")

    torch.manual_seed(SEED)
    m = BiLSTMModel(len(nums), sizes, n_channels=1, hidden=cfg["hidden"],
                    layers=cfg["layers"], d_model=cfg["hidden"])
    print(f"[{label}] params {count_params(m):,}  config {cfg}")
    t0 = time.time()
    m, vm, hist, predict = run_nn(m, S, None, Xn, C, y, tr, va, EPOCHS,
                                  cfg["bs"], cfg["lr"], patience=PATIENCE,
                                  label=label)
    te_m = metrics(y[te], predict(te))
    return _save(label, dict(config=cfg, params=count_params(m),
                       epochs_run=len(hist), valid=vm, test=te_m,
                       seconds=round(time.time() - t0, 1), history=hist))


def run_lightgbm(cfg_name, force=False):
    label = f"lightgbm-{cfg_name}"
    if not force:
        cached = _cached(label)
        if cached is not None:
            return cached
    import lightgbm as lgb
    cfg = CONFIGS["lightgbm"][cfg_name]
    tab, seq, nums, cats, tr, va, te = _load()
    y = tab[TARGET].to_numpy("float32")

    d = tab[nums + cats].copy()
    for c in cats:
        d[c] = d[c].astype("category")

    t0 = time.time()
    g = lgb.LGBMRegressor(random_state=SEED, verbose=-1, subsample=0.8,
                          colsample_bytree=0.8, **cfg)
    g.fit(d.loc[tr], y[tr], eval_set=[(d.loc[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    vm = metrics(y[va], g.predict(d.loc[va]))
    te_m = metrics(y[te], g.predict(d.loc[te]))
    importance = dict(sorted(zip(nums + cats, g.feature_importances_.tolist()),
                              key=lambda kv: -kv[1])[:15])
    return _save(label, dict(config=cfg, valid=vm, test=te_m,
                       seconds=round(time.time() - t0, 1),
                       best_iteration=int(g.best_iteration_ or cfg["n_estimators"]),
                       top_features=importance))


RUNNERS = {
    "transformer": run_transformer,
    "bilstm": run_bilstm,
    "lightgbm": run_lightgbm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RUNNERS))
    ap.add_argument("--config", choices=["a", "b"])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.all:
        for model in ["lightgbm", "bilstm", "transformer"]:
            for cfg in ["a", "b"]:
                RUNNERS[model](cfg)
        return

    if not a.model or not a.config:
        ap.error("pass --model and --config, or --all")
    RUNNERS[a.model](a.config)


if __name__ == "__main__":
    main()
