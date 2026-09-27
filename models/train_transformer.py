"""
Trains the rain transformer and the baselines it has to beat, on the same rows
and the same split, and prints one comparison table.

    python3 ml/train_transformer.py                  full run
    python3 ml/train_transformer.py --epochs 3       quick check
    python3 ml/train_transformer.py --cv             5 district folds

Baselines, in order of how embarrassing it would be to lose to them:
    zero          predict no change at all
    season mean   the training mean for that campaign month
    ridge         linear on the tabular features
    lightgbm      gradient boosting on the tabular features

The point of the comparison is not to win. A transformer that ties LightGBM on
216k rows is the expected result and is reportable as such. The point is that
the sequence arm is measured against the hand-built rain windows rather than
assumed better than them.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformer import RainTransformer, SeqOnlyTransformer, count_params

REPO = Path(__file__).resolve().parents[1]
T = REPO / "data" / "training"
SEED = 42
TARGET = "delta_h_m"


# --------------------------------------------------------------------------
def load():
    tab = pd.read_parquet(T / "tabular.parquet")
    spec = pd.read_csv(T / "feature_spec.csv")
    seq = np.load(T / "rain_seq.npz")["weeks"].astype("float32")
    assert len(tab) == len(seq), (len(tab), len(seq))

    feats = spec.loc[spec.role == "feature", "column"].tolist()
    cats = spec.loc[(spec.role == "feature") & (spec.kind == "categorical"), "column"].tolist()
    nums = [c for c in feats if c not in cats]

    forbidden = set(spec.loc[spec.role == "meta", "column"]) | {TARGET}
    leak = [c for c in feats if c in forbidden]
    assert not leak, f"leakage: {leak}"
    return tab, seq, nums, cats


def encode(tab, nums, cats, train_mask):
    """Standardise numerics and index categoricals using TRAIN rows only."""
    X = tab[nums].to_numpy("float32")
    med = np.nanmedian(X[train_mask], axis=0)
    X = np.where(np.isfinite(X), X, med)
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0)
    sd[sd < 1e-6] = 1.0
    Xn = ((X - mu) / sd).astype("float32")

    codes, sizes = [], []
    for c in cats:
        v = tab[c].astype(str).fillna("unknown")
        levels = pd.Index(sorted(v[train_mask].unique()))
        idx = levels.get_indexer(v)          # -1 for unseen
        codes.append(np.where(idx < 0, len(levels), idx).astype("int64"))
        sizes.append(len(levels) + 1)        # +1 bucket for unseen
    C = np.stack(codes, axis=1) if codes else np.zeros((len(tab), 0), "int64")
    return Xn, C, sizes


def scale_seq(seq, train_mask):
    """log1p then standardise. Weekly rain is heavily zero-inflated and
    right-skewed; log1p pulls the 500 mm weeks in without discarding them."""
    s = np.log1p(np.clip(seq, 0, None))
    mu = s[train_mask].mean()
    sd = s[train_mask].std()
    out = ((s - mu) / max(sd, 1e-6)).astype("float32")
    mask = ~np.isfinite(seq)                 # weeks with no rainfall record
    out[mask] = 0.0
    return out, mask


# --------------------------------------------------------------------------
def metrics(y, p):
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(mae=float(np.abs(e).mean()),
                rmse=float(np.sqrt((e ** 2).mean())),
                r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"))


def baselines(tab, nums, cats, tr, va):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    y_tr = tab.loc[tr, TARGET].to_numpy("float32")
    y_va = tab.loc[va, TARGET].to_numpy("float32")
    out = {}

    out["zero"] = metrics(y_va, np.zeros_like(y_va))

    m = tab.loc[tr].groupby("season")[TARGET].mean()
    out["season mean"] = metrics(y_va, tab.loc[va, "season"].map(m).fillna(y_tr.mean()).to_numpy("float32"))

    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                        ("s", StandardScaler())]), nums),
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats)])
    ridge = Pipeline([("p", pre), ("m", Ridge(alpha=1.0))])
    ridge.fit(tab.loc[tr, nums + cats], y_tr)
    out["ridge"] = metrics(y_va, ridge.predict(tab.loc[va, nums + cats]))

    try:
        import lightgbm as lgb
        d = tab[nums + cats].copy()
        for c in cats:
            d[c] = d[c].astype("category")
        g = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05,
                              num_leaves=63, min_child_samples=40,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1)
        g.fit(d.loc[tr], y_tr, eval_set=[(d.loc[va], y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        out["lightgbm"] = metrics(y_va, g.predict(d.loc[va]))
        out["_lgb_importance"] = dict(sorted(
            zip(nums + cats, g.feature_importances_.tolist()),
            key=lambda kv: -kv[1])[:15])
    except ImportError:
        pass
    return out


# --------------------------------------------------------------------------
def run_nn(model, S, M, Xn, C, y, tr, va, epochs=25, bs=512, lr=1e-3,
           patience=5, device="cpu", label="transformer"):
    torch.manual_seed(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(tr) // bs + 1))
    # Huber, not MSE: delta_h_m has a long tail of real but rare 10 m swings
    # and MSE would let a handful of them set every gradient.
    loss_fn = nn.HuberLoss(delta=1.0)

    def batches(idx, shuffle):
        order = np.random.permutation(idx) if shuffle else idx
        for a in range(0, len(order), bs):
            k = order[a:a + bs]
            yield (torch.from_numpy(S[k]).to(device),
                   torch.from_numpy(Xn[k]).to(device),
                   torch.from_numpy(C[k]).to(device),
                   torch.from_numpy(y[k]).to(device))

    @torch.no_grad()
    def predict(idx):
        model.eval()
        p = []
        for a in range(0, len(idx), 4096):
            k = idx[a:a + 4096]
            p.append(model(torch.from_numpy(S[k]).to(device),
                           torch.from_numpy(Xn[k]).to(device),
                           torch.from_numpy(C[k]).to(device)).cpu().numpy())
        return np.concatenate(p)

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for s, x, c, t in batches(tr, True):
            opt.zero_grad()
            l = loss_fn(model(s, x, c), t)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(l) * len(t)
        vm = metrics(y[va], predict(va))
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), **vm))
        print(f"  [{label}] epoch {ep:>2}  train {tot/len(tr):.4f}  "
              f"valid mae {vm['mae']:.4f}  rmse {vm['rmse']:.4f}  r2 {vm['r2']:.4f}")
        if vm["mae"] < best - 1e-4:
            best, bad = vm["mae"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  [{label}] early stop at epoch {ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, metrics(y[va], predict(va)), hist, predict


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--cv", action="store_true", help="5 district folds instead of the year split")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out", default=str(REPO / "data" / "training" / "results.json"))
    a = ap.parse_args()

    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tab, seq, nums, cats = load()
    seq = seq[:, -a.weeks:]
    y = tab[TARGET].to_numpy("float32")
    print(f"rows {len(tab):,}  numeric {len(nums)}  categorical {len(cats)}  "
          f"weeks {seq.shape[1]}  device {device}")

    results = {"config": vars(a), "rows": int(len(tab))}

    if a.cv:
        folds = sorted(tab.district_fold.unique())
        agg = []
        for f in folds:
            tr = np.flatnonzero((tab.district_fold != f).to_numpy())
            va = np.flatnonzero((tab.district_fold == f).to_numpy())
            mk = np.zeros(len(tab), bool); mk[tr] = True
            Xn, C, sizes = encode(tab, nums, cats, mk)
            S, _ = scale_seq(seq, mk)
            m = RainTransformer(len(nums), sizes, seq.shape[1], a.patch,
                                a.d_model, a.heads, a.layers)
            print(f"fold {f}: train {len(tr):,} valid {len(va):,}")
            _, vm, _, _ = run_nn(m, S, None, Xn, C, y, tr, va, a.epochs, a.bs,
                                 a.lr, device=device, label=f"fold{f}")
            agg.append(vm)
        results["cv_folds"] = agg
        results["cv_mean"] = {k: float(np.mean([d[k] for d in agg])) for k in agg[0]}
        print("\ndistrict-fold CV mean:", results["cv_mean"])
    else:
        tr = np.flatnonzero((tab.split == "train").to_numpy())
        va = np.flatnonzero((tab.split == "valid").to_numpy())
        te = np.flatnonzero((tab.split == "test").to_numpy())
        mk = np.zeros(len(tab), bool); mk[tr] = True
        Xn, C, sizes = encode(tab, nums, cats, mk)
        S, _ = scale_seq(seq, mk)

        if not a.no_baselines:
            t0 = time.time()
            results["baselines"] = baselines(tab, nums, cats, tr, va)
            print(f"baselines done in {time.time()-t0:.0f}s")

        m = RainTransformer(len(nums), sizes, seq.shape[1], a.patch,
                            a.d_model, a.heads, a.layers)
        print(f"transformer params {count_params(m):,}")
        m, vm, hist, predict = run_nn(m, S, None, Xn, C, y, tr, va, a.epochs,
                                      a.bs, a.lr, device=device, label="full")
        results["transformer"] = vm
        results["history"] = hist
        results["transformer_test"] = metrics(y[te], predict(te))
        torch.save(m.state_dict(), T / "transformer.pt")

        sm = SeqOnlyTransformer(seq.shape[1], a.patch, a.d_model, a.heads, a.layers)
        _, svm, _, _ = run_nn(sm, S, None, Xn, C, y, tr, va, a.epochs, a.bs,
                              a.lr, device=device, label="seq only")
        results["transformer_seq_only"] = svm

        rows = []
        for k, v in (results.get("baselines") or {}).items():
            if not k.startswith("_"):
                rows.append((k, v))
        rows += [("transformer", vm), ("transformer seq only", svm)]
        print("\nvalid 2015-2017")
        print(f"{'model':<22}{'mae':>9}{'rmse':>9}{'r2':>9}")
        for k, v in rows:
            print(f"{k:<22}{v['mae']:>9.4f}{v['rmse']:>9.4f}{v['r2']:>9.4f}")
        print(f"\ntransformer on held-out test 2018-2022: {results['transformer_test']}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
