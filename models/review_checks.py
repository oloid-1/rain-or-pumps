"""
Recomputes every number in models/MODEL_REVIEW.md from the saved models.

    python models/review_checks.py

Needs data/training/ with the 6-channel data (python models/build_sequences.py),
models/artifacts/bilstm.pt and simulator/artifacts/sim.pt. Trains one LightGBM
with the exact settings of train_transformer.baselines(); everything else is
inference. About two minutes on CPU.

Checks:
    1. test-year scores (2018-2022) for every model, LightGBM included
    2. how correlated the models' errors are, and what an ensemble gains
    3. the district residual ranking on all rows against held-out rows only
    4. how much of the skill the calendar alone carries (season-mean baseline)
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "models"), str(REPO / "simulator")]
from bilstm import RainBiLSTM  # noqa: E402
from train_bilstm import load, channel_stats, scale  # noqa: E402
from build_training_data import OUT_DIR  # noqa: E402
from train_sim import encode_static, STATIC_NUMERIC  # noqa: E402
from train_transformer import encode, metrics, TARGET, SEED  # noqa: E402


def predict(model, S, X, C):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(S[i:i + 8192]), torch.from_numpy(X[i:i + 8192]),
                                     torch.from_numpy(C[i:i + 8192])).numpy()
                               for i in range(0, len(S), 8192)])


def row(name, y, p, va, te):
    v, t = metrics(y[va], p[va]), metrics(y[te], p[te])
    print(f"  {name:<32} valid R2 {v['r2']:.3f} MAE {v['mae']:.3f} | test R2 {t['r2']:.3f} MAE {t['mae']:.3f}")


def ranking(df, mask):
    r = (df[mask].groupby("district").agg(resid=("resid", "mean"), wells=("well_uid", "nunique"))
         .query("wells >= 3").sort_values("resid", ascending=False))
    return r


def main():
    import lightgbm as lgb
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    tab, x, ch, nums, cats = load(OUT_DIR)
    y = tab[TARGET].to_numpy("float32")
    tr = (tab.split == "train").to_numpy()
    va = (tab.split == "valid").to_numpy()
    te = (tab.split == "test").to_numpy()

    # ---- baselines and LightGBM, exactly as train_transformer.baselines()
    m = tab[tr].groupby("season")[TARGET].mean()
    p_season = tab.season.map(m).to_numpy("float32")
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), nums),
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats)])
    ridge = Pipeline([("p", pre), ("m", Ridge(alpha=1.0))]).fit(tab.loc[tr, nums + cats], y[tr])
    p_ridge = ridge.predict(tab[nums + cats])
    d = tab[nums + cats].copy()
    for c in cats:
        d[c] = d[c].astype("category")
    g = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    g.fit(d[tr], y[tr], eval_set=[(d[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb = g.predict(d)

    # ---- the two sequence models, from their saved weights
    S = scale(x, ch, channel_stats(x, ch, tr))
    Xn, C, sizes = encode(tab, nums, cats, tr)
    mb = RainBiLSTM(len(nums), sizes, len(ch))
    mb.load_state_dict(torch.load(REPO / "models" / "artifacts" / "bilstm.pt", map_location="cpu"))
    p_bil = predict(mb, S, Xn, C)
    Xs, Cs, ss, _ = encode_static(tab, tr)
    ms = RainBiLSTM(len(STATIC_NUMERIC), ss, len(ch))
    ms.load_state_dict(torch.load(REPO / "simulator" / "artifacts" / "sim.pt", map_location="cpu"))
    p_sim = predict(ms, S, Xs, Cs)

    print("\n1. scores on validation (used for early stopping) and test (never used)")
    print(f"  target sd: valid {y[va].std():.3f}, test {y[te].std():.3f}; LightGBM stopped at {g.best_iteration_} trees")
    for name, p in (("season mean", p_season), ("ridge", p_ridge), ("LightGBM", p_lgb),
                    ("BiLSTM, 6 channels", p_bil), ("simulator, 6 channels", p_sim)):
        row(name, y, p, va, te)

    print("\n2. error correlation and ensembles (weights chosen on validation only)")
    print(f"  corr(LightGBM error, BiLSTM error) on valid: {np.corrcoef(y[va] - p_lgb[va], y[va] - p_bil[va])[0, 1]:.3f}")
    print(f"  corr(LightGBM error, simulator error) on valid: {np.corrcoef(y[va] - p_lgb[va], y[va] - p_sim[va])[0, 1]:.3f}")
    w = max(np.linspace(0, 1, 21), key=lambda w: metrics(y[va], w * p_lgb[va] + (1 - w) * p_bil[va])["r2"])
    row("0.5 LightGBM + 0.5 BiLSTM", y, 0.5 * p_lgb + 0.5 * p_bil, va, te)
    row(f"{w:.2f} LightGBM + {1 - w:.2f} BiLSTM", y, w * p_lgb + (1 - w) * p_bil, va, te)
    row("0.5 LightGBM + 0.5 simulator", y, 0.5 * p_lgb + 0.5 * p_sim, va, te)

    print("\n3. district residual ranking (BiLSTM, districts with >= 3 wells)")
    t = tab.assign(resid=y - p_bil)
    print(f"  mean |residual|: training rows {np.abs(t.resid[tr]).mean():.3f}, held-out rows {np.abs(t.resid[~tr]).mean():.3f}")
    share = t.groupby("district").split.apply(lambda s: (s == "train").mean()).mean()
    print(f"  share of a district's rows that are training rows: {share:.2f} on average")
    a, b = ranking(t, np.ones(len(t), bool)), ranking(t, ~tr)
    print("  top 10, all rows (as published):", ", ".join(a.index[:10]))
    print("  top 10, held-out rows only:      ", ", ".join(b.index[:10]))
    common = a.index.intersection(b.index)
    print(f"  overlap: top 10 {len(set(a.index[:10]) & set(b.index[:10]))}, top 20 {len(set(a.index[:20]) & set(b.index[:20]))}; "
          f"rank correlation over {len(common)} districts {a.loc[common, 'resid'].rank().corr(b.loc[common, 'resid'].rank()):.3f}")

    print("\n4. calendar against rain")
    for name, p in (("season mean", p_season), ("BiLSTM", p_bil)):
        print(f"  {name:<12} test R2 {metrics(y[te], p[te])['r2']:.3f}")
    print(f"  rain and well facts add {metrics(y[te], p_bil[te])['r2'] - metrics(y[te], p_season[te])['r2']:.3f} test R2 over the calendar alone (BiLSTM)")
    print(f"  ... and {metrics(y[te], p_lgb[te])['r2'] - metrics(y[te], p_season[te])['r2']:.3f} (LightGBM)")


if __name__ == "__main__":
    main()
