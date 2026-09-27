"""Scoring: how much of the water-level movement rain actually explains.

Scores are reported per campaign, never as one pooled number, because the four
campaigns are different questions: August and November follow the monsoon,
January and May follow months of drawdown. Skill is measured against the
"normal" baseline, so a positive skill means the model knows something about
rain that the well's own average does not.

Two splits are reported for every model: later years it never saw, and districts
it never saw. A model can do well on one and badly on the other, and the second
is what tells us whether the attribution travels beyond monitored districts.

Run from ml/:  python -m bits_ml.evaluate [--models ridge,lgbm] [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import models as M
from . import splits as S
from .config import DATA_PROCESSED
from .dataset import FEATURES, TARGET, build, matrix


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def skill(y_true, y_pred, y_base) -> float:
    """1 means perfect, 0 means no better than the baseline, below 0 means worse."""
    base = mean_squared_error(y_true, y_base)
    return float(1 - mean_squared_error(y_true, y_pred) / base) if base > 0 else np.nan


def score(table: pd.DataFrame, rows: np.ndarray, predictions: dict[str, np.ndarray], label: str) -> pd.DataFrame:
    """Per campaign and overall, for every prediction given."""
    y = table[TARGET].to_numpy()[rows]
    campaign = table["campaign"].to_numpy()[rows]
    out = []
    for name, pred in predictions.items():
        for group in ["all", *pd.unique(campaign)]:
            take = np.ones(len(rows), dtype=bool) if group == "all" else campaign == group
            if take.sum() < 50:
                continue
            row = {"split": label, "model": name, "campaign": group, "n": int(take.sum())}
            row.update(metrics(y[take], pred[take]))
            row["skill_vs_normal"] = skill(y[take], pred[take], predictions["baseline_normal"][take])
            out.append(row)
    return pd.DataFrame(out)


def fit_predict(name: str, table: pd.DataFrame, fit_rows: np.ndarray, predict_rows: np.ndarray, **params):
    model = M.build(name, **params)
    model.fit(matrix(table.iloc[fit_rows]), table[TARGET].to_numpy()[fit_rows])
    return model, model.predict(matrix(table.iloc[predict_rows]))


def evaluate(table: pd.DataFrame, names=("ridge", "lgbm"), quick: bool = False) -> tuple[pd.DataFrame, dict]:
    split = S.year_split(table)
    train_rows = np.flatnonzero(split.train)
    results, fitted = [], {}

    for label, mask in (("validation years", split.val), ("test years", split.test)):
        rows = np.flatnonzero(mask)
        preds = {"baseline_normal": M.baseline_normal(table.iloc[rows]),
                 "baseline_last_year": M.baseline_last_year(table)[rows]}
        fit_rows = train_rows if label == "validation years" else np.flatnonzero(split.train | split.val)
        for name in names:
            model, pred = fit_predict(name, table, fit_rows, rows)
            preds[name] = pred
            fitted[f"{name}:{label}"] = model
        results.append(score(table, rows, preds, label))

    if not quick:
        for name in names:
            pred = np.full(len(table), np.nan)
            held_all = []
            for fit_rows, held in S.group_folds(table):
                _, p = fit_predict(name, table, fit_rows, held)
                pred[held] = p
                held_all.append(held)
            rows = np.concatenate(held_all)
            preds = {"baseline_normal": M.baseline_normal(table.iloc[rows]), name: pred[rows]}
            results.append(score(table, rows, preds, "unseen districts"))

    return pd.concat(results, ignore_index=True), fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="ridge,lgbm", help="comma separated: ridge, lgbm, lgbm_free, mlp")
    parser.add_argument("--quick", action="store_true", help="skip the unseen-district folds")
    args = parser.parse_args()

    table = build()
    names = tuple(n.strip() for n in args.models.split(",") if n.strip())
    results, fitted = evaluate(table, names, quick=args.quick)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "model_results.csv"
    results.to_csv(out, index=False)
    shown = results[results["campaign"] == "all"] if len(results) > 12 else results
    print(shown.round(3).to_string(index=False))
    print(f"\nper-campaign detail written to {out}")

    ridge = next((m for k, m in fitted.items() if k.startswith("ridge")), None)
    if ridge is not None:
        coef = ridge.coefficients().dropna(subset=["response_m"])
        rain = coef[coef["feature"].isin(M.RAIN_FEATURES)]
        top = rain.reindex(rain["standardised"].abs().sort_values(ascending=False).index).head(8)
        print("\nstrongest rain responses (metres of water level; negative = water stands shallower):")
        print(top[["campaign", "feature", "response_m", "unit"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
