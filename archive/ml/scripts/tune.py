"""Search LightGBM settings, then score the test years once.

The same search the Colab notebook runs, kept in the repo so it is reproducible
and reviewable. LightGBM on this data is CPU work and finishes in minutes here,
so the notebook is a convenience, not a requirement.

Two rules the search obeys:

* while searching, the table is built from the training years alone, so the
  validation years are not inside any well's normal level or normal rain
* the winner is refit on training plus validation years, with the table rebuilt
  accordingly, and only then are the test years scored, once

Run from ml/:  python -m scripts.tune [--trials 24]
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from bits_ml import splits as S
from bits_ml.config import DATA_PROCESSED, MODELS_DIR
from bits_ml.dataset import FEATURES, MONOTONE, TARGET, build, matrix
from bits_ml.evaluate import metrics, skill
from bits_ml.models import baseline_normal
from bits_ml.train import bundle

SPACE = {
    "num_leaves": [31, 63, 127],
    "min_child_samples": [100, 200, 400],
    "learning_rate": [0.03, 0.05],
    "colsample_bytree": [0.6, 0.8],
    "reg_lambda": [1.0, 10.0],
}
FIXED = dict(objective="regression", n_estimators=3000, subsample=0.8, subsample_freq=1,
             monotone_constraints_method="advanced", random_state=42, n_jobs=-1, verbose=-1)


def constraints() -> list[int]:
    return [MONOTONE.get(f, 0) for f in FEATURES]


def score(table: pd.DataFrame, rows: np.ndarray, prediction: np.ndarray) -> dict:
    truth = table[TARGET].to_numpy()[rows]
    return metrics(truth, prediction) | {"skill_vs_normal": skill(truth, prediction, baseline_normal(table.iloc[rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=24, help=f"settings to try out of {len(list(itertools.product(*SPACE.values())))}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "lgbm_tuned.joblib")
    args = parser.parse_args()

    search_table = build(S.TRAIN_YEARS)
    year = search_table["year"].to_numpy()
    train_rows = np.flatnonzero(np.isin(year, list(S.TRAIN_YEARS)))
    val_rows = np.flatnonzero(np.isin(year, list(S.VAL_YEARS)))
    X, y = matrix(search_table), search_table[TARGET].to_numpy()
    print(f"search: {len(train_rows):,} training rows, {len(val_rows):,} validation rows "
          f"({min(S.TRAIN_YEARS)}-{max(S.TRAIN_YEARS)} against {min(S.VAL_YEARS)}-{max(S.VAL_YEARS)})", flush=True)

    everything = [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]
    random.seed(args.seed)
    combos = random.sample(everything, min(args.trials, len(everything)))

    trials, best = [], None
    for i, params in enumerate(combos, 1):
        model = lgb.LGBMRegressor(**FIXED, monotone_constraints=constraints(), **params)
        model.fit(X.iloc[train_rows], y[train_rows],
                  eval_set=[(X.iloc[val_rows], y[val_rows])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        row = params | score(search_table, val_rows, model.predict(X.iloc[val_rows])) | {"trees": model.best_iteration_}
        trials.append(row)
        if best is None or row["rmse"] < best["rmse"]:
            best = row
        print(f"   {i:2d}/{len(combos)}  rmse {row['rmse']:.4f}  skill {row['skill_vs_normal']:+.4f}  "
              f"trees {row['trees']:4d}  {params}", flush=True)

    print(f"\nbest on the validation years: rmse {best['rmse']:.4f}, skill {best['skill_vs_normal']:+.4f}")

    final_table = build(S.FINAL_TRAIN_YEARS)
    final_year = final_table["year"].to_numpy()
    fit_rows = np.flatnonzero(np.isin(final_year, list(S.FINAL_TRAIN_YEARS)))
    test_rows = np.flatnonzero(np.isin(final_year, list(S.TEST_YEARS)))
    chosen = {k: best[k] for k in SPACE}
    trees = max(200, int(best["trees"] * 1.2))  # a little longer, since there is more data to learn from now
    final_model = lgb.LGBMRegressor(**{**FIXED, "n_estimators": trees}, monotone_constraints=constraints(), **chosen)
    final_features = matrix(final_table)
    final_model.fit(final_features.iloc[fit_rows], final_table[TARGET].to_numpy()[fit_rows])

    test_score = score(final_table, test_rows, final_model.predict(final_features.iloc[test_rows]))
    per_campaign = {}
    campaign = final_table["campaign"].to_numpy()
    for name in pd.unique(campaign[test_rows]):
        rows = test_rows[campaign[test_rows] == name]
        per_campaign[str(name)] = score(final_table, rows, final_model.predict(final_features.iloc[rows]))

    scores = {"validation years": {k: best[k] for k in ("rmse", "mae", "r2", "skill_vs_normal")},
              "test years": test_score, "test years by campaign": per_campaign}
    saved = bundle(final_model, "lgbm_tuned", final_table, S.FINAL_TRAIN_YEARS, scores)
    saved["settings"] = chosen | {"n_estimators": trees}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved, args.out)
    pd.DataFrame(trials).to_csv(DATA_PROCESSED / "tuning_trials.csv", index=False)

    print(f"\nsettings: {chosen} with {trees} trees")
    print(f"test years: rmse {test_score['rmse']:.3f} m | mae {test_score['mae']:.3f} m | "
          f"skill {test_score['skill_vs_normal']:+.3f}")
    print("by campaign: " + "  ".join(f"{k} {v['skill_vs_normal']:+.3f}" for k, v in per_campaign.items()))
    print(f"saved -> {args.out}  |  trials -> {DATA_PROCESSED / 'tuning_trials.csv'}")


if __name__ == "__main__":
    main()
