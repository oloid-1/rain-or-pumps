"""Which target, which features, which model — measured under one honest split.

Tuning is already known not to be the lever here: a 24-setting search moved skill
by 0.007 and picked the settings the code already had. What is left to decide is
the three things that actually change the answer, and this measures all of them
side by side instead of arguing about them:

* **the target** — a level, a monsoon's recharge, or a year's net change. The
  business question is about the third, but the third is also the noisiest, and
  a model that cannot explain the second has no business claiming the third.
* **the features** — how much of the result is rain alone, and how much comes
  from knowing the well. Rain alone is the honest attribution model; the well
  properties are allowed because they cannot encode pumping.
* **the model** — a ridge that can be read, against gradient boosting.

Every row is scored the same way: fit on 2001-2014, tune nothing, score the
validation years, then refit on train plus validation and score the test years
once. Skill is measured against "this well sits at its own normal", so 0 means
the model learnt nothing about rain and 1 would be perfect.

Run from ml/:  python -m scripts.experiments [--targets level,rise,annual] [--spatial]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bits_ml import models as M
from bits_ml import splits as S
from bits_ml import targets as T
from bits_ml.config import DATA_PROCESSED
from bits_ml.dataset import CATEGORICAL, RAIN_FEATURES, WELL_FEATURES, build
from bits_ml.evaluate import metrics, skill

FEATURE_SETS = {
    "rain core": ["rain_12m_pct", "rain_36m_pct", "monsoon_last_mm_pct"],
    "rain full": list(RAIN_FEATURES),
    "rain + well": list(RAIN_FEATURES) + list(WELL_FEATURES) + list(CATEGORICAL),
}


def build_model(model_name: str, target: str):
    """The model, told which way its target runs so the rain constraint points the right way."""
    if model_name == "lgbm":
        return M.build("lgbm", wetter_is_higher=T.WETTER_IS_HIGHER[target])
    return M.build(model_name)


def score(y, prediction, label: str, **extra) -> dict:
    return {**extra, "split": label, "n": len(y)} | metrics(y, prediction) | {
        "skill_vs_normal": skill(y, prediction, np.zeros_like(y))
    }


def previous_year_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Last year's deviation, repeated. Uses past water levels, so it is a reference only."""
    previous = frame[["well_uid", "year", T.TARGET]].copy()
    previous["year"] += 1
    merged = frame[["well_uid", "year"]].merge(previous, on=["well_uid", "year"], how="left")
    return merged[T.TARGET].fillna(0.0).to_numpy()


def run_target(name: str, table: pd.DataFrame, spatial: bool) -> list[dict]:
    frame = T.build_target(name, table)
    year = frame["year"].to_numpy()
    train = np.flatnonzero(np.isin(year, list(S.TRAIN_YEARS)))
    val = np.flatnonzero(np.isin(year, list(S.VAL_YEARS)))
    test = np.flatnonzero(np.isin(year, list(S.TEST_YEARS)))
    final_train = np.flatnonzero(np.isin(year, list(S.FINAL_TRAIN_YEARS)))
    y = frame[T.TARGET].to_numpy()
    common = {"target": name, "rows": len(frame), "wells": frame["well_uid"].nunique()}
    print(f"\n{name}: {len(frame):,} rows, {common['wells']:,} wells, target sd {y.std():.2f} m", flush=True)

    rows = [
        score(y[val], previous_year_baseline(frame)[val], "validation years", **common,
              features="—", model="baseline: last year"),
        score(y[test], previous_year_baseline(frame)[test], "test years", **common,
              features="—", model="baseline: last year"),
    ]

    for feature_name, features in FEATURE_SETS.items():
        usable = [f for f in features if f in frame.columns]
        X = frame[usable]
        for model_name in ("ridge", "lgbm"):
            model = build_model(model_name, name).fit(X.iloc[train], y[train])
            rows.append(score(y[val], model.predict(X.iloc[val]), "validation years", **common,
                              features=feature_name, model=model_name))
            final = build_model(model_name, name).fit(X.iloc[final_train], y[final_train])
            rows.append(score(y[test], final.predict(X.iloc[test]), "test years", **common,
                              features=feature_name, model=model_name))
            print(f"   {feature_name:12s} {model_name:6s} "
                  f"val skill {rows[-2]['skill_vs_normal']:+.3f}  test skill {rows[-1]['skill_vs_normal']:+.3f}", flush=True)

            if spatial and feature_name == "rain + well":
                held = np.full(len(frame), np.nan)
                for fit_rows, held_rows in S.group_folds(frame, years=S.FINAL_TRAIN_YEARS):
                    fold = build_model(model_name, name).fit(X.iloc[fit_rows], y[fit_rows])
                    held[held_rows] = fold.predict(X.iloc[held_rows])
                seen = np.flatnonzero(~np.isnan(held))
                rows.append(score(y[seen], held[seen], "unseen districts", **common,
                                  features=feature_name, model=model_name))
                print(f"   {feature_name:12s} {model_name:6s} unseen districts "
                      f"skill {rows[-1]['skill_vs_normal']:+.3f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="level,rise,annual")
    parser.add_argument("--spatial", action="store_true", help="also hold out whole districts (slower)")
    args = parser.parse_args()

    table = build(S.TRAIN_YEARS)
    rows = []
    for name in (t.strip() for t in args.targets.split(",") if t.strip()):
        rows += run_target(name, table, args.spatial)

    results = pd.DataFrame(rows)
    out = DATA_PROCESSED / "experiment_results.csv"
    results.to_csv(out, index=False)

    print("\n\nskill against each well's own normal (higher is better, 0 means nothing learnt)\n")
    for split in ("validation years", "test years", "unseen districts"):
        block = results[results["split"] == split]
        if block.empty:
            continue
        pivot = block.pivot_table(index=["target", "features"], columns="model", values="skill_vs_normal")
        print(f"--- {split}")
        print(pivot.round(3).to_string(), "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
