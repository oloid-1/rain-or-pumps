"""Prove a model trained on Colab behaves the same at home.

The model travels back as a pickle, and the trip can change what it does without
raising anything. Library versions may differ. A category label may come back
renamed, or missing while rows still use it, in which case those readings turn
into gaps and predictions move by about a metre — both measured on this data.
Reordered categories are harmless: LightGBM remaps pandas categoricals by label.
Plain strings fail loudly instead, which dataset.apply_categories repairs.

None of the silent cases can be fixed after the fact, so they have to be caught.

The notebook saves `colab_predictions.csv`, its own predictions for 2,000 test
readings. This replays them here and reports the largest difference. Anything
above a rounding error means the bundle must not be used.

Run from ml/:  python -m scripts.check_colab_bundle --bundle models/lgbm_tuned.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bits_ml.config import DATA_PROCESSED, MODELS_DIR
from bits_ml.dataset import apply_categories, build, matrix
from bits_ml.train import load_bundle

TOLERANCE = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=MODELS_DIR / "lgbm.joblib")
    parser.add_argument("--predictions", type=Path, default=DATA_PROCESSED / "colab_predictions.csv")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"{args.predictions} is missing; it is written by notebooks/02_train_on_colab.ipynb")

    bundle = load_bundle(args.bundle)
    recorded = pd.read_csv(args.predictions)
    table = build(bundle["train_years"])  # the bundle's own training years define the normals
    rows = table.merge(recorded[["well_uid", "year", "campaign"]].assign(_wanted=1),
                       on=["well_uid", "year", "campaign"], how="inner")
    if len(rows) < len(recorded) * 0.99:
        print(f"warning: only {len(rows):,} of {len(recorded):,} recorded readings were found in the rebuilt table")

    here = pd.Series(bundle["model"].predict(matrix(apply_categories(rows, bundle.get("categories")))),
                     index=pd.MultiIndex.from_frame(rows[["well_uid", "year", "campaign"]]))
    there = recorded.set_index(["well_uid", "year", "campaign"])["prediction"]
    shared = here.index.intersection(there.index)
    difference = np.abs(here.loc[shared].to_numpy() - there.loc[shared].to_numpy())

    print(f"{args.bundle.name}: {len(shared):,} readings compared")
    print(f"   largest difference {difference.max():.2e} m | median {np.median(difference):.2e} m")
    if difference.max() > args.tolerance:
        worst = rows.iloc[int(np.argmax(difference))]
        print(f"   worst at {worst['well_uid']} {worst['campaign']} {worst['year']}")
        print("   the bundle does not reproduce its own predictions here: check the category labels and library versions")
        sys.exit(1)
    print("   identical within tolerance: the bundle is safe to use")


if __name__ == "__main__":
    main()
