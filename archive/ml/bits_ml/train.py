"""Fit a model and save it for the simulator and the API.

Training years are the ones that also define every well's normal level and
normal rain, so the saved bundle records them: anything scoring or predicting
later has to rebuild the table the same way or the numbers mean something else.

Usage, from ml/:
    python -m bits_ml.train --model lgbm          # fit on 2000-2017, save models/lgbm.joblib
    python -m bits_ml.train --model ridge --hold-out-val   # fit on 2000-2014, keeps 2015-2017 free for tuning
    python -m bits_ml.train --model lgbm --colab-table path/to/well_campaign_table.parquet
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import models as M
from . import splits as S
from .config import MODELS_DIR, MODEL_PATH
from .dataset import FEATURES, TARGET, build, categories_of, matrix
from .evaluate import metrics, skill


def fit(name: str, table: pd.DataFrame, train_years, **params):
    rows = np.flatnonzero(np.isin(table["year"].to_numpy(), list(train_years)))
    model = M.build(name, **params)
    model.fit(matrix(table.iloc[rows]), table[TARGET].to_numpy()[rows])
    return model, rows


def bundle(model, name: str, table: pd.DataFrame, train_years, scores: dict) -> dict:
    return {
        "model": model,
        "name": name,
        "features": list(FEATURES),
        "categories": categories_of(table),  # the label set is part of the model, see dataset.apply_categories
        "target": TARGET,
        "train_years": list(train_years),
        "n_rows": int(np.isin(table["year"].to_numpy(), list(train_years)).sum()),
        "scores": scores,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }


def load_bundle(path: Path) -> dict:
    """Load a saved model and complain loudly if it was built somewhere different.

    A bundle trained on Colab travels as a pickle: different library versions can
    change how it behaves, and without its recorded labels a table whose category
    columns arrive as plain text cannot be put back into the dtype it expects.
    """
    bundle = joblib.load(path)
    here = {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__}
    differences = [f"{k}: bundle {v}, running {here.get(k)}" for k, v in bundle.get("versions", {}).items()
                   if k in here and v != here[k]]
    if differences:
        print(f"warning: {path.name} was built with different libraries ({'; '.join(differences)})")
    if not bundle.get("categories"):
        print(f"warning: {path.name} records no category labels; a table whose campaign or well-type columns "
              f"arrive as text cannot be restored for it (rebuild it with bits_ml.train or the Colab notebook)")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="lgbm", help="ridge, lgbm, lgbm_free or mlp")
    parser.add_argument("--hold-out-val", action="store_true", help="train on 2000-2014 only, leaving 2015-2017 for tuning")
    parser.add_argument("--table", type=Path, help="prebuilt well_campaign_table.parquet (for Colab or Kaggle)")
    parser.add_argument("--out", type=Path, help="where to write the model bundle")
    parser.add_argument("--default", action="store_true", help="also save as the API default model")
    args = parser.parse_args()

    train_years = S.TRAIN_YEARS if args.hold_out_val else S.FINAL_TRAIN_YEARS
    table = pd.read_parquet(args.table) if args.table else build(train_years)
    if args.table:
        print(f"loaded {args.table} ({len(table):,} rows) - its normals must come from the same training years")

    model, rows = fit(args.model, table, train_years)
    scores = {}
    for label, years in (("validation years", S.VAL_YEARS), ("test years", S.TEST_YEARS)):
        if args.hold_out_val is False and label == "validation years":
            continue  # those years are inside training now
        take = np.flatnonzero(np.isin(table["year"].to_numpy(), list(years)))
        if take.size < 50:
            continue
        y = table[TARGET].to_numpy()[take]
        pred = model.predict(matrix(table.iloc[take]))
        scores[label] = metrics(y, pred) | {"skill_vs_normal": skill(y, pred, M.baseline_normal(table.iloc[take]))}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or MODELS_DIR / f"{args.model}.joblib"
    joblib.dump(bundle(model, args.model, table, train_years, scores), out)
    if args.default:
        joblib.dump(bundle(model, args.model, table, train_years, scores), MODEL_PATH)

    print(f"{args.model}: fitted on {len(rows):,} rows from {min(train_years)}-{max(train_years)}")
    for label, s in scores.items():
        print(f"   {label:<18} rmse {s['rmse']:.3f} m | mae {s['mae']:.3f} m | "
              f"skill vs each well's normal {s['skill_vs_normal']:+.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
