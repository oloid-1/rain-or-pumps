"""The one table the models train on: every water-level reading with the rain behind it.

Rows are readings (well x campaign x year), not wells, so one model learns from
all 2,759 wells at once. The target is how far the water stands from that well's
own normal for that campaign, in metres, positive meaning deeper.

What a feature may be is the whole attribution argument. Rain and fixed well
properties are allowed. Year, district, state, coordinates and any past water
level are not: each of them carries the pumping history we are trying to
measure, and a model given them would explain the decline with itself and leave
no gap to attribute.

Run from ml/:  python -m bits_ml.dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import groundwater as gw
from . import rain_model as rm
from .config import DATA_PROCESSED

TRAIN_YEARS = tuple(range(2000, 2015))

TARGET = "anomaly_m"
GROUP = "district"
RAIN_FEATURES = (
    [f"rain_{w}m" for w in rm.WINDOWS]
    + [f"rain_{w}m_pct" for w in rm.WINDOWS]
    + ["monsoon_last_mm", "monsoon_last_mm_pct", "monsoon_todate_mm", "wet_days_12m", "wet_days_12m_pct", "max_day_12m"]
)
WELL_FEATURES = ["sy", "well_depth_m", "rain_normal_mm"]
CATEGORICAL = ["campaign", "well_type", "aquifer"]
FEATURES = RAIN_FEATURES + WELL_FEATURES + CATEGORICAL
FORBIDDEN = ["year", "district", "state", "lat", "lon", "depth_mbgl", "normal_m", "well_uid"]

# More rain must never make the model predict deeper water (the target is depth).
# Every rain input is constrained, including the wettest day and the monsoon so
# far: a scenario raises all of them together, so leaving any one free lets a
# wetter year come back as deeper water and breaks the break-even search.
MONOTONE = {c: -1 for c in RAIN_FEATURES}


def build(train_years=TRAIN_YEARS) -> pd.DataFrame:
    """Readings joined to the rain history that precedes them.

    train_years decides two things and nothing else: which years define each
    well's normal level, and which years define its normal rain. Everything
    downstream inherits that choice, so a fold can rebuild the table with its own
    training years and stay leakage free.
    """
    readings = gw.add_normals(gw.load_readings(), train_years)
    series = rm.RainSeries.load()
    features = rm.campaign_features(series, sorted(readings["year"].unique()), gw.CAMPAIGN_MONTH)
    features = rm.add_relative(features, train_years)
    table = readings.merge(features.drop(columns="month"), on=["well_uid", "year", "campaign"], how="inner")
    for c in CATEGORICAL:
        table[c] = table[c].astype("category")
    table["in_train"] = table["year"].isin(list(train_years))
    return table.sort_values(["well_uid", "date"], ignore_index=True)


def matrix(table: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Just the model inputs, in a fixed column order."""
    leaked = [c for c in features if c in FORBIDDEN]
    if leaked:
        raise ValueError(f"these columns must never be model inputs: {leaked}")
    return table[list(features)]


def apply_categories(table: pd.DataFrame, categories: dict[str, list] | None) -> pd.DataFrame:
    """Re-cast the categorical columns to the exact dtype a model was trained with.

    Measured against the saved model rather than assumed:

    * a column of plain strings is refused outright ("train and valid dataset
      categorical_feature do not match"); this restores it and the predictions
      come back bit for bit. A table read from CSV, or rebuilt in a notebook,
      arrives exactly that way, which is the case this exists for.
    * a differently ordered list is already harmless: LightGBM keeps the training
      labels and remaps pandas categoricals by label.
    * a label dropped while rows still use it cannot be repaired here, since
      those values are NaN before this is called, and renamed labels cannot
      either. scripts/check_colab_bundle.py is what catches those.
    """
    if not categories:
        return table
    out = table.copy()
    for column, values in categories.items():
        if column in out.columns:
            out[column] = pd.Categorical(out[column].astype(str), categories=[str(v) for v in values])
    return out


def categories_of(table: pd.DataFrame) -> dict[str, list]:
    """The label set a table was built with, to be saved alongside a model."""
    return {c: [str(v) for v in table[c].cat.categories] for c in CATEGORICAL if c in table.columns}


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("well_uid", "key", "well identity, built from its coordinates"),
        ("year, campaign, date", "key", "which of the four yearly readings this row is"),
        ("depth_mbgl", "raw", "water level, metres below ground; deeper is a larger number"),
        ("normal_m", "raw", "that well's mean depth for this campaign across training years"),
        (TARGET, "target", "depth minus normal; positive means deeper than normal for that well"),
        ("rain_3m … rain_36m", "rain", "rain over the 3 to 36 months before the reading, mm"),
        ("rain_*_pct", "rain", "the same, as a percentage of that well's own normal for that window"),
        ("monsoon_last_mm", "rain", "the last completed Jun-Sep monsoon, mm (and _pct)"),
        ("monsoon_todate_mm", "rain", "monsoon rain so far this year, 0 before June"),
        ("wet_days_12m", "rain", "days above 2.5 mm in the last 12 months, counted at the grid nodes"),
        ("max_day_12m", "rain", "wettest single day in the last 12 months, mm"),
        ("rain_normal_mm", "well", "the well's normal annual rain: dry region or wet region"),
        ("sy, well_depth_m", "well", "specific yield and drilled depth"),
        ("well_type, aquifer", "well", "dug or bore well; unconfined, semi-confined or confined"),
        ("district, state, lat, lon", "grouping only", "used to split folds and to report, never as inputs"),
        ("far_from_district", "flag", "well sits far from its stated district; excluded from district results"),
        ("in_train", "flag", "row belongs to the training years that defined the normals"),
    ]
    return pd.DataFrame(rows, columns=["column", "role", "meaning"])


def main() -> None:
    table = build()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "well_campaign_table.parquet"
    table.to_parquet(out, index=False)
    dictionary = DATA_PROCESSED / "well_campaign_table_dictionary.csv"
    data_dictionary().to_csv(dictionary, index=False)

    print(f"rows {len(table):,} | wells {table['well_uid'].nunique():,} | districts {table[GROUP].nunique()} "
          f"| years {table['year'].min()}-{table['year'].max()}")
    print(f"features {len(FEATURES)} ({len(RAIN_FEATURES)} rain, {len(WELL_FEATURES)} well, {len(CATEGORICAL)} categorical)")
    print("\nrows and target spread per campaign:")
    summary = table.groupby("campaign", observed=True).agg(
        rows=(TARGET, "size"), wells=("well_uid", "nunique"),
        anomaly_sd=(TARGET, "std"), depth_median=("depth_mbgl", "median"))
    print(summary.round(2).to_string())
    missing = matrix(table).isna().mean().sort_values(ascending=False)
    print(f"\nfeatures with gaps: {dict(missing[missing > 0].round(3)) or 'none'}")
    print(f"wrote {out.name} and {dictionary.name} to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
