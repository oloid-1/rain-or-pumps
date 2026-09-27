"""Three ways of asking the same question, so the model is judged on the right one.

The decision this project serves is not "how deep is the water this November". It
is "is this district losing water it will not get back, and would more rain bring
it back". Three targets answer that with different sharpness, and each is
expressed as a deviation from that well's own normal, so the fixed depth of the
site — which says where it was drilled, not what is happening to it — drops out.

**level**   how far the water stands from its normal for that campaign.
            Four rows a year, the most data, but a level is the running total of
            every year that came before, so rain explains only part of it.

**rise**    the monsoon recharge, May to November, one row per well-year. The
            cleanest rain signal in the record: the water that arrives in a
            single season. If rain cannot explain this, it cannot explain
            anything, so this is the honest test of the rainfall model.

**annual**  the net change from one November to the next, one row per well-year.
            This is the business target. A district that recharges well every
            monsoon and still ends each year lower is exactly the "structural
            decline" the problem statement asks us to find, and the residual here
            is what a recharge intervention would have to overcome.

Rain features always come from the campaign that closes the interval, so no
reading is explained by rain that fell after it.

Run from ml/:  python -m bits_ml.targets
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import splits as S
from .dataset import CATEGORICAL, FEATURES, GROUP, build

MIN_TRAIN_YEARS = 5
GROUPING = ["district", "state", "far_from_district", "lat", "lon"]
TARGET = "target_m"


def _deviation(frame: pd.DataFrame, column: str, train_years, min_years: int = MIN_TRAIN_YEARS) -> pd.DataFrame:
    """Express a per-well series as a deviation from that well's training-year normal.

    A well with too few training years is dropped rather than compared against a
    normal built from two readings.
    """
    train = frame[frame["year"].isin(list(train_years))]
    normal = train.groupby("well_uid")[column].agg(well_normal="mean", n_train="size").reset_index()
    normal = normal[normal["n_train"] >= min_years]
    out = frame.merge(normal, on="well_uid", how="inner")
    out[TARGET] = out[column] - out["well_normal"]
    return out


def _closing_campaign_rows(table: pd.DataFrame, campaign: str = "Nov") -> pd.DataFrame:
    """The row whose rain history covers the interval being measured."""
    keep = ["well_uid", "year", *FEATURES, *GROUPING]
    # `campaign` stays, constant at the closing campaign, so the feature list is
    # the same shape for every target and the models need no special case.
    return table[table["campaign"] == campaign][keep].copy()


def level(table: pd.DataFrame, train_years=S.TRAIN_YEARS) -> pd.DataFrame:
    """Depth minus that well's normal for that campaign: the table as dataset.py builds it."""
    out = table.copy()
    out[TARGET] = out["anomaly_m"]
    return out


def rise(table: pd.DataFrame, train_years=S.TRAIN_YEARS) -> pd.DataFrame:
    """May to November recharge, as a deviation from that well's normal monsoon."""
    depth = table.pivot_table(index=["well_uid", "year"], columns="campaign", values="depth_mbgl", observed=True)
    if not {"May", "Nov"} <= set(depth.columns):
        raise ValueError("both the May and November campaigns are needed for the rise target")
    moves = (depth["May"] - depth["Nov"]).rename("rise_m").reset_index().dropna(subset=["rise_m"])
    frame = moves.merge(_closing_campaign_rows(table, "Nov"), on=["well_uid", "year"], how="inner")
    return _deviation(frame, "rise_m", train_years)


def annual(table: pd.DataFrame, train_years=S.TRAIN_YEARS) -> pd.DataFrame:
    """November to November net change, as a deviation from that well's normal year.

    Positive means the water ended the year higher than it started. The row is
    filed under the closing year, whose rain history is what produced it.
    """
    november = (table[table["campaign"] == "Nov"][["well_uid", "year", "depth_mbgl"]]
                .sort_values(["well_uid", "year"]))
    previous = november.assign(year=november["year"] + 1).rename(columns={"depth_mbgl": "previous_depth"})
    joined = november.merge(previous, on=["well_uid", "year"], how="inner")
    joined["change_m"] = joined["previous_depth"] - joined["depth_mbgl"]  # positive = water rose over the year
    frame = joined[["well_uid", "year", "change_m"]].merge(
        _closing_campaign_rows(table, "Nov"), on=["well_uid", "year"], how="inner")
    return _deviation(frame, "change_m", train_years)


BUILDERS = {"level": level, "rise": rise, "annual": annual}

# Which way each target runs when rain arrives. A depth below ground gets
# smaller as the water comes up; a rise and a year's net gain get larger. A
# model given the wrong direction is forced to fit backwards and scores nothing,
# so anything that constrains the rain response has to ask this first.
WETTER_IS_HIGHER = {"level": False, "rise": True, "annual": True}
MEANING = {
    "level": "how far the water stands from normal (4 readings a year)",
    "rise": "monsoon recharge, May to November (1 a year)",
    "annual": "net change, November to November (1 a year)",
}


def build_target(name: str, table: pd.DataFrame | None = None, train_years=S.TRAIN_YEARS) -> pd.DataFrame:
    if name not in BUILDERS:
        raise ValueError(f"unknown target {name!r}; choose from {sorted(BUILDERS)}")
    table = build(train_years) if table is None else table
    frame = BUILDERS[name](table, train_years)
    for column in CATEGORICAL:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return frame.sort_values(["well_uid", "year"], ignore_index=True)


def main() -> None:
    table = build(S.TRAIN_YEARS)
    print(f"{'target':8s} {'rows':>8s} {'wells':>7s} {'years':>10s}  {'sd (m)':>7s}  {'|median|':>8s}  meaning")
    for name in BUILDERS:
        frame = build_target(name, table)
        years = f"{frame['year'].min()}-{frame['year'].max()}"
        print(f"{name:8s} {len(frame):8,d} {frame['well_uid'].nunique():7,d} {years:>10s}  "
              f"{frame[TARGET].std():7.2f}  {abs(frame[TARGET].median()):8.3f}  {MEANING[name]}")

    annual_frame = build_target("annual", table)
    falling = annual_frame.groupby("well_uid")["change_m"].mean()
    print(f"\nwells losing water in an average year: {int((falling < 0).sum()):,} of {len(falling):,} "
          f"({(falling < 0).mean() * 100:.0f}%), median {falling.median():+.3f} m/yr")
    rise_frame = build_target("rise", table)
    print(f"monsoon recharge: median {rise_frame['rise_m'].median():.2f} m, "
          f"spread across wells {rise_frame.groupby('well_uid')['rise_m'].mean().std():.2f} m")


if __name__ == "__main__":
    main()
