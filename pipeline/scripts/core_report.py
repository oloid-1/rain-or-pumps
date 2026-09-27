"""What the rebuilt core tables actually contain, before any model is chosen.

Every number printed here comes from data_cleaning/data. The sections answer
the questions that decide what can be modelled at all: how much of the record
exists, where it exists, what the quality flags cost, whether wells standing in
one grid cell agree with each other, and whether rain moves the water level at
all at the cadence the panel holds.

Run from ml/:  python -m scripts.core_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bits_ml.config import CORE_DIR
from bits_ml.ingest import CAMPAIGNS

REPORT_DIR = CORE_DIR / "report"
pd.set_option("display.width", 200)


def load() -> dict[str, pd.DataFrame]:
    names = ["well_master", "well_obs", "dim_cell", "fact_well_month",
             "fact_cell_month_full", "fact_district_month"]
    return {name: pd.read_parquet(CORE_DIR / f"{name}.parquet") for name in names}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def inventory(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in tables.items():
        size = (CORE_DIR / f"{name}.parquet").stat().st_size / 1e6
        rows.append((name, len(frame), len(frame.columns), round(size, 1)))
    return pd.DataFrame(rows, columns=["table", "rows", "columns", "mb"])


def coverage_by_year(obs: pd.DataFrame, n_wells: int) -> pd.DataFrame:
    """Share of modelling wells read in each campaign of each year."""
    counts = obs.pivot_table(index="year", columns="campaign", values="depth_mbgl", aggfunc="size")
    return (counts.reindex(columns=list(CAMPAIGNS)) / n_wells * 100).round(0)


def flags_by_group(obs: pd.DataFrame, wells: pd.DataFrame, by: str) -> pd.DataFrame:
    joined = obs.merge(wells[["well_uid", by]], on="well_uid", how="left")
    flags = [c for c in joined.columns if c.startswith("flag_")]
    out = joined.groupby(by)[flags].mean() * 100
    out["readings"] = joined.groupby(by).size()
    return out.sort_values("readings", ascending=False).round(1)


def seasonal_shape(panel: pd.DataFrame) -> pd.DataFrame:
    """The water year in four readings, and the rain that arrives between them."""
    level = panel[panel["depth_mbgl"].notna()]
    out = level.groupby("campaign").agg(
        readings=("depth_mbgl", "size"),
        depth_median_m=("depth_mbgl", "median"),
        anomaly_sd_m=("anomaly_ref_m", "std"),
    )
    rain = panel.groupby("month_no")["rain_mm"].mean()
    out["rain_that_month_mm"] = [rain.get({"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}[c]) for c in out.index]
    return out.reindex(list(CAMPAIGNS)).round(2)


def within_cell_disagreement(cells: pd.DataFrame) -> pd.DataFrame:
    """How far apart the wells of one cell stand, by how many wells the cell has."""
    read = cells[cells["n_wells_read"] >= 2].copy()
    read["wells"] = pd.cut(read["n_wells_read"], [1, 2, 3, 5, 10, 100],
                           labels=["2", "3", "4-5", "6-10", "11+"])
    out = read.groupby("wells", observed=True).agg(
        cell_months=("anomaly_spread_m", "size"),
        median_spread_m=("anomaly_spread_m", "median"),
        p90_spread_m=("anomaly_spread_m", lambda s: s.quantile(0.9)),
        median_abs_anomaly_m=("anomaly_mean_m", lambda s: s.abs().median()),
    )
    return out.round(2)


def rain_response(cells: pd.DataFrame) -> pd.DataFrame:
    """Does more monsoon rain mean shallower water that November?

    Per cell, across years: the correlation between June-September rain and the
    November anomaly. A negative number is the physical expectation, since the
    anomaly is depth and more rain should make it smaller.
    """
    monsoon = cells[cells["is_monsoon"]].groupby(["cell_id", "year"], as_index=False)["rain_mm"].sum()
    november = cells[(cells["month_no"] == 11) & cells["anomaly_mean_m"].notna()]
    joined = monsoon.merge(november[["cell_id", "year", "anomaly_mean_m", "n_wells_read"]],
                           on=["cell_id", "year"], how="inner")
    rows = []
    for min_years in (8, 12):
        counts = joined.groupby("cell_id").size()
        usable = joined[joined["cell_id"].isin(counts[counts >= min_years].index)]
        correlation = usable.groupby("cell_id").apply(
            lambda g: g["rain_mm"].corr(g["anomaly_mean_m"]), include_groups=False).dropna()
        rows.append((min_years, len(correlation), round(correlation.median(), 3),
                     round(float((correlation < 0).mean() * 100)), round(correlation.quantile(0.1), 3),
                     round(correlation.quantile(0.9), 3)))
    return pd.DataFrame(rows, columns=["min_years", "cells", "median_r", "pct_negative", "p10", "p90"])


def november_trends(panel: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    """Per-well November trend in metres a year, split by view."""
    november = panel[(panel["month_no"] == 11) & panel["depth_mbgl"].notna()]
    rows = []
    for name, subset in (("all modelling wells", november),):
        grouped = subset.groupby("well_uid")
        slopes = grouped.apply(lambda g: np.polyfit(g["year"], g["depth_mbgl"], 1)[0]
                               if len(g) >= 8 else np.nan, include_groups=False).dropna()
        strict = wells.set_index("well_uid")["view_strict"].reindex(slopes.index).fillna(False)
        for label, values in (("all", slopes), ("in published 2,759", slopes[strict.to_numpy()]),
                              ("added by rebuild", slopes[~strict.to_numpy()])):
            rows.append((label, len(values), round(values.median(), 4), round(values.mean(), 4),
                         round(float((values > 0).mean() * 100))))
    return pd.DataFrame(rows, columns=["wells", "n", "median_m_per_yr", "mean_m_per_yr", "pct_deepening"])


def main() -> None:
    tables = load()
    wells, obs = tables["well_master"], tables["well_obs"]
    panel, cells = tables["fact_well_month"], tables["fact_cell_month_full"]
    modelling = wells[wells["view_modelling"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    section("1. what was built")
    print(inventory(tables).to_string(index=False))

    section("2. coverage in time, % of the 8,629 modelling wells read")
    table = coverage_by_year(obs[obs["well_uid"].isin(set(modelling["well_uid"])) & obs["valid"]], len(modelling))
    print(table.to_string())
    worst = table.stack().sort_values().head(4)
    print("\nthinnest campaigns:", ", ".join(f"{c} {y} at {v:.0f}%" for (y, c), v in worst.items()))
    table.to_csv(REPORT_DIR / "coverage_by_year.csv")

    section("3. coverage in space")
    by_state = modelling.groupby("state").agg(
        wells=("well_uid", "size"), districts=("district", "nunique"),
        dug_well_pct=("well_type", lambda s: round((s == "Dug well").mean() * 100)),
        median_depth_m=("well_depth_m", "median"),
    ).sort_values("wells", ascending=False)
    print(by_state.head(12).to_string())
    print(f"\nstates {len(by_state)} | districts {modelling['district'].nunique()} "
          f"| grid cells occupied {tables['dim_cell']['n_wells_modelling'].gt(0).sum():,} of "
          f"{int(tables['dim_cell']['is_land'].sum()):,} land cells")
    by_state.to_csv(REPORT_DIR / "coverage_by_state.csv")

    section("4. what the quality flags cost, by well type")
    print(flags_by_group(obs, wells, "well_type").to_string())
    print("\nby state, worst five for suspect zeros:")
    state_flags = flags_by_group(obs, wells, "state")
    print(state_flags.sort_values("flag_zero_suspect", ascending=False).head(5).to_string())
    state_flags.to_csv(REPORT_DIR / "flags_by_state.csv")

    section("5. the shape of a water year")
    print(seasonal_shape(panel).to_string())

    section("6. do wells in one grid cell agree?")
    print(within_cell_disagreement(cells).to_string())
    single = cells[cells["n_wells_read"] == 1]
    print(f"\ncell-months resting on a single well: {len(single):,} of "
          f"{int((cells['n_wells_read'] > 0).sum()):,} ({len(single) / max((cells['n_wells_read'] > 0).sum(), 1) * 100:.0f}%)")

    section("7. does rain move the level? monsoon rain against the November anomaly, per cell")
    print(rain_response(cells).to_string(index=False))

    section("8. November trend per well, rebuilt set against the published one")
    trends = november_trends(panel, wells)
    print(trends.to_string(index=False))
    trends.to_csv(REPORT_DIR / "november_trends.csv", index=False)

    print(f"\ntables written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
