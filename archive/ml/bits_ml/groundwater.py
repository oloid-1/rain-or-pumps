"""CGWB water levels as a long time series, with each well's own normal level.

The CGWB file holds one row per well and 92 reading columns (Jan, May, Aug and
Nov of 2000-2022). Models need the opposite shape: one row per well per reading,
so a single model can learn from every well at once instead of one model per
well.

Each reading is also expressed as an anomaly, how much deeper or shallower than
that well's own normal for that campaign. That removes the fixed depth of the
site, which says more about where the well was drilled than about rainfall or
pumping, and leaves what actually varies. Normals come from the training years
only, and a well-campaign with fewer than MIN_TRAIN_READINGS of them is dropped
rather than compared against a shaky normal.

Missing readings stay missing: no reading is ever filled in.

Run from ml/:  python -m bits_ml.groundwater
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import rainfall as rf
from .config import CGWB_CSV, DATA_PROCESSED

YEARS = tuple(range(2000, 2023))
CAMPAIGN_MONTH = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
MIN_TRAIN_READINGS = 5
LABEL_OUTLIER_KM = 150.0

_STATIC_COLUMNS = {
    "well_uid": "well_uid",
    "State": "state",
    "District": "district",
    "Station Name": "station",
    "Latitude": "lat",
    "Longitude": "lon",
    "Type of Well": "well_type",
    "Aquifer Type": "aquifer",
    "Well Depth": "well_depth_m",
    "Reference_Sy": "sy",
}


def label_outliers(static: pd.DataFrame, km: float = LABEL_OUTLIER_KM) -> pd.Series:
    """Wells sitting far from the other wells of their stated district.

    Some district labels are wrong (six wells labelled Anantapur stand in Delhi,
    Madhya Pradesh and Gujarat). Distance to the district's median well position
    catches them without needing a boundary map.
    """
    median = static.groupby("district")[["lat", "lon"]].transform("median")
    dy = (static["lat"] - median["lat"]) * 111.2
    dx = (static["lon"] - median["lon"]) * 111.2 * np.cos(np.radians(static["lat"]))
    return pd.Series(np.hypot(dx, dy) > km, index=static.index, name="far_from_district")


def load_static(path: Path = CGWB_CSV) -> pd.DataFrame:
    """One row per well: position, type, aquifer, specific yield, label flag."""
    wells = rf.load_wells(path)
    static = wells[list(_STATIC_COLUMNS)].rename(columns=_STATIC_COLUMNS)
    static["well_depth_m"] = pd.to_numeric(static["well_depth_m"], errors="coerce")
    static["sy"] = pd.to_numeric(static["sy"], errors="coerce")
    static["far_from_district"] = label_outliers(static)
    return static


def load_readings(path: Path = CGWB_CSV) -> pd.DataFrame:
    """Every water-level reading as its own row, with the well's attributes attached."""
    wells = rf.load_wells(path)
    reading_cols = [f"{c}-{y % 100:02d}" for y in YEARS for c in CAMPAIGN_MONTH]
    missing = [c for c in reading_cols if c not in wells.columns]
    if missing:
        raise ValueError(f"{path} is missing reading columns: {missing[:5]}")

    long = wells.melt(id_vars="well_uid", value_vars=reading_cols, var_name="column", value_name="depth_mbgl")
    long["depth_mbgl"] = pd.to_numeric(long["depth_mbgl"], errors="coerce")
    long = long.dropna(subset=["depth_mbgl"])

    parts = long["column"].str.split("-", n=1, expand=True)
    long["campaign"] = parts[0]
    long["year"] = 2000 + parts[1].astype(int)
    long["month"] = long["campaign"].map(CAMPAIGN_MONTH)
    long["date"] = pd.to_datetime(dict(year=long["year"], month=long["month"], day=15))
    long = long.drop(columns="column")
    return long.merge(load_static(path), on="well_uid", how="left").sort_values(["well_uid", "date"], ignore_index=True)


def add_normals(readings: pd.DataFrame, train_years, min_readings: int = MIN_TRAIN_READINGS) -> pd.DataFrame:
    """Attach each well-campaign's normal depth and the anomaly against it.

    The normal is the mean over training years only, so nothing from the
    validation or test years reaches the target.
    """
    train_years = list(train_years)
    train = readings[readings["year"].isin(train_years)]
    normal = (
        train.groupby(["well_uid", "campaign"])["depth_mbgl"]
        .agg(normal_m="mean", n_train_readings="size")
        .reset_index()
    )
    normal = normal[normal["n_train_readings"] >= min_readings]
    out = readings.merge(normal, on=["well_uid", "campaign"], how="inner")
    out["anomaly_m"] = out["depth_mbgl"] - out["normal_m"]
    return out.sort_values(["well_uid", "date"], ignore_index=True)


def seasonal_moves(readings: pd.DataFrame) -> pd.DataFrame:
    """Per well and year: the monsoon rise (May to Nov) and the dry-season fall (Nov to next May).

    Not model targets, but the two moves a groundwater year is made of, and the
    dry-season fall is the independent pumping check used in attribution.
    """
    wide = readings.pivot_table(index=["well_uid", "year"], columns="campaign", values="depth_mbgl").reset_index()
    for c in CAMPAIGN_MONTH:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide.sort_values(["well_uid", "year"])
    next_may = wide.groupby("well_uid")["May"].shift(-1)
    next_year = wide.groupby("well_uid")["year"].shift(-1)
    return pd.DataFrame({
        "well_uid": wide["well_uid"],
        "year": wide["year"],
        "monsoon_rise_m": wide["May"] - wide["Nov"],
        "dry_fall_m": np.where(next_year == wide["year"] + 1, wide["Nov"] - next_may, np.nan),
        "nov_depth_m": wide["Nov"],
    })


def coverage(readings: pd.DataFrame, n_wells: int) -> pd.DataFrame:
    """Share of wells read in each campaign of each year, as a percentage."""
    counts = readings.pivot_table(index="year", columns="campaign", values="depth_mbgl", aggfunc="size")
    return (counts.reindex(columns=list(CAMPAIGN_MONTH)) / n_wells * 100).round(0)


def main() -> None:
    static = load_static()
    readings = load_readings()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "well_readings_2000_2022.parquet"
    readings.to_parquet(out, index=False)

    print(f"wells {len(static):,} | readings {len(readings):,} of {len(static) * 92:,} possible "
          f"({len(readings) / (len(static) * 92) * 100:.1f}%)")
    print(f"wells far from their stated district: {int(static['far_from_district'].sum())}")
    print("\ncoverage, % of wells read:")
    print(coverage(readings, len(static)).to_string())
    moves = seasonal_moves(readings)
    print(f"\nmonsoon rise: {moves['monsoon_rise_m'].notna().sum():,} well-years, "
          f"median {moves['monsoon_rise_m'].median():.2f} m | "
          f"dry-season fall: {moves['dry_fall_m'].notna().sum():,}, median {moves['dry_fall_m'].median():.2f} m")
    print(f"wrote {out.name} to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
