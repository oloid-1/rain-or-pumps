"""Daily IMD rain turned into one monthly row per cell, for every land cell.

The rain grid is daily and the wells are read four times a year, so the two
cannot be joined until they share a cadence. Monthly is the finest cadence both
sides can hold: rain aggregates to it exactly, and a water level lands on four
of the twelve months and is left empty in the rest.

This is the only table that touches the NetCDF files. Rain at a well is not
stored again here: a well's rain is the weighted sum of its stencil cells, and
`bridge_well_cell.weight_land` holds those weights, so `well_rain()` builds it
on demand from this one source. One table, no copy to fall out of step.

Wet days and the wettest day are counted at the grid nodes before any
interpolation, which is what keeps them unbiased (see rainfall.well_monthly),
and both are linear in the cell values, so the same weighted sum applies.

Zeros that are really missing. Besides the -999 code, IMD writes 0.0 mm on
every day of a year in some land cells with no gauge near them (669 cell-years
in 109 cells, mostly the north-east and the Gujarat/Rajasthan edge), and 0 mm for
whole monsoon months in cells where that month's median is over 100 mm. Both are
flagged `suspect_zero` and their rain set to missing. Anything that averages
cells (`weighted_rain`) then uses only the cells with data and renormalises
their weights, so a missing cell never counts as a dry one.

Run from ml/:  python -m bits_ml.rain_panel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import grid as gr
from . import rainfall as rf
from .config import CORE_DIR, IMD_DIR

RAIN_VALUES = ["rain_mm", "wet_days", "max_day_mm"]
SUSPECT_MONTH_MEDIAN_MM = 100.0   # a monsoon month this wet in a normal year is never exactly 0


def flag_suspect_zeros(cells: pd.DataFrame, monsoon=gr.MONSOON_MONTHS,
                       month_median_mm: float = SUSPECT_MONTH_MEDIAN_MM) -> pd.DataFrame:
    """Mark cell-months whose 0 mm is missing data, and blank their rain.

    Two rules: every month of a cell-year whose annual total is exactly 0, and a
    monsoon month with exactly 0 in a cell whose median for that month is over
    `month_median_mm`. Medians are taken over all years; the zeros pull them
    down, which only makes the rule more cautious.
    """
    out = cells.copy()
    year, month = out["month"].dt.year, out["month"].dt.month
    annual = out.groupby([out["cell_id"], year])["rain_mm"].transform("sum")
    median = out.groupby([out["cell_id"], month])["rain_mm"].transform("median")
    whole_year = annual == 0
    dry_monsoon = month.isin(list(monsoon)) & (out["rain_mm"] == 0) & (median > month_median_mm)
    out["suspect_zero"] = (whole_year | dry_monsoon).to_numpy()
    out.loc[out["suspect_zero"], RAIN_VALUES] = np.nan
    return out


def weighted_rain(frame: pd.DataFrame, keys: list[str], weight: str, columns: list[str]) -> pd.DataFrame:
    """Weighted mean of cell values per group, over the cells that have data.

    `frame` has one row per group x cell, a weight and the values. Weights are
    renormalised over the rows where `rain_mm` is present, so a missing cell is
    left out rather than read as 0 mm. A group with no cell left gets NaN, and
    `rain_cover` says what share of its weight had data.
    """
    present = frame["rain_mm"].notna()
    work = frame[keys].copy()
    work["_w"] = frame[weight].where(present, 0.0)
    work["_wall"] = frame[weight]
    for column in columns:
        work[column] = frame[column].where(present) * work["_w"]
    out = work.groupby(keys, as_index=False)[["_w", "_wall", *columns]].sum()
    for column in columns:
        out[column] = out[column] / out["_w"].where(out["_w"] > 0)
    out["rain_cover"] = out["_w"] / out["_wall"].where(out["_wall"] > 0)
    return out.drop(columns=["_w", "_wall"])


def cell_month(imd_dir: Path = IMD_DIR) -> pd.DataFrame:
    """Monthly rain, wet days and wettest day for every land cell of the grid.

    One pass per yearly file. Non-land cells are dropped rather than written as
    zero: a cell IMD does not report is not a dry cell.
    """
    files = sorted(Path(imd_dir).glob("imd_rf25_*.nc"))
    if not files:
        raise FileNotFoundError(f"no imd_rf25_*.nc files in {imd_dir}")

    rows, cols = np.divmod(np.arange(rf.IMD.size), rf.IMD.nlon)
    ids = gr.cell_id(rows, cols)
    frames = []
    for path in files:
        dates, daily = rf.read_year(path)
        rain, wet = rf.monthly_totals(dates, daily)
        wettest = rf.monthly_max(dates, daily)
        months = pd.date_range(f"{dates[0].year}-01-01", periods=12, freq="MS")
        keep = ~np.isnan(rain[0])  # the land mask of this year, same every year
        frames.append(pd.DataFrame({
            "month": np.repeat(months, keep.sum()),
            "cell_id": np.tile(ids[keep], 12),
            "rain_mm": rain[:, keep].ravel(),
            "wet_days": wet[:, keep].ravel(),
            "max_day_mm": wettest[:, keep].ravel(),
        }))
    out = pd.concat(frames, ignore_index=True)
    return flag_suspect_zeros(out).sort_values(["cell_id", "month"], ignore_index=True)


def well_rain(bridge: pd.DataFrame, cells: pd.DataFrame, well_uids=None) -> pd.DataFrame:
    """Rain at each well: the weighted sum of the cells its stencil covers.

    Pass `well_uids` to build it for one view rather than all 32,299 wells; the
    full cross with 300 months is 9.4 million rows and is rarely what is wanted.
    """
    stencil = bridge[bridge["weight_land"] > 0]
    if well_uids is not None:
        stencil = stencil[stencil["well_uid"].isin(set(well_uids))]
    joined = stencil.merge(cells[["cell_id", "month", *RAIN_VALUES]], on="cell_id", how="inner")
    out = weighted_rain(joined, ["well_uid", "month"], "weight_land", RAIN_VALUES)
    return out.sort_values(["well_uid", "month"], ignore_index=True)


def main() -> None:
    cells = cell_month()
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    out = CORE_DIR / "fact_cell_month.parquet"
    cells.to_parquet(out, index=False)

    span = f"{cells['month'].min():%Y-%m} to {cells['month'].max():%Y-%m}"
    print(f"rows {len(cells):,} | cells {cells['cell_id'].nunique():,} | months {cells['month'].nunique()} ({span})")
    print(f"cell-months flagged suspect_zero and set missing: {int(cells['suspect_zero'].sum()):,} "
          f"in {cells.loc[cells['suspect_zero'], 'cell_id'].nunique()} cells")
    yearly = cells.assign(year=cells["month"].dt.year).groupby("year")["rain_mm"].sum() / cells["cell_id"].nunique()
    print(f"mean rain per cell per year: {yearly.mean():.0f} mm "
          f"(driest {yearly.idxmin()} {yearly.min():.0f}, wettest {yearly.idxmax()} {yearly.max():.0f})")
    monsoon = cells[cells["month"].dt.month.isin(gr.MONSOON_MONTHS)]["rain_mm"].sum() / cells["rain_mm"].sum()
    print(f"share of rain falling June-September: {monsoon * 100:.0f}%")
    print(f"wrote {out.name} to {CORE_DIR}")


if __name__ == "__main__":
    main()
