"""Evidence: how the IMD rain grid is converted to districts and states.

The rain grid knows nothing about districts. It is 129 x 135 points, 0.25 degrees
apart, each holding mm of rain per day. This script rebuilds the conversion to
district and state from scratch, checks it against the tables the pipeline
wrote, and works one district through by hand from the raw NetCDF, so anyone
can see every step and its result.

    step 1  the raw grid: size, spacing, the -999 code outside land
    step 2  each grid point becomes a 0.25 degree cell (point +/- 0.125 degrees)
    step 3  each cell is cut by the district outlines; the overlap in km2 is kept
    step 4  checks: shares add up, the mapping matches the stored bridge, areas are sane
    step 5  district -> state: the same overlaps, summed by state
    step 6  worked example: one district's season rain, by hand from the raw file

Writes to data_cleaning/data/report/:
    rain_grid_mapping.txt   this log
    grid_district_map.csv   land cell x district: overlap km2, share of the cell, weight in the district
    grid_state_map.csv      land cell x state: the same, by state

Run from ml/:  python -m scripts.rain_grid_mapping
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np
import pandas as pd
import shapely
import xarray as xr

from bits_ml import districts as ds
from bits_ml import grid as gr
from bits_ml import rainfall as rf
from bits_ml.config import CORE_DIR, IMD_DIR

REPORT_DIR = CORE_DIR / "report"
EXAMPLE = ("Pune", "Maharashtra")
EXAMPLE_SEASON = ("May-Aug", 2019)
pd.set_option("display.width", 160)


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nstep {n}. {title}\n{'=' * 78}")


def run() -> None:
    step(1, "the raw grid")
    path = IMD_DIR / "imd_rf25_2019.nc"
    with xr.open_dataset(path, mask_and_scale=False) as raw:
        values = raw["RAINFALL"].values
        lat, lon = raw["LATITUDE"].values, raw["LONGITUDE"].values
        print(f"file {path.name}: dimensions {dict(raw.sizes)}")
    print(f"latitude  {lat[0]} to {lat[-1]} in steps of {lat[1] - lat[0]} ({len(lat)} rows)")
    print(f"longitude {lon[0]} to {lon[-1]} in steps of {lon[1] - lon[0]} ({len(lon)} columns)")
    code = values[0] == -999
    print(f"grid points {code.size:,} | coded -999 (sea, outside India) {int(code.sum()):,} "
          f"| land {int((~code).sum()):,}")
    print("the -999 code is read as missing, never as 0 mm: a missing value summed as zero "
          "would invent a dry day")

    step(2, "grid points become cells")
    cells = pd.read_parquet(CORE_DIR / "dim_cell.parquet")
    land = cells[cells["is_land"]].copy()
    print(f"each point is the centre of a {rf.IMD.step} x {rf.IMD.step} degree box, "
          f"about {rf.IMD.step * gr.EARTH_KM_PER_DEG:.0f} km a side")
    print(f"land cells {len(land):,}; cell_id is the centre, e.g. {land['cell_id'].iloc[len(land) // 2]}")
    example_box = shapely.box(78.125, 17.125, 78.375, 17.375)
    print(f"example: cell 17.25_78.25 covers lon 78.125-78.375, lat 17.125-17.375, "
          f"{example_box.area * gr.EARTH_KM_PER_DEG ** 2 * np.cos(np.radians(17.25)):.0f} km2")

    step(3, "cells cut by the district outlines")
    outlines = ds.load_districts()
    print(f"outlines {len(outlines)} districts in {outlines['state'].nunique()} states/UTs "
          f"(data_cleaning/reference/districts.geojson, post-2020 boundaries)")
    rebuilt = ds.cell_district_area(land, outlines)
    print("for every land cell: intersect its box with every outline it touches, keep the "
          "overlap area\n  area km2 = overlap in degrees2 x 111.2^2 x cos(latitude)\n"
          "  frac_of_cell = overlap / cell box;  weight_in_district = overlap / district's total")
    print(f"\nrows (cell x district pairs) {len(rebuilt):,} | cells mapped {rebuilt['cell_id'].nunique():,} "
          f"| districts reached {rebuilt.groupby(['district', 'state']).ngroups} of {len(outlines)}")
    per_cell = rebuilt.groupby("cell_id").size()
    print("districts per cell: " + ", ".join(f"{k} -> {v:,} cells" for k, v in per_cell.value_counts().sort_index().items()))

    step(4, "checks")
    stored = pd.read_parquet(CORE_DIR / "bridge_cell_district.parquet")
    key = ["cell_id", "district", "state"]
    both = rebuilt.merge(stored, on=key, suffixes=("", "_stored"), how="outer", indicator=True)
    same = both["_merge"].eq("both").all() and np.allclose(both["area_km2"], both["area_km2_stored"])
    print(f"rebuilt mapping identical to the stored bridge_cell_district: {same}")
    weight_sum = rebuilt.groupby(["district", "state"])["weight_in_district"].sum()
    print(f"weights sum to 1 in every district: {np.allclose(weight_sum, 1)} "
          f"(min {weight_sum.min():.6f}, max {weight_sum.max():.6f})")
    cell_sum = rebuilt.groupby("cell_id")["frac_of_cell"].sum()
    print(f"cells fully inside outlines {int((cell_sum > 0.99).sum()):,} | on a coast or border "
          f"{int((cell_sum <= 0.99).sum()):,}")
    over = cell_sum[cell_sum > 1 + 1e-9]
    overlapping = rebuilt[rebuilt["cell_id"].isin(over.index)]
    print(f"cells whose shares add to more than 1 (outlines overlap): {len(over)}, worst {over.max():.3f}, "
          f"in {', '.join(sorted(overlapping['state'].unique()))} "
          f"({', '.join(sorted(overlapping['district'].unique()))})")
    print("  the boundary file draws the new Vav-Tharad district over part of its parent Banaskantha; "
          "the overlap is a few % of 9 cells and is left as drawn")
    unmapped = land[~land["cell_id"].isin(rebuilt["cell_id"])]
    print(f"land cells outside every outline {len(unmapped)} "
          f"(lat {unmapped['cell_lat'].min()}-{unmapped['cell_lat'].max()}, "
          f"lon {unmapped['cell_lon'].min()}-{unmapped['cell_lon'].max()}): beyond the drawn border")

    geod_km2 = outlines.assign(area=[g.area for g in outlines["geometry"]])
    geod_km2["area_km2"] = geod_km2["area"] * gr.EARTH_KM_PER_DEG ** 2 * np.cos(
        np.radians([g.centroid.y for g in outlines["geometry"]]))
    gridded = rebuilt.groupby(["district", "state"])["area_km2"].sum()
    ratio = (gridded / geod_km2.set_index(["district", "state"])["area_km2"]).dropna()
    print(f"gridded land area / outline area per district: median {ratio.median():.3f}, "
          f"5th-95th percentile {ratio.quantile(0.05):.3f}-{ratio.quantile(0.95):.3f}")
    print(f"  (below 1 where part of a district is sea or outside the IMD land mask)")
    print(f"total gridded land {rebuilt['area_km2'].sum() / 1e6:.2f} million km2 "
          f"(India's official area is 3.29 million km2)")
    missed = outlines.set_index(["district", "state"]).index.difference(gridded.index)
    print(f"districts no land cell touches: {', '.join(f'{d} ({s})' for d, s in missed)}")

    step(5, "district -> state")
    by_state = rebuilt.groupby(["cell_id", "state"], as_index=False).agg(
        area_km2=("area_km2", "sum"), frac_of_cell=("frac_of_cell", "sum"))
    by_state["weight_in_state"] = by_state["area_km2"] / by_state.groupby("state")["area_km2"].transform("sum")
    print("a cell's overlap with each district is summed by the district's state, so a cell on a "
          "state border is split between states by area, like a district border")
    print(f"rows {len(by_state):,} | states/UTs reached {by_state['state'].nunique()} | "
          f"cells split across states {int((by_state.groupby('cell_id').size() > 1).sum()):,}")
    summary = by_state.groupby("state").agg(cells=("cell_id", "nunique"), land_km2=("area_km2", "sum"))
    print(summary.sort_values("land_km2", ascending=False).round(0).astype(int).head(10).to_string())

    step(6, f"worked example: {EXAMPLE[0]}, {EXAMPLE_SEASON[0]} {EXAMPLE_SEASON[1]}, by hand")
    mine = rebuilt[(rebuilt["district"] == EXAMPLE[0]) & (rebuilt["state"] == EXAMPLE[1])].copy()
    seasons = pd.read_parquet(CORE_DIR / "dim_season.parquet")
    season = seasons[(seasons["season"] == EXAMPLE_SEASON[0]) & (seasons["season_year"] == EXAMPLE_SEASON[1])].iloc[0]
    with xr.open_dataset(IMD_DIR / f"imd_rf25_{season['start'].year}.nc") as nc:
        window = nc["RAINFALL"].sel(TIME=slice(season["start"], season["end"])).load()
    mine["season_rain_mm"] = [
        float(window.sel(LATITUDE=float(c.split("_")[0]), LONGITUDE=float(c.split("_")[1])).sum())
        for c in mine["cell_id"]]
    mine["contribution_mm"] = mine["season_rain_mm"] * mine["weight_in_district"]
    print(f"{season['start']:%d %b %Y} to {season['end']:%d %b %Y}, {season['n_days']} days, "
          f"summed straight from {IMD_DIR.name}/imd_rf25_{season['start'].year}.nc")
    print(mine[["cell_id", "area_km2", "frac_of_cell", "weight_in_district", "season_rain_mm", "contribution_mm"]]
          .round(3).to_string(index=False))
    by_hand = mine["contribution_mm"].sum()
    table = pd.read_parquet(CORE_DIR / "fact_district_season.parquet")
    stored_value = table[(table["district"] == EXAMPLE[0]) & (table["state"] == EXAMPLE[1])
                         & (table["season_idx"] == season["season_idx"])]["rain_mm"].item()
    print(f"\ndistrict rain = sum of (cell rain x weight) = {by_hand:.3f} mm")
    print(f"fact_district_season holds                    {stored_value:.3f} mm "
          f"(difference {abs(by_hand - stored_value):.6f} mm)")
    print(f"a plain mean of the cells would give          {mine['season_rain_mm'].mean():.3f} mm, "
          f"over-weighting the {int((mine['frac_of_cell'] < 0.5).sum())} cells that are mostly outside the district")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = rebuilt.merge(land[["cell_id", "cell_lat", "cell_lon"]], on="cell_id")
    out[["cell_id", "cell_lat", "cell_lon", "district", "state", "area_km2", "frac_of_cell",
         "weight_in_district"]].to_csv(REPORT_DIR / "grid_district_map.csv", index=False)
    by_state.merge(land[["cell_id", "cell_lat", "cell_lon"]], on="cell_id")[
        ["cell_id", "cell_lat", "cell_lon", "state", "area_km2", "frac_of_cell", "weight_in_state"]
    ].to_csv(REPORT_DIR / "grid_state_map.csv", index=False)
    print(f"\nwrote grid_district_map.csv, grid_state_map.csv, rain_grid_mapping.txt to {REPORT_DIR}")


def main() -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run()
    text = buffer.getvalue()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rain_grid_mapping.txt").write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
