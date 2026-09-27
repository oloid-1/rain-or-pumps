"""The IMD grid as the backbone every other table hangs off.

The rain grid is the one regular thing in this project: 0.25 degrees, 17,415
cells, the same every day since 1998. Wells are the irregular thing, 32,299 of
them clustered where the Board drilled. Making the grid the baseline and
attaching wells to it gives every table one join key and one geometry.

Two different attachments are needed and they are not the same:

* **home cell** — the cell a well stands in. This is how wells are grouped, so a
  cell with several wells can be summarised as one series.
* **rain stencil** — the cells a well's rain is read from. A well near a cell
  edge takes most of its rain from the neighbour it is closest to, so rain comes
  from up to four cells with weights, not from the home cell alone. Tested
  against the grid itself, that beats using the home cell: monthly error 17.7%
  against 25.6%.

Both are written out. `bridge_well_cell` carries the stencil with `is_home`
marking the containing cell, so a consumer can use either and say which.

Wells inside one cell are up to 27 km apart, and a single cell can hold 52 of
them, so summarising a cell means weighting its wells. `cell_weights` offers
inverse distance from the cell centre; `panel.py` also writes the plain mean and
the median beside it, because on a 27 km cell the distance to an arbitrary grid
centre is a weak claim about which well represents it and the three should be
compared rather than assumed equal.

Run from ml/:  python -m bits_ml.grid
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import rainfall as rf
from .config import CORE_DIR, IMD_DIR

PANEL_START, PANEL_END = 1998, 2022         # the years the rain grid covers
CAMPAIGN_MONTH = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
MONSOON_MONTHS = (6, 7, 8, 9)
IDW_FLOOR_KM = 2.0                          # a well on the centre must not take all the weight
EARTH_KM_PER_DEG = 111.2


def cell_id(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Readable cell key: the centre coordinates, e.g. "14.25_78.25"."""
    lat = rf.IMD.lat0 + rf.IMD.step * np.asarray(row)
    lon = rf.IMD.lon0 + rf.IMD.step * np.asarray(col)
    return np.array([f"{a:g}_{b:g}" for a, b in zip(lat, lon)])


def home_cell(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the cell each point stands in."""
    row = np.round((np.asarray(lat, dtype=float) - rf.IMD.lat0) / rf.IMD.step).astype(int)
    col = np.round((np.asarray(lon, dtype=float) - rf.IMD.lon0) / rf.IMD.step).astype(int)
    return np.clip(row, 0, rf.IMD.nlat - 1), np.clip(col, 0, rf.IMD.nlon - 1)


def land_mask(imd_dir: Path = IMD_DIR) -> np.ndarray:
    """True for cells IMD reports rain for, flat, one entry per cell.

    IMD writes -999 over sea and outside the national boundary, which xarray
    decodes to NaN. A cell is land if any day of the reference year has a value.
    """
    files = sorted(Path(imd_dir).glob("imd_rf25_*.nc"))
    if not files:
        raise FileNotFoundError(f"no imd_rf25_*.nc files in {imd_dir}")
    _, grid = rf.read_year(files[0])
    return ~np.isnan(grid).all(axis=0)


def dim_cell(wells: pd.DataFrame | None = None, imd_dir: Path = IMD_DIR) -> pd.DataFrame:
    """Every cell of the grid, land or not, with the wells that stand in it."""
    rows, cols = np.divmod(np.arange(rf.IMD.size), rf.IMD.nlon)
    cells = pd.DataFrame({
        "cell_id": cell_id(rows, cols),
        "cell_row": rows,
        "cell_col": cols,
        "cell_lat": rf.IMD.lat0 + rf.IMD.step * rows,
        "cell_lon": rf.IMD.lon0 + rf.IMD.step * cols,
        "is_land": land_mask(imd_dir),
    })
    if wells is not None:
        home = pd.Series(cell_id(*home_cell(wells["lat"], wells["lon"])), index=wells.index)
        cells["n_wells"] = cells["cell_id"].map(home.value_counts()).fillna(0).astype(int)
        modelled = home[wells["view_modelling"].to_numpy()]
        cells["n_wells_modelling"] = cells["cell_id"].map(modelled.value_counts()).fillna(0).astype(int)
    return cells


def name_cells(cells: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Each cell named after the district covering most of it; bridge_cell_district has the full split."""
    main = bridge.sort_values("area_km2", ascending=False).drop_duplicates("cell_id")
    split = bridge.groupby("cell_id").size().rename("n_districts")
    out = cells.merge(main[["cell_id", "district", "state"]], on="cell_id", how="left")
    out["n_districts"] = out["cell_id"].map(split).fillna(0).astype(int)
    return out


def dim_month(start: int = PANEL_START, end: int = PANEL_END) -> pd.DataFrame:
    """The monthly spine, 1998-01 to 2022-12.

    Rain exists for every month. Water levels exist for four of them, so
    `is_campaign_month` is 4/12 of the spine and the rest of the level columns
    stay empty by design: CGWB reads these wells four times a year, and filling
    the other eight months would invent the signal the project measures.
    """
    months = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    frame = pd.DataFrame({"month": months})
    frame["year"] = frame["month"].dt.year
    frame["month_no"] = frame["month"].dt.month
    campaign = {v: k for k, v in CAMPAIGN_MONTH.items()}
    frame["campaign"] = frame["month_no"].map(campaign)
    frame["is_campaign_month"] = frame["campaign"].notna()
    frame["is_monsoon"] = frame["month_no"].isin(MONSOON_MONTHS)
    # A water year runs June to May, so one monsoon and the dry season it feeds
    # stay in the same year instead of being split across the calendar boundary.
    frame["water_year"] = np.where(frame["month_no"] >= 6, frame["year"], frame["year"] - 1)
    frame["season"] = pd.cut(frame["month_no"], [0, 2, 5, 9, 12],
                             labels=["winter", "pre_monsoon", "monsoon", "post_monsoon"])
    return frame


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres between two sets of points."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def bridge_well_cell(wells: pd.DataFrame) -> pd.DataFrame:
    """Well to cell, one row per cell a well draws rain from.

    The stencil is the same masked-bilinear one the rainfall module samples
    with, widened to the rounding box for wells whose coordinates were reported
    to a tenth of a degree, so the weights here are exactly the weights the rain
    was read with. `is_home` marks the cell the well physically stands in, which
    is not always the heaviest weight.
    """
    lat, lon = wells["lat"].to_numpy(), wells["lon"].to_numpy()
    half = wells["coord_half_width_deg"].to_numpy()
    stencil = rf.box_stencil(lat, lon, half)

    n_wells, k = stencil.idx.shape
    flat = pd.DataFrame({
        "well_uid": np.repeat(wells["well_uid"].to_numpy(), k),
        "flat_index": stencil.idx.ravel(),
        "weight": stencil.w.ravel(),
    })
    flat = flat[flat["weight"] > 0]
    rows, cols = np.divmod(flat["flat_index"].to_numpy(), rf.IMD.nlon)
    flat["cell_id"] = cell_id(rows, cols)
    flat["cell_lat"] = rf.IMD.lat0 + rf.IMD.step * rows
    flat["cell_lon"] = rf.IMD.lon0 + rf.IMD.step * cols

    home = pd.Series(cell_id(*home_cell(lat, lon)), index=wells["well_uid"].to_numpy())
    flat["is_home"] = flat["cell_id"].to_numpy() == home.reindex(flat["well_uid"]).to_numpy()
    well_lat = wells.set_index("well_uid")["lat"].reindex(flat["well_uid"]).to_numpy()
    well_lon = wells.set_index("well_uid")["lon"].reindex(flat["well_uid"]).to_numpy()
    flat["distance_km"] = haversine_km(well_lat, well_lon, flat["cell_lat"], flat["cell_lon"])

    # A coastal well can have a sea cell in its stencil, and IMD reports nothing
    # there. weight_land drops those and renormalises over the rest, so summing
    # weight_land * cell rain reproduces the interpolation the rainfall module
    # does per day, instead of quietly counting a sea cell as a dry one.
    land = pd.Series(land_mask(), index=cell_id(*np.divmod(np.arange(rf.IMD.size), rf.IMD.nlon)))
    on_land = land.reindex(flat["cell_id"]).to_numpy()
    weight = np.where(on_land, flat["weight"].to_numpy(), 0.0)
    total = pd.Series(weight).groupby(flat["well_uid"].to_numpy()).transform("sum").to_numpy()
    flat["weight_land"] = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    return flat.drop(columns="flat_index").reset_index(drop=True)


def bridge_cell_district(cells: pd.DataFrame) -> pd.DataFrame:
    """How each land cell's area splits across district outlines.

    Cells cross district lines and districts hold several cells, so a district
    figure built by averaging whole cells double counts. Splitting by area gives
    rain to every district the grid reaches, with or without a well in it.
    """
    from . import districts as ds  # districts imports this module for its constants
    return ds.cell_district_area(cells[cells["is_land"]], ds.load_districts())


def cell_weights(wells: pd.DataFrame, floor_km: float = IDW_FLOOR_KM) -> pd.DataFrame:
    """Inverse-distance weight of each well within its own cell.

    A cell is up to 27 km across and can hold 52 wells, so which well speaks for
    the cell matters. Weight is 1 / (distance to the cell centre + floor), the
    floor keeping a well that happens to sit on the centre from taking the whole
    weight, and weights are normalised within the cell.
    """
    rows, cols = home_cell(wells["lat"], wells["lon"])
    out = pd.DataFrame({
        "well_uid": wells["well_uid"].to_numpy(),
        "cell_id": cell_id(rows, cols),
        "cell_lat": rf.IMD.lat0 + rf.IMD.step * rows,
        "cell_lon": rf.IMD.lon0 + rf.IMD.step * cols,
    })
    out["distance_km"] = haversine_km(wells["lat"], wells["lon"], out["cell_lat"], out["cell_lon"])
    raw = 1.0 / (out["distance_km"] + floor_km)
    out["idw_weight"] = raw / raw.groupby(out["cell_id"]).transform("sum")
    out["n_wells_in_cell"] = out.groupby("cell_id")["well_uid"].transform("size")
    return out


def main() -> None:
    wells = pd.read_parquet(CORE_DIR / "well_master.parquet")
    cells = dim_cell(wells)
    months = dim_month()
    bridge = bridge_well_cell(wells)
    district = bridge_cell_district(cells)
    cells = name_cells(cells, district)
    weights = cell_weights(wells)

    CORE_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(CORE_DIR / "dim_cell.parquet", index=False)
    months.to_parquet(CORE_DIR / "dim_month.parquet", index=False)
    bridge.to_parquet(CORE_DIR / "bridge_well_cell.parquet", index=False)
    district.to_parquet(CORE_DIR / "bridge_cell_district.parquet", index=False)
    weights.to_parquet(CORE_DIR / "well_cell_weights.parquet", index=False)

    land = cells[cells["is_land"]]
    occupied = cells[cells["n_wells"] > 0]
    modelled = cells[cells["n_wells_modelling"] > 0]
    print(f"cells {len(cells):,} | land {len(land):,} | with wells {len(occupied):,} "
          f"| with modelling wells {len(modelled):,}")
    print(f"months {len(months):,} ({months['month'].min():%Y-%m} to {months['month'].max():%Y-%m}), "
          f"campaign months {int(months['is_campaign_month'].sum())}")
    print("\nwells per occupied cell:")
    for name, column in (("all wells", "n_wells"), ("modelling view", "n_wells_modelling")):
        counts = cells.loc[cells[column] > 0, column]
        print(f"  {name:<16} cells {len(counts):>6,} | median {counts.median():.0f} "
              f"| p90 {counts.quantile(0.9):.0f} | max {counts.max()} "
              f"| single-well cells {int((counts == 1).sum()):,}")
    print(f"\nwell-cell stencil rows {len(bridge):,} "
          f"({len(bridge) / wells['well_uid'].nunique():.1f} cells per well) | "
          f"home cell is heaviest for {bridge.loc[bridge.groupby('well_uid')['weight'].idxmax(), 'is_home'].mean() * 100:.0f}% of wells")
    print(f"cells crossing a district line: {int((cells['n_districts'] > 1).sum()):,} | districts the grid reaches "
          f"{district.groupby(['district', 'state']).ngroups} | land cells outside every outline "
          f"{int((cells['is_land'] & (cells['n_districts'] == 0)).sum())}")
    print(f"\nwrote dim_cell, dim_month, bridge_well_cell, bridge_cell_district, "
          f"well_cell_weights to {CORE_DIR}")


if __name__ == "__main__":
    main()
