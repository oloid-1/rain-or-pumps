"""Rain and water level on one clock: four seasons cut at the well readings.

CGWB reads its wells in January, May, August and November. The file gives the
month, not the day, so every reading is placed on the 15th. A season is the
stretch between two readings, closed by the later one:

    season    from     to       days     closed by
    Nov-Jan   Nov 16   Jan 15   61       the January reading
    Jan-May   Jan 16   May 15   120/121  the May reading
    May-Aug   May 16   Aug 15   92       the August reading
    Aug-Nov   Aug 16   Nov 15   92       the November reading

They are deliberately unequal. Calendar quarters would split the monsoon across
the August reading and mix rain the level has already answered with rain it has
not. Here each season's rain is exactly the rain that fell between the two
levels bracketing it. The windows carry names rather than "winter" or
"monsoon" because `dim_month.season` already uses those for calendar months.

Within a season a well has one state and one change: the level at the reading
that closes it, and how far that moved since the reading before. Rain is the
variable: its total, wet days, wettest day, and how it compares with the
season's normal. The two meet in one unit through specific yield:

    delta_h_m          = depth now - depth at the previous reading (+ = water fell)
    storage_change_mm  = -delta_h_m x specific yield x 1000        (+ = water gained)
    recharge_ratio     = storage_change_mm / rain over the same span

A missing or invalid reading is never filled. The change is taken from the
last valid reading instead, up to MAX_SPAN_SEASONS back, and `span_seasons`
says how far that was; rain over the span is summed to match. A ratio is only
written where the span's rain reaches MIN_RAIN_FOR_RATIO_MM, because a dry
season divides a small change by almost nothing.

Tables, all in data_cleaning/data:

* `dim_season`            one row per season, 1998 Jan-May to 2022 Aug-Nov
* `fact_cell_season`      land cell x season: rain, from the daily grid
* `fact_well_season`      well x season: rain at the well and the level it closes on
* `fact_district_season`  district outline x season: area-weighted rain, wells summarised

Run from ml/ after panel.py:  python -m bits_ml.seasons
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import grid as gr
from . import panel as pn
from . import rain_panel as rp
from . import rainfall as rf
from .config import CORE_DIR, IMD_DIR
from .ingest import TRAIN_YEARS

READING_DAY = 15
# (season, label, the campaign that closes it, the month that campaign falls in)
SEASONS = [
    ("Nov-Jan", "post-monsoon", "Jan", 1),
    ("Jan-May", "dry", "May", 5),
    ("May-Aug", "early monsoon", "Aug", 8),
    ("Aug-Nov", "late monsoon", "Nov", 11),
]
SEASON_OF_CAMPAIGN = {campaign: season for season, _, campaign, _ in SEASONS}
MAX_SPAN_SEASONS = 4           # one year: beyond it, a change says little about any season's rain
MIN_RAIN_FOR_RATIO_MM = 25.0
RAIN_COLUMNS = ["rain_mm", "wet_days", "max_day_mm", "rain_normal_mm"]


def dim_season(first_day: str = f"{gr.PANEL_START}-01-01", last_day: str = f"{gr.PANEL_END}-12-31") -> pd.DataFrame:
    """Every season lying wholly inside [first_day, last_day], numbered in order.

    `season_year` is the year of the reading that closes the season, so Nov-Jan
    2005 runs from November 2004 to January 2005. `water_year` runs June to May
    as in `dim_month`, keeping a monsoon and the dry season after it together.
    """
    first, last = pd.Timestamp(first_day), pd.Timestamp(last_day)
    rows = []
    for year in range(first.year, last.year + 2):
        for season, label, campaign, month in SEASONS:
            rows.append((year, season, label, campaign, pd.Timestamp(year, month, READING_DAY)))
    frame = pd.DataFrame(rows, columns=["season_year", "season", "season_label", "campaign", "end"])
    frame["start"] = frame["end"].shift(1) + pd.Timedelta(days=1)
    frame = frame[(frame["start"] >= first) & (frame["end"] <= last)].reset_index(drop=True)
    frame["n_days"] = (frame["end"] - frame["start"]).dt.days + 1
    frame["season_idx"] = np.arange(len(frame))
    frame["water_year"] = np.where(frame["campaign"].isin(["Aug", "Nov"]),
                                   frame["season_year"], frame["season_year"] - 1)
    frame["is_monsoon_season"] = frame["campaign"].isin(["Aug", "Nov"])
    return frame[["season_idx", "season_year", "season", "season_label", "campaign",
                  "start", "end", "n_days", "water_year", "is_monsoon_season"]]


def season_of_days(dates: pd.DatetimeIndex, seasons: pd.DataFrame) -> np.ndarray:
    """The season_idx each date falls in, or -1 outside every season."""
    ends = seasons["end"].to_numpy(dtype="datetime64[ns]")
    starts = seasons["start"].to_numpy(dtype="datetime64[ns]")
    days = np.asarray(dates, dtype="datetime64[ns]")
    pos = np.searchsorted(ends, days, side="left")
    inside = pos < len(ends)
    inside[inside] &= days[inside] >= starts[pos[inside]]
    return np.where(inside, seasons["season_idx"].to_numpy()[np.minimum(pos, len(ends) - 1)], -1)


def accumulate(dates: pd.DatetimeIndex, daily: np.ndarray, seasons: pd.DataFrame, acc: dict | None = None) -> dict:
    """Add one block of daily rain (days x cells) to per-season running totals.

    Rain and wet days add; the wettest day takes the larger. A NaN day adds
    nothing and is not counted in `n_days_observed`, so a season missing days
    can be recognised afterwards rather than passing as a dry one.
    """
    n_seasons, n_cells = len(seasons), daily.shape[1]
    if acc is None:
        acc = {"rain_mm": np.zeros((n_seasons, n_cells)), "wet_days": np.zeros((n_seasons, n_cells)),
               "max_day_mm": np.full((n_seasons, n_cells), np.nan),
               "n_days_observed": np.zeros((n_seasons, n_cells))}
    idx = season_of_days(dates, seasons)
    for s in np.unique(idx[idx >= 0]):
        block = daily[idx == s]
        acc["rain_mm"][s] += np.nansum(block, axis=0)
        acc["wet_days"][s] += (block > rf.WET_DAY_MM).sum(axis=0)
        acc["max_day_mm"][s] = np.fmax(acc["max_day_mm"][s], np.nanmax(block, axis=0, initial=-np.inf))
        acc["n_days_observed"][s] += (~np.isnan(block)).sum(axis=0)
    acc["max_day_mm"][np.isinf(acc["max_day_mm"])] = np.nan
    return acc


def add_rain_normals(frame: pd.DataFrame, key: str, train_years=TRAIN_YEARS) -> pd.DataFrame:
    """Each place's normal rain for each season over the training years, and the share of it that fell."""
    train = frame[frame["season_year"].isin(list(train_years))]
    normal = train.groupby([key, "season"], observed=True)["rain_mm"].mean().rename("rain_normal_mm").reset_index()
    out = frame.merge(normal, on=[key, "season"], how="left")
    out["rain_pct_normal"] = 100 * out["rain_mm"] / out["rain_normal_mm"].where(out["rain_normal_mm"] > 0)
    return out


def fact_cell_season(seasons: pd.DataFrame, imd_dir: Path = IMD_DIR,
                     suspect: pd.DataFrame | None = None) -> pd.DataFrame:
    """Land cell x season, straight from the daily grid. One pass over the files.

    Cut at the day, not the month: the season boundary sits inside January,
    May, August and November, so no monthly table can produce these totals.
    `suspect` (cell_id, month) lists the cell-months `rain_panel` flagged as
    zeros that are really missing; their days are blanked before summing, so a
    season touching one is incomplete, not dry.
    """
    files = sorted(Path(imd_dir).glob("imd_rf25_*.nc"))
    if not files:
        raise FileNotFoundError(f"no imd_rf25_*.nc files in {imd_dir}")
    rows, cols = np.divmod(np.arange(rf.IMD.size), rf.IMD.nlon)
    ids = gr.cell_id(rows, cols)

    land, acc, position = None, None, None
    for path in files:
        dates, daily = rf.read_year(path)
        this_land = ~np.isnan(daily).all(axis=0)
        if land is None:
            land = this_land
            position = pd.Series(np.arange(int(land.sum())), index=ids[land])
        elif (this_land != land).any():
            raise ValueError(f"{path.name}: land mask differs from the first year's")
        block = daily[:, land]
        if suspect is not None:
            year = suspect[suspect["month"].dt.year == dates[0].year]
            for month, group in year.groupby(year["month"].dt.month):
                block[np.ix_(np.asarray(dates.month == month), position[group["cell_id"]].to_numpy())] = np.nan
        acc = accumulate(dates, block, seasons, acc)

    n_seasons, n_cells = len(seasons), int(land.sum())
    out = pd.DataFrame({
        "cell_id": np.tile(ids[land], n_seasons),
        "season_idx": np.repeat(seasons["season_idx"].to_numpy(), n_cells),
        **{name: values.ravel() for name, values in acc.items()},
    })
    out = out.merge(seasons[["season_idx", "season_year", "season", "n_days"]], on="season_idx")
    short = out["n_days_observed"] < out["n_days"]
    out.loc[short, ["rain_mm", "wet_days", "max_day_mm"]] = np.nan   # incomplete is missing, not dry
    out["complete"] = ~short
    out = add_rain_normals(out, "cell_id")
    return out.drop(columns=["season_year", "season", "n_days"]).sort_values(
        ["cell_id", "season_idx"], ignore_index=True)


def well_rain(bridge: pd.DataFrame, cells: pd.DataFrame, well_uids) -> pd.DataFrame:
    """Season rain at each well: the weighted sum of its stencil cells, as `rain_panel.well_rain`.

    Wet days and the wettest day are counted at the cells first and then
    weighted, which keeps them unbiased (see `rainfall.well_monthly`).
    """
    stencil = bridge[(bridge["weight_land"] > 0) & bridge["well_uid"].isin(set(well_uids))]
    joined = stencil[["well_uid", "cell_id", "weight_land"]].merge(
        cells[["cell_id", "season_idx", *RAIN_COLUMNS]], on="cell_id", how="inner")
    out = rp.weighted_rain(joined, ["well_uid", "season_idx"], "weight_land", RAIN_COLUMNS)
    out["rain_pct_normal"] = 100 * out["rain_mm"] / out["rain_normal_mm"].where(out["rain_normal_mm"] > 0)
    return out


def season_levels(obs: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Valid readings placed on the season they close, with their reference anomaly."""
    levels = pn.reference_normals(obs)
    levels["season"] = levels["campaign"].map(SEASON_OF_CAMPAIGN)
    keys = seasons[["season_idx", "season_year", "season"]].rename(columns={"season_year": "year"})
    return levels.merge(keys, on=["year", "season"], how="inner")[
        ["well_uid", "season_idx", "depth_mbgl", "normal_ref_m", "anomaly_ref_m"]]


def add_changes(panel: pd.DataFrame, max_span: int = MAX_SPAN_SEASONS,
                min_rain: float = MIN_RAIN_FOR_RATIO_MM) -> pd.DataFrame:
    """Change since the last valid reading, rain over the same span, and storage in mm.

    `panel` is well x season, complete in seasons, sorted, with rain_mm, sy and
    depth_mbgl (NaN where no valid reading).
    """
    panel = panel.sort_values(["well_uid", "season_idx"], ignore_index=True)
    rain_to_date = panel.groupby("well_uid")["rain_mm"].cumsum()

    read = panel["depth_mbgl"].notna()
    prev = panel[read].groupby("well_uid")[["season_idx", "depth_mbgl"]].shift(1)
    prev_row = pd.Series(np.nan, index=panel.index)
    prev_row[read] = panel.index[read].to_series().groupby(panel.loc[read, "well_uid"]).shift(1)

    panel["span_seasons"] = np.nan
    panel.loc[read, "span_seasons"] = panel.loc[read, "season_idx"] - prev["season_idx"]
    panel["depth_start_m"] = np.nan
    panel.loc[read, "depth_start_m"] = prev["depth_mbgl"]
    ok = read & panel["span_seasons"].le(max_span)
    panel.loc[~ok, ["span_seasons", "depth_start_m"]] = np.nan

    start = prev_row[ok].astype(int).to_numpy()
    panel["rain_span_mm"] = np.nan
    panel.loc[ok, "rain_span_mm"] = rain_to_date[ok].to_numpy() - rain_to_date.to_numpy()[start]

    panel["delta_h_m"] = panel["depth_mbgl"] - panel["depth_start_m"]
    panel["storage_change_mm"] = -panel["delta_h_m"] * panel["sy"] * 1000
    enough = panel["rain_span_mm"] >= min_rain
    panel["recharge_ratio"] = (panel["storage_change_mm"] / panel["rain_span_mm"]).where(enough)
    return panel


def fact_well_season(wells: pd.DataFrame, obs: pd.DataFrame, bridge: pd.DataFrame, cells: pd.DataFrame,
                     seasons: pd.DataFrame, view: str = "view_modelling") -> pd.DataFrame:
    """Well x season: rain every season, the level in those that close on a valid reading."""
    chosen = wells[wells[view]]
    uids = chosen["well_uid"]
    spine = pd.MultiIndex.from_product([uids, seasons["season_idx"]],
                                       names=["well_uid", "season_idx"]).to_frame(index=False)
    panel = spine.merge(well_rain(bridge, cells, uids), on=["well_uid", "season_idx"], how="left")
    panel = panel.merge(season_levels(obs[obs["well_uid"].isin(set(uids))], seasons),
                        on=["well_uid", "season_idx"], how="left")
    panel = panel.merge(chosen[["well_uid", "sy"]], on="well_uid", how="left")
    panel = add_changes(panel)

    home = gr.cell_weights(chosen)[["well_uid", "cell_id"]]
    panel = panel.merge(home, on="well_uid", how="left")
    panel = panel.merge(chosen[["well_uid", "district", "state", "district_how", "state_agrees", "well_type", "aquifer", "well_depth_m", "sy_source",
                                "lat", "lon", "view_strict"]], on="well_uid", how="left")
    panel = panel.merge(seasons[["season_idx", "season_year", "season", "season_label", "campaign",
                                 "start", "end", "n_days", "water_year", "is_monsoon_season"]],
                        on="season_idx", how="left")
    return panel.sort_values(["well_uid", "season_idx"], ignore_index=True)


def fact_district_season(panel: pd.DataFrame, cells: pd.DataFrame, area: pd.DataFrame,
                         seasons: pd.DataFrame) -> pd.DataFrame:
    """District outline x season: rain from every cell it covers, level from the wells inside it.

    Rain is weighted by the square kilometres of each cell inside the district,
    so it exists for all 720 districts the grid reaches, not only the ones with
    a well. Wells enter by the outline their coordinates fall in. Counts sit
    beside every summary.
    """
    rain = area[["cell_id", "district", "state", "weight_in_district"]].merge(
        cells[["cell_id", "season_idx", *RAIN_COLUMNS]], on="cell_id")
    n_cells = rain.groupby(["district", "state", "season_idx"]).size().rename("n_cells").reset_index()
    rain = rp.weighted_rain(rain, ["district", "state", "season_idx"], "weight_in_district", RAIN_COLUMNS)
    rain = rain.merge(n_cells, on=["district", "state", "season_idx"])
    rain["rain_pct_normal"] = 100 * rain["rain_mm"] / rain["rain_normal_mm"].where(rain["rain_normal_mm"] > 0)
    land_km2 = area.groupby(["district", "state"], as_index=False)["area_km2"].sum().rename(
        columns={"area_km2": "land_km2"})

    read = panel[panel["depth_mbgl"].notna()]
    level = read.groupby(["district", "state", "season_idx"], as_index=False).agg(
        n_wells_read=("well_uid", "nunique"),
        depth_median_m=("depth_mbgl", "median"),
        anomaly_median_m=("anomaly_ref_m", "median"),
        anomaly_mean_m=("anomaly_ref_m", "mean"),
        anomaly_spread_m=("anomaly_ref_m", "std"),
        n_wells_change=("delta_h_m", "count"),
        delta_h_median_m=("delta_h_m", "median"),
        storage_change_median_mm=("storage_change_mm", "median"),
        recharge_ratio_median=("recharge_ratio", "median"),
    )

    out = rain.merge(level, on=["district", "state", "season_idx"], how="left")
    out[["n_wells_read", "n_wells_change"]] = out[["n_wells_read", "n_wells_change"]].fillna(0).astype(int)
    out = out.merge(land_km2, on=["district", "state"], how="left")
    out = out.merge(seasons[["season_idx", "season_year", "season", "season_label", "campaign",
                             "water_year", "is_monsoon_season"]], on="season_idx", how="left")
    return out.sort_values(["state", "district", "season_idx"], ignore_index=True)


def main() -> None:
    seasons = dim_season()
    print(f"dim_season {len(seasons)} seasons, {seasons['start'].min():%Y-%m-%d} to {seasons['end'].max():%Y-%m-%d}")
    print("reading the daily grid:", flush=True)
    monthly = pd.read_parquet(CORE_DIR / "fact_cell_month.parquet", columns=["cell_id", "month", "suspect_zero"])
    cells = fact_cell_season(seasons, suspect=monthly[monthly["suspect_zero"]])

    wells = pd.read_parquet(CORE_DIR / "well_master.parquet")
    obs = pd.read_parquet(CORE_DIR / "well_obs.parquet")
    bridge = pd.read_parquet(CORE_DIR / "bridge_well_cell.parquet")
    area = pd.read_parquet(CORE_DIR / "bridge_cell_district.parquet")

    panel = fact_well_season(wells, obs, bridge, cells, seasons)
    district = fact_district_season(panel, cells, area, seasons)

    seasons.to_parquet(CORE_DIR / "dim_season.parquet", index=False)
    cells.to_parquet(CORE_DIR / "fact_cell_season.parquet", index=False)
    panel.to_parquet(CORE_DIR / "fact_well_season.parquet", index=False)
    district.to_parquet(CORE_DIR / "fact_district_season.parquet", index=False)

    print(f"fact_cell_season     rows {len(cells):>9,} | cells {cells['cell_id'].nunique():,} "
          f"| incomplete cell-seasons (suspect zeros blanked) {int((~cells['complete']).sum()):,}")
    by_season = cells.merge(seasons[["season_idx", "season"]]).groupby("season")["rain_mm"].mean()
    print("                     mean rain per cell: " +
          ", ".join(f"{s} {by_season[s]:.0f} mm" for s, *_ in SEASONS))
    read = panel["depth_mbgl"].notna()
    change = panel["delta_h_m"].notna()
    print(f"fact_well_season     rows {len(panel):>9,} | wells {panel['well_uid'].nunique():,} "
          f"| with a level {int(read.sum()):,} | with a change {int(change.sum()):,} "
          f"({(panel.loc[change, 'span_seasons'] == 1).mean() * 100:.0f}% from the season just before)")
    med = panel[change].groupby("season")[["delta_h_m", "rain_span_mm", "recharge_ratio"]].median()
    for season, *_ in SEASONS:
        row = med.loc[season]
        print(f"                     {season}: median change {row['delta_h_m']:+.2f} m, "
              f"rain {row['rain_span_mm']:.0f} mm, recharge ratio {row['recharge_ratio']:+.3f}")
    print(f"fact_district_season rows {len(district):>9,} | districts {district.groupby(['district', 'state']).ngroups} "
          f"| district-seasons with a well {int((district['n_wells_read'] > 0).sum()):,}")
    print(f"\nwrote dim_season, fact_cell_season, fact_well_season, fact_district_season to {CORE_DIR}")


if __name__ == "__main__":
    main()
