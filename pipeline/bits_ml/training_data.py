"""Model-ready data for the goal: observed rain in, groundwater change out.

Two architectures are to be compared on the same rows, the same target and the
same splits:

* **A, many inputs to one output** (ridge baseline, LightGBM, MLP): one row per
  well and season, with rain summarised into features.
* **B, recurrent** (LSTM/GRU): the same rows, with the rain of the 52 weeks up to
  the reading as a sequence, plus the same fixed well properties.

The target is `delta_h_m`, how far the water level moved since the previous
reading (+ = fell). Rain is only ever an input, and it is the rain that was
observed: next season's rain cannot be forecast from past rain (a district's
monsoon correlates +0.11 with the next year's; fed back into itself, a 30-day
rain model loses its skill within a week), so "future" groundwater is asked of
the model with scenario rain, not forecast rain.

Never inputs: any past water level (`depth_start_m`, `depth_mbgl`), the year,
coordinates, district or state. Each of them carries the pumping history the
project is trying to measure. They stay in the table as identifiers and for
grouping, and `feature_spec.csv` marks their role; `check_no_leakage` fails if
one is listed as a feature.

The wells are `view_training`: every well with a usable normal in at least three
of the four campaigns. Requiring all four, as `view_modelling` does, removes
every well in Kerala and Assam and all but four in West Bengal and Odisha,
because their raw record has no May reading; three of four brings them back,
and their Jan -> Aug change simply spans two seasons (`span_seasons` = 2).

Splits: train 2000-2014, validation 2015-2017, test 2018-2022, plus five folds of
whole districts for testing on places the model never saw.

Writes data_cleaning/data/training/:
    tabular.parquet        one row per well x season with a target
    sequence_cells.npz     weekly rain per land cell for the 52 weeks up to each season end
    stencil.parquet        which cells, with which weights, make each training well's rain
    feature_spec.csv       every column, its role and meaning

Run from ml/ after the core pipeline and clean_gaps.ipynb:  python -m bits_ml.training_data
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import grid as gr
from . import rainfall as rf
from . import seasons as ss
from .config import CORE_DIR, IMD_DIR
from .ingest import MIN_TRAIN_READINGS

OUT_DIR = CORE_DIR / "training"
TRAIN_YEARS = range(2000, 2015)
VALID_YEARS = range(2015, 2018)
TEST_YEARS = range(2018, 2023)
N_DISTRICT_FOLDS = 5
MIN_CAMPAIGNS = 3
HISTORY_SEASONS = (4, 8, 12)          # one, two and three years of rain before the reading
LAGS = (1, 2, 3)
SEQ_WEEKS = 52

TARGET = "delta_h_m"
IDENTIFIERS = ["well_uid", "season_idx", "season_year", "district", "state", "cell_id", "split", "district_fold"]
FORBIDDEN = {"depth_start_m", "depth_mbgl", "normal_ref_m", "season_year", "year", "water_year",
             "lat", "lon", "district", "state", "cell_id", "well_uid"}
CATEGORICAL = ["season", "aquifer", "well_type", "rock_class"]
STATIC = ["sy", "well_depth_m", "rain_normal_annual_mm"]


def training_view(wells: pd.DataFrame, min_campaigns: int = MIN_CAMPAIGNS,
                  min_readings: int = MIN_TRAIN_READINGS) -> pd.Series:
    """Wells with a usable normal (enough training-year readings) in at least `min_campaigns` campaigns."""
    ready = wells[["n_train_jan", "n_train_may", "n_train_aug", "n_train_nov"]] >= min_readings
    return ready.sum(axis=1) >= min_campaigns


def add_rain_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Rain before the reading, from the well's own season series: lags and 1-3 year totals.

    `panel` is well x season, complete and sorted, with rain_mm and
    rain_normal_mm. A window reaching back before the record is missing, not
    short. Every window ends at the season the reading closes, never after it.
    """
    panel = panel.sort_values(["well_uid", "season_idx"], ignore_index=True)
    by_well = panel.groupby("well_uid")
    for lag in LAGS:
        panel[f"rain_lag{lag}_mm"] = by_well["rain_mm"].shift(lag)
    for n in HISTORY_SEASONS:
        total = by_well["rain_mm"].rolling(n, min_periods=n).sum().reset_index(level=0, drop=True)
        normal = by_well["rain_normal_mm"].rolling(n, min_periods=n).sum().reset_index(level=0, drop=True)
        panel[f"rain_{n}s_mm"] = total
        panel[f"rain_{n}s_pct_normal"] = 100 * total / normal.where(normal > 0)
    panel["rain_normal_annual_mm"] = by_well["rain_normal_mm"].transform(
        lambda s: s.groupby(np.arange(len(s)) % 4).mean().sum())
    return panel


def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    year = frame["season_year"]
    frame["split"] = np.select([year.isin(TRAIN_YEARS), year.isin(VALID_YEARS), year.isin(TEST_YEARS)],
                               ["train", "valid", "test"], default="outside")
    key = (frame["state"].fillna("") + "|" + frame["district"].fillna("")).map(
        lambda k: zlib.crc32(k.encode()) % N_DISTRICT_FOLDS)
    frame["district_fold"] = key.astype(int)
    return frame


def feature_columns() -> list[str]:
    rain = ["rain_mm", "rain_span_mm", "span_seasons", "rain_pct_normal", "wet_days", "max_day_mm",
            "rain_normal_mm", "rain_cover", "n_days",
            *[f"rain_lag{k}_mm" for k in LAGS],
            *[c for n in HISTORY_SEASONS for c in (f"rain_{n}s_mm", f"rain_{n}s_pct_normal")]]
    return [*rain, *STATIC, *CATEGORICAL]


def check_no_leakage(features: list[str]) -> None:
    bad = sorted(set(features) & FORBIDDEN)
    if bad:
        raise ValueError(f"forbidden inputs listed as features: {bad}")


def build_tabular(wells: pd.DataFrame, obs: pd.DataFrame, bridge: pd.DataFrame, cells: pd.DataFrame,
                  seasons: pd.DataFrame) -> pd.DataFrame:
    wells = wells.assign(view_training=training_view(wells))
    panel = ss.fact_well_season(wells, obs, bridge, cells, seasons, view="view_training")
    panel = panel.merge(wells[["well_uid", "rock_class"]], on="well_uid", how="left")
    panel = add_rain_history(panel)
    panel = assign_splits(panel)
    rows = panel[panel[TARGET].notna()].copy()
    rows["view_modelling"] = rows["well_uid"].isin(set(wells.loc[wells["view_modelling"], "well_uid"]))
    features = feature_columns()
    check_no_leakage(features)
    keep = [*IDENTIFIERS, TARGET, "storage_change_mm", "anomaly_ref_m", "depth_start_m",
            *[f for f in features if f not in IDENTIFIERS], "view_modelling", "view_strict", "state_agrees"]
    return rows[list(dict.fromkeys(keep))].reset_index(drop=True)


def week_blocks(dates: pd.DatetimeIndex, daily: np.ndarray, ends: pd.DatetimeIndex,
                weeks: int = SEQ_WEEKS) -> np.ndarray:
    """Rain in each of the `weeks` 7-day blocks ending on each date in `ends`.

    Returns (len(ends), weeks, cells), oldest week first. A block reaching
    before the first day, or holding a missing day, is NaN.
    """
    day = pd.Series(np.arange(len(dates)), index=dates)
    out = np.full((len(ends), weeks, daily.shape[1]), np.nan, dtype=np.float32)
    span = 7 * weeks
    for i, end in enumerate(ends):
        if end not in day.index or day[end] + 1 < span:
            continue
        window = daily[day[end] + 1 - span: day[end] + 1].reshape(weeks, 7, -1)
        total = window.sum(axis=1)                      # NaN if any day in the week is missing
        out[i] = total
    return out


def build_sequences(seasons: pd.DataFrame, imd_dir: Path = IMD_DIR,
                    suspect: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weekly rain per land cell for the 52 weeks up to every season end, from the daily grid."""
    rows, cols = np.divmod(np.arange(rf.IMD.size), rf.IMD.nlon)
    ids = gr.cell_id(rows, cols)
    blocks, all_dates, land = [], [], None
    for path in sorted(Path(imd_dir).glob("imd_rf25_*.nc")):
        dates, daily = rf.read_year(path)
        if land is None:
            land = ~np.isnan(daily).all(axis=0)
            position = pd.Series(np.arange(int(land.sum())), index=ids[land])
        block = daily[:, land].astype(np.float32)
        if suspect is not None:
            year = suspect[suspect["month"].dt.year == dates[0].year]
            for month, group in year.groupby(year["month"].dt.month):
                block[np.ix_(np.asarray(dates.month == month), position[group["cell_id"]].to_numpy())] = np.nan
        blocks.append(block)
        all_dates.append(dates)
    daily = np.vstack(blocks)
    dates = pd.DatetimeIndex(np.concatenate([d.values for d in all_dates]))
    return week_blocks(dates, daily, pd.DatetimeIndex(seasons["end"])), ids[land]


def well_sequences(rows: pd.DataFrame, cube: np.ndarray, cell_ids: np.ndarray,
                   stencil: pd.DataFrame) -> np.ndarray:
    """(len(rows), 52) weekly rain at each row's well, for the season it closes.

    The weighted mean of the well's stencil cells, over the cells with data, as
    everywhere else in the pipeline.
    """
    col = pd.Series(np.arange(len(cell_ids)), index=cell_ids)
    st = stencil[stencil["well_uid"].isin(set(rows["well_uid"]))]
    out = np.full((len(rows), cube.shape[1]), np.nan, dtype=np.float32)
    groups = st.groupby("well_uid")
    for i, (uid, idx) in enumerate(zip(rows["well_uid"].to_numpy(), rows["season_idx"].to_numpy())):
        g = groups.get_group(uid)
        vals = cube[idx][:, col[g["cell_id"]].to_numpy()]          # (weeks, cells)
        w = np.where(np.isnan(vals), 0.0, g["weight_land"].to_numpy()[None, :])
        total = w.sum(axis=1)
        out[i] = np.where(total > 0, np.nansum(vals * w, axis=1) / np.where(total > 0, total, 1), np.nan)
    return out


SPEC = {
    "well_uid": ("id", "the well"), "season_idx": ("id", "the season, 0 = Jan-May 1998"),
    "season_year": ("id", "year of the reading that closes the season; never an input"),
    "district": ("group", "district outline the well stands in; for district folds and reporting, never an input"),
    "state": ("group", "state of that outline; never an input"),
    "cell_id": ("group", "the grid cell the well stands in; never an input"),
    "split": ("split", "train 2000-2014, valid 2015-2017, test 2018-2022"),
    "district_fold": ("split", "0-4: whole districts held out together, for spatial cross-validation"),
    "delta_h_m": ("target", "depth now minus depth at the previous reading, m; + = water fell"),
    "storage_change_mm": ("target_alt", "-delta_h x specific yield x 1000: water gained, mm"),
    "anomaly_ref_m": ("target_alt", "depth minus the well-campaign normal of 2000-2014, m; normals from training years only"),
    "depth_start_m": ("excluded", "the previous water level: kept to rebuild levels from changes, never an input"),
    "rain_mm": ("feature", "rain at the well over this season"),
    "rain_span_mm": ("feature", "rain since the previous reading; equals rain_mm when span_seasons is 1"),
    "span_seasons": ("feature", "seasons since the previous reading (1 = the one before)"),
    "rain_pct_normal": ("feature", "this season's rain as % of its 2000-2014 normal"),
    "wet_days": ("feature", "days above 2.5 mm this season"),
    "max_day_mm": ("feature", "wettest day this season"),
    "rain_normal_mm": ("feature", "this season's normal rain at the well"),
    "rain_cover": ("feature", "share of the well's stencil with rain data (below 1 where IMD zeros were dropped)"),
    "n_days": ("feature", "length of the season in days"),
    "rain_lag1_mm": ("feature", "rain in the season before"), "rain_lag2_mm": ("feature", "two seasons before"),
    "rain_lag3_mm": ("feature", "three seasons before"),
    "rain_4s_mm": ("feature", "rain over the last 4 seasons (one year) up to the reading"),
    "rain_4s_pct_normal": ("feature", "the same as % of normal"),
    "rain_8s_mm": ("feature", "rain over the last 8 seasons (two years)"),
    "rain_8s_pct_normal": ("feature", "the same as % of normal"),
    "rain_12s_mm": ("feature", "rain over the last 12 seasons (three years)"),
    "rain_12s_pct_normal": ("feature", "the same as % of normal"),
    "sy": ("feature", "specific yield from the hydrogeological map"),
    "well_depth_m": ("feature", "drilled depth of the well"),
    "rain_normal_annual_mm": ("feature", "the well's normal annual rain: dry and wet places respond differently"),
    "season": ("feature", "Nov-Jan, Jan-May, May-Aug or Aug-Nov (categorical)"),
    "aquifer": ("feature", "Unconfined, Semi-confined, Confined or Unknown (categorical)"),
    "well_type": ("feature", "dug, bore, tube well, piezometer (categorical)"),
    "rock_class": ("feature", "rock type behind the specific-yield class (categorical)"),
    "view_modelling": ("filter", "also in the stricter four-campaign set"),
    "view_strict": ("filter", "one of the 2,759 published wells"),
    "state_agrees": ("filter", "CGWB's state label agrees with the coordinates"),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wells = pd.read_parquet(CORE_DIR / "well_master.parquet")
    obs = pd.read_parquet(CORE_DIR / "well_obs.parquet")
    bridge = pd.read_parquet(CORE_DIR / "bridge_well_cell.parquet")
    cells = pd.read_parquet(CORE_DIR / "fact_cell_season.parquet")
    seasons = pd.read_parquet(CORE_DIR / "dim_season.parquet")
    monthly = pd.read_parquet(CORE_DIR / "fact_cell_month.parquet", columns=["cell_id", "month", "suspect_zero"])

    table = build_tabular(wells, obs, bridge, cells, seasons)
    table.to_parquet(OUT_DIR / "tabular.parquet", index=False)
    spec = pd.DataFrame([(c, *SPEC.get(c, ("", ""))) for c in table.columns], columns=["column", "role", "meaning"])
    missing = spec[spec["role"] == ""]
    if len(missing):
        raise ValueError(f"columns with no role in feature_spec: {missing['column'].tolist()}")
    spec["null_pct"] = [round(float(table[c].isna().mean()) * 100, 1) for c in spec["column"]]
    spec.to_csv(OUT_DIR / "feature_spec.csv", index=False)

    print("reading the daily grid for the weekly sequences:", flush=True)
    cube, cell_ids = build_sequences(seasons, suspect=monthly[monthly["suspect_zero"]])
    np.savez_compressed(OUT_DIR / "sequence_cells.npz", weeks=cube, cell_ids=cell_ids,
                        season_idx=seasons["season_idx"].to_numpy(), season_end=seasons["end"].astype(str).to_numpy())
    stencil = bridge[(bridge["weight_land"] > 0) & bridge["well_uid"].isin(set(table["well_uid"]))][
        ["well_uid", "cell_id", "weight_land"]]
    stencil.to_parquet(OUT_DIR / "stencil.parquet", index=False)

    wells_t = table.drop_duplicates("well_uid")
    print(f"\nview_training wells {table['well_uid'].nunique():,} (view_modelling {int(wells_t['view_modelling'].sum()):,}) "
          f"| states {wells_t['state'].nunique()} | districts {wells_t.groupby(['district', 'state']).ngroups}")
    print("wells in the four states the stricter set leaves out: " + ", ".join(
        f"{s} {int((wells_t['state'] == s).sum())}" for s in ("Kerala", "West Bengal", "Odisha", "Assam")))
    print(f"rows with a target {len(table):,} | " + " | ".join(
        f"{k} {v:,}" for k, v in table["split"].value_counts().reindex(["train", "valid", "test", "outside"]).fillna(0).astype(int).items()))
    print(f"span 1 {(table['span_seasons'] == 1).mean():.0%} | span 2 {(table['span_seasons'] == 2).mean():.0%} "
          f"| longer {(table['span_seasons'] > 2).mean():.0%}")
    print(f"features {len(feature_columns())}: {', '.join(feature_columns())}")
    print(f"feature nulls over 1%: " + ", ".join(f"{r.column} {r.null_pct}%" for r in spec.itertuples()
                                               if r.role == "feature" and r.null_pct > 1))
    print(f"sequence cube {cube.shape} (seasons x weeks x cells), {cube.nbytes / 1e6:.0f} MB in memory, "
          f"{(OUT_DIR / 'sequence_cells.npz').stat().st_size / 1e6:.0f} MB on disk")
    sample = table[table["split"] == "train"].sample(3, random_state=0)
    seq = well_sequences(sample, cube, cell_ids, stencil)
    print("check: 52-week sequence total vs rain over the last 4 seasons, three random rows:")
    for (_, r), s in zip(sample.iterrows(), seq):
        print(f"   {r['well_uid']:<18} season {r['season_idx']:>3}: 52 weeks {np.nansum(s):7.0f} mm, "
              f"4 seasons {r['rain_4s_mm']:7.0f} mm")
    print(f"\nwrote tabular.parquet, sequence_cells.npz, stencil.parquet, feature_spec.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
