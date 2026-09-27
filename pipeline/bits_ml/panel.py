"""The monthly panels everything downstream reads: well, cell, district.

Three tables, one cadence, one join key each:

* `fact_well_month`      well x month   rain every month, level in four of them
* `fact_cell_month_full` cell x month   the grid's own rain, plus the wells in it
* `fact_district_month`  district x month

The level columns are empty in eight months of twelve and that is correct: CGWB
reads these wells in January, May, August and November. Nothing is interpolated
across the gap. A consumer wanting a dense series takes the campaign rows.

Cells hold up to 49 modelling wells spread over 27 km, so a cell's level is
written three ways — plain mean, median, inverse-distance weighted from the cell
centre — and the count and spread of the wells behind it sit in the same row.
They usually agree; where they do not, the row says so instead of hiding it
behind one choice. All three are computed on the anomaly, how far a well stands
from its own normal, because the raw depth of a 5 m dug well and a 40 m bore
well in one cell cannot be averaged into anything meaningful. The raw mean is
written too, marked for what it is.

`normal_ref_m` and `anomaly_ref_m` are a fixed reference built from the training
years 2000-2014. They exist so the panel is usable and comparable on its own.
A model must not take them as given: every fold recomputes its own normals from
its own training years, which is what `dataset.build(train_years=...)` does.

Run from ml/:  python -m bits_ml.panel
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import grid as gr
from . import rain_panel as rp
from .config import CORE_DIR
from .ingest import CAMPAIGNS, MIN_TRAIN_READINGS, TRAIN_YEARS

LEVEL_COLUMNS = ["depth_mbgl", "normal_ref_m", "anomaly_ref_m"]


def reference_normals(obs: pd.DataFrame, train_years=TRAIN_YEARS,
                      min_readings: int = MIN_TRAIN_READINGS) -> pd.DataFrame:
    """Each well-campaign's normal depth over the training years, and the anomaly against it.

    Valid readings only, and a well-campaign with fewer than `min_readings` of
    them gets no normal at all rather than a shaky one, so its anomaly stays
    empty while its raw depth is still published.
    """
    valid = obs[obs["valid"]]
    train = valid[valid["year"].isin(list(train_years))]
    normal = (
        train.groupby(["well_uid", "campaign"])["depth_mbgl"]
        .agg(normal_ref_m="mean", n_train_readings="size")
        .reset_index()
    )
    normal = normal[normal["n_train_readings"] >= min_readings].drop(columns="n_train_readings")
    out = valid.merge(normal, on=["well_uid", "campaign"], how="left")
    out["anomaly_ref_m"] = out["depth_mbgl"] - out["normal_ref_m"]
    return out


def fact_well_month(wells: pd.DataFrame, obs: pd.DataFrame, bridge: pd.DataFrame,
                    cells: pd.DataFrame, months: pd.DataFrame, view: str = "view_modelling") -> pd.DataFrame:
    """Well x month: rain in every month, water level in the four that have one.

    Built for one view rather than all 32,299 wells, because the full cross with
    300 months is 9.4 million rows of mostly rain that `rain_panel.well_rain`
    can rebuild for any well on demand. The unfiltered reading table
    (`well_obs`) still holds every well.
    """
    chosen = wells[wells[view]]
    uids = chosen["well_uid"]
    spine = pd.MultiIndex.from_product([uids, months["month"]], names=["well_uid", "month"]).to_frame(index=False)

    rain = rp.well_rain(bridge, cells, uids)
    panel = spine.merge(rain, on=["well_uid", "month"], how="left")

    levels = reference_normals(obs)
    levels = levels[levels["well_uid"].isin(set(uids))].copy()
    levels["month"] = levels["date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    panel = panel.merge(levels[["well_uid", "month", "campaign", *LEVEL_COLUMNS]],
                        on=["well_uid", "month"], how="left")

    home = gr.cell_weights(chosen)[["well_uid", "cell_id", "distance_km", "idw_weight", "n_wells_in_cell"]]
    panel = panel.merge(home, on="well_uid", how="left")
    panel = panel.merge(chosen[["well_uid", "district", "state", "well_type", "aquifer",
                                "well_depth_m", "sy", "sy_source", "lat", "lon",
                                "state_agrees", "view_strict"]], on="well_uid", how="left")
    panel = panel.merge(months[["month", "year", "month_no", "water_year", "season",
                                "is_monsoon", "is_campaign_month"]], on="month", how="left")
    return panel.sort_values(["well_uid", "month"], ignore_index=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna()
    total = weights[ok].sum()
    return float((values[ok] * weights[ok]).sum() / total) if total > 0 else np.nan


def aggregate_cells(panel: pd.DataFrame) -> pd.DataFrame:
    """The wells of each cell summarised into one series per cell and month.

    Three summaries of the same thing, plus what they were built from. `spread`
    is the standard deviation of the anomalies in the cell: a large value means
    the wells of that cell disagree, which matters more than any one summary.
    """
    columns = ["cell_id", "month", "n_wells_read", "anomaly_mean_m", "anomaly_median_m",
               "anomaly_spread_m", "depth_mean_m", "depth_median_m", "anomaly_idw_m"]
    read = panel[panel["depth_mbgl"].notna()]
    if read.empty:
        return pd.DataFrame(columns=columns)
    grouped = read.groupby(["cell_id", "month"], observed=True)
    out = grouped.agg(
        n_wells_read=("well_uid", "nunique"),
        anomaly_mean_m=("anomaly_ref_m", "mean"),
        anomaly_median_m=("anomaly_ref_m", "median"),
        anomaly_spread_m=("anomaly_ref_m", "std"),
        depth_mean_m=("depth_mbgl", "mean"),           # wells of different depths: context only
        depth_median_m=("depth_mbgl", "median"),
    ).reset_index()

    idw = grouped.apply(lambda g: _weighted_mean(g["anomaly_ref_m"], g["idw_weight"]),
                        include_groups=False).rename("anomaly_idw_m").reset_index()
    return out.merge(idw, on=["cell_id", "month"], how="left")


def fact_cell_month_full(cells: pd.DataFrame, dim_cells: pd.DataFrame, panel: pd.DataFrame,
                         months: pd.DataFrame) -> pd.DataFrame:
    """The grid's own rain, with the wells standing in each cell beside it."""
    out = cells.merge(aggregate_cells(panel), on=["cell_id", "month"], how="left")
    out["n_wells_read"] = out["n_wells_read"].fillna(0).astype(int)
    out = out.merge(dim_cells[["cell_id", "cell_lat", "cell_lon", "district", "state",
                               "n_wells", "n_wells_modelling"]], on="cell_id", how="left")
    out = out.merge(months[["month", "year", "month_no", "water_year", "season",
                            "is_monsoon", "is_campaign_month"]], on="month", how="left")
    return out.sort_values(["cell_id", "month"], ignore_index=True)


def fact_district_month(panel: pd.DataFrame, cell_month: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """District x month: rain from the cells, level from the wells.

    Districts are the outlines the coordinates fall in. Rain is the mean of the
    cells a district overlaps, weighted by the square kilometres of each inside
    it (`bridge_cell_district`), so every district the grid reaches has rain
    whether or not a well stands there. Levels are averaged over the wells
    themselves, with the count kept: a district reading built on two wells is
    not the same claim as one built on forty, and the column says which.
    """
    read = panel[panel["depth_mbgl"].notna()]
    level = read.groupby(["district", "state", "month"], observed=True).agg(
        n_wells_read=("well_uid", "nunique"),
        anomaly_mean_m=("anomaly_ref_m", "mean"),
        anomaly_median_m=("anomaly_ref_m", "median"),
        anomaly_spread_m=("anomaly_ref_m", "std"),
        depth_median_m=("depth_mbgl", "median"),
    ).reset_index()

    rain = area[["cell_id", "district", "state", "weight_in_district"]].merge(
        cell_month[["cell_id", "month", *rp.RAIN_VALUES]], on="cell_id", how="inner")
    n_cells = rain.groupby(["district", "state", "month"]).size().rename("n_cells").reset_index()
    rain = rp.weighted_rain(rain, ["district", "state", "month"], "weight_in_district", rp.RAIN_VALUES)
    rain = rain.merge(n_cells, on=["district", "state", "month"])

    out = rain.merge(level, on=["district", "state", "month"], how="left")
    out["n_wells_read"] = out["n_wells_read"].fillna(0).astype(int)
    return out.sort_values(["state", "district", "month"], ignore_index=True)


def main() -> None:
    wells = pd.read_parquet(CORE_DIR / "well_master.parquet")
    obs = pd.read_parquet(CORE_DIR / "well_obs.parquet")
    bridge = pd.read_parquet(CORE_DIR / "bridge_well_cell.parquet")
    cells = pd.read_parquet(CORE_DIR / "fact_cell_month.parquet")
    dim_cells = pd.read_parquet(CORE_DIR / "dim_cell.parquet")
    months = pd.read_parquet(CORE_DIR / "dim_month.parquet")
    area = pd.read_parquet(CORE_DIR / "bridge_cell_district.parquet")

    panel = fact_well_month(wells, obs, bridge, cells, months)
    cell_full = fact_cell_month_full(cells, dim_cells, panel, months)
    district = fact_district_month(panel, cells, area)

    panel.to_parquet(CORE_DIR / "fact_well_month.parquet", index=False)
    cell_full.to_parquet(CORE_DIR / "fact_cell_month_full.parquet", index=False)
    district.to_parquet(CORE_DIR / "fact_district_month.parquet", index=False)

    levels = panel["depth_mbgl"].notna()
    print(f"fact_well_month      rows {len(panel):>9,} | wells {panel['well_uid'].nunique():,} "
          f"| with a level {int(levels.sum()):,} ({levels.mean() * 100:.0f}% of rows, "
          f"{levels.sum() / panel.loc[panel['is_campaign_month'], 'month'].size * 100:.0f}% of campaign rows)")
    print(f"                     rain missing {int(panel['rain_mm'].isna().sum()):,} "
          f"| anomaly missing where a level exists {int(panel.loc[levels, 'anomaly_ref_m'].isna().sum()):,}")
    read = cell_full[cell_full["n_wells_read"] > 0]
    print(f"fact_cell_month_full rows {len(cell_full):>9,} | cells {cell_full['cell_id'].nunique():,} "
          f"| rows carrying wells {len(read):,}")
    print(f"                     wells behind a cell reading: median {read['n_wells_read'].median():.0f}, "
          f"max {read['n_wells_read'].max()}, single-well {int((read['n_wells_read'] == 1).mean() * 100)}%")
    agreement = read[["anomaly_mean_m", "anomaly_idw_m", "anomaly_median_m"]].dropna()
    print(f"                     mean vs idw differ by median {(agreement['anomaly_mean_m'] - agreement['anomaly_idw_m']).abs().median():.3f} m, "
          f"mean vs median {(agreement['anomaly_mean_m'] - agreement['anomaly_median_m']).abs().median():.3f} m")
    print(f"fact_district_month  rows {len(district):>9,} | districts {district.groupby(['district', 'state']).ngroups:,} "
          f"| rows carrying wells {int((district['n_wells_read'] > 0).sum()):,}")
    print(f"\nwrote fact_well_month, fact_cell_month_full, fact_district_month to {CORE_DIR}")


if __name__ == "__main__":
    main()
