"""The data dictionary for data_cleaning/data, generated from the tables themselves.

Meanings are written here by hand; the column list, types and null counts are
read off the parquet files. A column that exists but has no meaning written for
it is reported as undocumented and makes this script exit non-zero, so the
dictionary cannot quietly fall behind the tables.

Run from ml/:  python -m scripts.core_dictionary
"""

from __future__ import annotations

import sys

import pandas as pd

from bits_ml.config import CORE_DIR

TABLES = {
    "well_master": "one row per well, all 32,299, nothing excluded",
    "well_obs": "one row per water-level reading, flagged but never dropped",
    "dim_cell": "one row per IMD grid cell, land or not",
    "dim_month": "the monthly spine, 1998-01 to 2022-12",
    "bridge_well_cell": "well to the cells its rain is read from, with weights",
    "bridge_cell_district": "land cell x district outline: km2 of the cell inside the district",
    "well_cell_weights": "each well's inverse-distance weight within its own cell",
    "fact_cell_month": "rain per cell per month, the grid baseline",
    "fact_well_month": "well x month: rain every month, level in four of them",
    "fact_cell_month_full": "fact_cell_month plus the wells standing in each cell",
    "fact_district_month": "district x month, rain from cells, level from wells",
    "dim_season": "the four seasons cut at the readings, 1998 Jan-May to 2022 Aug-Nov",
    "fact_cell_season": "land cell x season: rain from the daily grid",
    "fact_well_season": "well x season: rain at the well, the level it closes on, the change since the last",
    "fact_district_season": "district outline x season: area-weighted rain, wells summarised",
    "fact_cell_season_full": "fact_cell_season plus the wells standing in each cell (built by data_cleaning/clean_gaps.ipynb)",
}

MEANING = {
    # identity and position
    "well_uid": "well key: coordinates to four decimals, suffixed when wells share a coordinate",
    "site_key": "coordinates to four decimals; co-located wells share one, rain is sampled on it",
    "station_code": "CGWB station code as published, already collapsed by scientific notation; never an identity",
    "station": "station name as published",
    "station_type": "what the station measures",
    "agency": "the agency that reads the well",
    "acquisition": "manual or telemetric reading",
    "state": "state of the district outline the coordinates fall in",
    "district": "district outline (post-2020 boundaries) the coordinates fall in",
    "tehsil": "tehsil as labelled in the source file",
    "block": "block as labelled in the source file",
    "village": "village as labelled in the source file",
    "lat": "latitude, degrees north",
    "lon": "longitude, degrees east",
    "data_from": "first date the source file claims data for",
    "data_latest": "last date the source file claims data for",
    # construction
    "well_type": "dug well, bore well, tube well, piezometer; Unknown where the file is blank",
    "aquifer": "Unconfined, Semi-confined, Confined, or Unknown; blank dug wells filled as Unconfined (see aquifer_source)",
    "well_type_cgwb": "well type exactly as in the source file",
    "aquifer_cgwb": "aquifer type exactly as in the source file",
    "aquifer_source": "cgwb (as recorded), inferred_dug_well (blank dug well, 99.6% of recorded ones are unconfined) or unknown",
    "rock_class": "rock type behind the specific-yield class on the hydrogeological map",
    "sy_ingest": "specific yield as the ingest set it, before the hydrogeological map was applied",
    "sy_source_ingest": "where the ingest's value came from: published or a state/aquifer median",
    "well_depth_m": "drilled depth of the well, metres",
    "sy": "specific yield, the share of the rock that drains",
    "sy_source": "published, hydro_map (sampled at the well), hydro_map_nearest (nearest map cell within 2), or an imputed median",
    # coordinate quality
    "coord_half_width_deg": "how far the true position may lie from the reported one, degrees",
    "coord_rounded": "the coordinate was reported on a coarse lattice",
    "site_shared": "another well reports the same coordinate",
    "far_from_district": "the well stands far from the other wells of its labelled (cgwb_) district",
    "depth_missing": "drilled depth absent or not positive",
    # readings
    "depth_mbgl": "water level, metres below ground; deeper is a larger number",
    "campaign": "which of the four yearly readings this is: Jan, May, Aug, Nov",
    "year": "calendar year",
    "month": "first day of the month this row belongs to",
    "month_no": "month of the year, 1 to 12",
    "date": "reading date, the 15th of the campaign month",
    "water_year": "June to May, so a monsoon and the dry season it feeds stay together",
    "season": "winter, pre_monsoon, monsoon or post_monsoon",
    "is_monsoon": "the month falls in June to September",
    "is_campaign_month": "a water level can exist in this month",
    # flags
    "flag_negative": "water reported above ground level",
    "flag_zero": "reading is exactly zero",
    "flag_zero_suspect": "a zero in a well whose zeros exceed a fifth of its record: missing written as zero",
    "flag_below_bottom": "water reported deeper than the drilled depth of the well",
    "flag_outlier": "3 sigma from the well's own seasonal pattern, not from its raw level",
    "flag_repeat_run": "one of three or more identical consecutive readings",
    "valid": "no invalidating flag; a plain zero alone does not invalidate a reading",
    # per-well counts and views
    "n_obs": "readings present in the source file",
    "n_valid": "readings that survive the flags",
    "n_valid_2000_2022": "valid readings inside the years the rain grid covers",
    "zero_share": "share of the well's readings that are exactly zero",
    "first_year": "first year with a valid reading",
    "last_year": "last year with a valid reading",
    "n_train_jan": "valid January readings in the training years 2000-2014",
    "n_train_may": "valid May readings in the training years",
    "n_train_aug": "valid August readings in the training years",
    "n_train_nov": "valid November readings in the training years",
    "n_campaigns_covered": "how many of the four campaigns have enough training readings for a normal",
    "view_strict": "the well is one of the 2,759 the source paper published",
    "view_modelling": "all four campaigns have enough training readings: the set models use",
    # normals
    "normal_ref_m": "the well-campaign's mean depth over 2000-2014; a reference, recomputed per fold by models",
    "anomaly_ref_m": "depth minus that reference normal; positive means deeper than normal for that well",
    # grid
    "cell_id": "IMD cell key: the centre coordinates, e.g. 14.25_78.25",
    "cell_row": "row of the cell in the IMD grid",
    "cell_col": "column of the cell in the IMD grid",
    "cell_lat": "latitude of the cell centre",
    "cell_lon": "longitude of the cell centre",
    "is_land": "IMD reports rainfall for this cell",
    "n_wells": "wells standing in this cell",
    "n_wells_modelling": "wells of the modelling view standing in this cell",
    "weight": "share of the well's rain read from this cell, before the land mask",
    "weight_land": "the same share renormalised over land cells; these sum to one per well",
    "is_home": "this is the cell the well physically stands in",
    "distance_km": "distance from the well to the cell centre",
    "idw_weight": "the well's inverse-distance weight within its own cell, summing to one per cell",
    "n_wells_in_cell": "wells sharing this well's cell",
    # rain
    "rain_mm": "rainfall over the month",
    "wet_days": "days above 2.5 mm, counted at the grid nodes before interpolation, so it can be fractional",
    "max_day_mm": "wettest single day of the month",
    # cell and district aggregates
    "n_wells_read": "wells actually read in this cell or district this month",
    "anomaly_mean_m": "mean anomaly of those wells",
    "anomaly_median_m": "median anomaly of those wells",
    "anomaly_idw_m": "their anomaly weighted by inverse distance from the cell centre",
    "anomaly_spread_m": "standard deviation of those anomalies: how far the wells disagree",
    "depth_mean_m": "mean raw depth; wells of different drilled depths, so context only",
    "depth_median_m": "median raw depth; context only, for the same reason",
    "n_cells": "grid cells contributing rain to this district",
    "suspect_zero": "IMD wrote 0 mm where the value is missing (a whole year of zeros, or a zero monsoon month where the median is over 100 mm); rain set missing",
    "rain_cover": "share of the stencil or district weight whose cells had rain data; below 1 where suspect zeros were left out",
    # district outlines
    "frac_of_cell": "share of the cell's 0.25 degree box inside this district",
    "area_km2": "square kilometres of the cell inside this district",
    "weight_in_district": "that area as a share of the district's gridded land; sums to one per district",
    "cgwb_district": "district as labelled in the source file",
    "cgwb_state": "state as labelled in the source file",
    "district_distance_km": "distance from the well to its district outline; 0 inside it",
    "n_districts": "district outlines the cell overlaps",
    "district_how": "how the well was placed: inside an outline, nearest one within 5 km, or none",
    "name_agrees": "the source file's district name matches the outline's, allowing for spelling",
    "state_agrees": "the source file's state matches the outline's (Andhra Pradesh and Telangana count as one); where it does not, the water level follows the coordinates, so the label is taken to be wrong",
    "land_km2": "gridded land area of the district",
    # seasons
    "season_idx": "season number in order, 0 = Jan-May 1998",
    "season_year": "year of the reading that closes the season",
    "season_label": "post-monsoon, dry, early monsoon or late monsoon",
    "start": "first day of the season, the day after the previous reading",
    "end": "last day of the season, the 15th of the reading month",
    "n_days": "days in the season",
    "is_monsoon_season": "May-Aug or Aug-Nov",
    "n_days_observed": "days the grid reported rain for this cell in the season",
    "complete": "every day of the season observed; if not, the rain columns are empty",
    "rain_normal_mm": "the season's mean rain over the training years 2000-2014",
    "rain_pct_normal": "rain as a percentage of that normal",
    "depth_start_m": "depth at the last valid reading before this one, within a year",
    "span_seasons": "seasons back to that reading; 1 means the reading just before",
    "rain_span_mm": "rain over all seasons since that reading",
    "delta_h_m": "depth now minus depth then; positive means the water fell",
    "storage_change_mm": "minus delta_h x specific yield x 1000: water gained by the aquifer, mm",
    "recharge_ratio": "storage change over span rain; empty when span rain is under 25 mm",
    "n_wells_change": "wells with a change this season",
    "delta_h_median_m": "median change of those wells",
    "storage_change_median_mm": "median storage change of those wells",
    "recharge_ratio_median": "median recharge ratio of those wells",
}

# The same column name meaning something else in one table.
MEANING_IN = {
    ("bridge_cell_district", "district"): "district outline",
    ("bridge_cell_district", "state"): "state of that outline",
    ("dim_cell", "district"): "the district outline covering most of the cell",
    ("dim_cell", "state"): "state of that outline",
    ("dim_season", "season"): "the window between readings: Nov-Jan, Jan-May, May-Aug or Aug-Nov",
    ("fact_well_season", "season"): "the window between readings: Nov-Jan, Jan-May, May-Aug or Aug-Nov",
    ("fact_district_season", "season"): "the window between readings: Nov-Jan, Jan-May, May-Aug or Aug-Nov",
    ("fact_cell_season", "rain_mm"): "rainfall over the season; empty if a day is missing",
    ("fact_well_season", "rain_mm"): "rainfall over the season at the well",
    ("fact_district_season", "rain_mm"): "rainfall over the season, area-weighted over the district",
    ("fact_cell_season", "max_day_mm"): "wettest single day of the season",
    ("fact_well_season", "max_day_mm"): "wettest single day of the season",
    ("fact_district_season", "max_day_mm"): "wettest single day of the season, area-weighted",
    ("fact_well_season", "depth_mbgl"): "water level at the reading that closes the season; valid readings only",
    ("fact_district_season", "n_wells_read"): "wells read this season",
    ("fact_district_season", "n_cells"): "grid cells overlapping the district",
    ("fact_district_month", "n_cells"): "grid cells overlapping the district",
}


def describe(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        values = frame[column]
        rows.append({
            "table": name,
            "column": column,
            "dtype": str(values.dtype),
            "null_pct": round(float(values.isna().mean()) * 100, 1),
            "meaning": MEANING_IN.get((name, column), MEANING.get(column, "")),
        })
    return pd.DataFrame(rows)


def main() -> int:
    frames = []
    for name in TABLES:
        path = CORE_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"missing: {path.name}")
            continue
        frames.append(describe(name, pd.read_parquet(path)))
    dictionary = pd.concat(frames, ignore_index=True)

    out = CORE_DIR / "core_dictionary.csv"
    dictionary.to_csv(out, index=False)
    tables = pd.DataFrame({"table": list(TABLES), "grain": list(TABLES.values())})
    tables.to_csv(CORE_DIR / "core_tables.csv", index=False)

    undocumented = dictionary[dictionary["meaning"] == ""]
    print(f"{dictionary['table'].nunique()} tables | {len(dictionary)} columns documented")
    print(f"wrote {out.name} and core_tables.csv to {CORE_DIR}")
    if not undocumented.empty:
        print("\nundocumented columns:")
        print(undocumented[["table", "column"]].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
