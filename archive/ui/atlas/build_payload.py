"""Everything the one map page needs, in a single file it can load.

The page is static: no service behind it, so every number a dropdown can reach is
computed here and embedded. That includes the scenario predictions, which come
from the same regression the rest of the project uses — the model runs here, and
the page reads its answers off a grid.

Four things go in:

* district outlines, joined to CGWB's district labels
* a timeline per district: water level and rainfall, year by year
* what each district is doing: decline, its interval, how many wells
* the scenario grid: what a monsoon of -20% to +30% would gain, per district

The join is the fiddly part. CGWB and the boundary file disagree on 34 of 272
district names, so three rules are applied in order: exact match within the same
state, a close-spelling match restricted to that state (Puruliya/Purulia,
Dakshin kannada/Dakshina Kannada), and an explicit table for renames no string
metric should be trusted to guess (Trichur/Thrissur, Cuddapah/Y.S.R. Kadapa).
Anything still unmatched is reported and left off the map rather than guessed at.

Run from ui/atlas/:  python build_payload.py
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Point, shape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "ml"))

from bits_ml import groundwater as gw  # noqa: E402
from bits_ml import rain_model as rm  # noqa: E402
from bits_ml import splits as S  # noqa: E402
from bits_ml.config import DATA_PROCESSED  # noqa: E402

BOUNDARIES = HERE / "districts.geojson"
YEARS = list(range(2000, 2023))       # well readings: CGWB starts in 2000
RAIN_YEARS = list(range(1998, 2023))  # rainfall: the IMD grids we hold start two years earlier
DELTAS = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

# Renames and transliterations a string metric should not be left to guess. Checked one by one
# against the boundary file's own names; the value is the polygon, the key is what CGWB calls it.
ALIASES = {
    ("Trichur", "Kerala"): "Thrissur",
    ("Ropar", "Punjab"): "Rupnagar",
    ("Sonapur", "Odisha"): "Subarnapur",
    ("Mysore", "Karnataka"): "Mysuru",
    ("Bijapur", "Karnataka"): "Vijayapura",
    ("Belgaum", "Karnataka"): "Belagavi",
    ("Kheri", "Uttar pradesh"): "Lakhimpur Kheri",
    ("East nimar", "Madhya pradesh"): "Khandwa",
    ("West nimar", "Madhya pradesh"): "Khargone",
    ("Cuddapah", "Andhra pradesh"): "Y.S.R. Kadapa",
    ("Dohad", "Gujarat"): "Dahod",
    ("Kachchh", "Gujarat"): "Kutch",
    ("North west", "Delhi"): "North West (CGWB label)",
    ("Gaurella pendra marwahi", "Chhattisgarh"): "Bilaspur",  # carved out of Bilaspur in 2020
    # Same place, two CGWB spellings across vintages: both map to the one polygon and their
    # wells are pooled, rather than letting one silently win.
    ("Allahabad", "Uttar pradesh"): "Prayagraj",
    ("Praygraj", "Uttar pradesh"): "Prayagraj",
}

_norm = lambda text: re.sub(r"[^a-z]", "", (text or "").lower())


def load_boundaries() -> list[dict]:
    if not BOUNDARIES.exists():
        raise SystemExit(f"{BOUNDARIES} is missing. It is the India district outline used for drawing only;\n"
                         f"copy the districts FeatureCollection there (properties: n = district, s = state).")
    return json.loads(BOUNDARIES.read_text(encoding="utf-8"))["features"]


def match_districts(features: list[dict], districts: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """CGWB (district, state) -> polygon name, by exact, then close spelling, then alias table."""
    by_state: dict[str, list[str]] = {}
    for feature in features:
        properties = feature["properties"]
        by_state.setdefault(_norm(properties.get("s")), []).append(properties.get("n"))

    rows = []
    for district, state in districts[["district", "state"]].itertuples(index=False):
        names = by_state.get(_norm(state), [])
        lookup = {_norm(name): name for name in names}
        alias = ALIASES.get((district, state))
        if alias and _norm(alias) in lookup:
            rows.append((district, state, lookup[_norm(alias)], "alias"))
        elif _norm(district) in lookup:
            rows.append((district, state, lookup[_norm(district)], "exact"))
        else:
            close = difflib.get_close_matches(_norm(district), list(lookup), n=1, cutoff=0.78)
            rows.append((district, state, lookup[close[0]] if close else None, "spelling" if close else "unmatched"))
    table = pd.DataFrame(rows, columns=["district", "state", "polygon", "how"])
    mapping = {(r.district, r.state): r.polygon for r in table.itertuples() if r.polygon}
    return mapping, table


def water_levels(readings: pd.DataFrame) -> pd.DataFrame:
    """Per district and year: the post-monsoon water level, where wells exist."""
    november = readings[readings["campaign"] == "Nov"]
    return (november.groupby(["state", "district", "year"])["depth_mbgl"].median()
            .rename("depth_m").reset_index())


def rain_by_polygon(features: list[dict]) -> dict[str, dict]:
    """Rainfall for every district the grid reaches, straight from the IMD cells inside it.

    Rain is a grid over the whole country, so it does not depend on there being a
    monitored well: 702 of 724 districts contain at least one land cell, against
    270 with wells. Drawing rain only where wells happen to be left Rajasthan,
    Uttarakhand and the north-east blank on a rainfall map, which said something
    false about the data. The 22 districts with no cell at all — small islands and
    coastal slivers outside the grid's land mask — are marked as outside the grid
    rather than as missing.
    """
    from bits_ml import rainfall as rf

    dates, grid = rf.read_year(next(iter(sorted(rf.IMD_DIR.glob("imd_rf25_*.nc")))))
    land = ~np.isnan(grid[0])
    lat = np.repeat(rf.IMD.lats, rf.NLON)[land]
    lon = np.tile(rf.IMD.lons, rf.NLAT)[land]
    flat = np.flatnonzero(land)
    tree = STRtree([Point(x, y) for x, y in zip(lon, lat)])

    cells_of: dict[str, np.ndarray] = {}
    for feature in features:
        found = tree.query(shape(feature["geometry"]), predicate="intersects")
        if len(found):
            cells_of[feature["properties"]["n"]] = flat[np.asarray(found)]

    totals = {name: {"jjas": [], "annual": []} for name in cells_of}
    for year in RAIN_YEARS:
        path = rf.IMD_DIR / f"imd_rf25_{year}.nc"
        if not path.exists():
            for name in totals:
                totals[name]["jjas"].append(None)
                totals[name]["annual"].append(None)
            continue
        dates, grid = rf.read_year(path)
        monsoon = np.isin(dates.month, rm.MONSOON)
        jjas_grid = np.nansum(grid[monsoon], axis=0)
        annual_grid = np.nansum(grid, axis=0)
        for name, cells in cells_of.items():
            totals[name]["jjas"].append(round(float(np.mean(jjas_grid[cells])), 1))
            totals[name]["annual"].append(round(float(np.mean(annual_grid[cells])), 1))
        print(f"   rain {year}", flush=True)
    return totals


def main() -> None:
    features = load_boundaries()
    ranking = pd.read_csv(DATA_PROCESSED / "district_ranking.csv")
    whatif = pd.read_csv(DATA_PROCESSED / "district_whatif.csv")
    recharge_path = DATA_PROCESSED / "district_recharge.csv"
    recharge = pd.read_csv(recharge_path) if recharge_path.exists() else pd.DataFrame()
    readings = gw.load_readings()
    readings = readings[~readings["far_from_district"]]

    mapping, match_table = match_districts(features, ranking)
    print(match_table["how"].value_counts().to_string())
    missing = match_table[match_table["how"] == "unmatched"]
    if len(missing):
        print("\nleft off the map (no polygon found):")
        print(missing[["district", "state"]].to_string(index=False))

    print("reading the rain grid for every district it reaches:")
    rain = rain_by_polygon(features)

    frame = water_levels(readings)
    series_by_district: dict[str, dict] = {}
    for (state, district), group in frame.groupby(["state", "district"]):
        polygon = mapping.get((district, state))
        if not polygon:
            continue
        group = group.sort_values("year")
        entry = series_by_district.setdefault(polygon, {"depth": {}})
        for year, depth in group[["year", "depth_m"]].itertuples(index=False):
            # two CGWB spellings can share a polygon: pool their readings, then average
            if pd.notna(depth):
                entry["depth"].setdefault(int(year), []).append(float(depth))

    districts = {}
    gain_columns = {d: f"gain_{int(round(d * 100)):+d}pct" for d in DELTAS if d != 0}
    for row in ranking.itertuples():
        polygon = mapping.get((row.district, row.state))
        if not polygon:
            continue
        scenario = whatif[(whatif["district"] == row.district) & (whatif["state"] == row.state)]
        gains = {"0": 0.0}
        break_even = None
        if len(scenario):
            record = scenario.iloc[0]
            for delta, column in gain_columns.items():
                if column in scenario.columns and pd.notna(record[column]):
                    gains[str(delta)] = round(float(record[column]), 4)
            break_even = None if pd.isna(record.get("break_even_monsoon")) else round(float(record["break_even_monsoon"]), 3)
        entry = districts.setdefault(polygon, {
            "name": polygon, "state": row.state, "cgwb": [], "wells": 0,
            "decline": 0.0, "ci_low": 0.0, "ci_high": 0.0, "verdict": row.verdict,
            "gains": gains, "break_even": break_even,
        })
        entry["cgwb"].append(row.district)
        # pooled across spellings, weighted by wells
        total = entry["wells"] + int(row.wells)
        for key, value in (("decline", row.decline_m_per_year), ("ci_low", row.ci_low), ("ci_high", row.ci_high)):
            entry[key] = round((entry[key] * entry["wells"] + float(value) * int(row.wells)) / total, 4)
        entry["wells"] = total

    for polygon, entry in districts.items():
        values = series_by_district.get(polygon, {}).get("depth", {})
        entry["depth"] = [round(float(np.mean(values[year])), 2) if year in values else None for year in YEARS]

    # how much of the monsoon stays in the ground, by the water-table method
    for row in recharge.itertuples():
        polygon = mapping.get((row.district, row.state))
        if polygon in districts:
            districts[polygon]["infiltration"] = round(float(row.infiltration_pct), 1)
            districts[polygon]["rise_m"] = round(float(row.rise_m), 2)

    # Rain stands on its own: every district the grid reaches gets a series, whether or not a
    # well was ever drilled there, so a rainfall map shows the whole country.
    rainfall = {name: {"jjas": totals["jjas"], "annual": totals["annual"]} for name, totals in rain.items()}
    outside_grid = [f["properties"]["n"] for f in features if f["properties"]["n"] not in rain]

    payload = {
        "meta": {
            "years": YEARS, "rainYears": RAIN_YEARS, "deltas": DELTAS,
            "trainYears": [min(S.FINAL_TRAIN_YEARS), max(S.FINAL_TRAIN_YEARS)],
            "districts": len(districts), "rainDistricts": len(rainfall), "outsideGrid": len(outside_grid),
            "note": ("Scenario gains come from the gradient-boosting model of the November level, run for every "
                     "district at each rainfall level and stored here; the page reads them rather than computing "
                     "anything."),
        },
        "districts": districts,
        "rain": rainfall,
        "boundaries": {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "properties": {"n": f["properties"].get("n"), "s": f["properties"].get("s")},
             "geometry": f["geometry"]}
            for f in features]},
    }
    out = HERE / "atlas_payload.js"
    out.write_text("window.ATLAS=" + json.dumps(payload, separators=(",", ":")) + ";", encoding="utf-8")
    print(f"\nwrote {out.name} {out.stat().st_size / 1e6:.2f} MB | polygons {len(features)}")
    print(f"   rainfall in {len(rainfall)} districts ({RAIN_YEARS[0]}-{RAIN_YEARS[-1]}), "
          f"{len(outside_grid)} outside the grid's land mask")
    print(f"   water table in {len(districts)} districts ({YEARS[0]}-{YEARS[-1]}), the rest have no monitored wells")


if __name__ == "__main__":
    main()
