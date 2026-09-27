"""Grid cells and wells placed inside district outlines, by geometry rather than by name.

The CGWB file names a district for every well, but those names are a mix of
vintages and spellings (Rajnadgaon and Rajnandgaon, Ranga reddy and Rangareddy,
Telangana's ten districts before the 2016 split beside its thirty-three after).
A name cannot be joined to anything reliably. A coordinate can: each well is
placed in the outline that contains it, and each rain cell is split across the
outlines it overlaps, all against one boundary file of one vintage.

Two mappings, written by the steps that own them:

* `bridge_cell_district` (grid.py)  land cell x district: how many square kilometres
  of the cell's 0.25 degree box fall inside the district. District rain is the
  area-weighted mean of its cells, so it exists for every district the grid
  reaches, whether or not a well stands there.
* `district`, `state` on `well_master` (ingest.py)  the outline a well stands in. A well just off a
  coastline or border (outside every outline) takes the nearest one within
  NEAREST_MAX_KM and says so; beyond that it takes none. The CGWB name is kept
  beside it with flags for whether the two agree, which is a check on the
  coordinates, not a correction of them. Most district disagreements are
  renames and splits (Trichur is Thrissur, Khammam is now partly Bhadradri
  Kothagudem) and are exactly what geometry fixes. A *state* disagreement is
  rarer: 489 wells, most filed under Andhra Pradesh while standing in
  Maharashtra, Rajasthan or Gujarat. Their water levels follow the wells at
  their coordinates, not the wells of the labelled district, so it is the label
  that is wrong and the coordinates are kept. `state_agrees` records it.

Areas are computed in degrees and scaled by cos(latitude) of the cell centre.
Across one 0.25 degree cell that is exact to a fraction of a percent, which is
all a weight needs.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import shape

from . import rainfall as rf
from .config import DISTRICTS_GEOJSON
from .grid import EARTH_KM_PER_DEG

HALF_CELL = rf.IMD.step / 2
NEAREST_MAX_KM = 5.0      # a well this close to an outline is on the coast, not misplaced
NAME_MATCH_CUTOFF = 0.78  # spelling tolerance for comparing CGWB and outline names
# States split within the record: a well filed under either is not misplaced in the other.
SAME_STATE = {frozenset({"andhrapradesh", "telangana"})}

_norm = lambda text: re.sub(r"[^a-z]", "", str(text or "").lower().replace("&", "and"))


def load_districts(path: Path = DISTRICTS_GEOJSON) -> pd.DataFrame:
    """One row per district outline: name, state, repaired geometry."""
    features = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    geoms = [shapely.make_valid(shape(f["geometry"])) for f in features]
    return pd.DataFrame({
        "district": [f["properties"]["n"] for f in features],
        "state": [f["properties"]["s"] for f in features],
        "geometry": geoms,
    })


def _km2_per_deg2(lat) -> np.ndarray:
    return EARTH_KM_PER_DEG ** 2 * np.cos(np.radians(np.asarray(lat, dtype=float)))


def cell_district_area(cells: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """Land cell x district, with the overlap in km2 and both shares it implies.

    `cells` needs cell_id, cell_lat, cell_lon. `frac_of_cell` sums to less than
    one on a coast or the national border, where part of the box is outside
    every outline; `weight_in_district` always sums to one per district.
    """
    lat, lon = cells["cell_lat"].to_numpy(float), cells["cell_lon"].to_numpy(float)
    boxes = shapely.box(lon - HALF_CELL, lat - HALF_CELL, lon + HALF_CELL, lat + HALF_CELL)
    tree = STRtree(districts["geometry"].to_numpy())
    cell_ix, district_ix = tree.query(boxes, predicate="intersects")
    overlap = shapely.area(shapely.intersection(boxes[cell_ix], districts["geometry"].to_numpy()[district_ix]))

    out = pd.DataFrame({
        "cell_id": cells["cell_id"].to_numpy()[cell_ix],
        "district": districts["district"].to_numpy()[district_ix],
        "state": districts["state"].to_numpy()[district_ix],
        "frac_of_cell": overlap / (2 * HALF_CELL) ** 2,
        "area_km2": overlap * _km2_per_deg2(cells["cell_lat"].to_numpy()[cell_ix]),
    })
    out = out[out["area_km2"] > 0].copy()
    total = out.groupby(["district", "state"])["area_km2"].transform("sum")
    out["weight_in_district"] = out["area_km2"] / total
    return out.sort_values(["state", "district", "cell_id"], ignore_index=True)


def _names_agree(cgwb: str, polygon: str) -> bool:
    a, b = _norm(cgwb), _norm(polygon)
    return bool(a) and bool(b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= NAME_MATCH_CUTOFF)


def well_district(wells: pd.DataFrame, districts: pd.DataFrame,
                  nearest_max_km: float = NEAREST_MAX_KM) -> pd.DataFrame:
    """Each well's containing outline, or the nearest one within `nearest_max_km`.

    `wells` needs well_uid, lat, lon, cgwb_district and cgwb_state (the file's labels). `how` is
    "inside", "nearest" or "none"; a well on a shared border takes the first
    outline found, which is arbitrary but stable.
    """
    points = shapely.points(wells["lon"].to_numpy(), wells["lat"].to_numpy())
    geoms = districts["geometry"].to_numpy()
    tree = STRtree(geoms)

    hit = np.full(len(wells), -1)
    well_ix, poly_ix = tree.query(points, predicate="within")
    first = pd.Series(poly_ix).groupby(well_ix).first()
    hit[first.index.to_numpy()] = first.to_numpy()
    how = np.where(hit >= 0, "inside", "none").astype(object)
    distance_km = np.where(hit >= 0, 0.0, np.nan)

    outside = np.flatnonzero(hit < 0)
    if len(outside):
        max_deg = nearest_max_km / EARTH_KM_PER_DEG
        near_w, near_p = tree.query_nearest(points[outside], max_distance=max_deg, return_distance=False, all_matches=False)
        chosen = outside[near_w]
        hit[chosen] = near_p
        how[chosen] = "nearest"
        distance_km[chosen] = shapely.distance(points[chosen], geoms[near_p]) * EARTH_KM_PER_DEG

    found = hit >= 0
    out = pd.DataFrame({
        "well_uid": wells["well_uid"].to_numpy(),
        "cgwb_district": wells["cgwb_district"].to_numpy(),
        "cgwb_state": wells["cgwb_state"].to_numpy(),
        "district": np.where(found, districts["district"].to_numpy()[np.maximum(hit, 0)], None),
        "state": np.where(found, districts["state"].to_numpy()[np.maximum(hit, 0)], None),
        "how": how,
        "distance_km": distance_km,
    })
    out["name_agrees"] = [found_ and _names_agree(c, p)
                          for found_, c, p in zip(found, out["cgwb_district"], out["district"])]
    cgwb_state, state = out["cgwb_state"].map(_norm), out["state"].map(_norm)
    out["state_agrees"] = found & ((cgwb_state == state) | pd.Series(
        [frozenset({a, b}) in SAME_STATE for a, b in zip(cgwb_state, state)], index=out.index))
    return out
