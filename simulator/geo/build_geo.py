"""
Builds the map layers the simulator UI draws, clipped to India and thinned for a browser.

    python simulator/geo/build_geo.py

Reads (fetched once into data/geo/raw/, not tracked):
    HydroRIVERS_v10_as_shp/   HydroSHEDS river network, Asia
        https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip
    GeoDAR_v10_v11/GeoDAR_v11_dams.csv   georeferenced dams, CC BY 4.0, zenodo 6163413
        https://zenodo.org/records/6163413
    data_cleaning/reference/districts.geojson   724 post-2020 districts, official boundary

Writes simulator/ui/data/
    districts.geojson   simplified outlines, name + state
    india.geojson       the national outline, dissolved from the districts
    rivers.geojson      river reaches inside India with mean discharge above MIN_DISCHARGE_CMS
    dams.geojson        GeoDAR dams inside India, with storage where GeoDAR has it

Terrain needs nothing here: the UI reads public elevation tiles (AWS Terrain
Tiles, terrarium encoding) straight from the CDN.

The national outline is dissolved from the district file, not taken from Natural
Earth or OpenStreetMap, so the map shows India's boundary as the Survey of India
depicts it. Published maps of India are required to.
"""

import csv
import json
from pathlib import Path

import shapefile
from shapely.geometry import mapping, shape, LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "geo" / "raw"
OUT = REPO / "simulator" / "ui" / "data"
DISTRICTS = REPO / "data_cleaning" / "reference" / "districts.geojson"

MIN_DISCHARGE_CMS = 15     # long-term mean flow; keeps the named-river network, drops rills
RIVER_TOL = 0.01           # degrees, ~1 km: invisible at the zooms the rivers are drawn at
DISTRICT_TOL = 0.005
ROUND = 4                  # ~10 m


def rnd(geom):
    def r(c):
        return [round(c[0], ROUND), round(c[1], ROUND)]
    g = mapping(geom)

    def walk(x):
        if isinstance(x[0], (int, float)):
            return r(x)
        return [walk(y) for y in x]
    g["coordinates"] = walk(g["coordinates"])
    return g


def dump(name, feats):
    p = OUT / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                            separators=(",", ":")))
    print(f"  {name}: {len(feats):,} features, {p.stat().st_size / 1e6:.2f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    src = json.loads(DISTRICTS.read_text())
    dist = []
    for f in src["features"]:
        g = make_valid(shape(f["geometry"]))
        dist.append((f["properties"], g))
    india = unary_union([g for _, g in dist]).buffer(0)
    inside = prep(india.buffer(0.02))
    print(f"india bounds {[round(v, 2) for v in india.bounds]}")

    dump("districts.geojson", [
        {"type": "Feature", "properties": {"name": p["n"], "state": p["s"]},
         "geometry": rnd(g.simplify(DISTRICT_TOL, preserve_topology=True))}
        for p, g in dist])
    dump("india.geojson", [{"type": "Feature", "properties": {"name": "India"},
                            "geometry": rnd(india.simplify(DISTRICT_TOL))}])

    # ---- rivers
    r = shapefile.Reader(str(RAW / "HydroRIVERS_v10_as_shp" / "HydroRIVERS_v10_as"))
    names = [f[0] for f in r.fields[1:]]
    i_dis, i_ord, i_main, i_up = (names.index(k) for k in
                                  ("DIS_AV_CMS", "ORD_FLOW", "MAIN_RIV", "UPLAND_SKM"))
    bx0, by0, bx1, by1 = india.bounds
    rivers = []
    for sr in r.iterShapeRecords():
        rec = sr.record
        if rec[i_dis] < MIN_DISCHARGE_CMS:
            continue
        x0, y0, x1, y1 = sr.shape.bbox
        if x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1:
            continue
        line = LineString(sr.shape.points)
        if not inside.intersects(line):
            continue
        rivers.append({"type": "Feature",
                       "properties": {"dis": round(rec[i_dis], 1), "ord": rec[i_ord],
                                      "main": rec[i_main], "up_km2": round(rec[i_up])},
                       "geometry": rnd(line.simplify(RIVER_TOL))})
    dump("rivers.geojson", rivers)

    # ---- dams
    dams = []
    with open(RAW / "GeoDAR_v10_v11" / "GeoDAR_v11_dams.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lat, lon = float(row["lat"]), float(row["lon"])
            if not inside.contains(Point(lon, lat)):
                continue
            vol = float(row["rv_mcm_v11"])
            dams.append({"type": "Feature",
                         "properties": {"id": int(row["id_v11"]),
                                        "storage_mcm": round(vol, 1) if vol > 0 else None,
                                        "grand_id": int(row["id_grd_v13"]) if int(row["id_grd_v13"]) > 0 else None,
                                        "qa": row["qa_rank"]},
                         "geometry": {"type": "Point", "coordinates": [round(lon, ROUND), round(lat, ROUND)]}})
    dump("dams.geojson", dams)
    with_vol = sum(1 for d in dams if d["properties"]["storage_mcm"])
    print(f"  dams with storage volume: {with_vol:,} of {len(dams):,}")


if __name__ == "__main__":
    main()
