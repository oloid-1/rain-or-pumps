"""
Builds the town list behind the search box, so people can type a city or town
("Bangalore", "Gurgaon", "Ooty") without knowing which district it is in.

    python simulator/geo/build_places.py          (or: make places)

Reads
    data/geo/raw/geonames/cities1000.zip   GeoNames, every place with 1,000+ people
                                           (CC BY 4.0), from
                                           https://download.geonames.org/export/dump/cities1000.zip
    simulator/ui/data/districts.geojson    the map's district outlines

Writes simulator/ui/data/places.json, one row per Indian town:
    [name, other names, district, state, lat, lon, population]

Each town is placed in the district outline it falls in, the same outlines the
wells and the map use, so choosing a town lands on a district the rest of the UI
knows. Other names keep the Latin-script alternatives GeoNames lists (Bangalore
for Bengaluru, Bombay for Mumbai), which is how old and spoken names are found.
"""

import io
import json
import re
import zipfile
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "geo" / "raw" / "geonames" / "cities1000.zip"
UI = REPO / "simulator" / "ui" / "data"
MIN_POP = 5000            # below this the list is mostly villages that share names
LATIN = re.compile(r"^[A-Za-z][A-Za-z .'\-()]{1,40}$")
MAX_ALT = 8


def main():
    feats = json.loads((UI / "districts.geojson").read_text(encoding="utf-8"))["features"]
    polys = [make_valid(shape(f["geometry"])) for f in feats]
    tree = STRtree(polys)

    rows = []
    with zipfile.ZipFile(RAW) as z, z.open("cities1000.txt") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if c[8] != "IN" or int(c[14] or 0) < MIN_POP:
                continue
            name, ascii_name, lat, lon, pop = c[1], c[2], float(c[4]), float(c[5]), int(c[14])
            pt = Point(lon, lat)
            hit = [i for i in tree.query(pt) if polys[i].covers(pt)]
            if not hit:                       # coastal towns just off the outline
                i = int(tree.nearest(pt))
                if polys[i].distance(pt) > 0.1:
                    continue                  # not in India as the map draws it
                hit = [i]
            p = feats[hit[0]]["properties"]
            alts = []
            for a in [ascii_name] + c[3].split(","):
                a = a.strip()
                if a and LATIN.match(a) and a.lower() != name.lower() and a.lower() not in (x.lower() for x in alts):
                    alts.append(a)
            # clean spellings first (Bangalore before Ban'nkalor): capitalised, no apostrophe
            alts.sort(key=lambda a: (not a[0].isupper(), "'" in a, len(a)))
            rows.append([name, alts[:MAX_ALT], p["name"], p["state"], round(lat, 4), round(lon, 4), pop])

    rows.sort(key=lambda r: -r[6])
    (UI / "places.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(rows):,} towns to {UI / 'places.json'} "
          f"({(UI / 'places.json').stat().st_size / 1e3:.0f} kB)")
    for q in ("Bengaluru", "Mumbai", "Gurugram", "Ooty", "Udagamandalam"):
        hit = [r for r in rows if r[0] == q or q in r[1]]
        print(" ", q, "->", hit[0][:4] if hit else "not found")


if __name__ == "__main__":
    main()
