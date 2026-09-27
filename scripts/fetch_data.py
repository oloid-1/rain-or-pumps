"""
Fetches the raw inputs that are too large or too unclear in licence to track.

    python3 scripts/fetch_data.py --cgwb      the 2.8 MB quality-controlled extract
    python3 scripts/fetch_data.py --archive   the 26 MB raw 32,299-well archive
    python3 scripts/fetch_data.py --imd       the 607 MB of daily rainfall NetCDFs
    python3 scripts/fetch_data.py --all

None of this is needed to train. data/derived/rain_cube.npz carries the whole
rainfall record at 22 MB and is in the repository; the CGWB extract is small
enough to fetch in a second. The archive and the NetCDFs are only needed to
rebuild the full 32,299-well table from scratch, or to regenerate the cube.

Files already present are left alone unless --force is given.
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

FIGSHARE_ARTICLE = 29293877          # doi 10.6084/m9.figshare.29293877.v3
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}/files"

# Wanted members of the figshare record, by the name figshare reports.
CGWB_EXTRACT = "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv"
CGWB_ARCHIVE = "Quality_controlled_groundwater_levels_over_India.zip"

IMD_YEARS = range(1998, 2023)
IMD_NOTE = """
IMD rainfall is not on a stable direct-download URL. Get it once, by hand:

  1. open https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
  2. download one NetCDF per year, 1998 to 2022
  3. put them in data/imd_rainfall/ named imd_rf25_<year>.nc
  4. run  make cube

You almost certainly do not need to. data/derived/rain_cube.npz already holds
every one of those files, packed, and `make data` uses it automatically.
"""


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest, expect_md5=None):
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {dest.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    if expect_md5:
        import hashlib as _h
        got = _h.md5(tmp.read_bytes()).hexdigest()
        if got != expect_md5:
            tmp.unlink()
            raise SystemExit(f"checksum mismatch for {dest.name}: {got} != {expect_md5}")
    tmp.replace(dest)
    print(f"  {dest.relative_to(REPO)}  {dest.stat().st_size / 1e6:.1f} MB")


def figshare_files():
    import json
    import urllib.request
    with urllib.request.urlopen(FIGSHARE_API, timeout=60) as r:
        return {f["name"]: f for f in json.load(r)}


def fetch_figshare(name, dest, force):
    if dest.exists() and not force:
        print(f"  {dest.relative_to(REPO)} already present, skipping")
        return
    try:
        files = figshare_files()
    except Exception as e:
        print(f"  could not reach figshare ({e}).")
        print(f"  download {name} by hand from "
              f"https://doi.org/10.6084/m9.figshare.29293877.v3 into {dest.parent}")
        return
    if name not in files:
        print(f"  figshare record does not list {name}. It lists: {sorted(files)}")
        return
    f = files[name]
    download(f["download_url"], dest, f.get("computed_md5") or f.get("supplied_md5"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cgwb", action="store_true")
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--imd", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--checksums", action="store_true",
                    help="print sha256 of whatever is already on disk")
    a = ap.parse_args()

    if a.checksums:
        for p in sorted(DATA.rglob("*")):
            if p.is_file() and p.suffix in {".csv", ".zip", ".nc", ".npz"}:
                print(f"{sha256(p)}  {p.relative_to(REPO)}")
        return

    if not (a.cgwb or a.archive or a.imd or a.all):
        ap.print_help()
        return

    if a.cgwb or a.all:
        print("CGWB quality-controlled extract")
        fetch_figshare(CGWB_EXTRACT, DATA / "cgwb" / CGWB_EXTRACT, a.force)

    if a.archive or a.all:
        print("CGWB raw archive")
        fetch_figshare(CGWB_ARCHIVE, DATA / CGWB_ARCHIVE, a.force)

    if a.imd or a.all:
        print("IMD daily rainfall")
        have = sorted((DATA / "imd_rainfall").glob("imd_rf25_*.nc"))
        want = len(list(IMD_YEARS))
        if len(have) >= want and not a.force:
            print(f"  {len(have)} of {want} files already present, skipping")
        else:
            print(f"  {len(have)} of {want} files present")
            print(IMD_NOTE)

    cube = DATA / "derived" / "rain_cube.npz"
    if not cube.exists():
        print("\nnote: data/derived/rain_cube.npz is missing. "
              "It should be in the repository; run `make cube` to rebuild it "
              "once the IMD files are in place.")


if __name__ == "__main__":
    sys.exit(main())
