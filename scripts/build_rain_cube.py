"""
Compresses the 607 MB of IMD NetCDF files into one 22 MB array.

    python3 scripts/build_rain_cube.py

Reads  data/imd_rainfall/imd_rf25_1998.nc .. imd_rf25_2022.nc
Writes data/derived/rain_cube.npz

Why this exists. The NetCDF files are 25 MB each because they store a value for
every point of a 129 x 135 grid, and 12,451 of those 17,415 points are sea,
coded -999. Only 4,964 are land. Keeping the land columns only, as tenths of a
millimetre in int16, and letting zlib have the 72 percent of entries that are
exactly zero, turns 607 MB into about 22 MB with no loss beyond rounding to
0.1 mm. That fits in the repository, so the whole project can be rebuilt
without downloading anything.

Contents of the npz:
    rain      (n_days, n_land) int16, tenths of a mm, MISSING for no data
    lat_idx   (n_land,) row of each land column in the full grid
    lon_idx   (n_land,) column of each land column in the full grid
    lat       (n_lat,)  grid latitudes,  6.50 .. 38.50 step 0.25
    lon       (n_lon,)  grid longitudes, 66.50 .. 100.00 step 0.25
    days      (n_days,) days since 1970-01-01
"""

from pathlib import Path
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
IMD_DIR = REPO / "data" / "imd_rainfall"
OUT = REPO / "data" / "derived" / "rain_cube.npz"
YEARS = range(1998, 2023)
MISSING = np.int16(-9999)
SCALE = 10.0            # tenths of a millimetre


def main():
    files = [IMD_DIR / f"imd_rf25_{y}.nc" for y in YEARS]
    absent = [f.name for f in files if not f.exists()]
    if absent:
        raise SystemExit(
            f"missing {len(absent)} rainfall files, first is {absent[0]}.\n"
            f"run  python3 scripts/fetch_data.py  first")

    # A cell counts as land if it ever reports a finite value. Checked against
    # every year rather than a sample, because the answer has to be stable.
    ds0 = xr.open_dataset(files[0])
    lat = ds0["LATITUDE"].values.astype("float64")
    lon = ds0["LONGITUDE"].values.astype("float64")
    land = np.zeros((lat.size, lon.size), bool)
    ds0.close()

    for f in files:
        d = xr.open_dataset(f)
        land |= np.isfinite(d["RAINFALL"].values).any(axis=0)
        d.close()
    li, lj = np.nonzero(land)
    print(f"land cells {land.sum():,} of {land.size:,}")

    chunks, days = [], []
    for f in files:
        d = xr.open_dataset(f)
        r = d["RAINFALL"].values[:, li, lj]
        q = np.where(np.isfinite(r), np.rint(np.clip(r, 0, 3000) * SCALE), MISSING)
        chunks.append(q.astype("int16"))
        days.append(d["TIME"].values.astype("datetime64[D]").astype("int32"))
        d.close()

    rain = np.concatenate(chunks)
    days = np.concatenate(days)
    assert (np.diff(days) == 1).all(), "gap in the daily record"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, rain=rain, lat_idx=li.astype("int16"),
                        lon_idx=lj.astype("int16"), lat=lat, lon=lon, days=days)
    print(f"{rain.shape[0]:,} days x {rain.shape[1]:,} land cells")
    print(f"missing entries {(rain == MISSING).sum():,}  zeros {(rain == 0).mean():.1%}")
    print(f"wrote {OUT}  {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
