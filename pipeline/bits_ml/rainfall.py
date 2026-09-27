"""Rainfall at the exact coordinates of each well, from the IMD 0.25 degree grid.

Each well is sampled by masked bilinear interpolation: the four grid nodes
around its true latitude and longitude, weighted by proximity, with the weights
renormalised over land nodes. IMD stores non-land nodes as -999, which xarray
decodes to NaN. Plain bilinear interpolation returns NaN whenever any of the
four nodes is sea, and a skipna monthly sum then turns that NaN into 0 mm, a
fabricated dry month. Renormalising over land nodes avoids both.

Wells whose reported coordinates were rounded are sampled as the average over
their rounding box (box_stencil, coordinate_half_width). Wet days and the
wettest day are counted at the grid nodes first and then interpolated
(well_monthly), which keeps them unbiased.

Run from ml/:  python -m bits_ml.rainfall
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import CGWB_CSV, DATA_PROCESSED, IMD_DIR

LAT0, LON0, STEP = 6.5, 66.5, 0.25
NLAT, NLON = 129, 135
WET_DAY_MM = 2.5  # IMD rainy-day threshold


@dataclass(frozen=True)
class Grid:
    """A regular latitude-longitude grid of node centres, flattened row by row, latitude ascending."""

    lat0: float
    lon0: float
    step: float
    nlat: int
    nlon: int

    @property
    def lats(self) -> np.ndarray:
        return self.lat0 + self.step * np.arange(self.nlat)

    @property
    def lons(self) -> np.ndarray:
        return self.lon0 + self.step * np.arange(self.nlon)

    @property
    def size(self) -> int:
        return self.nlat * self.nlon


IMD = Grid(LAT0, LON0, STEP, NLAT, NLON)


@dataclass(frozen=True)
class Stencil:
    """Grid nodes and weights used to sample each well."""

    idx: np.ndarray  # (n_wells, k) flat indices into a grid
    w: np.ndarray    # (n_wells, k) weights, each row sums to 1


def _grid_position(lat, lon, grid: Grid = IMD):
    fi = (np.asarray(lat, dtype=float) - grid.lat0) / grid.step
    fj = (np.asarray(lon, dtype=float) - grid.lon0) / grid.step
    tol = 1e-9  # grids with a step like 0.05 are not exact in binary
    outside = (fi < -tol) | (fi > grid.nlat - 1 + tol) | (fj < -tol) | (fj > grid.nlon - 1 + tol)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} coordinate(s) fall outside the grid")
    return np.clip(fi, 0, grid.nlat - 1), np.clip(fj, 0, grid.nlon - 1)


def bilinear_stencil(lat, lon, grid: Grid = IMD) -> Stencil:
    fi, fj = _grid_position(lat, lon, grid)
    # clamp so a point on the last row or column still has an upper neighbour
    i0 = np.minimum(np.floor(fi).astype(int), grid.nlat - 2)
    j0 = np.minimum(np.floor(fj).astype(int), grid.nlon - 2)
    di, dj = fi - i0, fj - j0
    n = grid.nlon
    idx = np.stack([i0 * n + j0, i0 * n + j0 + 1, (i0 + 1) * n + j0, (i0 + 1) * n + j0 + 1], axis=1)
    w = np.stack([(1 - di) * (1 - dj), (1 - di) * dj, di * (1 - dj), di * dj], axis=1)
    return Stencil(idx, w)


def nearest_stencil(lat, lon, grid: Grid = IMD) -> Stencil:
    """The Week-1 notebook's nearest-node snapping, kept for comparison."""
    fi, fj = _grid_position(lat, lon, grid)
    idx = (np.round(fi).astype(int) * grid.nlon + np.round(fj).astype(int))[:, None]
    return Stencil(idx, np.ones(idx.shape))


def _merge(idx: np.ndarray, w: np.ndarray) -> Stencil:
    """Merge repeated nodes within each row, summing their weights."""
    order = np.argsort(idx, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    w = np.take_along_axis(w, order, axis=1)
    first = np.ones(idx.shape, dtype=bool)
    first[:, 1:] = idx[:, 1:] != idx[:, :-1]
    slot = np.cumsum(first, axis=1) - 1
    rows = np.broadcast_to(np.arange(len(idx))[:, None], idx.shape)
    out_idx = np.zeros((len(idx), int(slot.max()) + 1), dtype=idx.dtype)
    out_w = np.zeros(out_idx.shape)
    out_idx[rows, slot] = idx
    np.add.at(out_w, (rows, slot), w)  # padding slots keep weight 0
    return Stencil(out_idx, out_w)


def box_stencil(lat, lon, half_width, grid: Grid = IMD, n: int = 5) -> Stencil:
    """Average of bilinear samples over a square box around each point.

    half_width (degrees, scalar or one per point) is how far the true location
    may lie from the reported one; 0 gives the same result as bilinear_stencil.
    The box is sampled on an n x n midpoint grid and nodes shared between
    sub-points are merged, so the stencil stays small.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    _grid_position(lat, lon, grid)  # reject reported points outside the grid
    h = np.broadcast_to(np.asarray(half_width, dtype=float), lat.shape)
    offsets = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    da, db = (o.reshape(-1) for o in np.meshgrid(offsets, offsets, indexing="ij"))
    sub_lat = np.clip(lat[:, None] + h[:, None] * da, grid.lats[0], grid.lats[-1])
    sub_lon = np.clip(lon[:, None] + h[:, None] * db, grid.lons[0], grid.lons[-1])
    st = bilinear_stencil(sub_lat.reshape(-1), sub_lon.reshape(-1), grid)
    return _merge(st.idx.reshape(len(lat), -1), st.w.reshape(len(lat), -1) / da.size)


def coordinate_half_width(lat, lon) -> np.ndarray:
    """Half-width in degrees of the rounding box a reported coordinate stands for.

    Both axes on a 0.1 degree, 0.05 degree, whole-arcminute or 0.01 degree
    lattice (checked in that order of priority) mean the true location is
    anywhere within half a step. Anything finer is taken as exact (0).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    h = np.zeros(np.broadcast(lat, lon).shape)
    # finest first, so a coarser lattice that also matches overwrites it
    for per_degree, tol in ((100, 0.002), (60, 0.004), (20, 0.001), (10, 0.0005)):
        a, b = lat * per_degree, lon * per_degree
        on = (np.abs(a - np.round(a)) < tol) & (np.abs(b - np.round(b)) < tol)
        h[on] = 0.5 / per_degree
    return h


def sample(grid: np.ndarray, stencil: Stencil) -> np.ndarray:
    """Sample a flattened grid (..., grid size), NaN for non-land, at each well.

    Weights are renormalised over the land nodes of each stencil. A well whose
    nodes are all non-land gets NaN, never 0.
    """
    vals = grid[..., stencil.idx]  # (..., n_wells, k)
    land = ~np.isnan(vals)
    w = np.where(land, stencil.w, 0.0)
    den = w.sum(axis=-1)
    num = (np.where(land, vals, 0.0) * w).sum(axis=-1)
    out = np.full(den.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def land_weight(grid_day: np.ndarray, stencil: Stencil) -> np.ndarray:
    """Share of each well's stencil weight that falls on land (1.0 = fully inland)."""
    return np.where(~np.isnan(grid_day[stencil.idx]), stencil.w, 0.0).sum(axis=-1)


def read_year(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """One IMD yearly file as (dates, grid), grid shaped (days, NLAT * NLON)."""
    path = Path(path)
    with xr.open_dataset(path) as ds:  # mask_and_scale decodes -999 to NaN
        da = ds["RAINFALL"].transpose("TIME", "LATITUDE", "LONGITUDE")
        lat, lon = da["LATITUDE"].values, da["LONGITUDE"].values
        if lat.size != NLAT or lon.size != NLON or not (
            np.allclose(lat, IMD.lats) and np.allclose(lon, IMD.lons)
        ):
            raise ValueError(f"{path.name}: grid differs from the IMD 0.25 degree layout")
        grid = da.values.reshape(da.sizes["TIME"], -1).astype(np.float64)
        dates = pd.DatetimeIndex(da["TIME"].values)

    if np.nanmin(grid) < 0:
        raise ValueError(f"{path.name}: negative rainfall after masking")
    year = dates[0].year
    expected = np.arange(f"{year}-01-01", f"{year + 1}-01-01", dtype="datetime64[D]")
    if len(dates) != len(expected) or (dates.values.astype("datetime64[D]") != expected).any():
        raise ValueError(f"{path.name}: TIME is not one complete calendar year")
    return dates, grid


def monthly_totals(dates: pd.DatetimeIndex, daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monthly rainfall and wet-day counts, each (12, n_wells), for one calendar year.

    A month with no valid day stays NaN rather than summing to 0.
    """
    month = np.asarray(dates.month)
    n = daily.shape[1]
    rain = np.full((12, n), np.nan)
    wet = np.full((12, n), np.nan)
    for m in range(1, 13):
        block = daily[month == m]
        has = (~np.isnan(block)).any(axis=0)
        rain[m - 1, has] = np.nansum(block, axis=0)[has]
        wet[m - 1, has] = (block > WET_DAY_MM).sum(axis=0)[has]
    return rain, wet


def monthly_max(dates: pd.DatetimeIndex, daily: np.ndarray) -> np.ndarray:
    """Wettest day of each month, (12, n), NaN for a month with no valid day."""
    month = np.asarray(dates.month)
    out = np.full((12, daily.shape[1]), np.nan)
    for m in range(1, 13):
        block = daily[month == m]
        has = (~np.isnan(block)).any(axis=0)
        out[m - 1, has] = np.nanmax(block[:, has], axis=0)
    return out


def well_monthly(dates: pd.DatetimeIndex, grid: np.ndarray, stencil: Stencil):
    """Monthly rain, wet-day count and wettest day at each well, each (12, n_wells).

    All three are computed at the grid nodes and then interpolated. Rain totals
    come out the same either way, but counts and maxima do not: interpolating
    daily rain first spreads light rain onto dry days and flattens peaks. Hiding
    IMD nodes and predicting them back (scripts/interpolation_check.py) gave wet
    days +12% and the wettest day -12% that way, against +0.2% and -0.7% here.
    Wet days are therefore an expected count and can be fractional.
    """
    rain, wet = monthly_totals(dates, grid)
    return sample(rain, stencil), sample(wet, stencil), sample(monthly_max(dates, grid), stencil)


def load_wells(path: Path = CGWB_CSV) -> pd.DataFrame:
    """CGWB wells with a stable key built from coordinates (Station Code is unusable)."""
    wells = pd.read_csv(path, encoding="utf-8-sig")
    wells["well_uid"] = (
        wells["Latitude"].map("{:.4f}".format) + "_" + wells["Longitude"].map("{:.4f}".format)
    )
    if wells["well_uid"].duplicated().any():
        raise ValueError("well coordinates are not unique, so well_uid would collide")
    return wells


def extract(lat, lon, well_uid, imd_dir: Path = IMD_DIR, stencil: Stencil | None = None):
    """Daily and monthly rainfall at each well for every IMD year in imd_dir.

    Returns (dates, daily float32 array shaped (days, n_wells), monthly long DataFrame).
    The daily array is interpolated rain: sum it over any window, but take wet
    days and maxima from the monthly frame (see well_monthly).
    """
    stencil = stencil if stencil is not None else bilinear_stencil(lat, lon)
    well_uid = np.asarray(well_uid)
    files = sorted(Path(imd_dir).glob("imd_rf25_*.nc"))
    if not files:
        raise FileNotFoundError(f"no imd_rf25_*.nc files in {imd_dir}")

    all_dates, daily_chunks, monthly_frames = [], [], []
    for path in files:
        dates, grid = read_year(path)
        values = sample(grid, stencil)
        rain, wet, wettest = well_monthly(dates, grid, stencil)
        months = pd.date_range(f"{dates[0].year}-01-01", periods=12, freq="MS")
        monthly_frames.append(
            pd.DataFrame(
                {
                    "well_uid": np.tile(well_uid, 12),
                    "month": np.repeat(months, len(well_uid)),
                    "rain_mm": rain.reshape(-1).astype(np.float32),
                    "wet_days": wet.reshape(-1).astype(np.float32),
                    "max_day_mm": wettest.reshape(-1).astype(np.float32),
                }
            )
        )
        all_dates.append(dates)
        daily_chunks.append(values.astype(np.float32))

    dates = all_dates[0].append(all_dates[1:])
    steps = np.diff(dates.values.astype("datetime64[D]").astype(np.int64))
    if (steps != 1).any():
        raise ValueError("IMD years are not contiguous")
    return dates, np.concatenate(daily_chunks), pd.concat(monthly_frames, ignore_index=True)


def main() -> None:
    wells = load_wells()
    lat = wells["Latitude"].to_numpy()
    lon = wells["Longitude"].to_numpy()
    uid = wells["well_uid"].to_numpy().astype(str)

    half = coordinate_half_width(lat, lon)
    dates, daily, monthly = extract(lat, lon, uid, stencil=box_stencil(lat, lon, half))

    no_land = np.isnan(daily).all(axis=0)
    if no_land.any():
        cols = ["State", "District", "Station Name", "Latitude", "Longitude"]
        raise RuntimeError(
            f"{int(no_land.sum())} well(s) have no land node around them:\n{wells.loc[no_land, cols]}"
        )
    if np.isnan(daily).any():
        raise RuntimeError("some well-days are missing although every well has land nodes")
    never_wet = monthly.groupby("well_uid")["rain_mm"].sum() == 0
    if never_wet.any():
        raise RuntimeError(f"{int(never_wet.sum())} well(s) show zero rain across the whole record")

    y0, y1 = dates[0].year, dates[-1].year
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    monthly_path = DATA_PROCESSED / f"well_monthly_rain_{y0}_{y1}.parquet"
    daily_path = DATA_PROCESSED / f"well_daily_rain_{y0}_{y1}.npz"
    monthly.to_parquet(monthly_path, index=False)
    np.savez_compressed(
        daily_path, dates=dates.values.astype("datetime64[D]"), well_uid=uid, rain_mm=daily
    )

    print(
        f"wells {len(uid):,} | days {len(dates):,} "
        f"({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}) | well-months {len(monthly):,} "
        f"| rounded coordinates sampled as boxes {int((half > 0).sum())}"
    )
    print(f"wrote {monthly_path.name} and {daily_path.name} to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
