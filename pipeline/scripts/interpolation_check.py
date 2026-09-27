"""Which way of reading the IMD grid at a point is most accurate? A grid-denial test.

IMD rainfall exists only at 0.25 degree nodes, and no rain gauge sits at any well
to check an interpolated value against. The grid can stand in for one: drop
nodes so the grid is 0.5 (or 1.0) degree, predict the dropped nodes from the
ones kept, and compare with their real values. Running both spacings shows how
the error grows with node distance, which is extrapolated to the real 0.25
degree grid that wells are read from.

Only IMD data is used.

Run from ml/:  python -m scripts.interpolation_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bits_ml import rainfall as rf

DAILY_YEARS = (2019, 2020, 2021, 2022)


def keys_weights(t):
    """Cubic convolution weights (Keys, a = -0.5) for the nodes at offsets -1, 0, 1, 2."""
    t = np.asarray(t, dtype=float)[..., None]
    return np.concatenate(
        [
            -0.5 * t**3 + t**2 - 0.5 * t,
            1.5 * t**3 - 2.5 * t**2 + 1.0,
            -1.5 * t**3 + 2.0 * t**2 + 0.5 * t,
            0.5 * t**3 - 0.5 * t**2,
        ],
        axis=-1,
    )


def neighbourhood(lat, lon, g: rf.Grid):
    """The 4 x 4 block of nodes around each point."""
    fi, fj = rf._grid_position(lat, lon, g)
    i0 = np.minimum(np.floor(fi).astype(int), g.nlat - 2)
    j0 = np.minimum(np.floor(fj).astype(int), g.nlon - 2)
    ki = i0[:, None] + np.arange(-1, 3)
    kj = j0[:, None] + np.arange(-1, 3)
    inside = ((ki >= 0) & (ki < g.nlat))[:, :, None] & ((kj >= 0) & (kj < g.nlon))[:, None, :]
    idx = np.clip(ki, 0, g.nlat - 1)[:, :, None] * g.nlon + np.clip(kj, 0, g.nlon - 1)[:, None, :]
    n = len(fi)
    return idx.reshape(n, 16), inside.reshape(n, 16), fi - i0, fj - j0, ki, kj


def bicubic(values, lat, lon, g: rf.Grid):
    """Cubic convolution where all 16 nodes are land, masked bilinear elsewhere; never negative."""
    idx, inside, ti, tj, _, _ = neighbourhood(lat, lon, g)
    w = (keys_weights(ti)[:, :, None] * keys_weights(tj)[:, None, :]).reshape(len(ti), 16)
    vals = values[..., idx]
    full = inside.all(axis=1) & ~np.isnan(vals).any(axis=-1)
    cubic = np.clip((np.nan_to_num(vals) * w).sum(axis=-1), 0.0, None)
    return np.where(full, cubic, rf.sample(values, rf.bilinear_stencil(lat, lon, g)))


def idw_stencil(lat, lon, g: rf.Grid, power: float = 2.0) -> rf.Stencil:
    """Inverse-distance weights over the 4 x 4 block (longitude scaled by cos latitude)."""
    idx, inside, _, _, ki, kj = neighbourhood(lat, lon, g)
    dy = (g.lat0 + g.step * ki - lat[:, None])[:, :, None]
    dx = (g.lon0 + g.step * kj - lon[:, None])[:, None, :] * np.cos(np.radians(lat))[:, None, None]
    d = np.hypot(dy, dx).reshape(len(lat), 16)
    w = np.where(inside, 1.0 / np.maximum(d, 1e-9) ** power, 0.0)
    return rf.Stencil(idx, w / w.sum(axis=1, keepdims=True))


def predict(values, lat, lon, g: rf.Grid) -> dict[str, np.ndarray]:
    return {
        "nearest (Week 1)": rf.sample(values, rf.nearest_stencil(lat, lon, g)),
        "bilinear": rf.sample(values, rf.bilinear_stencil(lat, lon, g)),
        "bicubic": bicubic(values, lat, lon, g),
        "idw": rf.sample(values, idw_stencil(lat, lon, g)),
    }


def coarse_grid(f: int):
    rows, cols = np.arange(0, rf.NLAT, f), np.arange(0, rf.NLON, f)
    g = rf.Grid(rf.LAT0, rf.LON0, rf.STEP * f, rows.size, cols.size)
    return g, (rows[:, None] * rf.NLON + cols[None, :]).reshape(-1)


def targets(f: int, land: np.ndarray, g: rf.Grid):
    """Land nodes of the full grid that the coarse grid dropped."""
    i, j = np.divmod(np.arange(rf.IMD.size), rf.NLON)
    lat, lon = rf.IMD.lats[i], rf.IMD.lons[j]
    sel = land & ~((i % f == 0) & (j % f == 0)) & (lat <= g.lats[-1]) & (lon <= g.lons[-1])
    midpoint = ((i % f) * 2 % f == 0) & ((j % f) * 2 % f == 0)  # in-cell position 0 or 1/2 on both axes
    return np.flatnonzero(sel), lat[sel], lon[sel], midpoint[sel]


def systematic_error(obs, pred):
    """Per target: relative error of the long-term mean (%), which does not average out over time."""
    m = obs.mean(axis=0)
    out = np.full(m.shape, np.nan)  # a node with no rain in the whole record has no relative error
    np.divide(np.abs(pred.mean(axis=0) - m), m, out=out, where=m > 0)
    return out * 100


def main() -> None:
    monthly, months, daily = [], [], {}
    for path in sorted(rf.IMD_DIR.glob("imd_rf25_*.nc")):
        dates, grid = rf.read_year(path)
        rain, _ = rf.monthly_totals(dates, grid)
        monthly.append(rain)
        months.append(pd.date_range(f"{dates[0].year}-01-01", periods=12, freq="MS"))
        if dates[0].year in DAILY_YEARS:
            daily[dates[0].year] = grid
    monthly = np.concatenate(monthly)
    months = months[0].append(months[1:])
    land = ~np.isnan(monthly[0])
    year, moy = months.year.to_numpy(), months.month.to_numpy()
    years = np.unique(year)
    fields = {
        "monthly": monthly,
        "JJAS season": np.stack([monthly[(year == y) & (moy >= 6) & (moy <= 9)].sum(axis=0) for y in years]),
        "annual": np.stack([monthly[year == y].sum(axis=0) for y in years]),
    }

    mid_sys = {}
    for f in (2, 4):
        g, keep = coarse_grid(f)
        tidx, tlat, tlon, midpoint = targets(f, land, g)
        probe = predict(monthly[:1, keep], tlat, tlon, g)
        ok = np.all([~np.isnan(p[0]) for name, p in probe.items() if not name.startswith("nearest")], axis=0)
        near_sea = np.isnan(probe["nearest (Week 1)"][0]) & ok
        ok &= ~near_sea
        spacing = rf.STEP * f
        print(f"\n=== nodes {spacing:g} deg apart ({spacing * 111:.0f} km): predicting {ok.sum():,} dropped land nodes "
              f"(+{near_sea.sum()} where nearest lands on sea and gives nothing) ===")

        rows = []
        for scale, field in fields.items():
            obs = field[:, tidx[ok]]
            for name, p in predict(field[:, keep], tlat[ok], tlon[ok], g).items():
                e = p - obs
                rows.append({"scale": scale, "method": name,
                             "MAE mm": np.abs(e).mean(),
                             "rel MAE %": np.abs(e).sum() / obs.sum() * 100,
                             "bias %": e.sum() / obs.sum() * 100})
                if scale == "annual":
                    s = systematic_error(obs, p)
                    rows[-1].update({"long-term mean err: median %": np.nanmedian(s), "p90 %": np.nanquantile(s, .9)})
                    mid_sys.setdefault(name, {})[f] = (np.nanmedian(s[midpoint[ok]]), np.nanmedian(s))
        print(pd.DataFrame(rows).set_index(["scale", "method"]).round(2).to_string())

        rows, feats = [], []
        for y, grid in daily.items():
            obs = grid[:, tidx[ok]]
            pred = predict(grid[:, keep], tlat[ok], tlon[ok], g)
            for name, p in pred.items():
                rows.append({"method": name,
                             "MAE mm/day": np.abs(p - obs).mean(),
                             "wet-day mismatch %": ((p > rf.WET_DAY_MM) != (obs > rf.WET_DAY_MM)).mean() * 100,
                             "wet-day count bias %": ((p > rf.WET_DAY_MM).sum() / (obs > rf.WET_DAY_MM).sum() - 1) * 100,
                             "max-day bias %": (p.max(axis=0).sum() / obs.max(axis=0).sum() - 1) * 100})

            # threshold and extreme features: computed on interpolated rain, or at nodes and then interpolated
            node_wet = np.where(np.isnan(grid[0]), np.nan, (grid > rf.WET_DAY_MM).sum(axis=0))
            node_max = grid.max(axis=0)
            obs_wet, obs_max = node_wet[tidx[ok]], node_max[tidx[ok]]
            candidates = {f"{name}: interpolate rain, then feature": ((p > rf.WET_DAY_MM).sum(axis=0), p.max(axis=0))
                          for name, p in pred.items() if name != "idw"}
            bil = rf.bilinear_stencil(tlat[ok], tlon[ok], g)
            candidates["bilinear: feature at nodes, then interpolate"] = (
                rf.sample(node_wet[keep], bil), rf.sample(node_max[keep], bil))
            for name, (wet, mx) in candidates.items():
                feats.append({"method": name,
                              "wet days/yr MAE": np.abs(wet - obs_wet).mean(),
                              "wet days bias %": (wet.sum() / obs_wet.sum() - 1) * 100,
                              "wettest day MAE mm": np.abs(mx - obs_max).mean(),
                              "wettest day bias %": (mx.sum() / obs_max.sum() - 1) * 100})
        print(f"daily, {min(daily)}-{max(daily)}:")
        print(pd.DataFrame(rows).groupby("method", sort=False).mean().round(2).to_string())
        print(f"annual wet-day count and wettest day, {min(daily)}-{max(daily)}:")
        print(pd.DataFrame(feats).groupby("method", sort=False).mean().round(2).to_string())

    print("\n=== extrapolated to the real 0.25 deg grid (median long-term mean error at a random point in a cell) ===")
    for name, d in mid_sys.items():
        (mid_05, _), (mid_10, all_10) = d[2], d[4]
        alpha = np.log2(mid_10 / mid_05)  # growth of error per doubling of node spacing
        at_025 = mid_05 / 2**alpha * (all_10 / mid_10)  # halve spacing, then from midpoints to all positions
        print(f"   {name:<18} midpoint error {mid_10:5.2f}% at 1.0 deg, {mid_05:5.2f}% at 0.5 deg "
              f"(x{2**alpha:.2f} per doubling)  ->  about {at_025:.2f}% at 0.25 deg")


if __name__ == "__main__":
    main()
