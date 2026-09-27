"""Evidence for the rainfall-at-well extraction.

1. parity with the Week-1 notebook's reader
2. what moving from nearest node to masked bilinear changes
3. how much rainfall uncertainty the coordinate rounding itself implies
4. how far coastal stencils lean on renormalisation
5. what the mislabelled Anantapur wells do to a district mean

Run from ml/, after `python -m bits_ml.rainfall`:  python -m scripts.rainfall_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bits_ml import rainfall as rf
from bits_ml.config import DATA_PROCESSED


def rounded_to(v, per_degree, tol):
    x = v * per_degree
    return np.abs(x - np.round(x)) < tol


def main() -> None:
    wells = rf.load_wells()
    lat, lon = wells["Latitude"].to_numpy(), wells["Longitude"].to_numpy()
    uid = wells["well_uid"].to_numpy().astype(str)
    near_st, bil_st = rf.nearest_stencil(lat, lon), rf.bilinear_stencil(lat, lon)

    near_frames, annual_grids, land_share = [], [], None
    for path in sorted(rf.IMD_DIR.glob("imd_rf25_*.nc")):
        dates, grid = rf.read_year(path)
        if land_share is None:
            land_share = rf.land_weight(grid[0], bil_st)
        rain, wet = rf.monthly_totals(dates, rf.sample(grid, near_st))
        ym = pd.date_range(f"{dates[0].year}-01-01", periods=12, freq="MS").strftime("%Y-%m")
        near_frames.append(pd.DataFrame({
            "well_uid": np.tile(uid, 12), "ym": np.repeat(ym, len(uid)),
            "rain_mm": rain.reshape(-1), "wet_days": wet.reshape(-1),
        }))
        annual_grids.append(grid.sum(axis=0))  # NaN on non-land cells
    near = pd.concat(near_frames, ignore_index=True)
    mean_grid = np.stack(annual_grids).mean(axis=0)

    # 1 -----------------------------------------------------------------------
    print("\n1 · Reader parity with the Week-1 notebook (nearest node, per grid cell)")
    cells = pd.read_parquet(DATA_PROCESSED / "cell_monthly_rain_1998_2022.parquet")
    parts = cells["cell_id"].str.split("_")
    cells["node"] = (
        np.round((parts.str[0].astype(float) - rf.LAT0) / rf.STEP).astype(int) * rf.NLON
        + np.round((parts.str[1].astype(float) - rf.LON0) / rf.STEP).astype(int)
    )
    cells["ym"] = pd.to_datetime(cells["month"]).dt.strftime("%Y-%m")
    node_of = pd.Series(near_st.idx[:, 0], index=uid)
    mine = near.assign(node=near["well_uid"].map(node_of)).drop_duplicates(["node", "ym"])
    m = mine.merge(cells[["node", "ym", "rain_mm", "wet_days"]], on=["node", "ym"], suffixes=("", "_nb"))
    land = m["rain_mm"].notna()
    print(f"   cell-months matched          {len(m):,} (notebook table has {len(cells):,})")
    print(f"   land: max |Δ rain|           {(m.loc[land, 'rain_mm'] - m.loc[land, 'rain_mm_nb']).abs().max():.4f} mm")
    print(f"   land: max |Δ wet days|       {(m.loc[land, 'wet_days'] - m.loc[land, 'wet_days_nb']).abs().max():.0f}")
    sea = m.loc[~land]
    print(f"   sea cells                    {sea['node'].nunique()} cells, {len(sea):,} cell-months; "
          f"notebook stored rain = {sorted(sea['rain_mm_nb'].unique().tolist())[:3]}, extractor stores NaN")

    # 2 -----------------------------------------------------------------------
    print("\n2 · Nearest node → masked bilinear, mean annual rainfall 1998–2022")
    bil_path = sorted(DATA_PROCESSED.glob("well_monthly_rain_*.parquet"))[-1]
    bil = pd.read_parquet(bil_path)
    bil["year"] = pd.to_datetime(bil["month"]).dt.year
    near["year"] = near["ym"].str[:4].astype(int)
    bil_y = bil.groupby(["well_uid", "year"])["rain_mm"].sum(min_count=12).unstack().loc[uid]
    near_y = near.groupby(["well_uid", "year"])["rain_mm"].sum(min_count=12).unstack().loc[uid]
    mean_near, mean_bil = near_y.mean(axis=1), bil_y.mean(axis=1)
    ok = mean_near.notna().to_numpy()
    rel = ((mean_bil - mean_near) / mean_near * 100).to_numpy()
    r = np.abs(rel[ok])
    print(f"   wells compared {ok.sum():,}:  median |Δ| {np.median(r):.1f}%   p90 {np.quantile(r, .9):.1f}%   "
          f"p99 {np.quantile(r, .99):.1f}%   max {r.max():.1f}%")
    print(f"   wells shifting >10%: {(r > 10).sum()}   >25%: {(r > 25).sum()}   >50%: {(r > 50).sum()}")
    yearly_rel = ((bil_y - near_y) / near_y * 100).to_numpy()[ok]
    sign_stable = (np.sign(yearly_rel) == np.sign(rel[ok])[:, None]).mean(axis=1)
    big = r > 10
    print(f"   for wells shifting >10%, the shift keeps the same sign in a median "
          f"{np.median(sign_stable[big]) * 100:.0f}% of years: a systematic bias, not noise")

    w = wells.assign(near=mean_near.to_numpy(), bil=mean_bil.to_numpy(), rel=rel)
    top = w.loc[ok].reindex(w.loc[ok, "rel"].abs().sort_values(ascending=False).index).head(8)
    print("   largest well-level shifts:")
    for _, t in top.iterrows():
        print(f"      {t['District']:<18} {t['State']:<16} {t['near']:7.0f} → {t['bil']:7.0f} mm  ({t['rel']:+.0f}%)")
    print("   formerly sea-snapped wells (nearest node gave no rain):")
    for _, t in w.loc[~ok].iterrows():
        print(f"      {t['District']:<18} {t['State']:<16}    none → {t['bil']:7.0f} mm")

    # district means as the Week-1 pipeline actually computed them: sea wells counted as 0 mm
    w["near_pipeline"] = w["near"].fillna(0.0)
    d = w.groupby(["State", "District"])[["near_pipeline", "bil"]].mean()
    d["rel"] = (d["bil"] - d["near_pipeline"]) / d["near_pipeline"].where(d["near_pipeline"] > 0) * 100
    print("   district well-means shifting most (before = Week-1 pipeline, sea wells as 0 mm):")
    order = d["rel"].abs().fillna(np.inf).sort_values(ascending=False).index
    for (s, dist), t in d.reindex(order).head(10).iterrows():
        change = "was 0 mm" if np.isnan(t["rel"]) else f"{t['rel']:+.1f}%"
        print(f"      {dist:<18} {s:<16} {t['near_pipeline']:7.0f} → {t['bil']:7.0f} mm  ({change})")

    # 3 -----------------------------------------------------------------------
    print("\n3 · Rainfall uncertainty implied by coordinate rounding (mean annual grid, both axes)")

    def both(per_degree, tol):
        return rounded_to(lat, per_degree, tol) & rounded_to(lon, per_degree, tol)

    c01 = both(10, 0.0005)
    c005 = both(20, 0.001) & ~c01
    cmin = both(60, 0.004) & ~c01 & ~c005
    c001 = both(100, 0.002) & ~(c01 | c005 | cmin)
    fine = ~(c01 | c005 | cmin | c001)
    classes = [
        ("0.1° grid (±0.05°, ~5.6 km)", c01, 0.05),
        ("0.05° grid (±0.025°, ~2.8 km)", c005, 0.025),
        ("whole arcminute (±0.5′, ~0.9 km)", cmin, 1 / 120),
        ("0.01° (±0.005°, ~0.6 km)", c001, 0.005),
        ("finer (±0.00005°, ~6 m)", fine, 0.00005),
    ]

    def at(la, lo):
        return rf.sample(mean_grid, rf.bilinear_stencil(la, lo))

    for name, mask, h in classes:
        if not mask.any():
            continue
        la, lo = lat[mask], lon[mask]
        base = at(la, lo)
        corners = np.stack([at(la + sa * h, lo + so * h) for sa in (1, -1) for so in (1, -1)])
        spread = (corners.max(axis=0) - corners.min(axis=0)) / base * 100
        print(f"   {name:<34} {mask.sum():5d} wells   rainfall spread median {np.median(spread):5.2f}%   "
              f"p90 {np.quantile(spread, .9):5.2f}%   max {spread.max():5.2f}%")

    # 4 -----------------------------------------------------------------------
    partial = land_share < 1
    print(f"\n4 · Coastal stencils: {partial.sum()} wells have part of their bilinear weight on sea nodes")
    print(f"   land share of weight: min {land_share[partial].min():.2f}   median {np.median(land_share[partial]):.2f}")

    # 5 -----------------------------------------------------------------------
    print("\n5 · Mislabelled wells inside Anantapur, Andhra Pradesh")
    ana = w[(w["State"] == "Andhra pradesh") & (w["District"] == "Anantapur")]
    far = (ana["Latitude"] > 20) | (ana["Longitude"] < 74)
    for _, t in ana[far].iterrows():
        print(f"      {t['Station Name']:<12} {t['Latitude']:.2f}, {t['Longitude']:.2f}   {t['bil']:6.0f} mm/yr")
    print(f"   Anantapur well-mean rainfall: all {len(ana)} wells {ana['bil'].mean():.0f} mm   "
          f"true {int((~far).sum())} wells {ana.loc[~far, 'bil'].mean():.0f} mm   "
          f"({(ana['bil'].mean() / ana.loc[~far, 'bil'].mean() - 1) * 100:+.0f}%)")


if __name__ == "__main__":
    main()
