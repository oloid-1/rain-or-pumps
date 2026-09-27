"""Does reading rainfall more precisely at each well explain its water level better?

For each well and year the monsoon recharge signal is the rise in water level
from the May campaign to the November campaign (May depth minus November depth,
metres). Rain that fell in between (June to October) is read at the well by
each method. Two checks, on CGWB and IMD data only:

1. within a well, across years: correlation of June-October rain with the rise
2. across nearby wells: do wells that get more rain than their neighbours also
   rise more than their neighbours (long-term means, compared within 1 degree
   blocks so regional climate and geology largely cancel)

Run from ml/:  python -m scripts.gw_response_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from bits_ml import rainfall as rf
from scripts.interpolation_check import bicubic, idw_stencil

YEARS = np.arange(2000, 2023)
MIN_YEARS = 15
MIN_BLOCK_WELLS = 3
BOOTSTRAP = 1000


def main() -> None:
    wells = rf.load_wells()
    lat, lon = wells["Latitude"].to_numpy(float), wells["Longitude"].to_numpy(float)

    def campaign(mon):
        cols = [f"{mon}-{y % 100:02d}" for y in YEARS]
        return wells[cols].apply(pd.to_numeric, errors="coerce").to_numpy().T  # (years, wells)

    rise = campaign("May") - campaign("Nov")
    has_rise = ~np.isnan(rise)

    monsoon = []
    for y in YEARS:
        dates, grid = rf.read_year(rf.IMD_DIR / f"imd_rf25_{y}.nc")
        rain, _ = rf.monthly_totals(dates, grid)
        monsoon.append(rain[5:10].sum(axis=0))  # June to October; sea nodes stay NaN
    monsoon = np.stack(monsoon)

    methods = {
        "nearest (Week 1)": rf.sample(monsoon, rf.nearest_stencil(lat, lon)),
        "bilinear": rf.sample(monsoon, rf.bilinear_stencil(lat, lon)),
        "bilinear + rounding box": rf.sample(
            monsoon, rf.box_stencil(lat, lon, rf.coordinate_half_width(lat, lon))
        ),
        "bicubic": bicubic(monsoon, lat, lon, rf.IMD),
        "idw": rf.sample(monsoon, idw_stencil(lat, lon, rf.IMD)),
    }
    base = "nearest (Week 1)"

    # 1 within a well, across years ---------------------------------------------------------
    r_by = {}
    for name, rain in methods.items():
        r = np.full(len(wells), np.nan)
        for k in range(len(wells)):
            m = has_rise[:, k] & ~np.isnan(rain[:, k])
            if m.sum() >= MIN_YEARS and rain[m, k].std() > 0 and rise[m, k].std() > 0:
                r[k] = np.corrcoef(rain[m, k], rise[m, k])[0, 1]
        r_by[name] = r
    common = np.all([~np.isnan(v) for v in r_by.values()], axis=0)
    rows = []
    for name, r in r_by.items():
        d = r[common] - r_by[base][common]
        rows.append({
            "method": name,
            "median r": np.median(r[common]),
            "mean r": r[common].mean(),
            "wells r>0 %": (r[common] > 0).mean() * 100,
            "better than nearest %": (d > 1e-9).mean() * 100,
            "worse %": (d < -1e-9).mean() * 100,
            "mean change in r": d.mean(),
            "p (Wilcoxon)": np.nan if name == base or not np.any(d) else stats.wilcoxon(d[d != 0]).pvalue,
        })
    print(f"\n1 · Within each well, across years: Jun-Oct rain vs May->Nov rise "
          f"({common.sum():,} wells with >= {MIN_YEARS} years)")
    print(pd.DataFrame(rows).set_index("method").round(4).to_string())

    # 2 across neighbouring wells -----------------------------------------------------------
    def mean_over_rise_years(x):
        use = np.where(has_rise, x, np.nan)
        out = np.full(x.shape[1], np.nan)
        some = ~np.isnan(use).all(axis=0)
        out[some] = np.nanmean(use[:, some], axis=0)
        return out

    enough = has_rise.sum(axis=0) >= MIN_YEARS
    mean_rise = mean_over_rise_years(rise)
    sy = wells["Reference_Sy"].to_numpy(float)
    block = np.floor(lat).astype(int) * 1000 + np.floor(lon).astype(int)
    frame = pd.DataFrame({"block": block, "rise": mean_rise, "recharge": mean_rise * sy})
    for name, rain in methods.items():
        frame[name] = mean_over_rise_years(rain)
    frame = frame[enough & np.all([~np.isnan(frame[n]) for n in methods], axis=0)]
    frame = frame[frame.groupby("block")["rise"].transform("size") >= MIN_BLOCK_WELLS]
    cols = ["rise", "recharge", *methods]
    dev = frame[cols] - frame.groupby("block")[cols].transform("mean")
    dev["block"] = frame["block"]

    def corr(d, x, y):
        return stats.spearmanr(d[x], d[y]).statistic

    rng = np.random.default_rng(0)
    blocks = dev["block"].unique()
    groups = {b: g for b, g in dev.groupby("block")}
    boot = [pd.concat([groups[b] for b in rng.choice(blocks, blocks.size)]) for _ in range(BOOTSTRAP)]
    rows = []
    for name in methods:
        row = {"method": name,
               "rain spread within block (median |dev| mm)": np.median(np.abs(dev[name]))}
        for target in ("rise", "recharge"):
            row[f"rho vs {target}"] = corr(dev, name, target)
            diffs = np.array([corr(b, name, target) - corr(b, base, target) for b in boot])
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            row[f"{target}: change vs nearest (95% CI)"] = f"{diffs.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    print(f"\n2 · Across neighbouring wells: long-term mean rain vs mean rise, within 1 deg blocks "
          f"({len(dev):,} wells in {blocks.size} blocks with >= {MIN_BLOCK_WELLS} wells)")
    print(pd.DataFrame(rows).set_index("method").round(3).to_string())


if __name__ == "__main__":
    main()
