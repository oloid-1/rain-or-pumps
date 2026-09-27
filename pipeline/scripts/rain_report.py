"""The core analysis of the rain data: the IMD grid read at national, state and district scale.

Every number comes from the cleaned tables in data_cleaning/data and, for daily
extremes, the raw IMD files. District and state figures are area-weighted means
of the grid cells they cover (bridge_cell_district; scripts/rain_grid_mapping.py
shows the conversion step by step). National figures weight every land cell by
its area inside India's district outlines.

    1  the data and its quality
    2  national rain, year by year
    3  when it rains: months and the four reading seasons
    4  states
    5  districts: wettest, driest, most variable, trends
    6  heavy-rain days
    7  does one year's rain predict the next?
    8  the rain climate where the wells are
    9  how unusual is a +20% monsoon?
   10  zeros that are really missing data

"Normal" here is the 1998-2022 mean, the whole record, because this report
describes the climate. The season tables use 2000-2014 normals instead, so a
model never sees its test years in them.

Writes to data_cleaning/data/report/:
    rain_report.txt, rain_national_by_year.csv, rain_by_state.csv,
    rain_by_district.csv, rain_heavy_days_by_year.csv, rain_suspect_zeros.csv

Run from ml/ after scripts.rain_grid_mapping:  python -m scripts.rain_report
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bits_ml import grid as gr
from bits_ml import rainfall as rf
from bits_ml.config import CORE_DIR, IMD_DIR

REPORT_DIR = CORE_DIR / "report"
JJAS = (6, 7, 8, 9)
# IMD's own categories for a day at one place, mm
HEAVY_MM = {"heavy (64.5+)": 64.5, "very heavy (115.6+)": 115.6, "extremely heavy (204.5+)": 204.5}
pd.set_option("display.width", 170)
pd.set_option("display.max_rows", 60)


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def trend(frame: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    """Least-squares slope per group, in units per decade, with its p-value."""
    rows = []
    for key, g in frame.groupby(by):
        g = g.dropna(subset=[value])
        if len(g) < 15:                      # too few years left to call a trend
            rows.append((*((key,) if isinstance(key, str) else key), np.nan, np.nan))
            continue
        fit = stats.linregress(g["year"], g[value])
        rows.append((*((key,) if isinstance(key, str) else key), fit.slope * 10, fit.pvalue))
    return pd.DataFrame(rows, columns=[*by, f"{value}_trend_per_decade", "trend_p"])


def weighted(cells: pd.DataFrame, weights: pd.DataFrame, by: list[str], values: list[str]) -> pd.DataFrame:
    """Area-weighted mean of cell values over each group (district, state or nation).

    Cells with no value (the suspect zeros, now missing) are left out and the
    remaining weights renormalised, as `rain_panel.weighted_rain` does.
    """
    joined = weights.merge(cells, on="cell_id")
    ok = joined[values[0]].notna()
    joined["_w"] = joined["w"].where(ok, 0.0)
    for v in values:
        joined[v] = joined[v].where(ok) * joined["_w"]
    out = joined.groupby(by, as_index=False)[[*values, "_w"]].sum()
    for v in values:
        out[v] = out[v] / out["_w"].where(out["_w"] > 0)
    return out.drop(columns="_w")


def heavy_days() -> pd.DataFrame:
    """Cell-days above each IMD threshold, per year, over land cells, from the daily files."""
    rows = []
    for path in sorted(IMD_DIR.glob("imd_rf25_*.nc")):
        dates, daily = rf.read_year(path)
        land = daily[:, ~np.isnan(daily).all(axis=0)]
        row = {"year": dates[0].year, "wettest_day_mm": float(np.nanmax(land))}
        for name, mm in HEAVY_MM.items():
            row[name] = int((land >= mm).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    cells = pd.read_parquet(CORE_DIR / "fact_cell_month.parquet")
    seasons_cell = pd.read_parquet(CORE_DIR / "fact_cell_season.parquet")
    dim_season = pd.read_parquet(CORE_DIR / "dim_season.parquet")
    bridge = pd.read_parquet(CORE_DIR / "bridge_cell_district.parquet")
    dim_cell = pd.read_parquet(CORE_DIR / "dim_cell.parquet")

    cells["year"] = cells["month"].dt.year
    cells["month_no"] = cells["month"].dt.month
    # a year or monsoon with any missing month is missing, not short
    annual = cells.groupby(["cell_id", "year"], as_index=False).agg(
        annual_mm=("rain_mm", "sum"), wet_days=("wet_days", "sum"), n=("rain_mm", "count"))
    annual.loc[annual["n"] < 12, ["annual_mm", "wet_days"]] = np.nan
    jjas = cells[cells["month_no"].isin(JJAS)].groupby(["cell_id", "year"], as_index=False).agg(
        jjas_mm=("rain_mm", "sum"), n_jjas=("rain_mm", "count"))
    jjas.loc[jjas["n_jjas"] < len(JJAS), "jjas_mm"] = np.nan
    annual = annual.merge(jjas[["cell_id", "year", "jjas_mm"]], on=["cell_id", "year"]).drop(columns="n")

    w_district = bridge.rename(columns={"weight_in_district": "w"})[["cell_id", "district", "state", "w"]]
    by_state_area = bridge.groupby(["cell_id", "state"], as_index=False)["area_km2"].sum()
    w_state = by_state_area.assign(w=by_state_area["area_km2"] / by_state_area.groupby("state")["area_km2"].transform("sum"))
    cell_area = bridge.groupby("cell_id", as_index=False)["area_km2"].sum()
    w_nation = cell_area.assign(w=cell_area["area_km2"] / cell_area["area_km2"].sum(), nation="India")

    # ------------------------------------------------------------------ 1
    section(1, "the data and its quality")
    print(f"source: IMD 0.25 degree daily gridded rainfall, {cells['year'].min()}-{cells['year'].max()}, "
          f"{cells['cell_id'].nunique():,} land cells x {cells['month'].nunique()} months")
    print(f"missing values: {int(cells['rain_mm'].isna().sum())} | negative values: {int((cells['rain_mm'] < 0).sum())} "
          f"| every year complete (checked day by day in rainfall.read_year)")
    print(f"cells mapped to a district: {bridge['cell_id'].nunique():,} of {cells['cell_id'].nunique():,}; "
          f"districts with rain {bridge.groupby(['district', 'state']).ngroups} of 724; "
          f"states/UTs {bridge['state'].nunique()}")
    print(f"gridded land inside the outlines: {cell_area['area_km2'].sum() / 1e6:.2f} million km2")

    # ------------------------------------------------------------------ 2
    section(2, "national rain, year by year (area-weighted over India)")
    nation = weighted(annual, w_nation, ["nation", "year"], ["annual_mm", "jjas_mm", "wet_days"])
    normal_jjas, normal_annual = nation["jjas_mm"].mean(), nation["annual_mm"].mean()
    nation["jjas_pct_normal"] = 100 * nation["jjas_mm"] / normal_jjas
    nation["category"] = pd.cut(nation["jjas_pct_normal"], [0, 90, 110, 1000],
                                labels=["deficient (<90%)", "normal", "excess (>110%)"], right=False)
    print(nation[["year", "annual_mm", "jjas_mm", "jjas_pct_normal", "wet_days", "category"]].round(1).to_string(index=False))
    print(f"\nnormal annual {normal_annual:.0f} mm | normal June-September {normal_jjas:.0f} mm "
          f"({100 * normal_jjas / normal_annual:.0f}% of the year)")
    print(f"year-to-year variability (CV) of the monsoon: {nation['jjas_mm'].std() / normal_jjas * 100:.1f}%")
    deficient = nation.loc[nation["category"] == "deficient (<90%)", "year"].tolist()
    print(f"deficient monsoons: {deficient}  (IMD's own all-India record: 2002, 2004, 2009, 2014, 2015 were deficient)")
    t = stats.linregress(nation["year"], nation["jjas_mm"])
    print(f"monsoon trend {t.slope * 10:+.0f} mm per decade, p = {t.pvalue:.2f} "
          f"({'significant' if t.pvalue < 0.05 else 'not distinguishable from no trend in 25 years'})")
    nation.drop(columns="nation").to_csv(REPORT_DIR / "rain_national_by_year.csv", index=False)

    # ------------------------------------------------------------------ 3
    section(3, "when it rains")
    monthly = weighted(cells, w_nation, ["nation", "month_no", "year"], ["rain_mm"]).groupby("month_no")["rain_mm"].agg(["mean", "std"])
    monthly["share_of_year_%"] = 100 * monthly["mean"] / monthly["mean"].sum()
    monthly["cv_%"] = 100 * monthly["std"] / monthly["mean"]
    print("calendar months, national:")
    print(monthly.round(1).rename(columns={"mean": "mean_mm", "std": "sd_mm"}).to_string())

    sc = seasons_cell.merge(dim_season[["season_idx", "season", "season_year"]], on="season_idx")
    season_nation = weighted(sc.rename(columns={"season_year": "year"}), w_nation, ["nation", "season", "year"], ["rain_mm", "wet_days"])
    order = ["Nov-Jan", "Jan-May", "May-Aug", "Aug-Nov"]
    s = season_nation.groupby("season").agg(mean_mm=("rain_mm", "mean"), sd_mm=("rain_mm", "std"),
                                            wet_days=("wet_days", "mean")).reindex(order)
    s["share_of_year_%"] = 100 * s["mean_mm"] / s["mean_mm"].sum()
    s["cv_%"] = 100 * s["sd_mm"] / s["mean_mm"]
    print("\nthe four seasons between well readings (the 15th of Jan, May, Aug, Nov), national:")
    print(s.round(1).to_string())
    print("May-Aug and Aug-Nov together hold the monsoon; Nov-Jan and Jan-May are small and, relative to "
          "their size, the least reliable")

    # ------------------------------------------------------------------ 4
    section(4, "states (area-weighted over each state's cells)")
    st_year = weighted(annual, w_state[["cell_id", "state", "w"]], ["state", "year"], ["annual_mm", "jjas_mm", "wet_days"])
    st = st_year.groupby("state").agg(annual_mm=("annual_mm", "mean"), annual_sd=("annual_mm", "std"),
                                      jjas_mm=("jjas_mm", "mean"), wet_days=("wet_days", "mean"))
    st["cv_%"] = 100 * st["annual_sd"] / st["annual_mm"]
    st["monsoon_share_%"] = 100 * st["jjas_mm"] / st["annual_mm"]
    st = st.join(trend(st_year, ["state"], "annual_mm").set_index("state"))
    st["land_km2"] = by_state_area.groupby("state")["area_km2"].sum()
    st = st.sort_values("annual_mm", ascending=False)
    print(st[["annual_mm", "cv_%", "monsoon_share_%", "wet_days", "annual_mm_trend_per_decade", "trend_p", "land_km2"]]
          .round({"annual_mm": 0, "cv_%": 1, "monsoon_share_%": 0, "wet_days": 0,
                  "annual_mm_trend_per_decade": 0, "trend_p": 2, "land_km2": 0}).to_string())
    sig = st[st["trend_p"] < 0.05]
    print(f"\nstates with a trend at p < 0.05: {len(sig)} of {len(st)}"
          + (": " + ", ".join(f"{i} {r.annual_mm_trend_per_decade:+.0f} mm/decade" for i, r in sig.iterrows()) if len(sig) else ""))
    low_share = st[st["monsoon_share_%"] < 60].index.tolist()
    print(f"states where June-September is under 60% of the year: {', '.join(low_share)} "
          "(north-east monsoon or western disturbances matter there)")
    st.round(3).to_csv(REPORT_DIR / "rain_by_state.csv")

    # ------------------------------------------------------------------ 5
    section(5, "districts")
    d_year = weighted(annual, w_district, ["district", "state", "year"], ["annual_mm", "jjas_mm", "wet_days"])
    d = d_year.groupby(["district", "state"]).agg(annual_mm=("annual_mm", "mean"), annual_sd=("annual_mm", "std"),
                                                 jjas_mm=("jjas_mm", "mean"), jjas_sd=("jjas_mm", "std"),
                                                 wet_days=("wet_days", "mean"))
    d["cv_%"] = 100 * d["annual_sd"] / d["annual_mm"]
    d["jjas_cv_%"] = 100 * d["jjas_sd"] / d["jjas_mm"]
    d = d.join(trend(d_year, ["district", "state"], "annual_mm").set_index(["district", "state"]))
    whole = cells.groupby(["cell_id", "year"])["suspect_zero"].all()
    zero_years = whole[whole].reset_index()[["cell_id", "year"]]
    touched = w_district[w_district["cell_id"].isin(zero_years["cell_id"])].groupby(["district", "state"])["w"].sum()
    d["suspect_zero_weight"] = touched.reindex(d.index).fillna(0)
    d["suspect"] = np.where(d["suspect_zero_weight"] > 0.05, "*", "")
    print("districts marked * take over 5% of their area from cells with whole years missing (section 10);\n"
          "those years now come from the district's other cells or are missing, so their figures rest on\n"
          "less data; trends below leave them out\n")
    q = d["annual_mm"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"{len(d)} districts | annual rain 5th {q[0.05]:.0f}, 25th {q[0.25]:.0f}, median {q[0.5]:.0f}, "
          f"75th {q[0.75]:.0f}, 95th {q[0.95]:.0f} mm | a {q[0.95] / q[0.05]:.0f}-fold range")
    show = ["annual_mm", "cv_%", "wet_days", "suspect"]
    print("\nwettest ten:\n" + d.nlargest(10, "annual_mm")[show].round(1).to_string())
    print("\ndriest ten:\n" + d.nsmallest(10, "annual_mm")[show].round(1).to_string())
    print("\nmost variable ten (annual CV):\n" + d.nlargest(10, "cv_%")[show].round(1).to_string())
    clean = d[d["suspect"] == ""]
    print("\nmost variable ten, unmarked districts only:\n" + clean.nlargest(10, "cv_%")[show[:-1]].round(1).to_string())
    rho = stats.spearmanr(clean["annual_mm"], clean["cv_%"]).statistic
    print(f"\ndrier districts are less reliable: rank correlation of rain and its CV = {rho:+.2f}")
    ok = d["suspect"] == ""
    up = ok & (d["trend_p"] < 0.05) & (d["annual_mm_trend_per_decade"] > 0)
    down = ok & (d["trend_p"] < 0.05) & (d["annual_mm_trend_per_decade"] < 0)
    print(f"annual trend at p < 0.05, unmarked districts: {int(up.sum())} wetter, {int(down.sum())} drier, of {int(ok.sum())} "
          f"(about {0.05 * ok.sum():.0f} would pass by chance alone; neighbouring districts share weather, so these are not independent)")
    if up.any():
        print("  strongest wetting: " + ", ".join(f"{a} ({b}) {v:+.0f}" for (a, b), v in
                                                d[up]["annual_mm_trend_per_decade"].nlargest(5).items()) + " mm/decade")
    if down.any():
        print("  strongest drying:  " + ", ".join(f"{a} ({b}) {v:+.0f}" for (a, b), v in
                                                d[down]["annual_mm_trend_per_decade"].nsmallest(5).items()) + " mm/decade")
    d.round(3).to_csv(REPORT_DIR / "rain_by_district.csv")

    # ------------------------------------------------------------------ 6
    section(6, "heavy-rain days (IMD categories, counted per 0.25 degree cell per day)")
    heavy = heavy_days()
    print(heavy.round(1).to_string(index=False))
    for name in HEAVY_MM:
        fit = stats.linregress(heavy["year"], heavy[name])
        print(f"{name:<26} mean {heavy[name].mean():8.0f} cell-days a year, trend {fit.slope * 10 / heavy[name].mean() * 100:+.0f}% "
              f"per decade (p = {fit.pvalue:.2f})")
    heavy.to_csv(REPORT_DIR / "rain_heavy_days_by_year.csv", index=False)

    # ------------------------------------------------------------------ 7
    section(7, "does one year's rain predict the next?")
    d_sorted = d_year.sort_values(["district", "state", "year"])
    d_sorted["jjas_next"] = d_sorted.groupby(["district", "state"])["jjas_mm"].shift(-1)
    per = d_sorted.dropna().groupby(["district", "state"]).apply(
        lambda g: g["jjas_mm"].corr(g["jjas_next"]), include_groups=False)
    print(f"correlation of a district's monsoon with the next year's, across {len(per)} districts: "
          f"median {per.median():+.2f}, middle half {per.quantile(0.25):+.2f} to {per.quantile(0.75):+.2f}")
    print(f"national: {nation['jjas_mm'].autocorr():+.2f}")
    print("near zero: next year's monsoon cannot be forecast from this year's. The project therefore "
          "describes rain (normals, variability, scenarios) and does not forecast it")

    # ------------------------------------------------------------------ 8
    section(8, "the rain climate where the wells are")
    wells_in = dim_cell.set_index("cell_id")["n_wells_modelling"]
    cell_clim = annual.groupby("cell_id").agg(annual_mm=("annual_mm", "mean"), sd=("annual_mm", "std"))
    cell_clim["cv_%"] = 100 * cell_clim["sd"] / cell_clim["annual_mm"]
    cell_clim["has_wells"] = wells_in.reindex(cell_clim.index).fillna(0) > 0
    cell_clim = cell_clim.join(cell_area.set_index("cell_id")["area_km2"])
    for flag, label in ((True, "cells with modelling wells"), (False, "cells without")):
        g = cell_clim[cell_clim["has_wells"] == flag]
        print(f"{label:<28} {len(g):>5,} cells | median annual {g['annual_mm'].median():5.0f} mm | median CV {g['cv_%'].median():4.1f}%")
    bins = pd.cut(cell_clim["annual_mm"], [0, 500, 750, 1000, 1500, 2500, 20000])
    cover = cell_clim.groupby(bins, observed=True).agg(cells=("has_wells", "size"), with_wells=("has_wells", "sum"))
    cover["share_with_wells_%"] = (100 * cover["with_wells"] / cover["cells"]).round(0)
    print("\nshare of cells holding a modelling well, by how wet the cell is:")
    print(cover.to_string())
    print("wells thin out at both ends: the desert and the wettest hills are the least monitored, "
          "so findings there rest on few wells")

    # ------------------------------------------------------------------ 9
    section(9, "how unusual is a +20% monsoon?")
    d_year = d_year.merge(d[["jjas_mm"]].rename(columns={"jjas_mm": "jjas_normal"}), left_on=["district", "state"], right_index=True)
    ratio = d_year["jjas_mm"] / d_year["jjas_normal"]
    print(f"district-years with a monsoon at least 20% above that district's normal: {(ratio >= 1.2).mean():.1%}")
    print(f"at least 20% below: {(ratio <= 0.8).mean():.1%}")
    by_d = ratio.groupby([d_year["district"], d_year["state"]]).apply(lambda r: (r >= 1.2).mean())
    print(f"per district, a +20% monsoon happened in a median {by_d.median():.0%} of years "
          f"(range {by_d.min():.0%}-{by_d.max():.0%})")
    print("so +20% is a real but uncommon year, about one in "
          f"{1 / max((ratio >= 1.2).mean(), 1e-9):.0f}: a fair stress test for the what-if simulator, not a routine one")

    # ------------------------------------------------------------------ 10
    section(10, "zeros that are really missing data")
    print("IMD marks sea and foreign land with -999, which the pipeline reads as missing. But some land cells\n"
          "report exactly 0.0 mm on every day of a year, or of a monsoon month, where rain is certain: a cell\n"
          "with no rain gauge near it, written as zero instead of as missing.")
    print(f"\ncell-years with exactly 0 mm all year: {len(zero_years):,} in {zero_years['cell_id'].nunique()} of "
          f"{annual['cell_id'].nunique():,} land cells")
    print("by year: " + ", ".join(f"{y} {n}" for y, n in zero_years.groupby("year").size().items()))
    in_zero_year = cells.set_index(["cell_id", "year"]).index.isin(zero_years.set_index(["cell_id", "year"]).index)
    zero_months = cells[cells["suspect_zero"] & ~in_zero_year]
    print(f"further monsoon cell-months with exactly 0 mm where that month's median is over 100 mm: "
          f"{len(zero_months):,} in {zero_months['cell_id'].nunique()} cells")
    print(f"flagged in total (fact_cell_month.suspect_zero): {int(cells['suspect_zero'].sum()):,} cell-months")
    zc = dim_cell.set_index("cell_id").loc[zero_years["cell_id"].unique()]
    states = bridge[bridge["cell_id"].isin(zc.index)].drop_duplicates("cell_id")["state"].value_counts()
    print("cells by state: " + ", ".join(f"{k} {v}" for k, v in states.items()) + " (the edges of the grid)")
    print(f"modelling wells standing in these cells: {int(zc['n_wells_modelling'].sum())} "
          f"(in {int((zc['n_wells_modelling'] > 0).sum())} cells)")
    hit = d[d["suspect_zero_weight"] > 0].sort_values("suspect_zero_weight", ascending=False)
    print(f"\ndistricts drawing rain from these cells: {len(hit)} (column 1 = share of the district's area in them):")
    print(hit[["suspect_zero_weight", "annual_mm", "cv_%", "annual_mm_trend_per_decade"]].round(2).to_string())
    print("\ncorrected in the tables: rain_panel flags these cell-months `suspect_zero` and sets them missing;\n"
          "seasons blanks their days, so a season touching one is incomplete rather than dry; and every average\n"
          "over cells (well, district) uses only cells with data, with `rain_cover` giving the share of weight\n"
          "that had it. The figures in this report are computed the same way.")
    zero_years.merge(dim_cell[["cell_id", "cell_lat", "cell_lon", "district", "state"]], on="cell_id").to_csv(
        REPORT_DIR / "rain_suspect_zeros.csv", index=False)

    print(f"\nwrote rain_report.txt and CSVs to {REPORT_DIR}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run()
    text = buffer.getvalue()
    (REPORT_DIR / "rain_report.txt").write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
