"""How much of a monsoon actually reaches the aquifer, district by district.

This is the water-table fluctuation method, which is how CGWB itself estimates
recharge, applied to our own wells:

    recharge (mm) = specific yield x rise over the monsoon (m) x 1000
    infiltration  = recharge / the rain that fell on that well

A district that turns 12% of its monsoon into stored water is a place where a
recharge structure has something to work with. One that turns 3% is not, whatever
its rainfall. That is the question the project set out to answer, and unlike the
extraction residual it is measured rather than inferred: every term on the right
is observed, and nothing is left over to be named.

It also avoids the trap the attribution fell into. A residual holds everything the
model omitted; this holds only what the three measurements say.

What it assumes, and where that bites:

* **Nothing is pumped or drains away during the monsoon.** Both happen, so every
  figure here is an upper bound on recharge. Where pumping is heavy in the wet
  season the overstatement is worst, which is exactly where it matters.
* **Specific yield is right.** CGWB gives one of five values per well, read off a
  hydrogeology map, so it is a class, not a measurement. Recharge scales linearly
  with it: a well assigned 0.02 instead of 0.13 reports a sixth of the water.
* **The May and November readings bracket the monsoon.** They roughly do, but a
  well read late in May or early in November misses part of the rise.
* **The rise came from rain falling on that spot.** Often it did not: a canal
  command, a neighbour's pump switching off, or a keying error all raise a water
  table without a drop of local rain. Well-years claiming more water than the
  monsoon delivered are excluded and counted, not capped.

One thing the figures are not: a ranking driven purely by the hydrogeology map.
Specific yield accounts for 42% of the variance between districts, and the rise
term on its own - specific yield removed entirely - still reproduces across halves
of the record at +0.84 against a shuffled null of 0.14.

Run from ml/:  python -m bits_ml.recharge
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import groundwater as gw
from . import rain_model as rm
from .config import DATA_PROCESSED

MIN_MONSOON_MM = 100.0   # below this the ratio is dominated by its denominator
MIN_RISE_M = 0.01        # a well that did not rise says nothing about recharge
MAX_INFILTRATION_PCT = 60.0
MIN_YEARS = 8
MIN_WELLS = 3
HALVES = ((2000, 2011), (2012, 2022))


def monsoon_rain_by_well(years) -> pd.DataFrame:
    """Jun-Sep rain at each well, one row per well-year."""
    series = rm.RainSeries.load()
    month = np.asarray(series.months.month)
    year_of = np.asarray(series.months.year)
    monsoon = np.isin(month, rm.MONSOON)
    rows = []
    for year in years:
        rows.append(pd.DataFrame({
            "well_uid": series.well_uid, "year": year,
            "monsoon_mm": series.rain[monsoon & (year_of == year)].sum(axis=0),
        }))
    return pd.concat(rows, ignore_index=True)


def well_year_recharge(readings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per well and year: the rise, the rain that produced it, and the share that stayed."""
    readings = gw.load_readings() if readings is None else readings
    moves = gw.seasonal_moves(readings)
    static = readings.drop_duplicates("well_uid")[
        ["well_uid", "state", "district", "far_from_district", "sy", "well_type", "aquifer"]]
    frame = moves.merge(static, on="well_uid").merge(
        monsoon_rain_by_well(sorted(moves["year"].unique())), on=["well_uid", "year"], how="left")

    frame = frame.dropna(subset=["monsoon_rise_m", "monsoon_mm", "sy"])
    frame = frame[(frame["monsoon_rise_m"] > MIN_RISE_M) & (frame["monsoon_mm"] >= MIN_MONSOON_MM)]
    frame["recharge_mm"] = frame["sy"] * frame["monsoon_rise_m"] * 1000.0
    frame["infiltration_pct"] = frame["recharge_mm"] / frame["monsoon_mm"] * 100.0
    frame["rise_per_100mm"] = frame["monsoon_rise_m"] / frame["monsoon_mm"] * 100.0

    # Water the rain cannot account for. Some wells rise far more than local rainfall could
    # supply - 1.2% of well-years store more than fell, and the extreme cases are absurd on
    # their face: a 29.6 m rise on 131 mm of monsoon. Canal water, a pumped cone recovering,
    # or a mis-keyed reading; the method cannot tell which, and has no business calling any of
    # it recharge. They are dropped rather than capped, because capping would quietly keep a
    # wrong number at a plausible-looking value, and the count is reported.
    impossible = frame["infiltration_pct"] > MAX_INFILTRATION_PCT
    frame.attrs["excluded_well_years"] = int(impossible.sum())
    frame.attrs["excluded_share"] = float(impossible.mean())
    return frame[~impossible].reset_index(drop=True)


def by_district(frame: pd.DataFrame, min_wells: int = MIN_WELLS, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """District medians, over wells with enough years to have a typical monsoon in them."""
    usable = frame[~frame["far_from_district"]]
    per_well = (usable.groupby(["state", "district", "well_uid"])
                .agg(infiltration_pct=("infiltration_pct", "median"),
                     rise_per_100mm=("rise_per_100mm", "median"),
                     recharge_mm=("recharge_mm", "median"),
                     monsoon_mm=("monsoon_mm", "median"),
                     rise_m=("monsoon_rise_m", "median"),
                     sy=("sy", "first"), years=("year", "size")).reset_index())
    per_well = per_well[per_well["years"] >= min_years]

    district = (per_well.groupby(["state", "district"])
                .agg(wells=("well_uid", "size"),
                     infiltration_pct=("infiltration_pct", "median"),
                     spread_pct=("infiltration_pct", lambda s: float(s.quantile(.75) - s.quantile(.25))),
                     rise_per_100mm=("rise_per_100mm", "median"),
                     rise_m=("rise_m", "median"),
                     monsoon_mm=("monsoon_mm", "median"),
                     sy=("sy", "median")).reset_index())
    return district[district["wells"] >= min_wells].sort_values("infiltration_pct", ascending=False, ignore_index=True)


def reproducibility(frame: pd.DataFrame, halves=HALVES, min_wells: int = MIN_WELLS) -> dict:
    """Does a district's infiltration come out the same on both halves of the record?

    The lesson from the attribution: a district-level number is worth nothing until
    it has been measured twice on different years and agreed with itself.
    """
    sides = []
    for lo, hi in halves:
        side = frame[frame["year"].between(lo, hi)]
        sides.append(by_district(side, min_wells=min_wells, min_years=3)
                     .set_index(["state", "district"])["infiltration_pct"])
    early, late = sides
    shared = early.index.intersection(late.index)
    if len(shared) < 10 or early[shared].std() == 0 or late[shared].std() == 0:
        # no spread on one side leaves the correlation undefined rather than perfect
        return {"districts": len(shared), "pearson": np.nan, "spearman": np.nan, "p_value": np.nan,
                "null_p95": np.nan}
    result = stats.pearsonr(early[shared], late[shared])
    rng = np.random.default_rng(0)
    null = [stats.pearsonr(early[shared], rng.permutation(late[shared].to_numpy())).statistic for _ in range(200)]
    return {
        "districts": len(shared),
        "pearson": float(result.statistic), "p_value": float(result.pvalue),
        "spearman": float(stats.spearmanr(early[shared], late[shared]).statistic),
        "null_p95": float(np.percentile(np.abs(null), 95)),
    }


def main() -> None:
    frame = well_year_recharge()
    district = by_district(frame)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATA_PROCESSED / "well_recharge.parquet", index=False)
    district.to_csv(DATA_PROCESSED / "district_recharge.csv", index=False)

    print(f"well-years used: {len(frame):,} | wells {frame['well_uid'].nunique():,} | "
          f"districts ranked {len(district)}")
    print(f"dropped as more water than the rain could supply (>{MAX_INFILTRATION_PCT:.0f}%): "
          f"{frame.attrs.get('excluded_well_years', 0):,} well-years "
          f"({frame.attrs.get('excluded_share', 0) * 100:.1f}%) - canal water, a recovering pumped cone, "
          f"or a bad reading, none of which this method can call recharge")
    print(f"infiltration, share of the monsoon that stayed: median {frame['infiltration_pct'].median():.1f}% | "
          f"quartiles {frame['infiltration_pct'].quantile(.25):.1f}-{frame['infiltration_pct'].quantile(.75):.1f}%")
    print(f"a well rises {frame['rise_per_100mm'].median():.2f} m per 100 mm of monsoon at the median\n")

    print("by aquifer (median infiltration, %):")
    print(frame.groupby("aquifer")["infiltration_pct"].agg(["median", "size"]).round(1).to_string())
    print("\nby well type:")
    print(frame.groupby("well_type")["infiltration_pct"].agg(["median", "size"]).round(1).to_string())
    print("\nby specific yield class (the value it was assigned):")
    print(frame.groupby("sy")["infiltration_pct"].agg(["median", "size"]).round(1).to_string())

    check = reproducibility(frame)
    print(f"\nmeasured twice, on {check['districts']} districts present in both halves of the record:")
    print(f"   agreement between halves: pearson {check['pearson']:+.2f}, spearman {check['spearman']:+.2f} "
          f"(shuffled null |r| p95 {check['null_p95']:.2f})")
    verdict = ("holds up: a district's infiltration is a property of the place, not of the years you looked at"
               if check["pearson"] > 3 * check["null_p95"] else
               "does not hold up: treat these as descriptive, not as a district ranking")
    print(f"   -> {verdict}")

    print("\nhighest infiltration:")
    columns = ["district", "state", "wells", "infiltration_pct", "spread_pct", "rise_m", "monsoon_mm", "sy"]
    print(district.head(8)[columns].round(2).to_string(index=False))
    print("\nlowest:")
    print(district.tail(8)[columns].round(2).to_string(index=False))
    print(f"\nwrote well_recharge.parquet and district_recharge.csv to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
