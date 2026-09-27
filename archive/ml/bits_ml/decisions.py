"""What the numbers mean for a decision: how long the water lasts, and what would change it.

Metres per year is not a decision. A district decides whether to spend on
recharge structures, on demand management, or on neither, and that turns on three
questions the primary data can answer:

* **How much room is left.** The water stands 8 m down in a well drilled to
  12.5 m, so 4.5 m of usable column remains. At the rate that well is falling,
  that is a number of years. Dug wells are 94% of this network and shallow by
  construction, so "the water has passed the bottom of the well" is not an
  abstraction: it is the day the well stops working and the farm it serves
  changes what it can grow.

* **What is taking it — which this cannot answer.** The obvious move is to split
  the observed fall into the part rainfall accounts for and the part it does not,
  and send the first to recharge structures and the second to demand management.
  That split is not supported here. The rain-expected trend in the level and the
  observed trend correlate -0.03, so subtracting one from the other leaves a
  remainder with a wider spread than the fall itself, larger than the whole in
  38% of wells. The rain figures are carried as context and labelled as such;
  no district is classified by cause.

* **What a wet year buys.** The break-even monsoon from the simulator, set
  against how often a monsoon that size actually happens there. A break-even of
  +60% in a place that sees +20% once in five years is not a plan.

The output is deliberately blunt: years, wells, and which of the two levers has
any chance of working. Every figure carries the caveat that the gap is "not
rain", not "pumping" — canal water, cropping and land use sit in it too.

Run from ml/:  python -m bits_ml.decisions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import groundwater as gw
from .config import DATA_PROCESSED

MIN_YEARS = 10
MIN_WELLS = 3
URGENT_YEARS = 10.0
NEAR_BOTTOM_M = 2.0
DRIFT_CSV = DATA_PROCESSED / "well_drift.csv"


def _fall_rate(years: np.ndarray, depth: np.ndarray) -> float:
    """Metres a year the water is falling in this well: positive means deepening."""
    return float(stats.theilslopes(depth, years)[0])


def well_outlook(readings: pd.DataFrame, campaign: str = "Nov", min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Per well: how far the water has left to fall, and how fast it is falling.

    November is used: it is the best-covered campaign and it is the year's high
    water, so it is the kindest honest reading of how much is left.
    """
    rows = []
    post_monsoon = readings[readings["campaign"] == campaign]
    for well, group in post_monsoon.groupby("well_uid", observed=True):
        if len(group) < min_years:
            continue
        group = group.sort_values("year")
        depth = group["depth_mbgl"].to_numpy()
        well_depth = group["well_depth_m"].iloc[0]
        fall = _fall_rate(group["year"].to_numpy(dtype=float), depth)
        headroom = well_depth - depth[-1] if np.isfinite(well_depth) else np.nan
        rows.append({
            "well_uid": well, "state": group["state"].iloc[0], "district": group["district"].iloc[0],
            "far_from_district": bool(group["far_from_district"].iloc[0]),
            "well_type": group["well_type"].iloc[0], "well_depth_m": well_depth,
            "latest_year": int(group["year"].iloc[-1]), "latest_depth_m": float(depth[-1]),
            "headroom_m": headroom, "fall_m_per_year": fall,
            "years_left": headroom / fall if np.isfinite(headroom) and fall > 0.01 else np.inf,
        })
    return pd.DataFrame(rows)


def add_cause(outlook: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    """Split each well's fall into the part rainfall explains and the part it does not.

    `drift_m_per_year` is the trend the model could not explain with rain. What is
    left of the observed fall is the weather's share, and the two levers act on
    different halves: recharge work on the first, demand management on the second.
    """
    expected = "rain_expected_decline_m_per_year"
    if expected in drift.columns:
        # Both parts are trends of a level, built the same way, so they add up: the fall this well
        # actually has, the part rain alone would have produced, and the remainder.
        merged = outlook.merge(drift[["well_uid", expected]], on="well_uid", how="left")
        merged["weather_m_per_year"] = merged[expected]
        merged["unexplained_m_per_year"] = merged["fall_m_per_year"] - merged[expected]
    else:
        # An older attribution file measured the gap from yearly changes rather than levels, which
        # is not on the same footing as the observed fall and must not be subtracted from it.
        merged = outlook.merge(drift[["well_uid", "drift_m_per_year"]], on="well_uid", how="left")
        merged["unexplained_m_per_year"] = merged["drift_m_per_year"]
        merged["weather_m_per_year"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        without_pumping = merged["fall_m_per_year"] - merged["unexplained_m_per_year"].clip(lower=0)
        merged["years_left_if_gap_closed"] = np.where(
            without_pumping > 0.01, merged["headroom_m"] / without_pumping, np.inf)
    return merged


def district_outlook(wells: pd.DataFrame, min_wells: int = MIN_WELLS) -> pd.DataFrame:
    """District summary in decision units, ranked by how soon wells start failing."""
    usable = wells[~wells["far_from_district"]]
    rows = []
    for (state, district), group in usable.groupby(["state", "district"], observed=True):
        if len(group) < min_wells:
            continue
        # Median over every well, infinities included: a well that is rising has no horizon, and
        # dropping those leaves the median describing whichever minority happens to be falling.
        # Across this network only about a third of wells have a finite horizon at all, so the
        # filtered version reported single wells as if they were districts.
        falling_wells = group[np.isfinite(group["years_left"])]
        unexplained = group["unexplained_m_per_year"].median()
        fall = group["fall_m_per_year"].median()
        rows.append({
            "state": state, "district": district, "wells": len(group),
            "median_depth_m": float(group["latest_depth_m"].median()),
            "median_headroom_m": float(group["headroom_m"].median(skipna=True)),
            "fall_m_per_year": float(fall),
            "unexplained_m_per_year": float(unexplained) if np.isfinite(unexplained) else np.nan,
            "median_years_left": float(group["years_left"].median()),
            "years_left_of_falling_wells": float(falling_wells["years_left"].median()) if len(falling_wells) else np.inf,
            "wells_with_a_horizon": len(falling_wells),
            "wells_failing_within_10y": int((group["years_left"] <= URGENT_YEARS).sum()),
            "wells_near_bottom_now": int((group["headroom_m"] <= NEAR_BOTTOM_M).sum()),
            "years_left_if_gap_closed": float(group["years_left_if_gap_closed"].replace(np.inf, np.nan).median()),
        })
    frame = pd.DataFrame(rows)
    # Banded by how long the wells have, not by what is taking the water: the cause split does
    # not survive its own arithmetic (see the module docstring), while the fall and the room
    # left below the water are both measured directly.
    years = frame["median_years_left"]
    frame["outlook"] = np.where(
        frame["fall_m_per_year"] <= 0.01, "not falling",
        np.where(years <= URGENT_YEARS, "wells fail within 10 years",
                 np.where(years <= 25, "wells fail within 25 years", "falling, decades of room")))
    return frame.sort_values(["wells_failing_within_10y", "fall_m_per_year"], ascending=False, ignore_index=True)


def main() -> None:
    readings = gw.load_readings()
    outlook = well_outlook(readings)
    if DRIFT_CSV.exists():
        outlook = add_cause(outlook, pd.read_csv(DRIFT_CSV))
    else:
        print(f"{DRIFT_CSV.name} missing: run python -m bits_ml.attribution to split the cause\n")
        for column in ("drift_m_per_year", "unexplained_m_per_year", "weather_m_per_year", "years_left_if_gap_closed"):
            outlook[column] = np.nan

    districts = district_outlook(outlook)
    out = DATA_PROCESSED / "district_outlook.csv"
    districts.to_csv(out, index=False)

    falling = outlook[outlook["fall_m_per_year"] > 0.01]
    urgent = outlook[outlook["years_left"] <= URGENT_YEARS]
    print(f"wells with a usable record: {len(outlook):,} | falling: {len(falling):,} "
          f"({len(falling) / len(outlook) * 100:.0f}%) | within 10 years of their own bottom: {len(urgent):,}")
    print(f"median headroom left: {outlook['headroom_m'].median():.1f} m of well below the water\n")

    print("districts where wells start failing soonest:")
    columns = ["district", "state", "wells", "median_depth_m", "median_headroom_m", "fall_m_per_year",
               "median_years_left", "wells_failing_within_10y", "outlook"]
    print(districts.head(12)[columns].round(2).to_string(index=False))

    print("\nhow long the wells have:")
    print(districts["outlook"].value_counts().to_string())
    print("\nno district is classified by cause: the rain-explained and unexplained parts of the fall")
    print("correlate -0.03 with each other, so that split is context in the file, not a verdict.")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
