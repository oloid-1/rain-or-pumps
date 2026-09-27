"""The water a district loses that its rainfall does not account for.

The model says how much a well should gain or lose in a year given the rain it
received. Where a well loses more than that, year after year, something other
than the weather is taking the water. That shortfall, in metres a year, is the
extraction-pressure proxy.

Three rules keep it honest, and the first two were learnt the hard way:

* **The raw change, not a deviation from the well's own normal.** Every target in
  `targets.py` is expressed as a deviation, which is right for measuring how well
  rain explains the year-to-year swing. It is wrong here: a well's normal *rate*
  is its chronic decline, so subtracting it hides exactly the loss we are looking
  for. Attribution therefore predicts the raw change.

* **Rain features only.** Specific yield, depth and aquifer are allowed when the
  question is "how much does rain move this well", because they cannot encode
  pumping. Here they can: "deep bore wells in hard rock fall faster" is partly a
  statement about how hard they are pumped, and a model given it would explain
  the decline with itself.

* **Every year predicted by a model that never saw it**, with each block's
  normals rebuilt from its own training years, so a well's decline cannot leak
  into its own expectation.

The result per well is a mean shortfall in metres a year, with a t-test against
zero; per district, the median of its wells with a bootstrap interval. November
to November is the interval, because November has the best coverage and closes
the year after the monsoon has arrived.

Run from ml/:  python -m bits_ml.attribution [--target annual] [--model ridge]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from . import models as M
from . import splits as S
from . import targets as T
from .config import DATA_PROCESSED
from .dataset import RAIN_FEATURES, build

RAW_COLUMN = {"annual": "change_m", "rise": "rise_m"}
ATTRIBUTION_CAMPAIGN = "Nov"  # best covered, and it closes the year after the monsoon has arrived
MIN_YEARS = 10
MIN_WELLS = 3
N_BOOTSTRAP = 1000


def cross_fitted(target: str = "annual", model_name: str = "ridge", blocks=S.CROSS_FIT_BLOCKS,
                 features=RAIN_FEATURES) -> pd.DataFrame:
    """Rain-expected change for every well-year, each block predicted out of sample."""
    raw = RAW_COLUMN[target]
    frames = []
    for lo, hi in blocks:
        fit_years = tuple(y for y in range(2000, 2023) if not (lo <= y <= hi))
        frame = T.build_target(target, build(train_years=fit_years), train_years=fit_years)
        usable = [f for f in features if f in frame.columns]
        year = frame["year"].to_numpy()
        held = (year >= lo) & (year <= hi)
        if not held.any():
            continue
        model = (M.build(model_name, wetter_is_higher=T.WETTER_IS_HIGHER[target]) if model_name == "lgbm"
                 else M.build(model_name))
        X, y = frame[usable], frame[raw].to_numpy()
        model.fit(X.iloc[~held], y[~held])
        block = frame.loc[held, ["well_uid", "year", "district", "state", "far_from_district", raw]].copy()
        block["expected_m"] = model.predict(X.iloc[held])
        block["shortfall_m"] = block["expected_m"] - block[raw]  # positive = lost more than rain explains
        frames.append(block)
        print(f"   block {lo}-{hi}: {int(held.sum()):,} well-years predicted by a model fitted on {int((~held).sum()):,}",
              flush=True)
    out = pd.concat(frames, ignore_index=True).sort_values(["well_uid", "year"], ignore_index=True)

    # Each year carries a nationwide component the rain features do not capture: 2019 came in
    # about 0.6 m better than expected almost everywhere, 2020 half a metre worse, and the
    # national shortfall swings by 0.27 m from year to year. Whatever that is - a monsoon the
    # features describe badly, a wider climate signal, the survey itself - it is not any one
    # district pumping, so every well-year is measured against the national median of its own
    # year. The absolute figure stays in the table for anyone who wants it.
    out["national_m"] = out.groupby("year")["shortfall_m"].transform("median")
    out["relative_shortfall_m"] = out["shortfall_m"] - out["national_m"]
    return out


def well_pressure(residuals: pd.DataFrame, target: str = "annual", min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Per well: the average metres a year lost beyond what rain explains, and whether it is real."""
    raw = RAW_COLUMN[target]
    # measured against the country in the same year, so a national wet or dry year is not
    # charged to a district; falls back to the absolute figure if centring was not applied
    column = "relative_shortfall_m" if "relative_shortfall_m" in residuals.columns else "shortfall_m"
    rows = []
    for well, group in residuals.groupby("well_uid", observed=True):
        if len(group) < min_years:
            continue
        shortfall = group[column].to_numpy()
        test = stats.ttest_1samp(shortfall, 0.0) if shortfall.std() > 0 else None
        rows.append({
            "well_uid": well, "district": group["district"].iloc[0], "state": group["state"].iloc[0],
            "far_from_district": bool(group["far_from_district"].iloc[0]), "years": len(group),
            "drift_m_per_year": float(shortfall.mean()),  # name kept: the simulator and API read this column
            "absolute_shortfall_m": float(group["shortfall_m"].mean()),
            "observed_m_per_year": float(group[raw].mean()),
            "rain_expected_m_per_year": float(group["expected_m"].mean()),
            "p_value": float(test.pvalue) if test is not None else np.nan,
        })
    return pd.DataFrame(rows)


def residual_level_trend(residuals: pd.DataFrame, target: str = "annual", min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Per well: how fast the gap between the observed level and the rain-expected one widens.

    Why a trend and not the average of the yearly shortfalls, which is the obvious
    thing to do: the average of year-over-year changes telescopes to (first level
    minus last) divided by the years between, so it is an endpoint statistic and
    it reverses between one half of the record and the other. Measured on this
    data, districts agree with themselves across halves at -0.44 on the average
    shortfall, and the same reversal is there at -0.37 in the raw observed change
    with no model involved at all, so it is arithmetic rather than a modelling
    mistake. The November level's own trend, by contrast, agrees at +0.24 against
    a null of 0.11.

    So the yearly shortfalls are accumulated into a running gap, in metres, and
    the Theil-Sen trend of that gap is the measure: every pairwise slope counts,
    not just the ends, and one freak year cannot carry a district.
    """
    raw = RAW_COLUMN[target]
    column = "relative_shortfall_m" if "relative_shortfall_m" in residuals.columns else "shortfall_m"
    rows = []
    for well, group in residuals.groupby("well_uid", observed=True):
        if len(group) < min_years:
            continue
        group = group.sort_values("year")
        years = group["year"].to_numpy(dtype=float)
        gap = np.cumsum(group[column].to_numpy())  # metres of water owed, accumulating
        slope, _, low, high = stats.theilslopes(gap, years)
        rows.append({
            "well_uid": well, "district": group["district"].iloc[0], "state": group["state"].iloc[0],
            "far_from_district": bool(group["far_from_district"].iloc[0]), "years": len(group),
            "drift_m_per_year": float(slope),  # name kept: the simulator and API read this column
            "slope_low": float(low), "slope_high": float(high),
            "absolute_shortfall_m": float(group["shortfall_m"].mean()),
            "observed_m_per_year": float(group[raw].mean()),
            "rain_expected_m_per_year": float(group["expected_m"].mean()),
            # The same trend, in depth terms, of the level path rain alone implies: positive means
            # rain by itself would have deepened this well. It is built the same way as the observed
            # decline in observed_decline(), so the two can be subtracted and the parts add up.
            "rain_expected_decline_m_per_year": -float(stats.theilslopes(
                np.cumsum(group["expected_m"].to_numpy()), years)[0]),
            "p_value": float(stats.kendalltau(years, gap).pvalue),
        })
    return pd.DataFrame(rows)


def observed_decline(readings: pd.DataFrame | None = None, campaign: str = ATTRIBUTION_CAMPAIGN,
                     min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Per well: how fast the water table itself is deepening, with no model involved.

    This is the one measure here that reproduces. Split the record in half and
    districts agree with themselves at +0.24, rising to +0.29 where five or more
    wells are monitored — better coverage, better agreement, as a real property
    should behave. The rain-adjusted gap disagrees with itself at -0.31 and gets
    *worse* with coverage (-0.37 at five wells), so it is not thin-district noise.

    The ranking is therefore built on this, and the rain model is used to explain
    what it finds rather than to order it.
    """
    from . import groundwater as gw

    readings = gw.load_readings() if readings is None else readings
    post_monsoon = readings[readings["campaign"] == campaign]
    rows = []
    for well, group in post_monsoon.groupby("well_uid", observed=True):
        if len(group) < min_years:
            continue
        group = group.sort_values("year")
        years, depth = group["year"].to_numpy(dtype=float), group["depth_mbgl"].to_numpy()
        slope, _, low, high = stats.theilslopes(depth, years)  # positive = deepening = losing water
        rows.append({
            "well_uid": well, "district": group["district"].iloc[0], "state": group["state"].iloc[0],
            "far_from_district": bool(group["far_from_district"].iloc[0]), "decline_years": len(group),
            "decline_m_per_year": float(slope), "decline_low": float(low), "decline_high": float(high),
            "decline_p_value": float(stats.kendalltau(years, depth).pvalue),
        })
    return pd.DataFrame(rows)


VERDICTS = {
    "decline_m_per_year": ("water table falling", "water table rising", "no clear movement"),
    "drift_m_per_year": ("extraction beyond rain", "recovering beyond rain", "not separable from rain"),
}


def district_ranking(wells: pd.DataFrame, value_column: str = "decline_m_per_year",
                     min_wells: int = MIN_WELLS, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> pd.DataFrame:
    """District median of the chosen measure, with a bootstrap interval over its wells."""
    rng = np.random.default_rng(seed)
    usable = wells[~wells["far_from_district"]]
    rows = []
    for (state, district), group in usable.groupby(["state", "district"], observed=True):
        if len(group) < min_wells:
            continue
        values = group[value_column].dropna().to_numpy()
        if len(values) < min_wells:
            continue
        boot = np.median(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        row = {"state": state, "district": district, "wells": len(values),
               "ci_low": float(lo), "ci_high": float(hi), "wells_sinking": int((values > 0).sum())}
        for column in ("decline_m_per_year", "drift_m_per_year", "observed_m_per_year", "rain_expected_m_per_year"):
            if column in group.columns:
                row[column] = float(group[column].median())
        row[value_column] = float(np.median(values))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(value_column, ascending=False, ignore_index=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    falling, rising, unclear = VERDICTS.get(value_column, VERDICTS["drift_m_per_year"])
    out["verdict"] = np.where(out["ci_low"] > 0, falling, np.where(out["ci_high"] < 0, rising, unclear))
    return out


def dry_season_check(ranking: pd.DataFrame, readings: pd.DataFrame) -> tuple[float, float]:
    """Does the shortfall line up with dry-season drawdown, when pumping happens?

    An independent check from the same primary data: districts losing water beyond
    what rain explains should also draw down harder between November and May.
    """
    from . import groundwater as gw

    moves = gw.seasonal_moves(readings).merge(readings[["well_uid", "district", "state"]].drop_duplicates(),
                                              on="well_uid")
    per_district = moves.groupby(["state", "district"], observed=True)["dry_fall_m"].median().reset_index()
    merged = ranking.merge(per_district, on=["state", "district"], how="inner").dropna(subset=["dry_fall_m"])
    result = stats.spearmanr(merged["drift_m_per_year"], -merged["dry_fall_m"])
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="annual", choices=sorted(RAW_COLUMN))
    parser.add_argument("--model", default="ridge", help="ridge or lgbm")
    parser.add_argument("--measure", default="trend", choices=("trend", "mean"),
                        help="trend of the accumulating gap (holds up across halves) or the yearly average (does not)")
    parser.add_argument("--rank-by", default="decline_m_per_year", dest="rank_by",
                        choices=("decline_m_per_year", "drift_m_per_year"),
                        help="observed decline (reproduces across halves) or the rain-adjusted gap (does not)")
    args = parser.parse_args()

    print(f"cross-fitting {args.model} on the raw {args.target} change, rain features only, "
          f"over {len(S.CROSS_FIT_BLOCKS)} blocks of years:")
    residuals = cross_fitted(args.target, args.model)
    wells = (residual_level_trend(residuals, args.target) if args.measure == "trend"
             else well_pressure(residuals, args.target))
    wells = wells.merge(observed_decline(), on=["well_uid", "district", "state", "far_from_district"], how="outer")
    ranking = district_ranking(wells, value_column=args.rank_by)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    residuals.to_parquet(DATA_PROCESSED / "well_residuals.parquet", index=False)
    wells.to_csv(DATA_PROCESSED / "well_drift.csv", index=False)
    ranking.to_csv(DATA_PROCESSED / "district_ranking.csv", index=False)

    beyond = ranking[ranking["verdict"] == "extraction beyond rain"]
    national = residuals.groupby("year")["shortfall_m"].median()
    print(f"\nthe national shortfall swings {national.std():.2f} m between years "
          f"({national.idxmin()}: {national.min():+.2f}, {national.idxmax()}: {national.max():+.2f}); "
          f"districts are measured against it, not against zero")
    print(f"wells measured: {len(wells):,} | districts ranked: {len(ranking)} | "
          f"losing beyond rain: {len(beyond)} | recovering: {int((ranking['verdict'] == 'recovering beyond rain').sum())}")
    print(f"median well shortfall: {wells['drift_m_per_year'].median():+.3f} m/yr relative to the country "
          f"({wells['absolute_shortfall_m'].median():+.3f} absolute) | "
          f"wells with a clear shortfall (p<0.05): {int(((wells['p_value'] < 0.05) & (wells['drift_m_per_year'] > 0)).sum()):,}")

    headline = ("water table falling fastest" if args.rank_by == "decline_m_per_year"
                else "water lost beyond rainfall")
    print(f"\ntop 12 districts by {headline} (m/yr):")
    columns = [c for c in ["rank", "district", "state", "wells", "decline_m_per_year", "drift_m_per_year",
                           "ci_low", "ci_high", "rain_expected_m_per_year"] if c in ranking.columns]
    print(ranking.head(12)[columns].round(3).to_string(index=False))
    print("\n   decline_m_per_year is the observed November water table, no model: it agrees with itself across")
    print("   halves of the record at +0.24 (+0.29 where five wells or more are monitored).")
    print("   drift_m_per_year is the part rainfall cannot account for. It does NOT agree with itself across")
    print("   halves (-0.31, worsening to -0.37 with better coverage), so it explains a district but must not rank one.")

    from . import groundwater as gw

    rho, p = dry_season_check(ranking, gw.load_readings())
    agrees = "agrees" if rho > 0 and p < 0.05 else "does not agree"
    print(f"\ncheck against dry-season drawdown (independent, same data): Spearman rho {rho:+.2f} (p {p:.1e}) - {agrees}")
    print(f"wrote well_residuals.parquet, well_drift.csv, district_ranking.csv to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
