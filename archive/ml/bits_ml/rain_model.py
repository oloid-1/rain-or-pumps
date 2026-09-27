"""Rain where each well stands: its normal, its history, and what-if scenarios.

This is the rainfall half of the project and it is deliberately not a forecast.
Rain cannot be predicted from its own past (Week 1: a SARIMA model only matched
the long-term average, and one monsoon barely predicts the next), so this module
describes rain instead:

* the normal for each well and calendar month, from training years only
* history at any date: totals over the last 3 to 36 months, wet days, the
  wettest day, the last completed monsoon
* trends, per well or grid cell
* scenario years built as a percentage of that place's own normal, so "+20%
  monsoon" means +20% of what falls there, about 50 mm in western Rajasthan and
  about 600 mm on the Karnataka coast

Everything is region specific by construction: no national number is ever used.
Rain reaches the groundwater model only through the features built here.

Run from ml/:  python -m bits_ml.rain_model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED

WINDOWS = (3, 6, 12, 24, 36)  # months of history summed before a reading
MONSOON = (6, 7, 8, 9)  # Jun-Sep
RAIN_TABLE = DATA_PROCESSED / "well_monthly_rain_1998_2022.parquet"


@dataclass(frozen=True)
class RainSeries:
    """Monthly rain at every well: months x wells matrices, months ascending."""

    months: pd.DatetimeIndex
    well_uid: np.ndarray
    rain: np.ndarray
    wet_days: np.ndarray
    max_day: np.ndarray

    @classmethod
    def load(cls, path: Path = RAIN_TABLE) -> "RainSeries":
        table = pd.read_parquet(path)
        wide = {c: table.pivot(index="month", columns="well_uid", values=c) for c in ("rain_mm", "wet_days", "max_day_mm")}
        rain = wide["rain_mm"].sort_index()
        if rain.isna().to_numpy().any():
            raise ValueError(f"{path.name}: gaps in the monthly rain table")
        return cls(
            months=pd.DatetimeIndex(rain.index),
            well_uid=rain.columns.to_numpy().astype(str),
            rain=rain.to_numpy(dtype=np.float64),
            wet_days=wide["wet_days"].sort_index()[rain.columns].to_numpy(dtype=np.float64),
            max_day=wide["max_day_mm"].sort_index()[rain.columns].to_numpy(dtype=np.float64),
        )

    def index_of(self, year: int, month: int) -> int:
        """Row of one calendar month, or -1 if the record does not reach it."""
        hits = np.flatnonzero((self.months.year == year) & (self.months.month == month))
        return int(hits[0]) if hits.size else -1

    def normals(self, train_years) -> "RainNormals":
        """Each well's normal rain, wet days and wettest day per calendar month, training years only."""
        years = np.asarray(self.months.year)
        months = np.asarray(self.months.month)
        keep = np.isin(years, list(train_years))
        per_month = lambda a: np.stack([a[keep & (months == m)].mean(axis=0) for m in range(1, 13)])
        return RainNormals(per_month(self.rain), per_month(self.wet_days), per_month(self.max_day))

    def with_scenario(self, year: int, delta: float, normals: "RainNormals", months=MONSOON) -> "RainSeries":
        """A copy where one year becomes a normal year with its monsoon scaled by delta.

        delta = 0.2 gives every well 20% more than its own normal monsoon, which
        is a different amount of rain in every region. The rest of the year is
        that well's normal. Wet days grow more slowly than the total (extra rain
        arrives partly as heavier days) and never exceed the days in the month.
        """
        rain, wet, mx = self.rain.copy(), self.wet_days.copy(), self.max_day.copy()
        for m in range(1, 13):
            row = self.index_of(year, m)
            if row < 0:
                continue
            scale = 1.0 + delta if m in months else 1.0
            days = pd.Period(f"{year}-{m:02d}").days_in_month
            rain[row] = normals.rain[m - 1] * scale
            wet[row] = np.minimum(normals.wet_days[m - 1] * np.sqrt(scale), days)
            mx[row] = normals.max_day[m - 1] * scale
        return RainSeries(self.months, self.well_uid, rain, wet, mx)


@dataclass(frozen=True)
class RainNormals:
    """A normal year at every well: (12, n_wells) for rain, wet days and the wettest day."""

    rain: np.ndarray
    wet_days: np.ndarray
    max_day: np.ndarray

    def monsoon(self, months=MONSOON) -> np.ndarray:
        return self.rain[[m - 1 for m in months]].sum(axis=0)

    def annual(self) -> np.ndarray:
        return self.rain.sum(axis=0)


def _cumulative(matrix: np.ndarray) -> np.ndarray:
    """Running totals with a zero row on top, so a window sum is one subtraction."""
    return np.vstack([np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)])


def history_features(series: RainSeries, year: int, month: int) -> pd.DataFrame | None:
    """Rain history at one campaign, for every well.

    Everything is counted up to the end of the month *before* the reading, so a
    reading is never explained by rain that fell after it.
    """
    end = series.index_of(year, month)  # first month excluded from the history
    if end <= 0:
        return None
    rain_cum, wet_cum = _cumulative(series.rain), _cumulative(series.wet_days)
    out = {"well_uid": series.well_uid, "year": year, "month": month}
    for w in WINDOWS:
        if end - w < 0:
            return None
        out[f"rain_{w}m"] = rain_cum[end] - rain_cum[end - w]
    out["wet_days_12m"] = wet_cum[end] - wet_cum[end - 12]
    out["max_day_12m"] = series.max_day[end - 12:end].max(axis=0)

    months = np.asarray(series.months.month)[:end]
    years = np.asarray(series.months.year)[:end]
    is_monsoon = np.isin(months, MONSOON)
    this_year = is_monsoon & (years == year)
    out["monsoon_todate_mm"] = series.rain[:end][this_year].sum(axis=0) if this_year.any() else np.zeros(len(series.well_uid))
    last = year if this_year.sum() == len(MONSOON) else year - 1
    done = is_monsoon & (years == last)
    if done.sum() != len(MONSOON):
        return None
    out["monsoon_last_mm"] = series.rain[:end][done].sum(axis=0)
    return pd.DataFrame(out)


def campaign_features(series: RainSeries, years, campaign_months: dict[str, int]) -> pd.DataFrame:
    """Rain history at every campaign of every year: one row per well per campaign."""
    frames = []
    for year in years:
        for campaign, month in campaign_months.items():
            block = history_features(series, year, month)
            if block is not None:
                frames.append(block.assign(campaign=campaign))
    if not frames:
        raise ValueError("no campaign has enough rain history")
    return pd.concat(frames, ignore_index=True)


RELATIVE_COLUMNS = [f"rain_{w}m" for w in WINDOWS] + ["wet_days_12m", "monsoon_last_mm"]


def relative_normals(features: pd.DataFrame, train_years) -> pd.DataFrame:
    """Each well's normal value of every history feature, per campaign, training years only."""
    train = features[features["year"].isin(list(train_years))]
    return train.groupby(["well_uid", "campaign"])[RELATIVE_COLUMNS].mean().add_suffix("_normal").reset_index()


def apply_relative(features: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    """Express each feature as a percentage of the given normals.

    Absolute rain says how wet the place is; the percentage says how unusual the
    year was there. The model needs both, and only the percentage is comparable
    between a well in Kerala and one in Rajasthan. A scenario must reuse the
    normals the model was trained with, which is why they are passed in.
    """
    out = features.merge(normals, on=["well_uid", "campaign"], how="left")
    for c in RELATIVE_COLUMNS:
        base = out[f"{c}_normal"]
        out[f"{c}_pct"] = np.where(base > 0, out[c] / base * 100.0, np.nan)
    out["rain_normal_mm"] = out["rain_12m_normal"]  # the well's normal year, a plain climate descriptor
    return out.drop(columns=[f"{c}_normal" for c in RELATIVE_COLUMNS if c != "rain_12m"])


def add_relative(features: pd.DataFrame, train_years) -> pd.DataFrame:
    """Percentage-of-normal features, with the normals taken from the training years."""
    return apply_relative(features, relative_normals(features, train_years))


def trend(series: RainSeries, train_years=None, months=MONSOON) -> pd.DataFrame:
    """Monsoon rain trend at each well: mm per year, with Kendall's p.

    Answers "did the rain itself decline here?", which has to be separated from
    pumping before any decline is blamed on people.
    """
    from scipy import stats

    years = np.asarray(series.months.year)
    keep_month = np.isin(np.asarray(series.months.month), months)
    all_years = sorted(set(years[keep_month]))
    if train_years is not None:
        all_years = [y for y in all_years if y in set(train_years)]
    totals = np.stack([series.rain[keep_month & (years == y)].sum(axis=0) for y in all_years])
    x = np.asarray(all_years, dtype=float)
    slope, p = np.empty(totals.shape[1]), np.empty(totals.shape[1])
    for i in range(totals.shape[1]):
        slope[i] = stats.theilslopes(totals[:, i], x)[0]
        p[i] = stats.kendalltau(x, totals[:, i]).pvalue
    return pd.DataFrame({"well_uid": series.well_uid, "monsoon_mm_per_year": slope, "p_value": p,
                         "monsoon_mean_mm": totals.mean(axis=0)})


def scenario_percentile(series: RainSeries, delta: float, normals: RainNormals, months=MONSOON) -> pd.DataFrame:
    """How often each well actually gets a monsoon at least (1 + delta) times its normal.

    Keeps a scenario honest: +20% is routine in an erratic region and rare in a steady one.
    """
    years = np.asarray(series.months.year)
    keep = np.isin(np.asarray(series.months.month), months)
    all_years = sorted(set(years[keep]))
    totals = np.stack([series.rain[keep & (years == y)].sum(axis=0) for y in all_years])
    target = normals.monsoon(months) * (1.0 + delta)
    share = (totals >= target).mean(axis=0)
    return pd.DataFrame({"well_uid": series.well_uid, "share_of_years": share,
                         "one_year_in": np.where(share > 0, 1.0 / np.where(share > 0, share, np.nan), np.inf)})


def main() -> None:
    series = RainSeries.load()
    normals = series.normals(range(2000, 2015))
    monsoon = normals.monsoon()
    print(f"wells {len(series.well_uid):,} | months {len(series.months)} "
          f"({series.months[0]:%Y-%m} to {series.months[-1]:%Y-%m})")
    print(f"normal monsoon rain, mm: min {monsoon.min():.0f} median {np.median(monsoon):.0f} max {monsoon.max():.0f}"
          f" | +20% is worth {0.2 * monsoon.min():.0f} mm at the driest well and {0.2 * monsoon.max():.0f} mm at the wettest")

    t = trend(series)
    falling = (t["monsoon_mm_per_year"] < 0) & (t["p_value"] < 0.1)
    rising = (t["monsoon_mm_per_year"] > 0) & (t["p_value"] < 0.1)
    print(f"monsoon trend 1998-2022: median {t['monsoon_mm_per_year'].median():+.2f} mm/yr | "
          f"wells drying (p<0.1) {int(falling.sum())} | wetting {int(rising.sum())}")

    pct = scenario_percentile(series, 0.2, normals)
    print(f"a +20% monsoon happens at the median well in 1 year in "
          f"{np.median(pct['one_year_in'][np.isfinite(pct['one_year_in'])]):.1f}")


if __name__ == "__main__":
    main()
