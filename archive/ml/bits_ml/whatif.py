"""If this district got 20% more monsoon rain, would it recover?

The simulator puts the two halves of the project together without mixing them.
The rain model builds a scenario year: every well gets its own normal rain, with
the monsoon scaled by the chosen percentage, so +20% means +42 mm at the driest
well and +890 mm at the wettest. The groundwater model then says where the water
would stand after such a year. The difference against a normal year is the gain,
in metres of water level.

The gain is then set against the drift measured by attribution: how far the water
sinks below its rain-expected level each year. If a year of drift outweighs the
gain, extra rain does not fix it and the decline is structural. The break-even
rainfall is the monsoon that would exactly cancel one year of drift, found by
bisection, which is only sound because the model is built so that more rain can
never predict deeper water.

Run from ml/:  python -m bits_ml.whatif --district Kurukshetra --delta 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import groundwater as gw
from . import models as M
from . import rain_model as rm
from . import splits as S
from .config import DATA_PROCESSED
from .dataset import FEATURES, TARGET, apply_categories, build, categories_of, matrix

SCENARIO_YEAR = 2022  # the last year the rain record covers; scenarios replace it
SCENARIO_CAMPAIGN = "Nov"  # after the monsoon, where recharge shows
BREAK_EVEN_LIMIT = 1.5  # stop looking past +150% monsoon


class Simulator:
    """A fitted model plus everything needed to rebuild features for a scenario year."""

    def __init__(self, model, table: pd.DataFrame, train_years=S.FINAL_TRAIN_YEARS,
                 year: int = SCENARIO_YEAR, campaign: str = SCENARIO_CAMPAIGN, categories=None):
        self.model, self.table, self.year, self.campaign = model, table, year, campaign
        # a model carries its own category order; without it, codes and labels drift apart
        self.categories = categories or categories_of(table)
        self.series = rm.RainSeries.load()
        self.normals = self.series.normals(train_years)
        actual = rm.campaign_features(self.series, sorted(table["year"].unique()), gw.CAMPAIGN_MONTH)
        self.feature_normals = rm.relative_normals(actual, train_years)
        self.static = table.drop_duplicates("well_uid").set_index("well_uid")

    def predict(self, delta: float) -> pd.Series:
        """Expected water level for every well after a year whose monsoon is (1 + delta) of normal."""
        scenario = self.series.with_scenario(self.year, delta, self.normals)
        month = gw.CAMPAIGN_MONTH[self.campaign]
        history = rm.history_features(scenario, self.year, month)
        if history is None:
            raise ValueError(f"the rain record does not reach {self.campaign} {self.year}")
        rows = rm.apply_relative(history.assign(campaign=self.campaign), self.feature_normals)
        rows = rows[rows["well_uid"].isin(self.static.index)]
        for column in ("sy", "well_depth_m", "well_type", "aquifer"):
            rows[column] = self.static.loc[rows["well_uid"], column].to_numpy()
        rows["campaign"] = self.campaign
        rows = apply_categories(rows, self.categories)
        return pd.Series(self.model.predict(matrix(rows)), index=rows["well_uid"].to_numpy())

    def gain(self, delta: float) -> pd.Series:
        """Metres of water level gained against a normal monsoon; positive means shallower water."""
        return self.predict(0.0) - self.predict(delta)


LOSS_COLUMNS = ("decline_m_per_year", "drift_m_per_year")


def loss_rate(wells: pd.DataFrame) -> tuple[pd.Series, str]:
    """What the extra rain has to beat, and what that number is.

    The observed decline is preferred over the rain-adjusted gap because it is the
    one that reproduces: districts agree with themselves across halves of the
    record at +0.24, while the gap disagrees at -0.31. So the question this
    simulator answers is "would a wetter monsoon cover the fall this district
    actually has", not "would it cover the part we attribute to pumping".
    """
    for column in LOSS_COLUMNS:
        if column in wells.columns and wells[column].notna().any():
            return wells[column], column
    raise ValueError(f"no loss rate in the well table; expected one of {LOSS_COLUMNS}")


def district_gain(simulator: Simulator, gains: pd.Series, drift: pd.DataFrame, district: str, state: str | None = None) -> dict:
    """One district's answer: rain gain against the drift it loses every year."""
    wells = drift[(drift["district"] == district) & (~drift["far_from_district"])]
    if state:
        wells = wells[wells["state"] == state]
    if wells.empty:
        raise ValueError(f"no wells with a drift estimate in {district}")
    common = wells["well_uid"][wells["well_uid"].isin(gains.index)]
    gain = float(gains.loc[common].median())
    losses, basis = loss_rate(wells)
    per_year_drift = float(losses.median())
    return {
        "district": district, "state": state or wells["state"].iloc[0], "wells": int(len(common)),
        "gain_m": gain, "drift_m_per_year": per_year_drift, "net_m": gain - per_year_drift, "basis": basis,
        "verdict": "recovers with this rain" if gain >= per_year_drift else "structural: rain alone does not cover it",
    }


def break_even(simulator: Simulator, drift: pd.DataFrame, district: str, state: str | None = None,
               limit: float = BREAK_EVEN_LIMIT, tolerance: float = 0.01) -> float | None:
    """The monsoon increase that exactly cancels one year of drift, or None if even +150% cannot."""
    wells = drift[(drift["district"] == district) & (~drift["far_from_district"])]
    target = float(loss_rate(wells)[0].median())
    if target <= 0:
        return 0.0
    net = lambda d: district_gain(simulator, simulator.gain(d), drift, district, state)["net_m"]
    if net(limit) < 0:
        return None
    low, high = 0.0, limit
    while high - low > tolerance:
        middle = (low + high) / 2
        low, high = (low, middle) if net(middle) >= 0 else (middle, high)
    return high


def curve(simulator: Simulator, drift: pd.DataFrame, deltas=(-0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
          min_wells: int = 3) -> pd.DataFrame:
    """Every district's answer at several rainfall levels, in one pass.

    One row per district: the gain at each rainfall level, the drift it loses per
    year, and the monsoon that would cancel that drift. The break-even comes from
    the curve itself rather than a search, which is exact enough for a dashboard
    and needs no extra model calls.
    """
    gains = {d: simulator.gain(d) for d in sorted(deltas)}
    usable = drift[~drift["far_from_district"]]
    rows = []
    for (state, district), g in usable.groupby(["state", "district"], observed=True):
        wells = g["well_uid"][g["well_uid"].isin(next(iter(gains.values())).index)]
        if len(wells) < min_wells:
            continue
        per_year_drift = float(loss_rate(g)[0].median())
        row = {"state": state, "district": district, "wells": int(len(wells)),
               "drift_m_per_year": per_year_drift}
        curve_points = {d: float(gains[d].loc[wells].median()) for d in sorted(deltas)}
        for d, value in curve_points.items():
            row[f"gain_{int(round(d * 100)):+d}pct"] = value
        positive = [(d, v) for d, v in curve_points.items() if d > 0]
        row["break_even_monsoon"] = np.nan
        if per_year_drift <= 0:
            row["break_even_monsoon"] = 0.0
        elif positive:
            deltas_arr = np.array([d for d, _ in positive])
            values = np.array([v for _, v in positive])
            if values.max() >= per_year_drift:
                row["break_even_monsoon"] = float(np.interp(per_year_drift, values, deltas_arr))
        row["verdict"] = ("recovering already" if per_year_drift <= 0
                          else "recovers with +20% rain" if curve_points.get(0.2, 0) >= per_year_drift
                          else "structural")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("drift_m_per_year", ascending=False, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--delta", type=float, default=0.2, help="monsoon change, 0.2 means 20% above normal")
    parser.add_argument("--district", help="one district; default is the ten with the strongest drift")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--grid", action="store_true", help="write every district's curve to district_whatif.csv")
    args = parser.parse_args()

    drift_path = DATA_PROCESSED / "well_drift.csv"
    if not drift_path.exists():
        raise SystemExit(f"{drift_path.name} is missing; run  python -m bits_ml.attribution  first")
    drift = pd.read_csv(drift_path)

    table = build(S.FINAL_TRAIN_YEARS)
    rows = np.flatnonzero(np.isin(table["year"].to_numpy(), list(S.FINAL_TRAIN_YEARS)))
    model = M.build(args.model)
    model.fit(matrix(table.iloc[rows]), table[TARGET].to_numpy()[rows])
    simulator = Simulator(model, table)
    gains = simulator.gain(args.delta)

    if args.grid:
        all_districts = curve(simulator, drift)
        out = DATA_PROCESSED / "district_whatif.csv"
        all_districts.to_csv(out, index=False)
        print(f"{len(all_districts)} districts | verdicts: {all_districts['verdict'].value_counts().to_dict()}")
        print(f"wrote {out.name} to {DATA_PROCESSED}\n")

    if args.district:
        names = [(args.district, None)]
    else:
        ranked = (drift[~drift["far_from_district"]].groupby(["state", "district"], observed=True)["drift_m_per_year"]
                  .agg(["median", "size"]).query("size >= 3").sort_values("median", ascending=False).head(args.top))
        names = [(district, state) for state, district in ranked.index]

    print(f"a monsoon {args.delta:+.0%} against normal, measured at the {SCENARIO_CAMPAIGN} reading:\n")
    out = []
    for district, state in names:
        answer = district_gain(simulator, gains, drift, district, state)
        answer["break_even_monsoon"] = break_even(simulator, drift, district, state)
        out.append(answer)
    frame = pd.DataFrame(out)
    frame["break_even_monsoon"] = frame["break_even_monsoon"].map(
        lambda v: "beyond +150%" if v is None or pd.isna(v) else f"{v:+.0%}")
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
