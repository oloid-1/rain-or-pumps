"""Controls: the tests that are supposed to fail, and the noise floor under the ranking.

A score is only worth something if it collapses when the information behind it is
taken away. A ranking is only worth something if the districts in it stand above
what pure chance produces. Neither has been checked, and the attribution already
disagrees with two independent signals, so this is where to look before adding
any more features.

1. **Rain shuffled across years** — each well keeps its own rain, in the wrong
   order. Skill must fall to about zero. If it does not, the model is scoring on
   something other than the weather of that year, and the "rain-expected" level
   is not what it claims to be.

2. **Rain shuffled between wells** — each well is given another well's rain for
   the same years. Skill must fall to about zero. If it stays up, what looked
   like local rainfall response is really the national monsoon rhythm, which
   every well shares and which explains nothing about a particular district.

3. **Rain from years later** — information the model cannot have. It must not
   help. If it does, the features are misaligned in time and every result is
   suspect.

   It has to be shifted by more than one year to mean anything. The features
   include windows of 24 and 36 months, so "next year's" long windows still
   contain most of this year's rain, and the control passes information through
   the back door: shifted by one year it scored +0.086 against a real +0.123,
   which says nothing about alignment. The shift therefore has to clear the
   longest window, which is why it defaults to four years.

4. **Permuted drift** — the ranking's own numbers, with each well's residuals
   shuffled across years so that any real trend is destroyed. Running the same
   trend estimate on that gives the drift this method invents from noise alone,
   which is the bar a district has to clear before it belongs on a watchlist.

   Read it for what it is: shuffling the residuals destroys the trend, not the
   rainfall, so clearing the bar means "this district really is drifting", not
   "this drift is pumping rather than weather". Controls 1 to 3 are the ones that
   test the rain claim, and the ranking is only worth as much as they are.

Run from ml/:  python -m scripts.controls [--target annual] [--permutations 500]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from bits_ml import models as M
from bits_ml import splits as S
from bits_ml import targets as T
from bits_ml.config import DATA_PROCESSED
from bits_ml.dataset import CATEGORICAL, RAIN_FEATURES, WELL_FEATURES, build
from bits_ml.evaluate import metrics, skill

FEATURES = list(RAIN_FEATURES) + list(WELL_FEATURES) + list(CATEGORICAL)
MIN_YEARS = 15
MIN_WELLS = 3


def fit_and_score(frame: pd.DataFrame, features: list[str], label: str, model_name: str = "lgbm",
                  target: str = "level") -> dict:
    year = frame["year"].to_numpy()
    train = np.flatnonzero(np.isin(year, list(S.TRAIN_YEARS)))
    val = np.flatnonzero(np.isin(year, list(S.VAL_YEARS)))
    X, y = frame[features], frame[T.TARGET].to_numpy()
    builder = (M.build("lgbm", wetter_is_higher=T.WETTER_IS_HIGHER[target]) if model_name == "lgbm"
               else M.build(model_name))  # a rise runs the other way from a depth
    model = builder.fit(X.iloc[train], y[train])
    prediction = model.predict(X.iloc[val])
    return {"control": label, "n": len(val)} | metrics(y[val], prediction) | {
        "skill_vs_normal": skill(y[val], prediction, np.zeros(len(val)))
    }


def _reorder_features(frame: pd.DataFrame, features: list[str], order: np.ndarray) -> pd.DataFrame:
    """Rebuild the frame with its feature columns read in a different row order.

    Column by column on purpose: taking the whole block as an array mixes floats
    with categoricals, which pandas flattens to objects and LightGBM then refuses.
    """
    out = frame.copy()
    for column in features:
        out[column] = frame[column].iloc[order].set_axis(frame.index)
    return out


def _keys(frame: pd.DataFrame, *names: str) -> list[str]:
    """Only the keys this target actually has: the level target keeps four campaigns a year, the others one."""
    return [name for name in names if name in frame.columns]


def shuffle_within_well(frame: pd.DataFrame, features: list[str], seed: int = 0) -> pd.DataFrame:
    """Each well keeps its own rain, in the wrong year order.

    Shuffled inside each campaign, so January is still compared against January
    and only the year is wrong. Scrambling campaigns too would destroy the
    seasonal cycle as well, which is a different and much easier test to pass.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    for rows in frame.groupby(_keys(frame, "well_uid", "campaign"), observed=True).indices.values():
        order[rows] = rng.permutation(rows)
    return _reorder_features(frame, features, order)


def shuffle_between_wells(frame: pd.DataFrame, features: list[str], seed: int = 0) -> pd.DataFrame:
    """Each well is handed another well's rain for the same year and campaign."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    for rows in frame.groupby(_keys(frame, "year", "campaign"), observed=True).indices.values():
        order[rows] = rng.permutation(rows)
    return _reorder_features(frame, features, order)


def use_later_rain(frame: pd.DataFrame, features: list[str], shift_years: int = 4) -> pd.DataFrame:
    """Give every row the rain of some years later: impossible, and must not help.

    The shift has to be longer than the longest rain window (36 months), or the
    "future" features still carry the real year's rain inside them.
    """
    keys = _keys(frame, "well_uid", "year", "campaign")
    values = [f for f in features if f not in keys]  # campaign is a key here, not a value to move
    future = frame[[*keys, *values]].copy()
    future["year"] -= shift_years
    merged = frame[keys].merge(future, on=keys, how="left")
    if len(merged) != len(frame):
        raise ValueError(f"next-year join changed the row count ({len(frame):,} -> {len(merged):,}); keys are {keys}")
    out = frame.copy()
    for column in values:
        out[column] = merged[column].set_axis(frame.index)
    return out.dropna(subset=values)


def _slopes(years: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row of `series` against `years` (rows are permutations)."""
    centred = years - years.mean()
    return (series - series.mean(axis=1, keepdims=True)) @ centred / (centred**2).sum()


def permuted_drift(residuals: pd.DataFrame, permutations: int, campaign: str = "Nov", seed: int = 0) -> pd.DataFrame:
    """The drift this method produces from residuals that carry no trend at all.

    Each well's residuals are shuffled across years, which destroys any real
    trend but keeps its spread, and the same trend estimate is run again. Doing
    it many times, keeping permutations aligned across wells, gives the
    distribution of district drift under "nothing is happening here".
    """
    rng = np.random.default_rng(seed)
    wells = residuals[residuals["campaign"] == campaign].groupby("well_uid", observed=True)
    real, null, keys = {}, {}, {}
    for well, group in wells:
        if len(group) < MIN_YEARS:
            continue
        years = group["year"].to_numpy(dtype=float)
        values = group["residual_m"].to_numpy()
        draws = np.stack([rng.permutation(values) for _ in range(permutations)])
        real[well] = float(_slopes(years, values[None, :])[0])
        null[well] = _slopes(years, draws)
        keys[well] = (group["state"].iloc[0], group["district"].iloc[0], bool(group["far_from_district"].iloc[0]))

    frame = pd.DataFrame({"well_uid": list(real), "slope": [real[w] for w in real],
                          "state": [keys[w][0] for w in real], "district": [keys[w][1] for w in real],
                          "far_from_district": [keys[w][2] for w in real]})
    usable = frame[~frame["far_from_district"]]
    rows = []
    for (state, district), group in usable.groupby(["state", "district"], observed=True):
        if len(group) < MIN_WELLS:
            continue
        draws = np.stack([null[w] for w in group["well_uid"]])  # (wells, permutations)
        null_medians = np.median(draws, axis=0)
        observed = float(np.median(group["slope"]))
        rows.append({
            "state": state, "district": district, "wells": len(group), "drift_m_per_year": observed,
            "noise_p95": float(np.quantile(np.abs(null_medians), 0.95)),
            "p_value": float((np.abs(null_medians) >= abs(observed)).mean()),
        })
    return pd.DataFrame(rows).sort_values("drift_m_per_year", ascending=False, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="annual", help="level, rise or annual")
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--shift-years", type=int, default=4, dest="shift_years",
                        help="how far ahead the impossible-rain control looks; must clear the 36-month window")
    args = parser.parse_args()

    frame = T.build_target(args.target, build(S.TRAIN_YEARS))
    features = [f for f in FEATURES if f in frame.columns]
    print(f"controls on the {args.target} target: {len(frame):,} rows, {frame['well_uid'].nunique():,} wells\n")

    rows = [fit_and_score(frame, features, "as measured", args.model, args.target)]
    print(f"   as measured                skill {rows[-1]['skill_vs_normal']:+.4f}", flush=True)
    for label, scrambled in (
        ("rain shuffled across years", shuffle_within_well(frame, features)),
        ("rain shuffled between wells", shuffle_between_wells(frame, features)),
        (f"rain from {args.shift_years} years later", use_later_rain(frame, features, args.shift_years)),
    ):
        rows.append(fit_and_score(scrambled, features, label, args.model, args.target))
        print(f"   {label:26s} skill {rows[-1]['skill_vs_normal']:+.4f}", flush=True)

    controls = pd.DataFrame(rows)
    controls.to_csv(DATA_PROCESSED / "control_results.csv", index=False)
    real_skill = controls.iloc[0]["skill_vs_normal"]
    worst = controls.iloc[1:]["skill_vs_normal"].max()
    if real_skill < 0.02:
        verdict = (f"the model itself only reaches {real_skill:+.4f} on this target, so there is no rain signal here "
                   f"for a control to destroy. The controls say nothing either way; the target does.")
    elif worst < real_skill * 0.25:
        verdict = ("the controls collapse as they should: the skill comes from this well's own rain, in the right year")
    else:
        verdict = (f"a control reaches {worst:+.4f} against the real {real_skill:+.4f}: whatever the model is "
                   f"scoring on, it is not local rainfall in the right year")
    print(f"\n   -> {verdict}")

    residual_path = DATA_PROCESSED / "well_residuals.parquet"
    if not residual_path.exists():
        print(f"\n{residual_path.name} is missing; run python -m bits_ml.attribution for the drift control")
        return

    print(f"\npermuted drift, {args.permutations} shuffles per well:", flush=True)
    drift = permuted_drift(pd.read_parquet(residual_path), args.permutations)
    drift.to_csv(DATA_PROCESSED / "drift_noise_floor.csv", index=False)
    beats = drift[drift["p_value"] < 0.05]
    sinking = drift[(drift["drift_m_per_year"] > 0) & (drift["p_value"] < 0.05)]
    print(f"   districts ranked                {len(drift)}")
    print(f"   noise floor (median across districts)  ±{drift['noise_p95'].median():.3f} m/yr")
    print(f"   districts beating their own noise      {len(beats)} ({len(beats) / len(drift) * 100:.0f}%), "
          f"of which sinking: {len(sinking)}")
    print(f"   expected by chance alone               about {0.05 * len(drift):.0f}")
    print("\n   this says a district's drift is a real trend rather than noise. It does not say the trend is")
    print("   pumping rather than weather: shuffling residuals destroys the trend, not the rain. Controls 1-3 test that.")
    print("\n   top districts that clear the bar:")
    print(sinking.head(10)[["district", "state", "wells", "drift_m_per_year", "noise_p95", "p_value"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
