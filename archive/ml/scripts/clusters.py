"""District archetypes: which places behave alike, found rather than assumed.

The ranking says which districts are losing water. It does not say what kind of
place each one is, and a policy answer needs that: a shallow district whose wells
refill hard every monsoon is a different problem from a deep one that barely
moves whatever the rain does, even when both are falling at the same rate.

So districts are grouped on how they behave, not on where they are. Six measures,
all from the two project datasets:

* how much rain a normal year brings, and how variable that is
* how far the water stands below ground, and how much it swings between May and
  November
* how strongly the swing follows the rain, which is the recharge response
* how fast the table is falling

K-means on standardised measures, with the number of groups chosen by silhouette
rather than picked, and PCA only for drawing. The groups are then described by
what separates them, so each one can be named from its own numbers.

Run from ml/:  python -m scripts.clusters [--max-clusters 8]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from bits_ml import groundwater as gw
from bits_ml import rain_model as rm
from bits_ml import splits as S
from bits_ml.config import DATA_PROCESSED, RANDOM_SEED

MIN_WELLS = 3
MIN_YEARS = 10
FEATURES = {
    "normal_rain_mm": "a normal year's rain",
    "rain_variability": "how much that rain varies year to year",
    "depth_m": "how far down the water stands",
    "seasonal_swing_m": "how far it rises over the monsoon",
    "rain_response": "how closely the rise follows the rain",
    "decline_m_per_year": "how fast the table is falling",
}


def district_behaviour() -> pd.DataFrame:
    """One row per district: six numbers describing how its groundwater behaves."""
    readings = gw.load_readings()
    moves = gw.seasonal_moves(readings)
    static = readings.drop_duplicates("well_uid")[["well_uid", "state", "district", "far_from_district"]]
    moves = moves.merge(static, on="well_uid")
    moves = moves[~moves["far_from_district"]]

    series = rm.RainSeries.load()
    normals = series.normals(S.FINAL_TRAIN_YEARS)
    years = np.asarray(series.months.year)
    monsoon_months = np.isin(np.asarray(series.months.month), rm.MONSOON)
    monsoon_years = sorted(set(years[monsoon_months]))
    monsoon_totals = pd.DataFrame(
        np.stack([series.rain[monsoon_months & (years == y)].sum(axis=0) for y in monsoon_years]),
        index=monsoon_years, columns=series.well_uid)
    normal_annual = pd.Series(normals.annual(), index=series.well_uid)

    rows = []
    for (state, district), group in moves.groupby(["state", "district"], observed=True):
        wells = group["well_uid"].unique()
        known = [w for w in wells if w in monsoon_totals.columns]
        if len(wells) < MIN_WELLS or not known:
            continue
        rise = group.dropna(subset=["monsoon_rise_m"])
        if len(rise) < MIN_YEARS:
            continue
        # how closely this district's monsoon rise follows its own monsoon rain
        yearly = rise.groupby("year")["monsoon_rise_m"].median()
        rain_by_year = monsoon_totals[known].median(axis=1)
        shared = yearly.index.intersection(rain_by_year.index)
        response = (stats.pearsonr(rain_by_year.loc[shared], yearly.loc[shared]).statistic
                    if len(shared) >= MIN_YEARS else np.nan)
        depth = group["nov_depth_m"].median()
        decline = _decline(group)
        rows.append({
            "state": state, "district": district, "wells": len(wells),
            "normal_rain_mm": float(normal_annual[known].median()),
            "rain_variability": float((rain_by_year.std() / rain_by_year.mean()) if rain_by_year.mean() else np.nan),
            "depth_m": float(depth), "seasonal_swing_m": float(rise["monsoon_rise_m"].median()),
            "rain_response": float(response), "decline_m_per_year": decline,
        })
    return pd.DataFrame(rows).dropna(subset=list(FEATURES))


def _decline(group: pd.DataFrame) -> float:
    """Median deepening rate of the district's wells, metres a year."""
    slopes = []
    for _, well in group.groupby("well_uid", observed=True):
        well = well.dropna(subset=["nov_depth_m"]).sort_values("year")
        if len(well) >= MIN_YEARS:
            slopes.append(stats.theilslopes(well["nov_depth_m"].to_numpy(), well["year"].to_numpy(float))[0])
    return float(np.median(slopes)) if slopes else np.nan


def choose_k(X: np.ndarray, max_clusters: int, seed: int = RANDOM_SEED) -> tuple[int, pd.DataFrame]:
    """Pick the number of groups by silhouette rather than by eye."""
    scores = []
    for k in range(2, max_clusters + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores.append({"k": k, "silhouette": float(silhouette_score(X, labels))})
    table = pd.DataFrame(scores)
    return int(table.loc[table["silhouette"].idxmax(), "k"]), table


def stability(X: np.ndarray, n_clusters: int, resamples: int = 20, seed: int = RANDOM_SEED) -> dict:
    """How much of the grouping survives dropping a fifth of the districts.

    Reported with the groups, never separately, because on this data it is the
    number that decides how far they can be pushed: measured at about 0.71 mean
    agreement with a worst case near 0.44, which is a tendency along a continuum
    rather than a set of types. Separated synthetic districts score above 0.9.
    """
    from sklearn.metrics import adjusted_rand_score

    base = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
    scores = []
    for draw in range(resamples):
        rng = np.random.default_rng(seed + draw)
        keep = rng.choice(len(X), size=int(len(X) * 0.8), replace=False)
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X[keep])
        scores.append(adjusted_rand_score(base[keep], labels))
    return {"mean_agreement": float(np.mean(scores)), "worst_agreement": float(np.min(scores)),
            "resamples": resamples}


def null_silhouette(X: np.ndarray, n_clusters: int, draws: int = 30, seed: int = RANDOM_SEED) -> float:
    """Silhouette from data with the same spread per measure but no joint structure."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        shuffled = np.column_stack([rng.permutation(column) for column in X.T])
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(shuffled)
        scores.append(silhouette_score(shuffled, labels))
    return float(np.mean(scores))


def describe(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """What separates each group: its median on every measure, against the national median."""
    out = frame.copy()
    out["cluster"] = labels
    national = frame[list(FEATURES)].median()
    profile = out.groupby("cluster")[list(FEATURES)].median()
    profile.insert(0, "districts", out.groupby("cluster").size())
    for column in FEATURES:
        profile[f"{column}_vs_national"] = (profile[column] - national[column]).round(3)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-clusters", type=int, default=8, dest="max_clusters")
    args = parser.parse_args()

    frame = district_behaviour()
    print(f"districts with enough wells and years: {len(frame)}\n")

    X = StandardScaler().fit_transform(frame[list(FEATURES)].to_numpy())
    k, scores = choose_k(X, args.max_clusters)
    print("how many groups the data supports:")
    print(scores.round(3).to_string(index=False))
    print(f"\nchosen: {k} groups\n")

    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
    frame["cluster"] = labels
    frame["pc1"], frame["pc2"] = coordinates[:, 0], coordinates[:, 1]

    profile = describe(frame, labels)
    print("what each group looks like (medians):")
    print(profile[["districts", *FEATURES]].round(2).to_string())
    print("\nand how each differs from the national median:")
    print(profile[[f"{c}_vs_national" for c in FEATURES]].round(2).to_string())

    print("\nwhere each group is, by state:")
    for cluster, group in frame.groupby("cluster"):
        states = group["state"].value_counts().head(4)
        print(f"   group {cluster} ({len(group)} districts): " + ", ".join(f"{s} {n}" for s, n in states.items()))

    real = silhouette_score(X, labels)
    null = null_silhouette(X, k)
    held = stability(X, k)
    print(f"\nhow much of this to believe:")
    print(f"   separation      silhouette {real:.3f} against {null:.3f} for data with the same spread and no structure")
    print(f"   stability       dropping a fifth of the districts reproduces the grouping at "
          f"{held['mean_agreement']:.2f} on average, worst {held['worst_agreement']:.2f}")
    print("   so these are tendencies along a continuum, not district types: use them to describe a")
    print("   district's behaviour, not to assign it to a category or to drive a decision on its own.")

    out = DATA_PROCESSED / "district_clusters.csv"
    frame.to_csv(out, index=False)
    profile.to_csv(DATA_PROCESSED / "district_cluster_profiles.csv")
    print(f"\nwrote {out.name} and district_cluster_profiles.csv to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
