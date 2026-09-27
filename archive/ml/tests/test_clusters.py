import numpy as np
import pandas as pd
import pytest

from scripts import clusters as C


def behaviour(rows):
    """rows: (state, district, wells, rain, variability, depth, swing, response, decline)."""
    return pd.DataFrame(rows, columns=["state", "district", "wells", *C.FEATURES])


def separated_groups(per_group=12, spread=0.03, seed=0):
    """Three obviously different kinds of district, with a little noise on each measure."""
    rng = np.random.default_rng(seed)
    centres = {
        "wet_shallow": (2500.0, 0.15, 3.0, 3.0, 0.70, -0.05),
        "dry_deep": (600.0, 0.40, 18.0, 1.0, 0.20, 0.20),
        "middling": (1100.0, 0.25, 8.0, 2.0, 0.45, 0.01),
    }
    rows = []
    for kind, centre in centres.items():
        for i in range(per_group):
            rows.append(("Test", f"{kind}{i}", 5, *[v * (1 + rng.normal(0, spread)) for v in centre]))
    return behaviour(rows)


def standardised(frame):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(frame[list(C.FEATURES)].to_numpy())


def test_choose_k_finds_groups_that_really_are_there():
    frame = separated_groups()
    k, scores = C.choose_k(standardised(frame), max_clusters=6)
    assert k == 3
    assert scores.set_index("k").loc[3, "silhouette"] > 0.6  # genuinely separated data scores high


def test_silhouette_on_real_data_is_far_below_a_clean_split():
    """Guards the claim made in the module: on this data the groups are tendencies, not types.

    Separated synthetic districts score above 0.6. The real districts reach 0.237 against a
    column-shuffled null of 0.184, so nothing here should be presented as a discrete archetype.
    """
    frame = separated_groups()
    _, scores = C.choose_k(standardised(frame), max_clusters=6)
    assert scores["silhouette"].max() > 0.6


def test_describe_reports_every_measure_against_the_national_median():
    frame = separated_groups(per_group=6)
    X = standardised(frame)
    k, _ = C.choose_k(X, max_clusters=5)
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    profile = C.describe(frame, labels)
    assert profile["districts"].sum() == len(frame)
    for column in C.FEATURES:
        assert column in profile.columns and f"{column}_vs_national" in profile.columns
    # the deviations are measured against the national median, so they straddle zero
    deviations = profile[[f"{c}_vs_national" for c in C.FEATURES]].to_numpy()
    assert (deviations > 0).any() and (deviations < 0).any()


def test_stability_agrees_with_itself_on_clean_data_and_is_reported():
    """The module must publish how stable the grouping is, not just the grouping."""
    frame = separated_groups()
    X = standardised(frame)
    stability = C.stability(X, n_clusters=3, resamples=8, seed=0)
    assert stability["mean_agreement"] > 0.8  # separated data regroups the same way every time
    assert 0.0 <= stability["worst_agreement"] <= 1.0


@pytest.mark.skipif(not C.gw.CGWB_CSV.exists() or not C.rm.RAIN_TABLE.exists(),
                    reason="CGWB / rain table not present")
def test_real_districts_build_with_every_measure_present():
    frame = C.district_behaviour()
    assert len(frame) > 150
    assert frame[list(C.FEATURES)].notna().to_numpy().all()
    assert frame["wells"].min() >= C.MIN_WELLS
    assert frame["rain_response"].between(-1, 1).all()
    assert frame["normal_rain_mm"].between(100, 6000).all()
