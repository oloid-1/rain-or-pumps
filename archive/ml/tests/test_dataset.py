import numpy as np
import pandas as pd
import pytest

from bits_ml import dataset as ds
from bits_ml import splits as S

HAVE_DATA = ds.gw.CGWB_CSV.exists() and ds.rm.RAIN_TABLE.exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="CGWB / rain table not present")


@pytest.fixture(scope="module")
def table():
    return ds.build()


def test_model_inputs_never_include_the_banned_columns():
    assert not set(ds.FEATURES) & set(ds.FORBIDDEN)
    with pytest.raises(ValueError):
        ds.matrix(pd.DataFrame(columns=ds.FEATURES + ["year"]), features=ds.FEATURES + ["year"])


def test_every_rain_feature_that_should_lower_the_water_is_constrained(table):
    assert set(ds.MONOTONE) <= set(ds.FEATURES)
    assert set(ds.MONOTONE.values()) == {-1}
    for window in ds.rm.WINDOWS:
        assert ds.MONOTONE[f"rain_{window}m"] == -1


def test_changing_the_training_years_moves_the_normals_but_not_the_rain_history(table):
    other = ds.build(train_years=range(2005, 2018))
    keys = ["well_uid", "year", "campaign"]
    merged = table.merge(other, on=keys, suffixes=("_a", "_b"))
    assert len(merged) > 100_000
    np.testing.assert_allclose(merged["rain_12m_a"], merged["rain_12m_b"], rtol=1e-9)
    np.testing.assert_allclose(merged["max_day_12m_a"], merged["max_day_12m_b"], rtol=1e-9)
    assert not np.allclose(merged["normal_m_a"], merged["normal_m_b"])  # normals follow their training years
    assert not np.allclose(merged["rain_12m_pct_a"], merged["rain_12m_pct_b"])


def test_apply_categories_restores_columns_that_arrived_as_text(table):
    """What the helper is for, measured against a real model rather than assumed.

    A table read back from CSV, or rebuilt in a notebook, brings its category
    columns as plain strings, and LightGBM refuses those outright. Restoring the
    saved labels brings the predictions back exactly. Reordering, by contrast,
    needs no help: LightGBM remaps pandas categoricals by label.
    """
    import lightgbm as lgb

    sample = table.sample(4000, random_state=4)
    saved = ds.categories_of(sample)
    features, target = ds.matrix(sample), sample[ds.TARGET].to_numpy()
    model = lgb.LGBMRegressor(n_estimators=20, num_leaves=15, random_state=0, verbose=-1).fit(features, target)
    baseline = model.predict(features)

    as_text = sample.copy()
    for column in saved:
        as_text[column] = as_text[column].astype(str)
    with pytest.raises(ValueError):
        model.predict(ds.matrix(as_text))
    restored = ds.apply_categories(as_text, saved)
    np.testing.assert_array_equal(model.predict(ds.matrix(restored)), baseline)

    reordered = sample.copy()
    for column, values in saved.items():
        reordered[column] = pd.Categorical(reordered[column].astype(str), categories=list(reversed(values)))
    np.testing.assert_array_equal(model.predict(ds.matrix(reordered)), baseline)


def test_apply_categories_cannot_bring_back_a_label_whose_rows_were_dropped(table):
    """The limit worth knowing: lost values stay lost, which is why the round-trip check exists."""
    sample = table.sample(2000, random_state=5)
    saved = ds.categories_of(sample)
    lost = sample.copy()
    text = lost["campaign"].astype(str)
    lost["campaign"] = pd.Categorical(text.where(text != "Aug"), categories=[c for c in saved["campaign"] if c != "Aug"])

    assert lost["campaign"].isna().any()
    assert ds.apply_categories(lost, saved)["campaign"].isna().sum() == lost["campaign"].isna().sum()


def test_apply_categories_without_a_recorded_order_leaves_the_table_alone(table):
    assert ds.apply_categories(table, None) is table


def test_target_is_depth_minus_that_wells_normal(table):
    sample = table.sample(2000, random_state=0)
    np.testing.assert_allclose(sample[ds.TARGET], sample["depth_mbgl"] - sample["normal_m"], rtol=1e-9)


def test_table_has_no_gaps_in_the_model_inputs(table):
    numeric = ds.matrix(table).select_dtypes("number")
    assert numeric.notna().to_numpy().all()


def test_rain_history_stops_before_the_reading(table):
    """A November row must not carry rain from November itself: Jun-Sep is the whole monsoon so far."""
    november = table[table["campaign"] == "Nov"].sample(500, random_state=1)
    assert (november["monsoon_todate_mm"] <= november["rain_6m"] + 1e-6).all()
    august = table[table["campaign"] == "Aug"].sample(500, random_state=1)
    assert (august["monsoon_todate_mm"] <= august["rain_3m"] + 1e-6).all()


def test_year_splits_do_not_overlap(table):
    split = S.year_split(table)
    assert not (split.train & split.val).any()
    assert not (split.train & split.test).any()
    assert not (split.val & split.test).any()
    assert split.train.sum() > split.test.sum() > 10_000
    assert table.loc[split.test, "year"].min() >= min(S.TEST_YEARS)


def test_district_folds_never_share_a_district(table):
    districts = table["district"].to_numpy()
    seen = []
    for fit, held in S.group_folds(table):
        assert not set(districts[fit]) & set(districts[held])
        seen.append(set(districts[held]))
    assert len(set.union(*seen)) == table["district"].nunique()


def test_cross_fit_blocks_cover_every_year_exactly_once(table):
    covered = []
    for fit, held, fit_years in S.cross_fit_blocks(table):
        years = set(table.iloc[held]["year"].tolist())
        assert not years & set(fit_years)  # a block is never fitted on its own years
        covered.append(years)
    assert set.union(*covered) == set(table["year"].unique())
    assert sum(len(y) for y in covered) == len(set.union(*covered))
