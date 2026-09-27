import numpy as np
import pandas as pd
import pytest

from bits_ml import dataset as ds
from bits_ml import targets as T

pytestmark = pytest.mark.skipif(not ds.gw.CGWB_CSV.exists(), reason="CGWB data not present")


def fake_table(rows):
    """A minimal stand-in for dataset.build(): keys, depths, and zeroed features."""
    frame = pd.DataFrame(rows, columns=["well_uid", "year", "campaign", "depth_mbgl"])
    frame["anomaly_m"] = 0.0
    for column in ds.RAIN_FEATURES + ds.WELL_FEATURES:
        frame[column] = 1.0
    frame["well_type"] = "Dug well"
    frame["aquifer"] = "Unconfined"
    for column, value in (("district", "Alpha"), ("state", "Test"), ("far_from_district", False),
                          ("lat", 20.0), ("lon", 78.0)):
        frame[column] = value
    return frame


def well_years(well, years, may, nov):
    return [(well, y, c, d) for y in years for c, d in (("May", may(y)), ("Nov", nov(y)))]


# --- the arithmetic the sign conventions rest on -------------------------------------------

def test_rise_is_may_minus_november_so_a_recharging_well_scores_positive():
    table = fake_table(well_years("w", range(2001, 2012), may=lambda y: 12.0, nov=lambda y: 4.0))
    frame = T.rise(table, train_years=range(2001, 2012))
    assert (frame["rise_m"] == 8.0).all()  # water 8 m shallower after the monsoon
    assert (frame[T.TARGET] == 0.0).all()  # and exactly normal for this well, every year


def test_annual_change_is_positive_when_the_water_ends_the_year_higher():
    rising = well_years("w", range(2001, 2012), may=lambda y: 10.0, nov=lambda y: 10.0 - (y - 2001))
    frame = T.annual(fake_table(rising), train_years=range(2001, 2012))
    assert (frame["change_m"] == 1.0).all()  # a metre shallower each November
    falling = well_years("d", range(2001, 2012), may=lambda y: 10.0, nov=lambda y: 10.0 + (y - 2001))
    assert (T.annual(fake_table(falling), train_years=range(2001, 2012))["change_m"] == -1.0).all()


def test_annual_row_is_filed_under_the_year_whose_rain_produced_it():
    years = range(2001, 2013)  # enough years to form a normal, which needs five
    rows = well_years("w", years, may=lambda y: 10.0, nov=lambda y: 10.0)
    frame = T.annual(fake_table(rows), train_years=years)
    # the Nov 2001 -> Nov 2002 interval closes in 2002, so 2001 has no row of its own
    assert frame["year"].tolist() == list(range(2002, 2013))


def test_orientation_map_matches_the_targets_and_their_arithmetic():
    assert set(T.WETTER_IS_HIGHER) == set(T.BUILDERS)
    assert T.WETTER_IS_HIGHER["level"] is False  # a depth shrinks as water rises
    assert T.WETTER_IS_HIGHER["rise"] is True and T.WETTER_IS_HIGHER["annual"] is True


# --- leakage and thresholds ----------------------------------------------------------------

def test_deviation_uses_training_years_only():
    rows = well_years("w", range(2001, 2016), may=lambda y: 12.0,
                      nov=lambda y: 4.0 if y <= 2010 else 0.0)  # much wetter after the training years
    frame = T.rise(fake_table(rows), train_years=range(2001, 2011))
    assert frame.loc[frame["year"] <= 2010, T.TARGET].abs().max() == 0.0
    assert (frame.loc[frame["year"] > 2010, T.TARGET] == 4.0).all()  # later years read as 4 m above normal


def test_a_well_with_too_few_training_years_is_dropped():
    thin = well_years("thin", range(2001, 2004), may=lambda y: 12.0, nov=lambda y: 4.0)
    thick = well_years("thick", range(2001, 2012), may=lambda y: 12.0, nov=lambda y: 4.0)
    frame = T.rise(fake_table(thin + thick), train_years=range(2001, 2012))
    assert set(frame["well_uid"]) == {"thick"}


def test_targets_carry_the_features_of_the_closing_campaign():
    table = fake_table(well_years("w", range(2001, 2012), may=lambda y: 12.0, nov=lambda y: 4.0))
    table.loc[table["campaign"] == "May", "rain_12m"] = 999.0  # must not be picked up
    frame = T.rise(table, train_years=range(2001, 2012))
    assert (frame["rain_12m"] == 1.0).all()
    assert "campaign" in frame.columns and set(frame["campaign"].astype(str)) == {"Nov"}


# --- against the real table ----------------------------------------------------------------

@pytest.fixture(scope="module")
def real_table():
    return ds.build()


@pytest.mark.parametrize("name", ["level", "rise", "annual"])
def test_every_target_builds_from_the_real_data_without_gaps(name, real_table):
    frame = T.build_target(name, real_table)
    assert len(frame) > 30_000
    assert frame[T.TARGET].notna().all()
    assert frame[[f for f in ds.FEATURES if f in frame.columns]].notna().to_numpy().all()
    assert frame["year"].between(2001, 2022).all()


def test_the_three_targets_disagree_enough_to_be_worth_comparing(real_table):
    rise = T.build_target("rise", real_table)
    annual = T.build_target("annual", real_table)
    merged = rise.merge(annual, on=["well_uid", "year"], suffixes=("_rise", "_annual"))
    correlation = merged[f"{T.TARGET}_rise"].corr(merged[f"{T.TARGET}_annual"])
    assert 0.0 < correlation < 0.9  # related, since both are rain-fed, but not the same question
