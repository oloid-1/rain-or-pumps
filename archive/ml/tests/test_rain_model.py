import numpy as np
import pandas as pd
import pytest

from bits_ml import groundwater as gw
from bits_ml import rain_model as rm

MONTHS = pd.date_range("1998-01-01", "2022-12-01", freq="MS")


def series(rain=None, n_wells=2):
    """A rain series with a known pattern: 10 mm every month unless overridden."""
    values = np.full((len(MONTHS), n_wells), 10.0) if rain is None else rain
    return rm.RainSeries(
        months=MONTHS,
        well_uid=np.array([f"w{i}" for i in range(values.shape[1])]),
        rain=values,
        wet_days=np.full(values.shape, 2.0),
        max_day=np.full(values.shape, 5.0),
    )


def test_history_windows_sum_the_months_before_the_reading():
    features = rm.history_features(series(), 2010, 8)  # August campaign
    assert features["rain_3m"].unique().tolist() == [30.0]  # May, June, July
    assert features["rain_12m"].unique().tolist() == [120.0]
    assert features["rain_36m"].unique().tolist() == [360.0]


def test_rain_falling_in_the_campaign_month_or_later_is_never_used():
    rain = np.full((len(MONTHS), 1), 10.0)
    august_2010 = np.flatnonzero((MONTHS.year == 2010) & (MONTHS.month == 8))[0]
    rain[august_2010:, 0] = 1000.0  # a deluge from the reading month onwards
    features = rm.history_features(series(rain), 2010, 8)
    assert features["rain_3m"].iloc[0] == 30.0
    assert features["max_day_12m"].iloc[0] == 5.0


def test_monsoon_features_point_at_the_right_monsoon():
    rain = np.full((len(MONTHS), 1), 10.0)
    monsoon_2009 = np.isin(MONTHS.month, rm.MONSOON) & (MONTHS.year == 2009)
    monsoon_2010 = np.isin(MONTHS.month, rm.MONSOON) & (MONTHS.year == 2010)
    rain[monsoon_2009, 0] = 100.0
    rain[monsoon_2010, 0] = 200.0

    august = rm.history_features(series(rain), 2010, 8)  # only June and July are in
    assert august["monsoon_todate_mm"].iloc[0] == 400.0
    assert august["monsoon_last_mm"].iloc[0] == 400.0  # the 2009 monsoon, the last completed one

    november = rm.history_features(series(rain), 2010, 11)  # June to September are in
    assert november["monsoon_todate_mm"].iloc[0] == 800.0
    assert november["monsoon_last_mm"].iloc[0] == 800.0


def test_relative_features_average_to_one_hundred_percent_over_training_years():
    features = rm.campaign_features(series(), range(2005, 2015), gw.CAMPAIGN_MONTH)
    out = rm.add_relative(features, train_years=range(2005, 2015))
    assert out["rain_12m_pct"].dropna().round(6).unique().tolist() == [100.0]
    assert out["rain_normal_mm"].unique().tolist() == [120.0]


def test_scenario_scales_each_well_by_its_own_normal_monsoon():
    rain = np.full((len(MONTHS), 2), 10.0)
    rain[:, 1] = 100.0  # a much wetter well
    base = series(rain)
    normals = base.normals(range(2000, 2015))
    scenario = base.with_scenario(2023 - 1, 0.2, normals)  # 2022 stands in for the scenario year

    monsoon = np.isin(MONTHS.month, rm.MONSOON) & (MONTHS.year == 2022)
    np.testing.assert_allclose(scenario.rain[monsoon, 0], 12.0)
    np.testing.assert_allclose(scenario.rain[monsoon, 1], 120.0)  # +20% is ten times more rain at the wetter well
    other = ~np.isin(MONTHS.month, rm.MONSOON) & (MONTHS.year == 2022)
    np.testing.assert_allclose(scenario.rain[other, 0], 10.0)
    assert (scenario.wet_days[monsoon] <= 31).all()


def test_scenario_is_monotonic_in_delta():
    base = series()
    normals = base.normals(range(2000, 2015))
    totals = []
    for delta in (-0.2, 0.0, 0.2, 0.4):
        scenario = base.with_scenario(2022, delta, normals)
        totals.append(rm.history_features(scenario, 2022, 11)["rain_12m"].iloc[0])
    assert totals == sorted(totals)


@pytest.mark.skipif(not rm.RAIN_TABLE.exists(), reason="rain table not built")
def test_real_rain_table_loads_with_one_row_per_well_month():
    real = rm.RainSeries.load()
    assert real.rain.shape == (len(real.months), len(real.well_uid))
    assert real.months[0] == pd.Timestamp("1998-01-01") and real.months[-1] == pd.Timestamp("2022-12-01")
    assert np.isfinite(real.rain).all()
    normals = real.normals(range(2000, 2015))
    assert (normals.monsoon() > 100).all()  # every well gets a real monsoon
