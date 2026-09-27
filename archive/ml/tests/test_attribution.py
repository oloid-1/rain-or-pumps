import numpy as np
import pandas as pd
import pytest

from bits_ml import attribution as A
from bits_ml import whatif as W


def residual_frame(specs, years=range(2003, 2023), noise=0.0, seed=0):
    """specs: {well: (district, metres a year lost beyond rain, far_from_district)}."""
    rng = np.random.default_rng(seed)
    rows = []
    for well, (district, shortfall, far) in specs.items():
        for year in years:
            value = shortfall + (rng.normal(0, noise) if noise else 0.0)
            rows.append({
                "well_uid": well, "year": year, "district": district, "state": "Test",
                "far_from_district": far, "change_m": -value, "expected_m": 0.0, "shortfall_m": value,
            })
    return pd.DataFrame(rows)


# --- how much a well loses beyond its rain --------------------------------------------------

def test_pressure_is_the_average_shortfall_and_is_called_out_when_it_is_real():
    frame = residual_frame({"losing": ("Alpha", 0.2, False), "steady": ("Alpha", 0.0, False)}, noise=0.05)
    wells = A.well_pressure(frame).set_index("well_uid")
    assert wells.loc["losing", "drift_m_per_year"] == pytest.approx(0.2, abs=0.03)
    assert wells.loc["losing", "p_value"] < 0.01
    assert wells.loc["steady", "drift_m_per_year"] == pytest.approx(0.0, abs=0.03)
    assert wells.loc["steady", "p_value"] > 0.05


def test_a_well_gaining_more_than_rain_explains_reads_negative():
    wells = A.well_pressure(residual_frame({"gaining": ("Alpha", -0.15, False)}, noise=0.04))
    assert wells["drift_m_per_year"].iloc[0] == pytest.approx(-0.15, abs=0.03)


def test_observed_and_rain_expected_are_both_reported_so_the_split_is_visible():
    frame = residual_frame({"w": ("Alpha", 0.2, False)})
    wells = A.well_pressure(frame).iloc[0]
    assert wells["observed_m_per_year"] == pytest.approx(-0.2)  # the well lost 0.2 m a year
    assert wells["rain_expected_m_per_year"] == pytest.approx(0.0)  # rain said it should have held level
    assert wells["drift_m_per_year"] == pytest.approx(0.2)  # the gap between them


def test_wells_with_short_records_are_skipped():
    assert A.well_pressure(residual_frame({"short": ("Alpha", 0.3, False)}, years=range(2018, 2023))).empty


# --- the trend of the accumulating gap -------------------------------------------------------

def test_trend_of_the_gap_recovers_a_steady_loss():
    frame = residual_frame({"w": ("Alpha", 0.2, False)})
    wells = A.residual_level_trend(frame).iloc[0]
    assert wells["drift_m_per_year"] == pytest.approx(0.2)  # the gap widens 0.2 m every year
    assert wells["p_value"] < 0.01


def test_a_single_bad_reading_does_not_move_the_trend():
    """A spurious reading lifts one year's change and drops the next, so the accumulated gap
    steps away and straight back, leaving one stray point. Using every pairwise slope means
    that point cannot carry a well. A genuine lost year, by contrast, is a permanent step in
    the gap and is supposed to move the trend: the water does not come back."""
    steady = residual_frame({"w": ("Alpha", 0.1, False)})
    misread = steady.copy()
    misread.loc[misread["year"] == 2012, "shortfall_m"] += 3.0
    misread.loc[misread["year"] == 2013, "shortfall_m"] -= 3.0
    assert A.residual_level_trend(misread).iloc[0]["drift_m_per_year"] == pytest.approx(
        A.residual_level_trend(steady).iloc[0]["drift_m_per_year"], abs=0.02)


def test_order_matters_to_the_trend_but_not_to_the_average():
    """Why the ranking is not built on the average: it reads the same whatever order the years
    came in, so it cannot tell a well that is steadily losing from one that lost once and held.
    That is the arithmetic behind the -0.44 reversal between halves of the record."""
    frame = residual_frame({"w": ("Alpha", 0.1, False)}, noise=0.5, seed=3)
    shuffled = frame.copy()
    shuffled["shortfall_m"] = np.random.default_rng(1).permutation(shuffled["shortfall_m"].to_numpy())

    assert A.well_pressure(shuffled).iloc[0]["drift_m_per_year"] == pytest.approx(
        A.well_pressure(frame).iloc[0]["drift_m_per_year"])
    assert abs(A.residual_level_trend(shuffled).iloc[0]["drift_m_per_year"]
               - A.residual_level_trend(frame).iloc[0]["drift_m_per_year"]) > 1e-6


def test_a_well_recovering_beyond_rain_gives_a_negative_trend():
    wells = A.residual_level_trend(residual_frame({"w": ("Alpha", -0.15, False)}))
    assert wells["drift_m_per_year"].iloc[0] == pytest.approx(-0.15)


def test_short_records_are_skipped_by_the_trend_too():
    assert A.residual_level_trend(residual_frame({"w": ("Alpha", 0.2, False)}, years=range(2018, 2023))).empty


# --- districts ------------------------------------------------------------------------------

def test_district_ranking_needs_three_wells_and_drops_mislabelled_ones():
    frame = residual_frame({
        "a1": ("Alpha", 0.30, False), "a2": ("Alpha", 0.20, False), "a3": ("Alpha", 0.10, False),
        "a4": ("Alpha", 9.00, True),  # wrong district label: must not move Alpha
        "b1": ("Beta", 0.50, False), "b2": ("Beta", 0.50, False),  # only two wells: not ranked
    })
    ranking = A.district_ranking(A.well_pressure(frame), value_column="drift_m_per_year", n_boot=200)
    assert ranking["district"].tolist() == ["Alpha"]
    assert ranking["drift_m_per_year"].iloc[0] == pytest.approx(0.2)
    assert ranking["wells"].iloc[0] == 3


def test_verdict_separates_losing_recovering_and_unclear():
    losing = A.district_ranking(A.well_pressure(residual_frame(
        {f"s{i}": ("Sink", 0.2 + 0.01 * i, False) for i in range(5)})), value_column="drift_m_per_year", n_boot=400)
    recovering = A.district_ranking(A.well_pressure(residual_frame(
        {f"r{i}": ("Rise", -0.2 - 0.01 * i, False) for i in range(5)})), value_column="drift_m_per_year", n_boot=400)
    assert losing["verdict"].iloc[0] == "extraction beyond rain"
    assert recovering["verdict"].iloc[0] == "recovering beyond rain"


def test_dry_season_check_sees_the_expected_agreement():
    """Districts losing water beyond rain should also draw down harder before May."""
    readings = []
    for district, fall in {"A": -6.0, "B": -4.0, "C": -2.0, "D": -1.0}.items():
        for well in range(3):
            uid = f"{district}{well}"
            for year in range(2001, 2023):
                readings += [{"well_uid": uid, "year": year, "campaign": "May", "depth_mbgl": 10 - fall,
                              "district": district, "state": "Test"},
                             {"well_uid": uid, "year": year, "campaign": "Nov", "depth_mbgl": 10.0,
                              "district": district, "state": "Test"}]
    ranking = pd.DataFrame({"state": "Test", "district": list("ABCD"),
                            "drift_m_per_year": [0.30, 0.15, 0.02, -0.10]})
    rho, _ = A.dry_season_check(ranking, pd.DataFrame(readings))
    assert rho > 0.9


# --- the simulator on top of it --------------------------------------------------------------

class StubSimulator:
    """Stands in for the fitted simulator: every 10% of extra monsoon lifts the water by 0.05 m."""

    def __init__(self, wells, per_ten_percent=0.05):
        self.wells, self.per_ten_percent = wells, per_ten_percent

    def gain(self, delta):
        return pd.Series(self.per_ten_percent * (delta / 0.1), index=self.wells)


def drift_frame(district="Alpha", drift=0.1, wells=("w1", "w2", "w3")):
    return pd.DataFrame({"well_uid": list(wells), "district": district, "state": "Test",
                         "far_from_district": False, "drift_m_per_year": drift})


def test_what_if_compares_the_rain_gain_against_a_year_of_loss():
    simulator = StubSimulator(["w1", "w2", "w3"])
    drift = drift_frame(drift=0.1)
    weak = W.district_gain(simulator, simulator.gain(0.1), drift, "Alpha")
    strong = W.district_gain(simulator, simulator.gain(0.4), drift, "Alpha")
    assert weak["gain_m"] == pytest.approx(0.05)
    assert weak["verdict"].startswith("structural")  # 0.05 m of rain against 0.10 m lost
    assert strong["gain_m"] == pytest.approx(0.20)
    assert strong["verdict"] == "recovers with this rain"


def test_break_even_finds_the_rain_that_cancels_the_loss():
    simulator = StubSimulator(["w1", "w2", "w3"])
    found = W.break_even(simulator, drift_frame(drift=0.1), "Alpha", tolerance=0.001)
    assert found == pytest.approx(0.2, abs=0.01)  # 0.05 m per 10%, so 20% covers 0.10 m


def test_break_even_reports_impossible_when_no_rain_is_enough():
    simulator = StubSimulator(["w1", "w2", "w3"], per_ten_percent=0.001)
    assert W.break_even(simulator, drift_frame(drift=0.5), "Alpha") is None


def test_break_even_is_zero_where_the_water_is_not_being_lost():
    simulator = StubSimulator(["w1", "w2", "w3"])
    assert W.break_even(simulator, drift_frame(drift=-0.05), "Alpha") == 0.0
