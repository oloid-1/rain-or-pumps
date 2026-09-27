import numpy as np
import pandas as pd
import pytest

from bits_ml import decisions as D


def readings(wells):
    """wells: {name: (district, well_depth_m, depth at each November)}."""
    rows = []
    for well, (district, well_depth, depths) in wells.items():
        for offset, depth in enumerate(depths):
            rows.append({"well_uid": well, "year": 2001 + offset, "campaign": "Nov", "depth_mbgl": depth,
                         "well_depth_m": well_depth, "state": "Test", "district": district,
                         "far_from_district": False, "well_type": "Dug well"})
    return pd.DataFrame(rows)


def falling(rate, start=5.0, years=20):
    return [start + rate * year for year in range(years)]


def weather_frame(values):
    """What rain alone would have done to each well, as a trend in the level: positive = deepening.

    Attribution supplies this on the same footing as the observed fall, so the two subtract
    cleanly. Passing the leftover gap instead — measured from yearly changes — is what made the
    unexplained part come out larger than the fall itself.
    """
    return pd.DataFrame({"well_uid": list(values), "rain_expected_decline_m_per_year": list(values.values())})


# --- how long the water lasts ---------------------------------------------------------------

def test_years_left_is_the_room_below_the_water_divided_by_the_rate_it_falls():
    outlook = D.well_outlook(readings({"w": ("Alpha", 12.5, falling(0.2))})).iloc[0]
    assert outlook["fall_m_per_year"] == pytest.approx(0.2)
    assert outlook["latest_depth_m"] == pytest.approx(8.8)  # 5.0 + 0.2 * 19
    assert outlook["headroom_m"] == pytest.approx(3.7)  # 12.5 m of well, water at 8.8
    assert outlook["years_left"] == pytest.approx(18.5)


def test_a_well_that_is_not_falling_never_runs_out():
    outlook = D.well_outlook(readings({
        "steady": ("Alpha", 12.5, falling(0.0)),
        "rising": ("Alpha", 12.5, falling(-0.1)),
    })).set_index("well_uid")
    assert outlook.loc["steady", "years_left"] == np.inf
    assert outlook.loc["rising", "years_left"] == np.inf


def test_wells_with_short_records_are_left_out():
    frame = D.well_outlook(readings({
        "short": ("Alpha", 12.5, falling(0.2, years=4)),
        "long": ("Alpha", 12.5, falling(0.2)),
    }))
    assert frame["well_uid"].tolist() == ["long"]


# --- what is taking it ----------------------------------------------------------------------

def test_the_fall_splits_into_the_weather_and_the_part_rain_cannot_explain():
    outlook = D.well_outlook(readings({"w": ("Alpha", 12.5, falling(0.2))}))
    caused = D.add_cause(outlook, weather_frame({"w": 0.05})).iloc[0]
    assert caused["weather_m_per_year"] == pytest.approx(0.05)  # rain alone would take 5 cm a year
    assert caused["unexplained_m_per_year"] == pytest.approx(0.15)  # the rest of the 20 cm it loses
    # close that gap and the well lasts far longer: 3.7 m of room at 0.05 m a year
    assert caused["years_left_if_gap_closed"] == pytest.approx(74.0)


def test_the_parts_always_add_back_to_the_observed_fall():
    """The decomposition is only meaningful if it is one: no part may exceed the whole."""
    outlook = D.well_outlook(readings({"w": ("Alpha", 12.5, falling(0.2))}))
    for weather in (-0.1, 0.0, 0.05, 0.2, 0.3):
        caused = D.add_cause(outlook, weather_frame({"w": weather})).iloc[0]
        assert caused["weather_m_per_year"] + caused["unexplained_m_per_year"] == pytest.approx(
            caused["fall_m_per_year"])


def test_a_well_rain_alone_would_have_raised_never_runs_out_once_the_gap_is_closed():
    outlook = D.well_outlook(readings({"w": ("Alpha", 12.5, falling(0.2))}))
    caused = D.add_cause(outlook, weather_frame({"w": -0.3})).iloc[0]
    assert caused["unexplained_m_per_year"] == pytest.approx(0.5)  # 0.2 observed, rain said -0.3
    assert caused["years_left_if_gap_closed"] == np.inf


# --- which lever applies --------------------------------------------------------------------

def district(name, rate, weather, n=3, well_depth=12.5):
    wells = {f"{name}{i}": (name, well_depth, falling(rate)) for i in range(n)}
    return readings(wells), {f"{name}{i}": weather for i in range(n)}


def outlook_for(*districts):
    frames, weathers = zip(*districts)
    merged = {k: v for d in weathers for k, v in d.items()}
    return D.district_outlook(D.add_cause(D.well_outlook(pd.concat(frames, ignore_index=True)),
                                          weather_frame(merged)))


def test_the_outlook_bands_by_how_long_the_wells_have():
    """Banded by time left, not by cause: the cause split does not survive its own arithmetic."""
    frame = outlook_for(
        district("Soon", rate=0.20, weather=0.05, well_depth=10.3),   # 1.5 m of room: about 7 years
        district("Later", rate=0.20, weather=0.05, well_depth=12.5),  # 3.7 m of room: about 18 years
        district("Slow", rate=0.02, weather=0.01, well_depth=14.0),   # falling, but centuries of room
        district("Steady", rate=0.00, weather=0.00),
    ).set_index("district")
    assert frame.loc["Soon", "outlook"] == "wells fail within 10 years"
    assert frame.loc["Later", "outlook"] == "wells fail within 25 years"
    assert frame.loc["Slow", "outlook"] == "falling, decades of room"
    assert frame.loc["Steady", "outlook"] == "not falling"


def test_no_district_is_labelled_by_cause():
    """The weather and unexplained columns stay in the file, but never become a verdict."""
    frame = outlook_for(district("Pumped", rate=0.20, weather=0.01),
                        district("Rained", rate=0.20, weather=0.19))
    assert "lever" not in frame.columns
    assert set(frame["outlook"]) == {"wells fail within 25 years"}  # same fall, same outlook
    assert {"unexplained_m_per_year", "fall_m_per_year"} <= set(frame.columns)


def test_a_district_needs_three_wells_to_be_ranked():
    frame = outlook_for(district("Thin", rate=0.2, weather=0.05, n=2),
                        district("Full", rate=0.2, weather=0.05, n=3))
    assert frame["district"].tolist() == ["Full"]


def test_urgent_wells_are_counted_against_their_own_bottom():
    # 1.5 m of room left, falling 0.2 m a year: about 7 years, so it counts as urgent
    frame = outlook_for(district("Urgent", rate=0.2, weather=0.05, well_depth=10.3))
    assert frame.iloc[0]["wells_failing_within_10y"] == 3
    assert frame.iloc[0]["median_years_left"] == pytest.approx(7.5)


def test_a_district_of_mostly_rising_wells_has_no_horizon():
    """The median runs over every well. Dropping the rising ones would let the one falling well
    speak for the district, which is how Banaskantha came to report 3.7 years while its median
    well was rising."""
    wells = {f"rise{i}": ("Mixed", 12.5, falling(-0.1)) for i in range(5)}
    wells["fall0"] = ("Mixed", 10.3, falling(0.2))  # distinct names: same name would merge the series
    frame = D.district_outlook(D.add_cause(
        D.well_outlook(readings(wells)), weather_frame({name: 0.05 for name in wells}))).iloc[0]

    assert frame["median_years_left"] == np.inf  # five of six wells are rising
    assert frame["years_left_of_falling_wells"] == pytest.approx(7.5)  # the one that is not
    assert frame["wells_with_a_horizon"] == 1
    assert frame["outlook"] == "not falling"
