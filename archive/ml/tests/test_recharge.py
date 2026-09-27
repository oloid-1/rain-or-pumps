import numpy as np
import pandas as pd
import pytest

from bits_ml import recharge as R


def readings(wells, years=range(2001, 2015), may=12.0, nov=8.0):
    """wells: {name: (district, specific yield)} with a fixed 4 m monsoon rise."""
    rows = []
    for well, (district, sy) in wells.items():
        for year in years:
            for campaign, depth in (("May", may), ("Nov", nov)):
                rows.append({"well_uid": well, "year": year, "campaign": campaign, "depth_mbgl": depth,
                             "state": "Test", "district": district, "far_from_district": False,
                             "sy": sy, "well_type": "Dug well", "aquifer": "Unconfined"})
    return pd.DataFrame(rows)


def with_rain(frame, monsoon_mm):
    """Attach a fixed monsoon to every well-year, standing in for the rain table."""
    moves = R.gw.seasonal_moves(frame)
    static = frame.drop_duplicates("well_uid")[
        ["well_uid", "state", "district", "far_from_district", "sy", "well_type", "aquifer"]]
    out = moves.merge(static, on="well_uid")
    out["monsoon_mm"] = monsoon_mm
    out = out.dropna(subset=["monsoon_rise_m"])
    out = out[(out["monsoon_rise_m"] > R.MIN_RISE_M) & (out["monsoon_mm"] >= R.MIN_MONSOON_MM)]
    out["recharge_mm"] = out["sy"] * out["monsoon_rise_m"] * 1000.0
    out["infiltration_pct"] = out["recharge_mm"] / out["monsoon_mm"] * 100.0
    out["rise_per_100mm"] = out["monsoon_rise_m"] / out["monsoon_mm"] * 100.0
    return out.reset_index(drop=True)


# --- the arithmetic the method rests on -------------------------------------------------------

def test_recharge_is_specific_yield_times_the_rise():
    """4 m of rise at a specific yield of 0.13 is 520 mm of water; against a 1000 mm monsoon, 52%."""
    frame = with_rain(readings({"w": ("Alpha", 0.13)}), monsoon_mm=1000.0)
    assert frame["recharge_mm"].iloc[0] == pytest.approx(520.0)
    assert frame["infiltration_pct"].iloc[0] == pytest.approx(52.0)


def test_infiltration_scales_with_the_specific_yield_it_was_assigned():
    """Sy is a class read off a map, so the headline number inherits that choice directly."""
    low = with_rain(readings({"w": ("Alpha", 0.02)}), 1000.0)["infiltration_pct"].iloc[0]
    high = with_rain(readings({"w": ("Alpha", 0.13)}), 1000.0)["infiltration_pct"].iloc[0]
    assert high / low == pytest.approx(6.5)  # the same water table, six times the reported recharge


def test_a_wetter_monsoon_with_the_same_rise_infiltrates_less():
    dry = with_rain(readings({"w": ("Alpha", 0.05)}), 500.0)["infiltration_pct"].iloc[0]
    wet = with_rain(readings({"w": ("Alpha", 0.05)}), 1500.0)["infiltration_pct"].iloc[0]
    assert dry == pytest.approx(3 * wet)


def test_wells_that_did_not_rise_are_left_out():
    """A falling water table says nothing about recharge under this method."""
    frame = with_rain(readings({"w": ("Alpha", 0.05)}, may=8.0, nov=12.0), 1000.0)
    assert frame.empty


def test_a_trickle_of_rain_cannot_set_the_denominator():
    assert with_rain(readings({"w": ("Alpha", 0.05)}), monsoon_mm=40.0).empty


# --- aggregation ------------------------------------------------------------------------------

def test_district_needs_enough_wells_and_years():
    thin = with_rain(readings({"a": ("Thin", 0.05), "b": ("Thin", 0.05)}), 1000.0)
    full = with_rain(readings({f"c{i}": ("Full", 0.05) for i in range(3)}), 1000.0)
    district = R.by_district(pd.concat([thin, full], ignore_index=True))
    assert district["district"].tolist() == ["Full"]

    short = with_rain(readings({f"d{i}": ("Short", 0.05) for i in range(3)}, years=range(2001, 2005)), 1000.0)
    assert R.by_district(short).empty  # four years is not a district estimate


def test_mislabelled_wells_are_kept_out_of_district_medians():
    frame = with_rain(readings({f"w{i}": ("Alpha", 0.05) for i in range(3)}), 1000.0)
    stray = with_rain(readings({"stray": ("Alpha", 0.13)}), 1000.0)
    stray["far_from_district"] = True
    district = R.by_district(pd.concat([frame, stray], ignore_index=True))
    assert district["wells"].iloc[0] == 3
    # a 4 m rise at 0.05 against a 1000 mm monsoon: 200 mm stored, 20%
    assert district["infiltration_pct"].iloc[0] == pytest.approx(20.0)


def test_reproducibility_reports_agreement_between_halves():
    """Districts that keep their character across the record must come back as agreeing."""
    rng = np.random.default_rng(0)
    rows = []
    # twelve districts: the check needs enough of them for a correlation to mean anything
    for district, sy in [(f"D{n}", 0.02 + 0.01 * n) for n in range(12)]:
        for i in range(4):
            for year in range(2000, 2023):
                # a little year-to-year noise, so the correlation is defined rather than degenerate
                rise = 4.0 * (1 + rng.normal(0, 0.05))
                rows.append({"well_uid": f"{district}{i}", "year": year, "state": "Test", "district": district,
                             "far_from_district": False, "sy": sy, "well_type": "Dug well",
                             "aquifer": "Unconfined", "monsoon_rise_m": rise, "monsoon_mm": 1000.0,
                             "recharge_mm": sy * rise * 1000, "infiltration_pct": sy * rise * 100,
                             "rise_per_100mm": rise / 10})
    check = R.reproducibility(pd.DataFrame(rows))
    assert check["districts"] == 12
    assert check["pearson"] > 0.9


def test_reproducibility_refuses_to_report_a_correlation_it_cannot_compute():
    """Identical districts have no spread, so the agreement is undefined, not perfect."""
    rows = [{"well_uid": f"w{i}", "year": year, "state": "Test", "district": f"D{i}",
             "far_from_district": False, "sy": 0.05, "well_type": "Dug well", "aquifer": "Unconfined",
             "monsoon_rise_m": 4.0, "monsoon_mm": 1000.0, "recharge_mm": 200.0,
             "infiltration_pct": 20.0, "rise_per_100mm": 0.4}
            for i in range(12) for year in range(2000, 2023)]
    check = R.reproducibility(pd.DataFrame(rows))
    assert np.isnan(check["pearson"])
