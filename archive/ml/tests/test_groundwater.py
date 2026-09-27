import numpy as np
import pandas as pd
import pytest

from bits_ml import groundwater as gw

HAVE_DATA = gw.CGWB_CSV.exists()


def readings_frame(rows):
    """rows: (well, year, campaign, depth)."""
    frame = pd.DataFrame(rows, columns=["well_uid", "year", "campaign", "depth_mbgl"])
    frame["month"] = frame["campaign"].map(gw.CAMPAIGN_MONTH)
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=15))
    return frame


def test_normals_use_training_years_only():
    rows = [("w", y, "Nov", 10.0) for y in range(2000, 2006)] + [("w", y, "Nov", 40.0) for y in range(2006, 2010)]
    out = gw.add_normals(readings_frame(rows), train_years=range(2000, 2006), min_readings=5)
    assert out["normal_m"].unique().tolist() == [10.0]
    later = out[out["year"] >= 2006]
    assert (later["anomaly_m"] == 30.0).all()  # test years read as 30 m deeper than normal, not folded into it


def test_well_campaign_without_enough_training_readings_is_dropped():
    rows = [("keep", y, "Nov", 5.0) for y in range(2000, 2006)] + [("thin", y, "Nov", 5.0) for y in range(2000, 2003)]
    out = gw.add_normals(readings_frame(rows), train_years=range(2000, 2006), min_readings=5)
    assert set(out["well_uid"]) == {"keep"}


def test_each_campaign_gets_its_own_normal():
    rows = [("w", y, c, depth) for y in range(2000, 2006) for c, depth in (("May", 12.0), ("Nov", 4.0))]
    out = gw.add_normals(readings_frame(rows), train_years=range(2000, 2006), min_readings=5)
    normals = out.groupby("campaign", observed=True)["normal_m"].first()
    assert normals["May"] == 12.0 and normals["Nov"] == 4.0
    assert (out["anomaly_m"] == 0.0).all()


def test_label_outliers_flags_a_well_far_from_its_district():
    static = pd.DataFrame({
        "district": ["Anantapur"] * 4,
        "lat": [14.5, 14.6, 14.4, 28.6],  # the last one is in Delhi
        "lon": [77.5, 77.6, 77.4, 77.2],
    })
    flagged = gw.label_outliers(static)
    assert flagged.tolist() == [False, False, False, True]


def test_seasonal_moves_sign_convention():
    rows = [("w", 2001, "May", 12.0), ("w", 2001, "Nov", 4.0), ("w", 2002, "May", 13.0), ("w", 2002, "Nov", 6.0)]
    moves = gw.seasonal_moves(readings_frame(rows)).set_index("year")
    assert moves.loc[2001, "monsoon_rise_m"] == pytest.approx(8.0)  # water rose 8 m over the monsoon
    assert moves.loc[2001, "dry_fall_m"] == pytest.approx(-9.0)  # and fell 9 m before the next May
    assert np.isnan(moves.loc[2002, "dry_fall_m"])  # no May 2003 reading to compare with


@pytest.mark.skipif(not HAVE_DATA, reason="CGWB data not present")
def test_real_readings_match_the_non_missing_cells_of_the_csv():
    wells = pd.read_csv(gw.CGWB_CSV, encoding="utf-8-sig")
    cols = [f"{c}-{y % 100:02d}" for y in gw.YEARS for c in gw.CAMPAIGN_MONTH]
    expected = int(wells[cols].apply(pd.to_numeric, errors="coerce").notna().to_numpy().sum())
    readings = gw.load_readings()
    assert len(readings) == expected
    assert readings["depth_mbgl"].notna().all()
    assert set(readings["campaign"]) == set(gw.CAMPAIGN_MONTH)
    assert readings["year"].between(2000, 2022).all()


@pytest.mark.skipif(not HAVE_DATA, reason="CGWB data not present")
def test_may_2020_and_2021_are_almost_empty_so_no_model_should_lean_on_them():
    readings = gw.load_readings()
    static = gw.load_static()
    share = gw.coverage(readings, len(static))
    assert share.loc[2020, "May"] < 5 and share.loc[2021, "May"] < 5
    assert share.loc[2019, "Nov"] > 90
