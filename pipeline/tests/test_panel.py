import numpy as np
import pandas as pd
import pytest

from bits_ml import panel as pn


def obs_frame(rows):
    """rows: (well, year, campaign, depth, valid)."""
    frame = pd.DataFrame(rows, columns=["well_uid", "year", "campaign", "depth_mbgl", "valid"])
    frame["month"] = frame["campaign"].map({"Jan": 1, "May": 5, "Aug": 8, "Nov": 11})
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=15))
    return frame


def test_the_reference_normal_uses_training_years_only():
    rows = [("w", y, "Nov", 10.0, True) for y in range(2000, 2015)]
    rows += [("w", y, "Nov", 18.0, True) for y in range(2015, 2020)]
    out = pn.reference_normals(obs_frame(rows))
    assert out["normal_ref_m"].dropna().unique().tolist() == [10.0]
    later = out[out["year"] >= 2015]
    assert (later["anomaly_ref_m"] == 8.0).all()   # read as 8 m deeper, not folded into the normal


def test_invalid_readings_never_reach_the_normal_or_the_panel():
    rows = [("w", y, "Nov", 10.0, True) for y in range(2000, 2010)]
    rows += [("w", 2010, "Nov", 500.0, False)]
    out = pn.reference_normals(obs_frame(rows))
    assert out["normal_ref_m"].dropna().unique().tolist() == [10.0]
    assert 500.0 not in set(out["depth_mbgl"])


def test_a_well_campaign_with_too_few_training_readings_gets_no_normal_but_keeps_its_depth():
    rows = [("thin", y, "Nov", 6.0, True) for y in range(2000, 2003)]
    out = pn.reference_normals(obs_frame(rows))
    assert out["normal_ref_m"].isna().all()
    assert out["anomaly_ref_m"].isna().all()
    assert len(out) == 3           # the raw depth is still published


def cell_panel(rows):
    """rows: (well, cell, anomaly, depth, idw)."""
    frame = pd.DataFrame(rows, columns=["well_uid", "cell_id", "anomaly_ref_m", "depth_mbgl", "idw_weight"])
    frame["month"] = pd.Timestamp("2010-11-01")
    return frame


def test_the_three_cell_summaries_are_computed_on_the_same_wells():
    out = pn.aggregate_cells(cell_panel([
        ("a", "c", 1.0, 5.0, 0.8),
        ("b", "c", 3.0, 30.0, 0.2),
    ])).iloc[0]
    assert out["n_wells_read"] == 2
    assert out["anomaly_mean_m"] == pytest.approx(2.0)
    assert out["anomaly_median_m"] == pytest.approx(2.0)
    assert out["anomaly_idw_m"] == pytest.approx(1.4)     # the near well pulls it down
    assert out["anomaly_spread_m"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_a_cell_month_with_no_reading_produces_no_row():
    out = pn.aggregate_cells(cell_panel([("a", "c", np.nan, np.nan, 1.0)]))
    assert out.empty


def test_a_well_without_an_anomaly_still_counts_toward_the_cell_depth():
    out = pn.aggregate_cells(cell_panel([
        ("a", "c", np.nan, 8.0, 0.5),
        ("b", "c", 2.0, 4.0, 0.5),
    ])).iloc[0]
    assert out["n_wells_read"] == 2
    assert out["depth_mean_m"] == pytest.approx(6.0)
    assert out["anomaly_mean_m"] == pytest.approx(2.0)    # mean of what exists, not of zeros


def test_a_whole_year_of_zeros_and_a_dry_wet_month_are_flagged_missing():
    from bits_ml import rain_panel as rp
    months = pd.date_range("2000-01-01", "2004-12-01", freq="MS")
    rain = np.where(months.month == 7, 300.0, 10.0)
    rain[months.year == 2001] = 0.0                            # a year with no rain at all: missing
    frame = pd.DataFrame({"cell_id": "c", "month": months, "rain_mm": rain, "wet_days": 1.0, "max_day_mm": 1.0})
    frame.loc[frame["month"] == "2002-07-01", "rain_mm"] = 0.0  # a July with none, where July brings 300 mm
    out = rp.flag_suspect_zeros(frame)
    assert out.loc[out["month"].dt.year == 2001, "suspect_zero"].all()
    assert out.loc[out["month"] == "2002-07-01", "suspect_zero"].item()
    assert out["suspect_zero"].sum() == 13 and out.loc[out["suspect_zero"], "rain_mm"].isna().all()


def test_a_dry_month_that_is_normally_dry_is_kept():
    from bits_ml import rain_panel as rp
    months = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    frame = pd.DataFrame({"cell_id": "c", "month": months, "rain_mm": np.where(months.month == 1, 0.0, 50.0),
                          "wet_days": 1.0, "max_day_mm": 1.0})
    assert not rp.flag_suspect_zeros(frame)["suspect_zero"].any()
