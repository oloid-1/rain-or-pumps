import numpy as np
import pandas as pd
import pytest

from bits_ml import seasons as ss


def test_seasons_are_cut_on_the_15th_and_tile_the_year_without_gaps():
    dim = ss.dim_season("2000-01-01", "2001-12-31")
    one_year = dim[dim["water_year"] == 2000]           # Aug-2000, Nov-2000, Jan-2001, May-2001
    assert one_year["n_days"].sum() == 365
    assert (dim["start"].iloc[1:].to_numpy() == (dim["end"].iloc[:-1] + pd.Timedelta(days=1)).to_numpy()).all()
    aug = dim[(dim["season"] == "May-Aug") & (dim["season_year"] == 2000)].iloc[0]
    assert (aug["start"], aug["end"], aug["n_days"]) == (pd.Timestamp("2000-05-16"), pd.Timestamp("2000-08-15"), 92)
    assert dim[(dim["season"] == "Jan-May") & (dim["season_year"] == 2000)]["n_days"].item() == 121  # leap year


def test_only_seasons_wholly_inside_the_rain_record_are_kept():
    dim = ss.dim_season("1998-01-01", "2022-12-31")
    assert dim.iloc[0][["season", "season_year"]].tolist() == ["Jan-May", 1998]   # Nov-Jan 1998 starts in 1997
    assert dim.iloc[-1][["season", "season_year"]].tolist() == ["Aug-Nov", 2022]  # Nov-Jan 2023 ends in 2023
    assert len(dim) == 99


def test_a_reading_day_belongs_to_the_season_it_closes():
    dim = ss.dim_season("2000-01-01", "2000-12-31")
    days = pd.to_datetime(["2000-05-15", "2000-05-16", "2000-08-15", "2000-08-16", "2000-01-10"])
    idx = ss.season_of_days(days, dim)
    names = [dim.set_index("season_idx")["season"].get(i, None) for i in idx]
    assert names == ["Jan-May", "May-Aug", "May-Aug", "Aug-Nov", None]   # Jan 10 2000 closes a season that began in 1999
    assert idx[-1] == -1


def test_accumulate_sums_across_files_and_leaves_missing_days_countable():
    dim = ss.dim_season("2000-01-01", "2000-12-31")
    may_aug = dim[dim["season"] == "May-Aug"]["season_idx"].item()
    first = pd.date_range("2000-05-16", "2000-06-30")
    second = pd.date_range("2000-07-01", "2000-08-15")
    a = np.full((len(first), 2), 3.0)
    b = np.full((len(second), 2), 1.0)
    b[0, 1] = np.nan
    b[5, 0] = 40.0
    acc = ss.accumulate(first, a, dim)
    acc = ss.accumulate(second, b, dim, acc)
    assert acc["rain_mm"][may_aug, 0] == pytest.approx(3 * 46 + 1 * 45 + 40)
    assert acc["wet_days"][may_aug, 0] == 47                  # 46 days of 3 mm and the 40 mm day
    assert acc["max_day_mm"][may_aug, 0] == 40.0
    assert acc["n_days_observed"][may_aug].tolist() == [92, 91]


def season_panel(depths, rain, sy=0.1):
    """One well over consecutive seasons: depth (NaN = no valid reading) and rain per season."""
    return pd.DataFrame({"well_uid": "w", "season_idx": np.arange(len(depths)),
                         "depth_mbgl": depths, "rain_mm": rain, "sy": sy})


def test_change_and_storage_have_opposite_signs_and_the_right_units():
    out = ss.add_changes(season_panel([5.0, 3.0, 4.0], [0.0, 400.0, 30.0]))
    rise = out.iloc[1]
    assert rise["delta_h_m"] == pytest.approx(-2.0)            # water came up 2 m
    assert rise["storage_change_mm"] == pytest.approx(200.0)   # 2 m x 0.1 x 1000
    assert rise["recharge_ratio"] == pytest.approx(0.5)
    fall = out.iloc[2]
    assert fall["storage_change_mm"] == pytest.approx(-100.0)
    assert np.isnan(out.iloc[0]["delta_h_m"])                  # nothing before the first reading


def test_a_missing_reading_is_bridged_not_filled_and_rain_covers_the_whole_span():
    out = ss.add_changes(season_panel([5.0, np.nan, 4.0], [10.0, 300.0, 100.0]))
    assert np.isnan(out.iloc[1]["depth_mbgl"])
    last = out.iloc[2]
    assert last["span_seasons"] == 2
    assert last["depth_start_m"] == 5.0
    assert last["rain_span_mm"] == pytest.approx(400.0)        # both seasons since the reading, not the first


def test_a_gap_longer_than_a_year_gives_no_change():
    depths = [5.0] + [np.nan] * ss.MAX_SPAN_SEASONS + [4.0]
    out = ss.add_changes(season_panel(depths, [100.0] * len(depths)))
    assert out["delta_h_m"].isna().all()
    assert out["span_seasons"].isna().all()


def test_no_ratio_is_written_for_a_dry_span():
    out = ss.add_changes(season_panel([5.0, 5.5], [0.0, ss.MIN_RAIN_FOR_RATIO_MM - 1]))
    assert out.iloc[1]["storage_change_mm"] == pytest.approx(-50.0)
    assert np.isnan(out.iloc[1]["recharge_ratio"])


def test_changes_never_cross_from_one_well_to_the_next():
    a = season_panel([5.0, 4.0], [100.0, 100.0])
    b = season_panel([20.0, 21.0], [100.0, 100.0]).assign(well_uid="x")
    out = ss.add_changes(pd.concat([b, a], ignore_index=True))
    firsts = out.groupby("well_uid").head(1)
    assert firsts["delta_h_m"].isna().all()
    assert out.set_index("well_uid").loc["x", "delta_h_m"].dropna().tolist() == [1.0]


def test_well_rain_is_the_weighted_sum_of_its_cells():
    bridge = pd.DataFrame({"well_uid": ["w", "w", "w"], "cell_id": ["a", "b", "sea"],
                           "weight_land": [0.75, 0.25, 0.0]})
    cells = pd.DataFrame({"cell_id": ["a", "b"], "season_idx": [0, 0], "rain_mm": [100.0, 200.0],
                          "wet_days": [10.0, 20.0], "max_day_mm": [30.0, 50.0], "rain_normal_mm": [100.0, 100.0]})
    out = ss.well_rain(bridge, cells, ["w"]).iloc[0]
    assert out["rain_mm"] == pytest.approx(125.0)
    assert out["wet_days"] == pytest.approx(12.5)
    assert out["rain_pct_normal"] == pytest.approx(125.0)


def test_rain_normals_use_training_years_only():
    frame = pd.DataFrame({"cell_id": "c", "season": "May-Aug",
                          "season_year": [2000, 2001, 2020], "rain_mm": [400.0, 600.0, 5000.0]})
    out = ss.add_rain_normals(frame, "cell_id", train_years=range(2000, 2015))
    assert out["rain_normal_mm"].unique().tolist() == [500.0]
    assert out["rain_pct_normal"].tolist() == pytest.approx([80.0, 120.0, 1000.0])


def test_a_missing_cell_is_left_out_of_a_well_average_not_read_as_dry():
    bridge = pd.DataFrame({"well_uid": ["w", "w"], "cell_id": ["a", "b"], "weight_land": [0.75, 0.25]})
    cells = pd.DataFrame({"cell_id": ["a", "b"], "season_idx": [0, 0], "rain_mm": [np.nan, 200.0],
                          "wet_days": [np.nan, 20.0], "max_day_mm": [np.nan, 50.0], "rain_normal_mm": [100.0, 100.0]})
    out = ss.well_rain(bridge, cells, ["w"]).iloc[0]
    assert out["rain_mm"] == pytest.approx(200.0)       # all weight moves to the cell with data
    assert out["rain_cover"] == pytest.approx(0.25)


def test_suspect_days_are_blanked_so_the_season_is_incomplete_not_dry():
    dim = ss.dim_season("2000-01-01", "2000-12-31")
    may_aug = dim[dim["season"] == "May-Aug"]["season_idx"].item()
    days = pd.date_range("2000-05-16", "2000-08-15")
    daily = np.full((len(days), 1), 5.0)
    daily[np.asarray(days.month == 7)] = np.nan              # what fact_cell_season does to a flagged month
    acc = ss.accumulate(days, daily, dim)
    assert acc["n_days_observed"][may_aug, 0] == len(days) - 31
