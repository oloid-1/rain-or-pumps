import numpy as np
import pandas as pd
import pytest

from bits_ml import training_data as td


def test_three_of_four_campaigns_is_enough_for_the_training_view():
    wells = pd.DataFrame({"n_train_jan": [10, 10, 10], "n_train_may": [0, 10, 0],
                          "n_train_aug": [10, 10, 10], "n_train_nov": [10, 10, 2]})
    assert td.training_view(wells).tolist() == [True, True, False]


def history_panel(rain):
    return pd.DataFrame({"well_uid": "w", "season_idx": np.arange(len(rain)),
                         "rain_mm": rain, "rain_normal_mm": 100.0})


def test_rain_history_looks_only_backwards_and_is_missing_before_the_record():
    out = td.add_rain_history(history_panel([float(i) for i in range(1, 13)]))
    assert np.isnan(out.loc[0, "rain_lag1_mm"]) and out.loc[5, "rain_lag1_mm"] == 5.0
    assert np.isnan(out.loc[2, "rain_4s_mm"])                   # only three seasons exist yet
    assert out.loc[3, "rain_4s_mm"] == 1 + 2 + 3 + 4          # ends at the season the reading closes
    assert out.loc[11, "rain_12s_mm"] == sum(range(1, 13))
    assert out.loc[11, "rain_4s_pct_normal"] == pytest.approx(100 * (9 + 10 + 11 + 12) / 400)


def test_history_never_crosses_from_one_well_to_the_next():
    two = pd.concat([history_panel([1.0] * 4), history_panel([7.0] * 4).assign(well_uid="x")], ignore_index=True)
    out = td.add_rain_history(two)
    assert np.isnan(out.set_index("well_uid").loc["x", "rain_lag1_mm"].iloc[0])


def test_splits_are_by_year_and_district_folds_keep_a_district_together():
    frame = pd.DataFrame({"season_year": [2005, 2016, 2020, 2005], "district": ["A", "A", "B", "B"],
                          "state": ["S", "S", "S", "S"]})
    out = td.assign_splits(frame)
    assert out["split"].tolist() == ["train", "valid", "test", "train"]
    assert out.loc[0, "district_fold"] == out.loc[1, "district_fold"]


def test_past_levels_place_and_time_are_never_features():
    td.check_no_leakage(td.feature_columns())
    with pytest.raises(ValueError):
        td.check_no_leakage([*td.feature_columns(), "depth_start_m"])


def test_week_blocks_end_on_the_reading_day_and_a_missing_day_blanks_its_week():
    dates = pd.date_range("2000-01-01", periods=400)
    daily = np.ones((400, 2), dtype=np.float32)
    daily[399 - 3, 1] = np.nan                                # in the last week of cell 1
    out = td.week_blocks(dates, daily, pd.DatetimeIndex([dates[399], dates[100]]), weeks=52)
    assert out.shape == (2, 52, 2)
    assert (out[0, :, 0] == 7).all()
    assert np.isnan(out[0, -1, 1]) and out[0, -2, 1] == 7
    assert np.isnan(out[1]).all()                             # 52 weeks reach before the first day


def test_well_sequence_is_the_weighted_mean_of_its_cells():
    cube = np.zeros((1, 2, 2), dtype=np.float32)
    cube[0, :, 0], cube[0, :, 1] = 10.0, 30.0
    stencil = pd.DataFrame({"well_uid": ["w", "w"], "cell_id": ["a", "b"], "weight_land": [0.75, 0.25]})
    rows = pd.DataFrame({"well_uid": ["w"], "season_idx": [0]})
    out = td.well_sequences(rows, cube, np.array(["a", "b"]), stencil)
    assert out[0].tolist() == pytest.approx([15.0, 15.0])
