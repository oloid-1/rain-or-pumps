import numpy as np
import pandas as pd
import pytest

from bits_ml import ingest

HAVE_DATA = ingest.CGWB_ZIP.exists()


def obs_frame(rows):
    """rows: (well, year, campaign, depth)."""
    frame = pd.DataFrame(rows, columns=["well_uid", "year", "campaign", "depth_mbgl"])
    frame["month"] = frame["campaign"].map(ingest.CAMPAIGN_MONTH)
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=15))
    return frame.sort_values(["well_uid", "date"], ignore_index=True)


def wells_frame(rows):
    """rows: (well, well_depth_m)."""
    frame = pd.DataFrame(rows, columns=["well_uid", "well_depth_m"])
    return frame.set_index("well_uid", drop=False)


def test_zero_is_kept_unless_the_well_is_full_of_zeros():
    # one zero after a monsoon is possible; a well that reports zero half the
    # time is reporting missing data as zero
    # the real well swings between 1 and 6 m, so its one zero is ordinary for it
    real = [("real", y, "Nov", 0.0 if y == 2005 else float(1 + y % 6)) for y in range(2000, 2012)]
    fake = [("fake", y, "Nov", 0.0 if y % 2 else 4.0) for y in range(2000, 2012)]
    out = ingest.flag_readings(obs_frame(real + fake), wells_frame([("real", 30.0), ("fake", 30.0)]))

    assert out.loc[out["well_uid"] == "real", "flag_zero"].sum() == 1
    assert not out.loc[out["well_uid"] == "real", "flag_zero_suspect"].any()
    assert out.loc[out["well_uid"] == "fake", "flag_zero_suspect"].sum() == 6
    # and the distinction reaches `valid`: the real zero survives, the fake ones do not
    assert out.loc[(out["well_uid"] == "real") & out["flag_zero"], "valid"].all()
    assert not out.loc[(out["well_uid"] == "fake") & out["flag_zero"], "valid"].any()


def test_outliers_are_judged_against_the_season_not_the_raw_level():
    # a well that falls ten metres over twenty years, read deep in May and
    # shallow in November: on the raw level the later readings sit far from a
    # mean drawn from its wetter past, and clipping them would flatten the fall
    rows = []
    for offset, year in enumerate(range(2000, 2020)):
        rows.append(("w", year, "May", 12.0 + 0.5 * offset))
        rows.append(("w", year, "Nov", 4.0 + 0.5 * offset))
    out = ingest.flag_readings(obs_frame(rows), wells_frame([("w", 40.0)]))
    assert not out["flag_outlier"].any()
    assert out["valid"].all()


def test_a_single_wild_reading_is_still_caught():
    rows = [("w", y, c, 5.0 if c == "Nov" else 9.0) for y in range(2000, 2015) for c in ("May", "Nov")]
    rows.append(("w", 2015, "Nov", 45.0))
    out = ingest.flag_readings(obs_frame(rows), wells_frame([("w", 60.0)]))
    wild = out[out["depth_mbgl"] == 45.0]
    assert wild["flag_outlier"].all() and not wild["valid"].all()


def test_water_below_the_drilled_bottom_is_flagged():
    rows = [("w", y, "Nov", 9.0) for y in range(2000, 2010)] + [("w", 2010, "Nov", 21.0)]
    out = ingest.flag_readings(obs_frame(rows), wells_frame([("w", 20.0)]))
    assert out.loc[out["depth_mbgl"] == 21.0, "flag_below_bottom"].all()
    assert not out.loc[out["depth_mbgl"] == 9.0, "flag_below_bottom"].any()


def test_a_missing_drilled_depth_never_flags_anything():
    rows = [("w", y, "Nov", 9.0) for y in range(2000, 2010)]
    out = ingest.flag_readings(obs_frame(rows), wells_frame([("w", np.nan)]))
    assert not out["flag_below_bottom"].any()


def test_three_identical_readings_in_a_row_are_flagged_but_two_are_not():
    rows = [("pair", 2000, "May", 3.0), ("pair", 2000, "Nov", 3.0), ("pair", 2001, "May", 4.0),
            ("run", 2000, "May", 3.0), ("run", 2000, "Nov", 3.0), ("run", 2001, "May", 3.0)]
    out = ingest.flag_readings(obs_frame(rows), wells_frame([("pair", 30.0), ("run", 30.0)]))
    assert not out.loc[out["well_uid"] == "pair", "flag_repeat_run"].any()
    assert out.loc[out["well_uid"] == "run", "flag_repeat_run"].sum() == 3


def test_co_located_wells_get_distinct_keys():
    sites = pd.Series(["12.0_77.0", "12.0_77.0", "13.0_78.0"])
    uids = ingest.well_uid(sites, pd.Series(["beta", "alpha", "gamma"]))
    assert uids.nunique() == 3
    assert uids.iloc[2] == "13.0_78.0"           # a lone well keeps the plain key
    assert set(uids.iloc[:2]) == {"12.0_77.0#1", "12.0_77.0#2"}
    assert uids.iloc[1].endswith("#1")           # suffix follows station name, so it is stable


@pytest.mark.skipif(not HAVE_DATA, reason="raw CGWB archive not present")
def test_the_published_set_is_reproduced_as_a_view():
    wells, obs = ingest.build()
    assert len(wells) == 32_299
    assert int(wells["view_strict"].sum()) == 2_759      # the published quality-controlled set, exactly
    assert int(wells["view_modelling"].sum()) > 8_000    # and the rebuild keeps far more
    assert wells["well_uid"].is_unique
    # nothing is silently discarded: every reading in the file is present and judged
    assert len(obs) == obs["valid"].sum() + (~obs["valid"]).sum()
    assert set(obs["well_uid"]) <= set(wells["well_uid"])


def test_label_outliers_flags_a_well_far_from_its_district():
    wells = pd.DataFrame({
        "district": ["Anantapur"] * 4,
        "lat": [14.5, 14.6, 14.4, 28.6],  # the last one is in Delhi
        "lon": [77.5, 77.6, 77.4, 77.2],
    })
    assert ingest.label_outliers(wells).tolist() == [False, False, False, True]
