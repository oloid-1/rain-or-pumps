import numpy as np
import pandas as pd
import pytest

from bits_ml import grid as gr
from bits_ml import rainfall as rf

HAVE_IMD = any(gr.IMD_DIR.glob("imd_rf25_*.nc")) if gr.IMD_DIR.exists() else False


def wells_frame(rows):
    """rows: (well, lat, lon, half_width)."""
    frame = pd.DataFrame(rows, columns=["well_uid", "lat", "lon", "coord_half_width_deg"])
    frame["district"] = "D"
    frame["state"] = "S"
    frame["view_modelling"] = True
    return frame


def test_home_cell_is_the_cell_the_well_stands_in():
    # the grid starts at 6.5, 66.5 with a 0.25 step, so 14.30, 78.30 belongs to
    # the cell centred on 14.25, 78.25
    row, col = gr.home_cell([14.30], [78.30])
    assert gr.cell_id(row, col)[0] == "14.25_78.25"


def test_a_point_exactly_between_two_centres_lands_in_one_of_them():
    row, col = gr.home_cell([14.375], [78.25])
    assert gr.cell_id(row, col)[0] in {"14.25_78.25", "14.5_78.25"}


def test_cell_id_round_trips_through_the_grid():
    rows, cols = np.array([0, 40, 128]), np.array([0, 60, 134])
    ids = gr.cell_id(rows, cols)
    lat = [float(i.split("_")[0]) for i in ids]
    lon = [float(i.split("_")[1]) for i in ids]
    back_row, back_col = gr.home_cell(lat, lon)
    assert (back_row == rows).all() and (back_col == cols).all()


def test_distance_weighting_favours_the_well_nearest_the_centre_and_sums_to_one():
    wells = wells_frame([("near", 14.25, 78.26, 0.0), ("far", 14.36, 78.36, 0.0)])
    out = gr.cell_weights(wells).set_index("well_uid")
    assert out.loc["near", "idw_weight"] > out.loc["far", "idw_weight"]
    assert out["idw_weight"].sum() == pytest.approx(1.0)
    assert (out["n_wells_in_cell"] == 2).all()


def test_a_well_on_the_centre_does_not_take_the_whole_weight():
    wells = wells_frame([("centre", 14.25, 78.25, 0.0), ("other", 14.30, 78.30, 0.0)])
    out = gr.cell_weights(wells).set_index("well_uid")
    assert out.loc["centre", "idw_weight"] < 0.9   # the floor keeps a zero distance finite


def test_a_lone_well_carries_its_cell_entirely():
    out = gr.cell_weights(wells_frame([("only", 20.1, 77.1, 0.0)]))
    assert out["idw_weight"].iloc[0] == pytest.approx(1.0)


def test_the_monthly_spine_holds_a_level_in_four_months_of_twelve():
    months = gr.dim_month(2000, 2002)
    assert len(months) == 36
    assert int(months["is_campaign_month"].sum()) == 12
    assert set(months.loc[months["is_campaign_month"], "campaign"]) == {"Jan", "May", "Aug", "Nov"}
    # a water year runs June to May, so a monsoon and the dry season it feeds stay together
    june = months[(months["year"] == 2001) & (months["month_no"] == 6)].iloc[0]
    march = months[(months["year"] == 2002) & (months["month_no"] == 3)].iloc[0]
    assert june["water_year"] == march["water_year"] == 2001


@pytest.mark.skipif(not HAVE_IMD, reason="IMD grids not present")
def test_rain_weights_cover_the_whole_well_and_only_land():
    # sitting just above a grid node, so the rounding box reaches across it
    wells = wells_frame([("inland", 20.02, 77.02, 0.0), ("rounded", 20.02, 77.02, 0.05)])
    bridge = gr.bridge_well_cell(wells)
    totals = bridge.groupby("well_uid")["weight_land"].sum()
    assert totals.loc["inland"] == pytest.approx(1.0)
    assert totals.loc["rounded"] == pytest.approx(1.0)
    # a rounded coordinate is read over its rounding box, so it touches more cells
    assert (bridge["well_uid"] == "rounded").sum() > (bridge["well_uid"] == "inland").sum()
    land = gr.land_mask()
    ids = gr.cell_id(*np.divmod(np.arange(rf.IMD.size), rf.IMD.nlon))
    on_land = pd.Series(land, index=ids)
    assert on_land.reindex(bridge.loc[bridge["weight_land"] > 0, "cell_id"]).all()
