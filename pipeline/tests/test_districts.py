import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from bits_ml import districts as ds


def two_districts():
    """West and East split a 1 x 1 degree square at longitude 78.0."""
    return pd.DataFrame({
        "district": ["West", "East"],
        "state": ["Andhra Pradesh", "Andhra Pradesh"],
        "geometry": [box(77.0, 15.0, 78.0, 16.0), box(78.0, 15.0, 79.0, 16.0)],
    })


def test_a_cell_on_a_boundary_is_split_by_area():
    cells = pd.DataFrame({"cell_id": ["mid"], "cell_lat": [15.5], "cell_lon": [78.0]})
    out = ds.cell_district_area(cells, two_districts()).set_index("district")
    assert out.loc["West", "frac_of_cell"] == pytest.approx(0.5)
    assert out.loc["East", "frac_of_cell"] == pytest.approx(0.5)
    full_cell_km2 = (0.25 * ds.EARTH_KM_PER_DEG) ** 2 * np.cos(np.radians(15.5))
    assert out["area_km2"].sum() == pytest.approx(full_cell_km2)


def test_district_weights_sum_to_one_and_a_coastal_cell_keeps_only_its_land():
    cells = pd.DataFrame({"cell_id": ["inner", "edge"], "cell_lat": [15.5, 15.5], "cell_lon": [77.5, 76.95]})
    out = ds.cell_district_area(cells, two_districts())
    west = out[out["district"] == "West"].set_index("cell_id")
    assert west["weight_in_district"].sum() == pytest.approx(1.0)
    assert west.loc["edge", "frac_of_cell"] == pytest.approx(0.075 / 0.25)   # most of it is outside every outline
    assert west.loc["edge", "weight_in_district"] < west.loc["inner", "weight_in_district"]


def test_a_cell_outside_every_outline_is_not_invented_a_district():
    cells = pd.DataFrame({"cell_id": ["sea"], "cell_lat": [10.0], "cell_lon": [70.0]})
    assert ds.cell_district_area(cells, two_districts()).empty


def wells(rows):
    """rows: (uid, lat, lon, cgwb district, cgwb state)."""
    return pd.DataFrame(rows, columns=["well_uid", "lat", "lon", "cgwb_district", "cgwb_state"])


def test_wells_are_placed_by_coordinates_not_by_name():
    out = ds.well_district(wells([
        ("a", 15.5, 77.5, "West", "Andhra pradesh"),
        ("b", 15.5, 78.5, "West", "Andhra pradesh"),      # CGWB says West, it stands in East
    ]), two_districts()).set_index("well_uid")
    assert out.loc["a", "district"] == "West" and out.loc["a", "how"] == "inside"
    assert out.loc["b", "district"] == "East"
    assert out.loc["a", "name_agrees"] and not out.loc["b", "name_agrees"]
    assert out["state_agrees"].all()                       # case differs, the state is the same


def test_a_well_just_off_the_outline_takes_the_nearest_and_a_far_one_takes_none():
    out = ds.well_district(wells([
        ("coast", 15.5, 76.98, "West", "Andhra pradesh"),   # about 2 km outside
        ("lost", 25.0, 90.0, "West", "Andhra pradesh"),
    ]), two_districts()).set_index("well_uid")
    assert out.loc["coast", "how"] == "nearest" and out.loc["coast", "district"] == "West"
    assert out.loc["coast", "distance_km"] == pytest.approx(0.02 * ds.EARTH_KM_PER_DEG, rel=0.01)
    assert out.loc["lost", "how"] == "none" and pd.isna(out.loc["lost", "district"])
    assert not out.loc["lost", "state_agrees"]


def test_a_well_in_another_state_is_flagged_but_the_telangana_split_is_not():
    districts = two_districts()
    districts.loc[1, "state"] = "Telangana"
    districts.loc[2] = ["Far", "Maharashtra", box(79.0, 15.0, 80.0, 16.0)]
    out = ds.well_district(wells([
        ("split", 15.5, 78.5, "East", "Andhra pradesh"),
        ("wrong", 15.5, 79.5, "Anantapur", "Andhra pradesh"),
    ]), districts).set_index("well_uid")
    assert out.loc["split", "state_agrees"]
    assert not out.loc["wrong", "state_agrees"]


def test_ingest_takes_location_from_coordinates_and_keeps_the_labels():
    from bits_ml import ingest
    raw = pd.DataFrame({"well_uid": ["a", "b"], "lat": [15.5, 15.5], "lon": [77.5, 78.5],
                        "district": ["West", "Anantapur"], "state": ["Andhra pradesh", "Andhra pradesh"]})
    out = ingest.place_by_coordinates(raw, two_districts()).set_index("well_uid")
    assert out.loc["b", "district"] == "East"                  # where the coordinates put it
    assert out.loc["b", "cgwb_district"] == "Anantapur"        # what the file called it
    assert out.loc["a", "state"] == "Andhra Pradesh" and out.loc["a", "cgwb_state"] == "Andhra pradesh"
