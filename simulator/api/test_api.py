"""
Tests for the simulator API.

    pytest simulator/api -q

Needs the built UI data (python simulator/ui/build_ui_data.py).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app, store, HELD_OUT

client = TestClient(app)


def scen(**kw):
    r = client.post("/api/scenario", json={"details": False, **kw})
    assert r.status_code == 200, r.text
    return r.json()["summary"]


def test_health():
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["wells"] == 2759
    assert r["scenario_years"] == list(HELD_OUT)


def test_onnx_matches_the_torch_base_predictions():
    """The API's ONNX model, on unscaled inputs, reproduces the predictions that
    build_ui_data.py computed with torch. float16 storage of the inputs allows a
    few millimetres of drift; anything larger means the two have diverged."""
    s = store()
    for year in (2015, 2022):
        seq, num, cat = s.inputs(year)
        p = s.predict(s.scaled(seq, 1.0, np.arange(len(seq))), num, cat)
        base = np.asarray(s.scen["years"][str(year)]["base"], np.float32)
        assert np.abs(p - base).max() < 5e-3


def test_recorded_rain_is_the_baseline():
    s = scen(year=2022, rain_pct=0)
    assert s["level_vs_recorded_rain_m"] == 0 and s["waterlogged_vs_recorded_rain"] == 0


def test_more_rain_raises_the_water_level():
    drought, wet = scen(year=2022, rain_pct=-40), scen(year=2022, rain_pct=20)
    assert drought["level_vs_recorded_rain_m"] < 0 < wet["level_vs_recorded_rain_m"]
    assert drought["mean_change_m"] > scen(year=2022, rain_pct=0)["mean_change_m"] > wet["mean_change_m"]
    # a higher water table puts more wells within 2 m of the surface
    assert wet["waterlogged_vs_recorded_rain"] > 0 > drought["waterlogged_vs_recorded_rain"]


def test_state_filter_limits_the_change_to_that_state():
    s = store()
    out = client.post("/api/scenario", json={"year": 2022, "rain_pct": -40, "state": "Kerala"}).json()
    assert out["summary"]["wells"] == len(out["wells"]) > 0
    assert {s.wells[w["index"]]["state"] for w in out["wells"]} == {"Kerala"}


@pytest.mark.parametrize("body", [{"rain_pct": 80}, {"rain_pct": -51}, {"year": 2010}, {"year": 2023},
                                  {"state": "Atlantis"}])
def test_bad_scenarios_are_rejected(body):
    assert client.post("/api/scenario", json={"year": 2022, **body}).status_code == 422


def test_pressure_defaults_to_held_out_years():
    r = client.get("/api/pressure").json()
    assert (r["from_year"], r["to_year"], r["in_sample"]) == (2015, 2022, False)
    vals = [d["unexplained_m"] for d in r["ranking"]]
    assert vals == sorted(vals, reverse=True)


def test_pressure_refuses_training_years_unless_asked():
    assert client.get("/api/pressure", params={"from_year": 2005}).status_code == 422
    r = client.get("/api/pressure", params={"from_year": 2005, "include_training_years": True}).json()
    assert r["in_sample"] is True


def test_wells_and_districts():
    w = client.get("/api/wells", params={"state": "Kerala"}).json()
    assert w["count"] > 0 and all(x["state"] == "Kerala" for x in w["wells"])
    one = client.get(f"/api/wells/{w['wells'][0]['id']}").json()
    assert one["history"] and {"depth_m", "change_m", "rain_expected_change_m", "held_out"} <= one["history"][0].keys()
    assert client.get("/api/wells/nope").status_code == 404
    top = client.get("/api/pressure", params={"top": 1}).json()["ranking"][0]
    d = client.get(f"/api/districts/{top['district']}", params={"state": top["state"]}).json()
    assert d["years"] and d["state"] == top["state"]
    assert client.get("/api/districts/Nowhere").status_code == 404


def test_rain():
    r = client.get("/api/rain/2005-07-26", params={"cells": True}).json()
    assert r["max_mm"] > 200 and r["cells"] and len(r["cells"][0]) == 3
    assert client.get("/api/rain/1990-01-01").status_code == 422


def test_ui_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Rain or Pumps" in r.text
