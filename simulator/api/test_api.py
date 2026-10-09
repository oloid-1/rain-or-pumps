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


# ---------------------------------------------------------------- forecast and simulate
# these also need python simulator/forecast/build_forecast.py

def test_forecast_bands_are_ordered_and_start_at_the_last_reading():
    r = client.get("/api/forecast", params={"state": "Punjab", "to_year": 2030}).json()
    f = r["forecast"]
    assert f[0]["date"] == "2023-01-15" and f[-1]["date"] == "2030-11-15"
    for k in ("rain_only", "rain_plus_trend"):
        assert all(x[k]["p10"] <= x[k]["p50"] <= x[k]["p90"] for x in f)
    assert any(x["beyond_tested_skill"] for x in f) and not f[0]["beyond_tested_skill"]


def test_forecast_less_rain_means_deeper_water():
    dry = client.get("/api/forecast", params={"to_year": 2026, "rain_pct": -20}).json()["summary"]
    wet = client.get("/api/forecast", params={"to_year": 2026, "rain_pct": 20}).json()["summary"]
    assert dry["rain_only_change_m"] > wet["rain_only_change_m"]


def test_simulate_past_monsoon_plus_15_percent_raises_the_water():
    r = client.post("/api/simulate", json={"state": "Kerala", "start": "2019-06-01", "end": "2019-09-30",
                                           "rain_pct": 15}).json()
    assert r["past"] and r["members"] == 1
    assert r["peak"]["depth_effect_m"]["p50"] < 0         # negative: water table higher


def test_simulate_rain_today_in_bengaluru_is_a_future_ensemble():
    r = client.post("/api/simulate", json={"district": "Bengaluru", "start": "2026-10-09", "add_mm": 40}).json()
    assert "Bengaluru Urban" in r["place"] and not r["past"] and r["members"] > 1
    assert r["readings"][0]["date"] == "2026-11-15"
    assert r["peak"]["depth_effect_m"]["p50"] <= 0
    assert r["rain_context"]["added_mm"] == 40 and r["rain_context"]["season"] == "post-monsoon"


@pytest.mark.parametrize("body", [{"start": "2019-01-01"}, {"start": "2019-01-01", "end": "2018-01-01", "rain_pct": 10},
                                  {"start": "2019-01-01", "rain_pct": 10, "district": "Atlantis"}])
def test_bad_simulations_are_rejected(body):
    assert client.post("/api/simulate", json=body).status_code in (404, 422)
