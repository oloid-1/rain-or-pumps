"""
FastAPI service for the Rain or Pumps simulator.

    uvicorn app:app --app-dir simulator/api --port 8000      (or: make api)
    open http://localhost:8000          the UI, now backed by this API
         http://localhost:8000/docs     interactive API documentation

It serves the same model file as the browser (simulator/ui/data/sim.onnx), so
API and UI give identical numbers; simulator/api/test_api.py checks that. It reads the
data built by simulator/ui/build_ui_data.py and needs nothing else.

Endpoints, all under /api:
    GET  /health                    model, data and parity check
    GET  /wells                     every monitored well, without history
    GET  /wells/{well}              one well with its level history
    POST /scenario                  rain scenario for a November reading, 2015-2022
    GET  /pressure                  district ranking: observed minus rain-expected
    GET  /districts/{name}          one district's yearly observed and expected change
    GET  /rain/{date}               the IMD rainfall grid for one day

Pressure is ranked on held-out years only (2015-2022) unless training years are
asked for explicitly. Residuals on training years were fitted by the model, so
they are biased towards zero (see models/MODEL_REVIEW.md, section 3).
"""

from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

UI = Path(__file__).resolve().parents[1] / "ui"
DATA = UI / "data"
HELD_OUT = range(2015, 2023)          # validation 2015-17 and test 2018-22: never trained on
WATERLOGGED_M = 2.0                   # CGWB: water table within 2 m of the surface
BATCH = 1024


# ---------------------------------------------------------------- data, loaded once
def _json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class Store:
    """Everything the endpoints read, loaded at start-up."""

    def __init__(self):
        missing = [p for p in ("wells.json", "campaigns.json", "pressure.json", "sim.onnx", "sim_meta.json",
                               "scenario/index.json", "rain/grid.json") if not (DATA / p).exists()]
        if missing:
            raise RuntimeError(f"missing {missing} in {DATA}; run python simulator/ui/build_ui_data.py")
        self.wells = _json("wells.json")
        self.campaigns = _json("campaigns.json")
        self.pressure = _json("pressure.json")
        self.meta = _json("sim_meta.json")
        self.scen = _json("scenario/index.json")
        self.grid = _json("rain/grid.json")
        self.well_ix = {w["id"]: i for i, w in enumerate(self.wells)}
        self.states = sorted({w["state"] for w in self.wells})
        self.session = ort.InferenceSession(str(DATA / "sim.onnx"), providers=["CPUExecutionProvider"])
        self.output = self.session.get_outputs()[0].name

    @lru_cache(maxsize=8)
    def inputs(self, year: int):
        """Model-ready rows of the November reading of `year`, from scenario/<year>.bin."""
        idx = self.scen["years"][str(year)]
        n = idx["rows"]
        _, T, C, _ = self.scen["layout"]["seq"]
        nn, nc = self.scen["layout"]["num"][1], self.scen["layout"]["cat"][1]
        buf = (DATA / "scenario" / f"{year}.bin").read_bytes()
        o1 = n * T * C * 2
        o2 = o1 + n * nn * 4
        seq = np.frombuffer(buf, np.float16, n * T * C).astype(np.float32).reshape(n, T, C)
        num = np.frombuffer(buf, np.float32, n * nn, o1).reshape(n, nn)
        cat = np.frombuffer(buf, np.uint8, n * nc, o2).astype(np.int64).reshape(n, nc)
        return seq, num, cat

    def scaled(self, seq: np.ndarray, f: float, rows: np.ndarray) -> np.ndarray:
        """The same sequences with every week's rain multiplied by f, in the model's
        scaled space. Mirrors RainScaler in simulator/train_sim.py and scaledSeq in
        the UI: rain is multiplied, the anomaly moves by (f - 1) * rain, the
        calendar, interval and wet-day channels do not change."""
        out = seq.copy()
        ch, sc = self.meta["channels"], self.meta["channel_scaling"]
        r = ch.index("rain_mm")
        rs = sc["rain_mm"]
        rain = np.maximum(np.expm1(seq[rows, :, r] * rs["std"] + rs["mean"]), 0.0)
        out[rows, :, r] = (np.log1p(rain * f) - rs["mean"]) / rs["std"]
        if "rain_anom_mm" in ch:
            a, s_ = ch.index("rain_anom_mm"), sc["rain_anom_mm"]
            u = seq[rows, :, a] * s_["std"] + s_["mean"]
            anom = np.sign(u) * np.expm1(np.abs(u)) + (f - 1.0) * rain
            out[rows, :, a] = (np.sign(anom) * np.log1p(np.abs(anom)) - s_["mean"]) / s_["std"]
        return out

    def predict(self, seq, num, cat) -> np.ndarray:
        out = []
        for i in range(0, len(seq), BATCH):
            out.append(self.session.run([self.output], {"seq": seq[i:i + BATCH], "num": num[i:i + BATCH],
                                                        "cat": cat[i:i + BATCH]})[0])
        return np.concatenate(out)


STORE: Store | None = None


def store() -> Store:
    global STORE
    if STORE is None:
        STORE = Store()
    return STORE


# ---------------------------------------------------------------- app
app = FastAPI(
    title="Rain or Pumps simulator API",
    version="1.0.0",
    description="Groundwater change under recorded and scenario rainfall for 2,759 CGWB wells, "
                "and the per-district fall that rainfall does not explain. "
                "`delta_h_m` is positive when the water level fell.",
)


class ScenarioIn(BaseModel):
    year: int = Field(2022, ge=min(HELD_OUT), le=max(HELD_OUT),
                      description="November reading to start from; 2015-2022, the years the model never trained on")
    rain_pct: float = Field(0, ge=-50, le=50, description="change applied to the 104 weeks of rain before the reading, in percent")
    state: Optional[str] = Field(None, description="apply the change to wells in this state only; omit for all India")
    details: bool = Field(True, description="include the per-well results, not just the summary")


@app.get("/api/health", tags=["service"])
def health():
    s = store()
    return {
        "status": "ok",
        "model": "BiLSTM simulator, 6 rain channels, rain-response penalty",
        "channels": s.meta["channels"],
        "valid_metrics": s.meta.get("valid"),
        "rain_response_at_plus_20pct": s.meta.get("response_valid", {}).get("x1.2"),
        "wells": len(s.wells),
        "scenario_years": [int(y) for y in s.scen["years"]],
        "held_out_years": [min(HELD_OUT), max(HELD_OUT)],
    }


@app.get("/api/wells", tags=["wells"])
def wells(state: Optional[str] = None, district: Optional[str] = None):
    s = store()
    out = [{k: w[k] for k in ("id", "lat", "lon", "district", "state", "type", "aquifer", "well_depth_m", "sy")}
           | {"index": i, "readings": len(w["h"])}
           for i, w in enumerate(s.wells)
           if (state is None or w["state"] == state) and (district is None or w["district"] == district)]
    return {"count": len(out), "wells": out}


@app.get("/api/wells/{well}", tags=["wells"])
def well(well: str):
    """`well` is the well's id (lat_lon) or its index in /api/wells."""
    s = store()
    i = s.well_ix.get(well)
    if i is None and well.isdigit() and int(well) < len(s.wells):
        i = int(well)
    if i is None:
        raise HTTPException(404, f"no well {well!r}")
    w = s.wells[i]
    hist = [{"date": s.campaigns[c]["date"], "season": s.campaigns[c]["season"], "depth_m": d,
             "change_m": ch, "rain_expected_change_m": ex, "unexplained_m": round(ch - ex, 3),
             "held_out": s.campaigns[c]["year"] in HELD_OUT}
            for c, d, ch, ex in w["h"]]
    return {k: v for k, v in w.items() if k != "h"} | {"index": i, "history": hist}


@app.post("/api/scenario", tags=["scenario"])
def scenario(req: ScenarioIn):
    s = store()
    if req.state is not None and req.state not in s.states:
        raise HTTPException(422, f"unknown state {req.state!r}; one of {s.states}")
    idx = s.scen["years"][str(req.year)]
    seq, num, cat = s.inputs(req.year)
    base = np.asarray(idx["base"], np.float32)
    well_of = np.asarray(idx["well"])
    rows = np.flatnonzero([req.state is None or s.wells[w]["state"] == req.state for w in well_of])
    if len(rows) == 0:
        raise HTTPException(422, f"no wells read in November {req.year} in {req.state}")
    f = 1 + req.rain_pct / 100
    pred = base.copy() if f == 1 else s.predict(s.scaled(seq, f, rows), num, cat)

    depth0 = np.asarray(idx["depth_before"], np.float32)
    p, b = pred[rows], base[rows]
    after, after_base = depth0[rows] + p, depth0[rows] + b
    summary = {
        "year": req.year, "date": idx["date"], "rain_pct": req.rain_pct, "state": req.state, "wells": int(len(rows)),
        "mean_change_m": round(float(p.mean()), 4),
        # positive change = the water fell; the level difference is the negative of the change difference
        "level_vs_recorded_rain_m": round(float(-(p - b).mean()), 4),
        "falling": int((p > 0).sum()), "rising": int((p <= 0).sum()),
        "waterlogged": int((after <= WATERLOGGED_M).sum()),
        "waterlogged_vs_recorded_rain": int((after <= WATERLOGGED_M).sum() - (after_base <= WATERLOGGED_M).sum()),
        "note": "delta_h_m is positive when the water level fell; waterlogged = predicted water table within 2 m of the surface",
    }
    if not req.details:
        return {"summary": summary}
    obs = idx["observed"]
    detail = [{"row": int(i), "index": int(well_of[i]), "id": s.wells[well_of[i]]["id"], "district": s.wells[well_of[i]]["district"],
               "predicted_change_m": round(float(pred[i]), 4), "recorded_rain_change_m": round(float(base[i]), 4),
               "observed_change_m": obs[i], "depth_before_m": float(depth0[i]),
               "depth_after_m": round(float(depth0[i] + pred[i]), 3),
               "waterlogged": bool(depth0[i] + pred[i] <= WATERLOGGED_M)} for i in rows]
    return {"summary": summary, "wells": detail}


@app.get("/api/pressure", tags=["attribution"])
def pressure(from_year: int = Query(min(HELD_OUT), ge=2000, le=2022),
             to_year: int = Query(max(HELD_OUT), ge=2000, le=2022),
             include_training_years: bool = Query(False, description="allow 2000-2014, whose residuals the model was fitted on"),
             top: int = Query(25, ge=1, le=800)):
    """Districts ranked by mean observed change minus the simulator's rain-expected
    change (positive = fell further than rain explains: the extraction proxy).
    A proxy for extraction pressure, not a measurement of it."""
    a, b = min(from_year, to_year), max(from_year, to_year)
    if a < min(HELD_OUT) and not include_training_years:
        raise HTTPException(422, f"{a}-{min(HELD_OUT) - 1} are training years: the model was fitted on them, so "
                                 "their residuals are biased towards zero. Use 2015-2022, or set include_training_years=true.")
    s = store()
    rank = []
    for key, rec in s.pressure.items():
        name, state = key.split("|", 1)
        vals = [rec[str(y)] for y in range(a, b + 1) if str(y) in rec]
        if len(vals) < min(3, b - a + 1):
            continue
        rank.append({"district": name, "state": state, "unexplained_m": round(float(np.mean([v[0] for v in vals])), 4),
                     "observed_m": round(float(np.mean([v[1] for v in vals])), 4),
                     "rain_expected_m": round(float(np.mean([v[2] for v in vals])), 4),
                     "years": len(vals), "wells_latest_year": vals[-1][3]})
    rank.sort(key=lambda r: -r["unexplained_m"])
    return {"from_year": a, "to_year": b, "in_sample": a < min(HELD_OUT), "districts": len(rank), "ranking": rank[:top]}


@app.get("/api/districts/{name}", tags=["attribution"])
def district(name: str, state: Optional[str] = None):
    s = store()
    hits = [k for k in s.pressure if k.split("|")[0].lower() == name.lower() and (state is None or k.split("|")[1].lower() == state.lower())]
    if not hits:
        raise HTTPException(404, f"no district {name!r} with enough monitored wells")
    if len(hits) > 1:
        raise HTTPException(409, f"{name!r} is in several states: {[h.split('|')[1] for h in hits]}; pass state=")
    dname, dstate = hits[0].split("|", 1)
    rows = [{"year": int(y), "unexplained_m": v[0], "observed_m": v[1], "rain_expected_m": v[2], "wells": v[3],
             "held_out": int(y) in HELD_OUT} for y, v in sorted(s.pressure[hits[0]].items(), key=lambda kv: int(kv[0]))]
    return {"district": dname, "state": dstate, "years": rows}


@app.get("/api/rain/{date}", tags=["rainfall"])
def rain(date: dt.date, cells: bool = Query(False, description="include every wet cell as [lat, lon, mm]")):
    s = store()
    g = s.grid
    first = dt.date.fromisoformat(g["first_day"])
    if not (first <= date <= dt.date(2022, 12, 31)):
        raise HTTPException(422, f"rainfall covers {first} to 2022-12-31")
    f = DATA / "rain" / f"{date.year}.bin"
    n = g["n_cells"]
    i = (date - dt.date(date.year, 1, 1)).days
    q = np.frombuffer(f.read_bytes(), np.uint8, n, i * n)
    mm = np.expm1(q / g["quant"])
    out = {"date": date.isoformat(), "mean_mm": round(float(mm.mean()), 2), "max_mm": round(float(mm.max()), 1),
           "heavy_cells_over_64_5mm": int((mm >= 64.5).sum()), "cells_with_rain": int((mm >= 2.5).sum()),
           "source": g["source"]}
    if cells:
        lat, lon = np.asarray(g["lat"]), np.asarray(g["lon"])
        wet = np.flatnonzero(mm >= 0.6)
        out["cells"] = [[float(lat[g["lat_idx"][c]]), float(lon[g["lon_idx"][c]]), round(float(mm[c]), 1)] for c in wet]
    return out


@app.get("/api", include_in_schema=False)
def api_root():
    return RedirectResponse("/docs")


# the UI, served from the same origin so it can call /api without CORS
app.mount("/", StaticFiles(directory=UI, html=True), name="ui")
