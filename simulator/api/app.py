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
    GET  /forecast                  depth to water from 2022 to a chosen year, rain only and
                                    rain + 2015-22 trend, from the precomputed table
    POST /simulate                  change the rain on any past or future days and re-run the
                                    model on the readings it reaches

Pressure is ranked on held-out years only (2015-2022) unless training years are
asked for explicitly. Residuals on training years were fitted by the model, so
they are biased towards zero (see models/MODEL_REVIEW.md, section 3).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecast"))
from engine import SEASONS, W as WEEKS, Edit, Engine, reading_date, steps_between  # noqa: E402

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


# ---------------------------------------------------------------- places
class Place(BaseModel):
    """One of: well, district (+ state if the name repeats), state, or lat/lon with
    a radius. Nothing means all India."""
    well: Optional[str] = Field(None, description="well id (lat_lon) or index")
    district: Optional[str] = None
    state: Optional[str] = None
    lat: Optional[float] = Field(None, ge=5, le=38)
    lon: Optional[float] = Field(None, ge=66, le=99)
    radius_km: float = Field(50, gt=0, le=300)


def select_wells(p: Place) -> tuple[np.ndarray, str]:
    s = store()
    if p.well is not None:
        i = s.well_ix.get(p.well)
        if i is None and p.well.isdigit() and int(p.well) < len(s.wells):
            i = int(p.well)
        if i is None:
            raise HTTPException(404, f"no well {p.well!r}")
        w = s.wells[i]
        return np.asarray([i]), f"well {w['id']}, {w['district']}"
    if p.lat is not None and p.lon is not None:
        lat = np.radians([w["lat"] for w in s.wells]); lon = np.radians([w["lon"] for w in s.wells])
        la, lo = np.radians(p.lat), np.radians(p.lon)
        km = 12742 * np.arcsin(np.sqrt(np.sin((lat - la) / 2) ** 2 + np.cos(lat) * np.cos(la) * np.sin((lon - lo) / 2) ** 2))
        idx = np.flatnonzero(km <= p.radius_km)
        if not len(idx):
            raise HTTPException(404, f"no monitored well within {p.radius_km:g} km of {p.lat:.3f}, {p.lon:.3f}")
        return idx, f"{len(idx)} wells within {p.radius_km:g} km of {p.lat:.2f}°N {p.lon:.2f}°E"
    in_state = lambda w: p.state is None or w["state"].lower() == p.state.lower()
    if p.district is not None:
        want = p.district.strip().lower()
        idx = [i for i, w in enumerate(s.wells) if w["district"].lower() == want and in_state(w)]
        if not idx:          # "Bengaluru" -> Bengaluru Urban and Bengaluru Rural
            idx = [i for i, w in enumerate(s.wells) if w["district"].lower().startswith(want) and in_state(w)]
        if not idx:
            raise HTTPException(404, f"no monitored wells in a district called {p.district!r}")
        return np.asarray(idx), " + ".join(sorted({f"{s.wells[i]['district']}, {s.wells[i]['state']}" for i in idx}))
    if p.state is not None:
        idx = [i for i, w in enumerate(s.wells) if in_state(w)]
        if not idx:
            raise HTTPException(422, f"unknown state {p.state!r}; one of {s.states}")
        return np.asarray(idx), s.wells[idx[0]]["state"]
    return np.arange(len(s.wells)), "All India"


# ---------------------------------------------------------------- forecast
SKILL_YEARS = 5          # backtest: beats "no change" up to 5 years ahead, not beyond
HISTORY_FROM = 2012


class Forecast:
    """The per-step table written by simulator/forecast/build_forecast.py."""

    def __init__(self):
        d = DATA / "forecast"
        if not (d / "index.json").exists():
            raise RuntimeError(f"missing {d}; run python simulator/forecast/build_forecast.py")
        self.idx = json.loads((d / "index.json").read_text())
        nw, self.ny = len(self.idx["start"]), len(self.idx["years"])
        self.shifts = self.idx["shifts"]
        self.deltas = (np.frombuffer((d / "deltas.bin").read_bytes(), np.float16)
                       .reshape(len(self.shifts), nw, self.ny, 4).astype(np.float32))
        self.trend = np.frombuffer((d / "trend.bin").read_bytes(), np.float32).reshape(nw, 4)
        self.start = np.asarray(self.idx["start"])             # (wells, 2): last campaign, depth
        self.y0 = self.idx["years"][0]

    def table(self, pct: float) -> np.ndarray:
        """(wells, years, 4) predicted change at rain shift pct, linear between grid points."""
        sh = self.shifts
        pct = min(max(pct, sh[0]), sh[-1])
        j = min(int(np.searchsorted(sh, pct, side="right")) - 1, len(sh) - 2)
        t = (pct - sh[j]) / (sh[j + 1] - sh[j])
        return self.deltas[j] * (1 - t) + self.deltas[j + 1] * t


FC: Forecast | None = None


def forecast_store() -> Forecast:
    global FC
    if FC is None:
        FC = Forecast()
    return FC


def forecast_paths(wells: np.ndarray, to_year: int, pct: float):
    """Depth paths (members, steps, wells), rain only and rain + trend, from each
    well's last reading. Member m rains future year 2023 + k like year 2000 + (m + k) mod 23."""
    s, fc = store(), forecast_store()
    tab = fc.table(pct)[wells]                                # (w, years, 4)
    trend = fc.trend[wells]                                   # (w, 4)
    depth0 = fc.start[wells, 1].astype(np.float32)
    # wells last read before Nov 2022 first catch up to Nov on the real 2022 rain
    catch = np.zeros(len(wells), np.float32)
    catch_trend = np.zeros(len(wells), np.float32)
    for k, c in enumerate(fc.start[wells, 0].astype(int)):
        cm = s.campaigns[c]
        for si in range(SEASONS.index(cm["season"]) + 1 if cm["year"] == 2022 else 0, 4):
            catch[k] += tab[k, 2022 - fc.y0, si]
            catch_trend[k] += trend[k, si]
    steps = [(y, si) for y in range(2023, to_year + 1) for si in range(4)]
    rain = np.stack([np.cumsum(np.stack([tab[:, (m + y - 2023) % fc.ny, si] for y, si in steps]), axis=0)
                     for m in range(fc.ny)])
    trend_cum = np.cumsum(np.stack([trend[:, si] for _, si in steps]), axis=0)
    base = depth0 + catch
    return base + rain, base + catch_trend + rain + trend_cum, [reading_date(y, si).isoformat() for y, si in steps]


def _bands(paths: np.ndarray):
    """Mean over wells for each member, then 10/50/90th percentile over members."""
    q = np.percentile(paths.mean(axis=2), [10, 50, 90], axis=0)
    return [{"p10": round(float(a), 3), "p50": round(float(b), 3), "p90": round(float(c), 3)} for a, b, c in q.T]


@app.get("/api/forecast", tags=["forecast"])
def forecast(to_year: int = Query(2030, ge=2023, le=2045),
             rain_pct: float = Query(0, ge=-30, le=30, description="shift applied to every replayed year's rain"),
             well: Optional[str] = None, district: Optional[str] = None, state: Optional[str] = None,
             lat: Optional[float] = None, lon: Optional[float] = None, radius_km: float = 50):
    """Depth to water (m below ground, larger = deeper) from the 2022 reading to
    `to_year`: median and 10-90% over 23 replayed rain histories. `rain_plus_trend`
    adds each district's 2015-22 unexplained change per step, i.e. assumes pumping and
    everything else rain does not explain carries on."""
    s, fc = store(), forecast_store()
    wells, label = select_wells(Place(well=well, district=district, state=state, lat=lat, lon=lon, radius_km=radius_km))
    d_rain, d_trend, dates = forecast_paths(wells, to_year, rain_pct)
    hist = {}
    for i in wells:
        for c, depth, *_ in s.wells[i]["h"]:
            if s.campaigns[c]["year"] >= HISTORY_FROM:
                hist.setdefault(c, []).append(depth)
    # campaigns that read only a few of the wells make the mean jump; skip them
    full = max((len(v) for v in hist.values()), default=0)
    history = [{"date": s.campaigns[c]["date"], "depth_m": round(float(np.mean(v)), 3), "wells": len(v)}
               for c, v in sorted(hist.items()) if len(v) >= 0.5 * full]
    d0 = fc.start[wells, 1]
    rows = [{"date": d, "rain_only": r, "rain_plus_trend": t, "beyond_tested_skill": (k + 1) / 4 > SKILL_YEARS}
            for k, (d, r, t) in enumerate(zip(dates, _bands(d_rain), _bands(d_trend)))]
    return {
        "place": label, "wells": int(len(wells)), "to_year": to_year, "rain_pct": rain_pct,
        "start": {"date": "2022-11-15", "depth_m": round(float(d0.mean()), 3)},
        "history": history, "forecast": rows,
        "summary": {"rain_only_change_m": round(float(np.median(d_rain[:, -1].mean(1)) - d0.mean()), 3),
                    "rain_plus_trend_change_m": round(float(np.median(d_trend[:, -1].mean(1)) - d0.mean()), 3)},
        "per_well": [{"index": int(i), "change_m": round(float(v), 3)}
                     for i, v in zip(wells, np.median(d_rain[:, -1], axis=0) - d0)],
        "backtest": fc.idx["backtest"],
    }


# ---------------------------------------------------------------- simulate
ENGINE: Engine | None = None
ROW_BUDGET = 30000       # model rows per request; about 40 s on a laptop CPU


def get_engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine(store().session, store().meta)
    return ENGINE


def _season(d: dt.date) -> str:
    m = d.month
    return "south-west monsoon" if 6 <= m <= 9 else "post-monsoon" if m >= 10 else "winter" if m <= 2 else "pre-monsoon"


class SimulateIn(Place):
    start: dt.date
    end: Optional[dt.date] = Field(None, description="default: the start day")
    rain_pct: float = Field(0, ge=-100, le=500, description="change to the rain on those days, %")
    add_mm: float = Field(0, ge=0, le=500, description="extra rain on each of those days, mm")


@app.post("/api/simulate", tags=["forecast"])
def simulate(req: SimulateIn):
    """Change the rain on a day or range of days, past or future, and re-run the
    model on every reading that rain reaches (up to 104 weeks later). Past dates use
    the recorded rain around the edit; future dates use replayed rain and return a
    median and 10-90% range. Example: {"district": "Bengaluru", "start": "2026-10-09", "add_mm": 40}."""
    end = req.end or req.start
    if end < req.start:
        raise HTTPException(422, "end is before start")
    if req.start < dt.date(2000, 1, 1) or end > dt.date(2045, 12, 31):
        raise HTTPException(422, "dates must fall between 2000-01-01 and 2045-12-31")
    if (end - req.start).days > 730:
        raise HTTPException(422, "change at most two years of rain at once")
    if req.rain_pct == 0 and req.add_mm == 0:
        raise HTTPException(422, "set rain_pct or add_mm")
    wells, label = select_wells(req)
    steps = steps_between(req.start - dt.timedelta(days=1), end + dt.timedelta(days=WEEKS * 7))
    future = any(y > 2022 for y, _ in steps)
    rows = 2 * len(wells) * len(steps)                  # with and without the edit
    if rows > ROW_BUDGET:
        raise HTTPException(422, f"{len(wells)} wells x {len(steps)} readings is too many to run live; "
                                 "choose a state, district or point")
    members = min(23, ROW_BUDGET // rows) if future else 1
    eng = get_engine()
    eff, steps = eng.effect(wells, [Edit(req.start, end, 1 + req.rain_pct / 100, req.add_mm)], members=tuple(range(members)))
    cum = np.cumsum(eff, axis=1)                        # (members, steps, wells): effect on depth
    q = np.percentile(cum.mean(axis=2), [10, 50, 90], axis=0)
    readings = [{"date": reading_date(y, si).isoformat(),
                 "depth_effect_m": {"p10": round(float(a), 4), "p50": round(float(b), 4), "p90": round(float(c), 4)}}
                for (y, si), a, b, c in zip(steps, *q)]
    peak = int(np.argmax(np.abs(q[1]))) if steps else 0
    per_well = np.median(cum[:, peak, :], axis=0) if steps else np.zeros(len(wells))

    ctx = eng.context(wells, req.start, end)
    added = ctx.get("recorded_mm", ctx["normal_mm"]) * req.rain_pct / 100 + req.add_mm * ((end - req.start).days + 1)
    ctx = {k: round(v, 1) for k, v in ctx.items()} | {
        "added_mm": round(added, 1), "season": _season(req.start),
        "added_vs_normal": round(added / ctx["normal_mm"], 2) if ctx["normal_mm"] > 0.5 else None}
    return {
        "place": label, "wells": int(len(wells)), "past": not future, "members": members,
        "edit": {"start": str(req.start), "end": str(end), "rain_pct": req.rain_pct, "add_mm": req.add_mm},
        "rain_context": ctx, "readings": readings, "peak": readings[peak] if readings else None,
        "per_well_at_peak": [{"index": int(i), "depth_effect_m": round(float(v), 4)} for i, v in zip(wells, per_well)],
    }


@app.get("/api", include_in_schema=False)
def api_root():
    return RedirectResponse("/docs")


# the UI, served from the same origin so it can call /api without CORS
app.mount("/", StaticFiles(directory=UI, html=True), name="ui")
