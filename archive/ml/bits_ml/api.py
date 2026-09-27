"""The service the dashboard and the agent call.

Four questions, four endpoints: how is this district doing (/district_rank,
/district), what would more rain do (/whatif), and what counts as more rain here
(/rain). Every number comes from the trained model and the attribution tables;
nothing is computed in words.

The first request builds the feature table and fits the simulator, which takes
about a minute; later requests are served from memory.

Usage:  uvicorn bits_ml.api:app --reload --port 8000
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import models as M
from . import rain_model as rm
from . import splits as S
from . import whatif as W
from .config import DATA_PROCESSED, MODELS_DIR, MODEL_PATH
from .dataset import TARGET, build, matrix
from .train import load_bundle

app = FastAPI(title="Rain or Pumps? groundwater attribution", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RANKING_CSV = DATA_PROCESSED / "district_ranking.csv"
DRIFT_CSV = DATA_PROCESSED / "well_drift.csv"


class WhatIf(BaseModel):
    district: str
    state: str | None = None
    delta: float = Field(0.2, ge=-0.5, le=1.5, description="monsoon change, 0.2 means 20% above normal")
    include_break_even: bool = True


@lru_cache(maxsize=1)
def _tables() -> dict:
    if not RANKING_CSV.exists() or not DRIFT_CSV.exists():
        raise HTTPException(503, f"run `python -m bits_ml.attribution` first: {RANKING_CSV.name} is missing")
    return {"ranking": pd.read_csv(RANKING_CSV), "drift": pd.read_csv(DRIFT_CSV)}


@lru_cache(maxsize=1)
def _simulator() -> W.Simulator:
    path = MODEL_PATH if MODEL_PATH.exists() else MODELS_DIR / "lgbm.joblib"
    table = build(S.FINAL_TRAIN_YEARS)
    if path.exists():
        bundle = load_bundle(path)  # warns if it came from a different environment
        model, train_years, categories = bundle["model"], bundle["train_years"], bundle.get("categories")
    else:  # no saved model yet: fit the default one so the service still answers
        rows = np.flatnonzero(np.isin(table["year"].to_numpy(), list(S.FINAL_TRAIN_YEARS)))
        model = M.build("lgbm").fit(matrix(table.iloc[rows]), table[TARGET].to_numpy()[rows])
        train_years, categories = S.FINAL_TRAIN_YEARS, None
    return W.Simulator(model, table, train_years=train_years, categories=categories)


@app.get("/health")
def health():
    saved = MODEL_PATH.exists() or (MODELS_DIR / "lgbm.joblib").exists()
    return {
        "status": "ok",
        "saved_model": saved,
        "attribution_ready": RANKING_CSV.exists() and DRIFT_CSV.exists(),
        "note": "first /whatif call builds features and may take about a minute",
    }


@app.get("/district_rank")
def district_rank(limit: int = 25, state: str | None = None, verdict: str | None = None):
    """Districts ordered by how fast they sink below the level rain explains."""
    ranking = _tables()["ranking"]
    if state:
        ranking = ranking[ranking["state"].str.lower() == state.lower()]
    if verdict:
        ranking = ranking[ranking["verdict"] == verdict]
    return {"districts": ranking.head(limit).to_dict("records"), "total": int(len(ranking))}


@app.get("/district/{name}")
def district(name: str, state: str | None = None):
    """One district: its drift, how much of its trend rain explains, and its wells."""
    tables = _tables()
    rows = tables["ranking"][tables["ranking"]["district"].str.lower() == name.lower()]
    if state:
        rows = rows[rows["state"].str.lower() == state.lower()]
    if rows.empty:
        raise HTTPException(404, f"no ranked district called {name!r} (needs at least 3 wells with long records)")
    record = rows.iloc[0].to_dict()
    wells = tables["drift"].query("district.str.lower() == @name.lower() and not far_from_district")
    record["well_drifts"] = wells[["well_uid", "years", "drift_m_per_year", "p_value"]].round(4).to_dict("records")
    return record


@app.get("/rain/{district_name}")
def rain(district_name: str):
    """What a normal monsoon is worth here, and whether the rain itself is changing."""
    tables = _tables()
    wells = tables["drift"].query("district.str.lower() == @district_name.lower()")["well_uid"]
    if wells.empty:
        raise HTTPException(404, f"no wells known in {district_name!r}")
    series = rm.RainSeries.load()
    keep = np.isin(series.well_uid, wells.to_numpy())
    normals = series.normals(S.FINAL_TRAIN_YEARS)
    trend = rm.trend(series)
    trend = trend[trend["well_uid"].isin(wells)]
    return {
        "district": district_name,
        "normal_monsoon_mm": float(np.median(normals.monsoon()[keep])),
        "normal_annual_mm": float(np.median(normals.annual()[keep])),
        "monsoon_trend_mm_per_year": float(trend["monsoon_mm_per_year"].median()),
        "plus_20_percent_is_mm": float(np.median(normals.monsoon()[keep]) * 0.2),
    }


@app.post("/whatif")
def what_if(request: WhatIf):
    """If this district got that much more monsoon rain, would it recover?"""
    simulator, tables = _simulator(), _tables()
    try:
        answer = W.district_gain(simulator, simulator.gain(request.delta), tables["drift"], request.district, request.state)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    answer["delta"] = request.delta
    if request.include_break_even:
        value = W.break_even(simulator, tables["drift"], request.district, request.state)
        answer["break_even_monsoon"] = value
        answer["break_even_note"] = "beyond +150% of normal" if value is None else f"{value:+.0%} of normal monsoon"
    return answer
