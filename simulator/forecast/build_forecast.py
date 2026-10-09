"""
Builds the forecast table: the simulator's predicted change for every well, past
year and season at several rain shifts, each district's 2015-22 trend, and a
backtest of the chained forecast. Run after simulator/ui/build_ui_data.py.

    python simulator/forecast/build_forecast.py          (or: make forecast; ~18 min on CPU)

The API chains these into forecasts: readings fall on 15 Jan/May/Aug/Nov, the level
after a step is the level before plus the predicted change, and future year
2023 + k of ensemble member m is rained on like past year 2000 + (m + k) mod 23.
The trend is each district's mean of observed minus predicted change per step on
held-out years 2015-22 (regular transitions only); adding it assumes whatever rain
did not explain there carries on.

Writes simulator/ui/data/forecast/
    deltas.bin   float16 (shifts, wells, 23 years, 4 seasons) predicted change
    trend.bin    float32 (wells, 4 seasons) trend per step
    static.bin   float32 (wells, 4, 6) scaled static inputs, then uint8 (wells, 4, 4)
                 category codes, for engine.py
    index.json   layout, rain shifts, each well's last reading and position, backtest
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "models"))
import build_training_data as btd  # noqa: E402
from build_sequences import daily_normals, WET_MM  # noqa: E402

DATA = REPO / "simulator" / "ui" / "data"
OUT = DATA / "forecast"
YEARS = list(range(2000, 2023))                      # replayable record
SEASONS = ["JAN", "MAY", "AUG", "NOV"]               # order within a year
PREV = {"JAN": (-1, "11-15"), "MAY": (0, "01-15"), "AUG": (0, "05-15"), "NOV": (0, "08-15")}
MONTH_DAY = {"JAN": "01-15", "MAY": "05-15", "AUG": "08-15", "NOV": "11-15"}
SHIFTS = [-30, -20, -10, 0, 10, 20, 30]              # rain shift grid, percent
HELD_OUT = range(2015, 2023)
MIN_WELLS = 2                                        # a district rate needs at least this many wells
BATCH = 4096


def season_dates(years):
    """(year, season) -> (reading date, previous regular reading date)."""
    out = {}
    for y in years:
        for s in SEASONS:
            dy, md = PREV[s]
            out[(y, s)] = (pd.Timestamp(f"{y}-{MONTH_DAY[s]}"), pd.Timestamp(f"{y + dy}-{md}"))
    return out


def build_inputs(tab, ids, meta):
    """Raw 6-channel sequences and static inputs for every well x year x season,
    rows ordered (year, season, well)."""
    W = btd.SEQ_WEEKS
    first = tab.drop_duplicates("well_uid").set_index("well_uid").loc[ids]
    xy = first[["lat", "lon"]].reset_index()
    dates, rain = btd.well_daily_rain(xy)
    rain = np.nan_to_num(rain, nan=0.0)

    def cumsum(a):
        return np.vstack([np.zeros((1, a.shape[1]), "float64"), np.cumsum(a, axis=0, dtype="float64")])

    c_rain = cumsum(rain)
    c_anom = cumsum(rain - daily_normals(dates, rain))
    c_wet = cumsum((rain >= WET_MM).astype("float32"))
    doy_all = dates.dayofyear.to_numpy()

    nw = len(ids)
    sd = season_dates(YEARS)
    keys = [(y, s) for y in YEARS for s in SEASONS]
    x = np.zeros((len(keys) * nw, W, 6), "float32")
    days_prev = np.zeros(len(keys) * nw, "float32")
    for k, key in enumerate(keys):
        d, p = sd[key]
        endi = dates.searchsorted(np.datetime64(d), side="right")
        lo = dates.searchsorted(np.datetime64(p), side="right")
        start = endi - W * 7
        assert start >= 0, key
        wk0 = start + 7 * np.arange(W)
        edges = start + 7 * np.arange(W + 1)
        r = slice(k * nw, (k + 1) * nw)
        x[r, :, 0] = np.diff(c_rain[edges], axis=0).T
        x[r, :, 1] = (np.clip(wk0 + 7 - lo, 0, 7) / 7.0)[None, :]
        ang = 2 * np.pi * (doy_all[wk0 + 3] - 1) / 365.25
        x[r, :, 2] = np.sin(ang)[None, :]
        x[r, :, 3] = np.cos(ang)[None, :]
        x[r, :, 4] = np.diff(c_anom[edges], axis=0).T
        x[r, :, 5] = np.diff(c_wet[edges], axis=0).T / 7.0
        days_prev[r] = (d - p).days

    # static numerics: fixed per well, except the seasonal normal and the interval
    st = meta["static"]
    season_norm = (tab.groupby(["well_uid", "season"]).rain_normal_season_mm.mean()
                   .unstack().reindex(index=ids, columns=SEASONS))
    num = np.zeros((len(keys) * nw, len(st["numeric"])), "float32")
    cat = np.zeros((len(keys) * nw, 4), "int64")
    levels = st["categorical"]
    for k, (y, s) in enumerate(keys):
        r = slice(k * nw, (k + 1) * nw)
        cols = {"well_depth_m": first.well_depth_m.to_numpy(), "sy": first.sy.to_numpy(),
                "days_since_prev": days_prev[r],
                "rain_normal_annual_mm": first.rain_normal_annual_mm.to_numpy(),
                "rain_normal_season_mm": season_norm[s].to_numpy(),
                "rain_normal_monsoon_mm": first.rain_normal_monsoon_mm.to_numpy()}
        prev_season = SEASONS[(SEASONS.index(s) - 1) % 4]
        cats = {"season": s, "aquifer": first.aquifer.astype(str).to_numpy(),
                "well_type": first.well_type.astype(str).to_numpy(), "transition": f"{prev_season}_{s}"}
        for j, c in enumerate(st["numeric"]):
            v = np.asarray(cols[c], "float32")
            v = np.where(np.isfinite(v), v, st["median"][j])
            num[r, j] = (v - st["mean"][j]) / st["std"][j]
        for j, c in enumerate(["season", "aquifer", "well_type", "transition"]):
            lv = levels[c]
            v = np.broadcast_to(np.asarray(cats[c], dtype=object), (nw,))
            cat[r, j] = [lv.index(t) if t in lv else len(lv) for t in v]
    return x, num, cat


def scale_seq(x, meta, f):
    """Raw channels with rain multiplied by f, scaled the way the model was trained.
    Same transform as RainScaler (train_sim.py) and Store.scaled (api/app.py)."""
    ch, sc = meta["channels"], meta["channel_scaling"]
    r, a = ch.index("rain_mm"), ch.index("rain_anom_mm")
    out = x.copy()
    rain = np.maximum(x[..., r], 0.0)
    anom = x[..., a] + (f - 1.0) * rain
    out[..., r] = (np.log1p(rain * f) - sc["rain_mm"]["mean"]) / sc["rain_mm"]["std"]
    out[..., a] = (np.sign(anom) * np.log1p(np.abs(anom)) - sc["rain_anom_mm"]["mean"]) / sc["rain_anom_mm"]["std"]
    return out


def predict(sess, x, num, cat, meta, f):
    name = sess.get_outputs()[0].name
    out = np.empty(len(x), "float32")
    for i in range(0, len(x), BATCH):
        s = slice(i, i + BATCH)
        out[s] = sess.run([name], {"seq": scale_seq(x[s], meta, f).astype("float32"),
                                   "num": num[s], "cat": cat[s]})[0]
    return out


def regular(camps, a, b):
    """True when campaign b is the season right after campaign a. Campaign indices
    alone are not enough: a region can miss a campaign, as the sample does."""
    q = lambda c: camps[c]["year"] * 4 + SEASONS.index(camps[c]["season"])
    return q(b) - q(a) == 1


def regular_residuals(wells, camps, years):
    """Per well and season, observed minus rain-expected change over `years`,
    regular transitions only (the previous reading is the previous campaign)."""
    res = [[[] for _ in SEASONS] for _ in wells]
    for i, w in enumerate(wells):
        h = w["h"]
        for prev, cur in zip(h[:-1], h[1:]):
            c = camps[cur[0]]
            if regular(camps, prev[0], cur[0]) and c["year"] in years:
                res[i][SEASONS.index(c["season"])].append(cur[2] - cur[3])
    return res


def district_rates(wells, res):
    """Extraction rate per well and season: its district's mean residual, falling
    back to its state's, then to zero. District, not well, because one well's
    residuals are mostly measurement noise."""
    groups = {}
    for i, w in enumerate(wells):
        for key in (("d", w["district"], w["state"]), ("s", w["state"])):
            g = groups.setdefault(key, [[[] for _ in SEASONS], set()])
            for s in range(4):
                if res[i][s]:
                    g[0][s].append(float(np.mean(res[i][s])))
            if any(res[i]):
                g[1].add(i)
    rate = np.zeros((len(wells), 4), "float32")
    for i, w in enumerate(wells):
        for s in range(4):
            for key in (("d", w["district"], w["state"]), ("s", w["state"])):
                vals, members = groups[key]
                if len(members) >= MIN_WELLS and vals[s]:
                    rate[i, s] = float(np.mean(vals[s]))
                    break
    return rate


def backtest(wells, camps, cidx, D0, rate_fn):
    """Chain the forecast from an observed reading with the rain that actually fell
    and compare with what was observed. Returns depth error by years ahead, for
    the model with and without the trend, and two naive baselines."""
    out = {}
    for start_year, end_year, ext_years in ((2014, 2022, None), (2018, 2022, range(2015, 2019))):
        rate = rate_fn(ext_years) if ext_years else None
        c0 = cidx[(start_year, "NOV")]
        rows = {k: [] for k in ("rain_only", "rain_trend", "persistence", "own_trend")}
        leads = []
        for i, w in enumerate(wells):
            h = {c: d for c, d, *_ in w["h"]}
            if c0 not in h:
                continue
            # the well's own linear trend over the four years before the start, a naive baseline
            past = [(camps[c]["year"] + SEASONS.index(camps[c]["season"]) / 4, d) for c, d, *_ in w["h"]
                    if start_year - 4 < camps[c]["year"] <= start_year]
            slope = np.polyfit(*zip(*past), 1)[0] if len(past) >= 6 else 0.0
            d_rain = d_ext = h[c0]
            for y in range(start_year + 1, end_year + 1):
                for s, name in enumerate(SEASONS):
                    step = D0[i, YEARS.index(y), s]
                    d_rain += step
                    d_ext += step + (rate[i, s] if rate is not None else 0.0)
                    c = cidx.get((y, name))
                    if c in h:
                        t = y + s / 4 - (start_year + 3 / 4)
                        leads.append(int(np.ceil(t)))
                        rows["rain_only"].append(d_rain - h[c])
                        rows["rain_trend"].append(d_ext - h[c])
                        rows["persistence"].append(h[c0] - h[c])
                        rows["own_trend"].append(h[c0] + slope * t - h[c])
        leads = np.asarray(leads)
        res = {"start": f"{start_year}-11-15", "wells": int(sum(1 for w in wells if c0 in {c for c, *_ in w["h"]})),
               "trend_from": [min(ext_years), max(ext_years)] if ext_years else None, "by_years_ahead": []}
        for L in range(1, end_year - start_year + 1):
            m = leads == L
            row = {"years_ahead": L, "n": int(m.sum())}
            for k, v in rows.items():
                if k == "rain_trend" and rate is None:
                    continue
                e = np.asarray(v)[m]
                row[k] = {"mae_m": round(float(np.abs(e).mean()), 3), "bias_m": round(float(e.mean()), 3)}
            res["by_years_ahead"].append(row)
        out[f"from_{start_year}"] = res
    return out


def write_static(tab, ids, num, cat, nw):
    """Static inputs per well and season, and full-precision positions for engine.py."""
    last = len(YEARS) - 1
    n4 = num[last * 4 * nw:(last + 1) * 4 * nw].reshape(4, nw, -1).transpose(1, 0, 2)
    c4 = cat[last * 4 * nw:(last + 1) * 4 * nw].reshape(4, nw, -1).transpose(1, 0, 2)
    (OUT / "static.bin").write_bytes(np.ascontiguousarray(n4, "float32").tobytes()
                                     + np.ascontiguousarray(c4, "uint8").tobytes())
    xy = tab.drop_duplicates("well_uid").set_index("well_uid").loc[ids, ["lat", "lon"]]
    return dict(xy=xy.to_numpy().tolist(),
                static=dict(num=[nw, 4, int(n4.shape[2]), "float32"], cat=[nw, 4, int(c4.shape[2]), "uint8"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA / "sim_meta.json").read_text())
    wells = json.loads((DATA / "wells.json").read_text())
    camps = json.loads((DATA / "campaigns.json").read_text())
    cidx = {(c["year"], c["season"]): i for i, c in enumerate(camps)}
    ids = [w["id"] for w in wells]
    tab = pd.read_parquet(btd.OUT_DIR / "tabular.parquet")

    x, num, cat = build_inputs(tab, ids, meta)
    print(f"  inputs: {len(x):,} rows ({len(wells):,} wells x {len(YEARS)} years x 4 seasons)")
    st = write_static(tab, ids, num, cat, len(wells))

    sess = ort.InferenceSession(str(DATA / "sim.onnx"), providers=["CPUExecutionProvider"])
    nw = len(wells)
    deltas = np.empty((len(SHIFTS), nw, len(YEARS), 4), "float32")
    for p, pct in enumerate(SHIFTS):
        d = predict(sess, x, num, cat, meta, 1 + pct / 100)
        deltas[p] = d.reshape(len(YEARS), 4, nw).transpose(2, 0, 1)
        print(f"  rain {pct:+d}%: mean step {d.mean():+.3f} m")
    (OUT / "deltas.bin").write_bytes(deltas.astype("float16").tobytes())

    rate = district_rates(wells, regular_residuals(wells, camps, HELD_OUT))
    (OUT / "trend.bin").write_bytes(rate.tobytes())
    print(f"  trend: mean {rate.sum(1).mean():+.3f} m/year")

    D0 = deltas[SHIFTS.index(0)]
    # the rebuilt inputs must reproduce build_ui_data.py's predictions on real readings
    diff = [abs(D0[i, YEARS.index(camps[cur[0]]["year"]), SEASONS.index(camps[cur[0]]["season"])] - cur[3])
            for i, w in enumerate(wells) for prev, cur in zip(w["h"][:-1], w["h"][1:]) if regular(camps, prev[0], cur[0])]
    diff = np.asarray(diff)
    print(f"  parity on {len(diff):,} real readings: median |diff| {np.median(diff):.4f} m, "
          f"99th pct {np.percentile(diff, 99):.4f} m")
    assert np.median(diff) < 0.01, "rebuilt inputs do not match the training rows"

    bt = backtest(wells, camps, cidx, D0,
                  lambda yrs: district_rates(wells, regular_residuals(wells, camps, yrs)))
    for k, v in bt.items():
        print(f"  backtest {k}:")
        for r in v["by_years_ahead"]:
            print("    ", r["years_ahead"], {kk: vv["mae_m"] for kk, vv in r.items() if isinstance(vv, dict)})

    start = [w["h"][-1][:2] for w in wells]
    (OUT / "index.json").write_text(json.dumps(dict(
        layout=dict(deltas=["shifts", nw, len(YEARS), 4, "float16"], trend=[nw, 4, "float32"],
                    static=st["static"]), xy=st["xy"],
        shifts=SHIFTS, years=YEARS, seasons=SEASONS, held_out=[min(HELD_OUT), max(HELD_OUT)],
        rain_cube=btd.RAIN_CUBE.relative_to(REPO).as_posix(),      # engine.py reads the same rain
        start=start, backtest=bt), separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
