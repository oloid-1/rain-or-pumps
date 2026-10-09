"""
Live engine behind /api/simulate: edit daily rain on any days, past or future, and
re-run the simulator on every reading the edit reaches (the 104 weeks after it).

Rain up to 2022 is the IMD record at each well. From 2023 it is past rain replayed:
member m rains year Y like 2000 + (m + Y - 2023) mod 23, and the whole window behind
a reading is shifted by the same number of years, so this matches the precomputed
table from build_forecast.py. The anomaly and wet-day channels are recomputed from
the edited rain, so the season of the edit counts.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "simulator" / "ui" / "data"
FC = DATA / "forecast"

W = 104
SEASONS = ["JAN", "MAY", "AUG", "NOV"]
MONTH_DAY = {"JAN": (1, 15), "MAY": (5, 15), "AUG": (8, 15), "NOV": (11, 15)}
FIRST_YEAR, LAST_YEAR = 2000, 2022
N_REPLAY = LAST_YEAR - FIRST_YEAR + 1
WET_MM = 2.5


@dataclass
class Edit:
    start: dt.date
    end: dt.date
    factor: float = 1.0
    add_mm: float = 0.0       # added per day, after the factor


def reading_date(year: int, s: int) -> dt.date:
    m, d = MONTH_DAY[SEASONS[s]]
    return dt.date(year, m, d)


def steps_between(a: dt.date, b: dt.date):
    """Regular readings (year, season index) after a and up to b."""
    out = []
    for y in range(a.year, b.year + 1):
        for s in range(4):
            d = reading_date(y, s)
            if a < d <= b:
                out.append((y, s))
    return out


def replay_year(member: int, year: int) -> int:
    return FIRST_YEAR + (member + year - (LAST_YEAR + 1)) % N_REPLAY


def _shift_years(d: dt.date, k: int) -> dt.date:
    try:
        return d.replace(year=d.year + k)
    except ValueError:                       # 29 Feb into a common year
        return d.replace(year=d.year + k, day=28)


class Engine:
    def __init__(self, session, meta):
        sys.path.insert(0, str(REPO / "models"))
        import pandas as pd
        import build_training_data as btd
        from build_sequences import daily_normals

        idx = json.loads((FC / "index.json").read_text())
        self.n = len(idx["xy"])
        st = idx["layout"]["static"]
        buf = (FC / "static.bin").read_bytes()
        nnum = st["num"][2]
        self.num = np.frombuffer(buf, np.float32, self.n * 4 * nnum).reshape(self.n, 4, nnum)
        self.cat = (np.frombuffer(buf, np.uint8, self.n * 4 * 4, self.n * 4 * nnum * 4)
                    .astype(np.int64).reshape(self.n, 4, 4))
        btd.RAIN_CUBE = REPO / idx.get("rain_cube", "data/derived/rain_cube.npz")   # full data or sample
        xy = pd.DataFrame(idx["xy"], columns=["lat", "lon"])
        dates, rain = btd.well_daily_rain(xy)
        self.rain = np.nan_to_num(rain, nan=0.0).astype(np.float32)           # (days, wells)
        self.d0 = dates[0].date()
        # normal rain by calendar day (365, wells), so future dates have one too
        doy = np.clip(np.minimum(dates.dayofyear.to_numpy(), 365) - 1
                      - (dates.is_leap_year & (dates.month > 2)), 0, 364)
        self.normal_doy = np.zeros((365, self.n), np.float32)
        self.normal_doy[doy] = daily_normals(dates, self.rain)
        self.session, self.meta = session, meta
        self.output = session.get_outputs()[0].name

    # ------------------------------------------------------------ inputs
    def _day(self, d: dt.date) -> int:
        return (d - self.d0).days

    def _window(self, wells, y, s, member, edits):
        """Daily rain and normals (rows, 728) behind reading (y, s), with edits applied."""
        end = reading_date(y, s)
        shift = 0 if y <= LAST_YEAR else replay_year(member, y) - y
        days = [end - dt.timedelta(days=W * 7 - 1 - i) for i in range(W * 7)]
        src = [self._day(_shift_years(d, shift)) for d in days]
        rain = self.rain[np.asarray(src)][:, wells].T.copy()                   # (rows, 728)
        doy = np.asarray([min(d.timetuple().tm_yday, 365) - 1 - (1 if (d.month > 2 and d.year % 4 == 0) else 0)
                          for d in days]).clip(0, 364)
        normal = self.normal_doy[doy][:, wells].T
        for e in edits:
            m = np.asarray([e.start <= d <= e.end for d in days])
            if m.any():
                rain[:, m] = np.maximum(rain[:, m] * e.factor + e.add_mm, 0.0)
        return rain, normal, days

    def _channels(self, rain, normal, days, s):
        rows = rain.shape[0]
        x = np.zeros((rows, W, 6), np.float32)
        wk = rain.reshape(rows, W, 7)
        x[:, :, 0] = wk.sum(2)
        prev_m, prev_d = MONTH_DAY[SEASONS[(s - 1) % 4]]
        end = days[-1]
        prev = dt.date(end.year - (1 if s == 0 else 0), prev_m, prev_d)
        after = np.asarray([d > prev for d in days], np.float32).reshape(W, 7).mean(1)
        x[:, :, 1] = after[None, :]
        mid = [days[7 * k + 3] for k in range(W)]
        ang = 2 * np.pi * (np.asarray([d.timetuple().tm_yday for d in mid]) - 1) / 365.25
        x[:, :, 2] = np.sin(ang)[None, :]
        x[:, :, 3] = np.cos(ang)[None, :]
        x[:, :, 4] = (rain - normal).reshape(rows, W, 7).sum(2)
        x[:, :, 5] = (rain >= WET_MM).reshape(rows, W, 7).mean(2)
        return x

    def _scale(self, x):
        sc, ch = self.meta["channel_scaling"], self.meta["channels"]
        r, a = ch.index("rain_mm"), ch.index("rain_anom_mm")
        out = x.copy()
        out[..., r] = (np.log1p(np.maximum(x[..., r], 0)) - sc["rain_mm"]["mean"]) / sc["rain_mm"]["std"]
        v = x[..., a]
        out[..., a] = (np.sign(v) * np.log1p(np.abs(v)) - sc["rain_anom_mm"]["mean"]) / sc["rain_anom_mm"]["std"]
        return out

    def predict_steps(self, wells, steps, member=0, edits=()):
        """Predicted change (len(steps), len(wells)) at each regular reading."""
        wells = np.asarray(wells)
        seq, num, cat = [], [], []
        for y, s in steps:
            rain, normal, days = self._window(wells, y, s, member, edits)
            seq.append(self._scale(self._channels(rain, normal, days, s)))
            num.append(self.num[wells, s])
            cat.append(self.cat[wells, s])
        seq, num, cat = np.concatenate(seq), np.concatenate(num), np.concatenate(cat)
        out = np.empty(len(seq), np.float32)
        for i in range(0, len(seq), 4096):
            k = slice(i, i + 4096)
            out[k] = self.session.run([self.output], {"seq": seq[k], "num": num[k], "cat": cat[k]})[0]
        return out.reshape(len(steps), len(wells))

    def context(self, wells, start: dt.date, end: dt.date):
        """Normal rain (and recorded rain, for past days) over the edited days, mean over wells."""
        days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        doy = np.asarray([min(d.timetuple().tm_yday, 365) - 1 - (1 if (d.month > 2 and d.year % 4 == 0) else 0)
                          for d in days]).clip(0, 364)
        out = {"normal_mm": float(self.normal_doy[doy][:, wells].sum(0).mean())}
        if end.year <= LAST_YEAR:
            src = np.asarray([self._day(d) for d in days])
            out["recorded_mm"] = float(self.rain[src][:, wells].sum(0).mean())
        return out

    def effect(self, wells, edits, members=(0,)):
        """Effect of the edits on each reading they reach: (members, steps, wells), steps."""
        first = min(e.start for e in edits)
        last = max(e.end for e in edits)
        steps = steps_between(first - dt.timedelta(days=1), last + dt.timedelta(days=W * 7))
        steps = [st for st in steps if st[0] >= FIRST_YEAR]   # earlier windows start before the record
        if not steps:
            return np.zeros((len(members), 0, len(wells)), np.float32), []
        past = all(y <= LAST_YEAR for y, _ in steps)
        members = (0,) if past else members
        out = np.stack([self.predict_steps(wells, steps, m, edits) - self.predict_steps(wells, steps, m, ())
                        for m in members])
        return out, steps
