"""
Builds the BiLSTM copy of the training data: the same rows as data/training/,
with a multi-channel weekly sequence in place of the single rain channel.

    python bilstm-data/build_bilstm_data.py

Writes to bilstm-data/out/
    tabular.parquet        identical rows and columns to data/training/tabular.parquet
    feature_spec.csv       identical to data/training/feature_spec.csv
    rain_seq.npz           the transformer's single channel, kept for the 1-channel ablation
    seq_channels.npz       x: (rows, 104, 6) float16, oldest week first; channels: names
    build_report.txt

Why a separate copy rather than editing the main build: the transformer's
results were produced on data/training/ and must stay reproducible. Every row,
target, split, fold and tabular feature here comes from the same
models/build_training_data.build(), so the two architectures are compared on the
same rows; only the sequence input is richer. See bilstm-data/README.md.

Channels, per week (7 days), 104 weeks ending on the reading date:
    0 rain_mm          weekly rain total, same as the transformer's input
    1 in_interval      share of the week's days that fall after the previous
                       reading, i.e. inside the interval delta_h_m is measured over
    2 doy_sin          calendar position of the week's middle day
    3 doy_cos
    4 rain_anom_mm     weekly rain minus that well's normal for those calendar days,
                       normal from TRAINING YEARS ONLY (2000-2014), 31-day smoothed
    5 wet_frac         share of the week's days with at least 2.5 mm (IMD rain day)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "models"))
import build_training_data as btd  # noqa: E402

OUT_DIR = HERE / "out"
CHANNELS = ["rain_mm", "in_interval", "doy_sin", "doy_cos", "rain_anom_mm", "wet_frac"]
WET_MM = 2.5             # IMD's definition of a rain day
CLIM_SMOOTH_DAYS = 31    # 15 training years per calendar day is noisy; smooth across days


def daily_normals(dates, rain):
    """Per-well mean rain for each calendar day, from training years only.

    Returns (n_days, n_wells): the normal for every day of the record, so the
    anomaly of any day is rain minus this. Feb 29 is folded onto Feb 28.
    """
    doy = np.minimum(dates.dayofyear.to_numpy(), 365) - 1
    leap_shift = dates.is_leap_year & (dates.month > 2)
    doy = np.where(leap_shift, doy - 1, doy)
    doy = np.clip(doy, 0, 364)

    tr = np.asarray(dates.year.isin(list(btd.TRAIN_YEARS)))
    clim = np.zeros((365, rain.shape[1]), "float64")
    cnt = np.bincount(doy[tr], minlength=365).astype("float64")
    np.add.at(clim, doy[tr], rain[tr])
    clim /= np.maximum(cnt, 1)[:, None]

    # circular moving average over calendar days
    k = CLIM_SMOOTH_DAYS
    pad = np.concatenate([clim[-(k // 2):], clim, clim[:k // 2]])
    c = np.vstack([np.zeros((1, clim.shape[1])), np.cumsum(pad, axis=0)])
    clim = (c[k:] - c[:-k]) / k
    return clim[doy].astype("float32")


def weekly(cum, start, wcol, n_weeks):
    """Weekly sums of a daily series from its (n_days+1, n_wells) cumsum.

    start: first day index of each row's window. Returns (rows, n_weeks).
    """
    edges = start[:, None] + 7 * np.arange(n_weeks + 1)[None, :]
    v = cum[edges, wcol[:, None]]
    return np.diff(v, axis=1)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # same rows, same features, same target as the main build, written here
    btd.OUT_DIR = OUT_DIR
    long, seq, spec = btd.build()
    log = [(OUT_DIR / "build_report.txt").read_text(), "", "---- bilstm channels"]

    wells_xy = (long[["well_uid", "lat", "lon"]].drop_duplicates("well_uid")
                .reset_index(drop=True))
    widx = pd.Series(wells_xy.index.values, index=wells_xy.well_uid)
    dates, rain = btd.well_daily_rain(wells_xy)
    rain = np.nan_to_num(rain, nan=0.0)

    anom = rain - daily_normals(dates, rain)

    def cumsum(a):
        return np.vstack([np.zeros((1, a.shape[1]), "float64"),
                          np.cumsum(a, axis=0, dtype="float64")])

    cum_anom = cumsum(anom)
    del anom
    cum_wet = cumsum((rain >= WET_MM).astype("float32"))

    n, W = len(long), btd.SEQ_WEEKS
    wcol = widx.loc[long.well_uid].to_numpy()
    endi = dates.searchsorted(long.campaign_date.values.astype("datetime64[ns]"), side="right")
    lo = dates.searchsorted(long.prev_date.values.astype("datetime64[ns]"), side="right")
    start = endi - W * 7
    ok = start >= 0

    x = np.full((n, W, len(CHANNELS)), np.nan, dtype="float32")
    x[:, :, 0] = seq

    rows = np.flatnonzero(ok)
    CH = 20000
    doy_all = dates.dayofyear.to_numpy()
    for a in range(0, len(rows), CH):
        r = rows[a:a + CH]
        s, w = start[r], wcol[r]
        wk0 = s[:, None] + 7 * np.arange(W)[None, :]               # first day of each week
        x[r, :, 1] = np.clip(wk0 + 7 - lo[r][:, None], 0, 7) / 7.0
        ang = 2 * np.pi * (doy_all[wk0 + 3] - 1) / 365.25
        x[r, :, 2] = np.sin(ang)
        x[r, :, 3] = np.cos(ang)
        x[r, :, 4] = weekly(cum_anom, s, w, W)
        x[r, :, 5] = weekly(cum_wet, s, w, W) / 7.0

    # checks that the channels mean what they claim
    covered = np.isfinite(x).all(axis=(1, 2))
    days_in = x[covered, :, 1].sum(axis=1) * 7
    want = (endi - lo)[covered]
    assert np.abs(days_in - want).max() < 1e-3, "in_interval disagrees with days_since_prev"
    tr = (long.split == "train").to_numpy() & covered
    log.append(f"rows fully covered: {int(covered.sum()):,} of {n:,}")
    log.append(f"interval weeks per row: median {np.median((x[covered, :, 1] > 0).sum(1)):.0f}, "
               f"max {(x[covered, :, 1] > 0).sum(1).max()}")
    log.append(f"train-row mean weekly anomaly: {np.nanmean(x[tr, :, 4]):.3f} mm "
               "(near zero by construction)")
    for i, c in enumerate(CHANNELS):
        v = x[covered, :, i]
        log.append(f"  {c:<14} mean {v.mean():9.3f}  sd {v.std():9.3f}  "
                   f"min {v.min():9.3f}  max {v.max():9.3f}")

    np.savez_compressed(OUT_DIR / "seq_channels.npz", x=x.astype("float16"),
                        channels=np.array(CHANNELS))
    txt = "\n".join(log)
    (OUT_DIR / "build_report.txt").write_text(txt)
    print("\n".join(log[2:]))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    build()
