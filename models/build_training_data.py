"""
Builds the training table and the weekly rain sequences from raw CGWB + IMD.

Self-contained: needs only
    data/cgwb/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv
    data/imd_rainfall/imd_rf25_YYYY.nc          (1998..2022)

Writes to data/training/
    tabular.parquet        one row per (well, campaign) with a previous reading
    rain_seq.npz           weekly rain per row, oldest week first
    feature_spec.csv       column -> role
    build_report.txt

Contract matches the data_cleaning branch so the same model code runs on
either table:
    target      delta_h_m   = depth now - depth at previous reading
                              positive means the water level FELL
    splits      train 2000-2014, valid 2015-2017, test 2018-2022
    folds       5 district folds for grouped CV
    forbidden   never a feature: depth_start_m, depth_mbgl, normal_ref_m,
                season_year, year, lat, lon, district, state, well_uid
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# DATA_DIR=data/sample runs everything on the small sample instead of the full data
DATA = Path(os.environ.get("DATA_DIR", REPO / "data")).resolve()
CGWB_CSV = DATA / "cgwb" / "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv"
IMD_DIR = DATA / "imd_rainfall"
RAIN_CUBE = DATA / "derived" / "rain_cube.npz"
OUT_DIR = DATA / "training"

RAIN_YEARS = range(1998, 2023)
TRAIN_YEARS = range(2000, 2015)
VALID_YEARS = range(2015, 2018)
TEST_YEARS = range(2018, 2023)
N_DISTRICT_FOLDS = 5
MIN_READINGS = 4          # a well needs this many campaigns to be usable
SEQ_WEEKS = 104           # 2 years of weekly rain behind each reading
SEED = 42

TARGET = "delta_h_m"
FORBIDDEN = {
    "depth_start_m", "depth_mbgl", "normal_ref_m", "season_year", "year",
    "lat", "lon", "district", "state", "well_uid", "campaign_date",
    "prev_date", "split", "district_fold", TARGET,
}
CATEGORICAL = ["season", "aquifer", "well_type"]

# CGWB measures in these four months; the file gives no day, so readings are
# placed mid-month. Interval lengths are therefore exact to within a few days.
MONTHS = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
READING_DAY = 15


# --------------------------------------------------------------------------
# 1. groundwater: wide -> long
# --------------------------------------------------------------------------
def load_wells():
    raw = pd.read_csv(CGWB_CSV, low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]

    camp_cols = [c for c in raw.columns if c[:3] in MONTHS and c[3] == "-"]
    meta = raw[["Station Name", "State", "District", "Latitude", "Longitude",
                "Type of Well", "Aquifer Type", "Well Depth", "Reference_Sy"]].copy()
    meta.columns = ["station", "state", "district", "lat", "lon",
                    "well_type", "aquifer", "well_depth_m", "sy"]

    # identity from coordinates: 'Station Code' is not unique in this file
    meta["well_uid"] = (meta.lat.round(4).astype(str) + "_" +
                        meta.lon.round(4).astype(str))
    dup = meta.well_uid.duplicated(keep=False)
    meta.loc[dup, "well_uid"] += "_" + meta.loc[dup].groupby("well_uid").cumcount().astype(str)

    long = raw[camp_cols].copy()
    long["well_uid"] = meta.well_uid.values
    long = long.melt(id_vars="well_uid", var_name="campaign", value_name="depth_mbgl")
    long = long.dropna(subset=["depth_mbgl"])

    mon = long.campaign.str[:3].map(MONTHS)
    yy = long.campaign.str[4:].astype(int)
    year = np.where(yy < 50, 2000 + yy, 1900 + yy)
    long["campaign_date"] = pd.to_datetime(
        dict(year=year, month=mon, day=READING_DAY))
    long["season"] = long.campaign.str[:3].str.upper()
    long["season_year"] = year
    long = long.drop(columns="campaign")

    long = long.merge(meta.drop(columns="station"), on="well_uid", how="left")
    long = long[(long.depth_mbgl > -5) & (long.depth_mbgl < 200)]

    # aquifer / well type tidy-up
    for c in ("aquifer", "well_type"):
        long[c] = (long[c].fillna("unknown").astype(str).str.strip()
                   .str.lower().replace({"-": "unknown", "": "unknown"}))

    keep = long.groupby("well_uid").depth_mbgl.transform("size") >= MIN_READINGS
    long = long[keep]

    long = long.sort_values(["well_uid", "campaign_date"]).reset_index(drop=True)
    return long, meta


# --------------------------------------------------------------------------
# 2. rainfall: daily grid -> daily series at each well by masked bilinear
# --------------------------------------------------------------------------
def well_daily_rain(wells_xy):
    """wells_xy: DataFrame with well_uid, lat, lon (one row per well).

    Returns (dates, matrix) with matrix shape (n_days, n_wells) in mm.

    Uses data/derived/rain_cube.npz when it is present, which is the whole
    IMD record at 22 MB, and falls back to the 607 MB of NetCDF files when it
    is not. Both paths compute the same thing.
    """
    if RAIN_CUBE.exists():
        return _rain_from_cube(wells_xy)
    return _rain_from_netcdf(wells_xy)


def _bilinear_weights(wells_xy, lat, lon):
    """Four surrounding grid nodes per well, with their bilinear weights.

    Returns (idx, wts) each of shape (n_wells, 4): idx holds flat grid indices
    into a lat x lon raster, wts the products of the axis fractions.
    """
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    wlat = wells_xy.lat.to_numpy("float64")
    wlon = wells_xy.lon.to_numpy("float64")

    i0 = np.clip(np.floor((wlat - lat[0]) / dlat).astype(int), 0, lat.size - 2)
    j0 = np.clip(np.floor((wlon - lon[0]) / dlon).astype(int), 0, lon.size - 2)
    fy = np.clip((wlat - lat[i0]) / dlat, 0.0, 1.0)
    fx = np.clip((wlon - lon[j0]) / dlon, 0.0, 1.0)

    idx = np.stack([
        i0 * lon.size + j0,
        i0 * lon.size + (j0 + 1),
        (i0 + 1) * lon.size + j0,
        (i0 + 1) * lon.size + (j0 + 1),
    ], axis=1)
    wts = np.stack([
        (1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx,
    ], axis=1)
    return idx, wts


def _rain_from_cube(wells_xy):
    """Masked bilinear straight off the packed cube.

    A corner that is sea, or missing on that day, drops out and the remaining
    weights are renormalised. Without that, every coastal well would come back
    NaN, and a well beside a suspect-zero cell would be diluted by it.
    """
    z = np.load(RAIN_CUBE)
    lat, lon = z["lat"], z["lon"]
    col_of = np.full(lat.size * lon.size, -1, "int32")
    col_of[z["lat_idx"].astype("int64") * lon.size + z["lon_idx"].astype("int64")] = \
        np.arange(z["lat_idx"].size)

    idx, wts = _bilinear_weights(wells_xy, lat, lon)
    cols = col_of[idx]                       # (n_wells, 4), -1 where sea
    on_land = cols >= 0
    cols = np.where(on_land, cols, 0)

    rain16 = z["rain"]                       # (n_days, n_land) int16, tenths
    n_days, n_wells = rain16.shape[0], len(wells_xy)
    out = np.zeros((n_days, n_wells), "float32")

    CH = 512
    for a in range(0, n_wells, CH):
        s = slice(a, a + CH)
        v = rain16[:, cols[s]].astype("float32")          # (days, k, 4)
        ok = on_land[s][None, :, :] & (v != -9999.0)
        w = np.where(ok, wts[s][None, :, :], 0.0)
        den = w.sum(axis=2)
        num = (np.where(ok, v, 0.0) * w).sum(axis=2)
        out[:, s] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan) / 10.0

    days = pd.to_datetime(z["days"].astype("datetime64[D]"))
    return pd.DatetimeIndex(days), out


def _rain_from_netcdf(wells_xy):
    import xarray as xr   # only needed without the rain cube
    wlat = xr.DataArray(wells_xy.lat.values, dims="w")
    wlon = xr.DataArray(wells_xy.lon.values, dims="w")

    chunks, dates = [], []
    for y in RAIN_YEARS:
        f = IMD_DIR / f"imd_rf25_{y}.nc"
        ds = xr.open_dataset(f)
        r = ds["RAINFALL"]
        valid = r.notnull()
        num = r.fillna(0.0).interp(LATITUDE=wlat, LONGITUDE=wlon, method="linear")
        den = valid.astype("float32").interp(LATITUDE=wlat, LONGITUDE=wlon, method="linear")
        out = (num / den.where(den > 0.05)).values.astype("float32")
        chunks.append(out)
        dates.append(pd.to_datetime(ds["TIME"].values))
        ds.close()

    return pd.DatetimeIndex(np.concatenate(dates)), np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------
# 3. features
# --------------------------------------------------------------------------
def window_sums(cum, hi, wcol, days):
    """Sum of rain over the `days` ending just before index hi, per row.

    cum is the (n_days+1, n_wells) cumulative sum along time; hi and wcol are
    the per-row time and well indices. NaN where the window runs off the start
    of the rainfall record. Indexed row-wise on purpose: taking whole columns
    first would materialise rows x wells.
    """
    lo = hi - days
    bad = lo < 0
    out = cum[hi, wcol] - cum[np.clip(lo, 0, None), wcol]
    out[bad] = np.nan
    return out


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    long, meta = load_wells()
    log.append(f"readings after QC: {len(long):,} over {long.well_uid.nunique():,} wells")

    wells_xy = (long[["well_uid", "lat", "lon"]].drop_duplicates("well_uid")
                .reset_index(drop=True))
    widx = pd.Series(wells_xy.index.values, index=wells_xy.well_uid)

    dates, rain = well_daily_rain(wells_xy)
    log.append(f"rain grid sampled at {rain.shape[1]:,} wells over {rain.shape[0]:,} days")
    log.append(f"wells with no land neighbour: {int(np.isnan(rain).all(axis=0).sum())}")
    rain = np.nan_to_num(rain, nan=0.0)

    cum = np.vstack([np.zeros((1, rain.shape[1]), "float64"),
                     np.cumsum(rain.astype("float64"), axis=0)])

    # ---- target: change since the previous reading at the same well
    g = long.groupby("well_uid", sort=False)
    long["depth_start_m"] = g.depth_mbgl.shift(1)
    long["prev_date"] = g.campaign_date.shift(1)
    long["prev_season"] = g.season.shift(1)
    long = long.dropna(subset=["depth_start_m"]).reset_index(drop=True)
    long["delta_h_m"] = long.depth_mbgl - long.depth_start_m
    long["days_since_prev"] = (long.campaign_date - long.prev_date).dt.days
    long = long[(long.days_since_prev >= 45) & (long.days_since_prev <= 400)]
    # a 30 m jump between two campaigns is an instrument change, not hydrology
    long = long[long.delta_h_m.abs() <= 30].reset_index(drop=True)
    log.append(f"rows with a usable previous reading: {len(long):,}")

    wcol = widx.loc[long.well_uid].values
    end = long.campaign_date.values.astype("datetime64[ns]")
    rows = np.arange(len(long))
    hi = dates.searchsorted(end, side="right")
    lo = dates.searchsorted(long.prev_date.values.astype("datetime64[ns]"), side="right")

    def win(days):
        return window_sums(cum, hi, wcol, days)

    # ---- rain over the interval itself, the thing that should explain delta
    long["rain_interval_mm"] = (cum[hi, wcol] - cum[lo, wcol])
    long["rain_interval_per_day"] = long.rain_interval_mm / long.days_since_prev

    for d in (30, 60, 90, 180, 365, 730):
        long[f"rain_{d}d_mm"] = win(d)

    # ---- monsoon rain of the June-September window most recently completed
    m_year = np.where(long.campaign_date.dt.month >= 10,
                      long.season_year, long.season_year - 1)
    m_end = pd.to_datetime(dict(year=m_year, month=9, day=30)).values.astype("datetime64[ns]")
    mh = dates.searchsorted(m_end, side="right")
    ml = dates.searchsorted(
        pd.to_datetime(dict(year=m_year, month=6, day=1)).values.astype("datetime64[ns]"),
        side="left")
    long["rain_monsoon_mm"] = cum[mh, wcol] - cum[ml, wcol]

    # ---- rain intensity inside the interval
    seg_len = (hi - lo)
    idx = np.repeat(lo, seg_len) + (np.arange(seg_len.sum()) -
                                    np.repeat(np.cumsum(seg_len) - seg_len, seg_len))
    rowrep = np.repeat(rows, seg_len)
    vals = rain[idx, wcol[rowrep]]
    wet = pd.Series(vals >= 2.5).groupby(rowrep).sum()
    heavy = pd.Series(vals >= 35.0).groupby(rowrep).sum()
    mx = pd.Series(vals).groupby(rowrep).max()
    long["rain_days_interval"] = wet.reindex(rows).values
    long["heavy_days_interval"] = heavy.reindex(rows).values
    long["rain_max_day_mm"] = mx.reindex(rows).values
    long["rain_per_wet_day"] = long.rain_interval_mm / long.rain_days_interval.clip(lower=1)

    # ---- lags and history, in campaigns not days
    gl = long.groupby("well_uid", sort=False)
    for k in (1, 2, 3):
        long[f"rain_interval_lag{k}"] = gl.rain_interval_mm.shift(k)
    for k in (4, 8, 12):
        long[f"rain_hist{k}_mm"] = (gl.rain_interval_mm
                                    .transform(lambda s: s.shift(1).rolling(k, min_periods=k).sum()))

    # ---- splits, assigned before any normal is computed
    long["split"] = np.select(
        [long.season_year.isin(TRAIN_YEARS), long.season_year.isin(VALID_YEARS)],
        ["train", "valid"], default="test")
    long = long[long.season_year.isin(list(TRAIN_YEARS) + list(VALID_YEARS) + list(TEST_YEARS))]
    long = long.reset_index(drop=True)

    # ---- normals from TRAINING YEARS ONLY, then anomalies
    tr = long[long.split == "train"]
    ann = tr.groupby("well_uid").rain_365d_mm.mean().rename("rain_normal_annual_mm")
    seas = (tr.groupby(["well_uid", "season"]).rain_interval_mm.mean()
            .rename("rain_normal_season_mm"))
    long = long.merge(ann, on="well_uid", how="left")
    long = long.merge(seas, on=["well_uid", "season"], how="left")
    long["rain_anom_annual_mm"] = long.rain_365d_mm - long.rain_normal_annual_mm
    long["rain_anom_season_mm"] = long.rain_interval_mm - long.rain_normal_season_mm
    long["rain_anom_season_ratio"] = (long.rain_interval_mm /
                                      long.rain_normal_season_mm.replace(0, np.nan))
    mn = tr.groupby("well_uid").rain_monsoon_mm.mean().rename("rain_normal_monsoon_mm")
    long = long.merge(mn, on="well_uid", how="left")
    long["rain_anom_monsoon_mm"] = long.rain_monsoon_mm - long.rain_normal_monsoon_mm

    # ---- static
    long["sy"] = pd.to_numeric(long.sy, errors="coerce")
    long["well_depth_m"] = pd.to_numeric(long.well_depth_m, errors="coerce")
    long["transition"] = long.prev_season + "_" + long.season
    long = long.drop(columns=["prev_season"])

    # ---- grouped folds by district
    rng = np.random.default_rng(SEED)
    d = np.sort(long.district.fillna("unknown").unique())
    fold = pd.Series(rng.permutation(len(d)) % N_DISTRICT_FOLDS, index=d)
    long["district_fold"] = long.district.fillna("unknown").map(fold).astype("int8")

    # ---- weekly rain sequence, exactly SEQ_WEEKS whole weeks before the reading
    n = len(long)
    seq = np.full((n, SEQ_WEEKS), np.nan, dtype="float32")
    endi = dates.searchsorted(long.campaign_date.values.astype("datetime64[ns]"), side="right")
    wcol = widx.loc[long.well_uid].values        # long was refiltered above
    span = SEQ_WEEKS * 7
    start = endi - span
    ok = np.flatnonzero(start >= 0)
    offs = np.arange(span)
    CH = 20000
    for a in range(0, len(ok), CH):
        sel = ok[a:a + CH]
        ti = start[sel][:, None] + offs[None, :]
        block = rain[ti, wcol[sel][:, None]]
        seq[sel] = block.reshape(len(sel), SEQ_WEEKS, 7).sum(axis=2)
    log.append(f"sequence rows fully covered: {int(np.isfinite(seq).all(axis=1).sum()):,} of {n:,}")

    # ---- feature spec
    feature_cols = [c for c in long.columns if c not in FORBIDDEN]
    spec = pd.DataFrame({"column": long.columns})
    spec["role"] = np.where(spec.column == TARGET, "target",
                   np.where(spec.column.isin(FORBIDDEN), "meta", "feature"))
    spec["kind"] = np.where(spec.column.isin(CATEGORICAL), "categorical",
                   np.where(spec.column.isin(["transition"]), "categorical", "numeric"))

    bad = [c for c in feature_cols if c in FORBIDDEN]
    assert not bad, bad

    long.to_parquet(OUT_DIR / "tabular.parquet", index=False)
    np.savez_compressed(OUT_DIR / "rain_seq.npz",
                        weeks=seq.astype("float16"),
                        row_index=np.arange(n))
    spec.to_csv(OUT_DIR / "feature_spec.csv", index=False)

    log.append(f"features: {len(feature_cols)}")
    log.append("split sizes:\n" + long.split.value_counts().to_string())
    log.append("target by split:\n" +
               long.groupby("split")[TARGET].agg(["mean", "std", "count"]).to_string())
    txt = "\n".join(str(x) for x in log)
    (OUT_DIR / "build_report.txt").write_text(txt)
    print(txt)
    return long, seq, spec


if __name__ == "__main__":
    build()
