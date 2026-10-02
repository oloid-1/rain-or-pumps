"""
Exports everything the simulator UI reads, from the built data and a trained
simulator run.

    python simulator/ui/build_ui_data.py --run simulator/artifacts

Reads
    data/derived/rain_cube.npz        daily IMD rain, every land cell, 1998-2022
    bilstm-data/out/                  tabular rows + 6-channel sequences
    <run>/sim.pt, <run>/sim.onnx, <run>/sim_meta.json

Writes simulator/ui/data/
    rain/grid.json       grid geometry: lat/lon axes, land-cell positions, quantisation
    rain/<year>.bin      uint8, days x land cells; mm = expm1(q / RAIN_Q)
    rain/national.json   mean rain over all land cells, every day, for the timeline
    wells.json           one entry per well: position, district, fixed properties,
                         and its level history (campaign index -> depth, change)
    campaigns.json       the campaign calendar the histories index into
    scenario/<year>.bin  November rows of that year, model-ready: scaled sequence
                         (float16), static numerics (float32), category codes (uint8)
    scenario/index.json  which well each scenario row belongs to, and the byte layout
    pressure.json        per district and year: observed minus rain-expected change
    sim.onnx, sim_meta.json   copied from the run, for onnxruntime-web

Scenario years are 2015-2022 only, the validation and test years the model never
trained on, so every scenario starts from an out-of-sample reading.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "simulator"), str(REPO / "models")]
from bilstm import RainBiLSTM  # noqa: E402
from train_bilstm import load, channel_stats, scale  # noqa: E402
from train_sim import encode_static, predict, STATIC_NUMERIC  # noqa: E402

OUT = HERE / "data"
RAIN_Q = 33.0               # log1p(2,000 mm) * 33 = 251, so a day up to ~2 m of rain fits a byte
SCENARIO_YEARS = range(2015, 2023)
SCENARIO_SEASON = "NOV"     # after the monsoon, where recharge shows


def export_rain():
    z = np.load(REPO / "data" / "derived" / "rain_cube.npz")
    rain, days = z["rain"], z["days"].astype("datetime64[D]")
    d = OUT / "rain"
    d.mkdir(parents=True, exist_ok=True)
    years = pd.DatetimeIndex(days).year
    for y in np.unique(years):
        r = rain[years == y].astype("float32") / 10.0
        r = np.where(r < 0, 0, r)                       # -9999 no-data drawn as dry
        q = np.clip(np.round(np.log1p(r) * RAIN_Q), 0, 255).astype("uint8")
        (d / f"{y}.bin").write_bytes(q.tobytes())
    # India-wide daily mean over land cells with data, for the timeline chart
    r = rain.astype("float32")
    ok = r >= 0
    mean = np.where(ok, r, 0).sum(1) / np.maximum(ok.sum(1), 1) / 10.0
    (d / "national.json").write_text(json.dumps([round(float(v), 2) for v in mean]))
    (d / "grid.json").write_text(json.dumps(dict(
        lat=z["lat"].round(3).tolist(), lon=z["lon"].round(3).tolist(),
        lat_idx=z["lat_idx"].tolist(), lon_idx=z["lon_idx"].tolist(),
        n_cells=int(rain.shape[1]), first_day=str(days[0]), quant=RAIN_Q,
        decode="mm = exp(q / quant) - 1",
        source="IMD 0.25 degree gridded daily rainfall (Pai et al. 2014)")))
    print(f"  rain: {len(np.unique(years))} years x {rain.shape[1]:,} cells")


def place_wells(tab):
    """District and state from the post-2020 outlines by coordinates, the same
    rule the pipeline uses, so wells join the map's polygons by name|state.
    The CGWB file's own district labels mix vintages and spellings."""
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    from shapely.validation import make_valid
    feats = json.loads((REPO / "data_cleaning" / "reference" / "districts.geojson").read_text())["features"]
    polys = [make_valid(shape(f["geometry"])) for f in feats]
    tree = STRtree(polys)
    xy = tab[["well_uid", "lat", "lon"]].drop_duplicates("well_uid")
    place = {}
    for r in xy.itertuples():
        pt = Point(r.lon, r.lat)
        hit = [i for i in tree.query(pt) if polys[i].covers(pt)]
        if not hit:                                    # coastal wells just off the outline
            hit = [int(tree.nearest(pt))]
        p = feats[hit[0]]["properties"]
        place[r.well_uid] = (p["n"], p["s"])
    tab = tab.copy()
    tab["district"] = tab.well_uid.map(lambda u: place[u][0])
    tab["state"] = tab.well_uid.map(lambda u: place[u][1])
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REPO / "simulator" / "artifacts"))
    ap.add_argument("--data", default=str(REPO / "bilstm-data" / "out"))
    ap.add_argument("--skip-rain", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    OUT.mkdir(parents=True, exist_ok=True)

    if not a.skip_rain:
        export_rain()

    tab, x, channels, _, _ = load(a.data)
    meta = json.loads((run / "sim_meta.json").read_text())
    assert meta["channels"] == channels, (meta["channels"], channels)
    mk = (tab.split == "train").to_numpy()
    S = scale(x, channels, channel_stats(x, channels, mk))
    Xn, C, sizes, _ = encode_static(tab, mk)

    m = RainBiLSTM(len(STATIC_NUMERIC), sizes, len(channels))
    m.load_state_dict(torch.load(run / "sim.pt", map_location="cpu"))
    pred = predict(m, S, Xn, C, np.arange(len(tab)), "cpu")
    tab = place_wells(tab.assign(pred=pred, resid=tab.delta_h_m - pred))
    print(f"  predictions on {len(tab):,} rows")

    # ---- campaigns and wells
    camps = (tab[["season_year", "season", "campaign_date"]].drop_duplicates()
             .sort_values("campaign_date").reset_index(drop=True))
    cidx = {(r.season_year, r.season): i for i, r in camps.iterrows()}
    (OUT / "campaigns.json").write_text(json.dumps([
        dict(year=int(r.season_year), season=r.season, date=str(r.campaign_date.date()))
        for r in camps.itertuples()]))

    wells = []
    for uid, g in tab.groupby("well_uid", sort=True):
        f = g.iloc[0]
        wells.append(dict(
            id=uid, lat=round(float(f.lat), 4), lon=round(float(f.lon), 4),
            district=f.district, state=f.state, aquifer=f.aquifer, type=f.well_type,
            well_depth_m=None if pd.isna(f.well_depth_m) else round(float(f.well_depth_m), 1),
            sy=None if pd.isna(f.sy) else round(float(f.sy), 3),
            rain_normal_annual_mm=None if pd.isna(f.rain_normal_annual_mm) else round(float(f.rain_normal_annual_mm)),
            # campaign index, depth below ground (m), observed change, rain-expected change
            h=[[cidx[(r.season_year, r.season)], round(float(r.depth_mbgl), 2),
                round(float(r.delta_h_m), 2), round(float(r.pred), 2)]
               for r in g.sort_values("campaign_date").itertuples()]))
    wid = {w["id"]: i for i, w in enumerate(wells)}
    (OUT / "wells.json").write_text(json.dumps(wells, separators=(",", ":")))
    print(f"  wells: {len(wells):,}, campaigns: {len(camps)}")

    # ---- scenario inputs: November rows of the out-of-sample years
    sd = OUT / "scenario"
    sd.mkdir(exist_ok=True)
    index = {}
    for y in SCENARIO_YEARS:
        k = np.flatnonzero(((tab.season_year == y) & (tab.season == SCENARIO_SEASON)).to_numpy())
        blob = (S[k].astype("float16").tobytes() + Xn[k].astype("float32").tobytes()
                + C[k].astype("uint8").tobytes())
        (sd / f"{y}.bin").write_bytes(blob)
        index[y] = dict(rows=int(len(k)), well=[wid[u] for u in tab.well_uid.iloc[k]],
                        date=str(tab.campaign_date.iloc[k[0]].date()),
                        observed=tab.delta_h_m.iloc[k].round(2).tolist(),
                        depth_before=tab.depth_start_m.iloc[k].round(2).tolist(),
                        # the model at recorded rain, so the page opens without running it
                        base=tab.pred.iloc[k].round(4).tolist())
    (sd / "index.json").write_text(json.dumps(dict(
        layout=dict(seq=["rows", int(S.shape[1]), int(S.shape[2]), "float16"],
                    num=["rows", int(Xn.shape[1]), "float32"],
                    cat=["rows", int(C.shape[1]), "uint8"]),
        years=index), separators=(",", ":")))
    print(f"  scenario years {list(SCENARIO_YEARS)}")

    # ---- pressure: observed minus rain-expected, per district and year
    pr = (tab.groupby(["district", "state", "season_year"])
          .agg(resid=("resid", "mean"), obs=("delta_h_m", "mean"), exp=("pred", "mean"),
               n=("resid", "size"), wells=("well_uid", "nunique"))
          .reset_index())
    pr = pr[pr.wells >= 2]
    out = {}
    for (dname, st), g in pr.groupby(["district", "state"]):
        out[f"{dname}|{st}"] = {int(r.season_year): [round(r.resid, 3), round(r.obs, 3), round(r.exp, 3), int(r.wells)]
                                for r in g.itertuples()}
    (OUT / "pressure.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  pressure: {len(out)} districts")

    for f in ("sim.onnx", "sim_meta.json"):
        shutil.copy2(run / f, OUT / f)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
