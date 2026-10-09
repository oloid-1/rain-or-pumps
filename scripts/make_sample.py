"""
Cuts a small sample out of the full data: one state's wells and the rain grid
around them, in the same layout as data/, so every step runs on it unchanged.

    python scripts/make_sample.py                  Karnataka (217 wells)
    python scripts/make_sample.py --state Kerala

Writes data/sample/cgwb/ and data/sample/derived/rain_cube.npz. Then
    DATA_DIR=data/sample make sequences ui-data forecast api
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FULL = REPO / "data"
CSV = "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv"
MARGIN = 0.5        # degrees of rain grid kept around the wells, so edge wells keep all four neighbours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="Karnataka")
    ap.add_argument("--out", default=str(FULL / "sample"))
    a = ap.parse_args()
    out = Path(a.out)

    wells = pd.read_csv(FULL / "cgwb" / CSV, low_memory=False)
    wells = wells[wells.State.str.lower() == a.state.lower()]
    assert len(wells), f"no wells in {a.state}"
    (out / "cgwb").mkdir(parents=True, exist_ok=True)
    wells.to_csv(out / "cgwb" / CSV, index=False)

    # keep every land cell inside the wells' bounding box, on the full grid's axes
    z = np.load(FULL / "derived" / "rain_cube.npz")
    lat, lon = z["lat"][z["lat_idx"]], z["lon"][z["lon_idx"]]
    keep = ((lat >= wells.Latitude.min() - MARGIN) & (lat <= wells.Latitude.max() + MARGIN)
            & (lon >= wells.Longitude.min() - MARGIN) & (lon <= wells.Longitude.max() + MARGIN))
    (out / "derived").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "derived" / "rain_cube.npz", rain=z["rain"][:, keep], days=z["days"],
                        lat=z["lat"], lon=z["lon"], lat_idx=z["lat_idx"][keep], lon_idx=z["lon_idx"][keep])

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "training" not in p.parts)
    print(f"{a.state}: {len(wells)} wells, {keep.sum()} rain cells -> {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
