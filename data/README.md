# data/

Everything the code reads lives here. Set `DATA_DIR` to point the builds at another
copy with the same layout.

## What is in the repository

| Path | Size | What it is |
|---|---|---|
| `cgwb/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv` | 2.8 MB | CGWB quality-controlled levels, 2,759 wells, 2000-2022 |
| `derived/rain_cube.npz` | 22 MB | the whole IMD daily rain record, 1998-2022, packed (below) |
| `reference/districts.geojson` | 0.4 MB | 724 post-2020 district outlines |

The app's built data (wells, rain files, forecast table, model files) is committed
in `simulator/ui/data/`.

## Not in the repository

| Path | Size | How to get it |
|---|---|---|
| `imd_rainfall/imd_rf25_*.nc` | 607 MB | `python scripts/fetch_data.py --imd`; only to rebuild the cube (`make cube`) |
| `geo/raw/` | | river, dam and town downloads for the map layers, see `simulator/README.md` |
| `training/` | 130 MB | `make sequences` |

## Why the rainfall is packed into a cube

The IMD files store a value for every point of a 129 x 135 grid, and 12,451 of those
17,415 points are sea. Keeping the 4,964 land columns as tenths of a millimetre in
int16, compressed, turns 607 MB into 22 MB. Against the NetCDFs, over all wells and
days: maximum difference 0.05 mm, mean 0.005 mm.

| Array | Shape | Meaning |
|---|---|---|
| `rain` | (9131, 4964) int16 | tenths of a millimetre; -9999 means no data |
| `lat_idx`, `lon_idx` | (4964,) | each column's row and column in the full grid |
| `lat`, `lon` | (129,), (135,) | 6.50-38.50 and 66.50-100.00, step 0.25 |
| `days` | (9131,) | 1998-01-01 to 2022-12-31 |

## Built outputs (`training/`)

| File | Rows | What it is |
|---|---|---|
| `tabular.parquet` | 216,455 | one row per well per season with a target |
| `rain_seq.npz` | 216,455 x 104 | weekly rain behind each row (the transformer's input) |
| `seq_channels.npz` | 216,455 x 104 x 6 | the BiLSTM's six weekly channels |
| `feature_spec.csv` | 46 columns | every column tagged feature, target or meta: the leakage guard |
