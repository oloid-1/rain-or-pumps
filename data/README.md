# data/

## What is tracked and what is not

| Path | Tracked | Size | How to get it |
|---|---|---|---|
| `derived/rain_cube.npz` | **yes** | 22 MB | in the repository |
| `imd_rainfall/imd_rf25_*.nc` | no | 607 MB | `python3 scripts/fetch_data.py --imd` |
| `cgwb/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv` | no | 2.8 MB | `python3 scripts/fetch_data.py --cgwb` |
| `Quality_controlled_groundwater_levels_over_India.zip` | no | 26 MB | `python3 scripts/fetch_data.py --archive` |
| `training/` | no | 55 MB | `make data` |
| `processed/` | no | 20 MB | the week 1 notebook |
| `../data_cleaning/data/` | no | large | the pipeline steps |

Nothing untracked is needed to train the model. `make data` reads the packed cube
and the CGWB extract, and the CGWB extract is 2.8 MB and fetched in a second.

## Why the rainfall record is packed

The IMD files store a value for every point of a 129 x 135 grid, and 12,451 of those
17,415 points are sea, coded -999. Only 4,964 are land. Keeping the land columns
only, as tenths of a millimetre in int16, and letting zlib have the 72 percent of
entries that are exactly zero, turns 607 MB into 22 MB.

The loss is rounding to 0.1 mm. Measured against reading the NetCDFs directly, over
all 2,759 wells and all 9,131 days: **maximum difference 0.05 mm, mean difference
0.005 mm.** Both code paths exist in `models/build_training_data.py` and the cube is
used when present, so the check can be repeated at any time.

Rebuild it with `make cube` after `make fetch`.

### Contents of `derived/rain_cube.npz`

| Array | Shape | Meaning |
|---|---|---|
| `rain` | (9131, 4964) int16 | tenths of a millimetre; -9999 means no data |
| `lat_idx`, `lon_idx` | (4964,) int16 | each land column's row and column in the full grid |
| `lat` | (129,) float64 | 6.50 to 38.50, step 0.25 |
| `lon` | (135,) float64 | 66.50 to 100.00, step 0.25 |
| `days` | (9131,) int32 | days since 1970-01-01, 1998-01-01 to 2022-12-31, no gaps |

## Sources and licences

**CGWB quality-controlled groundwater levels with specific yield over India.**
figshare, doi 10.6084/m9.figshare.29293877.v3, CC BY 4.0. The archive holds 32,299
wells; the published extract is the 2,759 that survive the source paper's quality
control. Both are used: the extract as a control group, the archive as the modelling
base. The archive also carries the hydrogeological map that supplies specific yield.

**IMD 0.25 degree gridded daily rainfall.** imdpune.gov.in, open access, no
registration. Constructed by inverse distance weighting from one to four stations
within a 1.5 degree radius (Pai et al., 2014), so the 0.25 degree spacing is a
presentation resolution rather than an information resolution. Redistribution terms
are less explicit than CGWB's, which is the second reason the NetCDFs are fetched
rather than tracked.

## Built outputs

| File | Rows | What it is |
|---|---|---|
| `training/tabular.parquet` | 216,455 | one row per well per season with a target |
| `training/rain_seq.npz` | 216,455 x 104 | weekly rainfall behind each row, oldest week first |
| `training/feature_spec.csv` | 46 columns | every column tagged feature, target or meta. This is the leakage guard |
| `training/build_report.txt` | — | counts and split sizes from the last build |
