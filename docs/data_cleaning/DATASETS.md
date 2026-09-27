# Datasets — what each one is and what it is for

The project joins two very different sources: **groundwater levels** (CGWB: 32,299 wells, read
four times a year) and **rainfall** (IMD: a daily 0.25° grid). This file lists every dataset,
raw and cleaned, and what role it plays. For *why* a dataset was changed, see
[README.md](README.md). For every column, see `data/core_dictionary.csv`.

## How it fits together

```
 RAW INPUTS                           CLEANED TABLES (data_cleaning/data)
 ─────────                            ─────────────────────────────────────────────────────
 CGWB raw wells ──ingest──► well_master (1 row / well) ──┐
                         └► well_obs    (1 row / reading)│
 hydrogeological map ──notebook──► well_master.sy, rock_class
 district outlines ──ingest/grid──► district on every well and cell
                                                         │
 IMD daily grid ──grid──► dim_cell (1 row / grid cell)   │
                └rain_panel──► fact_cell_month           │
                └seasons────► fact_cell_season           │
                                                         ▼
 bridges link the two:   bridge_well_cell      well ─► its 4 surrounding grid cells (rain)
                         well_cell_weights     well ─► the cell it stands in (grouping)
                         bridge_cell_district  cell ─► districts it overlaps (area share)
                                                         │
 time spines:            dim_month (300 months) ·  dim_season (99 seasons)
                                                         ▼
 analysis tables:        fact_well_*   ·   fact_cell_*_full   ·   fact_district_*
                         (month or season, at three scales: well, grid cell, district)
```

**Keys.** `well_uid` (a well), `cell_id` (a grid cell, e.g. `14.25_78.25`), `month`
(first day of the month), `season_idx` (0 = Jan-May 1998 … 98 = Aug-Nov 2022), and
`district` + `state` together (a district outline; names repeat across states, so use both).

---

## 1. Raw inputs (`data/`, not in git)

| dataset | what it is | role |
|---|---|---|
| `Quality_controlled_groundwater_levels_over_India.zip` → `Input/1_India_GWLs_2000_2024_wells_within_India.csv` | CGWB's raw file: 32,299 wells, one column per reading (Jan/May/Aug/Nov, 2000–2024), depth to water in metres below ground, plus location and construction | **the source of every groundwater table**. Nothing is removed from it; bad readings are flagged |
| same zip → `Hydrogeological_map/Hydrogeological_map.tif` | 0.12° grid of specific yield in 5 rock classes (0.018–0.13) | **gives every well its specific yield**, which converts metres of water level into mm of water |
| same zip → `Output/…`, `Code/…` | the source paper's filtered outputs and notebooks | reference only: used to prove the rebuild reproduces their 2,759 wells |
| `cgwb/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv` | the paper's published 2,759-well set with specific yield | **control arm** (`view_strict`), and the published specific yield |
| `imd_rainfall/imd_rf25_1998.nc` … `2022.nc` | IMD daily rainfall, 0.25° grid, 129 × 135 cells, mm/day; -999 outside land | **the source of every rainfall value** |
| `data_cleaning/reference/districts.geojson` (in git) | 724 district outlines, post-2020 boundaries | **decides the district** of every well and grid cell |

---

## 2. Cleaned tables (`data_cleaning/data/`)

### Wells — who and where

| table | one row per | rows | role |
|---|---|---|---|
| `well_master` | well (all 32,299) | 32,299 | **the well register.** Location (district/state from coordinates; CGWB labels kept as `cgwb_*`), construction (well type, aquifer, depth), specific yield and its source, rock class, quality counts, and the two views: `view_strict` (published 2,759) and `view_modelling` (8,629 with enough history). Filter on a view before analysis |
| `well_obs` | water-level reading | 1,125,369 | **every reading, never dropped.** Depth plus six quality flags and `valid`. Use `valid == True` |

### Grid and time — the frames everything hangs on

| table | one row per | rows | role |
|---|---|---|---|
| `dim_cell` | IMD grid cell | 17,415 | **the spatial frame.** Centre lat/lon, land or not, wells in the cell, main district |
| `dim_month` | month, 1998-01 → 2022-12 | 300 | **the monthly frame.** Calendar season, water year (June–May), whether a reading can exist that month |
| `dim_season` | season, Jan-May 1998 → Aug-Nov 2022 | 99 | **the season frame.** The four windows between readings (Nov-Jan, Jan-May, May-Aug, Aug-Nov), cut on the 15th, with start, end and length |

### Bridges — how wells, cells and districts connect

| table | one row per | rows | role |
|---|---|---|---|
| `bridge_well_cell` | well × grid cell | 129,267 | **where a well's rain comes from**: up to 4 surrounding cells with weights that sum to 1 |
| `well_cell_weights` | well | 32,299 | **which cell a well belongs to**, and its weight among the wells in that cell |
| `bridge_cell_district` | land cell × district | 10,272 | **how a cell splits across districts**, in km² and as a weight; district rain is the area-weighted mean |

### Facts — the analysis tables

Two time steps (month, season) at three scales (well, grid cell, district).

| table | one row per | rows | contains | use it for |
|---|---|---|---|---|
| `fact_cell_month` | cell × month | 1,489,200 | rain, wet days, wettest day; `suspect_zero` marks IMD zeros that are really missing (set missing) | **rainfall only**, the base all rain is read from |
| `fact_cell_month_full` | cell × month | 1,489,200 | the above plus the wells in the cell (count, anomaly mean/median/weighted, spread) | monthly rain against water level on the grid |
| `fact_well_month` | well × month (modelling view) | 2,588,700 | rain every month; level in the 4 reading months, with its anomaly from the well's normal | monthly time series per well |
| `fact_district_month` | district × month | 216,000 | area-weighted rain; wells read and their anomaly | district maps and monthly summaries |
| `fact_cell_season` | cell × season | 491,436 | rain, wet days, wettest day, rain as % of the season's normal | **seasonal rainfall only** |
| `fact_cell_season_full` | cell × season | 491,436 | the above plus the wells in the cell: count, anomaly, change, storage | seasonal rain against water level on the grid |
| `fact_well_season` | well × season (modelling view) | 854,271 | rain at the well; the level the season closes on; **change since the last reading, storage in mm, recharge ratio** | **the main rain-vs-groundwater table** |
| `fact_district_season` | district × season | 71,280 | area-weighted rain; wells read, median change, storage and recharge ratio | district-level season comparison |

**Month or season?** Use *season* to relate rain to groundwater: each season's rain is exactly
the rain that fell between the two readings around it. Use *month* for rainfall on its own, or
where calendar months are needed. In monthly tables, 8 of 12 months have no water level, by
design.

**Well, cell or district?** *Well* is the measurement itself. *Cell* matches the rain grid, but
31% of cell-seasons rest on a single well, so read `n_wells_read` alongside. *District* is the
unit for reporting and policy; every district gets rain, 40,893 district-seasons also have wells.

### Training data (`data/training/`)

| file | one row per | role |
|---|---|---|
| `tabular.parquet` | well × season with a target (694,253 rows, 11,455 wells in `view_training`: 3 of 4 campaigns) | **the model input for every architecture**: 25 rain and well features, target `delta_h_m`, year split and district folds |
| `sequence_cells.npz` | season × week × land cell | weekly rain for the 52 weeks up to each reading, for recurrent models |
| `stencil.parquet` | training well × cell | turns cell sequences into a well's sequence |
| `feature_spec.csv` | column | role of every column (feature, target, id, excluded), meaning, % missing |

How they were chosen and what is still open: [MODELLING_GAPS.md](MODELLING_GAPS.md).

### Reference files in `data/`

| file | role |
|---|---|
| `core_dictionary.csv` | every column of every table, with type, % missing and meaning; regenerated from the tables |
| `core_tables.csv` | one line per table: what one row is |
| `qc_audit.csv` | how many readings and wells each quality flag removes |
| `report/core_report.txt` | the groundwater side: coverage, flags, the water year, rain response (`ml/scripts/core_report.py`) |
| `report/rain_report.txt` | the rain side: quality, national/state/district climate, seasons, trends, heavy-rain days, persistence, where the wells sit, how rare +20% is, suspect zeros (`ml/scripts/rain_report.py`) |
| `report/rain_grid_mapping.txt` | evidence of the grid → district → state conversion, with a worked example (`ml/scripts/rain_grid_mapping.py`) |
| `report/grid_district_map.csv`, `grid_state_map.csv` | every land cell with its districts / states, overlap km², and weights |
| `report/rain_by_state.csv`, `rain_by_district.csv`, `rain_national_by_year.csv`, `rain_heavy_days_by_year.csv`, `rain_suspect_zeros.csv` | the rain report's tables |
| `report/coverage_*.csv`, `flags_by_state.csv`, `november_trends.csv` | the groundwater report's tables |

---

## 3. Earlier outputs (`old/data/processed/`) — superseded, archived

These files come from the earlier model pipeline (now in `old/`), which read the published
2,759-well file directly and not the cleaned tables: `well_readings_2000_2022`, `well_monthly_rain_1998_2022`,
`well_campaign_table`, `well_season_obs`, `final_dataset_district_season.csv`, the
`district_*` and `model_*` results, and the Colab bundle. They are kept for reference. **For
new data work, use `data_cleaning/data/`.**

---

## 4. How the tables are made

| step | command (from `ml/`) | writes |
|---|---|---|
| 1 | `python -m bits_ml.ingest` | well_master, well_obs, qc_audit |
| 2 | `python -m bits_ml.grid` | dim_cell, dim_month, bridge_well_cell, bridge_cell_district, well_cell_weights |
| 3 | `python -m bits_ml.rain_panel` | fact_cell_month |
| 4 | `python -m bits_ml.panel` | fact_well_month, fact_cell_month_full, fact_district_month |
| 5 | `python -m bits_ml.seasons` | dim_season, fact_cell_season, fact_well_season, fact_district_season |
| 6 | run `data_cleaning/clean_gaps.ipynb` | fills specific yield and aquifer type in well_master, reruns steps 4–5, builds fact_cell_season_full, refreshes the dictionary |
| 7 | `python -m scripts.rain_grid_mapping` then `python -m scripts.rain_report` | the rain evidence and report in `report/` |
| 8 | `python -m bits_ml.training_data` | the training data in `training/` |

Step 6 must be re-run after any re-run of step 1, which rewrites `well_master`.

## Reading a table

```python
import pandas as pd
ws = pd.read_parquet("data_cleaning/data/fact_well_season.parquet")
monsoon = ws[(ws["season"] == "May-Aug") & ws["delta_h_m"].notna()]
```

The tables are Parquet: compressed, column-based, and they keep their data types. For Excel,
export a slice, e.g. `ws[ws["state"] == "Karnataka"].to_csv("karnataka.csv")`. The full tables
are larger than Excel's 1-million-row limit.
