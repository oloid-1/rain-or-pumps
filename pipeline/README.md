# ml — the code that builds the data layer

| module | builds |
|---|---|
| `bits_ml/ingest.py` | every CGWB well and reading from the raw archive, with quality flags; location from coordinates |
| `bits_ml/districts.py` | wells and grid cells placed in district outlines |
| `bits_ml/grid.py` | the grid spine, well → cell and cell → district bridges |
| `bits_ml/rainfall.py` | reading the IMD NetCDF files; rain at a point from its four grid neighbours |
| `bits_ml/rain_panel.py` | monthly rain per cell, IMD zeros that are really missing flagged |
| `bits_ml/panel.py` | monthly well, cell and district tables |
| `bits_ml/seasons.py` | the four seasons between readings: rain, level, change, storage |
| `bits_ml/training_data.py` | model-ready rows, features, splits and rain sequences |
| `scripts/core_report.py` | the groundwater report |
| `scripts/rain_grid_mapping.py` | evidence of the grid → district → state conversion |
| `scripts/rain_report.py` | the rain report |
| `scripts/core_dictionary.py` | the data dictionary, generated from the tables |
| `scripts/interpolation_check.py` | evidence that four-neighbour rain beats the nearest grid point |

Run order, design and findings: [CORE.md](CORE.md). What each output is for:
[../data_cleaning/DATASETS.md](../data_cleaning/DATASETS.md). Tests: `pytest -q` from here.

The earlier modelling code (ridge, LightGBM, attribution, what-if, API) is in
[../old/ml](../old/README.md).
