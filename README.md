# Rain or Pumps?

**Attributing India's Groundwater Decline District by District**

BITS Pilani PGCP AI and ML capstone, Group 18
Kakara Siva Kumar · Rainchwar Parth · Vipul Aggarwal · Mentor: Sudharshan Deshmukh

---

## The idea in three sentences

Groundwater levels across India are falling, but rainfall varies naturally from year
to year, so a decline can come from a run of dry years, from pumping, or from both.
We model where the water level *should* be given the rain that actually fell, and
treat the part of the decline that rainfall does not explain as a proxy for
extraction pressure.

Extraction is never an input. It is the output.

## Quick start

```bash
pip install -r requirements.txt

make data        # unpack the rainfall cube and build the training table  (~1 min)
make train       # baselines + transformer, prints the comparison table   (~25 min CPU)
make report      # attention figure and the district residual table
```

`make data` needs no download. The rainfall record ships with the repository as a
22 MB packed cube; see [data/README.md](data/README.md).

## Current results

Validation 2015-2017, every model on the same rows, target `delta_h_m`:

| model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| zero, predict no change | 2.163 | 3.227 | 0.000 |
| season mean | 1.813 | 2.756 | 0.271 |
| ridge on tabular | 1.753 | 2.872 | 0.208 |
| LightGBM on tabular | 1.526 | 2.435 | 0.431 |
| transformer, sequence + tabular | 1.531 | 2.461 | 0.418 |
| transformer, sequence only | 1.641 | 2.618 | 0.342 |

Held-out test 2018-2022, transformer: MAE 1.457 m, RMSE 2.302 m, R² 0.439.

The transformer matches LightGBM. That is the claim: attention over the raw weekly
rainfall series recovers gradient boosting on hand-engineered rain windows, without
being told which windows matter. The sequence-only ablation reaching R² 0.342 says
the rain history is carrying the model, which is what the attribution needs.

## Repository layout

| Folder | What is in it |
|---|---|
| `data/` | Inputs and built tables. Only the packed rainfall cube is tracked; see `data/README.md` |
| `data_cleaning/` | District outlines and the gap-closing notebook that the pipeline reads |
| `pipeline/` | The data pipeline: raw archive to clean core tables to a training set |
| `models/` | Model code: the direct training-table builder, the transformer, training and reporting |
| `notebooks/` | Exploration and the gap-closing notebook |
| `docs/` | Problem statement, execution plan, methodology notes, the end-to-end review |
| `decks/` | Review presentations |
| `reports/` | Generated figures and tables |
| `scripts/` | Data fetching and the rainfall cube builder |
| `archive/` | The superseded first pipeline. Kept for the FastAPI endpoint, the what-if function and the atlas dashboard, which weeks 7 and 8 still need. See `archive/ARCHIVE.md` |

### Where to start reading

| If you want | Read |
|---|---|
| The whole story in one pass | `docs/05_end_to_end_review.docx` |
| Every cleaning decision and its evidence | `docs/data_cleaning/DATA_CLEANING_SUMMARY.txt` |
| Why wells are not snapped to grid nodes | `docs/04_spatial_merge_methodology.docx` |
| What mbgl, Sy and the rest mean | `docs/03_glossary.docx` |
| The model contract and the commands | `models/START_HERE.md` |
| The measured results | `models/RESULTS.md` |

## The data

Two public sources, no common join key.

| Source | What it is | Licence |
|---|---|---|
| CGWB groundwater levels | Depth to water at monitoring wells, four readings a year, 2000-2022. Raw archive 32,299 wells; the published quality-controlled extract 2,759 | CC BY 4.0, figshare doi 10.6084/m9.figshare.29293877.v3 |
| IMD gridded rainfall | Daily rainfall on a 0.25 degree grid, 1998-2022, 4,964 land cells | Open access, imdpune.gov.in |

Wells sit at arbitrary surveyed coordinates and only 3 of 2,759 land on a grid node,
so rainfall is transferred to each well by **masked bilinear interpolation** over the
four surrounding nodes, with sea and missing corners dropped and the remaining
weights renormalised. Nearest-node snapping, which this replaced, differs by 10.1
percent of mean monthly rainfall.

## Two builds of the same contract

Both produce the same target, the same forbidden columns, the same split years and
the same five district folds, with matching column names.

| | `pipeline/` (full rebuild) | `models/build_training_data.py` (direct) |
|---|---|---|
| Source | Raw 32,299-well archive | Published 2,759-well extract |
| Rows with a target | 694,253 | 216,455 |
| Features | 25 | 33 |
| Rain sequence | 52 weeks per land cell + stencil | 104 weeks per row |
| Prerequisites | Raw zip, district outlines, geopandas, the gap-closing notebook | The two supplied files, or just the packed cube |
| Status | Code complete | Built and trained; current results come from this |

The direct build exists so the model work could start without the pipeline's
prerequisites. Switching to the full table is a path change, not a rewrite.

## The target, and the one rule that matters

```
delta_h_m = depth now  -  depth at the previous reading of the same well
```

**Positive means the water level fell.**

Never a feature, enforced by an assertion that fails the build: any past water level,
the year, latitude, longitude, district, state, well id. Feed the previous depth in
and the model reaches R² near 0.9 by learning mean reversion, and the residual stops
meaning anything. Rainfall normals are computed from training years only. The split
is blocked by year: train 2000-2014, validation 2015-2017, test 2018-2022.

## Rebuilding from the raw archive

Only needed for the full 32,299-well table.

```bash
python3 scripts/fetch_data.py --all        # raw CGWB archive and IMD NetCDFs
cd pipeline
python -m bits_ml.ingest
python -m bits_ml.grid
python -m bits_ml.rain_panel
python -m bits_ml.panel
python -m bits_ml.seasons
jupyter nbconvert --execute --inplace ../notebooks/02_clean_gaps.ipynb
python -m bits_ml.training_data
pytest                                     # 79 tests
```

## Attribution

CGWB quality-controlled groundwater level dataset with specific yield over India,
figshare, doi 10.6084/m9.figshare.29293877.v3, CC BY 4.0.

Pai, D.S. et al. (2014). Development of a new high spatial resolution (0.25 x 0.25)
long period (1901-2010) daily gridded rainfall data set over India. *MAUSAM* 65(1), 1-18.

District outlines: post-2020 vintage, 724 districts in 36 states and union territories.
