# Modelling: the suggested approach, the data gaps, and the training data

The goal: **given the rain that falls, what happens to groundwater?** Train two architectures
on the same data and compare them. This file records what was suggested, what the data
supports, the gaps found on the way, what was done about each, and how the training data is
laid out.

For gaps in the raw data see [README.md](README.md); for every table see [DATASETS.md](DATASETS.md).

---

## 1. The suggested approach, tested against the data

Suggestion: map wells to the grid with a distance function and take a centroid as the well's
position; train an RNN that reads 30 days of rain and predicts rain at the well on day 31, feeding
its own output back for day 32 onward; then relate that rain to future groundwater; compare two
architectures.

| part | verdict | evidence |
|---|---|---|
| lat/long as the key; distance weighting when many grid points serve one well | **kept, already in place** | each well reads rain from its 4 surrounding grid points, weighted by distance (`bridge_well_cell`). Tested against the grid itself: 17.7% monthly error against 25.6% for the nearest point |
| centroid of grid points as the well's lat/long | **rejected** | the well's position is known exactly; that is where rain must be read. A centroid would move every well off its own position |
| RNN predicting rain at the well from the last 30 days | **rejected** | rain at every well is already observed, every day 1998–2022. Tested on 300 random cells, 2018–22: the last 30 days predict the next day at R² 0.145, *worse* than the calendar average for that day (0.17) |
| feeding predictions back in (day 31, 32, …) | **rejected** | R² 0.22 one day ahead, 0.17 after a week, 0.16 after 30 days: back to the average. The forecast's spread falls to a third of reality (4.8 mm vs 15.0 mm), losing the dry spells and downpours that drive recharge. Yearly: one monsoon predicts the next at +0.04 nationally |
| "backpropagation will normalise" | **corrected** | backpropagation computes gradients; inputs must be scaled separately |
| cost = difference between prediction and truth | **kept** | squared or absolute error on the groundwater target |
| rain in → impact on future groundwater | **kept: this is the project** | the problem statement's rain-expected groundwater model |
| two architectures, trained and compared | **kept** | matches the required deliverables |

**What is built instead.** Both architectures take *observed* rain in and give *groundwater change*
out. "Future" groundwater is asked with **scenario rain** (a year's rain +20%, a dry year from the
record, the normal), which is what the what-if simulator does. Scenario rain is chosen, not
forecast, so it is reliable.

| | A — many inputs, one output | B — recurrent |
|---|---|---|
| models | ridge (baseline), LightGBM, optionally an MLP | LSTM or GRU |
| rain input | 25 features: this season, previous seasons, 1/2/3-year totals and % of normal | 52 weekly totals up to the reading |
| fixed input | specific yield, depth, aquifer, well type, rock class, normal rain | the same |
| output | `delta_h_m`: change in water level since the previous reading | the same |
| data | `training/tabular.parquet` | `training/tabular.parquet` + `sequence_cells.npz` |

---

## 2. Data gaps for training, and what was done

### Fixed

| gap | evidence | fix |
|---|---|---|
| **IMD zeros that are really missing** | 669 cell-years of exactly 0 mm in 109 cells, plus 665 more monsoon months at 0 where the median is over 100 mm; Saiha district read 0 mm in 11 of 25 years, giving a false +1,296 mm/decade trend | `rain_panel` flags 8,693 cell-months `suspect_zero` and sets them missing; seasons blank those days; every average over cells uses only cells with data (`rain_cover` records the share). Saiha now reads 2,538 mm with 31% variability, not 1,402 mm and 100% |
| **four states missing from the training wells** | requiring all four readings excluded every well in Kerala (0 of 1,947) and Assam (0 of 478), and all but 4 in West Bengal and Odisha: their raw record has no May reading | `view_training` needs 3 of 4: **11,455 wells** (from 8,629), 27 states. Kerala 703, West Bengal 701, Odisha 837, Assam 211. Their Jan → Aug change spans two seasons (`span_seasons` = 2), with rain summed to match |
| **leakage through past water levels, place and time** | a model given the previous level, the year or the district explains the decline with itself | none of them is a feature; `feature_spec.csv` gives every column a role, and `check_no_leakage` stops the build if a forbidden one is listed |
| **leakage through normals** | normals computed on all years let test years shape the features | rain and level normals come from 2000–2014 only, the training years |
| **wells sharing a cell are not independent** | 78% of training cells hold more than one well; a random row split puts near-copies in train and test | fixed splits by year (train 2000–14, valid 2015–17, test 2018–22) plus `district_fold` (5 folds of whole districts) |
| **no sequence input for a recurrent model** | only season totals existed | weekly rain for the 52 weeks up to every reading, from the daily grid, stored per cell (51 MB) with a loader for any well |

### Still open (to handle while training, or to state in the report)

| gap | size | how to handle |
|---|---|---|
| **rain explains little on its own** | correlation of season rain with the change: −0.08 to −0.16 across the four seasons | expected: the unexplained part is the pumping signal the project measures. Compare against a no-skill baseline, report per season. Pumping, irrigation, cropping, canals, soil and evaporation data would raise the ceiling but are not in the data |
| changes spanning more than one season | 17% of rows (13% span 2, 4% longer) | keep `span_seasons` as an input, or train on span 1 only and compare |
| extreme changes | 0.24% of rows beyond ±20 m; 1st–99th percentile −10.4 to +8.4 m | robust loss (Huber) or cap the target |
| wells drop out over time | 10,619 wells with a target in 2005, 9,910 in 2014, 8,053 in 2022 | also score on wells present in both train and test |
| thin test seasons | Jan-May has 16,806 test rows against 32–37k for the others; May 2020 and May 2021 reached 1% of wells | score per season, never one pooled number |
| imbalance | 80% dug wells; Unknown aquifer for 735 training wells (6.4%) | report per well type; `Unknown` stays a category, never guessed |
| specific yield is a class | five values from a national map | fine as an input; storage in mm inherits it, `delta_h_m` does not |
| three-year history needs three years | `rain_12s_*` missing for 3.4% of rows (the first seasons) | LightGBM takes NaN; ridge and the LSTM need an imputation or a later start |
| state label disagrees with coordinates | 13,628 rows | kept at the coordinates (README A5); `state_agrees` allows a check without them |
| Western Ghats rain around 2004–07 | Pune district's cells jump 2–3× in 2004–07 | likely real (the 2005 Maharashtra floods) but possibly gauge-network change; stated, not changed |
| rain-gauge density changes over time | whole grid | stated; the IMD files don't record it |

---

## 3. The training data (`data/training/`)

| file | what it holds |
|---|---|
| `tabular.parquet` | 694,253 rows: one per well × season with a target. Train 489,707 · valid 85,250 · test 119,296 |
| `feature_spec.csv` | every column with its role (`id`, `group`, `split`, `target`, `target_alt`, `feature`, `excluded`, `filter`), meaning and % missing |
| `sequence_cells.npz` | `weeks` (99 seasons × 52 weeks × 4,964 land cells, mm), `cell_ids`, `season_idx`, `season_end` |
| `stencil.parquet` | each training well's cells and weights, to turn cell sequences into well sequences |

**The 25 features:** this season's rain, rain since the previous reading, span in seasons, % of
normal, wet days, wettest day, normal, `rain_cover`, season length; rain 1, 2 and 3 seasons before;
rain over the last 4, 8 and 12 seasons and each as % of normal; specific yield, drilled depth,
normal annual rain; and the categories season, aquifer, well type, rock class.

**Targets:** `delta_h_m` (main). Alternatives: `storage_change_mm` (the same in mm of water) and
`anomaly_ref_m` (level against the well's normal).

### Loading

```python
import numpy as np, pandas as pd
from bits_ml import training_data as td

T = "data_cleaning/data/training/"
tab = pd.read_parquet(T + "tabular.parquet")
spec = pd.read_csv(T + "feature_spec.csv")
features = spec.loc[spec.role == "feature", "column"].tolist()

train = tab[tab.split == "train"]
X, y = train[features], train["delta_h_m"]          # architecture A

seq = np.load(T + "sequence_cells.npz", allow_pickle=True)
stencil = pd.read_parquet(T + "stencil.parquet")
weeks = td.well_sequences(train.head(1000), seq["weeks"], seq["cell_ids"], stencil)   # (1000, 52) for B
```

### Rules for training

1. Tune on `valid`; open `test` once, at the end.
2. Report per season, and on held-out districts (`district_fold`) as well as held-out years.
3. Compare every model against "no change" and against the ridge baseline.
4. Scale inputs for ridge, MLP and LSTM with statistics from `train` only.
5. For the what-if simulator, check that more rain never predicts deeper water (LightGBM can
   enforce it; for the LSTM, test it with rain scaled up and down).

### Rebuilding

From `ml/`, after the core pipeline and `clean_gaps.ipynb`: `python -m bits_ml.training_data`.
Tests: `tests/test_training_data.py`.
