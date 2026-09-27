# Data cleaning — what was wrong with the raw data, and what we changed

This folder holds the cleaned, aligned groundwater and rainfall tables. This file lists every
gap found in the raw data, the evidence for it, and the change it led to. Nothing here was
edited by hand: every table is rebuilt from the raw files by the code in `ml/bits_ml/` and by
`clean_gaps.ipynb`. What each table is *for* is in [DATASETS.md](DATASETS.md).

```
data_cleaning/
  README.md          this file: every gap and the change it caused (tracked in git)
  DATA_CLEANING_SUMMARY.txt  everything done so far, step by step, in one plain-text file (tracked)
  DATA_AND_MODEL_PLAN.md     the data with real sample rows, and how the models will be built (also .txt)
  DATASETS.md        every dataset, raw and cleaned, and its role (tracked in git)
  clean_gaps.ipynb   closes the gaps the pipeline leaves: specific yield, aquifer type, cell x season (tracked)
  MODELLING_GAPS.md  the suggested modelling approach tested, training-data gaps, the training data (tracked)
  data/              the cleaned tables, ~210 MB (not in git; rebuild with the commands below)
    report/        coverage and flag summaries
    qc_audit.csv   what each quality rule costs, in readings and wells
    core_dictionary.csv   every column of every table, with its meaning
```

**Inputs.** CGWB groundwater levels, raw file inside
`data/Quality_controlled_groundwater_levels_over_India.zip` (32,299 wells, 2000–2024).
IMD daily rainfall, 0.25° grid, `data/imd_rainfall/imd_rf25_1998.nc` … `2022.nc`.
District outlines, `data_cleaning/reference/districts.geojson` (724 districts, post-2020 boundaries).

**Scope.** 1998–2022 for rain, 2000–2022 for water levels. The 2023–24 readings in the raw file
are kept but not used, because there is no rain for them.

---

## A. Groundwater (CGWB)

### A1. The file we used was already filtered, and the filter was biased
**Gap.** The project used the published file with 2,759 wells. Rerunning the source paper's
quality control on the raw file gives exactly the same 2,759 wells, which shows where the other
29,540 went:

| step | wells left |
|---|---|
| raw file | 32,299 |
| has any reading 2000–22 | 29,537 |
| no negative reading anywhere (drops the whole well) | 28,612 |
| **at least two readings in every year 2000–22** | **2,876** |
| no value repeating more than twice in a row | 2,759 |

The fourth rule removes 25,736 wells by itself. It checks completeness, not quality: a well with
91 of its 92 readings is dropped if one year has only one reading. The dropped wells are also
more likely to be deepening, so the published set understates the decline (46% of the added
wells are deepening, against 38% of the published ones).

**Change.** We start from the raw file and drop nothing. Bad readings are flagged (A2), and wells
are selected only at the end, into two views:

| view | wells | districts | states | dug wells |
|---|---|---|---|---|
| `view_strict` — the published set, kept as a control | 2,759 | 409 | 21 | 94% |
| `view_modelling` — enough readings in 2000–2014 to define a normal for every campaign | 8,629 | 483 | 22 | 80% |

### A2. Bad readings
**Gap.** The raw readings include values that cannot be right. **Change.** Each one gets a flag
naming the problem. Flagged readings stay in `well_obs` but are excluded from everything built on
top of it.

| flag | meaning | readings | wells |
|---|---|---|---|
| `flag_negative` | water above ground: a flowing well or a sign error | 1,429 | 925 |
| `flag_zero_suspect` | a zero in a well where zeros are over a fifth of the record: missing data written as 0 | 13,488 | 753 |
| `flag_below_bottom` | water deeper than the well was drilled | 26,257 | 2,755 |
| `flag_outlier` | over 3σ from the well's own seasonal pattern (not its raw level, so real long declines survive) | 9,587 | 7,485 |
| `flag_repeat_run` | three or more identical readings in a row | 22,448 | 1,950 |
| **valid** | | **1,065,392 of 1,125,369 (94.7%)** | |

A zero on its own is not treated as invalid, because shallow wells can genuinely fill to the
surface in the monsoon. Bad data is concentrated: Telangana has 16.7% suspect zeros and 17.9%
below-bottom readings, against 0.2% and 0.8% for dug wells nationally. Bore wells and piezometers
account for nearly all of it.

### A3. Wells with no unique ID, and rounded coordinates
**Gap.** The station code was damaged by scientific notation, so it can't identify a well.
3,710 wells share a coordinate with another well, and 2,999 have coordinates rounded to a coarse
grid (e.g. 0.1°).
**Change.** Each well is keyed by its coordinates, with a stable suffix where wells share a site
(`well_uid`). Rounded wells are flagged (`coord_rounded`), and their rain is averaged over the
whole area the true position could be in, instead of read at one point.

### A4. District and state names can't be trusted
**Gap.** 610 district names mix spellings (Rajnadgaon/Rajnandgaon, Ranga reddy/Rangareddy,
Jhunjhunu/Jhunjhunun, …), old and new names (Trichur/Thrissur, Belgaum/Belagavi), and boundary
vintages (Telangana's 10 districts before the 2016 split beside its 33 after). Names can't be
joined reliably to boundaries or to each other.
**Change.** Location comes from **coordinates**. Each well is placed in the district outline it
stands in, and that becomes its `district` and `state`. The file's labels are kept as
`cgwb_district` and `cgwb_state`.

| placement | wells |
|---|---|
| inside an outline | 32,205 |
| just outside (coast or border), nearest outline within 5 km | 87 |
| more than 5 km from any outline, no district | 7 |

The label matches the outline for 82.5% of wells. Nearly all the rest are renames and splits,
which is exactly what placing by coordinates fixes.

### A5. State labels that are wrong
**Gap.** 489 wells (203 of them in `view_modelling`) stand in a different state from the one the
file gives, mostly wells filed under Andhra Pradesh that stand in Maharashtra, Rajasthan or
Gujarat. Either the label or the coordinates is wrong.
**Evidence.** A well's water level should move with the wells around it. For the 262 of these
wells with enough readings, the level correlates with wells in the district at its coordinates at
a median **0.39**, and with wells in the district it is labelled with at **0.17**. The coordinates
fit better for 69% of them.
**Change.** The label is taken to be wrong, and the wells stay where their coordinates put them.
`state_agrees = False` marks them. Andhra Pradesh and Telangana count as one state, because of
the 2014 split.

### A6. Specific yield is missing from the raw file
**Gap.** Specific yield (Sy, the share of the rock that holds water) is needed to turn a
water-level change into millimetres of water. The raw file has none. The source paper read it
off a hydrogeological map for its published wells only (2,880 values). The first pass filled the
other 29,419 wells with the median of their state and aquifer type, which is a guess.
**Evidence.** The map is in the archive (`Hydrogeological_map.tif`, 0.12° grid, five Sy classes
from 0.018 in hard rock to 0.13 in alluvium). Sampled at the published wells, it gives the
published value for **100%** of the 2,872 that fall on a map cell. Compared with the map, the
guesses were a different class for 8,996 wells, and for 75% of those they were more than 1.5×
off.
**Change** (`clean_gaps.ipynb`). Every well takes its Sy from the map: 29,307 directly, 112 on the
coast from the nearest map cell within two cells. Published values are unchanged. **All 8,629
modelling wells now have a measured Sy class**, and `rock_class` names the rock type. The first
pass's values are kept as `sy_ingest` / `sy_source_ingest`.

### A6b. Aquifer type and well type are blank for many wells
**Gap.** `aquifer` is "-" or "Not available" for 1,697 of the 8,629 modelling wells (13,312 of all
wells). `well_type` is "-" for 2,239 wells.
**Evidence.** A dug well is an open pit into the water table. Of dug wells with a recorded aquifer
type, 99.6% are unconfined (13,912 of 13,964), and dug-cum-bore wells 100% (17 of 17). Bore wells
(64%), tube wells (80%) and piezometers (77%) are too mixed to guess.
**Change** (`clean_gaps.ipynb`). Blank dug wells become Unconfined (6,308 wells, 1,229 of them
modelling); other blanks become `Unknown`, as do blank well types. `aquifer_source` records
`cgwb`, `inferred_dug_well` or `unknown`; the original values are kept as `aquifer_cgwb` and
`well_type_cgwb`. Aquifer type is now known or inferred for 94.6% of modelling wells.

### A7. Readings have a month but no day
**Gap.** Readings are labelled Jan, May, Aug and Nov of each year, with no day.
**Change.** Every reading is placed on the 15th of its month. Season boundaries are therefore
uncertain by about ±2 weeks (see C1).

### A8. Some survey rounds barely happened
**Gap.** May 2020 and May 2021 reached 1% of wells, August 2012 3%, January 2016 12%, and November
2018 45%.
**Change.** Nothing is filled in. Missing readings stay missing, and the season tables bridge them
explicitly (C3).

---

## B. Rainfall (IMD)

### B1. Sea and out-of-border cells are coded -999
**Gap.** The grid is 129 × 135 = 17,415 cells. Only 4,964 are land in India; the rest hold -999.
If handled naively, -999 becomes NaN, and summing a month over NaN turns it into a false 0 mm.
**Change.** -999 is read as missing. Non-land cells are dropped rather than written as zero, and a
month or season with any missing day is left empty, never zero.

### B2. The raw grid itself is clean
**Checked.** All 25 files have every day of their year (365/366). The 4,964 land cells are the same
in every year, with no missing days and no negative values. The largest daily value is 979 mm
(2022, north-east), which is consistent with real extreme events there, so it is kept. **No
change was needed.**

### B2b. Some land cells report 0 mm where the true value is missing
**Gap.** Besides the -999 code, IMD writes exactly 0.0 mm for every day of a year in some land
cells with no rain gauge nearby: 669 cell-years in 109 of 4,964 cells, mostly at the edges of the
grid (Arunachal Pradesh, Mizoram, Manipur, and the Gujarat/Rajasthan border). Saiha district
(Mizoram) reads 0 mm in 1998–2006, 2011 and 2012 and about 2,000–3,000 mm in other years. A further
2,054 monsoon cell-months are exactly 0 where that month's median is over 100 mm.
**Effect.** 26 districts take rain from these cells (5 of them entirely) and 30 modelling wells
stand in them. Their variability and trends are distorted: Saiha's apparent +1,296 mm/decade trend
comes from the zeros.
**Change.** `rain_panel` flags these cell-months `suspect_zero` (8,693 in total) and sets them
missing; the season tables blank their days, so a season touching one is incomplete, not dry; and
every average over cells (a well's four cells, a district's cells) uses only cells with data and
renormalises their weights, with `rain_cover` recording the share. Saiha now reads 2,538 mm a year
with 31% variability, instead of 1,402 mm and 100%. Evidence: `data/report/rain_report.txt`
section 10, `data/report/rain_suspect_zeros.csv`.

### B3. Rain is on a grid, wells are points
**Gap.** A well is rarely at a grid point. Using the cell a well stands in ignores that a well near
a cell edge gets most of its rain from the neighbouring cell.
**Change.** Rain at a well is interpolated from the four surrounding grid points, with weights
renormalised over land points (`bridge_well_cell`). Tested by hiding grid points and predicting
them back: monthly error 17.7%, against 25.6% for using the nearest point. Wet days and the
wettest day are counted at the grid points first and then interpolated, which keeps them
unbiased: +0.2% and −0.7%, against +12% and −12% the other way round.

---

## C. Putting the two on the same clock

### C1. Rain is daily, water level is quarterly
**Gap.** The two can't be compared until they share a time step, and calendar quarters would split
the monsoon across the August reading.
**Change.** Four **seasons**, each running from one reading to the next and cut on the 15th:

| season | from | to | days | ends with |
|---|---|---|---|---|
| Nov-Jan (post-monsoon) | Nov 16 | Jan 15 | 61 | January reading |
| Jan-May (dry) | Jan 16 | May 15 | 120/121 | May reading |
| May-Aug (early monsoon) | May 16 | Aug 15 | 92 | August reading |
| Aug-Nov (late monsoon) | Aug 16 | Nov 15 | 92 | November reading |

The cut falls mid-month, so season rain is summed from the daily grid. Checked against the raw
NetCDF: it matches to within 0.0001 mm, including seasons that cross a year boundary and leap years.
Each season has 99 instances, from Jan-May 1998 to Aug-Nov 2022.

### C2. Water level and rain are different kinds of quantity
**Gap.** A water level is how full the aquifer is at one moment (a stock). Rain is water arriving
over a period (a flow). Rain in a season doesn't explain the level; it explains the **change** in
level over that season.
**Change.** Each well-season has the level at its closing reading and the change since the
previous one. Specific yield converts the change into millimetres of water, so rain in and water
stored are in the same unit:

```
delta_h_m         = depth now − depth at previous reading     (+ = water fell)
storage_change_mm = −delta_h_m × specific yield × 1000        (+ = water gained)
recharge_ratio    = storage_change_mm / rain over the same span
```

Median result across the modelling wells:

| season | change in level | rain | share of rain stored |
|---|---|---|---|
| Nov-Jan | falls 0.65 m | 7 mm | — |
| Jan-May | falls 1.42 m | 39 mm | — |
| May-Aug | **rises 1.94 m** | 482 mm | **+11.6%** |
| Aug-Nov | falls 0.15 m | 359 mm | ≈ 0 |

The May-Aug figure is within CGWB's own published infiltration range (8–15%). Outside the two
monsoon seasons, the ratio is a net loss per millimetre of rain, not recharge.

### C3. A missing reading breaks the chain
**Gap.** If one reading is missing, the change over that season can't be measured.
**Change.** The change is taken from the last valid reading, up to one year back, with rain summed
over the whole span. `span_seasons` records how far back that was. 88% of changes come from the
season immediately before. Nothing is interpolated.

---

## D. Putting the two on the same map

### D0. Evidence for the conversion
`ml/scripts/rain_grid_mapping.py` rebuilds the grid → district → state mapping from scratch and
logs every step to `data/report/rain_grid_mapping.txt`: the raw grid, cells as boxes, the cut by
district outlines, checks (identical to the stored bridge; weights sum to 1 in every district;
gridded land 3.25 million km² against India's 3.29 million), the district → state roll-up, and one
district (Pune, May-Aug 2019) computed by hand from the raw NetCDF, matching the table to
0.00003 mm. A plain mean of Pune's cells would give 1,183 mm against the correct area-weighted
903 mm. The mappings are exported as `grid_district_map.csv` and `grid_state_map.csv`.

### D1. Grid cells cross district boundaries
**Gap.** A 0.25° cell (about 27 km) often lies in two or more districts. Earlier, a district's rain
was averaged over the cells its wells were in. That left districts without wells with no rain,
and counted cells shared between districts in full for each.
**Change.** Each cell's box is intersected with the district outlines, and a district's rain is the
area-weighted mean of the cells it covers (`bridge_cell_district`). 3,449 cells are split between
districts. **720 of 724 districts now get rain**, up from 702; the 4 missed are islands off the
land mask. 91 land cells lie outside every outline, beyond the drawn border.

---

## E. Gaps still open

Closed by `clean_gaps.ipynb`: specific yield (A6), aquifer and well type (A6b), and the missing
cell × season table with wells (`fact_cell_season_full`). What is left has nothing in the data
to fix it with:

| gap | size | effect | handling |
|---|---|---|---|
| aquifer type unknown | 468 of 8,629 modelling wells (bore, tube, piezometer) | these wells can't be split by confinement | labelled `Unknown`; never guessed |
| reading day unknown | all readings | season boundaries uncertain by about ±2 weeks | stated; readings placed on the 15th |
| thin survey rounds | May 2020 and May 2021 1%, Aug 2012 3%, Jan 2016 12% of modelling wells read | those seasons rest on few wells | left missing; `span_seasons` bridges them |
| Sy is a class, not a measurement | all wells | five values from a national map; local rock varies within a class | stated; `delta_h_m` in metres does not depend on it |
| rain-gauge density changes over time | IMD grid | the grid is smoother where there were fewer gauges, and the files don't record it | stated |
| wells are not a random sample | all | coverage follows where CGWB drilled; 80% of modelling wells are shallow dug wells | stated |
| rain is not the only driver | all | canals, cropping and pumping also move the level | stated; the change is "not only rain" |
| edge cases | 7 wells with no district, 4 island districts with no rain, 91 land cells outside every outline | small | reported in the notebook, left as they are |
| 2023–24 | 53,380 readings | not used | out of scope by decision |

---

## Tables in `data/`

| table | one row per | rows |
|---|---|---|
| `well_master` | well, all of them, with location, construction, flags and views | 32,299 |
| `well_obs` | reading, flagged, never dropped | 1,125,369 |
| `dim_cell` | IMD grid cell, land or not, with its main district | 17,415 |
| `dim_month` | month, 1998-01 to 2022-12 | 300 |
| `dim_season` | season, Jan-May 1998 to Aug-Nov 2022 | 99 |
| `bridge_well_cell` | well × the grid cells its rain comes from, with weights | 129,267 |
| `bridge_cell_district` | land cell × district, by area | 10,272 |
| `well_cell_weights` | well, with its weight inside its own cell | 32,299 |
| `fact_cell_month` | cell × month: rain | 1,489,200 |
| `fact_cell_month_full` | cell × month: rain and the wells in the cell | 1,489,200 |
| `fact_well_month` | well × month: rain every month, level in four | 2,588,700 |
| `fact_district_month` | district × month | 216,000 |
| `fact_cell_season` | cell × season: rain | 491,436 |
| `fact_cell_season_full` | cell × season: rain and the wells in the cell | 491,436 |
| `fact_well_season` | well × season: rain, level, change, storage | 854,271 |
| `fact_district_season` | district × season | 71,280 |

Every column is described in `data/core_dictionary.csv`. Well and district tables cover
`view_modelling` unless stated.

## Rebuilding

From `ml/`, with the raw files in `data/`:

```bash
python -m bits_ml.ingest        # raw CGWB -> well_master, well_obs, qc_audit (placement by coordinates)
python -m bits_ml.grid          # grid cells, months, well->cell and cell->district bridges
python -m bits_ml.rain_panel    # daily IMD -> fact_cell_month
python -m bits_ml.panel         # monthly well, cell and district tables
python -m bits_ml.seasons       # the four seasons: dim_season and the three season tables
python -m scripts.core_report   # summaries in data/report
# then run data_cleaning/clean_gaps.ipynb top to bottom: Sy from the map, aquifer type,
# reruns panel + seasons with them, builds fact_cell_season_full, refreshes the dictionary
pytest -q                       # 145 tests
```

The notebook must be re-run after any re-run of `bits_ml.ingest`, which rewrites `well_master`.

A full rebuild takes a few minutes. Design details are in `ml/CORE.md`.
