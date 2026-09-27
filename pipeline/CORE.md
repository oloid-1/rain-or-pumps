# core — the grid, the wells, and one monthly panel

Everything downstream reads `data_cleaning/data`. It is built from the raw
CGWB archive and the IMD grids, and nothing else.

## Why it was rebuilt

The file the project used until now is the published *output* of the source
paper's quality control. Replaying that cascade on the raw file reproduces it
exactly, which is how we know where the 29,540 missing wells went:

| step | wells left |
|---|---|
| raw file | 32,299 |
| has any reading 2000–22 | 29,537 |
| no negative reading anywhere (drops the whole well) | 28,612 |
| **at least two readings in every year 2000–22** | **2,876** |
| no value repeating more than twice in a row | 2,759 |

The fourth rule costs 25,736 wells on its own, and it is a completeness test,
not a quality test: a well with 91 of its 92 readings is thrown out if one year
holds only one of them. The wells it removes are not a random sample — they
deepen more often than the survivors, so the published set understates decline.

So this layer flags instead of dropping. Every reading is kept with a flag
naming what is wrong with it; the two *views* at the end are the only place
anything is excluded, and either can be rebuilt without touching the ingest.

| view | wells | districts | states | dug wells |
|---|---|---|---|---|
| `view_strict` — the published set, kept as a control arm | 2,759 | 409 | 21 | 94% |
| `view_modelling` — enough training readings to define a normal | 8,629 | 483 | 22 | 80% |

## Run order

From `ml/`, each step reading only what the step before wrote:

```bash
python -m bits_ml.ingest        # raw archive -> well_master, well_obs, qc_audit
python -m bits_ml.grid          # dim_cell, dim_month, the two bridges, cell weights
python -m bits_ml.rain_panel    # IMD NetCDF -> fact_cell_month  (the only NetCDF reader)
python -m bits_ml.panel         # fact_well_month, fact_cell_month_full, fact_district_month
python -m bits_ml.seasons       # dim_season, fact_cell_season, fact_well_season, fact_district_season
# then run data_cleaning/clean_gaps.ipynb: specific yield from the map, aquifer type, fact_cell_season_full
python -m scripts.core_report      # what the tables contain, printed and written to core/report
python -m scripts.core_dictionary  # core_dictionary.csv, generated from the tables themselves
```

## The tables

| table | grain | rows |
|---|---|---|
| `well_master` | well, all of them | 32,299 |
| `well_obs` | reading, flagged, never dropped | 1,125,369 |
| `dim_cell` | IMD cell, land or not | 17,415 |
| `dim_month` | month, 1998-01 to 2022-12 | 300 |
| `bridge_well_cell` | well to the cells its rain comes from | 129,267 |
| `bridge_cell_district` | land cell x district outline, by area | 10,272 |
| `well_cell_weights` | a well's inverse-distance weight in its own cell | 32,299 |
| `fact_cell_month` | cell x month: rain | 1,489,200 |
| `fact_well_month` | well x month: rain and level | 2,588,700 |
| `fact_cell_month_full` | cell x month: rain and the wells in the cell | 1,489,200 |
| `fact_district_month` | district outline x month | 216,000 |
| `dim_season` | season, 1998 Jan-May to 2022 Aug-Nov | 99 |
| `fact_cell_season` | cell x season: rain | 491,436 |
| `fact_well_season` | well x season: rain, level, change, storage | 854,271 |
| `fact_district_season` | district outline x season | 71,280 |

`core_dictionary.csv` describes all 251 columns and is regenerated from the
tables, so it cannot fall behind them.

### Two attachments, not one

A well is joined to the grid twice and they are different questions:

* **home cell** — the cell the well stands in (`well_cell_weights`, and
  `is_home` in the bridge). This is how wells are grouped so a cell can be
  summarised as one series.
* **rain stencil** — the up to four cells its rain is read from, with weights
  (`bridge_well_cell.weight_land`, summing to one per well). A well near a cell
  edge takes most of its rain from the neighbour. Against the grid itself this
  beats using the home cell: monthly error 17.7% against 25.6%.

Rain at a well is not stored twice. It is the weighted sum of its stencil
cells, which `rain_panel.well_rain()` builds on demand, so `fact_cell_month` is
the single source and nothing can drift out of step with it.

### Levels are quarterly; the panel is monthly

CGWB reads these wells in January, May, August and November. There is no
monthly well series in this source and none is invented: `fact_well_month`
carries rain in all 300 months and a level in four of twelve, and the other
eight stay empty. Filling them would manufacture the signal the project
measures.

### Four seasons, cut at the readings

Rain is daily and levels are quarterly, so the two share a clock only once rain
is summed between readings. The readings carry a month and no day, so each is
placed on the 15th and a season runs from the day after one reading to the
next:

| season | from | to | days | closed by |
|---|---|---|---|---|
| Nov-Jan | Nov 16 | Jan 15 | 61 | January reading |
| Jan-May | Jan 16 | May 15 | 120/121 | May reading |
| May-Aug | May 16 | Aug 15 | 92 | August reading |
| Aug-Nov | Aug 16 | Nov 15 | 92 | November reading |

Not calendar quarters: a Jul-Sep quarter would straddle the August reading.
The cut falls mid-month, so `fact_cell_season` is summed from the daily grid,
not from `fact_cell_month`; it matches the raw NetCDF to 0.0001 mm.

Within a season the well is one reading and rain is the variable. A level is a
stock and rain a flow, so rain pairs with the *change* in level over the same
span, and specific yield puts both in millimetres of water:

    delta_h_m         = depth now - depth at the previous valid reading   (+ = fell)
    storage_change_mm = -delta_h_m x sy x 1000                            (+ = gained)
    recharge_ratio    = storage_change_mm / rain over the same span

A missing reading is bridged, never filled: the change runs from the last valid
reading up to a year back, `span_seasons` says how far, and the rain is summed
over the whole span. 88% of changes come from the season just before.

### Location from coordinates, not from names

CGWB district names mix vintages and spellings, so every table now takes its
district from the post-2020 outlines (`data_cleaning/reference/districts.geojson`, 724
districts). `ingest.py` places each well in the outline its coordinates fall in
and writes that as `district` and `state`; the file's own labels stay as
`cgwb_district` and `cgwb_state`. `grid.py` splits each rain cell across the
outlines its 0.25 degree box overlaps, by area (`bridge_cell_district`), and
every district table weights rain by that area. That gives rain to 720
districts (the four missed are islands off the land mask) against 702 when
cells were matched by centre point, and it removes the spelling duplicates
without an alias table.

The CGWB name agrees with the outline for 82.5% of wells; nearly all of the rest
are renames (Trichur, Thrissur) and splits (Khammam, Bhadradri Kothagudem),
which is what geometry is for. The state disagrees for 489 wells, mostly filed under
Andhra Pradesh while standing in Maharashtra, Rajasthan or Gujarat. The water
levels settle which is right: those wells correlate with the wells around their
coordinates at a median 0.39, against 0.17 with the wells of their labelled
district, and the coordinates fit better for 69% of the 262 with enough
readings. The label is what is wrong, so they stay where their coordinates put
them; `state_agrees` records the disagreement.

## What the tables say

From `scripts/core_report.py`:

* **Rain moves the level, and it is visible at cell level.** Monsoon rain
  against the November anomaly, per cell across years: median correlation
  **−0.355**, negative in **86%** of 2,786 cells. Negative is the physical
  expectation, since the anomaly is a depth.
* **The water year has the right shape.** Median depth 7.52 m in May, 4.15 m in
  August after 239 mm of monsoon rain in the month, 4.50 m in November, 5.57 m
  in January.
* **Wells in one cell disagree by about as much as the signal.** Median spread
  within a cell is 0.90 m for two wells and 1.29 m for six to ten, against a
  median absolute anomaly of 0.48–0.95 m. 31% of cell-months rest on a single
  well. Any cell-level claim has to carry `n_wells_read` and `anomaly_spread_m`
  beside it.
* **Distance weighting barely matters at this density.** Inverse-distance and
  plain mean differ by a median of 0.038 m, because most cells hold two or three
  wells. All three summaries are written; none is privileged.
* **The rebuild changes the decline.** November trend per well: the published
  2,759 give a median −0.0181 m/yr with 38% of wells deepening; the 6,050 wells
  the rebuild adds give −0.0103 m/yr with **46%** deepening and a mean of
  +0.0217 m/yr against the published set's −0.0345.
* **Bad data is concentrated, not spread.** Telangana carries 16.7% suspect
  zeros and 17.9% readings below the drilled bottom, against 0.2% and 0.8% for
  dug wells nationally. Bore wells and piezometers hold nearly all of it.
* **Some campaigns barely happened.** May 2020 and May 2021 reached 1% of wells,
  August 2012 3%, January 2016 12%, November 2018 45%. Nothing is filled in.

* **The seasons have the right physics.** Median change per season over all
  wells: the water rises 1.94 m through May-Aug on 482 mm of rain, about 11.5% of
  which reaches storage (CGWB's own infiltration factors run 8-15%). It falls
  1.42 m through Jan-May on 39 mm, and barely moves through Aug-Nov, when late
  rain and drainage cancel. `recharge_ratio` is a recharge figure only in the
  two monsoon seasons; outside them it is a net loss per millimetre of rain.

## Known gaps

* **Specific yield and aquifer type are finished by a notebook.** The ingest
  gives unpublished wells a state-and-aquifer median; `data_cleaning/clean_gaps.ipynb`
  then reads every well from the hydrogeological map the published values came
  from (it reproduces all of them) and fills blank dug wells as Unconfined.
  Run it after any re-run of `ingest`. Details in `data_cleaning/README.md`.
* **Specific yield is a class, not a measurement.** Five values from a
  national map; storage in millimetres inherits that, `delta_h_m` does not.
* **3,449 land cells cross a district line**, and 91 sit outside every outline
  (beyond the drawn border). District rollups use the area weights.
* **2023 and 2024 readings exist in `well_obs` but have no rain.** The IMD
  grids in the repo stop at 2022, so the panel stops there too; extending it is
  out of scope for now, by decision.
* **Surface water is not in yet.** Rivers, reservoirs, canal command areas and
  land cover were designed for as separate fact tables keyed on `cell_id`, so
  they attach without reshaping anything. None is built.
