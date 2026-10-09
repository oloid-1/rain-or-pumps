# simulator/

The model the application uses (a BiLSTM trained with a rain-response penalty),
the FastAPI service around it, and the map UI.

```bash
make api                                       # serve the committed data at :8000
# rebuild the data (make app-data does all three):
python models/build_sequences.py               # training table + 6-channel data (~35 s)
python simulator/ui/build_ui_data.py           # everything the UI reads (~2 min)
python simulator/forecast/build_forecast.py    # forecast table (~18 min)
uvicorn main:app --port 8000                     # UI at http://localhost:8000, API docs at /docs
```

Map layers and the town list are tracked; `make geo places` rebuilds them from the
downloads below.

## Why a separate model

The BiLSTM in `models/` is built to *fit* `delta_h_m`. A simulator has to do one
more thing: answer "what if the rain had been different?" with a response that at
least points the right way. `BILSTM_RESULTS.md` showed the fitted BiLSTM does not.
Its response to a 20% rain change was about 1 cm, and the sign reversed when one
channel was removed. Two changes fix that.

**1. Rain enters through the sequence only.** The BiLSTM also takes 23
rain-derived tabular features: windows, lags and anomalies. A scenario would have
to rebuild all of them consistently. The simulator keeps only static tabular
inputs:
- numeric: `well_depth_m`, `sy`, `days_since_prev`, and the three
  training-years rainfall normals;
- categorical: `season`, `aquifer`, `well_type`, `transition`.

Scaling the 104-week sequence is then the whole scenario. The same leakage rules
hold: no past water level, year, location or identifier.

**2. A rain-response penalty in training.** Each batch is re-run with its rain
scaled up and down by a random 10–40%. Any row where more rain predicts a larger
fall is penalised:

```
relu(pred(rain x up) - pred(rain)) + relu(pred(rain) - pred(rain x down))
```

`delta_h_m` is positive when the level falls, so the target ordering is
pred(up) ≤ pred(base) ≤ pred(down). The scaling is done inside the network's
scaled input space, mirroring `train_bilstm.rain_scaled`. The anomaly channel
moves with the rain; the calendar, interval and wet-day channels do not. The
torch and numpy versions agree to 1e-5.

The penalty is soft, so the share of rows that still move the wrong way is
measured and reported rather than assumed to be zero.

## Results

Kaggle T4, one seed, the same rows and the same split as every other model.
"Wrong-way" is the share of validation rows where +20% rain predicts a *larger*
fall.

| simulator | valid MAE | valid R² | test R² | +20% rain, mean change | wrong-way at +20% |
|---|---|---|---|---|---|
| **6 channels, penalty on** | **1.521** | **0.426** | **0.453** | **−11.1 cm** | **2.2%** |
| 6 channels, penalty off | 1.522 | 0.422 | 0.455 | −2.7 cm | 54% |
| main data (rain only), penalty on | 1.531 | 0.419 | 0.446 | −9.2 cm | 3.1% |
| main data (rain only), penalty off | 1.536 | 0.414 | 0.449 | −1.4 cm | 55% |
| *BiLSTM, full tabular (BILSTM_RESULTS.md)* | *1.521* | *0.419* | *0.441* | *−0.9 cm* | — |

The penalty costs nothing in fit. The 6-channel simulator matches or beats every
model in `BILSTM_RESULTS.md` on validation and test, and it does so without
the 23 rain-derived tabular features.

Without the penalty, more than half the rows respond to extra rain in the wrong
direction. The fit metrics cannot show this, because they never vary the rain.
With the penalty, the response is about four times stronger and points the right
way for 97–98% of rows.

The response grows with the size of the change and stays the right way round:

| rain change | ×0.5 | ×0.8 | ×1.2 | ×1.5 |
|---|---|---|---|---|
| mean change in predicted fall | +36.5 cm | +13.4 cm | −11.1 cm | −24.2 cm |
| wrong-way rows | 2.0% | 2.1% | 2.2% | 1.9% |

**Unseen districts (5 folds):** MAE 1.437, R² **0.488**. The folds are
0.433, 0.529, 0.503, 0.496 and 0.478. The transformer scored 0.486 and the
BiLSTM 0.481, so the simulator is the best of the three on districts it has
never seen.

**Main data and 6-channel data both work.** The penalty helps on either, and the
6 channels add about 0.007 R² on validation and test.

### What to say about it

- **Supported:** a sequence model given only rain history and fixed well facts,
  trained with a rain-response penalty, fits as well as gradient boosting on 33
  engineered features. Its rain response points the right way for 98% of wells.
- **Not supported:** that it predicts how much any single well will recover. The
  response is a few tens of centimetres across ±50% rain, the p10–p90 spread is
  wide, and the penalty only fixes the sign, not the size.
- One seed throughout. The R² differences between penalty on and off (0.004) are
  noise. The wrong-way share (2% against 55%) is not.

## The UI (`simulator/ui/`)

A map page (MapLibre GL) served by the FastAPI service, which runs the model. Four
modes:

| mode | the question it answers |
|---|---|
| **Past rain** (Replay) | What rain fell, and what did the wells do? IMD daily rainfall, every day 1998–2022, over a 25-year timeline. Wells are coloured by the change at the last quarterly reading. Zoomed in, the rain falls as particles scaled to the recorded amount |
| **What if it rains** | What would changed rain on *these days* do to the groundwater here? Any day or range of days from 2000 to 2045, by a percentage or extra mm per day, for any place. The model re-runs every reading that rain reaches (up to 104 weeks later) and reports the effect on the water table at each, with the season and the rain against the normal for those days. Past dates use the rain that fell around them; future dates give a median and range over 23 replayed rain histories |
| **Future** (Forecast) | Where is the water table heading? From each well's last reading in 2022 to a year you choose (2023–2045), with a rain shift of ±30%, as a median and 10–90% band over 23 replayed rain histories. Two lines: rain only, and rain + the 2015–22 trend (whatever rain did not explain carrying on). Years past the tested 5 are shaded |
| **Pumping hotspots** (Pressure) | Where did the water fall further than rain explains? Per district, observed minus rain-expected change, held-out years 2015–2022. Ranked, with a per-district timeline |

One search box in the header, on every tab, finds a state, a district, or any of
6,533 towns of 5,000+ people by its current or older name (Bangalore, Bombay,
Gurgaon, Ootacamund). A town leads to the district it lies in, because that is how
the wells are organised: Past rain flies there, What if and Future use it as the
place, Pumping hotspots opens the district. 72 towns of over 200,000 people
(Hyderabad, Jaipur, Kolkata among them) are in districts with no monitored well;
for those the place becomes the smallest circle around the town, 25 to 300 km,
that holds at least 3 wells, and the page says so. Press / to jump to the search.
You can also click a well, or click the map for the wells around that spot.

Design: a light, print-atlas look; Source Serif 4, IBM Plex Sans and Plex Mono;
colour reserved for meaning (blue rise, terracotta fall).
- Keyboard: Space to play, ← → to step a day, Shift for a month; arrow keys and
  Enter in the place search.
- Reduced motion is respected: no particles and no animated camera.
- At phone width the panel moves to the bottom.

Performance:
- The rain texture uploads once per day shown, wells are one `setData` per
  repaint, and small rivers appear only from regional zoom.
- Forecast reads a precomputed table and answers in under half a second.
- What if runs the model live: about 1–2 s for a past edit over a state, about
  5–12 s for a future edit (23 rain histories), most of it the model.

## Forecast and What if (`simulator/forecast/`)

**Why one model.** Of the three, only this BiLSTM simulator responds to rain in a
way a simulation can use: 98% of readings move the right way under changed rain.
LightGBM fits held-out years better (R² 0.484 against 0.453) but needs its 23 rain
features rebuilt for every edit and has no guarantee on direction; the transformer
and the plain BiLSTM respond to rain by about 1 cm, and not reliably in sign.

**How a forecast is made.** Readings fall on 15 Jan, May, Aug and Nov, so the
future is a chain of regular quarterly steps. Each step's change is the model's
prediction from the 104 weeks of rain behind it; the level after a step is the
level before plus that change. Future rain is past rain replayed: ensemble member
*m* rains year 2023 + *k* like year 2000 + (*m* + *k*) mod 23, so dry and wet years
keep their real runs.

**Backtest** (`index.json`, `backtest`): chained from an observed reading with the
rain that actually fell, mean absolute error of depth in metres:

| years ahead | rain-only forecast | no change | the well's own trend |
|---|---|---|---|
| 1 | **1.47** | 1.78 | 1.87 |
| 3 | **1.89** | 2.11 | 2.68 |
| 5 | **2.33** | 2.38 | 3.35 |
| 6–8 | 2.45–2.94 | **1.79–2.07** | 3.2–4.2 |

The forecast beats "no change" up to 5 years ahead and loses after; the UI shades
those years. Carrying the 2015–18 unexplained change forward made 2019–22 worse
(2.91 m against 2.17 m at 4 years): those years were dry and the next ones wet, so
the trend line is labelled an assumption.

**It is not a weather forecast.** Nothing here predicts what rain will fall. It
says what the groundwater does under rain like the record, or under rain you set.

| file | what |
|---|---|
| `build_forecast.py` | builds model inputs for every well, past year and season; predicts each step at rain shifts −30% to +30%; the trend rates; the backtest; checks the rebuilt inputs against the training rows (median 2.5 mm). About 18 minutes on a laptop CPU: `make forecast` |
| `engine.py` | the live engine behind What if: edits daily rain anywhere in time and re-runs the readings it reaches. Matches the table to under 1 mm on past readings |

### No flood or reservoir simulation

The model predicts quarterly groundwater change. It knows nothing about river
flow, reservoir storage or dam releases, so the UI shows dams as context only.

## The API (`simulator/api/`)

A FastAPI service: it serves the UI and the model behind it (the week 7
deliverable, an API endpoint for the model).

```bash
make api            # uvicorn on http://localhost:8000; interactive docs at /docs
make test-api       # 22 tests
```

| endpoint | what it returns |
|---|---|
| `GET /api/health` | model, channels, validation metrics, rain response, scenario years |
| `GET /api/forecast?to_year=&rain_pct=&district=&state=&well=&lat=&lon=&radius_km=` | depth to water from Nov 2022 to `to_year`: median and 10–90% for rain only and rain + 2015–22 trend, observed history, per-well change, the backtest |
| `POST /api/simulate` | body `{start, end, rain_pct, add_mm, district / state / well / lat+lon+radius_km}`. Runs the model live on every reading the changed rain reaches; returns the effect on the water table at each, the peak, per-well effects and the rain context (season, normal, recorded, added) |
| `GET /api/wells?state=&district=` | every monitored well: position, district, type, aquifer, depth, specific yield |
| `GET /api/wells/{id}` | one well with its level history; each reading has observed and rain-expected change, and a held-out flag |
| `POST /api/scenario` | body `{year, rain_pct, state, details}`. Per-well predicted change for a November reading (2015–2022) under scaled rain |
| `GET /api/pressure?from_year=&to_year=&top=` | districts ranked by observed minus rain-expected change |
| `GET /api/districts/{name}?state=` | one district's yearly observed, expected and unexplained change |
| `GET /api/rain/{date}?cells=` | IMD rain for one day: mean, maximum, heavy cells; optionally every wet cell |
| `/` | the UI |

Examples:

```bash
# 40 mm of rain today in Bangalore: the water table about 25 cm higher at the 15 Nov reading
curl -s -X POST localhost:8000/api/simulate -H "Content-Type: application/json" \
     -d '{"district": "Bengaluru", "start": "2026-10-09", "add_mm": 40}'
# +15% rain through the 2019 monsoon in Karnataka
curl -s -X POST localhost:8000/api/simulate -H "Content-Type: application/json" \
     -d '{"state": "Karnataka", "start": "2019-06-01", "end": "2019-09-30", "rain_pct": 15}'
# Punjab to 2030
curl -s "localhost:8000/api/forecast?state=Punjab&to_year=2030"
curl -s -X POST localhost:8000/api/scenario -H "Content-Type: application/json"      -d '{"year": 2022, "rain_pct": -40, "details": false}'
# summary.level_vs_recorded_rain_m = -0.258: the water level 25.8 cm lower than with the rain that fell
```

Design:
- **One model file.** The API runs `simulator/ui/data/sim.onnx` with onnxruntime.
  A test checks it against the predictions computed in torch when the data was
  built, and the forecast build checks its rebuilt inputs against the training rows.
- **What runs live.** What if (`/api/simulate`) and Scenario run the model on every
  request. Forecast reads a table of the model's per-step predictions, because a
  live all-India forecast is about 2 million predictions (an hour on a CPU).
  `/api/simulate` refuses requests over about 30,000 model rows, and runs fewer
  rain histories for big places on future dates.
- **Validation.** Requests are checked by Pydantic and give 422 with a reason:
  - `rain_pct` from −50 to +50;
  - `year` from 2015 to 2022;
  - `state` one of the states with monitored wells.
- **Pressure uses held-out years by default (2015–2022).** The model was fitted on
  2000–2014, so residuals there are biased towards zero (`models/MODEL_REVIEW.md`,
  section 3). Earlier years need `include_training_years=true`, and the response
  is flagged `in_sample: true`. The UI's Pressure slider is limited to 2015–2022
  for the same reason.

Tests (`simulator/api/test_api.py`):
- forecast bands ordered, starting at the last reading, untested years flagged,
  less rain meaning deeper water;
- simulate: +15% monsoon in the past raises the water; 40 mm today in Bangalore
  is a future ensemble with the right season; bad requests rejected;
- ONNX against the torch predictions;
- recorded rain is the baseline;
- more rain raises the water level and the waterlogging count;
- the state filter;
- rejected inputs;
- the Pressure year rule;
- wells, districts, rain and the UI route.

## Map data (`simulator/geo/build_geo.py`)

| layer | source | licence | in the UI |
|---|---|---|---|
| Districts and India outline | `data/reference/districts.geojson`, 724 post-2020 districts | — | outlines; India dissolved from them, so the boundary follows the Survey of India depiction (Leh to 37.08°N, Aksai Chin, PoK) |
| Rivers | HydroSHEDS HydroRIVERS v1.0, Asia | free, with attribution | 31,062 reaches inside India with mean flow ≥ 15 m³/s; ≥ 120 m³/s at national zoom |
| Dams | GeoDAR v1.1 (zenodo 6163413) | CC BY 4.0 | 1,299 dams in India, 334 with storage volume. That is about a fifth of the national register: context, not inventory |
| Relief | AWS Terrain Tiles (terrarium) | open | hillshade, read straight from the CDN |
| Rain | IMD 0.25° daily grid, from `data/derived/rain_cube.npz` | open | one uint8 file per year, log-quantised (mm = e^(q/33) − 1) |
| Towns | GeoNames `cities1000.zip`, built by `geo/build_places.py` | CC BY 4.0 | 6,533 Indian towns of 5,000+ people, each placed in its district outline, with Latin-script older names; `places.json` (530 kB, tracked) |

The raw downloads go in `data/geo/raw/` (gitignored):

```bash
curl -L -o data/geo/raw/HydroRIVERS_v10_as_shp.zip https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip
curl -L -o data/geo/raw/GeoDAR_v10_v11.zip "https://zenodo.org/api/records/6163413/files/GeoDAR_v10_v11.zip/content"
curl -L --create-dirs -o data/geo/raw/geonames/cities1000.zip https://download.geonames.org/export/dump/cities1000.zip
cd data/geo/raw && unzip HydroRIVERS_v10_as_shp.zip && unzip GeoDAR_v10_v11.zip
```

The four map layers and the town list are tracked in `simulator/ui/data/`. The
model files, rain files, wells, scenario inputs and pressure table are written there
by `build_ui_data.py` (about 80 MB on the full data), and the forecast table by
`simulator/forecast/build_forecast.py`.

## Training and reproducing

```bash
python simulator/train_sim.py --out simulator/out/sim_6ch                       # CPU: slow
python kaggle/push.py --script run_sim.py                                        # Kaggle T4
python kaggle/push.py --fetch --script run_sim.py
python simulator/export_onnx.py --run data/kaggle/rain-or-pumps-bilstm-sim/sim_6ch  # checks ONNX against torch
```

The penalty passes run with cuDNN disabled, because cuDNN refuses an RNN
backward pass in eval mode. That makes a penalised run about 8× slower than an
unpenalised one on the T4. Most of the 3.5 hours is the 5-fold run.

## Files

| file | what |
|---|---|
| `train_sim.py` | static tabular encoder, `RainScaler`, penalised training loop, rain-response check, ONNX export |
| `export_onnx.py` | exports a run trained elsewhere and checks ONNX against torch |
| `geo/build_geo.py` | districts, India outline, rivers and dams, clipped and thinned |
| `ui/build_ui_data.py` | rain files, wells, campaigns, scenario inputs, base predictions, pressure table |
| `forecast/build_forecast.py`, `forecast/engine.py` | the forecast table, the backtest and the live What if engine |
| `ui/index.html`, `ui/style.css`, `ui/app.js` | the page |
| `api/app.py`, `api/test_api.py` | the FastAPI service and its tests |
| `artifacts/` | the reported 6-channel simulator (`sim.pt`, `sim.onnx`, `sim_meta.json`) and every run's results |
| `../kaggle/run_sim.py` | the Kaggle kernel for all of the above |
