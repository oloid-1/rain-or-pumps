# simulator/

A rain-scenario simulator built on top of the BiLSTM, and the map UI that runs it.
This is new work on the `bilstm` branch. It is kept separate from the BiLSTM
comparison: `models/` and `models/BILSTM_RESULTS.md` are unchanged by it.

```bash
python bilstm-data/build_bilstm_data.py        # the 6-channel data (~25 s)
python simulator/geo/build_geo.py              # map layers (needs data/geo/raw/, below)
python simulator/ui/build_ui_data.py           # everything the UI reads (~2 min)
python -m http.server 8765 -d simulator/ui     # then open http://localhost:8765
```

Or with make: `make bilstm-data geo ui-data ui`.

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

A static page with no server code: MapLibre GL for the map, onnxruntime-web
running the model in a Web Worker. Three modes:

| mode | what it shows |
|---|---|
| **Replay** | IMD daily rainfall, every day 1998–2022, over a 25-year timeline of India's daily rain. Wells are coloured by the change at the last quarterly reading. Zoomed in, the rain falls as particles scaled to the recorded amount. Jump buttons go to Mumbai 2005, Uttarakhand 2013, Chennai 2015, Kerala 2018 and the 2009 drought |
| **Scenario** | Pick a November reading from 2015–2022 (the years the model never trained on) and change the two years of rain before it by −50% to +50%. The model re-predicts every well in the browser. It reports the mean change against recorded rain and how many wells reach the waterlogged band (water table within 2 m of the surface) |
| **Pressure** | Per district, mean observed change minus the simulator's rain-expected change. Warm colours are where the water fell further than rain explains: the extraction proxy. Ranked, with a per-district timeline |

Design: a light, print-atlas look; Source Serif 4, IBM Plex Sans and Plex Mono;
colour reserved for meaning (blue rise, terracotta fall).
- Keyboard: Space to play, ← → to step a day, Shift for a month.
- Reduced motion is respected: no particles and no animated camera.
- At phone width the panel moves to the bottom.

Performance:
- The base predictions at recorded rain are precomputed, so opening Scenario
  runs no model.
- Inference runs in `model-worker.js`, so the page never blocks; the longest pause
  measured was 62 ms.
- The rain texture uploads once per day shown, wells are one `setData` per
  repaint, and small rivers appear only from regional zoom.

### No flood or reservoir simulation

The model predicts quarterly groundwater change. It knows nothing about river
flow, reservoir storage or dam releases, so the UI shows dams as context only. The
"water table within 2 m of the surface" flag is the model-backed version of
"water comes out": CGWB's own waterlogged category.

## Map data (`simulator/geo/build_geo.py`)

| layer | source | licence | in the UI |
|---|---|---|---|
| Districts and India outline | `data_cleaning/reference/districts.geojson`, 724 post-2020 districts | — | outlines; India dissolved from them, so the boundary follows the Survey of India depiction (Leh to 37.08°N, Aksai Chin, PoK) |
| Rivers | HydroSHEDS HydroRIVERS v1.0, Asia | free, with attribution | 31,062 reaches inside India with mean flow ≥ 15 m³/s; ≥ 120 m³/s at national zoom |
| Dams | GeoDAR v1.1 (zenodo 6163413) | CC BY 4.0 | 1,299 dams in India, 334 with storage volume. That is about a fifth of the national register: context, not inventory |
| Relief | AWS Terrain Tiles (terrarium) | open | hillshade, read straight from the CDN |
| Rain | IMD 0.25° daily grid, from `data/derived/rain_cube.npz` | open | one uint8 file per year, log-quantised (mm = e^(q/33) − 1) |

The raw downloads go in `data/geo/raw/` (gitignored):

```bash
curl -L -o data/geo/raw/HydroRIVERS_v10_as_shp.zip https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip
curl -L -o data/geo/raw/GeoDAR_v10_v11.zip "https://zenodo.org/api/records/6163413/files/GeoDAR_v10_v11.zip/content"
cd data/geo/raw && unzip HydroRIVERS_v10_as_shp.zip && unzip GeoDAR_v10_v11.zip
```

The four map layers and the model (`sim.onnx`, `sim_meta.json`) are tracked in
`simulator/ui/data/`. The rain files, wells, scenario inputs and pressure table
(about 80 MB) are rebuilt by `build_ui_data.py`.

## Training and reproducing

```bash
python simulator/train_sim.py --data bilstm-data/out --out simulator/out/sim_6ch       # CPU: slow
python kaggle/push.py --main-data --script run_sim.py                                  # Kaggle T4: all runs, ~3.5 h
python kaggle/push.py --fetch --script run_sim.py
python simulator/export_onnx.py --run bilstm-data/out/kaggle_sim/sim_6ch                # checks ONNX against torch
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
| `ui/index.html`, `ui/style.css`, `ui/app.js`, `ui/model-worker.js` | the page |
| `artifacts/` | the reported 6-channel simulator (`sim.pt`, `sim.onnx`, `sim_meta.json`) and every run's results |
| `../kaggle/run_sim.py` | the Kaggle kernel for all of the above |
