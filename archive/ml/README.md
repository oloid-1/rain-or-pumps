# ml — rainfall, groundwater, attribution

Two datasets, two models, one link.

| | Rainfall (IMD) | Groundwater (CGWB) |
|---|---|---|
| Data | 0.25° daily grid, 1998–2022 | 2,759 wells, four readings a year, 2000–2022 |
| Module | `rainfall.py`, `rain_model.py` | `groundwater.py` |
| Model | statistics per location: normals, variability, trend, scenarios | regression: Model 1 ridge, Model 2 LightGBM |
| Why | rain cannot be forecast from its own past (Week 1) | rain explains part of the level; the rest is the signal we want |
| Link | rain read at each well's exact coordinates | those rain values are the model's inputs |

The gap between the level rain explains and the level actually measured, tracked over years, is the extraction-pressure proxy.

## Run it end to end

From `ml/`, in order. Steps 1–2 are data preparation; the rest can be re-run freely.

```bash
python -m bits_ml.rainfall       # rain at each well from the IMD grid  -> well_monthly_rain_1998_2022.parquet
python -m bits_ml.groundwater    # CGWB readings as a long series       -> well_readings_2000_2022.parquet
python -m bits_ml.dataset        # the modelling table + dictionary     -> well_campaign_table.parquet
python -m bits_ml.splits         # prints the split sizes and the leakage checks
python -m bits_ml.evaluate --models ridge,lgbm   # scores per campaign  -> model_results.csv
python -m bits_ml.train --model lgbm --default   # saves models/lgbm.joblib
python -m bits_ml.attribution --model lgbm       # district ranking     -> district_ranking.csv, well_drift.csv
python -m bits_ml.whatif --delta 0.2             # what 20% more monsoon would do
uvicorn bits_ml.api:app --port 8000              # /district_rank, /district/{name}, /rain/{name}, /whatif
pytest -q                                        # 50 tests
```

## Tuning on Colab or Kaggle

The table is built here; the search runs there. Note that LightGBM on this data is CPU work — it trains in seconds locally — so the gain is the wider search and a free machine, not a GPU.

1. **Upload** `data/processed/colab_bundle.zip` (25 MB: the table and its dictionary), rebuilt by zipping those two files.
2. **Run** `notebooks/02_train_on_colab.ipynb` top to bottom. It repeats the same features, splits and constraints as these modules, then searches 24 settings with early stopping on the validation years. The test years stay closed until the final cell.
3. **Bring back** three files: `lgbm.joblib` → `ml/models/lgbm_tuned.joblib`, and `model_results_colab.csv` plus `colab_predictions.csv` → `data/processed/`.
4. **Check it survived the trip**, which is not a formality:

   ```bash
   python -m scripts.check_colab_bundle --bundle models/lgbm_tuned.joblib
   ```

   Measured on this data: reordered categories are harmless (LightGBM remaps by label) and plain strings fail loudly, but a renamed label, or one missing while its rows still use it, moves predictions by about a metre in silence. Nothing can repair that afterwards, so it has to be caught: the bundle records its labels and library versions, and this replays the notebook's own predictions, failing if they differ.
5. **Regenerate what depends on it** — attribution and what-if numbers come from the model, so they are re-run, never edited:

   ```bash
   python -m bits_ml.attribution --model lgbm
   python -m bits_ml.whatif --grid --delta 0.2
   ```

## What the model is asked to predict

The target is **how far the water stands from that well's own normal for that campaign**, in metres, positive meaning deeper. A well's fixed depth says more about where it was drilled than about rain or pumping, so it is removed; what is left is what varies.

Inputs are rain history (3 to 36 months, both as millimetres and as a percentage of that well's own normal), wet days, the wettest day, the last monsoon, plus fixed well properties (specific yield, depth, well type, aquifer) and the well's normal annual rain so dry and wet regions can respond differently.

**Never inputs:** year, district, state, coordinates, or any past water level. Each of them carries the pumping history the project is trying to measure; a model given them would explain the decline with itself and leave nothing to attribute.

## Rules the code enforces

- **Normals come from training years only.** `dataset.build(train_years=...)` recomputes both the level normals and the rain normals, so every fold rebuilds the table its own way.
- **Test years stay closed.** Train 2001–2014, tune on 2015–2017, and 2018–2022 is scored once.
- **Two kinds of hold-out.** Later years, and whole unseen districts (`splits.group_folds`).
- **More rain never means deeper water.** LightGBM runs with monotone constraints, which is also what makes the break-even search in `whatif.py` valid.
- **Attribution never scores itself.** Each block of years is predicted by a model that never saw it (`attribution.cross_fitted`).
- **Missing readings stay missing.** May 2020 and May 2021 were barely surveyed (1% of wells); nothing is filled in.

## What the numbers look like so far

**Three targets, measured side by side** (`python -m scripts.experiments`, results in `data/processed/experiment_results.csv`). Skill is against "this well sits at its own normal", on test years 2018–2022 the model never saw:

| Target | What it asks | Ridge | LightGBM |
|---|---|---|---|
| `level` | how far from normal the water stands | 0.086 | **0.191** |
| `rise` | the monsoon's recharge, May→Nov | 0.110 | **0.120** |
| `annual` | the year's net change, Nov→Nov | **0.113** | 0.053 |

`annual` is the business question and ridge is the stabler model on it (validation 0.123 → test 0.113, while LightGBM swings 0.120 → 0.053).

**The skill is real.** Controls that must fail, do (`python -m scripts.controls`). On all three targets, skill collapses when rain is shuffled across years (−0.001 to +0.004) or between wells (−0.007 to −0.010), and when the model is handed rain from four years later (−0.017 to +0.015). The shift has to clear the 36-month window: at one year it leaks, scoring +0.086 against a real +0.123 and proving nothing.

**Tuning is not the lever.** A 24-setting search moved validation skill between +0.071 and +0.078 and chose the settings already in `models.py`, differing only in tree count. Test skill +0.161 against +0.159, predictions correlating 0.999, 1.2 cm apart at the median well. `models/lgbm_tuned.joblib` is kept as evidence. The features are the ceiling, not the settings.

**The extraction proxy does not reproduce, and that is the finding.** Splitting the record into 2002–2012 and 2013–2022 and measuring each district independently:

| Measure | ≥3 wells | ≥5 wells | ≥8 wells |
|---|---|---|---|
| Observed November decline | **+0.24** | **+0.29** | **+0.27** |
| Rain-adjusted gap | −0.31 | −0.37 | −0.35 |

Shuffled null is ±0.11 to ±0.18. The gap disagrees with itself and gets *worse* where coverage is better, so it is not thin-district noise. The cause is arithmetic plus climate: a mean of year-over-year changes telescopes to (first level − last) ÷ years, and the rain-expected change is the most reversing quantity of all (−0.53) because rain anomalies mean-revert between decades.

So districts are **ranked on observed decline**, which reproduces, and the rain model **explains** that decline rather than ordering it. Three independent checks agree that the residual cannot carry a ranking: dry-season drawdown (rho −0.09 to 0.00 across every design tried), split-half reproducibility, and the absence of known hotspots from the earlier residual-based top.

**The decline cannot be split into a weather part and a pumping part either.** The obvious policy move is to send districts whose fall follows the weather to recharge structures and the rest to demand management. That split is not supported here: the rain-expected trend in the level and the observed trend correlate **−0.03**, so subtracting one from the other leaves a remainder with a wider spread (0.164 m/yr) than the fall it decomposes (0.127 m/yr), larger than the whole in 38% of wells. `decisions.py` therefore bands districts by **how long their wells have** — the fall and the room left below the water, both measured directly — and carries the rain figures as labelled context. No district is classified by cause.

## District groups (unsupervised)

`python -m scripts.clusters` groups districts by how their groundwater behaves — normal rain and its variability, depth to water, the May→November swing, how closely that swing follows the rain, and the rate of decline — rather than by where they are. K-means, with the number of groups chosen by silhouette and both the separation and the stability reported alongside.

Read them as tendencies, not types. Silhouette reaches 0.237 against 0.184 for data with the same spread and no joint structure, and dropping a fifth of the districts reproduces the grouping at about 0.71 (worst case 0.44); separated synthetic districts score above 0.9 on the same test. The groups describe a district's behaviour; they should not be used to assign it to a category or to drive a decision on their own. Outputs: `district_clusters.csv`, `district_cluster_profiles.csv`.

## Caveats to carry into the report

- Quality control kept 2,759 of 31,169 wells, and the survivors are the steadier ones, so decline is likely understated.
- 94% are dug wells in unconfined aquifers: the findings are about shallow groundwater.
- The gap is measured relative to how these wells behave in a normal-rain year; it is not absolute pumping in cubic metres.
- Canal irrigation, cropping changes and land use also push the level away from rain. The gap is "not rain", not "pumping" alone.

## Files

| File | Job |
|---|---|
| `rainfall.py` | rain at each well from the IMD grid: masked bilinear, rounding boxes, wet days counted at nodes |
| `rain_model.py` | normals, history features, trends, scenario years |
| `groundwater.py` | readings as a long series, normals, anomalies, seasonal moves, label-outlier flag |
| `dataset.py` | the one modelling table, the feature list, and what may never be a feature |
| `splits.py` | year splits, unseen-district folds, cross-fitting blocks |
| `models.py` | baselines, per-campaign ridge, LightGBM with constraints, MLP benchmark |
| `evaluate.py` | scores per campaign against the baselines |
| `train.py` | fits and saves a model bundle |
| `attribution.py` | cross-fitted residuals, per-well drift, district ranking, dry-season check |
| `whatif.py` | scenario rain in, metres of water out, break-even monsoon |
| `api.py` | the endpoints the dashboard and agent call |
| `scripts/` | evidence runs: rainfall report, interpolation check, groundwater response check |
