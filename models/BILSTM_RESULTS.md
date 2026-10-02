# BiLSTM results

These results use the same 216,455 rows as the transformer: 2,759 wells, 33
tabular features and the same blocked year split. The sequence input is the
6-channel copy in `bilstm-data/` (see `bilstm-data/README.md`). The target is
`delta_h_m`, positive when the water level fell. No past water level appears
anywhere in the inputs.

All runs used a Kaggle T4 GPU, about 3 minutes per model. Each figure comes from
one run with seed 42.

## Headline

Validation 2015-2017, every model on the same rows:

| model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| zero (predict no change) | 2.163 | 3.227 | 0.000 |
| season mean | 1.813 | 2.756 | 0.271 |
| ridge on tabular | 1.753 | 2.870 | 0.209 |
| **LightGBM on tabular** | **1.515** | **2.426** | **0.435** |
| **BiLSTM, 6 channels + tabular** | **1.521** | **2.460** | **0.419** |
| BiLSTM, rain only + tabular | 1.533 | 2.466 | 0.416 |
| BiLSTM, 5 channels (no anomaly) + tabular | 1.538 | 2.469 | 0.415 |
| BiLSTM, sequence only | 1.554 | 2.499 | 0.400 |
| BiLSTM, sequence only, no anomaly | 1.578 | 2.513 | 0.394 |
| BiLSTM on the main data (`data/training/`), rain only + tabular | 1.529 | 2.458 | 0.420 |
| BiLSTM on the main data, sequence only | 1.618 | 2.582 | 0.360 |
| *transformer + tabular (RESULTS.md)* | *1.531* | *2.461* | *0.418* |
| *transformer, sequence only (RESULTS.md)* | *1.641* | *2.618* | *0.342* |

LightGBM reads 0.435 here against 0.431 in `RESULTS.md`. Same code; the
LightGBM version differs between the two machines.

Held-out test 2018-2022, BiLSTM: **MAE 1.449, RMSE 2.298, R² 0.441**. The
transformer scored 1.457, 2.302 and 0.439. Train R² is 0.530, so the gap to
unseen years is about 0.1, and test scores sit above validation, the same
pattern as the transformer. 183,122 parameters; early stopping ended training
at epoch 19, and the best epoch was 14.

## Unseen districts: five district folds

Both models ran on the same folds for 20 epochs each:

| fold | BiLSTM MAE | BiLSTM R² | transformer MAE | transformer R² |
|---|---|---|---|---|
| 0 | 1.419 | 0.426 | 1.424 | 0.432 |
| 1 | 1.450 | 0.526 | 1.445 | 0.531 |
| 2 | 1.452 | 0.497 | 1.445 | 0.508 |
| 3 | 1.354 | 0.484 | 1.354 | 0.487 |
| 4 | 1.527 | 0.471 | 1.528 | 0.472 |
| **mean** | **1.441** | **0.481** | **1.439** | **0.486** |

The two models tie on every fold. Both score better on unseen districts
(R² about 0.48) than on unseen years (about 0.42). So the harder generalisation
problem is the shift in time, not the shift in place. That is good news for a
district-level attribution product. This is the first transformer
district-fold run; `RESULTS.md` listed it as pending.

## The same BiLSTM on the main data

The same model and settings, trained on `data/training/` directly: the
transformer's build, one rain channel, with no `bilstm-data` copy involved.

| | valid MAE | valid R² | test R² | district CV R² |
|---|---|---|---|---|
| BiLSTM, main data, rain only | 1.529 | 0.420 | 0.440 | 0.483 |
| BiLSTM, bilstm-data, rain only (`bilstm 1ch`) | 1.533 | 0.416 | — | — |
| BiLSTM, bilstm-data, 6 channels | 1.521 | 0.419 | 0.441 | 0.481 |

The rain-only model trained from the copy and the one trained from the main
build agree to 0.004 m MAE. So the copy reproduces the main data, and the
comparisons above hold for either source. The district folds (0.425 / 0.525 /
0.508 / 0.487 / 0.470) match the 6-channel model's fold by fold.

Main-data run in `models/artifacts/bilstm_main_results.json`,
`bilstm_main_results_cv.json` and `bilstm_main.pt`.

## What to claim

**On the full model: a three-way tie.** The BiLSTM, the transformer and
LightGBM are within 0.016 m MAE of each other. From one seed, none of these gaps
is a result. The claim that holds up is the transformer's claim, now made twice:

> Two different sequence architectures, given the raw weekly rainfall, both
> recover gradient boosting on hand-engineered rain windows.

**The architecture makes no difference to the full model.** The BiLSTM and the
transformer were given the same rain-only input. They differ by 0.002 m MAE on
validation and 0.002 m on the district folds.

**The five extra channels add little.** Rain only gives MAE 1.533; six channels
give 1.521. The direction is consistent, but the gain is too small to report as
a finding without repeat seeds.

## The result that does hold: sequence only

With no tabular features at all:

| | R² |
|---|---|
| transformer, sequence only | 0.342 |
| BiLSTM, sequence only, rain only (main data) | 0.360 |
| BiLSTM, sequence only, 5 channels (no anomaly) | 0.394 |
| BiLSTM, sequence only, 6 channels | 0.400 |
| BiLSTM, full model | 0.419 |

The BiLSTM's rain history alone recovers **95%** of its full model's explained
variance. For the transformer, the figure is 82%. A gap of 0.05 to 0.06 in R² is
too large to be seed noise.

The anomaly channel was the obvious suspect. It is computed from each well's
normal, so it carries information about which well a row belongs to, which is
the same issue flagged for LightGBM's top features in `RESULTS.md`. Removing
that channel costs only 0.006 R². So the anomaly is not what drives the
sequence-only result.

The main-data run splits the remaining gain. On the same single rain channel,
the BiLSTM reaches 0.360 against the transformer's 0.342: weekly steps and
recurrence account for about +0.02. The `in_interval`, calendar and wet-day
channels take it to 0.394: about +0.035. **Most of the gap is the data, not the
architecture.** That is the honest reading. It also says the transformer should
be tried on the same channels before the comparison is closed.

This matters for the project. The more of the explained variance comes from the
rain sequence rather than from static well properties, the more the residual
can be read as extraction pressure.

## Rain sensitivity: read this before building the what-if tool

On validation rows, every week's rain was scaled by 0.8 and by 1.2. The table
shows the mean change in the predicted fall:

| model | rain x0.8 | rain x1.2 | expected direction |
|---|---|---|---|
| BiLSTM, 6 channels | +0.012 m | −0.009 m | yes |
| BiLSTM, no anomaly channel | **−0.023 m** | **+0.018 m** | **no** |
| BiLSTM, main data, rain only | **−0.007 m** | **+0.006 m** | **no** |

The 6-channel model moves the right way: less rain, larger fall. The effect is
small, though, about 1 cm for a 20% change in two years of rain. Drop the
anomaly channel and the sign reverses. That model predicts a *smaller* fall
after a drier two years.

So the models fit `delta_h_m` without having learned a reliable dose-response
to rain. The fit metrics cannot show this, and the what-if simulator depends on
it. Before any of these models drives a scenario tool, it needs one of these:
- A monotonic constraint, which LightGBM supports directly.
- A penalty during training that pushes the rain response in the right
  direction.

At minimum, this check should run on every candidate model.

## Pooling map

`reports/figures/bilstm_pooling_by_season.png` shows the attention-pooling
weight on each of the 104 weeks, averaged over 2,000 validation rows per
campaign. Uniform pooling would put 1/104 on every week.

| campaign | heaviest week before the reading | calendar time | peak / uniform | trough / uniform |
|---|---|---|---|---|
| JAN | 20 | early September | 1.43 | 0.56 |
| MAY | 37 | late August | 1.62 | 0.43 |
| AUG | 102 (secondary peak at about 50) | late August, two years and one year back | 1.53 | 0.54 |
| NOV | 7 | late September | 1.71 | 0.58 |

Every peak falls in the late monsoon, and the peaks repeat about 52 weeks
apart. The model picked out the monsoon of each of the two past years without
being given a monsoon feature. The `doy` channels tell it where each week falls
in the calendar, not which part of the calendar matters.

Two caveats:
- **Strength.** The contrast is about the same size as the transformer's
  (0.4 to 1.7 times uniform here, 0.7 to 1.9 for the transformer). The
  difference is that the BiLSTM's pattern is smooth and consistent across
  seasons, and it lacks the transformer's edge-patch artifact.
- **Interpretation.** Pooling weights show where the model reads its summary
  from, not how much each week changes the prediction. Each LSTM state already
  mixes in its neighbours. Present this as "where the model looks", not as
  "which rain causes recharge".

## Residual hook

`reports/bilstm_residual_by_district.csv` uses the same convention as the
transformer's: positive means the level fell more than rainfall accounts for.
Districts with at least 3 wells, top of the list:

Junagadh 0.154, Amreli 0.146, Neemuch 0.146, Shajapur 0.101, Rajgarh 0.098.

Junagadh tops both models' lists. Amreli, Bhavnagar (both Saurashtra) and
Neemuch, Shajapur, Rajgarh (western Madhya Pradesh) cluster geographically.
This remains a hook, not the attribution, for the same reasons given in
`RESULTS.md`.

## Files

| file | what |
|---|---|
| `models/artifacts/bilstm.pt` | 6-channel BiLSTM weights, year split |
| `models/artifacts/bilstm_results.json` | year split: baselines, three BiLSTM variants, test, sensitivity, history |
| `models/artifacts/bilstm_results_cv.json` | BiLSTM district folds |
| `models/artifacts/bilstm_no_anom_results.json` | year split without `rain_anom_mm` |
| `models/artifacts/transformer_results_cv.json` | transformer district folds |
| `reports/figures/bilstm_pooling_by_season.png`, `reports/bilstm_pooling_by_season.csv` | pooling map |
| `reports/bilstm_residual_by_district.csv` | residual hook |

## Reproduce

```bash
python bilstm-data/build_bilstm_data.py                       # ~25 s
python kaggle/push.py                                         # dataset + main kernel (~25 min on T4)
python kaggle/push.py --kernel-only --script run_followup.py  # transformer CV + no-anomaly run (~30 min)
python kaggle/push.py --fetch
python kaggle/push.py --fetch --script run_followup.py
```

## Next

1. **Repeat seeds.** Run 3 to 5 seeds before any of the 0.01 m gaps between
   models goes on a slide.
2. **Rain-response constraint.** A fixed sign on the rain response, for the what-if path.
3. **Transformer with the 6 channels.** Separates the effect of the channels
   from the architecture on the sequence-only result.
4. **The full 694k-row table** from `pipeline/`, where the sequence models may
   finally have a data advantage over boosting.
