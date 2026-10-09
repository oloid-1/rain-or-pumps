# Review of the model results, and which models to try next

A critical read of the two results files, checked against the saved models.
- `models/RESULTS.md`: the transformer.
- `models/BILSTM_RESULTS.md`: the BiLSTM.

It answers three questions:
- What do the results actually support?
- What should change before they are presented?
- Which models are worth training next?

Every number here that is not in the two results files comes from
`models/review_checks.py`, which reloads the saved models and recomputes them in
about two minutes on CPU:

```bash
python models/build_sequences.py         # if data/training/ is not built
python models/review_checks.py
```

The transformer's test score (0.439) and district-fold scores come from
`RESULTS.md` and `models/artifacts/transformer_results_cv.json`. The script does
not reload the transformer.

---

## Summary

| # | finding | severity | changes what we say? |
|---|---|---|---|
| 1 | On held-out test years, **LightGBM beats every sequence model** (R² 0.484 against 0.439–0.453). The "tie" exists only on the validation years that every model used for early stopping | high | yes: the headline claim |
| 2 | All models make **nearly the same errors** (error correlation 0.96), so ensembling gains almost nothing. The ceiling, about R² 0.45–0.48, comes from the data, not the model | high, and a finding in its own right | yes: it supports the project's premise |
| 3 | The **district residual ranking is mostly built from training rows**. On held-out rows only, 4 of the top 10 districts survive and Junagadh drops out | high for attribution | yes: the named districts |
| 4 | **The calendar alone gives R² 0.26.** Rain and well facts add 0.18–0.23 test R² on top, so "rain carries the model" needs the qualifier "with the calendar" | medium | yes: one phrase |
| 5 | The comparison of unseen districts against unseen years is confounded; one seed throughout | medium | soften one sentence |
| 6 | **Ridge is broken**: test R² −0.013, worse than predicting no change | medium | fix it or drop it from the table |
| 7 | Out-of-date paths and next-step lists | low | housekeeping |

What the files already get right is listed in section 8. Both are more careful
than most write-ups of this kind, and most of this review tightens claims they
already hedge.

---

## 1. On held-out years, LightGBM wins

Both results files compare models on **validation (2015–2017)**. Every model also
used those years to decide when to stop:
- LightGBM stops adding trees when validation error stops improving (it stopped
  at 623 of 800).
- The transformer and the BiLSTM keep the epoch with the best validation MAE.

So validation scores are mildly flattering for every model. Comparing models on
validation is fine for a first look, but the fair head-to-head is **test
(2018–2022)**, which no model saw in any form. Neither file reports LightGBM's
test score.

| model | valid R² | valid MAE | **test R²** | **test MAE** |
|---|---|---|---|---|
| zero (predict no change) | 0.000 | 2.163 | — | — |
| season mean | 0.271 | 1.813 | 0.258 | 1.762 |
| ridge | 0.209 | 1.753 | **−0.013** | 1.737 |
| **LightGBM** | **0.435** | **1.515** | **0.484** | **1.427** |
| simulator, 6 channels | 0.426 | 1.521 | 0.453 | 1.439 |
| BiLSTM, 6 channels | 0.419 | 1.521 | 0.441 | 1.449 |
| transformer (`RESULTS.md`) | 0.418 | 1.531 | 0.439 | 1.457 |

LightGBM leads by 0.03–0.045 R² and 0.012–0.030 m MAE on test. That is larger than
any of the gaps the files call ties.

Two notes:
- **The test target is a little less variable** (sd 3.07 m against 3.23 on
  validation). This is why every model's test R² sits above its validation R².
  It is not evidence that the models "improve forward in time", which
  `RESULTS.md` implies.
- **The ordering on test is LightGBM, then the simulator, then the BiLSTM and the
  transformer.** The simulator is the best sequence model on test, without the 23
  rain-derived tabular features.

### What to say instead

| current claim | revised claim |
|---|---|
| "The transformer matches LightGBM." / "a three-way tie" | "The sequence models match boosting on the validation years and trail it by about 0.04 R² on held-out years." |
| "Attention recovers gradient boosting without being told which windows matter." | "Attention and recurrence, given only raw weekly rain, get within 0.04 R² of boosting on 33 hand-built features, without being told which windows matter." |
| "Test scores above valid, so the model is not degrading forward in time." | "Test R² is higher for every model because the test years are less variable; it says nothing about degradation." |

The defensible case for the sequence models was never accuracy. It is what
boosting cannot give:
- a rain response usable for scenarios (the simulator: right-way for 98% of wells);
- a per-week attention map;
- no hand-engineered rain windows.

Present them on that.

---

## 2. Every model hits the same ceiling

| | value |
|---|---|
| correlation of LightGBM's errors with the BiLSTM's (valid) | **0.961** |
| correlation of LightGBM's errors with the simulator's (valid) | **0.962** |

| ensemble (weights chosen on validation only) | valid R² | test R² | test MAE |
|---|---|---|---|
| LightGBM alone | 0.435 | **0.484** | 1.427 |
| 0.5 LightGBM + 0.5 BiLSTM | 0.438 | 0.476 | 1.418 |
| 0.70 LightGBM + 0.30 BiLSTM | 0.439 | 0.482 | **1.416** |
| 0.5 LightGBM + 0.5 simulator | 0.441 | 0.480 | **1.416** |

Averaging trims about 0.01 m of test MAE and does not raise test R². The models
are wrong on the same rows.

This is the most useful result in the review, because it is evidence for the
project's premise. Four different model families, given different
representations of the same inputs, converge on R² about 0.45–0.48. More model
will not close the gap; more data of the right kind would. The roughly half of the
variance no model explains is variance that rainfall and fixed well facts cannot
explain. The project's hypothesis is that much of it is extraction. That is not
proof (soil, canals, cropping and irrigation return flows are also missing), but
"the ceiling is the data, not the model" is now measured, not assumed.

It also means another forecaster of the same kind (a GRU, a temporal CNN, a bigger
transformer) is very unlikely to change anything. See section 9.

---

## 3. The district residual ranking uses training rows

Both files rank districts by the mean residual (actual minus predicted) over
**all rows**. About two thirds of each district's rows (0.67 on average) are
training rows, which the model has fitted, so their residuals are smaller and
biased towards zero:

| | mean absolute residual |
|---|---|
| training rows | 1.376 m |
| held-out rows (valid + test) | 1.477 m |

Ranking on held-out rows only changes the answer at the top:

| rank | all rows (as published in `BILSTM_RESULTS.md`) | held-out rows only |
|---|---|---|
| 1 | Junagadh | Bangalore urban |
| 2 | Amreli | Dakshin Kannada |
| 3 | Neemuch | Uttara Kannada |
| 4 | Shajapur | Jalpaiguri |
| 5 | Rajgarh | Neemuch |
| 6 | Bhavnagar | Narmada |
| 7 | Banaskantha | Bhavnagar |
| 8 | Chhindwara | Udupi |
| 9 | Medak | Banaskantha |
| 10 | Porbandar | Amreli |

The two lists share 4 of their top 10 and 9 of their top 20. The rank correlation
over all 272 districts is 0.79: the broad picture holds, the names at the top do
not. Junagadh, quoted in both results files, is not in the held-out top 10.

Both files already call this "a hook, not the attribution", which is right. But no
district name from either published list should be quoted. The fix is in section
9, item 1: rank on **out-of-fold** residuals, where every row's prediction comes
from a model that never trained on it.

The UI's Pressure mode defaults to 2015–2022 (held-out years), so its default
ranking is clean. A range that reaches back into 2000–2014 averages training
residuals and inherits the problem.

---

## 4. "Rain carries the model" needs "with the calendar"

| model | test R² |
|---|---|
| season mean (knows only the month of the reading) | 0.258 |
| BiLSTM | 0.441 |
| LightGBM | 0.484 |

The month alone explains a quarter of the variance. Rain and well facts together
add **0.18 (BiLSTM) to 0.23 (LightGBM)** test R² on top.

This matters for the sequence-only results:
- The transformer's sequence-only 0.342 and the BiLSTM's 0.400 include the
  seasonal signal. A 104-week series that ends in August implicitly knows it is
  August.
- The 6-channel BiLSTM is told the calendar directly (`doy_sin`, `doy_cos`) and
  the interval length (`in_interval`).

So "the rain history alone recovers 95% of the full model" overstates how much is
rain. The supportable sentence is: "**rain history plus calendar** recovers 95%."

**The clean test, not yet run:** the sequence-only BiLSTM with the calendar and
interval channels kept and the rain channels zeroed. Whatever that scores is the
calendar's share; the rest is rain.

---

## 5. Two smaller problems in the BiLSTM results

**Unseen districts against unseen years is not a like-for-like comparison.**
`BILSTM_RESULTS.md` says the district folds score higher (R² about 0.48) than the
year split (about 0.42), "so the harder problem is the shift in time, not the
shift in place". But the two set-ups differ in more than the kind of shift:

| | year split | district folds |
|---|---|---|
| years trained on | 2000–2014 | all, 2000–2022 |
| share of rows trained on | 67% | 80% |
| shift tested | time | place only |

The folds also train on more data and on the same years they are tested on. The
sentence should say what is measured: "the models transfer to unseen districts
at least as well as to unseen years". It should not rank the two difficulties.

**One seed throughout.** `BILSTM_RESULTS.md` says so; `RESULTS.md` does not. Any
gap under about 0.01 m MAE or 0.01 R² between two models should be treated as
noise until it holds across 3–5 seeds.

---

## 6. Ridge is broken

Ridge has the season as an input, so it should at least match the season-mean
baseline. Instead:

| | valid R² | test R² |
|---|---|---|
| season mean | 0.271 | 0.258 |
| ridge | 0.209 | **−0.013** |

On test it is worse than predicting no change at all.

The likeliest cause is `rain_anom_season_ratio`, which divides by each well's
seasonal rain normal. In dry seasons that normal is near zero, so the ratio is
unbounded. A linear model extrapolates those extreme values straight into its
predictions, where the trees in LightGBM simply split them off.

Either fix it, for example by clipping or log-transforming the ratio features for
ridge, or drop ridge from the comparison table. As it stands it makes the linear
baseline look weaker than it is, which flatters every other model.

---

## 7. Housekeeping

| file | issue |
|---|---|
| `RESULTS.md` | reproduce commands use the old `ml/` paths (`python3 ml/build_training_data.py`); the scripts now live in `models/` |
| `RESULTS.md` | says the attention figure and residual table are written to `data/training/`; the published copies are in `reports/` |
| `RESULTS.md` | "train R² 0.513" is quoted without the matching validation-year caveat from section 1 |
| `BILSTM_RESULTS.md` | the "Next" list still has the rain-response constraint as to-do; the simulator (`simulator/README.md`) has since done it. The rain-sensitivity section should point there |
| both | neither reports LightGBM's test score |

---

## 8. What the two files get right

So that the review is not read as a teardown:

- **Leakage is designed out and enforced.** No past water level, year, location or
  identifier is a feature, and an assertion fails the build if one appears.
  Normals use training years only. This is the most important design decision in
  the project, and both files explain it.
- **Ties are reported as ties.** `RESULTS.md` explicitly says not to claim a win on
  0.005 m. The review only extends that honesty to the test years.
- **Caveats are raised before an examiner does:**
  - the edge-patch artifact in the attention map;
  - the rainfall normals acting as well identifiers in LightGBM;
  - pooling weights describing where the model looks, not causation;
  - the residual table as a hook.
- **Ablations are run, not assumed:** sequence only, rain only, no anomaly, main
  data versus copy.
- **The rain-sensitivity check in `BILSTM_RESULTS.md` found a real problem** (the
  sign flipping) that fit metrics could not show. The simulator was built to fix
  it, and it did.

---

## 9. Which models to try next

Section 2 decides the order. The models already share one ceiling, so the useful
next models either use the residual (attribution), quantify uncertainty, or give
a scenario model with a hard guarantee. Another forecaster is near the bottom.

### 1. Attribution model on out-of-fold residuals (recommended)

**What:** a mixed-effects model on the residuals:

```
residual(well w, district d, year t) = a_d + b_d * (t - 2000) + u_w + e
```

- `a_d`: a district's typical unexplained change.
- `b_d`: how that gap is trending, in metres per year.
- `u_w`: a random effect per well, absorbing well quirks.
- Districts are partially pooled, so a district with three wells is shrunk towards
  its state rather than ranked on noise.

**Why first:** this is the project's actual deliverable: which districts are
falling faster than rain explains, by how much a year, and how sure we are. It
uses the best model, LightGBM, rather than competing with it, and it fixes
section 3.

**How:**
1. Get **out-of-fold predictions** for every row from the 5 district folds (each
   row is predicted by the model trained on the other four folds), for LightGBM
   and the simulator.
2. Fit the mixed model, with `statsmodels` `MixedLM` or `pymer4`, on the residuals.
3. Report each district's `a_d` and `b_d` with confidence intervals, ranked by the
   lower bound, not the point estimate.
4. **Validate** against CGWB's published stage-of-extraction categories
   (over-exploited, critical, semi-critical, safe). Over-exploited districts
   should show positive, significant `b_d` more often than safe ones. The
   first pipeline's attribution step (since removed) already did this comparison
   for the old target, so most of the code exists.

**Success looks like:** a ranked table with intervals, and a clear separation
between CGWB's over-exploited and safe districts. If the separation is absent,
that is also a reportable result: rain-adjusted decline then does not track
official extraction estimates.

**Effort:** about half a day; CPU only.

### 2. Quantile models (uncertainty)

**What:**
- LightGBM with `objective="quantile"` at α = 0.1, 0.5 and 0.9.
- The simulator with a pinball loss on three outputs.

**Why:** it turns single numbers into ranges. That makes it possible to say whether
a district's residual is outside what rain variation alone produces, and to show
scenario results as bands rather than a point.

**Success looks like:** an 80% interval that covers about 80% of test rows.

**Effort:** 2–3 hours.

### 3. Monotone-constrained LightGBM as a second simulator

**What:** LightGBM with `monotone_constraints` set to "more rain, smaller fall" on
every rain-amount feature.

**Why:** a hard guarantee on the direction of the response, against the
simulator's soft penalty (2.2% of rows still go the wrong way). It is also the
best model on test. Comparing the two simulators' rain responses is a clean
experiment.

**The cost:** each scenario has to rebuild LightGBM's rain features from scaled
daily rain: windows, lags, history sums and anomalies. That means running the
feature code in `models/build_training_data.py` on modified rain, not just
rescaling a sequence. It is also why the BiLSTM simulator was built the way it was.

**Effort:** about half a day.

### 4. Frozen foundation-model encoder (TimesFM or Moirai)

**What:** item 4 of the original plan (`models/START_HERE.md`). The 104-week rain
series goes through a frozen pretrained encoder, with a small trained head on top.

**Why or why not:** given section 2, expect about R² 0.45. Worth doing if the
course brief requires a foundation-model comparison, and as a transfer-learning
data point.

**Rule:** water depth must never enter the input, or the model forecasts depth
from depth.

**Effort:** about a day, on a Kaggle GPU.

### Not worth the time

| model | why not |
|---|---|
| GRU, temporal CNN, a bigger transformer | the same family as the BiLSTM and transformer; section 2 says they will make the same errors |
| a graph neural network over neighbouring wells | would almost certainly use neighbours' water levels, which carry the pumping signal: the leakage the whole design avoids. Neighbours' rain is already in the 0.25° grid |
| more ensembling | measured in section 2: −0.01 m MAE, no R² gain |

---

## 10. Recommended order of work

1. **Correct the two results files** (sections 1–7):
   - Add the test column, with LightGBM.
   - Revise the tie claim.
   - Recompute the residual tables on held-out rows, and remove the quoted
     district names.
   - Add "with the calendar".
   - Soften the time-against-place sentence.
   - Fix or drop ridge.
   - Fix the paths and the "Next" lists.
2. **Run the clean calendar test** (section 4): sequence-only with the rain zeroed.
   One Kaggle run, about 5 minutes.
3. **Build the attribution model** (section 9, item 1). This is the project's
   result.
4. **Quantile intervals** (item 2), for the attribution table and the UI.
5. Only then: the monotone LightGBM simulator, the foundation-model encoder, and
   seed repeats for any comparison that will be quoted.
