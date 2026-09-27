# Transformer task — what you need to start

Your two items: (2) transformer architecture, (4) foundation-model exploration.
Everything below is yours alone. Nothing here waits on anyone else's branch.

## 1. Three commands

From `~/Downloads/BITS-Capstone-Groundwater`:

```
pip3 install torch pandas pyarrow xarray netcdf4 scikit-learn lightgbm
python3 ml/build_training_data.py
python3 ml/train_transformer.py --epochs 25
```

The build takes about 20 seconds and writes `data/training/`:

| file | what it is |
|---|---|
| `tabular.parquet` | 216,455 rows, one per (well, campaign) that has a previous reading |
| `rain_seq.npz` | `weeks` array, 216,455 x 104, weekly rainfall behind each reading, oldest week first |
| `feature_spec.csv` | every column tagged `feature`, `target` or `meta` |
| `build_report.txt` | row counts and split sizes |

Train once with `--epochs 3` first to confirm it runs, then the full 25.

## 2. Why this data and not Parth's

Parth's `data_cleaning` branch produces the same contract from the 32,299-well
archive, but it needs the CGWB zip unpacked, a districts geojson, geopandas,
and a manual gap-cleaning notebook run before `training_data.py` will emit
anything. None of that is your task and none of it is on disk yet.

`ml/build_training_data.py` goes from the two raw sources you already have
straight to the same target, the same split years, the same forbidden columns
and the same 5 district folds. Column names match his. When his table lands you
change one path and rerun. The architecture work is not blocked in the meantime.

Difference to state plainly if asked: 2,759 wells here against his 32,299,
because this reads the filtered CSV and his reads the full archive. Readings are
placed mid-month because the CSV gives a month and not a day, so interval
lengths are exact to within a few days. Say both out loud rather than being
caught on them.

## 3. The contract you are coding against

**Target** `delta_h_m` = depth now minus depth at the previous reading of the
same well, in metres. **Positive means the water level fell.** Get this the
wrong way round in a chart and the whole story inverts.

Mean is near zero in every split, standard deviation about 3.1 m. That near-zero
mean is the point: there is no trend to coast on, so a model has to actually use
the rain.

**Never a feature** (`role = meta` in the spec, and asserted in `load()`):
`depth_start_m`, `depth_mbgl`, `season_year`, `lat`, `lon`, `district`, `state`,
`well_uid`, `campaign_date`, `prev_date`, `split`, `district_fold`.

`depth_start_m` is the one that matters. Feed it in and you get R2 near 0.9 and
a model that has learned mean reversion instead of hydrology. The examiner will
ask. Point at the `assert not leak` in `load()`.

**Splits** train 2000-2014, valid 2015-2017, test 2018-2022. Blocked by year, so
no future rain informs a past prediction. `--cv` switches to the 5 district
folds instead, which tests whether the model transfers to districts it has never
seen. Report both; they answer different questions.

**33 features**, all rain or static well properties. No past water level
anywhere, which is what makes the residual interpretable as extraction
pressure later.

## 4. The architecture, and the reason for each choice

```
weekly rain (104,)  ->  26 patches of 4 weeks  ->  linear + learned position
                    ->  3 encoder blocks, 4 heads, d=64, pre-norm
                    ->  mean pool                                   (64,)
tabular (30 numeric + 4 categorical embeddings)  ->  MLP            (64,)
concat (128,)  ->  MLP  ->  scalar
```

177k parameters. Small on purpose: 144k training rows does not support a large
model, and a transformer that only ties LightGBM is a perfectly reportable
result, whereas one that overfits visibly is not.

**Why a transformer when the table already has `rain_30d`, `rain_90d`,
`rain_365d`.** Those windows are a guess about which part of the rain history
matters. Attention learns the weighting instead of being told it. That is the
whole argument, and it is also the honest framing: the transformer's job is to
beat your own hand-built windows, not to be impressive.

**Why patches of 4 weeks, not one token per week.** 104 tokens costs 104^2 per
head for nothing. Rainfall decorrelates in days, so a 4-week patch discards no
structure, and the attention map then reads as "which month mattered", which is
a slide you can show.

**Why no causal mask.** You are not forecasting rain. Every one of the 104 weeks
is already in the past at prediction time, so the encoder should see all of them
at once. This is also the answer to the leakage question about foundation
models in section 6.

**Why pre-norm.** Post-norm needs warmup to train at this depth. Pre-norm just
trains.

**Why Huber loss, not MSE.** `delta_h_m` has a long tail of real 10 m swings.
Under MSE a handful of them set every gradient.

**Why log1p on the rain sequence.** Weekly rain is zero-inflated and
right-skewed. log1p pulls the 500 mm weeks in without throwing them away.

**Why mean pool and not a CLS token.** With 26 patches and no pretraining a CLS
token has nothing to inherit, and mean pooling is one fewer thing to defend.

## 5. What to run and report

```
python3 ml/train_transformer.py --epochs 25          # year split, all baselines
python3 ml/train_transformer.py --cv --epochs 20     # 5 district folds
python3 ml/train_transformer.py --weeks 52 --epochs 25   # does 2 years beat 1
python3 ml/train_transformer.py --patch 1 --epochs 25    # does patching cost anything
python3 ml/train_transformer.py --layers 1 --epochs 25   # is depth earning its keep
```

The script already prints this table, on the same rows and the same split:

| model | what it tests |
|---|---|
| zero | predict no change at all |
| season mean | the training mean for that campaign month |
| ridge | linear on the tabular features |
| lightgbm | gradient boosting on the tabular features |
| transformer | sequence arm + tabular arm |
| transformer seq only | rain history alone, no tabular features |

The `seq only` ablation is the one worth thinking about. If it lands close to
the full model, the tabular arm is decoration. If it is far worse, the static
well properties carry the signal and the sequence is decoration. Either answer
is a finding, and having the answer ready is worth more in the viva than a good
number.

Lose to LightGBM and say so. 216k rows of 33 features is boosting's home
ground. The defensible claim is "attention over the raw weekly series recovers
gradient boosting on hand-built windows without being told which windows to
use", not "the transformer won".

## 6. Foundation models, item 4

**There is no pretrained groundwater model.** Nothing on HuggingFace is trained
on groundwater levels. Establish that first, with the search, so nobody spends
a week looking. What exists is general time-series foundation models:

| model | covariates | note |
|---|---|---|
| Chronos / Chronos-Bolt (Amazon) | no | univariate only, so it cannot see rain |
| TimesFM (Google) | yes, as of 2.x | closest usable fit |
| Moirai (Salesforce) | yes, any-variate | designed for exogenous inputs |
| Lag-Llama | no | univariate |
| Granite-TimeSeries (IBM) | yes | small, patch-based, close cousin of what you built |

**The trap, and say it before the examiner does.** These models forecast a
series from its own past. Point one at groundwater depth and it will predict
next season's depth from last season's depth, score beautifully, and answer a
question nobody asked. Our target is explicitly the part of the level that rain
does *not* explain, so any model given past depth destroys the attribution. A
foundation model is only admissible here if depth never enters its input.

**The two framings that survive that.**

*Frozen encoder.* Run the 104-week rain sequence through a frozen TimesFM or
Moirai encoder, take the representation, and train only a small head to predict
`delta_h_m`. Depth never enters. This is transfer learning on the rain side
only, and it drops into `RainTransformer` by swapping `SeqEncoder` for the
frozen model. That is your comparison: your own 177k-parameter encoder against
a pretrained one, same head, same rows, same split.

*Rain nowcast, strictly separate.* A foundation model forecasts rainfall, and
the forecast feeds the scenario tool. Useful for the what-if screen, irrelevant
to attribution. Keep it in a different section of the report so the two are
never confused.

**Selection criteria, the "scope overlapping" the mentor asked for.** Score each
candidate on: accepts exogenous covariates, does not require the target's own
history, licence permits use, fits in memory on a laptop, has a frozen-encoder
path, and gives an attention or attribution map you can show. TimesFM and Moirai
pass most of it. Chronos and Lag-Llama fail the first criterion, which is enough
to drop them, and saying why you dropped them is a slide.

## 7. Files

```
ml/build_training_data.py   raw CGWB + IMD -> data/training/
ml/transformer.py           SeqEncoder, TabEncoder, RainTransformer, SeqOnlyTransformer
ml/train_transformer.py     baselines + training loop + ablation, prints the table
```

`SeqEncoder.attention_maps()` returns per-layer attention, shape
(batch, 26, 26). Average it over a season and plot which of the 26 four-week
patches the model reads. That is your interpretability slide and it is the one
thing a boosted tree cannot give you.
