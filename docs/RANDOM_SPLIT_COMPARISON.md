# Random validation split: BiLSTM, LightGBM, transformer - two configs each

Requested at the last review: replace the year-blocked validation split
(train 2000-2014 / valid 2015-2017 / test 2018-2022) with a **random**
split, and run **two hyperparameter configurations** for each of BiLSTM,
LightGBM and the transformer. This is the write-up that ties the four new
notebooks together.

| notebook | what it does |
|---|---|
| `notebooks/03_random_split_eda.ipynb` | builds the random split, checks it for leakage, compares it to the year-blocked split |
| `notebooks/04_bilstm_random_split.ipynb` | BiLSTM, 2 configs |
| `notebooks/05_lightgbm_random_split.ipynb` | LightGBM, 2 configs |
| `notebooks/06_transformer_random_split.ipynb` | transformer, 2 configs |

All results: `data/training/random_split_results.json`. Split assignment
and a text report: `data/training/random_split_assignment.csv`,
`data/training/random_split_report.txt`. Code: `models/random_split.py`,
`models/bilstm.py`, `models/train_random.py`.

## 1. The split: random, but grouped by well

A plain row-level random split would shuffle each well's ~78 readings
independently, so some of a well's rows land in train and others in
validation. Those rows share the same coordinates, aquifer, well depth
and overlapping rain history - a model that has seen 60 of a well's 78
readings in training has effectively already seen that well, and every
metric on the other 18 would be optimistic. That is leakage, just a
different shape of it than the year-block split had.

So this is a **well-grouped** random split: every well's rows go to one
side only (train, valid or test), the side is chosen by a shuffle, seed
42, 70/15/15 by well count. It is still "random, not by year" exactly as
asked - no calendar information decides who is in validation.

Checked in the EDA notebook: 0 wells cross splits, and `delta_h_m`'s mean
and spread are near-identical across train/valid/test (as a correct
random split should look) - unlike the year-block split, whose test
window (2018-2022) really does carry a different mean than its own train
window, a genuine multi-year drift rather than a split artefact.

**A second caveat raised in the same notebook, not fixed by this split:**
a few features - `well_depth_m`, the rainfall normals - barely change
across a well's own readings, so they can act partly as a soft well
identifier rather than a physical signal. Grouping by well keeps the
*split* honest; it does not stop a tree model from partly keying on which
well a row belongs to through those features. Confirmed in the LightGBM
notebook: `well_depth_m` is comfortably the top feature in both configs,
more dominant here than on the year-split.

## 2. Configs - chosen to answer a question, not just to have two numbers

| model | config a | config b | question it answers |
|---|---|---|---|
| BiLSTM | hidden 64, 2 layers, lr 2e-3 (the documented default) | hidden 32, 2 layers, lr 1e-3 | two open items from `BILSTM_PARAMETERS.md` section 7 at once: the learning-rate caveat against the transformer, and capacity sensitivity |
| LightGBM | 800 trees, lr 0.05, 63 leaves (the repo's existing baseline) | 300 trees, lr 0.1, 31 leaves | is the headline number a tuning artefact of one leaf/tree count? |
| Transformer | d_model 64, 3 layers, 4 heads, lr 1e-3 (the repo's existing default) | d_model 32, 2 layers, lr 2e-3 | same capacity question, transformer side |

Both neural configs trained up to **12 epochs, patience 4** on validation
MAE - capped down from the usual 25/5 **for wall-clock reasons on a
2-core CPU box with no GPU**; each of the four NN runs took 25-60 minutes
here. That cap is the honest limitation on these numbers: BILSTM_RESULTS.md
and the transformer's own 25-epoch run both kept improving past epoch 12
on the year split, so these random-split numbers are plausibly a little
short of fully converged. The *comparison between configs and between
models* is still fair, since every run shares the same cap.

The BiLSTM here is also, by necessity, the **1-channel ("rain only")**
variant from `BILSTM_PARAMETERS.md` - this repo never had the 6-channel
`bilstm-data` pipeline merged in, only the single `rain_mm` sequence the
transformer also uses. Rebuilding that 6-channel cube was out of scope
for this pass. It's the fairer comparison anyway: both sequence arms see
exactly the same rainfall input, so the difference is architecture, not
input richness.

## 3. Results

Validation / test MAE (metres) and R², random split:

| model | params | valid MAE | valid R² | test MAE | test R² |
|---|---|---|---|---|---|
| LightGBM a (800 trees) | - | 1.423 | **0.533** | 1.433 | 0.508 |
| LightGBM b (300 trees) | - | 1.423 | 0.533 | 1.439 | 0.503 |
| BiLSTM a (hidden 64) | 182,962 | 1.413 | 0.514 | 1.433 | 0.488 |
| BiLSTM b (hidden 32) | 52,722 | 1.418 | 0.513 | 1.439 | 0.482 |
| Transformer a (d_model 64) | 176,817 | 1.415 | 0.517 | 1.433 | 0.488 |
| Transformer b (d_model 32) | 34,865 | 1.410 | 0.520 | 1.431 | 0.490 |

## 4. What to take from this, and what not to

**It is still a three-way tie**, now at a higher R² than the year-split
reported. All six configs land in a tight 0.51-0.53 valid R² band.
LightGBM has a small, consistent edge (~0.013-0.015 R² over both neural
models, in both its own configs) - small enough that it could be the
12-epoch cap rather than a real architecture gap, but it showed up
consistently enough to report rather than wave away.

**Within each neural model, the smaller config matched or slightly beat
the larger one** (transformer b > a; BiLSTM a and b are within 0.001 of
each other). Read together with the LightGBM result above - two very
differently-sized LightGBM fits landing within 0.0002 R² of each other -
the pattern across all three models is the same one the week-5 deck
already made about architecture: **capacity is not what's limiting these
numbers.** A quarter of the parameters gets the same answer.

**The random split is easier than the year-blocked split for every
model** - roughly 0.51-0.53 R² here versus ~0.42-0.44 R² there, for the
same features, same target, same architectures. `06_transformer_random_split.ipynb`
puts this side by side on one chart. This is the same finding the week-5
deck already made about district folds (unseen districts, R²≈0.48, easier
than unseen years, R²≈0.42) landing again on a third axis: **predicting a
randomly withheld 15% of wells is easier than predicting withheld years,
which is easier than nothing at all.** Time, not space, not well
identity, is the harder generalisation axis in this data. None of this
should be read as "the random-split model is the real one, 0.53 R² is our
number" - it's a different, easier question, and reporting it as the
headline without the year-split number beside it would be the kind of
overclaim worth catching before someone else does.

## 5. Honest limitations of this pass

- 12-epoch cap on the neural models, for wall-clock reasons on CPU-only
  hardware - plausibly a little short of fully converged relative to the
  25-epoch runs elsewhere in the repo.
- BiLSTM here is the 1-channel variant, not Parth's documented 6-channel
  one (that pipeline isn't in this repo).
- One seed. `BILSTM_PARAMETERS.md` section 7 already flags seed variance
  as the thing to check before trusting any gap under 0.01 MAE - that
  applies here too, more so with only 12 epochs.
- No rain-sensitivity (dose-response) check was repeated on these random-split
  models. That check (scaling validation rainfall ×0.8/×1.2) is a
  property of each trained model, not of the split, and worth re-running
  here if time allows before the review.
