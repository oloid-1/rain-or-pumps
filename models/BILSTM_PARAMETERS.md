# BiLSTM training: every parameter, its value and the reason for it

This covers both models on the `bilstm` branch:
- **BiLSTM:** the comparison model in `models/bilstm.py` and `models/train_bilstm.py`.
- **Simulator:** the scenario model in `simulator/train_sim.py`, which reuses the same network.

Results are in `BILSTM_RESULTS.md` and `simulator/README.md`. This file says what was set,
where it is set, and why.

Every value below was read from the code and from the saved run configs in
`models/artifacts/*.json` and `simulator/artifacts/*.json`. None is quoted from memory.

**No hyperparameter search was run.** Every value is a reasoned default, most of
them carried over from the transformer so that the two architectures are compared
on equal terms. The test years were never used to choose anything. Section 7 lists
what a search would try first.

---

## 1. At a glance

| group | parameter | value | set in | flag |
|---|---|---|---|---|
| input | weeks of rain per row | 104 | `train_bilstm.py` | `--weeks` |
| input | sequence channels | 6 (`all`), or 1 (`rain_mm`) | `train_bilstm.py` | `--channels` |
| network | input projection width `d_in` | 32 | `bilstm.py` | `--d-in` |
| network | LSTM hidden size, per direction | 64 | `bilstm.py` | `--hidden` |
| network | LSTM layers | 2, bidirectional | `bilstm.py` | `--layers` |
| network | pooling over weeks | attention | `bilstm.py` | `--pool attn\|mean\|last` |
| network | shared width `d_model` | 64 | `bilstm.py` | `--d-model` |
| network | categorical embedding size | 8 | `transformer.py` (`TabEncoder`) | — |
| network | dropout | 0.1 | `bilstm.py` | `--dropout` |
| loss | Huber, δ | 1.0 m | `train_transformer.run_nn` | — |
| optimiser | AdamW, weight decay | 1e-4 | `run_nn` | — |
| optimiser | peak learning rate | 2e-3 | `train_bilstm.py` | `--lr` |
| schedule | OneCycle (warm-up, then cosine decay) | over all epochs | `run_nn` | — |
| batches | batch size | 512 | `train_bilstm.py` | `--bs` |
| batches | gradient clipping, max norm | 1.0 | `run_nn` | — |
| stopping | max epochs | 25 (year split), 20 (district folds) | `train_bilstm.py`, `kaggle/*.py` | `--epochs` |
| stopping | patience | 5 epochs without a better valid MAE | `run_nn` | `--patience` |
| stopping | minimum improvement | 1e-4 m of MAE | `run_nn` | — |
| stopping | checkpoint kept | the best valid-MAE epoch | `run_nn` | — |
| reproducibility | seed | 42 (numpy and torch) | `train_transformer.SEED` | — |
| simulator only | rain-response penalty weight | 1.0 | `train_sim.py` | `--mono-weight` |
| simulator only | rain shift per batch | ×(1 ± u), u ~ U(0.10, 0.40) | `train_sim.py` | — |

## 2. Network

```
sequence (104, C)  -> Linear(C, 32) + GELU
                   -> BiLSTM, 2 layers, 64 per direction, dropout 0.1 between layers   (104, 128)
                   -> attention pooling: softmax(Linear(128, 1)) over the 104 weeks    (128,)
                   -> LayerNorm -> Dropout -> Linear(128, 64)                          (64,)
tabular            -> standardised numerics + 8-dim embeddings per category
                   -> Linear -> GELU -> Dropout -> Linear(128, 64) -> GELU             (64,)
concat (128,)      -> Linear(128, 64) -> GELU -> Dropout -> Linear(64, 1)              delta_h_m
```

### Parameter counts, from the code

| block | BiLSTM, 6 channels | BiLSTM, rain only | sequence only, 6 ch | simulator, 6 ch |
|---|---|---|---|---|
| input projection | 224 | 64 | 224 | 224 |
| **BiLSTM (2 layers, both directions)** | **149,504** | 149,504 | 149,504 | 149,504 |
| attention score | 129 | 129 | 129 | 129 |
| LayerNorm + output Linear | 8,512 | 8,512 | 8,512 | 8,512 |
| category embeddings (4 × 8) | 240 | 240 | — | 240 |
| tabular MLP | 16,192 (29 numeric) | 16,192 | — | 13,248 (6 numeric) |
| head | 8,321 | 8,321 | 4,225 | 8,321 |
| **total** | **183,122** | **182,962** | **162,594** | **180,178** |

The LSTM is 82% of the model. Its 149,504 parameters break down as:
- layer 1: 2 directions × (4 gates × 64 × (32 input + 64 hidden) + 512 bias) = 50,176
- layer 2: 2 × (4 × 64 × (128 + 64) + 512) = 99,328

The category embeddings have 5 / 5 / 5 / 15 levels for season, well type, aquifer
and transition. Each size includes one extra "unseen" bucket for a level that never
appears in training.

### Why each choice

- **Hidden 64, 2 layers, about 183k parameters.** This is deliberately the
  transformer's size (176,817), so a difference in results cannot be put down to
  capacity. There are 144,558 training rows, about 0.8 per parameter, so a larger
  model would overfit before it learnt anything new. The fitted model's train R²
  is 0.530 against 0.419 on validation: a gap, but not runaway overfitting.
- **Bidirectional.** All 104 weeks are in the past when the prediction is made, so
  nothing is forecast and the encoder may read in both directions. The same argument
  explains why the transformer has no causal mask.
- **Input projection to 32 before the LSTM.** Six raw channels mix scales and kinds
  (a log-rain value, a 0–1 share, a sine). A learnt projection lets the LSTM gates
  see combinations rather than raw columns. It costs 224 parameters.
- **Attention pooling, not the last hidden state.** In a 104-step LSTM the last state
  is dominated by the last few weeks, and the two directions end at opposite ends of
  the sequence. A learnt weight per week lets the model pick its weeks. It also
  yields the per-week map in `reports/figures/bilstm_pooling_by_season.png`.
  `--pool mean` and `--pool last` exist for the ablation; they were not run.
- **No 4-week patches,** unlike the transformer. An LSTM's cost grows linearly with
  sequence length, so patching saves little. Weekly steps also keep the edge of the
  `in_interval` channel exact to the week.
- **Dropout 0.1** in three places: between the LSTM layers, before the pooled
  projection, and inside the tabular arm and the head. It is the transformer's value;
  it was not tuned.
- **`d_model` 64 for both arms.** The sequence and tabular summaries enter the head
  with equal width, so neither dominates the concatenation by size alone.

## 3. Inputs and how they are scaled

All statistics come from **training rows only** (2000–2014). They are applied
unchanged to validation and test.

| input | treatment |
|---|---|
| `rain_mm` | `log1p`, then standardised. Weekly rain is zero-inflated with a long right tail (max 1,679 mm); log1p pulls in the extreme weeks without cutting them |
| `rain_anom_mm` | `sign(x)·log1p(|x|)`, then standardised. It can be negative, so a plain log does not apply |
| `in_interval`, `doy_sin`, `doy_cos`, `wet_frac` | left as they are; already in [−1, 1] |
| missing weeks | set to 0 after scaling. None occur in the current data (all 216,455 rows fully covered) |
| numeric tabular (29 for the BiLSTM, 6 for the simulator) | missing values filled with the training median, then standardised |
| categorical tabular | indexed against training levels, plus one "unseen" index |

`--channels` chooses the sequence input:
- `all`: the 6 channels.
- `rain_mm`: the transformer's exact input.
- any comma list: for ablations, for example the no-anomaly run.

`--weeks` takes the most recent N weeks of the 104.

## 4. Optimisation

| setting | value | reason |
|---|---|---|
| loss | Huber, δ = 1 m | `delta_h_m` has a long tail of real 10 m swings. Under MSE a handful of them would set every gradient. Below 1 m the loss is squared error, above it absolute error |
| optimiser | AdamW, weight decay 1e-4 | weight decay kept separate from the adaptive step; light regularisation on top of dropout |
| peak learning rate | **2e-3** (the transformer uses 1e-3) | Recurrent gradients through 104 steps are smaller and better conditioned than attention's at this size. Clipping at 1.0 plus OneCycle's warm-up made the higher peak safe. **This is the one optimisation setting that differs from the transformer, and it was not tuned.** See section 7 |
| schedule | OneCycle over `epochs × (rows / 512 + 1)` steps | 283 steps per epoch, 7,075 in total at 25 epochs. Warm-up guards the first steps; the decay ends at a small rate |
| batch size | 512 | large enough for steady gradients on a T4, small enough for 283 updates per epoch |
| gradient clipping | max norm 1.0 | the standard guard against exploding gradients in an LSTM |
| early stopping | patience 5, on valid MAE, minimum improvement 1e-4 | MAE, not the Huber loss, because MAE is the reported metric. The best epoch's weights are restored at the end |
| seed | 42 | the same for every run, so differences between models are not seed differences. The flip side: everything is one seed |

One thing to know about stopping: OneCycle's schedule is fixed by `--epochs`. A run
that stops early ends before the learning rate has fully decayed. The year-split
runs stopped between epochs 10 and 20 of 25, and the fold runs between 10 and 20 of 20.

### What the runs actually did

| run | epochs run | best epoch | config file |
|---|---|---|---|
| BiLSTM, 6 channels | 19 | 14 | `models/artifacts/bilstm_results.json` |
| BiLSTM, main data, rain only | 14 | 9 | `models/artifacts/bilstm_main_results.json` |
| Simulator, 6 channels, penalty on | 17 | 12 | `simulator/artifacts/sim_6ch_results.json` |
| District folds (all models, 20-epoch cap) | 10–20 per fold; some ran the full 20 | — | `*_results_cv.json`, epochs from the Kaggle logs |

All on a Kaggle T4. BiLSTM runs take about 2–3 minutes each; penalised simulator
runs take about 25 minutes each (see section 5).

## 5. Simulator-specific parameters

The simulator keeps every setting above. It changes three things.

**Tabular inputs, static only.** The six numerics are:
- `well_depth_m`
- `sy`
- `days_since_prev`
- `rain_normal_annual_mm`, `rain_normal_season_mm`, `rain_normal_monsoon_mm`

The four categoricals are unchanged: season, aquifer, well type, transition.

The 23 rain-derived features are removed so that scaling the sequence is the whole
scenario. This is where the parameter count drops to 180,178.

**Rain-response penalty.** Each training batch is re-run with its rain multiplied by
`1 + u` and `1 − u`, with `u` drawn per row from U(0.10, 0.40):

```
loss = Huber(pred, target) + w · [ mean relu(pred(1+u) − pred(1)) + mean relu(pred(1) − pred(1−u)) ]
```

| parameter | value | reason |
|---|---|---|
| `w` (`--mono-weight`) | 1.0 | Same scale as the Huber term. It was large enough to cut wrong-way rows from 54% to 2.2%, at no cost in R². Not tuned; 0 (off) was the only other value run |
| range of `u` | 0.10 to 0.40 | Under 10% the prediction changes are within float noise. Over 40% the scaled rain leaves what the training data covers. The UI slider spans ±50%, and the checks at ×0.5 and ×1.5 still show 2% wrong-way, so the property extends beyond the trained range |
| one random `u` per row | — | covers the whole range in each batch instead of fixing a single test point |
| penalty passes in eval mode | dropout off | the three predictions must differ only in the rain, not in a dropout mask |
| cuDNN off for those passes | — | cuDNN refuses an LSTM backward pass in eval mode. The native kernel is exact but about 8× slower, which is why penalised runs take about 25 minutes, not 3 |

How rain is scaled: inside the network's scaled input space. Raw rain is recovered,
multiplied, and re-scaled; the anomaly channel moves by `(f − 1) · rain`. The torch
version (`RainScaler`) matches the numpy version used for checking to within 1e-5.

## 6. Settings carried over from the transformer, and the ones that differ

Same as the transformer: rows, split, target, leakage guard, Huber δ, AdamW and
weight decay, OneCycle, batch size, patience, gradient clipping, dropout, `d_model`,
category embedding size, seed, sequence scaling, and the tabular encoder (the same
code, `TabEncoder`).

Different from the transformer:

| | transformer | BiLSTM | why |
|---|---|---|---|
| sequence encoder | 4-week patches, 3 attention layers, 4 heads | 2-layer BiLSTM over weekly steps | the architecture under test |
| peak learning rate | 1e-3 | 2e-3 | see section 4 |
| sequence channels | 1 | 6 (or 1) | the `bilstm-data` copy. The 1-channel runs give the like-for-like comparison |
| pooling | mean over patches | attention over weeks | see section 2 |

Because the learning rate differs, the clean architecture comparison is the
rain-only BiLSTM against the transformer: 1.533 against 1.531 MAE. The learning-rate
difference is a stated caveat, not a hidden one.

## 7. What was not tuned, and what to try first

Nothing below has been run. In order of expected value:

1. **Seeds.** Run 3–5 seeds of each model before quoting any gap under 0.01 m MAE.
   This matters more than any single hyperparameter.
2. **Learning rate.** Try 1e-3 (the transformer's) and 5e-4 on the rain-only BiLSTM.
   This closes the one caveat in section 6.
3. **Penalty weight.** Try 0.3 and 3. The question is whether a lower weight keeps
   the wrong-way share near 2% (and gives back any fit), or whether a higher weight
   pushes it to 0%.
4. **Pooling.** `--pool mean` and `--pool last`, to show that attention pooling
   earns its place.
5. **Hidden size.** 32 and 128. The aim is to show the result is not sensitive to
   capacity, not to beat it.
6. **History length.** `--weeks 52`: does the second year of rain matter? The
   pooling map peaks in both monsoons, which suggests it does.

The rule for any of these: choose on validation 2015–2017. Open test 2018–2022 once,
for the final model only.

### Commands

```bash
cd models
python train_bilstm.py                                  # BiLSTM, 6 channels, everything at the defaults above
python train_bilstm.py --channels rain_mm               # the transformer's input
python train_bilstm.py --data ../data/training          # main data
python train_bilstm.py --lr 1e-3 --no-baselines         # learning-rate check
python train_bilstm.py --pool mean --no-baselines       # pooling ablation
python train_bilstm.py --hidden 128 --no-baselines      # capacity check
python train_bilstm.py --weeks 52 --no-baselines        # one year of history
python train_bilstm.py --cv --epochs 20                 # 5 district folds

cd ..
python simulator/train_sim.py --mono-weight 0.3         # penalty strength
python simulator/train_sim.py --mono-weight 0           # penalty off
```

These are slow on CPU. The full BiLSTM run is about 3 minutes per epoch per model.
Kaggle versions: `python kaggle/push.py`. Add `--script run_sim.py` for the
simulator, then `python kaggle/push.py --fetch` for the outputs.
