# Transformer results

216,455 rows, 2,759 wells, 33 features, 104 weeks of rainfall behind each
reading. Target `delta_h_m`, the change in depth since the previous reading of
the same well, positive when the water level fell. Blocked year split, no past
water level anywhere in the inputs.

## Headline

Validation 2015-2017, every model on the same rows:

| model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| zero (predict no change) | 2.163 | 3.227 | 0.000 |
| season mean | 1.813 | 2.756 | 0.271 |
| ridge on tabular | 1.753 | 2.872 | 0.208 |
| **LightGBM on tabular** | **1.526** | **2.435** | **0.431** |
| **transformer (sequence + tabular)** | **1.531** | **2.461** | **0.418** |
| transformer, sequence only | 1.641 | 2.618 | 0.342 |

Held-out test 2018-2022, transformer: **MAE 1.457, RMSE 2.302, R² 0.439.**

Train R² 0.513 against valid 0.418 and test 0.439. A 0.07-0.10 gap, no
collapse, and test scores *above* valid, so the model is not degrading as it
moves forward in time. 176,817 parameters, early stopped at epoch 14, best at
epoch 9.

## What to say about the LightGBM tie

The transformer and LightGBM are the same model to two decimal places
(MAE 1.531 vs 1.526). Claim this, not a win:

> Attention over the raw weekly rainfall series recovers gradient boosting on
> hand-engineered rain windows, without being told which windows matter.

That is the real result and it is defensible. LightGBM was handed
`rain_30d`, `rain_60d`, `rain_90d`, `rain_180d`, `rain_365d`, `rain_730d`,
three campaign lags and three history sums — eleven windows a human chose. The
transformer got 104 raw weekly numbers and found its own. Matching a strong
baseline from a weaker starting position is a result; claiming victory on
0.005 m of MAE is not, and the examiner will take that apart.

If pressed on why boosting is not simply better here: 216k rows of 33 tabular
features is boosting's home ground, and a 177k-parameter transformer on 144k
training rows is in the regime where deep models have no data advantage. Both
statements are true and neither is a defect in the architecture.

## The ablation is the more interesting number

Sequence only, no tabular features at all: **R² 0.342 against 0.418.**

The rainfall history alone recovers 82% of the full model's explained variance.
The 29 numeric plus 4 categorical tabular features — well depth, specific
yield, aquifer type, all the rainfall normals — add the remaining 18%.

So the rain sequence is carrying the model, which is exactly what the project
needs to be true. If the static well properties had dominated, the residual
would be mostly well-specific noise and the extraction attribution would be
much weaker.

LightGBM's own importance ranking agrees and adds a caution:

| rank | feature | gain |
|---|---|---|
| 1 | `well_depth_m` | 1318 |
| 2 | `rain_normal_monsoon_mm` | 782 |
| 3 | `rain_normal_annual_mm` | 764 |
| 4 | `rain_normal_season_mm` | 707 |
| 5 | `rain_interval_lag1` | 517 |

Places 2-4 are rainfall *normals*, which are per-well constants. They are doing
the job of a well fixed effect: telling the model which well it is looking at
without naming it. Worth flagging before someone else does. It is not leakage —
the normals are computed on training years only — but it does mean part of the
tabular arm's 18% is well identity rather than hydrology, which makes the
sequence-only result look better still.

## Attention

`data/training/attention_by_season.png`. Attention received by each of the 26
four-week patches, averaged over 2,000 validation rows per campaign, last
encoder layer.

Read it honestly. Uniform attention would be 1/26 = 0.038 and the bars run
roughly 0.025 to 0.071, so the pattern is real but moderate, not a sharp
monsoon spike. Two things are visible:

- The **August** campaign concentrates around 28-30 weeks and 14-16 weeks
  before the reading, which is roughly the previous January and the pre-monsoon
  weeks. The November campaign peaks near 48-52 weeks, a full year back.
- The **first and last patches** are elevated in every season. Edge patches in
  a learned positional encoding commonly attract weight regardless of content.
  Do not read the 104-week-ago bar as hydrology.

Say both parts. An examiner who has seen attention maps before will know the
edge artifact, and pre-empting it is worth more than a cleaner-looking claim.

## Residual hook

`data/training/residual_by_district.csv`. Residual = actual minus predicted, so
**positive means the level fell more than rainfall accounts for**, which is the
extraction-pressure direction. Districts with at least 3 wells, top of the list:

Junagadh 0.181, Bangalore urban 0.176, Sundargarh 0.160, Ganjam 0.152,
Gajapati 0.149, Rayagada 0.147.

This is a hook, not the attribution. No district effects are fitted, no trend
is estimated, and several entries sit on 3 wells. Do not put this on a slide as
a finding. It exists to show the residual path is wired end to end.

## Reproduce

```
python3 ml/build_training_data.py
python3 ml/train_transformer.py --epochs 25
python3 ml/attention_report.py
```

About 25 minutes on CPU for the training, 20 seconds for the build.

## Ablations still worth running

```
python3 ml/train_transformer.py --weeks 52 --epochs 25    # is the 2nd year earning its keep
python3 ml/train_transformer.py --patch 1 --epochs 25     # does patching cost anything
python3 ml/train_transformer.py --layers 1 --epochs 25    # is depth 3 needed
python3 ml/train_transformer.py --cv --epochs 20          # transfer to unseen districts
```

The `--cv` run is the one that matters most for the report. The year split says
the model generalises forward in time; the district-fold run says whether it
generalises to places it has never seen, which is the harder and more relevant
claim for a district-level attribution product.
