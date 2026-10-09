# bilstm-data/

The training data for the BiLSTM: a copy of the main training table with a
richer weekly sequence. The main build in `data/training/` is untouched, so the
transformer's results stay reproducible.

## Build

```bash
python models/build_training_data.py      # optional; the copy runs the same build itself
python bilstm-data/build_bilstm_data.py   # ~25 s, writes bilstm-data/out/
```

Needs the same inputs as `make data`: `data/derived/rain_cube.npz` (tracked) and
`data/cgwb/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv`.

**Note on the CGWB CSV.** `scripts/fetch_data.py --cgwb` no longer finds it. The
figshare record (v3) now lists only `AUG.csv`, `JAN.csv`, `POST.csv`, `PRE.csv` and
the raw archive zip. Copy the CSV from an older checkout until the fetch script is
updated.

## What is in `out/`

| file | shape | what it is |
|---|---|---|
| `tabular.parquet` | 216,455 rows | **identical** to `data/training/tabular.parquet` (checked with `DataFrame.equals`) |
| `feature_spec.csv` | 46 columns | identical, same leakage roles |
| `rain_seq.npz` | 216,455 x 104 | identical, the transformer's single channel |
| `seq_channels.npz` | `x` 216,455 x 104 x 6, float16; `channels` | the BiLSTM input. Channel 0 is byte-identical to `rain_seq.npz` |
| `build_report.txt` | | counts, split sizes, per-channel statistics |

Each sequence holds 104 weeks ending on the reading date, oldest week first.
Rows, target, splits (train 2000-14, valid 2015-17, test 2018-22), district
folds and the 33 tabular features are the same as the main build, because the
copy calls the same `models/build_training_data.build()`.

## The six channels, and why each one is there

| # | channel | range | what it is | why the BiLSTM needs it |
|---|---|---|---|---|
| 0 | `rain_mm` | 0 to 1,679 | weekly rain total | the signal, unchanged from the transformer's input |
| 1 | `in_interval` | 0 to 1 | share of the week's days after the previous reading | `delta_h_m` is measured from the previous reading. The transformer can infer the interval from learned position plus the `days_since_prev` feature. An LSTM has no position input, and the boundary moves from row to row (median 14 weeks, max 53), so it is marked on the timeline itself |
| 2 | `doy_sin` | -1 to 1 | calendar position of the week's middle day | an LSTM sees order but not the calendar. The same 104 weeks begin in a different month for each campaign, so without this the model cannot tell monsoon weeks from winter weeks except by their rain |
| 3 | `doy_cos` | -1 to 1 | calendar position, cosine part | sin and cos together keep December next to January |
| 4 | `rain_anom_mm` | -389 to 1,349 | weekly rain minus that well's normal for those calendar days | the target responds to rain *against what the well usually gets*. The tabular arm already has season and annual anomalies; this puts them at weekly resolution. The normal uses **training years 2000-2014 only**, the same rule as every other normal, with a 31-day smoothing because 15 years per calendar day is noisy |
| 5 | `wet_frac` | 0 to 1 | share of the week's days with at least 2.5 mm (IMD's rain-day threshold) | a weekly total does not say whether 70 mm fell in one storm (mostly runoff) or over seven days (more infiltration). The tabular arm has `rain_days_interval` and `heavy_days_interval` for the interval only; this covers the full two years |

No channel uses a water level, a year, a location or an identifier, so the
leakage rule holds. The build checks that `in_interval` adds up to
`days_since_prev` for every row, exactly.

Scaling happens at training time (`models/train_bilstm.py`), with statistics
from training rows only:
- `rain_mm` is log1p, then standardised.
- `rain_anom_mm` is sign * log1p(|x|), then standardised.
- The other four channels are already in [-1, 1] and are left as they are.

## Why a separate copy and not a change to the main build

1. **Reproducibility.** The transformer numbers in `README.md` and `models/RESULTS.md`
   were produced on `data/training/`. Editing that build would break them.
2. **A fair comparison.** Every row, target and tabular feature is the same, so a
   difference between the models comes from the sequence encoder and its input.
3. **Separating architecture from data.** `train_bilstm.py` also trains the BiLSTM on
   `rain_mm` alone (`bilstm 1ch`), which is the transformer's exact input:
   - transformer vs `bilstm 1ch`: what the architecture changes.
   - `bilstm 1ch` vs `bilstm`: what the five extra channels add.

## Training

Locally, smoke test (about 3 minutes on CPU):

```bash
cd models && python train_bilstm.py --epochs 2 --limit 20000 --out ../bilstm-data/out/smoke
```

The full run takes about 3 minutes per epoch per model on CPU, so it goes to
Kaggle, on a T4 GPU:

```bash
python kaggle/push.py            # private dataset (out/ + model code) and private GPU kernel
kaggle kernels status parthrainchwar/rain-or-pumps-bilstm
python kaggle/push.py --fetch    # results to bilstm-data/out/kaggle_run/
```

The kernel (`kaggle/run_bilstm.py`) runs two things:
- The year split: baselines, `bilstm`, `bilstm 1ch` and `bilstm seq only`, 25 epochs.
- The 5 district folds, 20 epochs.

## Model: `models/bilstm.py`

- The input projection maps the 6 channels to 32.
- A 2-layer bidirectional LSTM with 64 hidden units per direction reads the sequence.
- Attention pooling over the 104 weeks produces one vector.
- The transformer's `TabEncoder` is reused for the tabular features.
- An MLP head produces the prediction.

About 183k parameters, against the transformer's 177k. Huber loss and the same
`run_nn` training loop, optimiser and early stopping as the transformer.

The training script also reports:
- **Rain sensitivity.** The mean change in prediction when every week's rain is
  scaled by 0.8 and by 1.2. More rain should give a smaller fall, as the what-if
  tool requires.
- **Pooling map.** `pooling_by_season.png`, comparable to the transformer's
  attention figure.
- **Residual table.** `residual_by_district.csv`, with the same sign convention
  as the transformer's.
