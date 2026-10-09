# Rain or Pumps?

**Attributing India's Groundwater Decline District by District**

BITS Pilani PGCP AI and ML capstone, Group 18
Kakara Siva Kumar · Rainchwar Parth · Vipul Aggarwal · Mentor: Sudharshan Deshmukh

**Live app: [rain-or-pumps.vercel.app](https://rain-or-pumps.vercel.app/#forecast)**
([Past rain](https://rain-or-pumps.vercel.app/#replay) ·
[What if it rains](https://rain-or-pumps.vercel.app/#whatif) ·
[Future](https://rain-or-pumps.vercel.app/#forecast) ·
[Pumping hotspots](https://rain-or-pumps.vercel.app/#pressure) ·
[API docs](https://rain-or-pumps.vercel.app/docs))

---

## The idea

Groundwater levels across India are falling, but rainfall varies naturally from year
to year, so a decline can come from a run of dry years, from pumping, or from both.
We model where the water level *should* be given the rain that actually fell, and
treat the part of the decline that rainfall does not explain as a proxy for
extraction pressure. Extraction is never an input. It is the output.

The result is a map application for all of India: replay the rain, ask what changed
rain on any past or future day would do to the groundwater, forecast water levels to
2045, and rank the districts falling faster than rain explains.

## Quick start

All the data is in the repository: the raw inputs in `data/` and the app's built
data in `simulator/ui/data/`.

```bash
pip install -r requirements.txt     # the app: FastAPI, ONNX runtime, pandas
make api                            # http://localhost:8000, API docs at /docs
```

`requirements-train.txt` adds what rebuilding the data, training the models, the
notebooks and the tests need (PyTorch, LightGBM, pyarrow, Jupyter, pytest).
`make app-data` rebuilds the app's data from `data/` (about 20 minutes).

## Deploying on Vercel

Deployed at **https://rain-or-pumps.vercel.app**. `vercel.json` deploys the app as
one FastAPI service (`main.py`) with its committed data, so there is no build step:
import the repository in Vercel, or run `vercel deploy`. It installs `requirements.txt` only and allows 60 s per request for
the live what-if runs.

## Which model, and why

Three model families were trained on the same rows, target `delta_h_m` (change in
depth since the previous reading; positive = the water level fell). Held-out test
years 2018-2022:

| model | test R² | test MAE (m) | rain response usable for what-if? |
|---|---|---|---|
| LightGBM on 33 hand-built features | **0.484** | **1.427** | no: every edit means rebuilding its rain features, and no guarantee on direction |
| Transformer, weekly rain + well facts | 0.439 | 1.457 | no: about 1 cm, sign not reliable |
| BiLSTM, weekly rain + well facts | 0.441 | 1.449 | no: about 1 cm, sign flips between variants |
| **BiLSTM simulator** (rain-response penalty) | 0.453 | 1.439 | **yes: right direction for 98% of wells** |

All models make nearly the same errors (correlation 0.96), so the ceiling of about
R² 0.45-0.48 comes from the data, not the model: roughly half the change is not
explained by rain and fixed well facts, which is the gap the project is about. The
simulator gives up 0.03 R² against LightGBM and is the only model whose answer to
"what if it rained more?" can be trusted, so it is the model the application uses.
Details: [models/MODEL_REVIEW.md](models/MODEL_REVIEW.md),
[models/BILSTM_RESULTS.md](models/BILSTM_RESULTS.md), [simulator/README.md](simulator/README.md).
A random (well-grouped) split puts all three at R² 0.49-0.51: see `notebooks/03`-`06`.

## Repository layout

| Folder | What is in it |
|---|---|
| `data/` | The raw inputs: CGWB wells, the packed IMD rain cube, district outlines. See `data/README.md` |
| `models/` | Training-table and sequence builders, the three model families, their results and write-ups |
| `main.py`, `vercel.json` | The app's entrypoint and the Vercel deployment |
| `simulator/` | The chosen model (`train_sim.py`, `artifacts/`), the FastAPI service (`api/`), forecast and what-if engine (`forecast/`), map layers (`geo/`) and the UI (`ui/`) |
| `kaggle/` | Runs training on a Kaggle GPU |
| `notebooks/` | Exploration and the random-split comparison of the three models |
| `reports/` | Figures and tables from the model runs |
| `scripts/` | Data fetching, the rain cube builder, the app data build |

## The data

| Source | What it is | Licence |
|---|---|---|
| CGWB groundwater levels | Depth to water at 2,759 monitoring wells, four readings a year, 2000-2022 | CC BY 4.0, figshare doi 10.6084/m9.figshare.29293877.v3 |
| IMD gridded rainfall | Daily rainfall on a 0.25 degree grid, 1998-2022, 4,964 land cells | Open access, imdpune.gov.in |
| Map layers | District outlines, HydroRIVERS, GeoDAR dams, GeoNames towns | see `simulator/README.md` |

Rainfall reaches each well by masked bilinear interpolation over the four
surrounding grid nodes. Behind every reading the model sees 104 weeks of rain, each
week's place in the calendar, and the rain against that well's normal for those days.

## The one rule that matters

Never a feature, enforced by an assertion that fails the build: any past water level,
the year, latitude, longitude, district, state, well id. With the previous depth as
an input a model reaches R² near 0.9 by learning mean reversion, and the residual
stops meaning anything. Rainfall normals use training years only. The split is
blocked by year: train 2000-2014, validation 2015-2017, test 2018-2022.

## Attribution

CGWB quality-controlled groundwater level dataset with specific yield over India,
figshare, doi 10.6084/m9.figshare.29293877.v3, CC BY 4.0.

Pai, D.S. et al. (2014). Development of a new high spatial resolution (0.25 x 0.25)
long period (1901-2010) daily gridded rainfall data set over India. *MAUSAM* 65(1), 1-18.

District outlines: post-2020 vintage, 724 districts in 36 states and union territories.
