# old — the earlier pipeline, kept for reference

Everything here was built on the published 2,759-well file (`data/cgwb/…_ref_sy_2000_2022.csv`)
before the data layer was rebuilt from the raw archive. It is superseded by `data_cleaning/`
and `ml/`, and kept so earlier results and decisions can be traced.

| folder | what it was |
|---|---|
| `ml/bits_ml/` | the first pipeline: readings (`groundwater`), features (`dataset`, `rain_model`), models (ridge, LightGBM, MLP), evaluation, attribution, what-if, recharge, decisions, FastAPI `api` |
| `ml/scripts/` | evidence runs: controls, experiments, tuning, clusters, Colab bundle check, rainfall and groundwater-response checks |
| `ml/tests/` | tests for the modules above |
| `ml/models/` | trained LightGBM bundles (not in git) |
| `ml/README.md` | that pipeline's run order and findings (skill scores, the extraction proxy that did not reproduce) |
| `ui/` | the Groundwater Atlas: map, timeline and scenario dropdowns, built from the old model |
| `notebooks/` | Week-1 EDA and feature engineering; training on Colab or Kaggle |
| `docs/` | the model report and regression brief from the old models |
| `data/processed/` | every table and result the old pipeline wrote (not in git) |

**Why it was replaced.** The published file had dropped 29,540 wells through a biased completeness
rule, district names mixed spellings and vintages, specific yield was missing for most wells, and
IMD's missing values written as 0 mm had passed as dry years. See `data_cleaning/README.md`.

**Running it.** The code here no longer runs in place: its modules import each other as the
`bits_ml` package, which now holds only the new pipeline. To run it as it was, check out commit
`6012fc2` (the last commit before the move) in a separate worktree.
