# archive/

The first modelling pipeline, superseded by `pipeline/` and `models/` but kept
because parts of it are still needed and none of it is reproducible from what
replaced it.

| Path | Why it is still here |
|---|---|
| `ml/bits_ml/api.py` | FastAPI endpoint. Week 7 of the plan asks for one; this is the starting point |
| `ml/bits_ml/whatif.py` | The plus or minus 20 percent rainfall scenario function, also week 7 |
| `ui/atlas/` | The district watchlist dashboard, week 8. `groundwater-atlas.html`, `atlas_app.js`, `assemble.py`, `build_payload.py` |
| `ml/bits_ml/attribution.py` | Residual attribution and district ranking against the CGWB stress categories |
| `ml/bits_ml/{dataset,splits,targets,models,train,evaluate}.py` | The earlier training stack, built on the 2,759-well extract and the `anomaly_m` target |
| `ml/bits_ml/{groundwater,rain_model,recharge,decisions}.py` | Earlier domain code |
| `ml/tests/` | Eight test files covering the above |
| `notebooks/` | Week 1 notebook variants and the Colab training notebook |
| `docs/` | `model_report.html`, `regression_brief.html` |

Two things to know before reusing any of it.

**The target changed sign and meaning.** This code targets `anomaly_m`, positive
when the level is deeper than normal. The current target is `delta_h_m`, the change
since the previous reading, positive when the level fell. They are not
interchangeable.

**The data layer underneath it is gone.** It reads the old `dataset.py` tables, not
`data/training/`. Anything lifted from here needs repointing at the current
contract in `models/START_HERE.md`.
