# ui — the Groundwater Atlas

One page, no service behind it: district outlines, a year-by-year timeline, and scenario predictions chosen from dropdowns.

```bash
cd ui/atlas
python build_payload.py     # joins boundaries to districts, gathers timelines and scenarios
python assemble.py          # folds payload + app into one file
start groundwater-atlas.html        # macOS/Linux: open groundwater-atlas.html
```

Rebuild the payload whenever the pipeline changes — it reads `data/processed/district_ranking.csv`, `district_whatif.csv`, the CGWB readings and the IMD rain table. `assemble.py` alone is enough after an edit to the page itself.

## What the dropdowns do

| Control | Effect |
|---|---|
| **Show** | Water depth, monsoon rain, annual rain, rate of decline, how much rain soaks in, scenario gain, or gain against decline |
| **Year** | 2000–2022, with Play to run the timeline; applies to the three measured views |
| **Monsoon** | −20% to +30% of each district's own normal, driving the two scenario views and the panel |
| **District** | Jumps to a district; clicking the map does the same |

Scenario numbers are **precomputed**, not live: the LightGBM model of the November level anomaly is run for every district at every rainfall level by `build_payload.py`, and the page reads the result. That is what keeps it a single static file with no API.

The page's wording is deliberately plain — "how deep the water is" rather than "water table depth, mbgl", "how fast the water is dropping" rather than "Theil–Sen trend". The statistics behind each phrase are unchanged and are named in `docs/model_report.html`.

## Files

| File | Role |
|---|---|
| `atlas/build_payload.py` | Joins CGWB districts to boundary polygons, assembles timelines and the scenario grid |
| `atlas/atlas_shell.html` | Markup, tokens and layout |
| `atlas/atlas_app.js` | Map rendering, dropdown logic, timeline and scenario charts |
| `atlas/assemble.py` | Produces the single-file page |
| `atlas/districts.geojson` | District outlines, drawing only — no number comes from it |

## The district join

CGWB and the boundary file disagree on 34 of 272 district names. Three rules, in order: exact match inside the same state, close-spelling match inside that state (Puruliya/Purulia, Dakshin kannada/Dakshina Kannada), then an explicit alias table for renames no string metric should guess (Trichur/Thrissur, Cuddapah/Y.S.R. Kadapa, Mysore/Mysuru). That lands 238 exact, 18 by spelling, 16 by alias — **270 of 272 districts**. Allahabad and Praygraj are the same place under two CGWB spellings and are pooled onto one polygon. Anything unmatched is reported by the build and left off the map rather than guessed.

## Removed

`ui/src` (React dashboard), `ui/watchlist`, `ui/maps` and the npm scaffolding were earlier, separate UIs. They were deleted once the atlas replaced all three: the React app needed `uvicorn bits_ml.api:app` running beside it, and this page needs nothing at all.

The service itself stays in `ml/bits_ml/api.py`, because the brief asks for a FastAPI endpoint as a deliverable. It is simply not required to view the atlas.
