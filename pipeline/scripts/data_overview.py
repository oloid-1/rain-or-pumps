"""The data as it stands, with real sample rows, and the plan for the models built on it.

Every sample row is read from the tables in data_cleaning/data at run time, so the
file never shows data that does not exist. One well (a dug well in Pune) is
followed through every table, so the rows tell one story: raw reading -> season
-> district -> training row.

Writes the same content three ways:
    data_cleaning/DATA_AND_MODEL_PLAN.txt   plain text
    data_cleaning/DATA_AND_MODEL_PLAN.md    Markdown
    README.md                               the repository README: the project, what the cleaning
                                            did, the data with samples, code snippets, the model plan

Run from ml/ after bits_ml.training_data:  python -m scripts.data_overview
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from bits_ml import training_data as td
from bits_ml.config import CORE_DIR, REPO_DIR

OUT_TXT = REPO_DIR / "data_cleaning" / "DATA_AND_MODEL_PLAN.txt"
OUT_MD = REPO_DIR / "data_cleaning" / "DATA_AND_MODEL_PLAN.md"
OUT_README = REPO_DIR / "README.md"
WELL = "18.1111_74.2028"
YEAR = 2019
W = 100

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}")

lines: list[str] = []
MD = False          # set by main(): the same build() writes plain text, then Markdown


def out(text: str = "") -> None:
    lines.append(text)


def para(text: str) -> None:
    text = " ".join(text.split())
    out(text if MD else textwrap.fill(text, W))
    out()


def title(text: str, top: bool = False) -> None:
    if MD:
        out(("# " if top else "## ") + text)
    else:
        out("=" * W)
        out(text.upper() if top else text)
        out("=" * W)
    out()


def sub(text: str) -> None:
    if MD:
        out("### " + text)
        out()
    else:
        out(text)
        out("-" * W)


def cell(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–" if MD else "NaN"
    if isinstance(v, (bool, np.bool_)):
        return ("yes" if v else "no") if MD else str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10_000 else str(v)          # 32,299 but not 2,019
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    return str(v)


def table(frame: pd.DataFrame, header: bool = True) -> None:
    """Text columns left-aligned, numbers right-aligned, in either format."""
    if MD:
        numeric = [pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
                   for c in frame.columns]
        names = [str(c) for c in frame.columns] if header else [" " for _ in frame.columns]
        out("| " + " | ".join(names) + " |")
        out("|" + "|".join("---:" if n else ":---" for n in numeric) + "|")
        for row in frame.itertuples(index=False):
            out("| " + " | ".join(cell(v).replace("|", "\\|") for v in row) + " |")
        out()
        return
    shown = frame.copy()
    for column in shown.columns:
        col = shown[column]
        if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
            texts = shown[column].astype(str)
            width = max(texts.str.len().max(), len(str(column)) if header else 0)
            shown[column] = texts.str.ljust(width)
            shown = shown.rename(columns={column: str(column).ljust(width)})
    text = shown.to_string(index=False, header=header)
    for row in text.splitlines():
        out("  " + row.rstrip())
    out()


def code_block(block: list[str]) -> None:
    if MD:
        out("```")
    for line in block:
        out(line if MD else "  " + line)
    if MD:
        out("```")
    out()


def rows(name: str) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(CORE_DIR / f"{name}.parquet").metadata.num_rows


def dataset(name: str, grain: str, role: str, frame: pd.DataFrame, note: str = "") -> None:
    if MD:
        out(f"#### `{name}`")
        out()
        out(f"*{rows(name):,} rows · one row per {grain}*")
        out()
    else:
        sub(f"{name}   ({rows(name):,} rows; one row per {grain})")
    para(role)
    table(frame)
    if note:
        para(note)


def build(readme: bool = False) -> None:
    wells = pd.read_parquet(CORE_DIR / "well_master.parquet")
    me = wells[wells["well_uid"] == WELL].iloc[0]
    seasons = pd.read_parquet(CORE_DIR / "dim_season.parquet")
    year_seasons = seasons[seasons["season_year"] == YEAR]["season_idx"].tolist()
    monsoon = seasons[(seasons["season_year"] == YEAR) & (seasons["season"] == "May-Aug")]["season_idx"].item()

    if not readme:
        title("Rain or Pumps? — the data we have, with sample rows, and how the models will be built", top=True)
    if readme:
        title("Part A — The data")
    para(f"""
        Part A walks through every dataset, with real rows read from data_cleaning/data. One well is
        followed through all of them: {WELL}, a dug well at {me['station']} in {me['district']}
        district, {me['state']} (lat {me['lat']}, lon {me['lon']}), one of the 2,759 published wells.
        Its rows show how a raw reading becomes a season, a district figure and a training row. Part B is
        the plan for the models. How the data was cleaned is in DATA_CLEANING_SUMMARY.txt; every column is
        described in data/core_dictionary.csv.
    """)
    if not readme:
        para("Generated by ml/scripts/data_overview.py; re-run it after any rebuild and the samples update.")
    if MD and not readme:
        out("**Contents**")
        out()
        out("- [Part A — The data](#part-a--the-data): raw inputs · wells · grid and time · rain · "
            "rain and groundwater together · training data")
        out("- [Part B — How the models will be built](#part-b--how-the-models-will-be-built): question · "
            "models · training · comparison · attribution · what-if · limits · order of work")
        out()
        out("Related: [how the data was cleaned](DATA_CLEANING_SUMMARY.txt) · [gaps and fixes](README.md) · "
            "[every dataset](DATASETS.md) · [modelling gaps](MODELLING_GAPS.md)")
        out()

    # ================================================================== A
    if not readme:
        title("Part A — The data")

    sub("A0. The raw inputs (not changed; everything below is built from them)")
    table(pd.DataFrame([
        ("CGWB raw archive", "32,299 wells x 100 reading columns (Jan/May/Aug/Nov 2000-2024), depth in m below ground"),
        ("CGWB published set", "2,759 quality-controlled wells with specific yield (kept as the control group)"),
        ("Hydrogeological map", "0.12 degree grid, 5 specific-yield classes by rock type"),
        ("IMD daily rainfall", "25 yearly NetCDF files, 129 x 135 grid at 0.25 degree, mm/day, 1998-2022"),
        ("District outlines", "724 districts, 36 states/UTs, post-2020 boundaries (GeoJSON)"),
    ], columns=["input", "what it is"]))

    # ---------------------------------------------------------- wells
    sub("A1. Wells")
    dataset("well_master", "well (all 32,299)",
            """The well register: who each well is, where it stands (district and state from its
            coordinates; the file's own labels kept as `cgwb_*`), how it is built, its specific yield and
            rock, quality counts, and which views it belongs to. Filter on a view before analysis.""",
            wells[wells["well_uid"].isin([WELL, "10.0011_78.7694"]) | (wells["state"] == "Kerala")].head(3)[
                ["well_uid", "district", "state", "cgwb_district", "well_type", "aquifer", "aquifer_source",
                 "well_depth_m", "sy", "sy_source", "rock_class", "n_valid", "view_strict", "view_modelling"]])

    obs = pd.read_parquet(CORE_DIR / "well_obs.parquet")
    mine = obs[(obs["well_uid"] == WELL) & (obs["year"] == YEAR)]
    flagged = obs[obs["flag_below_bottom"]].head(1)
    dataset("well_obs", "water-level reading (flagged, never dropped)",
            """Every reading in the raw file, with six quality flags and `valid`. The four 2019 readings of
            our well, then one reading elsewhere that fails a check (water reported below the bottom of
            the well):""",
            pd.concat([mine, flagged])[["well_uid", "campaign", "date", "depth_mbgl", "flag_negative",
                                        "flag_zero_suspect", "flag_below_bottom", "flag_outlier",
                                        "flag_repeat_run", "valid"]],
            "Depth is metres below ground, so a larger number means deeper water. May is deepest (end of "
            "the dry season), August shallowest (after the monsoon's first half).")

    # ---------------------------------------------------------- grid & time
    sub("A2. Grid, time and the links between them")
    cells = pd.read_parquet(CORE_DIR / "dim_cell.parquet")
    bridge = pd.read_parquet(CORE_DIR / "bridge_well_cell.parquet")
    my_cells = bridge[bridge["well_uid"] == WELL]
    home = my_cells.loc[my_cells["is_home"], "cell_id"].item()
    dataset("dim_cell", "IMD grid cell (land or sea)",
            "The spatial frame: every 0.25 degree cell, its centre, whether IMD reports rain there, how "
            "many wells stand in it, and the district covering most of it. Our well's cell:",
            cells[cells["cell_id"] == home][["cell_id", "cell_lat", "cell_lon", "is_land", "n_wells",
                                            "n_wells_modelling", "district", "state", "n_districts"]])
    dataset("dim_season", "season",
            "The time frame the models use: four seasons between readings, cut on the 15th. The four "
            f"seasons of {YEAR}:",
            seasons[seasons["season_year"] == YEAR][["season_idx", "season_year", "season", "season_label",
                                                     "campaign", "start", "end", "n_days", "water_year"]])
    dataset("bridge_well_cell", "well x grid cell it takes rain from",
            "Where a well's rain comes from: the four grid points around it, weighted by distance "
            "(weights sum to 1). Our well sits near a corner, so its rain mixes four cells:",
            my_cells[["well_uid", "cell_id", "weight_land", "is_home", "distance_km"]])
    area = pd.read_parquet(CORE_DIR / "bridge_cell_district.parquet")
    dataset("bridge_cell_district", "land cell x district it overlaps",
            "How grid cells are shared between districts, by area. District rain is the area-weighted "
            "mean of its cells. Three of Pune's 36 cells:",
            area[area["district"] == "Pune"].sort_values("frac_of_cell").iloc[[0, 15, 35]][
                ["cell_id", "district", "state", "area_km2", "frac_of_cell", "weight_in_district"]])

    # ---------------------------------------------------------- rain
    sub("A3. Rain")
    cm = pd.read_parquet(CORE_DIR / "fact_cell_month.parquet")
    sample = cm[(cm["cell_id"] == home) & (cm["month"].dt.year == YEAR) & cm["month"].dt.month.isin([5, 6, 7, 8])]
    zero = cm[cm["suspect_zero"]].head(1)
    dataset("fact_cell_month", "grid cell x month",
            "Rain per cell per month, the base all rain is read from. Our well's cell, May-August "
            f"{YEAR}, then one cell-month where IMD wrote 0 mm for missing data (flagged, set missing):",
            pd.concat([sample, zero])[["cell_id", "month", "rain_mm", "wet_days", "max_day_mm", "suspect_zero"]])
    cs = pd.read_parquet(CORE_DIR / "fact_cell_season.parquet")
    dataset("fact_cell_season", "grid cell x season",
            "Rain per cell per season, summed from the daily grid (the season boundary falls mid-month). "
            "rain_pct_normal compares with the 2000-2014 normal for that season:",
            cs[(cs["cell_id"] == home) & cs["season_idx"].isin(year_seasons)][
                ["cell_id", "season_idx", "rain_mm", "wet_days", "max_day_mm", "rain_normal_mm",
                 "rain_pct_normal", "complete"]])

    # ---------------------------------------------------------- joined
    sub("A4. Rain and groundwater together")
    ws = pd.read_parquet(CORE_DIR / "fact_well_season.parquet")
    dataset("fact_well_season", "well x season (8,629 modelling wells)",
            f"""The main joined table. For each season: rain at the well, the level at the reading that
            closes the season, the change since the previous reading, and that change in millimetres of
            water. Our well through {YEAR}:""",
            ws[(ws["well_uid"] == WELL) & ws["season_idx"].isin(year_seasons)][
                ["season", "rain_mm", "rain_pct_normal", "depth_mbgl", "depth_start_m", "delta_h_m",
                 "span_seasons", "storage_change_mm", "rain_span_mm", "recharge_ratio"]],
            f"""Read the May-Aug row: {ws.loc[(ws['well_uid'] == WELL) & (ws['season_idx'] == monsoon), 'rain_mm'].item():,.0f}
            mm of rain fell, the water rose {-ws.loc[(ws['well_uid'] == WELL) & (ws['season_idx'] == monsoon), 'delta_h_m'].item():.1f} m
            (delta_h negative means rising), and recharge_ratio says what share of that rain reached the
            aquifer. The Nov-Jan row bridges a missing reading: span_seasons is 2, so its change and its rain
            both run from the last valid reading, in August 2018.""")
    ds_ = pd.read_parquet(CORE_DIR / "fact_district_season.parquet")
    dataset("fact_district_season", "district x season",
            "The same at district scale: rain area-weighted over every cell of the district (so it exists "
            "for all 720 districts the grid reaches), wells summarised with their count. Pune:",
            ds_[(ds_["district"] == "Pune") & ds_["season_idx"].isin(year_seasons)][
                ["district", "season", "rain_mm", "rain_pct_normal", "n_wells_read", "anomaly_median_m",
                 "delta_h_median_m", "storage_change_median_mm", "recharge_ratio_median"]])
    cf = pd.read_parquet(CORE_DIR / "fact_cell_season_full.parquet")
    dataset("fact_cell_season_full", "grid cell x season",
            "The same at grid-cell scale, matching the rain grid exactly; n_wells_read says how many wells "
            "stand behind each row (31% of cell-seasons rest on one):",
            cf[(cf["cell_id"] == home) & cf["season_idx"].isin(year_seasons)][
                ["cell_id", "season", "rain_mm", "n_wells_read", "anomaly_median_m", "delta_h_median_m",
                 "storage_change_median_mm"]])
    para("""Monthly versions exist too (fact_well_month, fact_cell_month_full, fact_district_month): rain
         every month, a level in the four reading months only. The models use the season tables.""")

    # ---------------------------------------------------------- training
    sub("A5. Training data (data_cleaning/data/training/)")
    tab = pd.read_parquet(CORE_DIR / "training" / "tabular.parquet")
    spec = pd.read_csv(CORE_DIR / "training" / "feature_spec.csv")
    para(f"""tabular.parquet: {len(tab):,} rows, one per well x season with a target, from
         {tab['well_uid'].nunique():,} wells in {tab['state'].nunique()} states (three of four campaigns
         required, which keeps Kerala, Assam, West Bengal and Odisha). Split by year: train 2000-2014
         {int((tab['split'] == 'train').sum()):,}, valid 2015-2017 {int((tab['split'] == 'valid').sum()):,},
         test 2018-2022 {int((tab['split'] == 'test').sum()):,}; plus five folds of whole districts.""")
    if MD:
        table(spec.groupby("role", sort=False)["column"].agg(lambda c: ", ".join(f"`{x}`" for x in c))
              .reset_index().rename(columns={"column": "columns"}))
    else:
        for role, cols in spec.groupby("role", sort=False)["column"]:
            wrapped = textwrap.wrap(", ".join(cols), W - 16)
            out(f"  {role:<12}  {wrapped[0]}")
            for line in wrapped[1:]:
                out(f"  {'':<12}  {line}")
        out()
    row = tab[(tab["well_uid"] == WELL) & (tab["season_idx"] == monsoon)].iloc[0]
    sub(f"One training row: our well, May-Aug {YEAR} (split = {row['split']})")
    shown = spec[spec["role"].isin(["target", "feature"])]
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.3f}".rstrip("0").rstrip(".")
        return str(v)
    table(pd.DataFrame({"column": shown["column"], "role": shown["role"],
                        "value": [fmt(row[c]) for c in shown["column"]]}))
    seq = np.load(CORE_DIR / "training" / "sequence_cells.npz", allow_pickle=True)
    stencil = pd.read_parquet(CORE_DIR / "training" / "stencil.parquet")
    weeks = td.well_sequences(tab[(tab["well_uid"] == WELL) & (tab["season_idx"] == monsoon)],
                              seq["weeks"], seq["cell_ids"], stencil)[0]
    sub(f"The same row as a rain sequence for a recurrent model: 52 weekly totals (mm), ending {YEAR}-08-15")
    code_block(["weeks {:>2}-{:>2}: ".format(k + 1, k + 13) + " ".join(f"{v:5.0f}" for v in weeks[k:k + 13])
                for k in range(0, 52, 13)])
    para(f"Total {np.nansum(weeks):,.0f} mm over 52 weeks; the table's rain over the last 4 seasons is "
         f"{row['rain_4s_mm']:,.0f} mm (365 days against 364).")

    if readme:
        snippets(features_hint=True)

    # ================================================================== B
    title("Part B — How the models will be built")

    sub("B1. The question each model answers")
    para("""Given the rain that fell at a well, how far should its water level have moved this season?
         Rain is always an input and always observed rain; groundwater change is the output. Rain is never
         a target: it is already measured at every well, and it cannot be forecast from its own past (tested:
         a model reading 30 days of rain falls back to the calendar average within a week). "Future"
         groundwater is asked with scenario rain instead (B6).""")
    table(pd.DataFrame([
        ("rows", "tabular.parquet: one per well x season with a valid change"),
        ("target", "delta_h_m: depth now minus depth at the previous reading (m; + = water fell)"),
        ("alternatives", "storage_change_mm (the same in mm of water); anomaly_ref_m (level vs the well's normal)"),
        ("inputs", "25 features: rain this season and before, 1-3 year totals, % of normal, well properties"),
        ("never inputs", "past water level, year, coordinates, district, state (they carry the pumping history)"),
    ], columns=["key", "value"]), header=False)

    sub("B2. The models: a baseline and two architectures, on identical rows and splits")
    for name, arch, what in [
        ("0. No-skill baselines", "none",
         "predict 'no change', and each well's average change for that season in the training years; "
         "every model must beat both"),
        ("1. Ridge regression", "A: many inputs, one output",
         "the required linear baseline; scaled features, one-hot categories, NaN filled from train"),
        ("2. LightGBM", "A: many inputs, one output",
         "gradient-boosted trees on the 25 features; handles NaN; monotone constraint so that more rain "
         "never predicts deeper water"),
        ("3. LSTM (or GRU)", "B: recurrent",
         "reads the 52-week rain sequence; well properties and season join after the recurrent layer; "
         "same target"),
    ]:
        if MD:
            out(f"- **{name}** — *{arch}*: {what}")
        else:
            out(f"  {name}   [{arch}]")
            for line in textwrap.wrap(what, W - 6):
                out(f"      {line}")
    out()
    para("""Why these two architectures: A summarises rain into numbers a model can weigh (how much, how
         unusual, how long a run of wet or dry years); B reads the timing directly (one heavy week versus
         scattered showers). If B cannot beat A, the timing inside a season adds little beyond the totals,
         which is itself a result for the report.""")

    sub("B3. Training procedure")
    table(pd.DataFrame([
        ("1", "fit on train (2000-2014); tune on valid (2015-2017); open test (2018-2022) once, at the end"),
        ("2", "scale inputs (ridge, LSTM) with statistics from train only"),
        ("3", "loss: squared error, with Huber loss tried because 0.24% of changes exceed +/-20 m"),
        ("4", "LSTM: 52 weeks x 1 rain channel; 1-2 layers, 32-64 units; early stopping on valid"),
        ("5", "LightGBM: early stopping on valid; a small search over depth, leaves and learning rate"),
        ("6", "repeat on the five district folds: train on four folds' districts, test on the fifth"),
    ], columns=["step", "what"]), header=False)

    sub("B4. How they are compared")
    table(pd.DataFrame([
        ("RMSE, MAE (m)", "size of the error in metres"),
        ("R2, skill", "share of variation explained; skill = 1 - error / error of the no-skill baseline"),
        ("per season", "Jan-May, May-Aug, Aug-Nov, Nov-Jan reported separately, never one pooled number"),
        ("two hold-outs", "unseen years (test) and unseen districts (district folds)"),
        ("stable wells", "also scored on wells present in both train and test (8,053 wells remain by 2022)"),
        ("monotonicity", "rain scaled 0.8x .. 1.2x must never make predicted water deeper"),
        ("sanity", "May-Aug predicted storage should fall near CGWB's 8-15% infiltration range"),
    ], columns=["measure", "what it shows"]))

    sub("B5. From predictions to the project's answer (attribution)")
    para("""The chosen model gives, for every well and season, the change rain alone would explain
         ("rain-expected"). Observed minus rain-expected is the part rain does not explain. Summed over the
         years and over a district's wells, a gap that keeps growing is the extraction-pressure signal. Each
         block of years is predicted by a model that did not train on it, so the gap is never the model
         scoring itself. Districts are then ranked, and the ranking checked for stability (two halves of the
         record should agree) and against CGWB's official Safe / Semi-critical / Critical / Over-exploited
         categories (a dataset still to be added).""")

    sub("B6. The what-if simulator")
    para("""Question: if a district gets 20% more monsoon rain, does the water come back? The model is run
         twice on the same wells and years: once with observed rain, once with the May-Aug and Aug-Nov rain
         scaled by 1.2 (every rain feature and every weekly value recomputed consistently). The difference is
         the recovery rain could buy. A district where even +20% leaves the water falling has a structural
         decline. The rain report shows +20% is a real but uncommon monsoon, about 1 year in 5, so the
         scenario is a fair test. Scenario rain is chosen, not forecast, which is what makes it reliable.""")

    sub("B7. Expectations and limits")
    para("""Rain explains a modest share of the change: season rain correlates -0.08 to -0.16 with it
         across the four seasons. That is expected; the unexplained part is exactly what the project
         measures. The ceiling is set by the inputs, not the architecture: pumping, irrigation, cropping and
         canal data are not in the dataset. Specific yield is a five-class map value, so storage in mm is
         approximate; the change in metres does not depend on it. 17% of rows span more than one season
         (span_seasons is an input). Wells thin out over time and in the desert and the wettest hills.""")

    sub("B8. Order of work")
    table(pd.DataFrame([
        ("1", "baselines and ridge on tabular.parquet; per-season scores"),
        ("2", "LightGBM with monotone constraints; compare with ridge"),
        ("3", "LSTM on the weekly sequences; compare with LightGBM on the same rows"),
        ("4", "choose the model; district-fold results; error analysis by season, state, well type"),
        ("5", "cross-fitted rain-expected change -> district gap -> ranking; stability checks"),
        ("6", "what-if runs (+/-10%, +/-20% monsoon); CGWB category check once the data is added"),
    ], columns=["step", "what"]), header=False)


def snippets(features_hint: bool = False) -> None:
    """Code for using the data. Every snippet here was run against the tables before publishing."""
    sub("A6. Using the data: code snippets")
    para("Run from `ml/` with the project environment. Paths are relative to `ml/`.")
    out("**Read any table, and follow one well through the seasons**")
    out()
    out("```python")
    out('''import pandas as pd

D = "../data_cleaning/data/"
wells = pd.read_parquet(D + "well_master.parquet")
ws = pd.read_parquet(D + "fact_well_season.parquet")

me = ws[ws.well_uid == "18.1111_74.2028"]
me[["season_year", "season", "rain_mm", "depth_mbgl", "delta_h_m", "storage_change_mm"]].tail(8)

# district rain and wells, one season type across years
dist = pd.read_parquet(D + "fact_district_season.parquet")
dist.query("district == 'Pune' and season == 'May-Aug'")[["season_year", "rain_mm", "n_wells_read", "delta_h_median_m"]]''')
    out("```")
    out()
    out("**Only valid readings, only one view of wells**")
    out()
    out("```python")
    out('''obs = pd.read_parquet(D + "well_obs.parquet")
valid = obs[obs.valid]                                   # drops the six flagged problems
model_wells = wells.loc[wells.view_modelling, "well_uid"]  # or view_strict for the published 2,759''')
    out("```")
    out()
    out("**Training inputs for architecture A (ridge, LightGBM)**")
    out()
    out("```python")
    out('''import numpy as np
T = D + "training/"
tab = pd.read_parquet(T + "tabular.parquet")
spec = pd.read_csv(T + "feature_spec.csv")
features = spec.loc[spec.role == "feature", "column"].tolist()      # never past levels, year or place
categorical = ["season", "aquifer", "well_type", "rock_class"]
numeric = [f for f in features if f not in categorical]

train, valid = tab[tab.split == "train"], tab[tab.split == "valid"]   # test stays closed until the end
y_train, y_valid = train["delta_h_m"], valid["delta_h_m"]''')
    out("```")
    out()
    out("**Baselines, ridge and LightGBM**")
    out()
    out("```python")
    out('''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

rmse = lambda y, p: mean_squared_error(y, p) ** 0.5

# baseline: each well's average change for that season in the training years
mean_change = train.groupby(["well_uid", "season"])["delta_h_m"].mean().rename("base")
baseline = valid.join(mean_change, on=["well_uid", "season"])["base"].fillna(0)

ridge = make_pipeline(
    ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]),
    Ridge(alpha=1.0),
).fit(train[features], y_train)

as_lgb = lambda f: f[features].astype({c: "category" for c in categorical})
rain_fell = [f for f in numeric if f.startswith("rain_")
             and f not in ("rain_normal_mm", "rain_normal_annual_mm", "rain_cover")]
monotone = [-1 if f in rain_fell else 0 for f in features]   # more rain never predicts deeper water
lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                         monotone_constraints=monotone, verbose=-1)
lgbm.fit(as_lgb(train), y_train, eval_set=[(as_lgb(valid), y_valid)],
         callbacks=[lgb.early_stopping(50, verbose=False)])

for name, pred in [("no change", np.zeros(len(valid))), ("well-season mean", baseline),
                   ("ridge", ridge.predict(valid[features])), ("lightgbm", lgbm.predict(as_lgb(valid)))]:
    print(f"{name:<18} RMSE {rmse(y_valid, pred):.3f} m")''')
    out("```")
    out()
    out("**Rain sequences for architecture B (LSTM)**")
    out()
    out("```python")
    out('''from bits_ml import training_data as td

seq = np.load(T + "sequence_cells.npz", allow_pickle=True)
stencil = pd.read_parquet(T + "stencil.parquet")
weeks = td.well_sequences(train, seq["weeks"], seq["cell_ids"], stencil)   # (rows, 52) mm per week''')
    out("```")
    out()
    para("""A sketch of the recurrent model (planned, not yet run: it needs PyTorch, which is not in the
         environment yet):""")
    out("```python")
    out('''import torch, torch.nn as nn

class RainLSTM(nn.Module):
    def __init__(self, n_static, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, weeks, static):            # weeks: (batch, 52, 1), static: (batch, n_static)
        _, (h, _) = self.lstm(weeks)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)   # predicted delta_h_m

# weeks scaled with train statistics; static = scaled Sy, depth, normal rain + one-hot season, aquifer, ...
# loss: nn.HuberLoss(); early stopping on valid; same rows and splits as architecture A''')
    out("```")
    out()


def readme_head() -> None:
    wells = pd.read_parquet(CORE_DIR / "well_master.parquet")
    audit = pd.read_csv(CORE_DIR / "qc_audit.csv")
    tab = pd.read_parquet(CORE_DIR / "training" / "tabular.parquet", columns=["well_uid", "state", "split"])
    out("# Rain or Pumps?")
    out()
    out("**Attributing India's groundwater decline district by district** — BITS Pilani PGCP in AI & ML "
        "capstone, group 18.")
    out()
    para("""Groundwater is falling in many districts, but rainfall swings from year to year on its own. This
         project builds what the water level *should* be given the rain that actually fell, and treats the part
         that stays unexplained as the pressure people are putting on the aquifer. The end product answers one
         question per district: **if the monsoon were 20% stronger next year, would the water come back, or
         has the decline become structural?**""")
    out("> This README is generated by `ml/scripts/data_overview.py` from the data itself: every sample row "
        "below is real. Re-run it after a rebuild.")
    out()
    out("## Contents")
    out()
    for anchor, text in [("where-it-stands", "Where it stands"), ("repository-layout", "Repository layout"),
                         ("data-cleaning-what-was-done", "Data cleaning: what was done"),
                         ("part-a--the-data", "Part A — The data, with sample rows and code snippets"),
                         ("part-b--how-the-models-will-be-built", "Part B — How the models will be built"),
                         ("gaps-still-open", "Gaps still open"), ("rebuilding-everything", "Rebuilding everything"),
                         ("documentation-and-sources", "Documentation and sources")]:
        out(f"- [{text}](#{anchor})")
    out()

    title("Where it stands")
    table(pd.DataFrame([
        ("1. Data layer", "done", f"{len(wells):,} CGWB wells rebuilt from the raw archive; IMD rain cleaned; "
                                  "wells, grid cells and districts on one clock and one map"),
        ("2. Training data", "done", f"{len(tab):,} well-season rows from {tab['well_uid'].nunique():,} wells in "
                                     f"{tab['state'].nunique()} states; 25 features; 52-week rain sequences"),
        ("3. Models", "next", "ridge and LightGBM (architecture A) against an LSTM (architecture B), same rows"),
        ("4. Attribution, what-if", "after 3", "rain-expected change vs observed; district ranking; +20% monsoon"),
    ], columns=["stage", "status", "what"]))

    title("Repository layout")
    code_block([
        "data_cleaning/      the cleaned data and everything written about it",
        "  README.md           every gap in the raw data and the change it caused",
        "  DATASETS.md         every dataset and table, and its role",
        "  MODELLING_GAPS.md   the suggested modelling approach tested; training-data gaps",
        "  DATA_CLEANING_SUMMARY.txt   everything done, step by step",
        "  DATA_AND_MODEL_PLAN.md/.txt the content of this README as separate files",
        "  clean_gaps.ipynb    closes the gaps the pipeline leaves (specific yield, aquifer type)",
        "  reference/          district outlines",
        "  data/               cleaned tables, reports, training data (not in git; rebuilt)",
        "ml/                 the code: bits_ml/ (pipeline), scripts/ (reports), tests/",
        "docs/               problem statement, execution plan",
        "data/               raw inputs (not in git): CGWB archive, IMD NetCDF grids",
        "old/                the earlier pipeline, models, Atlas UI and notebooks (archived)",
    ])

    title("Data cleaning: what was done")
    para("""The two datasets were cleaned on their own terms, then aligned in space (same places), time (same
         periods) and unit (same quantity). Nothing was edited by hand; every table is rebuilt from the raw
         files. Full detail: [data_cleaning/DATA_CLEANING_SUMMARY.txt](data_cleaning/DATA_CLEANING_SUMMARY.txt)
         and [data_cleaning/README.md](data_cleaning/README.md).""")
    sub("1. The file we had been using was already filtered, and the filter was biased")
    para("""Replaying the source paper's quality control on the raw file reproduces its 2,759 wells exactly, and
         shows where the other 29,540 went:""")
    table(pd.DataFrame([
        ("raw file", 32299), ("has any reading 2000-22", 29537),
        ("no negative reading anywhere (drops the whole well)", 28612),
        ("**at least two readings in every year 2000-22**", 2876),
        ("no value repeating more than twice in a row", 2759),
    ], columns=["step", "wells left"]))
    para("""The completeness rule alone removes 25,736 wells, and the wells it removes deepen more often (46%
         against 38%), so the published set understates the decline. **Change:** start from the raw file, drop
         nothing, flag problems, and select wells only at the end, as views:""")
    table(pd.DataFrame([
        ("view_strict", "the published set, kept as a control", int(wells["view_strict"].sum())),
        ("view_modelling", "a usable normal in all four campaigns", int(wells["view_modelling"].sum())),
        ("view_training", "a usable normal in three of four (keeps Kerala, Assam, West Bengal, Odisha)",
         tab["well_uid"].nunique()),
    ], columns=["view", "rule", "wells"]))
    sub("2. Bad readings: flagged, never deleted")
    meaning = {
        "flag_negative": "water above ground (flowing well or sign error)",
        "flag_zero": "exactly 0.000 m (not invalid on its own)",
        "flag_zero_suspect": "a zero where zeros are over 1/5 of the well's record: missing written as 0",
        "flag_below_bottom": "water deeper than the well was drilled",
        "flag_outlier": "over 3 sigma from the well's own seasonal pattern",
        "flag_repeat_run": "three or more identical readings in a row",
    }
    flags = audit[audit["step"].isin(meaning)].assign(meaning=lambda f: f["step"].map(meaning))
    table(flags[["step", "meaning", "readings", "wells"]].rename(columns={"step": "flag"}))
    valid = audit.set_index("step").loc["valid", "readings"]
    total = audit.set_index("step").loc["readings read", "readings"]
    para(f"Valid: **{valid:,} of {total:,} readings ({valid / total:.1%})**.")
    sub("3. Everything else that was fixed")
    table(pd.DataFrame([
        ("Well identity", "station code damaged by scientific notation", "wells keyed by coordinates; shared sites suffixed"),
        ("District names", "610 names mixing spellings and boundary vintages",
         "district and state from the coordinates, against 724 post-2020 outlines"),
        ("State labels", "489 wells stand in another state than labelled",
         "levels follow the neighbours at the coordinates (r 0.39 vs 0.17): labels were wrong"),
        ("Specific yield", "missing in the raw file; first guessed from state medians",
         "read from the hydrogeological map (matches 100% of published values)"),
        ("Aquifer type", "blank for 13,312 wells", "dug wells -> Unconfined (99.6% are); the rest Unknown"),
        ("IMD -999", "sea and foreign land", "read as missing, never as 0 mm"),
        ("IMD zeros", "0 mm written for missing data: 669 cell-years, 109 cells",
         "8,693 cell-months flagged and set missing; averages use cells with data"),
        ("Rain at a well", "wells are points, rain is a grid", "four surrounding grid points, distance-weighted"),
        ("Rain per district", "cells cross district lines", "area-weighted; 720 of 724 districts get rain"),
        ("Time", "rain daily, levels four times a year", "four seasons cut on the 15th at the readings"),
        ("Unit", "level is a stock, rain a flow", "change in level x specific yield -> mm of water"),
        ("Missing readings", "some survey rounds reached 1% of wells", "never filled; changes bridge the gap"),
    ], columns=["area", "problem", "change"]))
    sub("4. Checks that were run")
    table(pd.DataFrame([
        ("rebuild reproduces the published 2,759 wells", "exact"),
        ("season rain vs raw NetCDF (incl. year boundary, leap year)", "within 0.00004 mm"),
        ("grid -> district mapping rebuilt from scratch vs stored", "identical"),
        ("Pune district rain by hand from raw NetCDF vs table", "within 0.00003 mm"),
        ("gridded land vs India's area", "3.25 vs 3.29 million km2"),
        ("hydrogeological map vs published specific yield", "100% of 2,872"),
        ("deficient monsoons found vs IMD's record (2002, 2004, 2009, 2014, 2015)", "all five match"),
        ("May-Aug share of rain stored vs CGWB's 8-15% infiltration range", "11.6%"),
        ("automated tests", "79 pass"),
    ], columns=["check", "result"]))


def readme_tail() -> None:
    title("Gaps still open")
    table(pd.DataFrame([
        ("rain explains little on its own", "season rain vs change: r -0.08 to -0.16",
         "expected: the rest is the signal; pumping, crops and canals are not in the data"),
        ("aquifer type unknown", "468 modelling wells", "kept as Unknown, never guessed"),
        ("reading day unknown", "all readings", "readings on the 15th; +/- 2 weeks"),
        ("thin survey rounds", "May 2020 and May 2021 at 1%", "left missing; bridged"),
        ("changes spanning several seasons", "17% of training rows", "span_seasons is an input"),
        ("wells drop out over time", "10,619 (2005) -> 8,053 (2022)", "also score on stable wells"),
        ("specific yield is a class", "five map values", "delta_h_m in metres is unaffected"),
        ("rain-gauge density changes", "IMD grid", "stated"),
    ], columns=["gap", "size", "handling"]))
    para("More in [data_cleaning/MODELLING_GAPS.md](data_cleaning/MODELLING_GAPS.md).")

    title("Rebuilding everything")
    para("""Put the CGWB archive in `data/`, the published CSV in `data/cgwb/` and the IMD yearly NetCDF files
         in `data/imd_rainfall/`. Then:""")
    out("```bash")
    out('''python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r ml/requirements.txt
cd ml
python -m bits_ml.ingest && python -m bits_ml.grid && python -m bits_ml.rain_panel
python -m bits_ml.panel && python -m bits_ml.seasons
# run data_cleaning/clean_gaps.ipynb top to bottom
python -m scripts.core_report && python -m scripts.rain_grid_mapping && python -m scripts.rain_report
python -m bits_ml.training_data
python -m scripts.data_overview        # this README and DATA_AND_MODEL_PLAN.md/.txt
pytest -q''')
    out("```")
    out()

    title("Documentation and sources")
    table(pd.DataFrame([
        ("[data_cleaning/DATA_CLEANING_SUMMARY.txt](data_cleaning/DATA_CLEANING_SUMMARY.txt)", "everything done, step by step"),
        ("[data_cleaning/README.md](data_cleaning/README.md)", "every gap in the raw data and its fix"),
        ("[data_cleaning/DATASETS.md](data_cleaning/DATASETS.md)", "every dataset and table, and its role"),
        ("[data_cleaning/MODELLING_GAPS.md](data_cleaning/MODELLING_GAPS.md)", "the suggested approach tested; training gaps"),
        ("[ml/CORE.md](ml/CORE.md)", "how the data layer is built"),
        ("[old/README.md](old/README.md)", "the earlier pipeline, archived"),
    ], columns=["document", "what it covers"]))
    out("- CGWB quality-controlled groundwater levels 2000–2022, figshare, CC BY 4.0 (doi: 10.6084/m9.figshare.29293877.v3)")
    out("- IMD 0.25° gridded daily rainfall, imdpune.gov.in")
    out("- District outlines, post-2020 boundaries (`data_cleaning/reference/districts.geojson`)")
    out()


def main() -> None:
    global MD
    for md, path in ((False, OUT_TXT), (True, OUT_MD)):
        MD = md
        lines.clear()
        build()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(lines)} lines)")
    MD = True
    lines.clear()
    readme_head()
    build(readme=True)
    readme_tail()
    OUT_README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_README} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
