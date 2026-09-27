"""Every CGWB well, unfiltered, with quality flagged instead of removed.

The file the project used until now is the published *output* of the source
paper's quality control, which keeps 2,759 of 32,299 wells. Replaying that
cascade shows where the loss comes from: 884 wells are dropped for holding one
negative reading anywhere in 25 years, and 25,736 more for the rule that every
single year must carry at least two of its four readings. That second rule is a
completeness test, not a quality test, and the wells it removes are not the same
wells: they deepen slightly more often than the survivors.

So this module starts from the raw file inside the archive and drops nothing.
Every reading that looks wrong gets a flag naming what is wrong with it, every
well carries the counts behind those flags, and the views at the bottom
(`view_strict`, `view_modelling`) are the only place anything is excluded.
Anyone can disagree with a rule and rebuild a view without touching the ingest.

Flags on readings:

* `flag_negative`       water above ground: a flowing well, or a sign error
* `flag_zero`           exactly 0.000 m
* `flag_zero_suspect`   a zero in a well where zeros are more than a fifth of
                        the record, which is missing data written as zero
                        (13,734 of them, mostly Telangana bore wells)
* `flag_below_bottom`   water reported deeper than the drilled depth of the well
* `flag_outlier`        3 sigma from the well's own seasonal pattern; measured
                        against that well-campaign's mean, not the raw level, so
                        a well with a real long decline keeps its deepest
                        readings instead of having them clipped
* `flag_repeat_run`     one of three or more identical consecutive readings

Location comes from coordinates. `district` and `state` are the post-2020
district outline each well's coordinates fall in (`districts.well_district`),
not the labels in the file: those mix vintages and spellings, and where the two
disagree on the state, a well's water level follows its neighbours at the
coordinates far better than the wells of its labelled district (median
correlation 0.39 against 0.17, coordinates closer for 69% of 262 such wells).
The labels are kept as `cgwb_district` and `cgwb_state`.

Run from ml/:  python -m bits_ml.ingest
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import districts as ds
from .config import CGWB_CSV, CGWB_RAW_MEMBER, CGWB_ZIP, CORE_DIR
from .rainfall import coordinate_half_width

CAMPAIGNS = ("Jan", "May", "Aug", "Nov")
CAMPAIGN_MONTH = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}
RAW_YEARS = tuple(range(2000, 2025))      # the raw file runs two years past the rain grid
TRAIN_YEARS = tuple(range(2000, 2015))
MIN_TRAIN_READINGS = 5                    # what a per-campaign normal needs to mean anything
ZERO_SHARE_SUSPECT = 0.20
BELOW_BOTTOM_TOL_M = 0.5                  # drilled depth is reported to a tenth of a metre at best
SIGMA = 3.0
MIN_FOR_SIGMA = 8                         # too few readings and the sigma is the outlier

_META = {
    "Station Code": "station_code",
    "Station Name": "station",
    "Station Type": "station_type",
    "Agency Name": "agency",
    "State": "state",
    "District": "district",
    "Tehsil": "tehsil",
    "Data Acquisition": "acquisition",
    "Block": "block",
    "Village": "village",
    "Latitude": "lat",
    "Longitude": "lon",
    "Data Available From": "data_from",
    "Latest  Data Available": "data_latest",
    "Type of Well": "well_type",
    "Aquifer Type": "aquifer",
    "Well Depth": "well_depth_m",
}
FLAGS = ["flag_negative", "flag_zero", "flag_zero_suspect", "flag_below_bottom",
         "flag_outlier", "flag_repeat_run"]
# A zero that is not suspect is kept: water can stand at the surface after a
# monsoon. flag_zero on its own therefore does not make a reading invalid.
INVALIDATING = ["flag_negative", "flag_zero_suspect", "flag_below_bottom",
                "flag_outlier", "flag_repeat_run"]


LABEL_OUTLIER_KM = 150.0


def label_outliers(wells: pd.DataFrame, km: float = LABEL_OUTLIER_KM) -> pd.Series:
    """Wells more than `km` from the median position of the other wells with the same district label.

    Kept as information on the file's labels; location itself comes from the
    coordinates (`place_by_coordinates`).
    """
    median = wells.groupby("district")[["lat", "lon"]].transform("median")
    dy = (wells["lat"] - median["lat"]) * 111.2
    dx = (wells["lon"] - median["lon"]) * 111.2 * np.cos(np.radians(wells["lat"]))
    return pd.Series(np.hypot(dx, dy) > km, index=wells.index, name="far_from_district")


def reading_columns(years=RAW_YEARS) -> list[str]:
    return [f"{c}-{y % 100:02d}" for y in years for c in CAMPAIGNS]


def read_raw(zip_path: Path = CGWB_ZIP, member: str = CGWB_RAW_MEMBER) -> pd.DataFrame:
    """The unfiltered well file, read straight out of the archive.

    Station Code is read as text. It arrives already written in scientific
    notation ("1.40136E+14"), which has collapsed 32,299 codes into 22,399
    distinct strings, so it is kept for reference and never used as identity.
    """
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        return pd.read_csv(handle, dtype={"Station Code": str}, low_memory=False)


def site_key(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Position rounded to four decimals: the key rainfall is sampled on."""
    return lat.map("{:.4f}".format) + "_" + lon.map("{:.4f}".format)


def well_uid(sites: pd.Series, order: pd.Series) -> pd.Series:
    """A unique, readable, stable key per well.

    Position alone is not unique: 3,710 raw rows share a coordinate with another
    well, either because two wells sit in one village or because the coordinate
    was rounded to a tenth of a degree. Co-located wells get a suffix, assigned
    in station-name order so the key does not move when the file is re-read.
    """
    rank = order.groupby(sites).rank(method="first").astype(int)
    shared = sites.duplicated(keep=False)
    return sites.where(~shared, sites + "#" + rank.astype(str))


def load_wells(zip_path: Path = CGWB_ZIP, member: str = CGWB_RAW_MEMBER) -> pd.DataFrame:
    """One row per well: identity, position, construction. No readings."""
    raw = read_raw(zip_path, member)
    wells = raw[list(_META)].rename(columns=_META)
    for column in ("lat", "lon", "well_depth_m"):
        wells[column] = pd.to_numeric(wells[column], errors="coerce")
    for column in ("state", "district", "station", "well_type", "aquifer", "agency"):
        wells[column] = wells[column].astype(str).str.strip()

    wells["site_key"] = site_key(wells["lat"], wells["lon"])
    wells["well_uid"] = well_uid(wells["site_key"], wells["station"].fillna(""))
    wells["site_shared"] = wells["site_key"].duplicated(keep=False)
    wells["coord_half_width_deg"] = coordinate_half_width(wells["lat"], wells["lon"])
    wells["coord_rounded"] = wells["coord_half_width_deg"] > 0
    wells["far_from_district"] = label_outliers(wells).to_numpy()
    wells["depth_missing"] = wells["well_depth_m"].isna() | (wells["well_depth_m"] <= 0)
    return place_by_coordinates(wells).set_index("well_uid", drop=False)


def place_by_coordinates(wells: pd.DataFrame, outlines: pd.DataFrame | None = None) -> pd.DataFrame:
    """District and state from the outline each well stands in; the file's labels kept beside them."""
    out = wells.rename(columns={"district": "cgwb_district", "state": "cgwb_state"})
    placed = ds.well_district(out, ds.load_districts() if outlines is None else outlines)
    placed = placed.rename(columns={"how": "district_how", "distance_km": "district_distance_km"})
    columns = ["district", "state", "district_how", "district_distance_km", "name_agrees", "state_agrees"]
    return out.merge(placed[["well_uid", *columns]], on="well_uid", how="left", validate="1:1")


def melt_readings(zip_path: Path = CGWB_ZIP, member: str = CGWB_RAW_MEMBER) -> pd.DataFrame:
    """Every reading as its own row: well, campaign, year, date, metres below ground.

    Blank cells are dropped here. Cells holding a value are all kept, however
    implausible; judging them is what the flags are for.
    """
    raw = read_raw(zip_path, member)
    wells = load_wells(zip_path, member)
    columns = [c for c in reading_columns() if c in raw.columns]
    values = raw[columns].apply(pd.to_numeric, errors="coerce")
    values.insert(0, "well_uid", wells["well_uid"].to_numpy())

    long = values.melt(id_vars="well_uid", var_name="column", value_name="depth_mbgl")
    long = long.dropna(subset=["depth_mbgl"])
    parts = long["column"].str.split("-", n=1, expand=True)
    long["campaign"] = parts[0]
    long["year"] = 2000 + parts[1].astype(int)
    long["month"] = long["campaign"].map(CAMPAIGN_MONTH)
    long["date"] = pd.to_datetime(dict(year=long["year"], month=long["month"], day=15))
    return long.drop(columns="column").sort_values(["well_uid", "date"], ignore_index=True)


def _repeat_runs(depth: pd.Series) -> pd.Series:
    """True for every member of a run of three or more identical readings."""
    block = (depth != depth.shift()).cumsum()
    return depth.groupby(block).transform("size") > 2


def _seasonal_outliers(obs: pd.DataFrame, usable: pd.Series) -> pd.Series:
    """3 sigma against the well's own seasonal pattern rather than its raw level.

    Subtracting each well-campaign's mean first is what keeps a long decline
    intact: on the raw level, a well that has fallen ten metres has its recent
    readings sitting three sigma from a mean drawn mostly from its wetter past,
    and clipping them flattens the very trend being measured.
    """
    depth = obs["depth_mbgl"].where(usable)
    season = depth.groupby([obs["well_uid"], obs["campaign"]]).transform("mean")
    deviation = depth - season
    by_well = deviation.groupby(obs["well_uid"])
    spread = by_well.transform("std")
    enough = by_well.transform("size") >= MIN_FOR_SIGMA
    return ((deviation.abs() > SIGMA * spread) & enough & spread.gt(0) & usable).fillna(False)


def flag_readings(obs: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    """Attach the six quality flags and the `valid` column they imply."""
    out = obs.sort_values(["well_uid", "date"], ignore_index=True)
    depth = out["depth_mbgl"]
    out["flag_negative"] = depth < 0
    out["flag_zero"] = depth == 0

    zero_share = out.groupby("well_uid")["flag_zero"].transform("mean")
    out["flag_zero_suspect"] = out["flag_zero"] & (zero_share > ZERO_SHARE_SUSPECT)

    bottom = wells["well_depth_m"].reindex(out["well_uid"]).to_numpy()
    out["flag_below_bottom"] = depth > bottom + BELOW_BOTTOM_TOL_M  # a NaN depth gives False

    usable = ~(out["flag_negative"] | out["flag_zero_suspect"] | out["flag_below_bottom"])
    out["flag_outlier"] = _seasonal_outliers(out, usable)
    out["flag_repeat_run"] = out.groupby("well_uid")["depth_mbgl"].transform(_repeat_runs)
    out["valid"] = ~out[INVALIDATING].any(axis=1)
    return out


def strict_view(wells: pd.DataFrame, path: Path = CGWB_CSV) -> pd.Series:
    """Membership of the published 2,759-well set, kept as the control arm.

    The published file identifies a well only by its coordinates, and 121 raw
    wells share a coordinate with one of its rows, so the station name is
    matched too and the first remaining candidate wins. That reproduces 2,759
    rows exactly rather than 2,880.
    """
    if not Path(path).exists():
        return pd.Series(False, index=wells.index, name="view_strict")
    published = pd.read_csv(path, encoding="utf-8-sig")
    published["site_key"] = site_key(pd.to_numeric(published["Latitude"]),
                                     pd.to_numeric(published["Longitude"]))
    published["station"] = published["Station Name"].astype(str).str.strip()

    pairs = set(zip(published["site_key"], published["station"]))
    named = pd.Series([pair in pairs for pair in zip(wells["site_key"], wells["station"])],
                      index=wells.index)
    rank = named.astype(int).groupby(wells["site_key"].to_numpy()).cumsum()
    chosen = named & rank.eq(1)
    missed = set(published["site_key"]) - set(wells.loc[chosen, "site_key"])
    if missed:  # a name that does not match: fall back to the first well at that coordinate
        chosen = chosen | (wells["site_key"].isin(missed) & ~wells["site_key"].duplicated())
    return pd.Series(chosen.to_numpy(), index=wells.index, name="view_strict")


def attach_specific_yield(wells: pd.DataFrame, path: Path = CGWB_CSV) -> pd.DataFrame:
    """Specific yield per well, and an honest label for where each value came from.

    The raw file has no specific yield: the source paper read it off a
    hydrogeological map, and only the 2,759 wells it published carry the result.
    Those are taken as given. The rest are filled with the median of the wells
    that do have a value in the same state and aquifer, falling back to the
    state and then to the whole country, because specific yield is a property of
    the rock a well sits in and these are the coarsest stand-ins for that rock.

    `sy_source` says which of the three happened, so a model can drop the
    imputed ones and see whether anything changes.
    """
    out = wells.copy()
    out["sy"] = np.nan
    out["sy_source"] = "missing"
    if Path(path).exists():
        published = pd.read_csv(path, encoding="utf-8-sig")
        published["site_key"] = site_key(pd.to_numeric(published["Latitude"]),
                                         pd.to_numeric(published["Longitude"]))
        known = published.groupby("site_key")["Reference_Sy"].median()
        out["sy"] = out["site_key"].map(known)
        out.loc[out["sy"].notna(), "sy_source"] = "published"

    for keys, label in ((["state", "aquifer"], "imputed_state_aquifer"), (["state"], "imputed_state")):
        filler = out.groupby(keys)["sy"].transform("median")
        fill = out["sy"].isna() & filler.notna()
        out.loc[fill, ["sy", "sy_source"]] = pd.DataFrame(
            {"sy": filler[fill], "sy_source": label}, index=out.index[fill])
    national = out["sy"].median()
    fill = out["sy"].isna() & pd.notna(national)
    out.loc[fill, ["sy", "sy_source"]] = [national, "imputed_national"]
    return out


def summarise_wells(wells: pd.DataFrame, obs: pd.DataFrame, train_years=TRAIN_YEARS,
                    min_readings: int = MIN_TRAIN_READINGS) -> pd.DataFrame:
    """Per-well counts, and the two views that decide who gets modelled.

    `view_modelling` replaces the source paper's "every year must carry two
    readings" with what the model actually needs: enough training-year readings
    in each campaign to define a normal worth subtracting.
    """
    out = wells.copy()
    valid = obs[obs["valid"]]
    in_window = valid[valid["year"] <= 2022]
    train = valid[valid["year"].isin(list(train_years))]

    out["n_obs"] = obs.groupby("well_uid").size().reindex(out.index).fillna(0).astype(int)
    out["n_valid"] = valid.groupby("well_uid").size().reindex(out.index).fillna(0).astype(int)
    out["n_valid_2000_2022"] = in_window.groupby("well_uid").size().reindex(out.index).fillna(0).astype(int)
    out["zero_share"] = obs.groupby("well_uid")["flag_zero"].mean().reindex(out.index).fillna(0.0)
    out["first_year"] = valid.groupby("well_uid")["year"].min().reindex(out.index)
    out["last_year"] = valid.groupby("well_uid")["year"].max().reindex(out.index)

    per_campaign = (
        train.groupby(["well_uid", "campaign"]).size().unstack(fill_value=0).reindex(out.index, fill_value=0)
    )
    for campaign in CAMPAIGNS:
        out[f"n_train_{campaign.lower()}"] = per_campaign.get(campaign, 0)
    counts = out[[f"n_train_{c.lower()}" for c in CAMPAIGNS]]
    out["n_campaigns_covered"] = (counts >= min_readings).sum(axis=1)

    out["view_strict"] = strict_view(out)
    out["view_modelling"] = out["n_campaigns_covered"] == len(CAMPAIGNS)
    return out


def qc_audit(obs: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    """What each flag costs, in readings and in wells, so no rule is silent."""
    rows = [("readings read", len(obs), obs["well_uid"].nunique())]
    for flag in FLAGS:
        hit = obs[flag]
        rows.append((flag, int(hit.sum()), int(obs.loc[hit, "well_uid"].nunique())))
    rows.append(("invalid (any invalidating flag)", int((~obs["valid"]).sum()),
                 int(obs.loc[~obs["valid"], "well_uid"].nunique())))
    rows.append(("valid", int(obs["valid"].sum()), int(obs.loc[obs["valid"], "well_uid"].nunique())))
    rows.append(("view_strict wells", int(wells["view_strict"].sum()), int(wells["view_strict"].sum())))
    rows.append(("view_modelling wells", int(wells["view_modelling"].sum()), int(wells["view_modelling"].sum())))
    return pd.DataFrame(rows, columns=["step", "readings", "wells"])


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    wells = load_wells()
    obs = flag_readings(melt_readings(), wells)
    return summarise_wells(attach_specific_yield(wells), obs), obs


def main() -> None:
    wells, obs = build()
    audit = qc_audit(obs, wells)
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    wells.reset_index(drop=True).to_parquet(CORE_DIR / "well_master.parquet", index=False)
    obs.to_parquet(CORE_DIR / "well_obs.parquet", index=False)
    audit.to_csv(CORE_DIR / "qc_audit.csv", index=False)

    print(f"wells {len(wells):,} | readings {len(obs):,} | years {obs['year'].min()}-{obs['year'].max()}")
    print("\nquality audit:")
    print(audit.to_string(index=False))
    print("\nviews:")
    for view in ("view_strict", "view_modelling"):
        sub = wells[wells[view]]
        print(f"  {view:<16} wells {len(sub):>6,} | districts {sub['district'].nunique():>4} "
              f"| states {sub['state'].nunique():>3} | dug wells {(sub['well_type'] == 'Dug well').mean() * 100:4.0f}%")
    print(f"\nwrote well_master.parquet, well_obs.parquet, qc_audit.csv to {CORE_DIR}")


if __name__ == "__main__":
    main()
