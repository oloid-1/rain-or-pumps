"""Shared paths and settings."""

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = ML_DIR.parent

DATA_DIR = REPO_DIR / "data"
CGWB_CSV = DATA_DIR / "cgwb" / "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv"
IMD_DIR = DATA_DIR / "imd_rainfall"
DATA_PROCESSED = DATA_DIR / "processed"

MODELS_DIR = ML_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Raw CGWB archive (figshare bundle) and the member holding every well, unfiltered.
CGWB_ZIP = DATA_DIR / "Quality_controlled_groundwater_levels_over_India.zip"
CGWB_RAW_MEMBER = (
    "Quality_controlled_groundwater_levels_over_India/Input/"
    "1_India_GWLs_2000_2024_wells_within_India.csv"
)
HYDRO_MAP_MEMBER = (
    "Quality_controlled_groundwater_levels_over_India/Hydrogeological_map/Hydrogeological_map.tif"
)

# The cleaned core tables live in their own folder, apart from the earlier processed artefacts.
# data_cleaning/README.md explains every change made to the raw data and why.
CORE_DIR = REPO_DIR / "data_cleaning" / "data"

# District outlines (post-2020 vintage, properties n = district, s = state). Tracked in git.
DISTRICTS_GEOJSON = REPO_DIR / "data_cleaning" / "reference" / "districts.geojson"
