"""Central paths and constants – always import from here, never hardcode them."""

from pathlib import Path
from typing import Literal, get_args

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"  # scratch figures, not in git
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "daten" / "img"  # figures shown in the docs, in git
FIGURE_DPI = 150

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

ID_COL = "id"
CROP_COL = "Crop"
STAGE_COL = "Stage"
AEZ_COL = "AEZ"
MONTH_COL = "Month"
TARGET_COLS = (CROP_COL, STAGE_COL)
META_COLS = (AEZ_COL, MONTH_COL)
BAND_PREFIX = "X"
BAND_PATTERN = rf"^{BAND_PREFIX}\d+$"

AEZ_RANGE = (1, 20)
MONTH_RANGE = (1, 12)

Crop = Literal["corn", "cotton", "rice", "soybean", "winter_wheat"]
Stage = Literal["Emerge_VEarly", "Early_Mid", "Critical", "Late", "Mature_Senesc", "Harvest"]
CROPS: tuple[str, ...] = get_args(Crop)
STAGES: tuple[str, ...] = get_args(Stage)
LABEL_SEP = "|"  # "_" occurs inside crop and stage names, so it cannot separate them

SEED = 42
VAL_SIZE = 0.3  # 70/30 holdout required for the M1 baseline
CV_FOLDS = 5
SCORE_DECIMALS = 4
RUN_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"  # lowercase, no spaces: usable as file/run id
