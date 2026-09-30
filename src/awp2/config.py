"""Central paths and constants – always import from here, never hardcode them."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"  # scratch figures, not in git
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "daten" / "img"  # figures shown in the docs, in git

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

ID_COL = "id"
TARGET_COLS = ("Crop", "Stage")
META_COLS = ("AEZ", "Month")
BAND_PREFIX = "X"
BAND_PATTERN = rf"^{BAND_PREFIX}\d+$"

CROPS = ("corn", "cotton", "rice", "soybean", "winter_wheat")
STAGES = ("Emerge_VEarly", "Early_Mid", "Critical", "Late", "Mature_Senesc", "Harvest")

SEED = 42
VAL_SIZE = 0.3  # 70/30 holdout required for the M1 baseline
