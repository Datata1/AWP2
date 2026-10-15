"""Central paths and constants – always import from here, never hardcode them."""

import os
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
MODEL_DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "modelle" / "img"  # model results, in git
FIGURE_DPI = 150

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
CLEAN_TRAIN_FILE = INTERIM_DATA_DIR / "train_clean.parquet"
SPLIT_FILE = PROCESSED_DATA_DIR / "split.csv"
SPLIT_CONFIG_FILE = PROCESSED_DATA_DIR / "split_config.json"

ID_COL = "id"
SUBSET_COL = "subset"
FOLD_COL = "cv_fold"
CROP_COL = "Crop"
STAGE_COL = "Stage"
AEZ_COL = "AEZ"
MONTH_COL = "Month"
MEAN_ALBEDO_COL = "mean_albedo"
TARGET_COLS = (CROP_COL, STAGE_COL)
META_COLS = (AEZ_COL, MONTH_COL)
MONTH_SIN_COL = f"{MONTH_COL}_sin"
MONTH_COS_COL = f"{MONTH_COL}_cos"
BAND_PREFIX = "X"
BAND_PATTERN = rf"^{BAND_PREFIX}\d+$"
NDVI_RED_BAND = "X671"
NDVI_NIR_BAND = "X854"
NDRE_RED_EDGE_BAND = "X702"
PRI_FIRST_BAND = "X529"
PRI_SECOND_BAND = "X569"
NDWI_SWIR_BAND = "X1235"
VEGETATION_INDEX_BANDS = (
    NDVI_RED_BAND,
    NDVI_NIR_BAND,
    NDRE_RED_EDGE_BAND,
    PRI_FIRST_BAND,
    PRI_SECOND_BAND,
    NDWI_SWIR_BAND,
)
VEGETATION_INDEX_NAMES = ("NDVI", "NDRE", "PRI", "NDWI")

# Deleted band blocks leave 60-303 nm gaps at 10 nm sampling; smoothing must not cross them.
SAVGOL_SPLIT_GAP_NM = 20

AEZ_RANGE = (1, 20)
MONTH_RANGE = (1, 12)

Crop = Literal["corn", "cotton", "rice", "soybean", "winter_wheat"]
Stage = Literal["Emerge_VEarly", "Early_Mid", "Critical", "Late", "Mature_Senesc", "Harvest"]
CROPS: tuple[str, ...] = get_args(Crop)
STAGES: tuple[str, ...] = get_args(Stage)
# stages before and with a closed canopy, compared in the error analysis
EARLY_STAGES: tuple[Stage, ...] = ("Emerge_VEarly", "Early_Mid")
CLOSED_CANOPY_STAGES: tuple[Stage, ...] = ("Critical", "Late")
Subset = Literal["train", "val"]
TRAIN_SUBSET, VAL_SUBSET = get_args(Subset)
Routing = Literal["hard", "soft"]
HARD_ROUTING, SOFT_ROUTING = get_args(Routing)
LABEL_SEP = "|"  # "_" occurs inside crop and stage names, so it cannot separate them

SEED = 42
VAL_SIZE = 0.3
CV_FOLDS = 5
SCORE_DECIMALS = 4
RECALL_COL = "recall"
SUPPORT_COL = "support"
RANDOM_FOREST_N_ESTIMATORS = 300
EXTRA_TREES_N_ESTIMATORS = 300
LOGISTIC_REGRESSION_MAX_ITER = 5000  # lbfgs needs more than the default 100 on ~140 features
MLP_MAX_ITER = 500  # an upper bound only; early stopping ends training before
# a CV gain below this is within the fold-to-fold noise of the reference (std about 0.010)
MIN_CV_GAIN = 0.005
CNN_DEFAULT_FILTERS = 16
CNN_DEFAULT_KERNEL_SIZE = 5
CNN_DEFAULT_DROPOUT = 0.2
CNN_DEFAULT_LEARNING_RATE = 0.001
CNN_DEFAULT_MAX_EPOCHS = 50
CNN_DEFAULT_BATCH_SIZE = 64
CNN_EARLY_STOPPING_PATIENCE = 8
CNN_INNER_VALIDATION_FRACTION = 0.1
MLP_DEFAULT_HIDDEN_UNITS = 64
MLP_DEFAULT_HIDDEN_LAYERS = 2
MLP_DEFAULT_DROPOUT = 0.2
MLP_DEFAULT_LEARNING_RATE = 0.001
MLP_DEFAULT_WEIGHT_DECAY = 0.0
MLP_DEFAULT_MAX_EPOCHS = 50
MLP_DEFAULT_BATCH_SIZE = 64
MLP_EARLY_STOPPING_PATIENCE = 8
MLP_INNER_VALIDATION_FRACTION = 0.1
SLUG_PATTERN = (
    r"^[a-z0-9][a-z0-9_-]*$"  # lowercase, no spaces: names of runs, approaches, experiments
)

# Local MLflow store, not in git; the env vars allow a shared server or a throwaway test store.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_ARTIFACTS_DIR = Path(
    os.environ.get("MLFLOW_ARTIFACTS_DIR", PROJECT_ROOT / "mlruns")
).resolve()
MLFLOW_EXPERIMENT = "crop-stage"  # main question: which approach predicts crop and stage best
# Marks the experiment as classic ML so the MLflow UI opens the "Model training" view.
MLFLOW_EXPERIMENT_KIND_TAG = ("mlflow.experimentKind", "custom_model_development")
MLFLOW_SYSTEM_METRICS_INTERVAL_S = (
    1  # our runs take seconds; MLflow's default of 10 s records nothing
)
MLFLOW_PARAM_MAX_CHARS = 250  # keeps logged parameter values short and readable in the UI
