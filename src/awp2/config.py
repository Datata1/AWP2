"""Central paths and constants – always import from here, never hardcode them."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"  # third-party reference data, downloaded, not in git

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
SPLIT_CONFIG_HASH_COL = "split_config_hash"
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
INDEX_WAVELENGTH_TOLERANCE_NM = 5
# Hyperion samples every ~10 nm; a larger step marks a block of deleted bands.
BAND_GAP_NM = 20
REFLECTANCE_PERCENT_SCALE = 100.0
MCARI_GREEN_WEIGHT = 0.2
TCARI_MULTIPLIER = 3.0
OSAVI_SOIL_ADJUSTMENT = 0.16
REIP_LAGRANGIAN_WINDOW_NM = (680, 750)
REIP_LAGRANGIAN_MIN_BANDS = 5
REIP_POLYNOMIAL_WINDOW_NM = (650, 760)
REIP_POLYNOMIAL_DEGREE = 4
PCA_FEATURE_NAMES = ("PC_1", "PC_2")


@dataclass(frozen=True)
class VegetationIndexSpec:
    """Static definition of one candidate vegetation index.

    Args:
        name: Stable output feature name.
        formula: Human-readable formula using ``R_<wavelength>`` reflectance values.
        target_wavelengths_nm: Wavelengths required by the formula in nanometres.
        category: The catalogue group from the kickoff table.
        source: Primary source of the formula or the kickoff-table citation.
        status: Whether the common transformer computes, defers or excludes the candidate.
        reason: Explanation for a deferred or excluded candidate.
    """

    name: str
    formula: str
    target_wavelengths_nm: tuple[int, ...]
    category: str
    source: str
    status: Literal["implemented", "deferred", "excluded"]
    reason: str | None = None


VEGETATION_INDEX_SPECS: tuple[VegetationIndexSpec, ...] = (
    VegetationIndexSpec(
        "SR_750_705",
        "R_750 / R_705",
        (750, 705),
        "Chlorophyll",
        "Gitelson & Merzlyak (1994)",
        "implemented",
    ),
    VegetationIndexSpec(
        "SR_800_550",
        "R_800 / R_550",
        (800, 550),
        "Chlorophyll",
        "Gitelson & Merzlyak (1994)",
        "implemented",
    ),
    VegetationIndexSpec(
        "ND_750_660",
        "(R_750 - R_660) / (R_750 + R_660)",
        (750, 660),
        "Chlorophyll",
        "Sims & Gamon (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "ND_800_680",
        "(R_800 - R_680) / (R_800 + R_680)",
        (800, 680),
        "Chlorophyll",
        "Sims & Gamon (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "ND_750_705",
        "(R_750 - R_705) / (R_750 + R_705)",
        (750, 705),
        "Chlorophyll",
        "Gitelson & Merzlyak (1994)",
        "implemented",
    ),
    VegetationIndexSpec(
        "MSR_750_445",
        "(R_750 - R_445) / (R_705 - R_445)",
        (750, 705, 445),
        "Chlorophyll",
        "Sims & Gamon (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "MND_750_705",
        "(R_750 - R_445) / (R_750 + R_705 - 2 R_445)",
        (750, 705, 445),
        "Chlorophyll",
        "Sims & Gamon (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "SR_750_550",
        "R_750 / R_550",
        (750, 550),
        "Chlorophyll",
        "Gitelson & Merzlyak (1994)",
        "implemented",
    ),
    VegetationIndexSpec(
        "MEAN_750_850",
        "mean(R_750:R_850)",
        (750, 850),
        "Chlorophyll",
        "Stachniewicz et al. (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "MCARI",
        "[(R_700 - R_670) - 0.2 (R_700 - R_550)] (R_700 / R_670)",
        (700, 670, 550),
        "Chlorophyll",
        "Daughtry et al. (2000)",
        "implemented",
    ),
    VegetationIndexSpec(
        "TCARI",
        "3 [(R_700 - R_670) - 0.2 (R_700 - R_550) (R_700 / R_670)]",
        (700, 670, 550),
        "Chlorophyll",
        "Haboudane et al. (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "OSAVI",
        "1.16 (R_800 - R_670) / (R_800 + R_670 + 0.16)",
        (800, 670),
        "Chlorophyll",
        "Rondeaux et al. (1996)",
        "implemented",
    ),
    VegetationIndexSpec(
        "TCARI_OSAVI",
        "TCARI / OSAVI",
        (700, 670, 550, 800),
        "Chlorophyll",
        "Haboudane et al. (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "PSRI",
        "(R_680 - R_500) / R_750",
        (680, 500, 750),
        "Senescence",
        "Merzlyak et al. (1999)",
        "implemented",
    ),
    VegetationIndexSpec(
        "SIPI",
        "(R_800 - R_445) / (R_800 - R_680)",
        (800, 445, 680),
        "Senescence",
        "Peñuelas et al. (1995)",
        "implemented",
    ),
    VegetationIndexSpec(
        "PRI",
        "(R_531 - R_570) / (R_531 + R_570)",
        (531, 570),
        "Chlorophyll",
        "Gamon et al. (1992)",
        "implemented",
    ),
    VegetationIndexSpec(
        "PSSRA", "R_800 / R_680", (800, 680), "Chlorophyll", "Blackburn (1998)", "implemented"
    ),
    VegetationIndexSpec(
        "PSSRB", "R_800 / R_635", (800, 635), "Chlorophyll", "Blackburn (1998)", "implemented"
    ),
    VegetationIndexSpec(
        "SR_695_760", "R_695 / R_760", (695, 760), "Chlorophyll", "Carter (1994)", "implemented"
    ),
    VegetationIndexSpec(
        "DSWI_1", "R_800 / R_1650", (800, 1650), "Water", "Spatial Sciences (2003)", "implemented"
    ),
    VegetationIndexSpec(
        "DSWI_2", "R_1660 / R_550", (1660, 550), "Water", "Spatial Sciences (2003)", "implemented"
    ),
    VegetationIndexSpec(
        "DSWI_3", "R_1660 / R_680", (1660, 680), "Water", "Spatial Sciences (2003)", "implemented"
    ),
    VegetationIndexSpec(
        "DSWI_4", "R_550 / R_680", (550, 680), "Water", "Spatial Sciences (2003)", "implemented"
    ),
    VegetationIndexSpec(
        "DSWI_5",
        "(R_800 + R_550) / (R_1660 + R_680)",
        (800, 550, 1660, 680),
        "Water",
        "Spatial Sciences (2003)",
        "implemented",
    ),
    VegetationIndexSpec(
        "NDWI_1070_1200",
        "(R_1070 - R_1200) / (R_1070 + R_1200)",
        (1070, 1200),
        "Water",
        "Ustin et al. (2002)",
        "implemented",
    ),
    VegetationIndexSpec(
        "NDWI_860_1240",
        "(R_860 - R_1240) / (R_860 + R_1240)",
        (860, 1240),
        "Water",
        "Gao (1996)",
        "implemented",
    ),
    VegetationIndexSpec(
        "MSI", "R_1600 / R_820", (1600, 820), "Water", "Hunt & Rock (1989)", "implemented"
    ),
    VegetationIndexSpec(
        "SR_895_420",
        "R_895 / R_420",
        (895, 420),
        "Chlorophyll",
        "Carter (1994)",
        "excluded",
        "420 nm has no non-empty Hyperion band within 5 nm.",
    ),
    VegetationIndexSpec(
        "SR_695_420",
        "R_695 / R_420",
        (695, 420),
        "Chlorophyll",
        "Carter (1994)",
        "excluded",
        "420 nm has no non-empty Hyperion band within 5 nm.",
    ),
    VegetationIndexSpec(
        "WI_900_970",
        "R_900 / R_970",
        (900, 970),
        "Water",
        "Peñuelas et al. (1997)",
        "excluded",
        "970 nm has no non-empty Hyperion band within 5 nm.",
    ),
    VegetationIndexSpec(
        "CHLOROPHYLL_WELL",
        "unpublished well metric",
        (),
        "Chlorophyll",
        "CSIRO VegSpectra (unpublished)",
        "excluded",
        "The source gives no reproducible formula.",
    ),
    VegetationIndexSpec(
        "GREEN_PEAK_WELL",
        "unpublished well metric",
        (),
        "Chlorophyll",
        "CSIRO VegSpectra (unpublished)",
        "excluded",
        "The source gives no reproducible formula.",
    ),
    VegetationIndexSpec(
        "WATER_WELL_893",
        "unpublished three-point well metric",
        (893,),
        "Water",
        "CSIRO VegSpectra (unpublished)",
        "excluded",
        "The source gives no reproducible formula.",
    ),
    VegetationIndexSpec(
        "WATER_WELL_1205",
        "unpublished three-point well metric",
        (1205,),
        "Water",
        "CSIRO VegSpectra (unpublished)",
        "excluded",
        "The source gives no reproducible formula.",
    ),
    VegetationIndexSpec(
        "WATER_WELL_893_PSSRA",
        "WATER_WELL_893 / PSSRA",
        (893, 800, 680),
        "Water",
        "CSIRO VegSpectra (unpublished)",
        "excluded",
        "It depends on an unpublished well metric.",
    ),
    VegetationIndexSpec(
        "WATER_WELL_1205_PSSRA",
        "WATER_WELL_1205 / PSSRA",
        (1205, 800, 680),
        "Water",
        "CSIRO VegSpectra (unpublished)",
        "excluded",
        "It depends on an unpublished well metric.",
    ),
    VegetationIndexSpec(
        "REIP_LAGRANGIAN",
        "maximum of a quadratic Lagrange interpolation of the first derivative",
        REIP_LAGRANGIAN_WINDOW_NM,
        "Red edge / REIP",
        "Dawson & Curran (1998)",
        "implemented",
    ),
    VegetationIndexSpec(
        "REIP_POLYNOMIAL",
        "maximum first derivative of a fourth-degree polynomial fit",
        REIP_POLYNOMIAL_WINDOW_NM,
        "Red edge / REIP",
        "Broge & Leblanc (2001)",
        "implemented",
    ),
    VegetationIndexSpec(
        "PC_1",
        "first principal component",
        (),
        "Red edge / REIP",
        "Spatial Sciences (2003)",
        "implemented",
    ),
    VegetationIndexSpec(
        "PC_2",
        "second principal component",
        (),
        "Red edge / REIP",
        "Spatial Sciences (2003)",
        "implemented",
    ),
)
VEGETATION_INDEX_NAMES = tuple(
    spec.name
    for spec in VEGETATION_INDEX_SPECS
    if spec.status == "implemented" and spec.name not in PCA_FEATURE_NAMES
)
AEZ_MONTH_PREFIX = "aez_month"
AEZ_MONTH_SEP = "_"

# Deleted band blocks leave 60-303 nm gaps at 10 nm sampling; smoothing must not cross them.
SAVGOL_SPLIT_GAP_NM = 20

AEZ_RANGE = (1, 20)
MONTH_RANGE = (1, 12)


@dataclass(frozen=True)
class SpectralRegion:
    """A named wavelength interval of the vegetation spectrum.

    Args:
        name: Display name.
        start_nm: Lower bound in nanometres (inclusive).
        end_nm: Upper bound in nanometres (exclusive).
    """

    name: str
    start_nm: int
    end_nm: int


# Boundaries follow docs/domaene/vegetation.md: red edge 680-750 nm, SWIR from about 1300 nm.
SPECTRAL_REGIONS: tuple[SpectralRegion, ...] = (
    SpectralRegion("Blau", 400, 500),
    SpectralRegion("Grün", 500, 600),
    SpectralRegion("Rot", 600, 680),
    SpectralRegion("Red Edge", 680, 750),
    SpectralRegion("NIR", 750, 1300),
    SpectralRegion("SWIR 1", 1300, 1800),
    SpectralRegion("SWIR 2", 1800, 2500),
)

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
LSTMContextMode = Literal["none", "standard", "embeddings"]
LSTM_NO_CONTEXT, LSTM_STANDARD_CONTEXT, LSTM_EMBEDDINGS_CONTEXT = get_args(LSTMContextMode)
SequenceCell = Literal["lstm", "gru"]
LSTM_CELL, GRU_CELL = get_args(SequenceCell)
BandScore = Literal["f_classif", "mutual_info"]
F_CLASSIF_SCORE, MUTUAL_INFO_SCORE = get_args(BandScore)
Sensor = Literal["hyperion", "sentinel2"]
HYPERION_SENSOR, SENTINEL2_SENSOR = get_args(Sensor)


@dataclass(frozen=True)
class SensorBand:
    """One band of a simulated multispectral sensor.

    Args:
        name: Band name in the sensor documentation.
        center_nm: Central wavelength.
        width_nm: Full bandwidth; the simulation averages all bands inside it.
    """

    name: str
    center_nm: int
    width_nm: int


# Sentinel-2A MSI surface bands (ESA). B1, B9 and B10 are left out: they sense aerosols,
# water vapour and cirrus for atmospheric correction, not the crop.
SENTINEL2_BANDS: tuple[SensorBand, ...] = (
    SensorBand("B2", 492, 66),
    SensorBand("B3", 560, 36),
    SensorBand("B4", 665, 31),
    SensorBand("B5", 704, 15),
    SensorBand("B6", 740, 15),
    SensorBand("B7", 783, 20),
    SensorBand("B8", 833, 106),
    SensorBand("B8A", 865, 21),
    SensorBand("B11", 1614, 91),
    SensorBand("B12", 2202, 175),
)

# Measured Sentinel-2 MSI spectral response functions (ESA, COPE-GSEG-EOPG-TN-15-0007).
# SentiWiki attachment URLs carry a content hash, so a pinned URL keeps serving this version.
SENTINEL2_SRF_VERSION = "5.0"
SENTINEL2_SRF_URL = (
    "https://sentiwiki.copernicus.eu/__attachments/"
    "a_ad595cc9fa0018a4e9eb28e1e234124a751149b5933b0b8b1f6b80612eaab586/"
    "sentinel-2-spectral-response-functions-v5.xlsx"
)
SENTINEL2_SRF_SHA256 = "42be3ebd8bfcbde11c37caeb2e2fdda65fefa8d2b6f8329559b7806d02cdc9e2"
SENTINEL2_SRF_FILE = EXTERNAL_DATA_DIR / f"sentinel2_srf_v{SENTINEL2_SRF_VERSION}.xlsx"
Sentinel2Platform = Literal["S2A", "S2B"]
SENTINEL2A_PLATFORM, SENTINEL2B_PLATFORM = get_args(Sentinel2Platform)
SENTINEL2_SRF_SHEET = "Spectral Responses ({platform})"
SENTINEL2_SRF_WAVELENGTH_COL = "SR_WL"
SENTINEL2_SRF_BAND_PATTERN = r"^S2[AB]_SR_AV_(B\w+)$"  # group 1 is the band name, e.g. "B8A"
DOWNLOAD_TIMEOUT_S = 60

ReductionFamily = Literal[
    "all_bands", "pca", "band_selection", "band_step", "sentinel2", "indices_only", "meta_only"
]
REDUCTION_FAMILY_LABELS: dict[str, str] = {
    "all_bands": "Alle Bänder",
    "pca": "PCA",
    "band_selection": "Bandauswahl",
    "band_step": "Jedes n-te Band",
    "sentinel2": "Sentinel-2",
    "indices_only": "Nur Indizes",
    "meta_only": "Nur Meta",
}
REDUCTION_PCA_COMPONENTS = (2, 5, 10, 20, 40)
REDUCTION_SELECT_K = (5, 10, 20, 40, 80)
REDUCTION_BAND_STEPS = (2, 4, 8)
REDUCTION_FAMILY_COL = "family"
REDUCTION_VARIANT_COL = "variant"
REDUCTION_TUNING_RUN_COL = "tuning_run"
REDUCTION_BEST_PARAMS_COL = "best_params"
# PCA study (#213): dense at few components, where the curve bends, sparse above
PCA_STUDY_COMPONENTS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80)
PCA_STUDY_SENTINEL2_COMPONENTS = (2, 3, 5, 8)
N_COMPONENTS_COL = "n_components"
FIT_TIME_COL = "fit_s"  # wall time of preprocessing + model fit on one training fold
PREDICT_TIME_COL = "predict_s"  # wall time of preprocessing + prediction of one held-out fold
# a PCA variant within the loss rule is only worth validating if it at least halves the time
PCA_MIN_SPEEDUP = 2.0
# a reduced input counts as equivalent if it loses at most this much CV-BAcc against all bands
REDUCTION_MAX_CV_LOSS = 0.01
N_INPUT_FEATURES_COL = "n_input_features"  # field names of preprocessing.InputSize
N_MEASURED_BANDS_COL = "n_measured_bands"
STD_SUFFIX = "_std"
# group folds keep every group in one fold, so a model is scored on acquisitions it never saw
FoldGrouping = Literal["random", "aez_month", "month", "aez"]
RANDOM_FOLDS, AEZ_MONTH_FOLDS, MONTH_FOLDS, AEZ_FOLDS = get_args(FoldGrouping)
LABEL_SEP = "|"  # "_" occurs inside crop and stage names, so it cannot separate them

SEED = 42
# 80/20 gives the models more training rows; CV rose from 0.848 to 0.859 (combined_svm_split_80_20)
VAL_SIZE = 0.2
CV_FOLDS = 5
SCORE_DECIMALS = 4

# Regions for grouped permutation importance: neighbouring bands are nearly collinear, so
# permuting one band alone lets its neighbours stand in and hides what the region carries.
SPECTRAL_REGIONS_NM: dict[str, tuple[int, int]] = {
    "VIS": (400, 680),  # pigments: chlorophyll and carotenoid absorption
    "Red Edge": (680, 750),  # steep chlorophyll slope, shifts with stage and stress
    "NIR": (750, 1300),  # leaf structure and biomass
    "SWIR1": (1300, 1900),  # leaf water
    "SWIR2": (1900, 2500),  # water, lignin, cellulose: drying and senescence
}
META_GROUP = "Meta"
IMPORTANCE_GROUP_COL = "group"
IMPORTANCE_METRIC_COL = "metric"
IMPORTANCE_MEAN_COL = "mean"
IMPORTANCE_STD_COL = "std"
IMPORTANCE_REPEAT_COL = "repeat"
IMPORTANCE_DROP_COL = "drop"
IMPORTANCE_BASELINE_COL = "baseline"
RECALL_COL = "recall"
SUPPORT_COL = "support"
RANDOM_FOREST_N_ESTIMATORS = 300
EXTRA_TREES_N_ESTIMATORS = 300
LOGISTIC_REGRESSION_MAX_ITER = 5000  # lbfgs needs more than the default 100 on ~140 features
MLP_MAX_ITER = 500  # an upper bound only; early stopping ends training before
# a CV gain below this is within the fold-to-fold noise of the reference (std about 0.010)
MIN_CV_GAIN = 0.005

# Data-quality flags (#196) only mark rows for review; nothing is removed by them.
QualityFlag = Literal["haze", "dark", "ndvi_outlier", "rare_context", "class_outlier"]
LABEL_SUSPECT_COL = "label_suspect"
HAZE_FLAG, DARK_FLAG, NDVI_OUTLIER_FLAG, RARE_CONTEXT_FLAG, CLASS_OUTLIER_FLAG = get_args(
    QualityFlag
)
ANY_FLAG_COL = "any_flag"
# The blue reflectance (median 16 %) lies far above the 3-6 % of vegetation at the surface, so
# absolute literature thresholds (e.g. Fmask) do not transfer. Every rule therefore flags only
# Tukey's "far out" values: below Q1 - 3 IQR or above Q3 + 3 IQR (Tukey 1977).
TUKEY_FAR_OUT_FACTOR = 3.0
# Haze Optimized Transform as in Fmask (Zhu & Woodcock 2012): haze raises blue against red
HAZE_BLUE_NM = 470
HAZE_RED_NM = 660
HAZE_RED_WEIGHT = 0.5
# very low NIR points to shadow or open water instead of vegetation or soil
DARK_NIR_NM = 860
NDVI_INDEX = "ND_800_680"
# month x crop x stage combinations this rare in training are phenologically suspicious
RARE_CONTEXT_MAX_COUNT = 3
# class-wise outliers: robust Mahalanobis distance on PCA scores of each crop/stage class
OUTLIER_PCA_COMPONENTS = 5
OUTLIER_MIN_CLASS_ROWS = 30
# the consensus filter needs at least two independent opinions on a label
MIN_CONSENSUS_MODELS = 2
CNN_DEFAULT_FILTERS = 16
CNN_DEFAULT_KERNEL_SIZE = 5
CNN_DEFAULT_DROPOUT = 0.2
CNN_DEFAULT_LEARNING_RATE = 0.001
CNN_DEFAULT_MAX_EPOCHS = 50
CNN_DEFAULT_BATCH_SIZE = 64
CNN_EARLY_STOPPING_PATIENCE = 8
CNN_INNER_VALIDATION_FRACTION = 0.1
LSTM_DEFAULT_HIDDEN_SIZE = 64
LSTM_DEFAULT_NUM_LAYERS = 2
LSTM_DEFAULT_DROPOUT = 0.2
LSTM_DEFAULT_LEARNING_RATE = 0.001
LSTM_DEFAULT_WEIGHT_DECAY = 0.0
LSTM_DEFAULT_AEZ_EMBEDDING_DIM = 4
LSTM_DEFAULT_MONTH_EMBEDDING_DIM = 3
LSTM_DEFAULT_MAX_EPOCHS = 50
LSTM_DEFAULT_BATCH_SIZE = 64
LSTM_EARLY_STOPPING_PATIENCE = 8
LSTM_INNER_VALIDATION_FRACTION = 0.1
LSTM_DEFAULT_BANDS_PER_STEP = 1
LSTM_DEFAULT_HEAD_HIDDEN_SIZE = 0  # 0 keeps the single linear layer after the fusion
MLP_DEFAULT_HIDDEN_UNITS = 64
MLP_DEFAULT_HIDDEN_LAYERS = 2
MLP_DEFAULT_DROPOUT = 0.2
MLP_DEFAULT_LEARNING_RATE = 0.001
MLP_DEFAULT_WEIGHT_DECAY = 0.0
MLP_DEFAULT_MAX_EPOCHS = 50
MLP_DEFAULT_BATCH_SIZE = 64
MLP_EARLY_STOPPING_PATIENCE = 8
MLP_INNER_VALIDATION_FRACTION = 0.1
PERMUTATION_REPEATS = 3
PERMUTATION_N_JOBS = -1
REGION_PERMUTATION_REPEATS = 5  # regions are few and cheap, so they get more shuffles than bands
RANDOM_BAND_GROUP_COUNT = 40  # windows per region; fewer exist where a region fills most bands
NULL_QUANTILE = 0.95
COMBINED_METRIC = "bacc_combined"
GroupVariantKind = Literal["all", "only", "only_no_meta", "without", "meta_only"]
(
    ALL_GROUPS_KIND,
    ONLY_GROUP_KIND,
    ONLY_GROUP_NO_META_KIND,
    WITHOUT_GROUP_KIND,
    META_ONLY_KIND,
) = get_args(GroupVariantKind)
ALL_BANDS_VARIANT = "all-bands"
ALL_BANDS_NO_META_VARIANT = "all-bands-no-meta"
META_ONLY_VARIANT = "meta-only"
ONLY_VARIANT_PREFIX = "only"
WITHOUT_VARIANT_PREFIX = "without"
NO_META_VARIANT_SUFFIX = "no-meta"
GROUP_STUDY_RUN_PREFIX = "spectral_groups"
GROUP_STUDY_EXPERIMENT = "spectral-groups"  # own MLflow experiment: a question besides the ranking
GROUP_STUDY_APPROACH = "spectral-groups"
GROUP_STUDY_C_GRID = (10.0, 100.0, 1000.0)
GROUP_STUDY_GAMMA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)  # few bands need a wide kernel
GROUP_STUDY_C_PARAM = "estimator__C"
GROUP_STUDY_GAMMA_PARAM = "estimator__gamma"
GROUP_STUDY_N_JOBS = -1
GROUP_STUDY_CACHE_DIR = INTERIM_DATA_DIR / "spectral_groups_cache"  # resume after an interruption
ImportanceGrouping = Literal["band", "region"]
ImportanceKind = Literal["metric", "crop", "stage", "pair"]
BAND_GROUPING, REGION_GROUPING = get_args(ImportanceGrouping)
METRIC_KIND, CROP_KIND, STAGE_KIND, PAIR_KIND = get_args(ImportanceKind)
IMPORTANCE_LEVELS = ("group", "fold", "repeat")
IMPORTANCE_COLUMN_LEVELS = ("kind", "target")
IMPORTANCE_METRICS = ("bacc_crop", "bacc_stage", "bacc_combined")
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
MLFLOW_REDUCTION_EXPERIMENT = "band-reduction"  # how much input does the quality need
MLFLOW_SPLIT_EXPERIMENT = "split-leakage"  # do the scores hold for unseen acquisitions
# Marks the experiment as classic ML so the MLflow UI opens the "Model training" view.
MLFLOW_EXPERIMENT_KIND_TAG = ("mlflow.experimentKind", "custom_model_development")
MLFLOW_SYSTEM_METRICS_INTERVAL_S = (
    1  # our runs take seconds; MLflow's default of 10 s records nothing
)
MLFLOW_PARAM_MAX_CHARS = 250  # keeps logged parameter values short and readable in the UI
