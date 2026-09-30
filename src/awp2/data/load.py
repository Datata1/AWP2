"""Load the raw data as validated DataFrames (index = id)."""

import re
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from awp2.config import BAND_PATTERN, BAND_PREFIX, ID_COL, TEST_FILE, TRAIN_FILE
from awp2.data.schema import feature_schema, train_schema


def _load(path: Path, schema: pa.DataFrameSchema) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)  # first column is only a row index
    df = schema.validate(df)
    return df.set_index(ID_COL)


def load_train(path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Labeled data: AEZ, Month, Crop, Stage and all spectral bands."""
    return _load(path, train_schema)


def load_test(path: Path = TEST_FILE) -> pd.DataFrame:
    """Unlabeled data: AEZ, Month and all spectral bands."""
    return _load(path, feature_schema)


def band_columns(df: pd.DataFrame) -> list[str]:
    """Names of the spectral band columns in wavelength order."""
    return [c for c in df.columns if re.fullmatch(BAND_PATTERN, str(c))]


def wavelengths(df: pd.DataFrame) -> list[int]:
    """Band wavelengths in nm, e.g. X427 -> 427."""
    return [wavelength(c) for c in band_columns(df)]


def wavelength(band: str) -> int:
    """Wavelength in nm of a band column name, e.g. ``"X427"`` -> ``427``."""
    if not re.fullmatch(BAND_PATTERN, band):
        raise ValueError(f"Not a band column: {band!r}")
    return int(band.removeprefix(BAND_PREFIX))
