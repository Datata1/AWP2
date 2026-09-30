"""Load the raw data as validated DataFrames and access their band columns."""

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
    """Load the labelled training data, validated against the raw-data schema.

    Args:
        path: CSV file to read.

    Returns:
        One row per spectrum (index ``id``) with ``AEZ``, ``Month``, ``Crop``, ``Stage`` and all
            band columns.
    """
    return _load(path, train_schema)


def load_test(path: Path = TEST_FILE) -> pd.DataFrame:
    """Load the unlabelled test data, validated against the raw-data schema.

    Args:
        path: CSV file to read.

    Returns:
        One row per spectrum (index ``id``) with ``AEZ``, ``Month`` and all band columns.
    """
    return _load(path, feature_schema)


def band_columns(df: pd.DataFrame) -> list[str]:
    """Select the spectral band columns.

    Args:
        df: Any frame that contains band columns (``X<nm>``).

    Returns:
        Band column names in wavelength order.
    """
    return [c for c in df.columns if re.fullmatch(BAND_PATTERN, str(c))]


def wavelengths(df: pd.DataFrame) -> list[int]:
    """Wavelengths of all band columns.

    Args:
        df: Any frame that contains band columns (``X<nm>``).

    Returns:
        Wavelengths in nm, in column order.
    """
    return [wavelength(c) for c in band_columns(df)]


def wavelength(band: str) -> int:
    """Wavelength of one band column.

    Args:
        band: Band column name, e.g. ``"X427"``.

    Returns:
        The wavelength in nm, e.g. ``427``.

    Raises:
        ValueError: If ``band`` is not a band column name.
    """
    if not re.fullmatch(BAND_PATTERN, band):
        raise ValueError(f"Not a band column: {band!r}")
    return int(band.removeprefix(BAND_PREFIX))
