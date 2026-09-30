"""Dataset preparation and the shared train/validation split."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from awp2.config import META_COLS, SEED, TARGET_COLS, VAL_SIZE
from awp2.data.load import band_columns

LABEL_SEP = "|"  # "_" is ambiguous: it occurs in crop and stage names


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labelled data into features ``X`` (metadata + bands) and targets ``y`` (Crop, Stage).

    Exact duplicates (same spectrum, metadata and labels) are dropped so that the same
    measurement cannot end up in both train and validation.
    """
    bands = band_columns(df)
    df = df.drop_duplicates(subset=[*META_COLS, *bands, *TARGET_COLS])
    return df[[*META_COLS, *bands]], df[list(TARGET_COLS)]


def combined_label(y: pd.DataFrame) -> pd.Series:
    """Crop and stage as one label, e.g. ``corn|Late``."""
    return y[TARGET_COLS[0]].astype(str) + LABEL_SEP + y[TARGET_COLS[1]].astype(str)


def split_combined_label(
    labels: pd.Series | np.ndarray, index: pd.Index | None = None
) -> pd.DataFrame:
    """Inverse of :func:`combined_label`."""
    parts = pd.Series(labels, index=index).str.split(LABEL_SEP, n=1, expand=True)
    parts.columns = list(TARGET_COLS)
    return parts


def valid_combinations(y: pd.DataFrame) -> set[tuple[str, str]]:
    """Crop/stage combinations that occur in ``y``."""
    return set(y.itertuples(index=False, name=None))


def balanced_sample_weight(y: pd.DataFrame) -> np.ndarray:
    """Sample weights that balance the crop+stage combinations (for models without
    ``class_weight``, e.g. XGBoost or MLP)."""
    return compute_sample_weight("balanced", combined_label(y))


def train_val_split(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The fixed holdout split everyone uses: ``VAL_SIZE`` of the data, stratified on crop+stage,
    seeded. Deliberately not configurable, so all results stay comparable.

    Returns ``X_train, X_val, y_train, y_val``.
    """
    return train_test_split(X, y, test_size=VAL_SIZE, stratify=combined_label(y), random_state=SEED)


def cv_splits(
    X: pd.DataFrame, y: pd.DataFrame, n_splits: int = 5, groups: pd.Series | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified cross-validation folds (train, validation index arrays).

    Use it on the **training part only** (``X_train, y_train`` from :func:`train_val_split`),
    e.g. for hyperparameter tuning – the 30 % validation split stays untouched for the final
    comparison. Pass ``groups`` (e.g. a field/cluster id) to keep each group within one fold.
    """
    kind = StratifiedGroupKFold if groups is not None else StratifiedKFold
    splitter = kind(n_splits=n_splits, shuffle=True, random_state=SEED)
    yield from splitter.split(X, combined_label(y), groups)
