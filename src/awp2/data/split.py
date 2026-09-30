"""Dataset preparation and the shared train/validation split."""

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from awp2.config import (
    CROP_COL,
    CV_FOLDS,
    LABEL_SEP,
    META_COLS,
    SEED,
    STAGE_COL,
    TARGET_COLS,
    VAL_SIZE,
    Crop,
    Stage,
)
from awp2.data.load import band_columns


class CropStage(BaseModel):
    """One valid pair of labels, e.g. ``CropStage(crop="corn", stage="Late")``."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    crop: Crop
    stage: Stage


class Dataset(NamedTuple):
    """Features (metadata + bands) and targets (``Crop``, ``Stage``) of labelled data."""

    X: pd.DataFrame
    y: pd.DataFrame


class TrainValSplit(NamedTuple):
    """The shared holdout split; unpacks as ``X_train, X_val, y_train, y_val``."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame


class Fold(NamedTuple):
    """Row positions of one cross-validation fold."""

    train: np.ndarray
    val: np.ndarray


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Separate features and targets of labelled data.

    Exact duplicates (same spectrum, metadata and labels) are dropped so that the same
    measurement cannot end up in both train and validation.
    """
    bands = band_columns(df)
    df = df.drop_duplicates(subset=[*META_COLS, *bands, *TARGET_COLS])
    return Dataset(X=df[[*META_COLS, *bands]], y=df[list(TARGET_COLS)])


def combined_label(y: pd.DataFrame) -> pd.Series:
    """Crop and stage as one string per row, e.g. ``corn|Late``.

    Stratification and ``StratifiedKFold`` accept only one label per row; this key lets them
    keep the proportions of every crop/stage pair.
    """
    return y[CROP_COL].astype(str) + LABEL_SEP + y[STAGE_COL].astype(str)


def valid_combinations(y: pd.DataFrame) -> frozenset[CropStage]:
    """All crop/stage pairs that occur in ``y``."""
    pairs = y[[CROP_COL, STAGE_COL]].drop_duplicates().itertuples(index=False)
    return frozenset(CropStage(crop=crop, stage=stage) for crop, stage in pairs)


def balanced_sample_weight(y: pd.DataFrame) -> np.ndarray:
    """Sample weights that balance the crop/stage pairs (for models without ``class_weight``,
    e.g. XGBoost or MLP)."""
    return compute_sample_weight("balanced", combined_label(y))


def train_val_split(X: pd.DataFrame, y: pd.DataFrame) -> TrainValSplit:
    """The fixed holdout split everyone uses: ``VAL_SIZE`` of the data, stratified on crop+stage,
    seeded. Deliberately not configurable, so all results stay comparable."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=combined_label(y), random_state=SEED
    )
    return TrainValSplit(X_train, X_val, y_train, y_val)


def cv_splits(X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series | None = None) -> Iterator[Fold]:
    """``CV_FOLDS`` stratified cross-validation folds.

    Use it on the **training part only** (``X_train, y_train`` from :func:`train_val_split`),
    e.g. for hyperparameter tuning – the validation split stays untouched for the final
    comparison. Pass ``groups`` (e.g. a field/cluster id) to keep each group within one fold.
    """
    kind = StratifiedGroupKFold if groups is not None else StratifiedKFold
    splitter = kind(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for train, val in splitter.split(X, combined_label(y), groups):
        yield Fold(train, val)
