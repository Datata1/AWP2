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
    """One valid pair of labels, e.g. ``CropStage(crop="corn", stage="Late")``.

    Attributes:
        crop: Crop label; only values from ``awp2.config.Crop`` are accepted.
        stage: Growth-stage label; only values from ``awp2.config.Stage`` are accepted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    crop: Crop
    stage: Stage


class Dataset(NamedTuple):
    """Labelled data split into features and targets.

    Attributes:
        X: Features: ``AEZ``, ``Month`` and all band columns.
        y: Targets: ``Crop`` and ``Stage``.
    """

    X: pd.DataFrame
    y: pd.DataFrame


class TrainValSplit(NamedTuple):
    """The shared holdout split; unpacks as ``X_train, X_val, y_train, y_val``.

    Attributes:
        X_train: Features of the training part.
        X_val: Features of the validation part.
        y_train: Targets of the training part.
        y_val: Targets of the validation part.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame


class Fold(NamedTuple):
    """One cross-validation fold; can be passed directly as ``cv=`` to sklearn.

    Attributes:
        train: Row positions (within ``X_train``) used for fitting.
        val: Row positions (within ``X_train``) used for scoring.
    """

    train: np.ndarray
    val: np.ndarray


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Separate features and targets of labelled data and drop exact duplicates.

    Duplicates (same spectrum, metadata and labels) are dropped so that the same measurement
    cannot end up in both train and validation.

    Args:
        df: Labelled data, e.g. from ``load_train()``.

    Returns:
        Features and targets without duplicate rows.
    """
    bands = band_columns(df)
    df = df.drop_duplicates(subset=[*META_COLS, *bands, *TARGET_COLS])
    return Dataset(X=df[[*META_COLS, *bands]], y=df[list(TARGET_COLS)])


def combined_label(y: pd.DataFrame) -> pd.Series:
    """Join crop and stage into one key per row, e.g. ``corn|Late``.

    Stratification and ``StratifiedKFold`` accept only one label per row; this key lets them
    keep the proportions of every crop/stage pair.

    Args:
        y: Targets with the columns ``Crop`` and ``Stage``.

    Returns:
        One string per row, same index as ``y``.
    """
    return y[CROP_COL].astype(str) + LABEL_SEP + y[STAGE_COL].astype(str)


def valid_combinations(y: pd.DataFrame) -> frozenset[CropStage]:
    """Collect the crop/stage pairs that occur in ``y``.

    Args:
        y: Targets with the columns ``Crop`` and ``Stage``, usually the training part.

    Returns:
        Every pair that occurs at least once.
    """
    pairs = y[[CROP_COL, STAGE_COL]].drop_duplicates().itertuples(index=False)
    return frozenset(CropStage(crop=crop, stage=stage) for crop, stage in pairs)


def balanced_sample_weight(y: pd.DataFrame) -> np.ndarray:
    """Weights that give every crop/stage pair the same total weight.

    For models without a ``class_weight`` option (e.g. XGBoost, HistGradientBoosting, MLP):
    pass the result as ``sample_weight`` to ``fit``.

    Args:
        y: Targets of the rows the model is fitted on.

    Returns:
        One weight per row of ``y``.
    """
    return compute_sample_weight("balanced", combined_label(y))


def train_val_split(X: pd.DataFrame, y: pd.DataFrame) -> TrainValSplit:
    """Compute the shared holdout split (used by ``make data``; load it with ``load_split``).

    ``VAL_SIZE`` of the rows go to validation, stratified on crop+stage and seeded. Deliberately
    not configurable, so all results stay comparable.

    Args:
        X: Features, e.g. ``Dataset.X``.
        y: Targets, e.g. ``Dataset.y``.

    Returns:
        Training and validation part.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=combined_label(y), random_state=SEED
    )
    return TrainValSplit(X_train, X_val, y_train, y_val)


def cv_splits(X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series | None = None) -> Iterator[Fold]:
    """Compute ``CV_FOLDS`` stratified folds (used by ``make data``; load them with ``load_folds``).

    Only for the **training part** – the validation part stays untouched for the final
    comparison.

    Args:
        X: Features of the training part.
        y: Targets of the training part.
        groups: Optional group id per row (e.g. a field); each group then stays within one fold.

    Yields:
        One fold after the other, as row positions within ``X``.
    """
    kind = StratifiedGroupKFold if groups is not None else StratifiedKFold
    splitter = kind(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for train, val in splitter.split(X, combined_label(y), groups):
        yield Fold(train, val)
