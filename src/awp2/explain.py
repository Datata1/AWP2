"""Permutation importance of single bands, spectral regions and metadata for a fitted pipeline."""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from awp2.config import (
    AEZ_COL,
    BAND_GROUPING,
    BAND_PATTERN,
    CROP_COL,
    CROP_KIND,
    IMPORTANCE_COLUMN_LEVELS,
    IMPORTANCE_LEVELS,
    IMPORTANCE_METRICS,
    METRIC_KIND,
    MONTH_COL,
    PAIR_KIND,
    PERMUTATION_N_JOBS,
    PERMUTATION_REPEATS,
    SEED,
    SPECTRAL_REGIONS,
    STAGE_COL,
    STAGE_KIND,
    ImportanceGrouping,
    SpectralRegion,
)
from awp2.data import combined_label, load_folds, load_split, wavelength
from awp2.data.split import Fold
from awp2.evaluation import as_target_frame
from awp2.experiment import (
    _MODEL_STEP,
    _PREPROCESS_STEP,
    RunConfig,
    TuneConfig,
    _fit_params,
    _pipeline,
)

Predictor = Callable[[pd.DataFrame], pd.DataFrame]
"""Maps model-ready features to predictions with the columns ``Crop`` and ``Stage``."""


class PermutationImportance(NamedTuple):
    """Drop in recall and balanced accuracy when a feature group is shuffled.

    The columns of all frames are a two-level index of ``kind`` (``metric``, ``crop``, ``stage``
    or ``pair``) and ``target`` (a metric name or a class). A larger drop means a more important
    feature group.

    Attributes:
        baseline: Unpermuted score per column, mean over the folds.
        mean: Mean drop per feature group (rows) over folds and repeats.
        std: Standard deviation of the drop over folds and repeats.
        drops: Every single drop, indexed by group, fold and repeat.
    """

    baseline: pd.Series
    mean: pd.DataFrame
    std: pd.DataFrame
    drops: pd.DataFrame


def feature_groups(
    columns: Sequence[str],
    grouping: ImportanceGrouping,
    regions: Sequence[SpectralRegion] = SPECTRAL_REGIONS,
) -> dict[str, list[str]]:
    """Group the model-ready feature columns for permutation.

    Spectral bands form one group each (``band``) or one group per spectral region (``region``).
    One-hot AEZ and the month encoding are always permuted as a whole; every other column
    (albedo, indices, ...) is its own group.

    Args:
        columns: Column names of the preprocessed features.
        grouping: ``band`` or ``region`` for the spectral bands.
        regions: Wavelength intervals used for ``region``; bands outside all of them are ignored.

    Returns:
        Group name to the columns it contains, in the order of the columns.
    """
    groups: dict[str, list[str]] = {}
    for column in columns:
        if re.fullmatch(BAND_PATTERN, column):
            name = _band_group(column, grouping, regions)
        elif column.startswith(AEZ_COL):
            name = AEZ_COL
        elif column.startswith(MONTH_COL):
            name = MONTH_COL
        else:
            name = column
        if name is not None:
            groups.setdefault(name, []).append(column)
    return groups


def _band_group(
    band: str, grouping: ImportanceGrouping, regions: Sequence[SpectralRegion]
) -> str | None:
    if grouping == BAND_GROUPING:
        return band
    center = wavelength(band)
    return next((r.name for r in regions if r.start_nm <= center < r.end_nm), None)


def _scores(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Score recall per class and balanced accuracy for crop, stage and their combination.

    Returns:
        Score per ``(kind, target)``.
    """
    targets = (
        (CROP_KIND, y_true[CROP_COL], y_pred[CROP_COL]),
        (STAGE_KIND, y_true[STAGE_COL], y_pred[STAGE_COL]),
        (PAIR_KIND, combined_label(y_true), combined_label(y_pred)),
    )
    scores: dict[tuple[str, str], float] = {}
    for (kind, true, pred), metric in zip(targets, IMPORTANCE_METRICS, strict=True):
        hit = pd.Series(true.to_numpy() == pred.to_numpy(), index=true.to_numpy())
        recall = hit.groupby(level=0).mean()
        scores.update({(kind, str(label)): float(value) for label, value in recall.items()})
        scores[(METRIC_KIND, metric)] = float(recall.mean())
    return scores


def _permutation_drops(
    predict: Predictor,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    n_repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str], float], list[tuple[str, int, dict[tuple[str, str], float]]]]:
    """Score ``predict`` on ``X`` once and after shuffling each group ``n_repeats`` times.

    Returns:
        The unpermuted scores and, per group and repeat, the drop of each score.
    """
    baseline = _scores(y, predict(X))
    drops = []
    for name, columns in groups.items():
        values = X[list(columns)].to_numpy()
        for repeat in range(n_repeats):
            shuffled = X.copy()
            shuffled[list(columns)] = values[rng.permutation(len(X))]
            permuted = _scores(y, predict(shuffled))
            drops.append((name, repeat, {key: baseline[key] - permuted[key] for key in baseline}))
    return baseline, drops


def _fold_drops(
    model: BaseEstimator,
    config: TuneConfig | RunConfig,
    fold: Fold,
    fold_number: int,
    grouping: ImportanceGrouping,
    n_repeats: int,
) -> tuple[dict[tuple[str, str], float], list[tuple[str, int, int, dict[tuple[str, str], float]]]]:
    split = load_split()
    y_fold = split.y_train.iloc[fold.train]
    pipeline = _pipeline(model, config.preprocessing)
    pipeline.fit(split.X_train.iloc[fold.train], y_fold, **_fit_params(config, y_fold))
    X_val = pipeline.named_steps[_PREPROCESS_STEP].transform(split.X_train.iloc[fold.val])
    y_val = split.y_train.iloc[fold.val]
    estimator = pipeline.named_steps[_MODEL_STEP]

    def predict(features: pd.DataFrame) -> pd.DataFrame:
        return as_target_frame(estimator.predict(features), features.index)

    baseline, drops = _permutation_drops(
        predict,
        X_val,
        y_val,
        feature_groups(list(X_val.columns), grouping),
        n_repeats,
        np.random.default_rng([SEED, fold_number]),
    )
    return baseline, [(group, fold_number, repeat, drop) for group, repeat, drop in drops]


def permutation_importance(
    model: BaseEstimator,
    config: TuneConfig | RunConfig,
    grouping: ImportanceGrouping = BAND_GROUPING,
    n_repeats: int = PERMUTATION_REPEATS,
) -> PermutationImportance:
    """Shuffle feature groups of the preprocessed validation folds and measure the score drop.

    Fits preprocessing + ``model`` on each cross-validation fold of the training part exactly
    like :func:`awp2.experiment.tune`, shuffles each feature group of the held-out fold and
    compares the recall per class and the balanced accuracy with the unpermuted prediction. The
    validation part of the shared split is never touched, so the result may inform later
    feature decisions without leakage.

    Args:
        model: Estimator that predicts ``Crop`` **and** ``Stage``; it is cloned per fold.
        config: Configuration the model was tuned or evaluated with, e.g. ``TuneResult.config``.
        grouping: ``band`` shuffles every band alone, ``region`` shuffles whole spectral regions
            (adjacent bands are strongly correlated, so single bands understate their region).
        n_repeats: Shuffles per group and fold.

    Returns:
        Baseline scores and the drop per feature group, aggregated over folds and repeats.
    """
    results = Parallel(n_jobs=PERMUTATION_N_JOBS)(
        delayed(_fold_drops)(model, config, fold, number, grouping, n_repeats)
        for number, fold in enumerate(load_folds())
    )
    baseline = pd.DataFrame([fold_baseline for fold_baseline, _ in results]).mean()
    records = [record for _, fold_records in results for record in fold_records]
    return _aggregate(baseline, records)


def _aggregate(
    baseline: pd.Series,
    records: Sequence[tuple[str, int, int, Mapping[tuple[str, str], float]]],
) -> PermutationImportance:
    index = pd.MultiIndex.from_tuples(
        [(group, fold, repeat) for group, fold, repeat, _ in records], names=IMPORTANCE_LEVELS
    )
    drops = pd.DataFrame([drop for *_, drop in records], index=index)
    drops.columns = pd.MultiIndex.from_tuples(drops.columns, names=IMPORTANCE_COLUMN_LEVELS)
    baseline.index = pd.MultiIndex.from_tuples(baseline.index, names=IMPORTANCE_COLUMN_LEVELS)
    by_group = drops.groupby(level=IMPORTANCE_LEVELS[0], sort=False)
    return PermutationImportance(baseline, by_group.mean(), by_group.std(), drops)
