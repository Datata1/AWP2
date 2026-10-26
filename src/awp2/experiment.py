"""Tune and evaluate models on the shared split and track both steps in MLflow."""

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from awp2.config import (
    AEZ_COL,
    AEZ_FOLDS,
    AEZ_MONTH_FOLDS,
    AEZ_MONTH_SEP,
    FIT_TIME_COL,
    MLFLOW_EXPERIMENT,
    MONTH_COL,
    PREDICT_TIME_COL,
    RANDOM_FOLDS,
    SLUG_PATTERN,
    FoldGrouping,
)
from awp2.data import (
    Fold,
    TrainValSplit,
    balanced_sample_weight,
    cv_splits,
    load_folds,
    load_split,
    load_split_config,
    valid_combinations,
)
from awp2.evaluation import (
    MetricName,
    Metrics,
    as_target_frame,
    evaluate,
    plot_confusion_matrices,
    scorer,
)
from awp2.preprocessing import (
    InputSize,
    PreprocessingConfig,
    build_preprocessor,
    describe_preprocessor,
    input_size,
)
from awp2.tracking import TrackedRun, estimator_params, log_run, log_tuning

TrainFilter = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
"""Marks training rows (``True``) to leave out, given raw features and targets."""

_MODEL_STEP = "model"
_PREPROCESS_STEP = "preprocess"


class _ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        pattern=SLUG_PATTERN,
        description="Short name, lowercase without spaces, e.g. 'rf_baseline'.",
    )
    approach: str = Field(
        pattern=SLUG_PATTERN,
        description="Modelling approach, e.g. 'baseline' or 'hierarchical' – used to filter and "
        "group runs in MLflow.",
    )
    description: str = Field(default="", description="What was tried and why (free text).")
    experiment: str = Field(
        default=MLFLOW_EXPERIMENT,
        pattern=SLUG_PATTERN,
        description="MLflow experiment = the question the run answers; the default is the main "
        "task, other questions (e.g. the band-reduction study) get their own.",
    )
    folds: FoldGrouping = Field(
        default=RANDOM_FOLDS,
        description="Cross-validation folds: the shared random folds, or group folds that keep "
        "each AEZ x month cell, month or AEZ in one fold to score unseen acquisitions.",
    )
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing options – set use_spectral_standard_scale/minmax_scale "
        "for SVM, logistic regression, MLP.",
    )
    balance_samples: bool = Field(
        default=False,
        description="Pass balanced sample_weight (by crop+stage) to fit – for models without "
        "a class_weight option (XGBoost, HistGradientBoosting, MLP).",
    )
    track: bool = Field(default=True, description="Record the run in MLflow.")
    system_metrics: bool = Field(
        default=False, description="Record CPU/memory usage – worthwhile for long trainings."
    )


class RunConfig(_ExperimentConfig):
    """Everything that defines an evaluation run besides the model itself."""

    log_model: bool = Field(
        default=True,
        description="Store the fitted pipeline in MLflow (Models tab); switch off for quick tests.",
    )
    tuning_run: str | None = Field(
        default=None,
        description="Run id of the tuning run the hyperparameters came from (from `tune()`).",
    )


class TuneConfig(_ExperimentConfig):
    """Everything that defines a hyperparameter search besides the model itself."""

    param_grid: dict[str, list[Any]] = Field(
        min_length=1,
        description="Hyperparameters of the model and the values to try, e.g. "
        "{'max_depth': [10, 20, None]}; for wrapped models use sklearn's names "
        "('estimator__max_depth').",
    )
    metric: MetricName = Field(
        default="bacc_combined", description="Cross-validation score used to rank candidates."
    )


@dataclass(frozen=True)
class RunResult:
    """Outcome of :func:`run`.

    Attributes:
        config: The configuration the run was started with.
        metrics: Scores on the validation part.
        pipeline: The fitted preprocessing + model.
        y_val: True labels of the validation part.
        y_pred: Predicted labels of the validation part.
        run_id: MLflow run id, ``None`` if the run was not tracked.
        model_uri: URI to load the stored model with ``mlflow.sklearn.load_model``, ``None`` if
            no model was stored.
        input_size: Model features and measured bands of the fitted preprocessing.
    """

    config: RunConfig
    metrics: Metrics
    pipeline: Pipeline
    y_val: pd.DataFrame
    y_pred: pd.DataFrame
    run_id: str | None
    model_uri: str | None
    input_size: InputSize


@dataclass(frozen=True)
class TuneResult:
    """Outcome of :func:`tune`.

    Attributes:
        config: The configuration the search was started with.
        best_params: Hyperparameters of the best candidate.
        best_score: Its mean cross-validation score.
        best_model: Unfitted copy of the model with ``best_params`` – pass it to :func:`run`.
        candidates: One row per candidate: ``params``, ``mean``, ``std``, ``rank``, ``fold_<i>``.
        run_id: MLflow run id of the tuning run, ``None`` if it was not tracked.
        input_size: Model features and measured bands of the preprocessing, fitted on the
            whole training part.
    """

    config: TuneConfig
    best_params: dict[str, Any]
    best_score: float
    best_model: BaseEstimator
    candidates: pd.DataFrame
    run_id: str | None
    input_size: InputSize


def cv_folds(grouping: FoldGrouping = RANDOM_FOLDS) -> list[Fold]:
    """Cross-validation folds of the training part.

    Rows of one AEZ x month cell probably come from the same acquisition (same atmosphere and
    illumination); random folds then let a model recognise the acquisition. Group folds keep a
    whole group in one fold and score how well the model carries over to unseen ones.

    Args:
        grouping: ``"random"`` for the shared folds of ``make data``, or ``"aez_month"``,
            ``"month"`` or ``"aez"`` for stratified group folds.

    Returns:
        ``CV_FOLDS`` folds as row positions within ``load_split().X_train``.
    """
    if grouping == RANDOM_FOLDS:
        return load_folds()
    split = load_split()
    X = split.X_train
    if grouping == AEZ_MONTH_FOLDS:
        groups = X[AEZ_COL].astype(str) + AEZ_MONTH_SEP + X[MONTH_COL].astype(str)
    elif grouping == AEZ_FOLDS:
        groups = X[AEZ_COL]
    else:
        groups = X[MONTH_COL]
    return list(cv_splits(X, split.y_train, groups=groups))


def _pipeline(model: BaseEstimator, preprocessing: PreprocessingConfig) -> Pipeline:
    return Pipeline(
        [(_PREPROCESS_STEP, build_preprocessor(preprocessing)), (_MODEL_STEP, clone(model))]
    )


def _fit_params(config: _ExperimentConfig, y_train: pd.DataFrame) -> dict[str, np.ndarray]:
    if not config.balance_samples:
        return {}
    return {f"{_MODEL_STEP}__sample_weight": balanced_sample_weight(y_train)}


def _pipeline_summary(config: _ExperimentConfig, pipeline: Pipeline | None = None) -> str:
    preprocessor = (
        pipeline.named_steps[_PREPROCESS_STEP]
        if pipeline is not None
        else build_preprocessor(config.preprocessing)
    )
    steps = " | ".join(describe_preprocessor(preprocessor))
    if config.description:
        return f"{config.description}\nPipeline: {steps}"
    return f"Pipeline: {steps}"


def _shared_params(config: _ExperimentConfig) -> dict[str, str | bool | float]:
    return {
        **{
            f"preprocessing.{key}": value
            for key, value in config.preprocessing.model_dump().items()
        },
        "balance_samples": config.balance_samples,
        "split.validation_size": load_split_config().validation_size,
        "cv.folds": config.folds,
    }


def tune(model: BaseEstimator, config: TuneConfig) -> TuneResult:
    """Search hyperparameters by cross-validation on the training part.

    Uses the folds of ``config.folds`` (``cv_folds``) and never touches the validation part, so the
    subsequent :func:`run` on the validation part stays an honest estimate. If
    ``config.track``, MLflow gets one tuning run with a nested run per candidate.

    Args:
        model: Unfitted estimator that predicts ``Crop`` **and** ``Stage``; it is cloned.
        config: Name, grid and options of the search.

    Returns:
        Best hyperparameters and score, an unfitted best model and all candidates.

    Raises:
        ValueError: If ``param_grid`` names a hyperparameter the model does not have.

    Example:
        ```python
        tuned = tune(
            RandomForestClassifier(class_weight="balanced", random_state=SEED),
            TuneConfig(name="rf_depth", approach="baseline", param_grid={"max_depth": [10, None]}),
        )
        result = run(
            tuned.best_model,
            RunConfig(name="rf_tuned", approach="baseline", tuning_run=tuned.run_id),
        )
        ```
    """
    unknown = set(config.param_grid) - set(model.get_params(deep=True))
    if unknown:
        raise ValueError(f"{type(model).__name__} has no hyperparameter(s) {sorted(unknown)}.")

    split = load_split()
    search = GridSearchCV(
        _pipeline(model, config.preprocessing),
        {f"{_MODEL_STEP}__{key}": values for key, values in config.param_grid.items()},
        scoring=scorer(config.metric),
        cv=cv_folds(config.folds),
        refit=False,
    )
    search.fit(split.X_train, split.y_train, **_fit_params(config, split.y_train))

    results = search.cv_results_
    fold_scores = {
        key.removeprefix("split").removesuffix("_test_score"): results[key]
        for key in results
        if key.startswith("split") and key.endswith("_test_score")
    }
    candidates = pd.DataFrame(
        {
            "params": [
                {key.removeprefix(f"{_MODEL_STEP}__"): value for key, value in params.items()}
                for params in results["params"]
            ],
            "mean": results["mean_test_score"],
            "std": results["std_test_score"],
            "rank": results["rank_test_score"],
            **{f"fold_{number}": scores for number, scores in fold_scores.items()},
        }
    )
    best = candidates.loc[candidates["rank"].idxmin()]
    size = input_size(build_preprocessor(config.preprocessing).fit(split.X_train, split.y_train))
    run_id = _track_tuning(config, model, candidates, split, size) if config.track else None
    return TuneResult(
        config=config,
        best_params=best["params"],
        best_score=float(best["mean"]),
        best_model=clone(model).set_params(**best["params"]),
        candidates=candidates,
        run_id=run_id,
        input_size=size,
    )


def run(model: BaseEstimator, config: RunConfig) -> RunResult:
    """Fit standard preprocessing + ``model`` on the shared split and evaluate it.

    Trains on the training part of ``load_split()``, evaluates on the validation part and – if
    ``config.track`` – records parameters, metrics, confusion matrices and git state in MLflow.
    Use it to evaluate a finished model once; search hyperparameters with :func:`tune` first.

    Args:
        model: Unfitted estimator that predicts ``Crop`` **and** ``Stage`` (natively
            multi-output or wrapped); it is cloned, the passed object stays untouched.
        config: Name and options of the run.

    Returns:
        Metrics, fitted pipeline, validation labels and predictions, MLflow run id and model URI.

    Example:
        ```python
        result = run(
            RandomForestClassifier(class_weight="balanced"),
            RunConfig(name="rf", approach="baseline"),
        )
        result.metrics.bacc_combined
        ```
    """
    split = load_split()
    pipeline = _pipeline(model, config.preprocessing)
    pipeline.fit(split.X_train, split.y_train, **_fit_params(config, split.y_train))
    y_pred = as_target_frame(pipeline.predict(split.X_val), split.y_val.index)
    metrics = evaluate(split.y_val, y_pred, valid_combinations(split.y_train))

    if metrics.invalid_combinations:
        warnings.warn(
            f"{config.name}: {metrics.invalid_combinations:.1%} of the predictions are impossible "
            "crop/stage combinations.",
            stacklevel=2,
        )

    size = input_size(pipeline.named_steps[_PREPROCESS_STEP])
    tracked = (
        _track_run(config, model, metrics, pipeline, split, y_pred, size) if config.track else None
    )
    return RunResult(
        config,
        metrics,
        pipeline,
        split.y_val,
        y_pred,
        run_id=tracked.run_id if tracked else None,
        model_uri=tracked.model_uri if tracked else None,
        input_size=size,
    )


class TimedPredictions(NamedTuple):
    """Out-of-fold predictions together with the time each fold took.

    Attributes:
        predictions: Columns ``Crop`` and ``Stage``, in the row order of
            ``load_split().y_train``.
        timings: One row per fold with the wall times ``fit_s`` (preprocessing + model fit on
            the training folds) and ``predict_s`` (preprocessing + prediction of the held-out
            fold) in seconds.
    """

    predictions: pd.DataFrame
    timings: pd.DataFrame


def out_of_fold_predictions(
    model: BaseEstimator,
    config: TuneConfig | RunConfig,
    train_filter: TrainFilter | None = None,
) -> pd.DataFrame:
    """Predict every training row with preprocessing + ``model`` fitted on the other folds.

    Uses the folds of the training part from ``cv_folds(config.folds)`` and never touches the
    validation part – for analysing errors before the single :func:`run`. Fits exactly like
    :func:`tune`/:func:`run` (``config.preprocessing``, ``config.balance_samples``); nothing is
    tracked.

    Args:
        model: Estimator that predicts ``Crop`` **and** ``Stage``; it is cloned per fold.
        config: Configuration the model was tuned or evaluated with, e.g. ``TuneResult.config``.
        train_filter: Optional rule that marks rows to leave out of training, e.g. outliers.
            It sees only the training rows of each fold; every held-out row is still predicted.

    Returns:
        Out-of-fold predictions with the columns ``Crop`` and ``Stage``, in the row order of
            ``load_split().y_train``.
    """
    return timed_out_of_fold_predictions(model, config, train_filter).predictions


def timed_out_of_fold_predictions(
    model: BaseEstimator,
    config: TuneConfig | RunConfig,
    train_filter: TrainFilter | None = None,
) -> TimedPredictions:
    """Like :func:`out_of_fold_predictions`, and measure fit and prediction time per fold.

    Args:
        model: Estimator that predicts ``Crop`` **and** ``Stage``; it is cloned per fold.
        config: Configuration the model was tuned or evaluated with, e.g. ``TuneResult.config``.
        train_filter: Optional rule that marks rows to leave out of training, e.g. outliers.

    Returns:
        The out-of-fold predictions and the wall times of every fold.
    """
    split = load_split()
    predictions = []
    timings = []
    for fold in cv_folds(config.folds):
        X_fold, y_fold = split.X_train.iloc[fold.train], split.y_train.iloc[fold.train]
        if train_filter is not None:
            keep = ~train_filter(X_fold, y_fold)
            X_fold, y_fold = X_fold[keep], y_fold[keep]
        pipeline = _pipeline(model, config.preprocessing)
        start = time.perf_counter()
        pipeline.fit(X_fold, y_fold, **_fit_params(config, y_fold))
        fitted = time.perf_counter()
        predicted = pipeline.predict(split.X_train.iloc[fold.val])
        done = time.perf_counter()
        timings.append({FIT_TIME_COL: fitted - start, PREDICT_TIME_COL: done - fitted})
        predictions.append(as_target_frame(predicted, split.X_train.index[fold.val]))
    return TimedPredictions(
        predictions=pd.concat(predictions).loc[split.y_train.index],
        timings=pd.DataFrame(timings),
    )


def _track_run(
    config: RunConfig,
    model: BaseEstimator,
    metrics: Metrics,
    pipeline: Pipeline,
    split: TrainValSplit,
    y_pred: pd.DataFrame,
    size: InputSize,
) -> TrackedRun | None:
    scores = {name: value for name, value in metrics.model_dump().items() if value is not None}
    scores.update(size._asdict())
    figure = plot_confusion_matrices(split.y_val, y_pred)
    try:
        return log_run(
            experiment=config.experiment,
            name=config.name,
            approach=config.approach,
            description=_pipeline_summary(config, pipeline),
            params={**_shared_params(config), **estimator_params(model, "model")},
            metrics=scores,
            train=pd.concat([split.X_train, split.y_train], axis=1),
            validation=pd.concat([split.X_val, split.y_val], axis=1),
            model=pipeline if config.log_model else None,
            figures={"confusion_matrices.png": figure},
            tags={"tuning_run": config.tuning_run} if config.tuning_run else None,
            system_metrics=config.system_metrics,
        )
    except Exception as error:  # noqa: BLE001 – a tracking problem must not discard the trained run
        warnings.warn(f"{config.name}: run could not be tracked ({error}).", stacklevel=3)
        return None
    finally:
        plt.close(figure)


def _track_tuning(
    config: TuneConfig,
    model: BaseEstimator,
    candidates: pd.DataFrame,
    split: TrainValSplit,
    size: InputSize,
) -> str | None:
    try:
        return log_tuning(
            experiment=config.experiment,
            name=config.name,
            approach=config.approach,
            description=_pipeline_summary(config),
            params={**_shared_params(config), **estimator_params(model, "model")},
            metric=config.metric,
            candidates=candidates,
            train=pd.concat([split.X_train, split.y_train], axis=1),
            extra_metrics=size._asdict(),
            system_metrics=config.system_metrics,
        )
    except Exception as error:  # noqa: BLE001 – a tracking problem must not discard the search
        warnings.warn(f"{config.name}: tuning could not be tracked ({error}).", stacklevel=3)
        return None
