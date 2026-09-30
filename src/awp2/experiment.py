"""Run a model on the shared split, evaluate it and track the run in MLflow."""

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from awp2.config import MLFLOW_EXPERIMENT, SLUG_PATTERN
from awp2.data import TrainValSplit, balanced_sample_weight, load_split, valid_combinations
from awp2.evaluation import Metrics, as_target_frame, evaluate, plot_confusion_matrices
from awp2.preprocessing import PreprocessingConfig, build_preprocessor
from awp2.tracking import TrackedRun, estimator_params, log_run


class RunConfig(BaseModel):
    """Everything that defines a run besides the model itself."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        pattern=SLUG_PATTERN,
        description="Short run name, lowercase without spaces, e.g. 'rf_baseline'.",
    )
    approach: str = Field(
        pattern=SLUG_PATTERN,
        description="Modelling approach the run belongs to, e.g. 'baseline' or 'hierarchical' – "
        "used to filter and group runs in MLflow.",
    )
    description: str = Field(default="", description="What was tried and why (free text).")
    experiment: str = Field(
        default=MLFLOW_EXPERIMENT,
        pattern=SLUG_PATTERN,
        description="MLflow experiment = the question the run answers; the default is the main "
        "task, other questions (e.g. the band-reduction study) get their own.",
    )
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing options – set scale=True for SVM, logistic regression, MLP.",
    )
    balance_samples: bool = Field(
        default=False,
        description="Pass balanced sample_weight (by crop+stage) to fit – for models without "
        "a class_weight option (XGBoost, HistGradientBoosting, MLP).",
    )
    track: bool = Field(default=True, description="Record the run in MLflow.")
    log_model: bool = Field(
        default=True,
        description="Store the fitted pipeline in MLflow (Models tab); switch off for quick tests.",
    )
    system_metrics: bool = Field(
        default=False, description="Record CPU/memory usage – worthwhile for long trainings."
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
    """

    config: RunConfig
    metrics: Metrics
    pipeline: Pipeline
    y_val: pd.DataFrame
    y_pred: pd.DataFrame
    run_id: str | None
    model_uri: str | None


def run(model: BaseEstimator, config: RunConfig) -> RunResult:
    """Fit standard preprocessing + ``model`` on the shared split and evaluate it.

    Trains on the training part of ``load_split()``, evaluates on the validation part and – if
    ``config.track`` – records parameters, metrics, confusion matrices and git state in MLflow.
    Use it to compare finished models; tune hyperparameters with ``load_folds()`` instead.

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
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(config.preprocessing)), ("model", clone(model))]
    )
    fit_params = (
        {"model__sample_weight": balanced_sample_weight(split.y_train)}
        if config.balance_samples
        else {}
    )
    pipeline.fit(split.X_train, split.y_train, **fit_params)
    y_pred = as_target_frame(pipeline.predict(split.X_val), split.y_val.index)
    metrics = evaluate(split.y_val, y_pred, valid_combinations(split.y_train))

    if metrics.invalid_combinations:
        warnings.warn(
            f"{config.name}: {metrics.invalid_combinations:.1%} of the predictions are impossible "
            "crop/stage combinations.",
            stacklevel=2,
        )

    tracked = _track(config, model, metrics, pipeline, split, y_pred) if config.track else None
    return RunResult(
        config,
        metrics,
        pipeline,
        split.y_val,
        y_pred,
        run_id=tracked.run_id if tracked else None,
        model_uri=tracked.model_uri if tracked else None,
    )


def _track(
    config: RunConfig,
    model: BaseEstimator,
    metrics: Metrics,
    pipeline: Pipeline,
    split: TrainValSplit,
    y_pred: pd.DataFrame,
) -> TrackedRun | None:
    params: dict[str, str | bool] = {
        **{
            f"preprocessing.{key}": value
            for key, value in config.preprocessing.model_dump().items()
        },
        "balance_samples": config.balance_samples,
        **estimator_params(model, "model"),
    }
    scores = {name: value for name, value in metrics.model_dump().items() if value is not None}
    figure = plot_confusion_matrices(split.y_val, y_pred)
    try:
        return log_run(
            experiment=config.experiment,
            name=config.name,
            approach=config.approach,
            description=config.description,
            params=params,
            metrics=scores,
            train=pd.concat([split.X_train, split.y_train], axis=1),
            validation=pd.concat([split.X_val, split.y_val], axis=1),
            model=pipeline if config.log_model else None,
            figures={"confusion_matrices.png": figure},
            system_metrics=config.system_metrics,
        )
    except Exception as error:  # noqa: BLE001 – a tracking problem must not discard the trained run
        warnings.warn(f"{config.name}: run could not be tracked ({error}).", stacklevel=3)
        return None
    finally:
        plt.close(figure)
