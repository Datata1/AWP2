"""Run a model on the shared split, evaluate it and track the run in MLflow."""

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from awp2.config import RUN_NAME_PATTERN
from awp2.data import balanced_sample_weight, load_split, valid_combinations
from awp2.evaluation import Metrics, as_target_frame, evaluate, plot_confusion_matrices
from awp2.preprocessing import PreprocessingConfig, build_preprocessor
from awp2.tracking import estimator_params, log_run


class RunConfig(BaseModel):
    """Everything that defines a run besides the model itself."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        pattern=RUN_NAME_PATTERN,
        description="Short run name, lowercase without spaces, e.g. 'rf_baseline'.",
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
        default=False, description="Also store the fitted pipeline in MLflow (can be large)."
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
    """

    config: RunConfig
    metrics: Metrics
    pipeline: Pipeline
    y_val: pd.DataFrame
    y_pred: pd.DataFrame
    run_id: str | None


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
        Metrics, fitted pipeline, validation labels and predictions, MLflow run id.

    Example:
        ```python
        result = run(RandomForestClassifier(class_weight="balanced"), RunConfig(name="rf"))
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

    run_id = _track(config, model, metrics, pipeline, split.y_val, y_pred) if config.track else None
    return RunResult(config, metrics, pipeline, split.y_val, y_pred, run_id)


def _track(
    config: RunConfig,
    model: BaseEstimator,
    metrics: Metrics,
    pipeline: Pipeline,
    y_val: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> str | None:
    params: dict[str, str | bool] = {
        **{
            f"preprocessing.{key}": value
            for key, value in config.preprocessing.model_dump().items()
        },
        "balance_samples": config.balance_samples,
        **estimator_params(model, "model"),
    }
    scores = {name: value for name, value in metrics.model_dump().items() if value is not None}
    figure = plot_confusion_matrices(y_val, y_pred)
    try:
        return log_run(
            config.name,
            params,
            scores,
            model=pipeline if config.log_model else None,
            figures={"confusion_matrices.png": figure},
        )
    except Exception as error:  # noqa: BLE001 – a tracking problem must not discard the trained run
        warnings.warn(f"{config.name}: run could not be tracked ({error}).", stacklevel=3)
        return None
    finally:
        plt.close(figure)
