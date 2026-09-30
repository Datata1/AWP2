"""Experiment tracking with a local MLflow store (``mlflow.db`` + ``mlruns/``, not in git).

Only this module talks to MLflow, so the tracking tool can be swapped without touching the
rest of the code. Browse the runs with ``make mlflow``.
"""

import subprocess
import warnings
from collections.abc import Mapping
from typing import NamedTuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from mlflow.exceptions import MlflowException
from sklearn.base import BaseEstimator

from awp2.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_KIND_TAG,
    MLFLOW_PARAM_MAX_CHARS,
    MLFLOW_SYSTEM_METRICS_INTERVAL_S,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    SPLIT_FILE,
)

_LOCAL_STORE_SCHEMES = ("sqlite:", "file:")
# MLflow warns about a doubly registered local dataset source and about integer columns in the
# inferred schema; we never use the dataset schema to validate inputs, so both are noise here.
_IRRELEVANT_DATASET_WARNINGS = ("can be interpreted in multiple ways", "Hint: Inferred schema")
_UNKNOWN = "unknown"


class TrackedRun(NamedTuple):
    """Where a run ended up in MLflow.

    Attributes:
        run_id: MLflow run id.
        model_uri: URI to load the stored model with ``mlflow.sklearn.load_model``
            (``models:/<id>``), ``None`` if no model was stored.
    """

    run_id: str
    model_uri: str | None


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT)
    except FileNotFoundError:
        return _UNKNOWN
    return out.stdout.strip() if out.returncode == 0 else _UNKNOWN


def _use_experiment(name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    kind_key, kind_value = MLFLOW_EXPERIMENT_KIND_TAG
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        # A shared server decides itself where artifacts go; only a local store gets ours.
        is_local = MLFLOW_TRACKING_URI.startswith(_LOCAL_STORE_SCHEMES)
        location = (MLFLOW_ARTIFACTS_DIR / name).as_uri() if is_local else None
        try:
            mlflow.create_experiment(name, artifact_location=location, tags={kind_key: kind_value})
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":  # created by a parallel run
                raise
    elif experiment.lifecycle_stage == "deleted":
        mlflow.MlflowClient().restore_experiment(experiment.experiment_id)
    active = mlflow.set_experiment(name)
    if active.tags.get(kind_key) != kind_value:
        mlflow.set_experiment_tag(kind_key, kind_value)


def estimator_params(estimator: BaseEstimator, prefix: str) -> dict[str, str]:
    """Flatten the hyperparameters of an sklearn estimator for logging.

    Keys keep sklearn's naming (``estimator__max_depth`` for wrapped models), so they match what
    ``GridSearchCV`` expects.

    Args:
        estimator: Any sklearn estimator, including pipelines and wrappers.
        prefix: Prefix for all keys, e.g. ``"model"``.

    Returns:
        ``{prefix: class name, "prefix.param": value, ...}``; nested estimators appear as their
            class name, all other values as their (shortened) ``repr``.
    """
    params = {prefix: type(estimator).__name__}
    for key, value in estimator.get_params(deep=True).items():
        text = type(value).__name__ if isinstance(value, BaseEstimator) else repr(value)
        params[f"{prefix}.{key}"] = text[:MLFLOW_PARAM_MAX_CHARS]
    return params


def log_run(
    *,
    experiment: str,
    name: str,
    approach: str,
    description: str,
    params: Mapping[str, str | bool],
    metrics: Mapping[str, float],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model: BaseEstimator | None = None,
    figures: Mapping[str, plt.Figure] | None = None,
    system_metrics: bool = False,
) -> TrackedRun:
    """Record one run in MLflow.

    Besides parameters and metrics, the run gets the git state (commit, branch, author, whether
    ``src/`` or ``notebooks/`` had uncommitted changes), the approach as tag, the training and
    validation data as datasets (metadata and hash only) and – if given – the fitted model, to
    which the metrics are linked so it shows up with its scores in the experiment's *Models* tab.

    Args:
        experiment: MLflow experiment, one per question (e.g. ``"crop-stage"``).
        name: Run name shown in the MLflow UI.
        approach: Modelling approach, stored as tag ``approach`` for filtering and grouping.
        description: Free text on what was tried and why; shown as the run description.
        params: Settings of the run (configuration and hyperparameters).
        metrics: Scores on the validation data.
        train: Training data (features and targets) the model was fitted on.
        validation: Validation data (features and targets) the metrics were computed on.
        model: Fitted model to store; omit it for quick runs.
        figures: Plots to store, keyed by file name (e.g. ``"confusion_matrices.png"``).
        system_metrics: Also record CPU/memory usage during the run.

    Returns:
        Run id and – if a model was stored – the URI to load it.
    """
    _use_experiment(experiment)
    if system_metrics:
        mlflow.set_system_metrics_sampling_interval(MLFLOW_SYSTEM_METRICS_INTERVAL_S)
        mlflow.set_system_metrics_samples_before_logging(1)
    tags = {
        "approach": approach,
        "git.commit": _git("rev-parse", "--short", "HEAD"),
        "git.branch": _git("branch", "--show-current"),
        "git.dirty": str(bool(_git("status", "--porcelain", "--", "src", "notebooks"))),
        "author": _git("config", "user.name"),
    }
    with mlflow.start_run(
        run_name=name,
        tags=tags,
        description=description or None,
        log_system_metrics=system_metrics,
    ) as active:
        mlflow.log_params(dict(params))
        datasets = {}
        with warnings.catch_warnings():
            for message in _IRRELEVANT_DATASET_WARNINGS:
                warnings.filterwarnings("ignore", message=f".*{message}")
            for context, frame in (("training", train), ("validation", validation)):
                datasets[context] = from_pandas(frame, source=str(SPLIT_FILE), name=context)
                mlflow.log_input(datasets[context], context=context)
        for file_name, fig in (figures or {}).items():
            mlflow.log_figure(fig, file_name)
        model_id = model_uri = None
        if model is not None:
            # skops cannot serialise our own transformers; the models are our own and stay
            # local, so cloudpickle is safe here.
            logged = mlflow.sklearn.log_model(
                model, name=name, serialization_format="cloudpickle", params=dict(params)
            )
            model_id, model_uri = logged.model_id, logged.model_uri
        mlflow.log_metrics(dict(metrics), model_id=model_id, dataset=datasets["validation"])
    return TrackedRun(active.info.run_id, model_uri)
