"""Experiment tracking with a local MLflow store (``mlflow.db`` + ``mlruns/``, not in git).

Only this module talks to MLflow, so the tracking tool can be swapped without touching the
rest of the code. Browse the runs with ``make mlflow``.
"""

import hashlib
import subprocess
from collections.abc import Mapping

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from sklearn.base import BaseEstimator

from awp2.config import (
    CLEAN_TRAIN_FILE,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_PARAM_MAX_CHARS,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    SPLIT_FILE,
)

_LOCAL_STORE_SCHEMES = ("sqlite:", "file:")
_UNKNOWN = "unknown"
_DATA_HASH_CHARS = 8


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT)
    except FileNotFoundError:
        return _UNKNOWN
    return out.stdout.strip() if out.returncode == 0 else _UNKNOWN


def _data_version() -> str:
    digest = hashlib.sha256()
    for path in (CLEAN_TRAIN_FILE, SPLIT_FILE):
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:_DATA_HASH_CHARS]


def _use_project_experiment() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        # A shared server decides itself where artifacts go; only a local store gets ours.
        is_local = MLFLOW_TRACKING_URI.startswith(_LOCAL_STORE_SCHEMES)
        location = MLFLOW_ARTIFACTS_DIR.as_uri() if is_local else None
        try:
            mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=location)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":  # created by a parallel run
                raise
    elif experiment.lifecycle_stage == "deleted":
        mlflow.MlflowClient().restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


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
    name: str,
    params: Mapping[str, str | bool],
    metrics: Mapping[str, float],
    model: BaseEstimator | None = None,
    figures: Mapping[str, plt.Figure] | None = None,
) -> str:
    """Record one run in the MLflow store.

    Every run is tagged with git commit, branch, author, whether ``src/`` or ``notebooks/`` had
    uncommitted changes, and a short hash of the data artifacts – so a result can always be
    traced back to the code and the split that produced it.

    Args:
        name: Run name shown in the MLflow UI.
        params: Settings of the run (configuration and hyperparameters).
        metrics: Scores of the run.
        model: Fitted model to store; omit it for quick runs because models can be large.
        figures: Plots to store, keyed by file name (e.g. ``"confusion_matrices.png"``).

    Returns:
        The MLflow run id.
    """
    _use_project_experiment()
    tags = {
        "git.commit": _git("rev-parse", "--short", "HEAD"),
        "git.branch": _git("branch", "--show-current"),
        "git.dirty": str(bool(_git("status", "--porcelain", "--", "src", "notebooks"))),
        "author": _git("config", "user.name"),
        "data.version": _data_version(),
    }
    with mlflow.start_run(run_name=name, tags=tags) as active:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        for file_name, fig in (figures or {}).items():
            mlflow.log_figure(fig, file_name)
        if model is not None:
            # skops cannot serialise our own transformers; the models are our own and stay
            # local, so cloudpickle is safe here.
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    return active.info.run_id
