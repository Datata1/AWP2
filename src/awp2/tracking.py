"""Experiment tracking with a local MLflow store (``mlflow.db`` + ``mlruns/``, not in git).

Only this module talks to MLflow, so the tracking tool can be swapped without touching the
pipeline. Browse runs with ``make mlflow``.
"""

import subprocess
from collections.abc import Mapping

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

from awp2.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
)

_SIMPLE = (str, int, float, bool, type(None))


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT)
    return out.stdout.strip()


def _setup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def estimator_params(estimator: BaseEstimator, prefix: str) -> dict[str, object]:
    """Flat, loggable parameters of an estimator (nested objects become their class name)."""
    params: dict[str, object] = {prefix: type(estimator).__name__}
    for key, value in estimator.get_params(deep=True).items():
        if isinstance(value, BaseEstimator):
            value = type(value).__name__
        elif not isinstance(value, _SIMPLE):
            continue
        params[f"{prefix}.{key}"] = str(value)[:250]
    return params


def log_run(
    name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    model: BaseEstimator | None = None,
    figures: Mapping[str, plt.Figure] | None = None,
) -> str:
    """Log one run and return its MLflow run id.

    Tags the run with git commit, branch and author so results stay traceable. The fitted
    model is only stored when passed (models can be large).
    """
    _setup()
    tags = {
        "git.commit": _git("rev-parse", "--short", "HEAD"),
        "git.branch": _git("branch", "--show-current"),
        "git.dirty": str(bool(_git("status", "--porcelain", "--", "src"))),
        "author": _git("config", "user.name"),
    }
    with mlflow.start_run(run_name=name, tags=tags) as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        for file_name, fig in (figures or {}).items():
            mlflow.log_figure(fig, file_name)
        if model is not None:
            # cloudpickle: skops cannot store our own transformers; these are our own local
            # models, so unpickling them is safe.
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    return run.info.run_id
