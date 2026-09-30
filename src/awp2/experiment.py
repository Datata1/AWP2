"""Run a model on the shared split with the standard preprocessing and evaluation."""

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from awp2.data import (
    balanced_sample_weight,
    combined_label,
    load_train,
    prepare_dataset,
    split_combined_label,
    train_val_split,
    valid_combinations,
)
from awp2.evaluation import as_target_frame, evaluate, plot_confusion_matrices
from awp2.preprocessing import build_preprocessor
from awp2.tracking import estimator_params, log_run


class CombinedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Train any single-output classifier on the combined ``Crop|Stage`` label.

    Labels are integer-encoded (works with XGBoost & co.) and predictions are split back into
    the two target columns, so only combinations seen in training can be predicted.
    ``score()`` returns the combined balanced accuracy, so ``GridSearchCV`` works out of the box.
    """

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **fit_params) -> "CombinedLabelClassifier":
        """Fit on the combined label; ``fit_params`` (e.g. ``sample_weight``) are forwarded."""
        self.encoder_ = LabelEncoder().fit(combined_label(y))
        self.classes_ = self.encoder_.classes_
        codes = self.encoder_.transform(combined_label(y))
        self.estimator_ = clone(self.estimator).fit(X, codes, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict ``Crop`` and ``Stage`` as a two-column DataFrame."""
        labels = self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X)))
        return split_combined_label(labels, index=getattr(X, "index", None))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, columns ordered like ``classes_``."""
        return self.estimator_.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None) -> float:
        """Balanced accuracy of the combined crop/stage label."""
        return evaluate(y, self.predict(X))["bacc_combined"]


@dataclass
class RunResult:
    """Outcome of :func:`run`: metrics, the fitted pipeline and the validation predictions."""

    name: str
    metrics: dict[str, float]
    pipeline: Pipeline
    y_val: pd.DataFrame
    y_pred: pd.DataFrame
    run_id: str | None = None


def run(
    model: BaseEstimator,
    name: str,
    preprocessor: BaseEstimator | None = None,
    balance_samples: bool = False,
    track: bool = True,
    log_model: bool = False,
) -> RunResult:
    """Fit ``preprocessor + model`` on the shared training split and evaluate on validation.

    The model must predict both targets: natively multi-output (e.g. random forest), wrapped in
    ``MultiOutputClassifier``, or wrapped in :class:`CombinedLabelClassifier`.

    Args:
        model: Unfitted estimator (it is cloned, the passed object stays untouched).
        name: Short run name, e.g. ``"rf_combined"``.
        preprocessor: Defaults to ``build_preprocessor()`` **without scaling** – pass
            ``build_preprocessor(scale=True)`` for SVM, logistic regression or MLP.
        balance_samples: Pass balanced ``sample_weight`` (by crop+stage) to ``fit`` – for models
            without a ``class_weight`` option.
        track: Log parameters, metrics and confusion matrices to MLflow (``make mlflow``).
        log_model: Also store the fitted pipeline in MLflow (can be large).

    The 30 % validation split is for comparing finished models. Tune hyperparameters with
    cross-validation on the training part (``awp2.data.cv_splits(X_train, y_train)``).
    """
    X, y = prepare_dataset(load_train())
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    if preprocessor is None:
        preprocessor = build_preprocessor()
    pipeline = Pipeline([("preprocess", preprocessor), ("model", clone(model))])
    fit_params = (
        {"model__sample_weight": balanced_sample_weight(y_train)} if balance_samples else {}
    )
    pipeline.fit(X_train, y_train, **fit_params)
    y_pred = as_target_frame(pipeline.predict(X_val), y_val.index)

    metrics = evaluate(y_val, y_pred)
    valid = valid_combinations(y_train)
    invalid = sum(combo not in valid for combo in y_pred.itertuples(index=False, name=None))
    metrics["invalid_combinations"] = round(invalid / len(y_pred), 4)
    if invalid:
        warnings.warn(
            f"{name}: {invalid} predictions are impossible crop/stage combinations "
            "– consider CombinedLabelClassifier.",
            stacklevel=2,
        )

    run_id = None
    if track:
        params = {
            **estimator_params(preprocessor, "prep"),
            **estimator_params(model, "model"),
            "balance_samples": balance_samples,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }
        fig = plot_confusion_matrices(y_val, y_pred)
        run_id = log_run(
            name,
            params,
            metrics,
            model=pipeline if log_model else None,
            figures={"confusion_matrices.png": fig},
        )
        plt.close(fig)
    return RunResult(name, metrics, pipeline, y_val, y_pred, run_id)
