"""Run a model on the shared split with the standard preprocessing and evaluation."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
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
from awp2.evaluation import Metrics, as_target_frame, evaluate
from awp2.preprocessing import PreprocessingConfig, build_preprocessor


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

    def score(
        self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None = None
    ) -> float:
        """Balanced accuracy of the combined crop/stage label (``sample_weight`` is ignored)."""
        return evaluate(y, self.predict(X)).bacc_combined


class RunConfig(BaseModel):
    """Everything that defines a run besides the model itself."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Short run name, lowercase, e.g. 'rf_combined'.",
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


@dataclass(frozen=True)
class RunResult:
    """Outcome of :func:`run`: metrics, the fitted pipeline and the validation predictions."""

    config: RunConfig
    metrics: Metrics
    pipeline: Pipeline
    y_val: pd.DataFrame
    y_pred: pd.DataFrame


def run(model: BaseEstimator, config: RunConfig) -> RunResult:
    """Fit standard preprocessing + ``model`` on the shared training split, evaluate on validation.

    The model must predict both targets: natively multi-output (e.g. random forest), wrapped in
    ``MultiOutputClassifier``, or wrapped in :class:`CombinedLabelClassifier`. It is cloned, the
    passed object stays untouched.

    The 30 % validation split is for comparing finished models. Tune hyperparameters with
    cross-validation on the training part (``awp2.data.cv_splits(X_train, y_train)``).
    """
    X, y = prepare_dataset(load_train())
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    pipeline = Pipeline(
        [("preprocess", build_preprocessor(config.preprocessing)), ("model", clone(model))]
    )
    fit_params = (
        {"model__sample_weight": balanced_sample_weight(y_train)} if config.balance_samples else {}
    )
    pipeline.fit(X_train, y_train, **fit_params)
    y_pred = as_target_frame(pipeline.predict(X_val), y_val.index)

    metrics = evaluate(y_val, y_pred, valid_combinations=valid_combinations(y_train))
    if metrics.invalid_combinations:
        warnings.warn(
            f"{config.name}: {metrics.invalid_combinations:.1%} of the predictions are impossible "
            "crop/stage combinations – consider CombinedLabelClassifier.",
            stacklevel=2,
        )
    return RunResult(config, metrics, pipeline, y_val, y_pred)
