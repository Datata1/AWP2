"""Approach "combined label": crop and stage predicted as one class ``Crop|Stage``.

Background, advantages and drawbacks: ``docs/modelle/ansaetze.md`` (section "Kombinierte Klasse").
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

from awp2.config import LABEL_SEP, TARGET_COLS
from awp2.data import combined_label
from awp2.evaluation import evaluate


def split_combined_label(
    labels: pd.Series | np.ndarray, index: pd.Index | None = None
) -> pd.DataFrame:
    """Inverse of :func:`awp2.data.combined_label`: back to the columns ``Crop`` and ``Stage``."""
    parts = pd.Series(labels, index=index).str.split(LABEL_SEP, n=1, expand=True)
    parts.columns = list(TARGET_COLS)
    return parts


class CombinedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Train any single-output classifier on the combined ``Crop|Stage`` label.

    Only pairs seen in training can be predicted. Labels are integer-encoded, so estimators that
    require ``0..n-1`` classes (e.g. XGBoost) work as well.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **fit_params: object) -> Self:
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
