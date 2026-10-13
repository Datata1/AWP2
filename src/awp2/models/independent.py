"""Classifiers that predict crop and growth stage with independent estimators."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from awp2.config import CROP_COL, STAGE_COL, TARGET_COLS
from awp2.data import combined_label


class IndependentClassifier(ClassifierMixin, BaseEstimator):
    """Predict crop and growth stage with two independently trained estimators.

    The estimator for the growth stage is trained on all rows and does not receive the predicted
    crop. Predictions can therefore contain crop/stage pairs that did not occur in training.

    Args:
        crop_estimator: Unfitted classifier for the crop target.
        stage_estimator: Unfitted classifier for the growth-stage target.
    """

    def __init__(self, crop_estimator: ClassifierMixin, stage_estimator: ClassifierMixin) -> None:
        """Initialise the independent estimators.

        Args:
            crop_estimator: Classifier trained on the crop target.
            stage_estimator: Classifier trained on the growth-stage target.
        """
        self.crop_estimator = crop_estimator
        self.stage_estimator = stage_estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "IndependentClassifier":
        """Fit one estimator for each project target.

        Args:
            X: Model features.
            y: Targets with the columns ``Crop`` and ``Stage``.
            sample_weight: Optional weight for each sample, passed to both estimators.

        Returns:
            This fitted classifier.

        Raises:
            ValueError: If the target columns do not match the project contract.
        """
        if list(y.columns) != list(TARGET_COLS):
            raise ValueError(f"Expected target columns {list(TARGET_COLS)}, got {list(y.columns)}.")
        fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.crop_estimator_ = clone(self.crop_estimator).fit(X, y[CROP_COL], **fit_params)
        self.stage_estimator_ = clone(self.stage_estimator).fit(X, y[STAGE_COL], **fit_params)
        self.classes_ = np.unique(combined_label(y).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict crop and stage independently.

        Args:
            X: Model features.

        Returns:
            Predicted targets with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, ["crop_estimator_", "stage_estimator_"])
        return pd.DataFrame(
            {
                CROP_COL: self.crop_estimator_.predict(X),
                STAGE_COL: self.stage_estimator_.predict(X),
            },
            index=X.index,
        )


def make_independent_svm() -> IndependentClassifier:
    """Create independent RBF-SVMs for crop and growth stage.

    Returns:
        Unfitted classifier with balanced class weights. Scale features through
            ``PreprocessingConfig(scale=True)`` before fitting.
    """
    return IndependentClassifier(SVC(class_weight="balanced"), SVC(class_weight="balanced"))