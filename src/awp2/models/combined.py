"""Classifiers that predict crop and growth stage as one combined label."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    CROP_COL,
    LABEL_SEP,
    RANDOM_FOREST_N_ESTIMATORS,
    SEED,
    STAGE_COL,
    TARGET_COLS,
)
from awp2.data import combined_label


class CombinedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Adapt a single-output classifier to predict the two project target columns.

    Args:
        estimator: Unfitted classifier trained on the combined ``Crop|Stage`` label.
    """

    def __init__(self, estimator: ClassifierMixin) -> None:
        """Initialise the adapter around an unfitted single-output classifier.

        Args:
            estimator: Classifier trained on the combined ``Crop|Stage`` label.
        """
        self.estimator = estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "CombinedLabelClassifier":
        """Fit the wrapped classifier on one crop/stage class per sample.

        Args:
            X: Model features.
            y: Targets with the columns ``Crop`` and ``Stage``.
            sample_weight: Optional weight for each sample.

        Returns:
            This fitted classifier.

        Raises:
            ValueError: If the target columns do not match the project contract.
        """
        if list(y.columns) != list(TARGET_COLS):
            raise ValueError(f"Expected target columns {list(TARGET_COLS)}, got {list(y.columns)}.")
        self.estimator_ = clone(self.estimator)
        fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.estimator_.fit(X, combined_label(y), **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict crop and stage without producing unseen label combinations.

        Args:
            X: Model features.

        Returns:
            Predicted targets with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, "estimator_")
        labels = pd.Series(self.estimator_.predict(X), index=X.index, name="combined")
        targets = labels.str.split(LABEL_SEP, n=1, expand=True)
        return targets.set_axis([CROP_COL, STAGE_COL], axis=1)


def make_combined_random_forest() -> CombinedLabelClassifier:
    """Create the Random Forest classifier for the combined-label approach.

    Returns:
        Unfitted classifier with balanced class weights and the project seed.
    """
    return CombinedLabelClassifier(
        RandomForestClassifier(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    )


def make_combined_svm() -> CombinedLabelClassifier:
    """Create the RBF-SVM classifier for the combined-label approach.

    Returns:
        Unfitted classifier with balanced class weights. Scale its features through
            ``PreprocessingConfig(use_spectral_standard_scale=True)`` before fitting.
    """
    return CombinedLabelClassifier(SVC(class_weight="balanced"))
