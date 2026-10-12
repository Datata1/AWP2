"""Classifiers that predict the crop first and then the growth stage with one model per crop."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from awp2.config import CROP_COL, EXTRA_TREES_N_ESTIMATORS, SEED, STAGE_COL, TARGET_COLS
from awp2.data import combined_label


class HierarchicalClassifier(ClassifierMixin, BaseEstimator):
    """Predict the crop, then the stage with a stage model trained only on that crop.

    Each stage model only knows the stages observed for its crop, so every prediction is a
    crop/stage pair that occurs in training. Hyperparameters of both levels are reachable for
    ``tune()`` as ``crop_estimator__<name>`` and ``stage_estimator__<name>``; all stage models
    share the same settings.

    Args:
        crop_estimator: Unfitted classifier for the crop.
        stage_estimator: Unfitted classifier for the stage, cloned once per crop.
    """

    def __init__(self, crop_estimator: ClassifierMixin, stage_estimator: ClassifierMixin) -> None:
        """Initialise the hierarchy from two unfitted single-output classifiers.

        Args:
            crop_estimator: Classifier for the crop.
            stage_estimator: Classifier for the stage, cloned once per crop.
        """
        self.crop_estimator = crop_estimator
        self.stage_estimator = stage_estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "HierarchicalClassifier":
        """Fit the crop model on all samples and one stage model per crop.

        The stage models are trained on the true crop, so they learn the stage given a correct
        crop; crop errors at prediction time propagate into the stage.

        Args:
            X: Model features.
            y: Targets with the columns ``Crop`` and ``Stage``.
            sample_weight: Optional weight for each sample. It is rescaled per level: the crop
                model sees the same total weight for every crop, each stage model sees weights
                with mean 1, so shared hyperparameters such as ``C`` act alike for all crops.

        Returns:
            This fitted classifier.

        Raises:
            ValueError: If the target columns do not match the project contract.
        """
        if list(y.columns) != list(TARGET_COLS):
            raise ValueError(f"Expected target columns {list(TARGET_COLS)}, got {list(y.columns)}.")
        crops = y[CROP_COL].to_numpy()
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.crop_estimator_ = clone(self.crop_estimator).fit(
            X, crops, **_weight_params(None if weights is None else _equal_per_crop(weights, crops))
        )
        self.stage_estimators_ = {}
        for crop in self.crop_estimator_.classes_:
            is_crop = crops == crop
            crop_weights = None if weights is None else weights[is_crop] / weights[is_crop].mean()
            self.stage_estimators_[crop] = clone(self.stage_estimator).fit(
                X[is_crop], y.loc[is_crop, STAGE_COL], **_weight_params(crop_weights)
            )
        self.classes_ = np.unique(combined_label(y).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the crop and then the stage within the predicted crop.

        Args:
            X: Model features.

        Returns:
            Predicted targets with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, "crop_estimator_")
        crop = pd.Series(self.crop_estimator_.predict(X), index=X.index, name=CROP_COL)
        return pd.concat([crop, self.predict_stage(X, crop)], axis=1)

    def predict_stage(self, X: pd.DataFrame, crop: pd.Series | np.ndarray) -> pd.Series:
        """Predict the stage for a given crop per sample.

        Passing the true crop measures the stage error without crop errors (oracle crop). ``X``
        must be preprocessed like inside the pipeline, i.e. ``result.pipeline[:-1].transform(X)``.

        Args:
            X: Model features.
            crop: Crop per sample, matched to ``X`` by position.

        Returns:
            Predicted stage per sample, named ``Stage``, with the index of ``X``.

        Raises:
            ValueError: If ``crop`` has another length than ``X`` or contains a crop that was not
                seen in training.
        """
        check_is_fitted(self, "stage_estimators_")
        crops = np.asarray(crop)
        if len(crops) != len(X):
            raise ValueError(f"Got {len(crops)} crops for {len(X)} samples.")
        unknown = set(crops) - set(self.stage_estimators_)
        if unknown:
            raise ValueError(f"No stage model for crop(s) {sorted(unknown)}.")
        stage = np.empty(len(X), dtype=object)
        for value in np.unique(crops):
            is_crop = crops == value
            stage[is_crop] = self.stage_estimators_[value].predict(X[is_crop])
        return pd.Series(stage, index=X.index, name=STAGE_COL)


def _weight_params(sample_weight: np.ndarray | None) -> dict[str, np.ndarray]:
    return {} if sample_weight is None else {"sample_weight": sample_weight}


def _equal_per_crop(weights: np.ndarray, crops: np.ndarray) -> np.ndarray:
    # balanced_sample_weight balances crop/stage pairs, which favours crops with many stages
    _, crop_index = np.unique(crops, return_inverse=True)
    crop_totals = np.bincount(crop_index, weights=weights)
    return weights * len(weights) / (len(crop_totals) * crop_totals[crop_index])


def make_hierarchical_svm() -> HierarchicalClassifier:
    """Create the hierarchy with an RBF-SVM on both levels.

    Returns:
        Unfitted classifier with balanced class weights, so keep ``balance_samples=False``.
            Scale its features through
            ``PreprocessingConfig(use_spectral_standard_scale=True)`` before fitting.
    """
    return HierarchicalClassifier(SVC(class_weight="balanced"), SVC(class_weight="balanced"))


def make_hierarchical_extra_trees() -> HierarchicalClassifier:
    """Create the hierarchy with Extra Trees on both levels.

    Returns:
        Unfitted classifier with balanced class weights and the project seed, so keep
            ``balance_samples=False``.
    """
    return HierarchicalClassifier(_extra_trees(), _extra_trees())


def _extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=EXTRA_TREES_N_ESTIMATORS,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
