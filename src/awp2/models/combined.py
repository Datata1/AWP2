"""Classifiers that predict crop and growth stage as one combined label."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    CROP_COL,
    EXTRA_TREES_N_ESTIMATORS,
    LABEL_SEP,
    LSTM_STANDARD_CONTEXT,
    RANDOM_FOREST_N_ESTIMATORS,
    SEED,
    SPECTRAL_JITTER_DEFAULT_MULTIPLIER,
    SPECTRAL_JITTER_DEFAULT_NOISE_STD,
    STAGE_COL,
    SVM_REFERENCE_C,
    SVM_REFERENCE_GAMMA,
    TARGET_COLS,
    LSTMContextMode,
)
from awp2.data import band_columns, combined_label
from awp2.models.lstm import SpectralLSTMClassifier


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


class SpectralJitterSVC(ClassifierMixin, BaseEstimator):
    """Train an RBF-SVM with noisy copies of scaled spectral bands.

    Augmented copies are created only in ``fit``. Context features are kept unchanged, so
    cross-validation validation folds and inference inputs are never augmented.

    Args:
        C: RBF-SVM regularization parameter.
        gamma: RBF kernel coefficient.
        augmentation_multiplier: Number of noisy copies generated per training spectrum.
        noise_std: Gaussian noise standard deviation in standard-scaled spectral units.
        random_state: Seed for reproducible spectral perturbations.
    """

    def __init__(
        self,
        C: float = SVM_REFERENCE_C,
        gamma: float = SVM_REFERENCE_GAMMA,
        augmentation_multiplier: int = SPECTRAL_JITTER_DEFAULT_MULTIPLIER,
        noise_std: float = SPECTRAL_JITTER_DEFAULT_NOISE_STD,
        random_state: int = SEED,
    ) -> None:
        """Store sklearn-cloneable jitter and SVM hyperparameters."""
        self.C = C
        self.gamma = gamma
        self.augmentation_multiplier = augmentation_multiplier
        self.noise_std = noise_std
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "SpectralJitterSVC":
        """Fit the RBF-SVM using original and optionally jittered training spectra.

        Args:
            X: Standard-scaled spectral bands with optional static context features.
            y: One combined crop/stage label per row.
            sample_weight: Optional non-negative weight per original training row.

        Returns:
            This fitted classifier.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If hyperparameters, features or sample weights are invalid.
        """
        self._validate_hyperparameters()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X)}.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self.band_columns_ = band_columns(X)
        if not self.band_columns_:
            raise ValueError("X must contain at least one spectral band")
        augmented_X = self._augment_features(X)
        augmented_y = np.tile(np.asarray(y), self.augmentation_multiplier + 1)
        augmented_weights = self._augment_weights(sample_weight, len(X))
        self.estimator_ = SVC(C=self.C, gamma=self.gamma, class_weight="balanced")
        fit_params = {"sample_weight": augmented_weights} if augmented_weights is not None else {}
        self.estimator_.fit(augmented_X, augmented_y, **fit_params)
        self.classes_ = self.estimator_.classes_
        self.n_training_samples_ = len(augmented_X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict combined labels without augmenting inference spectra.

        Args:
            X: Standard-scaled spectral bands with optional static context features.

        Returns:
            One combined crop/stage label per row.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def _validate_hyperparameters(self) -> None:
        if self.C <= 0 or self.gamma <= 0:
            raise ValueError("C and gamma must be positive")
        if not isinstance(self.augmentation_multiplier, int) or self.augmentation_multiplier < 0:
            raise ValueError("augmentation_multiplier must be a non-negative integer")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

    def _augment_features(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = [X]
        generator = np.random.default_rng(self.random_state)
        spectra = X[self.band_columns_].to_numpy(dtype=float)
        for _ in range(self.augmentation_multiplier):
            augmented = X.copy()
            noise = generator.normal(loc=0.0, scale=self.noise_std, size=spectra.shape)
            augmented.loc[:, self.band_columns_] = spectra + noise
            frames.append(augmented)
        return pd.concat(frames, ignore_index=True)

    def _augment_weights(
        self, sample_weight: np.ndarray | None, n_samples: int
    ) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n_samples,):
            raise ValueError(f"sample_weight must have shape ({n_samples},), got {weights.shape}")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() == 0:
            raise ValueError("sample_weight must be finite, non-negative and not all zero")
        return np.tile(weights, self.augmentation_multiplier + 1)


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


def make_combined_extra_trees() -> CombinedLabelClassifier:
    """Create the Extra Trees classifier for the combined-label approach.

    Returns:
        Unfitted classifier with balanced class weights and the project seed.
    """
    return CombinedLabelClassifier(
        ExtraTreesClassifier(
            n_estimators=EXTRA_TREES_N_ESTIMATORS,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    )


def make_combined_hist_gradient_boosting() -> CombinedLabelClassifier:
    """Create the HistGradientBoosting classifier for the combined-label approach.

    Returns:
        Unfitted classifier with the project seed. Pass balanced sample weights through
            ``TuneConfig(balance_samples=True)`` or ``RunConfig(balance_samples=True)``.
    """
    return CombinedLabelClassifier(HistGradientBoostingClassifier(random_state=SEED))


def make_combined_svm() -> CombinedLabelClassifier:
    """Create the RBF-SVM classifier for the combined-label approach.

    Returns:
        Unfitted classifier with balanced class weights. Scale its features through
            ``PreprocessingConfig(use_spectral_standard_scale=True)`` before fitting.
    """
    return CombinedLabelClassifier(SVC(class_weight="balanced"))


def make_combined_spectral_jitter_svm() -> CombinedLabelClassifier:
    """Create the documented RBF-SVM reference with train-only spectral jitter.

    Returns:
        Unfitted combined-label SVM with the reference ``C`` and ``gamma`` values.
    """
    return CombinedLabelClassifier(SpectralJitterSVC())


def make_combined_spectral_lstm(
    context_mode: LSTMContextMode = LSTM_STANDARD_CONTEXT,
) -> CombinedLabelClassifier:
    """Create the LSTM classifier for the combined-label approach.

    Args:
        context_mode: How the LSTM receives AEZ and Month context.

    Returns:
        Unfitted LSTM wrapped to predict valid combined crop/stage labels.
    """
    return CombinedLabelClassifier(SpectralLSTMClassifier(context_mode=context_mode))
