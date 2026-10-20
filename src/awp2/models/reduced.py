"""Classifiers that choose their own spectral bands before fitting."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from awp2.config import F_CLASSIF_SCORE, HARD_ROUTING, BandScore, Routing
from awp2.data import band_columns
from awp2.models.hierarchical import HierarchicalClassifier, make_hierarchical_svm
from awp2.preprocessing import SelectBands


def _classifier_has_predict_proba(wrapper: "BandReducedClassifier") -> bool:
    return hasattr(wrapper.classifier, "predict_proba")


class BandReducedClassifier(ClassifierMixin, BaseEstimator):
    """Select the ``k`` most separating bands for this classifier's own target, then fit it.

    Inside ``HierarchicalClassifier`` each level is fitted on its own rows and target, so the
    crop model picks bands for the crop and every stage model picks bands for the stages of its
    crop. Non-band columns such as AEZ and Month pass through unchanged.

    Args:
        classifier: Unfitted single-target classifier, cloned on fit.
        k: Number of bands to keep.
        band_score: Band score, ``"f_classif"`` or ``"mutual_info"``.
    """

    def __init__(
        self, classifier: ClassifierMixin, k: int = 20, band_score: BandScore = F_CLASSIF_SCORE
    ) -> None:
        """Store the unfitted classifier and the selection settings.

        Args:
            classifier: Unfitted single-target classifier, cloned on fit.
            k: Number of bands to keep.
            band_score: Band score, ``"f_classif"`` or ``"mutual_info"``.
        """
        self.classifier = classifier
        self.k = k
        self.band_score = band_score

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Choose the bands on ``X`` and fit the classifier on them plus all non-band columns.

        Args:
            X: Preprocessed features with band columns (``X<nm>``).
            y: One target label per row.
            sample_weight: Optional weight per row, passed to the classifier.

        Returns:
            This fitted classifier.
        """
        bands = band_columns(X)
        self.selector_ = SelectBands(self.k, self.band_score).fit(X[bands], y)
        self.context_columns_ = [column for column in X.columns if column not in bands]
        weights = {} if sample_weight is None else {"sample_weight": sample_weight}
        self.classifier_ = clone(self.classifier).fit(self._reduce(X), y, **weights)
        self.classes_ = self.classifier_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the bands chosen in ``fit``.

        Args:
            X: Preprocessed features with the columns seen in ``fit``.

        Returns:
            One predicted label per row.
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(self._reduce(X))

    @available_if(_classifier_has_predict_proba)
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities with the bands chosen in ``fit`` (needed for soft routing).

        Args:
            X: Preprocessed features with the columns seen in ``fit``.

        Returns:
            One row of probabilities per sample, columns in the order of ``classes_``.
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(self._reduce(X))

    @property
    def selected_wavelengths_(self) -> list[int]:
        """Wavelengths of the bands this classifier uses."""
        return self.selector_.selected_wavelengths_

    def _reduce(self, X: pd.DataFrame) -> pd.DataFrame:
        bands = band_columns(X)
        return pd.concat([self.selector_.transform(X[bands]), X[self.context_columns_]], axis=1)


def make_hierarchical_reduced_svm(
    k: int = 20, routing: Routing = HARD_ROUTING
) -> HierarchicalClassifier:
    """Create the SVM hierarchy where every level selects its own ``k`` bands.

    Tune the band count per level as ``crop_estimator__k`` and ``stage_estimator__k``, the SVM
    as ``crop_estimator__classifier__C`` etc.

    Args:
        k: Initial number of bands per level.
        routing: ``"hard"`` or ``"soft"``, see ``make_hierarchical_svm``.

    Returns:
        Unfitted hierarchy; scale the spectra through ``PreprocessingConfig`` before fitting.
    """
    svm_hierarchy = make_hierarchical_svm(routing)
    return HierarchicalClassifier(
        BandReducedClassifier(svm_hierarchy.crop_estimator, k),
        BandReducedClassifier(svm_hierarchy.stage_estimator, k),
        routing=routing,
    )
