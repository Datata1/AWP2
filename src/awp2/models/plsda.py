"""Partial least squares discriminant analysis (PLS-DA) – a standard method in spectroscopy."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from awp2.models.linear import BalancedLinearDiscriminantAnalysis


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Compress the features with PLS towards the classes, then classify the scores with LDA.

    PLS finds the directions of the (highly correlated) bands that covary most with the one-hot
    class labels – a supervised alternative to PCA. A shrinkage LDA with equal priors classifies
    the scores, so frequent classes are not favoured.

    Args:
        n_components: Number of PLS components; capped at the number of features.
    """

    def __init__(self, n_components: int = 10) -> None:
        """Initialise the classifier.

        Args:
            n_components: Number of PLS components.
        """
        self.n_components = n_components

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "PLSDAClassifier":
        """Fit PLS on one-hot labels and LDA on the resulting scores.

        Args:
            X: Training features, already scaled.
            y: Class label per sample.

        Returns:
            This fitted classifier.
        """
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        one_hot = (labels[:, np.newaxis] == self.classes_).astype(float)
        components = min(self.n_components, np.shape(X)[1])
        self.pls_ = PLSRegression(n_components=components, scale=False).fit(X, one_hot)
        self.lda_ = BalancedLinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            self.pls_.transform(X), labels
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the class of each sample.

        Args:
            X: Features with the columns used in ``fit``.

        Returns:
            Predicted class labels.
        """
        check_is_fitted(self, "lda_")
        return self.lda_.predict(self.pls_.transform(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features with the columns used in ``fit``.

        Returns:
            One probability per class (columns in the order of ``classes_``).
        """
        check_is_fitted(self, "lda_")
        return self.lda_.predict_proba(self.pls_.transform(X))
