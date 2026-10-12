"""Linear classifiers adapted to the balanced-accuracy objective."""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class BalancedLinearDiscriminantAnalysis(LinearDiscriminantAnalysis):
    """LDA that treats all classes as equally likely a priori.

    Plain LDA uses the class frequencies as priors and so favours frequent classes; equal priors
    match the balanced accuracy, the way ``class_weight="balanced"`` does for other models. All
    other options are those of :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`;
    ``priors`` is ignored.
    """

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "BalancedLinearDiscriminantAnalysis":
        """Fit LDA with equal priors for the classes present in ``y``.

        Args:
            X: Training features.
            y: Class label per sample.

        Returns:
            This fitted classifier.
        """
        configured_priors = self.priors
        # the classes are only known here; the parameter is restored so clone/get_params see
        # the configured value
        n_classes = len(np.unique(y))
        self.priors = np.full(n_classes, 1 / n_classes)
        try:
            super().fit(X, y)
        finally:
            self.priors = configured_priors
        return self
