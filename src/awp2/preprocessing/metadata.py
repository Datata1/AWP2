"""Metadata for the shared preprocessing pipeline."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    AEZ_COL,
    AEZ_MONTH_PREFIX,
    AEZ_MONTH_SEP,
    MONTH_COL,
    MONTH_COS_COL,
    MONTH_SIN_COL,
)
from awp2.data import band_columns


class CyclicMonthEncoder(TransformerMixin, BaseEstimator):
    """Encode calendar month as sine and cosine features."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate the month column and store its feature name.

        Args:
            X: DataFrame containing one month column.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If input does not contain one valid month column.
        """
        months = self._check(X)
        self.feature_names_in_ = np.asarray(months.columns, dtype=object)
        self.n_features_in_ = months.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return sine and cosine features for each calendar month.

        Args:
            X: DataFrame containing the month column used during fitting.

        Returns:
            Two columns, ``Month_sin`` and ``Month_cos``.
        """
        months = self._check(X).iloc[:, 0].to_numpy(dtype=float)
        angle = 2 * np.pi * (months) / 12
        return pd.DataFrame(
            {MONTH_SIN_COL: np.sin(angle), MONTH_COS_COL: np.cos(angle)},
            index=X.index,
        )

    @staticmethod
    def _check(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or list(X.columns) != [MONTH_COL]:
            raise ValueError(f"Expected a DataFrame with the {MONTH_COL!r} column.")
        months = X[MONTH_COL]
        valid = (
            pd.api.types.is_numeric_dtype(months)
            and months.notna().all()
            and months.between(1, 12).all()
            and np.equal(months, np.floor(months)).all()
        )
        if not valid:
            raise ValueError("Month values must be integers in 1-12.")
        return X

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output feature names following the sklearn transformer API.

        Args:
            input_features: Ignored; accepted for sklearn compatibility.

        Returns:
            Names of the sine and cosine features.
        """
        return np.asarray([MONTH_SIN_COL, MONTH_COS_COL], dtype=object)


class AcquisitionCentering(TransformerMixin, BaseEstimator):
    """Subtract the mean training spectrum of each AEZ x month acquisition.

    Rows of one AEZ x month cell probably share an acquisition, i.e. its atmosphere and
    illumination; removing the cell mean takes out that common offset. Cells unseen in training
    fall back to the overall mean. Takes AEZ, Month and the bands, returns the bands only.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Learn the mean spectrum per acquisition and overall.

        Args:
            X: ``AEZ``, ``Month`` and band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        self.bands_ = band_columns(X)
        self.cell_means_ = X[self.bands_].groupby(_acquisition(X)).mean()
        self.overall_mean_ = X[self.bands_].mean()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subtract the mean spectrum of each row's acquisition.

        Args:
            X: ``AEZ``, ``Month`` and band columns.

        Returns:
            The centered band columns; missing values stay missing.
        """
        check_is_fitted(self, "cell_means_")
        means = self.cell_means_.reindex(_acquisition(X).to_numpy())
        means = means.fillna(self.overall_mean_).set_axis(X.index)
        return X[self.bands_] - means

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output feature names following the sklearn transformer API.

        Args:
            input_features: Ignored; accepted for sklearn compatibility.

        Returns:
            The band column names.
        """
        return np.asarray(self.bands_, dtype=object)


def _acquisition(X: pd.DataFrame) -> pd.Series:
    return X[AEZ_COL].astype(int).astype(str) + AEZ_MONTH_SEP + X[MONTH_COL].astype(int).astype(str)


class AezMonthEncoder(TransformerMixin, BaseEstimator):
    """One-hot encode each AEZ × month combination seen in training.

    Crop errors cluster in particular regions at particular times of the season; one indicator
    per combination makes that context explicit (and gives it more weight in distance-based
    models than separate AEZ and month features). Combinations unseen in training get all zeros.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record the observed combinations.

        Args:
            X: DataFrame with the ``AEZ`` and ``Month`` columns.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        combinations = self._combinations(X)
        self.categories_ = np.asarray(sorted(combinations.unique()), dtype=object)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return one indicator column per training combination.

        Args:
            X: DataFrame with the ``AEZ`` and ``Month`` columns.

        Returns:
            One 0/1 column per combination, named ``aez_month_<AEZ>_<Month>``.
        """
        combinations = self._combinations(X).to_numpy()
        indicators = (combinations[:, np.newaxis] == self.categories_).astype(float)
        return pd.DataFrame(indicators, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output feature names following the sklearn transformer API.

        Args:
            input_features: Ignored; accepted for sklearn compatibility.

        Returns:
            One name per training combination.
        """
        check_is_fitted(self, "categories_")
        return np.asarray(
            [f"{AEZ_MONTH_PREFIX}{AEZ_MONTH_SEP}{combination}" for combination in self.categories_],
            dtype=object,
        )

    @staticmethod
    def _combinations(X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame) or list(X.columns) != [AEZ_COL, MONTH_COL]:
            raise ValueError(f"Expected a DataFrame with the columns {[AEZ_COL, MONTH_COL]}.")
        return (
            X[AEZ_COL].astype(int).astype(str)
            + AEZ_MONTH_SEP
            + X[MONTH_COL].astype(int).astype(str)
        )
