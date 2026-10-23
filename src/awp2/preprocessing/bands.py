"""Bands for the shared preprocessing pipeline."""

import re
from typing import Self

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

from awp2.config import (
    BAND_PATTERN,
    BAND_PREFIX,
    MEAN_ALBEDO_COL,
    SAVGOL_SPLIT_GAP_NM,
)
from awp2.data import wavelengths


class _BandTransformer(TransformerMixin, BaseEstimator):
    """Base for transformers that work on a DataFrame of band columns (``X<nm>``)."""

    feature_names_out_: list[str]

    @staticmethod
    def _check(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame with band columns, got {type(X)}.")
        wrong = [c for c in X.columns if not re.fullmatch(BAND_PATTERN, str(c))]
        if wrong:
            raise ValueError(f"Not band columns (expected {BAND_PREFIX}<nm>): {wrong[:5]}")
        return X

    def _set_input(self, X: pd.DataFrame) -> None:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of the output columns (sklearn API).

        Args:
            input_features: Ignored; accepted for sklearn compatibility.

        Returns:
            Column names after the transformation.
        """
        return np.asarray(self.feature_names_out_, dtype=object)


class DropEmptyBands(_BandTransformer):
    """Drop bands that contain no values at all in the training data."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Learn which bands are completely empty.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        X = self._check(X)
        self._set_input(X)
        self.empty_bands_ = [c for c in X.columns if X[c].isna().all()]
        self.feature_names_out_ = [c for c in X.columns if c not in self.empty_bands_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove the bands that were empty during ``fit``.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            ``X`` without the empty bands.
        """
        return self._check(X)[self.feature_names_out_]


class InterpolateBands(_BandTransformer):
    """Fill missing band values per spectrum along the wavelength.

    - Between nearby measured bands: linear interpolation by wavelength.
    - Across wider gaps: use the nearest measured band to avoid interpolating across missing
      spectral regions.
    - At the edges of a spectrum: value of the nearest measured band (spectra differ a lot in
      brightness, so the neighbour is a better estimate than a global median).
    - Spectrum without any value: training median per band.
    """

    def __init__(self, max_interpolation_gap_nm: int = 15) -> None:
        """Initialize the maximum wavelength span for linear interpolation.

        Args:
            max_interpolation_gap_nm: Maximum distance between measured endpoints in nm.
        """
        self.max_interpolation_gap_nm = max_interpolation_gap_nm

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Store the training medians as last-resort fallback.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the maximum interpolation gap is not positive.
        """
        X = self._check(X)
        if self.max_interpolation_gap_nm <= 0:
            raise ValueError("max_interpolation_gap_nm must be positive")
        self._set_input(X)
        self.medians_ = X.median()
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill all missing band values.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            ``X`` without missing values.
        """
        X = self._check(X)
        if not X.isna().any().any():
            return X
        return _interpolate_along_wavelength(X, self.max_interpolation_gap_nm).fillna(self.medians_)


def _interpolate_along_wavelength(
    spectra: pd.DataFrame, max_interpolation_gap_nm: int
) -> pd.DataFrame:
    wavelength_values = np.asarray(wavelengths(spectra))
    values = spectra.to_numpy(dtype=float, na_value=np.nan, copy=True)

    for row in values:
        measured_indices = np.flatnonzero(~np.isnan(row))
        if measured_indices.size == 0:
            continue

        for missing_index in np.flatnonzero(np.isnan(row)):
            insertion_point = np.searchsorted(measured_indices, missing_index)
            left_position = insertion_point - 1
            right_position = insertion_point
            has_left = left_position >= 0
            has_right = right_position < measured_indices.size

            if has_left and has_right:
                left_index = measured_indices[left_position]
                right_index = measured_indices[right_position]
                left_wavelength = wavelength_values[left_index]
                right_wavelength = wavelength_values[right_index]
                if right_wavelength - left_wavelength <= max_interpolation_gap_nm:
                    fraction = (wavelength_values[missing_index] - left_wavelength) / (
                        right_wavelength - left_wavelength
                    )
                    row[missing_index] = row[left_index] + fraction * (
                        row[right_index] - row[left_index]
                    )
                elif (
                    wavelength_values[missing_index] - left_wavelength
                    <= right_wavelength - wavelength_values[missing_index]
                ):
                    row[missing_index] = row[left_index]
                else:
                    row[missing_index] = row[right_index]
            elif has_left:
                row[missing_index] = row[measured_indices[left_position]]
            else:
                row[missing_index] = row[measured_indices[right_position]]

    return pd.DataFrame(values, index=spectra.index, columns=spectra.columns)


def _contiguous_segments(wavelength_values: np.ndarray) -> list[tuple[int, int]]:
    bounds = [0]
    for position in range(1, len(wavelength_values)):
        if wavelength_values[position] - wavelength_values[position - 1] > SAVGOL_SPLIT_GAP_NM:
            bounds.append(position)
    bounds.append(len(wavelength_values))
    return list(zip(bounds[:-1], bounds[1:], strict=True))


class SavitzkyGolaySmoothing(_BandTransformer):
    """Smooth each spectrum with a Savitzky-Golay filter, one wavelength segment at a time.

    Deleted band blocks leave wide gaps between the surviving bands; smoothing across them
    would smear real measurements with distant neighbours, so each contiguous segment is
    filtered independently with a single vectorised call over all spectra.
    """

    def __init__(self, window_length: int, polyorder: int) -> None:
        """Set the filter size and polynomial degree.

        Args:
            window_length: Window size in bands (odd).
            polyorder: Degree of the fitted polynomial.
        """
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate the parameters and record the smoothing segments.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the window is not odd and at least 3, or the polynomial degree
                is not below the window.
        """
        X = self._check(X)
        if self.window_length < 3 or self.window_length % 2 == 0:
            raise ValueError("window_length must be odd and at least 3")
        if not 0 <= self.polyorder < self.window_length:
            raise ValueError("polyorder must satisfy 0 <= polyorder < window_length")
        self._set_input(X)
        self.segments_ = _contiguous_segments(np.asarray(wavelengths(X)))
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Smooth every segment; segments shorter than the window pass through untouched.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            ``X`` with smoothed band values.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        for start, end in self.segments_:
            if end - start >= self.window_length:
                values[:, start:end] = savgol_filter(
                    values[:, start:end],
                    self.window_length,
                    self.polyorder,
                    axis=1,
                    mode="interp",
                )
        return pd.DataFrame(values, index=X.index, columns=X.columns)


class StandardNormalVariate(_BandTransformer):
    """Per-spectrum scatter correction: zero mean and unit variance across the bands.

    Runs right after interpolation on the measured reflectance spectrum, i.e. before
    index engineering, scaling and PCA.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate the band columns; no statistics are fitted (the correction is per row).

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        X = self._check(X)
        self._set_input(X)
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Centre each spectrum to zero mean and unit variance across its bands.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            ``X`` with each row standardised; constant spectra become all zeros.
        """
        X = self._check(X)
        values = X.to_numpy(dtype=float, copy=True)
        stds = values.std(axis=1)
        stds[stds == 0.0] = 1.0
        corrected = (values - values.mean(axis=1, keepdims=True)) / stds[:, None]
        return pd.DataFrame(corrected, index=X.index, columns=X.columns)


class MeanAlbedo(TransformerMixin, BaseEstimator):
    """Per-spectrum mean reflectance (albedo) as one extra feature.

    SNV removes brightness by construction, so the row-wise mean is kept in its own
    column to stay available to the model. Runs after interpolation in its own branch,
    in parallel to the spectra branch, so missing values never bias the mean.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate the band columns and store their names.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        bands = _BandTransformer._check(X)
        self.feature_names_in_ = np.asarray(bands.columns, dtype=object)
        self.n_features_in_ = bands.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the mean across the available (non-missing) bands per spectrum.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            One column, ``mean_albedo``; fully missing spectra give NaN.
        """
        bands = _BandTransformer._check(X)
        means = np.nanmean(bands.to_numpy(dtype=float), axis=1)
        return pd.DataFrame({MEAN_ALBEDO_COL: means}, index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output feature names following the sklearn transformer API.

        Args:
            input_features: Ignored; accepted for sklearn compatibility.

        Returns:
            The single mean-albedo column name.
        """
        return np.asarray([MEAN_ALBEDO_COL], dtype=object)
