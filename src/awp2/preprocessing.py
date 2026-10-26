"""Preprocessing steps as sklearn transformers – always fitted on the training split only."""

import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.linalg import eigh
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.neighbors import kneighbors_graph
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    AEZ_COL,
    AEZ_MONTH_PREFIX,
    AEZ_MONTH_SEP,
    BAND_PATTERN,
    BAND_PREFIX,
    F_CLASSIF_SCORE,
    HYPERION_SENSOR,
    INDEX_WAVELENGTH_TOLERANCE_NM,
    MCARI_GREEN_WEIGHT,
    MEAN_ALBEDO_COL,
    MONTH_COL,
    MONTH_COS_COL,
    MONTH_SIN_COL,
    OSAVI_SOIL_ADJUSTMENT,
    PCA_FEATURE_NAMES,
    REFLECTANCE_PERCENT_SCALE,
    REIP_LAGRANGIAN_MIN_BANDS,
    REIP_LAGRANGIAN_WINDOW_NM,
    REIP_POLYNOMIAL_DEGREE,
    REIP_POLYNOMIAL_WINDOW_NM,
    SAVGOL_SPLIT_GAP_NM,
    SEED,
    SENTINEL2_BANDS,
    SENTINEL2_SENSOR,
    SPECTRAL_REGIONS,
    TCARI_MULTIPLIER,
    VEGETATION_INDEX_NAMES,
    VEGETATION_INDEX_SPECS,
    BandScore,
    Sensor,
    SensorBand,
)
from awp2.data import band_columns, combined_label, wavelength, wavelengths


def nearest_band(target_wavelength_nm: int, band_names: Sequence[str]) -> str:
    """Return the closest available band within the configured wavelength tolerance.

    Ties resolve to the lower wavelength so that the mapping is independent of input order.

    Args:
        target_wavelength_nm: Wavelength required by an index formula in nanometres.
        band_names: Available Hyperion band names in the ``X<nm>`` convention.

    Returns:
        The closest band name at most ``INDEX_WAVELENGTH_TOLERANCE_NM`` away.

    Raises:
        ValueError: If no band is supplied or the closest band exceeds the tolerance.
    """
    if not band_names:
        raise ValueError("Cannot map an index wavelength without available bands")
    closest_band = min(
        band_names,
        key=lambda band_name: (
            abs(wavelength(band_name) - target_wavelength_nm),
            wavelength(band_name),
        ),
    )
    distance_nm = abs(wavelength(closest_band) - target_wavelength_nm)
    if distance_nm > INDEX_WAVELENGTH_TOLERANCE_NM:
        raise ValueError(
            f"No band within {INDEX_WAVELENGTH_TOLERANCE_NM} nm of "
            f"{target_wavelength_nm} nm; closest is {closest_band} ({distance_nm} nm away)"
        )
    return closest_band


def _band_window(
    band_names: Sequence[str], target_window_nm: tuple[int, int], minimum_bands: int
) -> tuple[str, ...]:
    """Resolve a target window and return its available bands in wavelength order.

    Args:
        band_names: Available Hyperion band names.
        target_window_nm: Required lower and upper wavelength in nanometres.
        minimum_bands: Minimum number of resolved bands needed by the algorithm.

    Returns:
        Band names within the resolved window, sorted by wavelength.

    Raises:
        ValueError: If the resolved window contains too few available bands.
    """
    start_band, end_band = (nearest_band(target_nm, band_names) for target_nm in target_window_nm)
    start_nm, end_nm = sorted((wavelength(start_band), wavelength(end_band)))
    window_bands = tuple(
        sorted(
            (band for band in band_names if start_nm <= wavelength(band) <= end_nm),
            key=wavelength,
        )
    )
    if len(window_bands) < minimum_bands:
        raise ValueError(
            f"REIP window {start_nm}-{end_nm} nm needs at least {minimum_bands} bands, "
            f"got {len(window_bands)}"
        )
    return window_bands


def _lagrangian_reip(reflectance: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:
    """Estimate the red-edge inflection point from local derivative interpolation.

    Args:
        reflectance: One row per spectrum and one column per red-edge band.
        wavelength_nm: Wavelength of each reflectance column in nanometres.

    Returns:
        One inflection-point wavelength per spectrum; invalid local maxima are ``NaN``.
    """
    derivatives = np.gradient(reflectance, wavelength_nm, axis=1)
    positions = np.full(reflectance.shape[0], np.nan)
    for row_number, derivative in enumerate(derivatives):
        peak = int(np.argmax(derivative[1:-1])) + 1
        support_wavelengths = wavelength_nm[peak - 1 : peak + 2]
        support_derivatives = derivative[peak - 1 : peak + 2]
        coefficients = np.polyfit(support_wavelengths, support_derivatives, 2)
        curvature, slope = coefficients[:2]
        if np.isclose(curvature, 0.0):
            continue
        position = -slope / (2 * curvature)
        if curvature < 0 and support_wavelengths[0] <= position <= support_wavelengths[-1]:
            positions[row_number] = position
    return positions


def _polynomial_reip(reflectance: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:
    """Estimate the red-edge inflection point from a fourth-degree reflectance polynomial.

    Args:
        reflectance: One row per spectrum and one column per red-edge band.
        wavelength_nm: Wavelength of each reflectance column in nanometres.

    Returns:
        One inflection-point wavelength per spectrum; missing inner maxima are ``NaN``.
    """
    center_nm = float(np.mean(wavelength_nm))
    centered_wavelengths = wavelength_nm - center_nm
    positions = np.full(reflectance.shape[0], np.nan)
    for row_number, spectrum in enumerate(reflectance):
        coefficients = np.polynomial.polynomial.polyfit(
            centered_wavelengths, spectrum, REIP_POLYNOMIAL_DEGREE
        )
        second_derivative = np.array(
            [2 * coefficients[2], 6 * coefficients[3], 12 * coefficients[4]]
        )
        roots = np.polynomial.polynomial.polyroots(second_derivative)
        candidates = [
            float(root.real)
            for root in roots
            if np.isclose(root.imag, 0.0)
            and centered_wavelengths[0] <= root.real <= centered_wavelengths[-1]
            and np.polynomial.polynomial.polyval(root.real, second_derivative) < 0
        ]
        if candidates:
            first_derivative = np.array(
                [coefficients[1], 2 * coefficients[2], 3 * coefficients[3], 4 * coefficients[4]]
            )
            best_candidate = max(
                candidates,
                key=lambda candidate: np.polynomial.polynomial.polyval(candidate, first_derivative),
            )
            positions[row_number] = best_candidate + center_nm
    return positions


class _BandTransformer(TransformerMixin, BaseEstimator):
    """Base for transformers that work on a DataFrame of band columns (``X<nm>``)."""

    feature_names_out_: list[str]  # set in ``fit`` of the subclasses

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


class KeepEveryNthBand(_BandTransformer):
    """Keep every ``step``-th band, starting with the first, to coarsen the spectral sampling.

    Counts the bands it receives, so after ``DropEmptyBands`` the step runs over the usable
    bands and does not land in the empty water-absorption gaps.
    """

    def __init__(self, step: int) -> None:
        """Set the distance between kept bands.

        Args:
            step: Distance between kept bands, in bands; 1 keeps all.
        """
        self.step = step

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Choose the kept bands from the column order.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If ``step`` is smaller than 1.
        """
        if self.step < 1:
            raise ValueError(f"step must be at least 1, got {self.step}")
        X = self._check(X)
        self._set_input(X)
        self.feature_names_out_ = list(X.columns[:: self.step])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the bands chosen in ``fit``.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            ``X`` reduced to every ``step``-th band.
        """
        return self._check(X)[self.feature_names_out_]


class SimulateSensor(_BandTransformer):
    """Resample the hyperspectral bands to the bands of a multispectral sensor.

    Each sensor band is the plain mean of the bands inside its bandwidth, a box-shaped
    approximation of the real spectral response. Outputs keep the band-column format
    (``X<center nm>``), so all later spectral steps and wavelength plots still apply.
    """

    def __init__(self, bands: tuple[SensorBand, ...] = SENTINEL2_BANDS) -> None:
        """Set the sensor bands to simulate.

        Args:
            bands: Central wavelength and bandwidth of every sensor band.
        """
        self.bands = bands

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Assign the input bands to the sensor bands.

        Args:
            X: Gap-free band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If a sensor band contains no input band.
        """
        X = self._check(X)
        self._set_input(X)
        input_wavelengths = np.asarray(wavelengths(X))
        self.members_: dict[str, list[str]] = {}
        for band in self.bands:
            inside = np.abs(input_wavelengths - band.center_nm) <= band.width_nm / 2
            if not inside.any():
                raise ValueError(f"no input band inside sensor band {band.name}")
            self.members_[f"{BAND_PREFIX}{band.center_nm}"] = list(X.columns[inside])
        self.feature_names_out_ = list(self.members_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Average the input bands per sensor band.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            One column per sensor band, named by its central wavelength.
        """
        X = self._check(X)
        return pd.DataFrame(
            {name: X[members].mean(axis=1) for name, members in self.members_.items()},
            index=X.index,
        )


class SelectBands(_BandTransformer):
    """Keep the ``k`` bands that separate the crop/stage pairs best on their own.

    Scores every band univariately against the combined crop/stage label, so a band that
    helps for either target counts. Belongs inside the model pipeline: fitted on a training
    fold, the choice never sees validation labels.
    """

    def __init__(self, k: int, score: BandScore = F_CLASSIF_SCORE) -> None:
        """Set the number of bands and the scoring function.

        Args:
            k: Number of bands to keep.
            score: ``"f_classif"`` (ANOVA F, linear separability) or ``"mutual_info"``
                (also non-linear dependence, slower).
        """
        self.k = k
        self.score = score

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray) -> Self:
        """Score the bands and remember the ``k`` best.

        Args:
            X: Band columns of the training part.
            y: Targets with ``Crop`` and ``Stage``, or one label per row.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If ``k`` exceeds the number of bands.
        """
        X = self._check(X)
        if not 1 <= self.k <= X.shape[1]:
            raise ValueError(f"k must be between 1 and {X.shape[1]} bands, got {self.k}")
        self._set_input(X)
        labels = combined_label(y) if isinstance(y, pd.DataFrame) else y
        if self.score == F_CLASSIF_SCORE:
            scores, _p_values = f_classif(X, labels)
        else:
            scores = mutual_info_classif(X, labels, random_state=SEED)
        self.scores_ = pd.Series(np.nan_to_num(scores), index=X.columns)
        best = set(self.scores_.nlargest(self.k).index)
        self.feature_names_out_ = [band for band in X.columns if band in best]
        self.selected_wavelengths_ = [wavelength(band) for band in self.feature_names_out_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the bands chosen in ``fit``.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            ``X`` reduced to the selected bands, in wavelength order.
        """
        check_is_fitted(self, "feature_names_out_")
        return self._check(X)[self.feature_names_out_]


class OrthogonalSubspaceProjection(_BandTransformer):
    """Unsupervised OSP by automatic target generation: iterative endmember signatures.

    Each round projects the training spectra onto the orthogonal complement of the
    endmembers found so far and picks the highest-energy pixel as the next endmember.
    New spectra are projected onto the endmembers. Runs on raw spectra, so it must not
    combine with scaling.
    """

    def __init__(self, n_components: int) -> None:
        """Set the number of endmembers.

        Args:
            n_components: Number of endmember signatures to extract.
        """
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Extract endmembers greedily from the training spectra.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the component count is not positive or exceeds the bands.
        """
        X = self._check(X)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.n_components > X.shape[1]:
            raise ValueError("n_components must not exceed the bands")
        self._set_input(X)
        values = X.to_numpy(dtype=float)
        found = np.zeros((X.shape[1], 0))
        for _ in range(self.n_components):
            if found.shape[1]:
                projector = np.eye(X.shape[1]) - found @ np.linalg.pinv(found)
            else:
                projector = np.eye(X.shape[1])
            energies = ((values @ projector) ** 2).sum(axis=1)
            found = np.column_stack([found, values[np.argmax(energies)]])
        self.endmembers_ = found
        self.feature_names_out_ = [f"osp{position}" for position in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project spectra onto the fitted endmembers.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Endmember projections, one ``osp<i>`` column each.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        return pd.DataFrame(
            values @ self.endmembers_, index=X.index, columns=self.feature_names_out_
        )


class VerySparseRandomProjection(_BandTransformer):
    """Achlioptas very sparse random projection of mean-centered spectra.

    Entries are ``±sqrt(c)`` with probability ``1/(2c)`` each and zero otherwise, with
    ``c`` as the square root of the band count. The projection is data-independent; the
    training mean is the only fitted statistic, so centering holds for new spectra too.
    """

    def __init__(self, n_components: int) -> None:
        """Set the output size.

        Args:
            n_components: Number of random components.
        """
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Store the training mean and draw the fixed projection matrix.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the component count is not positive.
        """
        X = self._check(X)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        self._set_input(X)
        self.mean_ = X.to_numpy(dtype=float).mean(axis=0)
        sparsity = float(np.sqrt(X.shape[1]))
        draw = np.random.default_rng(SEED).random((X.shape[1], self.n_components))
        self.matrix_ = np.where(
            draw < 1 / (2 * sparsity), 1.0, np.where(draw < 1 / sparsity, -1.0, 0.0)
        ) * np.sqrt(sparsity)
        self.feature_names_out_ = [f"vsrp{position}" for position in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Center spectra with the training mean and apply the fixed projection.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Random components, one ``vsrp<i>`` column each.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        return pd.DataFrame(
            (values - self.mean_) @ self.matrix_, index=X.index, columns=self.feature_names_out_
        )


class LocalityPreservingProjection(_BandTransformer):
    """Linear locality-preserving reduction of the spectral bands.

    Builds a symmetric binary k-nearest-neighbour graph over the training spectra and keeps
    the projection that best preserves neighbour relations (smallest eigenvectors of the
    generalized problem ``XLX'a = lXDX'a``). New spectra transform linearly, so validation
    data needs no graph of its own.
    """

    def __init__(self, n_components: int, n_neighbors: int) -> None:
        """Set the output size and the neighbourhood size.

        Args:
            n_components: Number of preserved components.
            n_neighbors: Neighbours per spectrum for the adjacency graph.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Build the neighbourhood graph and solve for the preserving projection.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the component or neighbour counts are not positive.
        """
        X = self._check(X)
        if self.n_components < 1 or self.n_neighbors < 1:
            raise ValueError("n_components and n_neighbors must be positive")
        self._set_input(X)
        values = X.to_numpy(dtype=float)
        graph = kneighbors_graph(values, self.n_neighbors, include_self=False)
        adjacency = graph.maximum(graph.T)
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        spread = values * degrees[:, None] - adjacency @ values
        left = spread.T @ values
        right = (values * degrees[:, None]).T @ values
        _, eigenvectors = eigh(left, right, subset_by_index=[0, self.n_components - 1])
        self.projection_ = eigenvectors
        self.feature_names_out_ = [f"lpp{position}" for position in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project spectra onto the fitted preserving components.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Preserved components, one ``lpp<i>`` column each.
        """
        values = self._check(X).to_numpy(dtype=float, copy=True)
        return pd.DataFrame(
            values @ self.projection_, index=X.index, columns=self.feature_names_out_
        )


class DeepBeliefFeatures(_BandTransformer):
    """DBN-style features from greedily stacked BernoulliRBMs.

    Each layer is trained on the hidden probabilities of the previous one. Visible units
    assume ``[0, 1]`` input, hence the min-max scaler requirement.
    """

    def __init__(self, hidden_layers: tuple[int, ...]) -> None:
        """Set the stacked architecture.

        Args:
            hidden_layers: Hidden units per layer; the last entry is the output size.
        """
        self.hidden_layers = hidden_layers

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Train the stacked machines greedily, bottom layer first.

        Args:
            X: Band columns of the training part, scaled to ``[0, 1]``.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the architecture is empty or holds non-positive sizes.
        """
        X = self._check(X)
        if not self.hidden_layers or any(size < 1 for size in self.hidden_layers):
            raise ValueError("hidden_layers must hold at least one positive size")
        self._set_input(X)
        activations = X.to_numpy(dtype=float)
        self.rbms_ = []
        for size in self.hidden_layers:
            machine = BernoulliRBM(n_components=size, random_state=SEED)
            activations = machine.fit_transform(activations)
            self.rbms_.append(machine)
        self.feature_names_out_ = [f"dbn{position}" for position in range(self.hidden_layers[-1])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Propagate spectra through the stacked machines.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            Top-layer hidden probabilities, one ``dbn<i>`` column each.
        """
        activations = self._check(X).to_numpy(dtype=float, copy=True)
        for machine in self.rbms_:
            activations = machine.transform(activations)
        return pd.DataFrame(activations, index=X.index, columns=self.feature_names_out_)


class AddVegetationIndices(_BandTransformer):
    """Append the reproducible kickoff-catalogue indices to spectral bands."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Resolve every catalogue wavelength against the fitted band columns.

        Args:
            X: Band columns after missing-value processing.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If a required target wavelength has no band within the tolerance.
        """
        X = self._check(X)
        self._set_input(X)
        implemented_specs = [
            spec
            for spec in VEGETATION_INDEX_SPECS
            if spec.status == "implemented" and spec.name not in PCA_FEATURE_NAMES
        ]
        self.index_bands_ = {
            spec.name: tuple(
                nearest_band(target_nm, list(X.columns)) for target_nm in spec.target_wavelengths_nm
            )
            for spec in implemented_specs
        }
        self.reip_lagrangian_bands_ = _band_window(
            list(X.columns), REIP_LAGRANGIAN_WINDOW_NM, REIP_LAGRANGIAN_MIN_BANDS
        )
        self.reip_polynomial_bands_ = _band_window(
            list(X.columns),
            REIP_POLYNOMIAL_WINDOW_NM,
            REIP_POLYNOMIAL_DEGREE + 1,
        )
        self.feature_names_out_ = [*X.columns, *VEGETATION_INDEX_NAMES]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the configured catalogue values to the input bands.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            The input bands plus the reproducible vegetation-index catalogue.
        """
        X = self._check(X)
        reflectance = X / REFLECTANCE_PERCENT_SCALE
        index_bands = self.index_bands_
        r750, r705 = (reflectance[band] for band in index_bands["SR_750_705"])
        r800, r550 = (reflectance[band] for band in index_bands["SR_800_550"])
        _, r660 = (reflectance[band] for band in index_bands["ND_750_660"])
        _, r680 = (reflectance[band] for band in index_bands["ND_800_680"])
        r750_msr, r705_msr, r445 = (reflectance[band] for band in index_bands["MSR_750_445"])
        r700, r670, _ = (reflectance[band] for band in index_bands["MCARI"])
        _, r500, _ = (reflectance[band] for band in index_bands["PSRI"])
        r531, r570 = (reflectance[band] for band in index_bands["PRI"])
        _, r635 = (reflectance[band] for band in index_bands["PSSRB"])
        r695, r760 = (reflectance[band] for band in index_bands["SR_695_760"])
        r1660, _ = (reflectance[band] for band in index_bands["DSWI_2"])
        r1070, r1200 = (reflectance[band] for band in index_bands["NDWI_1070_1200"])
        r860, r1240 = (reflectance[band] for band in index_bands["NDWI_860_1240"])
        r1600, r820 = (reflectance[band] for band in index_bands["MSI"])
        mean_start, mean_end = index_bands["MEAN_750_850"]
        mean_bands = [
            band
            for band in X.columns
            if wavelength(mean_start) <= wavelength(band) <= wavelength(mean_end)
        ]
        tcari = TCARI_MULTIPLIER * (
            (r700 - r670) - MCARI_GREEN_WEIGHT * (r700 - r550) * (r700 / r670)
        )
        osavi = (1 + OSAVI_SOIL_ADJUSTMENT) * (r800 - r670) / (r800 + r670 + OSAVI_SOIL_ADJUSTMENT)
        reip_lagrangian = _lagrangian_reip(
            reflectance[list(self.reip_lagrangian_bands_)].to_numpy(),
            np.asarray([wavelength(band) for band in self.reip_lagrangian_bands_], dtype=float),
        )
        reip_polynomial = _polynomial_reip(
            reflectance[list(self.reip_polynomial_bands_)].to_numpy(),
            np.asarray([wavelength(band) for band in self.reip_polynomial_bands_], dtype=float),
        )
        indices = pd.DataFrame(
            {
                "SR_750_705": r750 / r705,
                "SR_800_550": r800 / r550,
                "ND_750_660": (r750 - r660) / (r750 + r660),
                "ND_800_680": (r800 - r680) / (r800 + r680),
                "ND_750_705": (r750 - r705) / (r750 + r705),
                "MSR_750_445": (r750_msr - r445) / (r705_msr - r445),
                "MND_750_705": (r750 - r445) / (r750 + r705 - 2 * r445),
                "SR_750_550": r750 / r550,
                "MEAN_750_850": reflectance[mean_bands].mean(axis=1),
                "MCARI": ((r700 - r670) - MCARI_GREEN_WEIGHT * (r700 - r550)) * (r700 / r670),
                "TCARI": tcari,
                "OSAVI": osavi,
                "TCARI_OSAVI": tcari / osavi,
                "PSRI": (r680 - r500) / r750,
                "SIPI": (r800 - r445) / (r800 - r680),
                "PRI": (r531 - r570) / (r531 + r570),
                "PSSRA": r800 / r680,
                "PSSRB": r800 / r635,
                "SR_695_760": r695 / r760,
                "DSWI_1": r800 / (reflectance[index_bands["DSWI_1"][1]]),
                "DSWI_2": r1660 / r550,
                "DSWI_3": r1660 / r680,
                "DSWI_4": r550 / r680,
                "DSWI_5": (r800 + r550) / (r1660 + r680),
                "NDWI_1070_1200": (r1070 - r1200) / (r1070 + r1200),
                "NDWI_860_1240": (r860 - r1240) / (r860 + r1240),
                "MSI": r1600 / r820,
                "REIP_LAGRANGIAN": reip_lagrangian,
                "REIP_POLYNOMIAL": reip_polynomial,
            },
            index=X.index,
        )
        return pd.concat([X, indices], axis=1)


class SelectVegetationIndices(TransformerMixin, BaseEstimator):
    """Keep only the vegetation index columns computed by AddVegetationIndices."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate the index columns.

        Args:
            X: Bands plus vegetation indices, as produced by ``AddVegetationIndices``.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If any vegetation index column is missing.
        """
        self._check(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the vegetation index columns, dropping the bands.

        Args:
            X: Bands plus vegetation indices, as produced by ``AddVegetationIndices``.

        Returns:
            The vegetation-index catalogue columns.
        """
        return self._check(X)[list(VEGETATION_INDEX_NAMES)]

    @staticmethod
    def _check(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X)}.")
        missing = [c for c in VEGETATION_INDEX_NAMES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing vegetation index columns: {missing}")
        return X

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output feature names following the sklearn transformer API.

        Args:
            input_features: Ignored; accepted for sklearn compatibility.

        Returns:
            The vegetation-index catalogue names.
        """
        return np.asarray(VEGETATION_INDEX_NAMES, dtype=object)


class PrincipalComponentFeatures(_BandTransformer):
    """Reduce standardized spectral bands to the two named PCA features."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Fit PCA to the standardized training spectra.

        Args:
            X: Standardized band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        X = self._check(X)
        self._set_input(X)
        self.pca_ = PCA(n_components=len(PCA_FEATURE_NAMES), random_state=SEED).fit(X)
        self.feature_names_out_ = list(PCA_FEATURE_NAMES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project spectra onto the fitted principal-component axes.

        Args:
            X: Standardized band columns with the same columns as in ``fit``.

        Returns:
            One ``PC_1`` and one ``PC_2`` feature per spectrum.
        """
        values = self.pca_.transform(self._check(X))
        return pd.DataFrame(values, index=X.index, columns=self.feature_names_out_)


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


class PreprocessingConfig(BaseModel):
    """Options of the standard preprocessing.

    New preprocessing steps get a field here (and a transformer in this module) instead of
    ad-hoc code in notebooks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    spectral_regions: tuple[str, ...] | None = Field(
        default=None,
        description="Restrict every spectral branch to these SPECTRAL_REGIONS before "
        "interpolation. None keeps all bands; () gives a metadata-only baseline.",
    )
    use_meta: bool = Field(
        default=True, description="Use AEZ (one-hot) and Month as features next to the spectrum."
    )
    use_spectra: bool = Field(
        default=True,
        description="Use the spectral bands as features; False keeps only the other enabled "
        "groups (meta, indices, albedo, PC_1/PC_2), e.g. for an indices-only model.",
    )
    spectral_sensor: Sensor = Field(
        default=HYPERION_SENSOR,
        description="Spectral sampling of the input: hyperion keeps all bands, sentinel2 "
        "averages them into the 10 Sentinel-2 surface bands (after interpolation).",
    )
    spectral_band_step: int | None = Field(
        default=None,
        ge=2,
        description="Keep only every n-th usable band (after interpolation, smoothing and SNV) "
        "to simulate a coarser spectral resolution; None keeps all bands.",
    )
    use_categorical_context: bool = Field(
        default=False,
        description=(
            "Keep AEZ and Month as unscaled categorical columns for a model that learns "
            "context embeddings; overrides one-hot AEZ and cyclic Month encoding."
        ),
    )
    scale: bool = Field(
        default=False,
        description="Deprecated, will be removed soon – use use_spectral_standard_scale instead. "
        "Kept for old runs: standardises spectra and month.",
    )
    use_spectral_standard_scale: bool = Field(
        default=False,
        description="Standardise spectral bands only (per-band mean/std from the training part); "
        "leaves AEZ and Month untouched.",
    )
    use_spectral_minmax_scale: bool = Field(
        default=False,
        description="Scale spectral bands to [0, 1] per band from the training part; "
        "leaves AEZ and Month untouched.",
    )
    use_spectral_snv: bool = Field(
        default=False,
        description="Apply Standard Normal Variate scatter correction per spectrum (zero mean, "
        "unit variance across bands); applied after interpolation, before indices and scaling.",
    )
    use_spectral_centering: bool = Field(
        default=False,
        description="Subtract the per-band training mean without variance scaling "
        "(paper-faithful PCA input); mutually exclusive with the scalers.",
    )
    use_mean_albedo: bool = Field(
        default=False,
        description="Append the per-spectrum mean reflectance (mean albedo, computed "
        "after interpolation) as an extra feature; preserves the brightness that SNV removes. "
        "Scaled with the same method as the spectra.",
    )
    use_savgol: bool = Field(
        default=False,
        description="Smooth spectra with a Savitzky-Golay filter after interpolation, one "
        "wavelength segment at a time.",
    )
    savgol_window_length: int | None = Field(
        default=None,
        description="Savitzky-Golay window size in bands (odd); required with use_savgol.",
    )
    savgol_polyorder: int | None = Field(
        default=None,
        description="Savitzky-Golay polynomial degree; required with use_savgol.",
    )
    spectral_pca_components: int | None = Field(
        default=None,
        gt=0,
        description="Reduce the spectral bands to this many principal components; "
        "None keeps all bands. Requires use_spectral_standard_scale or use_spectral_centering.",
    )
    spectral_select_k: int | None = Field(
        default=None,
        gt=0,
        description="Keep the k bands with the highest univariate score against the crop/stage "
        "label, fitted per training fold; None keeps all bands.",
    )
    spectral_select_score: BandScore = Field(
        default=F_CLASSIF_SCORE,
        description="Band score for spectral_select_k: f_classif (ANOVA F) or mutual_info.",
    )
    spectral_ica_components: int | None = Field(
        default=None,
        gt=0,
        description="Independent components of the spectral bands; "
        "None disables the method. Requires use_spectral_standard_scale or use_spectral_centering.",
    )
    spectral_nmf_components: int | None = Field(
        default=None,
        gt=0,
        description="Non-negative parts of the raw spectra (multiplicative updates); "
        "None disables the method. Needs non-negative input: must not combine with "
        "savgol, snv, scalers or indices.",
    )
    spectral_osp_components: int | None = Field(
        default=None,
        gt=0,
        description="Automatic target generation: iterative endmember signatures of the raw "
        "spectra; None disables the method. Must not combine with snv, scalers or indices.",
    )
    spectral_lpp_components: int | None = Field(
        default=None,
        gt=0,
        description="Locality-preserving components of the spectral bands; "
        "None disables the method. Requires spectral_lpp_neighbors.",
    )
    spectral_lpp_neighbors: int | None = Field(
        default=None,
        gt=0,
        description="Neighbours per spectrum for the LPP adjacency graph; "
        "required with spectral_lpp_components.",
    )
    spectral_vsrp_components: int | None = Field(
        default=None,
        gt=0,
        description="Very sparse random projection of mean-centered spectra; "
        "None disables the method. Needs no scaler.",
    )
    spectral_dbn_layers: tuple[int, ...] | None = Field(
        default=None,
        description="Stacked BernoulliRBM architecture, e.g. (64, 20); the last entry is the "
        "output size. Requires use_spectral_minmax_scale for [0, 1] input.",
    )
    use_vegetation_indices: bool = Field(
        default=False,
        description="Compute the reproducible kickoff vegetation-index catalogue from interpolated "
        "bands in a parallel branch; ratios need physical reflectance, so SNV must not feed them.",
    )
    use_principal_components: bool = Field(
        default=False,
        description="Append PC_1 and PC_2 from a PCA fitted on standardized interpolated bands "
        "in a parallel branch.",
    )
    use_aez_month: bool = Field(
        default=False,
        description="One-hot encode each AEZ × month combination seen in training next to the "
        "separate AEZ and month features; requires use_meta.",
    )
    use_acquisition_centering: bool = Field(
        default=False,
        description="Subtract the mean training spectrum of each AEZ x month acquisition from "
        "the spectra before all other spectral steps; unseen acquisitions use the overall mean.",
    )
    use_cyclic_month: bool = Field(
        default=True,
        description="Replace numeric Month with sine/cosine annual-cycle features.",
    )
    max_interpolation_gap_nm: int = Field(
        default=15,
        gt=0,
        description=(
            "Maximum wavelength span for linear interpolation; wider gaps use the nearest "
            "measured band."
        ),
    )

    @model_validator(mode="after")
    def _check_options(self) -> Self:
        """Reject ambiguous or incomplete preprocessing combinations.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If ``use_aez_month`` is set without ``use_meta`` or with
                ``use_categorical_context``; if ``spectral_regions`` repeats or does not know a
                region, is empty without ``use_meta`` or with a spectral feature branch; if more
                than one reduction method is selected; if centering,
                scalers, PCA, ICA or the deprecated ``scale`` flag are combined ambiguously;
                if NMF, OSP or DBN get unsuitable input; if Savitzky-Golay or LPP
                arguments are set without their method or missing with it; or if
                ``use_spectra=False`` keeps spectral steps or leaves no features.
        """
        if not self.use_spectra:
            self._check_without_spectra()
        if self.use_acquisition_centering and (
            self.spectral_nmf_components is not None or self.spectral_osp_components is not None
        ):
            raise ValueError("acquisition centering must not combine with NMF or OSP (raw spectra)")
        if self.spectral_sensor == SENTINEL2_SENSOR and (
            self.use_savgol or self.spectral_band_step is not None
        ):
            raise ValueError(
                "spectral_sensor='sentinel2' must not combine with savgol or spectral_band_step"
            )
        if self.spectral_regions is not None:
            unknown = set(self.spectral_regions) - {region.name for region in SPECTRAL_REGIONS}
            if unknown:
                raise ValueError(f"spectral_regions requires known names; unknown: {unknown}")
            if len(set(self.spectral_regions)) != len(self.spectral_regions):
                raise ValueError("spectral_regions requires unique names")
            if not self.spectral_regions:
                if not self.use_meta:
                    raise ValueError("Empty spectral_regions requires use_meta=True")
                if (
                    self.use_mean_albedo
                    or self.use_vegetation_indices
                    or self.use_principal_components
                ):
                    raise ValueError("Spectral feature branches require nonempty spectral_regions")
        if self.use_aez_month and (not self.use_meta or self.use_categorical_context):
            raise ValueError("use_aez_month requires use_meta without use_categorical_context")
        if self.use_spectral_standard_scale and self.use_spectral_minmax_scale:
            raise ValueError("use only one of use_spectral_standard_scale/minmax_scale")
        if self.use_spectral_centering and (
            self.scale or self.use_spectral_standard_scale or self.use_spectral_minmax_scale
        ):
            raise ValueError("use_spectral_centering must not combine with scalers")
        if self.scale and (self.use_spectral_standard_scale or self.use_spectral_minmax_scale):
            raise ValueError("deprecated 'scale' must not be combined with the spectral scalers")
        reductions = [
            self.spectral_pca_components,
            self.spectral_select_k,
            self.spectral_ica_components,
            self.spectral_nmf_components,
            self.spectral_osp_components,
            self.spectral_lpp_components,
            self.spectral_vsrp_components,
        ]
        if (
            sum(selected is not None for selected in reductions)
            + (self.spectral_dbn_layers is not None)
            > 1
        ):
            raise ValueError("use at most one dimensionality reduction method")
        needs_centering = (
            self.spectral_pca_components,
            self.spectral_ica_components,
        )
        if any(selected is not None for selected in needs_centering):
            if not (self.use_spectral_standard_scale or self.use_spectral_centering):
                raise ValueError(
                    "pca/ica reduction requires use_spectral_standard_scale or "
                    "use_spectral_centering"
                )
        if self.spectral_nmf_components is not None and (
            self.scale
            or self.use_spectral_standard_scale
            or self.use_spectral_minmax_scale
            or self.use_spectral_centering
            or self.use_spectral_snv
            or self.use_savgol
            or self.use_vegetation_indices
        ):
            raise ValueError(
                "nmf reduction needs raw spectra: must not combine with savgol, snv, "
                "scalers or vegetation indices"
            )
        if self.spectral_osp_components is not None and (
            self.scale
            or self.use_spectral_standard_scale
            or self.use_spectral_minmax_scale
            or self.use_spectral_centering
            or self.use_spectral_snv
            or self.use_vegetation_indices
        ):
            raise ValueError(
                "osp reduction needs raw spectra: must not combine with snv, "
                "scalers or vegetation indices"
            )
        if self.spectral_lpp_components is None and self.spectral_lpp_neighbors is not None:
            raise ValueError("spectral_lpp_neighbors requires spectral_lpp_components")
        if self.spectral_lpp_components is not None and self.spectral_lpp_neighbors is None:
            raise ValueError("spectral_lpp_components requires spectral_lpp_neighbors")
        if self.spectral_lpp_components is not None and self.use_vegetation_indices:
            raise ValueError("lpp reduction must not combine with vegetation indices")
        if self.spectral_dbn_layers is not None:
            if not self.use_spectral_minmax_scale:
                raise ValueError("dbn reduction requires use_spectral_minmax_scale=True")
            if self.use_vegetation_indices:
                raise ValueError("dbn reduction must not combine with vegetation indices")
            if len(self.spectral_dbn_layers) == 0 or any(
                size < 1 for size in self.spectral_dbn_layers
            ):
                raise ValueError("spectral_dbn_layers must hold at least one positive size")
        if self.use_savgol:
            if self.savgol_window_length is None or self.savgol_polyorder is None:
                raise ValueError("use_savgol requires savgol_window_length and savgol_polyorder")
            if self.savgol_window_length < 3 or self.savgol_window_length % 2 == 0:
                raise ValueError("savgol_window_length must be odd and at least 3")
            if not 0 <= self.savgol_polyorder < self.savgol_window_length:
                raise ValueError("savgol_polyorder must satisfy 0 <= polyorder < window_length")
        elif self.savgol_window_length is not None or self.savgol_polyorder is not None:
            raise ValueError("savgol_window_length/polyorder require use_savgol=True")
        return self

    def _check_without_spectra(self) -> None:
        spectral_steps = {
            "use_savgol": self.use_savgol,
            "use_spectral_snv": self.use_spectral_snv,
            "spectral_sensor": self.spectral_sensor != HYPERION_SENSOR,
            "spectral_band_step": self.spectral_band_step is not None,
            "spectral_pca_components": self.spectral_pca_components is not None,
            "spectral_select_k": self.spectral_select_k is not None,
            "spectral_ica_components": self.spectral_ica_components is not None,
            "spectral_nmf_components": self.spectral_nmf_components is not None,
            "spectral_osp_components": self.spectral_osp_components is not None,
            "spectral_lpp_components": self.spectral_lpp_components is not None,
            "spectral_vsrp_components": self.spectral_vsrp_components is not None,
            "spectral_dbn_layers": self.spectral_dbn_layers is not None,
        }
        enabled = [name for name, active in spectral_steps.items() if active]
        if enabled:
            raise ValueError(f"use_spectra=False must not combine with spectral steps: {enabled}")
        if not (
            self.use_meta
            or self.use_vegetation_indices
            or self.use_mean_albedo
            or self.use_principal_components
        ):
            raise ValueError("use_spectra=False needs another feature group")


def _region_columns(frame: pd.DataFrame, regions: tuple[str, ...]) -> list[str]:
    intervals = [region for region in SPECTRAL_REGIONS if region.name in regions]
    return [
        column
        for column in frame.columns
        if re.fullmatch(BAND_PATTERN, column)
        and any(region.start_nm <= wavelength(column) < region.end_nm for region in intervals)
    ]


def build_preprocessor(config: PreprocessingConfig | None = None) -> ColumnTransformer:
    """Build the standard preprocessing as one sklearn transformer.

    Put it as first step into the model pipeline, so it is fitted on the training part only.

    Args:
        config: Options; the defaults of ``PreprocessingConfig`` if omitted.

    Returns:
        Transformer that turns ``X`` (metadata + bands) into model-ready features as a
            DataFrame with named columns.

    Example:
        ```python
        model = Pipeline([
            ("preprocess", build_preprocessor(PreprocessingConfig(
                use_spectral_standard_scale=True
            ))),
            ("model", SVC()),
        ])
        ```
    """
    config = config or PreprocessingConfig()
    band_selector = (
        make_column_selector(pattern=BAND_PATTERN)
        if config.spectral_regions is None
        else partial(_region_columns, regions=config.spectral_regions)
    )
    if config.scale:
        warnings.warn(
            "PreprocessingConfig(scale=...) is deprecated and will be removed soon; "
            "use use_spectral_standard_scale instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    spectra: list[tuple[str, BaseEstimator]] = []
    if config.use_acquisition_centering:
        spectra.append(("acquisition_centering", AcquisitionCentering()))
    spectra.append(("drop_empty", DropEmptyBands()))
    spectra.append(("interpolate", InterpolateBands(config.max_interpolation_gap_nm)))
    if config.spectral_sensor == SENTINEL2_SENSOR:
        spectra.append(("sensor", SimulateSensor(SENTINEL2_BANDS)))
    if config.use_savgol:
        assert config.savgol_window_length is not None and config.savgol_polyorder is not None
        spectra.append(
            ("savgol", SavitzkyGolaySmoothing(config.savgol_window_length, config.savgol_polyorder))
        )
    if config.use_spectral_snv:
        spectra.append(("snv", StandardNormalVariate()))
    if config.spectral_band_step is not None:
        spectra.append(("band_step", KeepEveryNthBand(config.spectral_band_step)))
    spectra.extend(_spectral_scaler(config))
    pca_components = config.spectral_pca_components
    ica_components = config.spectral_ica_components
    nmf_components = config.spectral_nmf_components
    osp_components = config.spectral_osp_components
    lpp_components = config.spectral_lpp_components
    vsrp_components = config.spectral_vsrp_components
    dbn_layers = config.spectral_dbn_layers
    if pca_components is not None:
        spectra.append(("pca", PCA(n_components=pca_components, random_state=SEED)))
    elif config.spectral_select_k is not None:
        spectra.append(
            ("select_k", SelectBands(config.spectral_select_k, config.spectral_select_score))
        )
    elif ica_components is not None:
        spectra.append(("ica", FastICA(n_components=ica_components, random_state=SEED)))
    elif nmf_components is not None:
        spectra.append(("nmf", NMF(n_components=nmf_components, random_state=SEED, solver="mu")))
    elif osp_components is not None:
        spectra.append(("osp", OrthogonalSubspaceProjection(osp_components)))
    elif lpp_components is not None:
        assert config.spectral_lpp_neighbors is not None
        spectra.append(
            ("lpp", LocalityPreservingProjection(lpp_components, config.spectral_lpp_neighbors))
        )
    elif vsrp_components is not None:
        spectra.append(("vsrp", VerySparseRandomProjection(vsrp_components)))
    elif dbn_layers is not None:
        spectra.append(("dbn", DeepBeliefFeatures(dbn_layers)))

    transformers: list[tuple[str, BaseEstimator | str, object]] = []
    if config.use_meta:
        if config.use_categorical_context:
            transformers.append(("context", "passthrough", [AEZ_COL, MONTH_COL]))
        else:
            transformers.append(
                ("aez", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [AEZ_COL])
            )
            if config.use_cyclic_month:
                month_transformers: list[tuple[str, BaseEstimator]] = [
                    ("cyclic", CyclicMonthEncoder())
                ]
                if config.scale:
                    month_transformers.append(("scale", StandardScaler()))
                transformers.append(("month", Pipeline(month_transformers), [MONTH_COL]))
            else:
                transformers.append(
                    ("month", StandardScaler() if config.scale else "passthrough", [MONTH_COL])
                )
    if config.use_mean_albedo:
        transformers.append(
            (
                "albedo",
                Pipeline(
                    [
                        ("drop_empty", DropEmptyBands()),
                        ("interpolate", InterpolateBands(config.max_interpolation_gap_nm)),
                        ("albedo", MeanAlbedo()),
                        # unscaled, the albedo (std ~5 %) dominates distances to standardised bands
                        *_spectral_scaler(config),
                    ]
                ),
                band_selector,
            )
        )
    if config.use_vegetation_indices:
        index_steps: list[tuple[str, BaseEstimator]] = [
            ("drop_empty", DropEmptyBands()),
            ("interpolate", InterpolateBands(config.max_interpolation_gap_nm)),
            ("indices", AddVegetationIndices()),
            ("select", SelectVegetationIndices()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
        if config.use_spectral_standard_scale or config.scale:
            index_steps.append(("standard_scale", StandardScaler()))
        elif config.use_spectral_minmax_scale:
            index_steps.append(("minmax_scale", MinMaxScaler()))
        elif config.use_spectral_centering:
            index_steps.append(("centering", StandardScaler(with_mean=True, with_std=False)))
        transformers.append(
            (
                "indices",
                Pipeline(index_steps),
                band_selector,
            )
        )
    if config.use_principal_components:
        transformers.append(
            (
                "principal_components",
                Pipeline(
                    [
                        ("drop_empty", DropEmptyBands()),
                        ("interpolate", InterpolateBands(config.max_interpolation_gap_nm)),
                        ("standard_scale", StandardScaler()),
                        ("pca", PrincipalComponentFeatures()),
                    ]
                ),
                band_selector,
            )
        )
    if config.use_aez_month:
        transformers.append(("aez_month", AezMonthEncoder(), [AEZ_COL, MONTH_COL]))
    if config.use_spectra:
        spectra_columns = (
            partial(_with_acquisition_context, band_selector=band_selector)
            if config.use_acquisition_centering
            else band_selector
        )
        transformers.append(("spectra", Pipeline(spectra), spectra_columns))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")


class InputSize(NamedTuple):
    """How much input a fitted preprocessor uses.

    Attributes:
        n_input_features: Columns the model receives after preprocessing.
        n_measured_bands: Spectral bands a sensor has to measure: Hyperion bands, or the bands
            of a simulated sensor; PCA still needs all bands, band selection fewer.
    """

    n_input_features: int
    n_measured_bands: int


def input_size(preprocessor: ColumnTransformer) -> InputSize:
    """Count the model features and the measured bands of a fitted preprocessor.

    Assumes that every kept band is measured, so interpolation never has to fill it from
    neighbours.

    Args:
        preprocessor: Built by ``build_preprocessor`` and fitted.

    Returns:
        Number of model features and of distinct measured bands over all branches.
    """
    check_is_fitted(preprocessor)
    bands: set[str] = set()
    for _name, branch, _columns in preprocessor.transformers_:
        if isinstance(branch, Pipeline) and "drop_empty" in branch.named_steps:
            bands |= _branch_bands(branch)
    return InputSize(len(preprocessor.get_feature_names_out()), len(bands))


def _branch_bands(branch: Pipeline) -> set[str]:
    sources = {band: {band} for band in branch.named_steps["drop_empty"].feature_names_out_}
    usable = set(sources)
    mixes_bands = False
    for step in branch.named_steps.values():
        if isinstance(step, SavitzkyGolaySmoothing | StandardNormalVariate):
            # every output band then depends on its neighbours or on the whole spectrum
            mixes_bands = True
        elif isinstance(step, SimulateSensor):
            # the simulated sensor measures each of its broad bands once
            sources = {name: {name} for name in step.members_}
        elif isinstance(step, KeepEveryNthBand | SelectBands):
            sources = {band: sources[band] for band in step.feature_names_out_}
        elif isinstance(step, AddVegetationIndices):
            index_bands = {band for bands in step.index_bands_.values() for band in bands}
            reip_bands = {*step.reip_lagrangian_bands_, *step.reip_polynomial_bands_}
            return index_bands | reip_bands
    return usable if mixes_bands else set().union(*sources.values())


def _with_acquisition_context(
    frame: pd.DataFrame, band_selector: Callable[[pd.DataFrame], list[str]]
) -> list[str]:
    # acquisition centering groups the spectra by AEZ and month, so it needs both columns
    return [AEZ_COL, MONTH_COL, *band_selector(frame)]


def _spectral_scaler(config: PreprocessingConfig) -> list[tuple[str, BaseEstimator]]:
    if config.use_spectral_standard_scale or config.scale:
        return [("standard_scale", StandardScaler())]
    if config.use_spectral_minmax_scale:
        return [("minmax_scale", MinMaxScaler())]
    if config.use_spectral_centering:
        return [("centering", StandardScaler(with_mean=True, with_std=False))]
    return []


def _describe_step(name: str, step: BaseEstimator | str) -> str:
    if name == "context":
        return "AEZ and Month as categorical context"
    if isinstance(step, str):
        return f"Month {step}"
    known: dict[str, str] = {
        "drop_empty": "Drop empty bands",
        "acquisition_centering": "Subtract the mean spectrum of the AEZ x month acquisition",
        "snv": "SNV scatter correction per spectrum",
        "indices": "Vegetation-index catalogue (from interpolated bands)",
        "standard_scale": "Standardize bands (per-band mean and std)",
        "minmax_scale": "Min-max scale bands to [0, 1]",
        "centering": "Center bands (per-band mean)",
        "aez": "One-hot encode AEZ",
        "albedo": "Mean albedo per spectrum (brightness after interpolation, before SNV)",
        "cyclic": "Cyclic month encoding (sin/cos)",
        "month": "Standardize month",
        "scale": "Standardize month",
        "aez_month": "One-hot encode AEZ x month combinations",
    }
    if name in known:
        return known[name]
    if name == "interpolate" and isinstance(step, InterpolateBands):
        return f"Interpolate band gaps (max {step.max_interpolation_gap_nm} nm)"
    if name == "sensor" and isinstance(step, SimulateSensor):
        return f"Simulate {len(step.bands)} multispectral sensor bands"
    if name == "band_step" and isinstance(step, KeepEveryNthBand):
        return f"Keep every {step.step}. band"
    if name == "select" and isinstance(step, SelectVegetationIndices):
        return "Keep only the vegetation indices"
    if name == "savgol" and isinstance(step, SavitzkyGolaySmoothing):
        return (
            f"Savitzky-Golay smoothing (window {step.window_length} bands, "
            f"polyorder {step.polyorder})"
        )
    if name == "pca" and isinstance(step, PCA):
        return f"PCA to {step.n_components} components"
    if name == "select_k" and isinstance(step, SelectBands):
        return f"Select {step.k} bands ({step.score})"
    if name == "pca" and isinstance(step, PrincipalComponentFeatures):
        return "PCA to PC_1 and PC_2"
    if name == "ica" and isinstance(step, FastICA):
        return f"ICA to {step.n_components} components"
    if name == "nmf" and isinstance(step, NMF):
        return f"NMF to {step.n_components} components"
    if name == "osp" and isinstance(step, OrthogonalSubspaceProjection):
        return f"OSP to {step.n_components} endmembers"
    if name == "lpp" and isinstance(step, LocalityPreservingProjection):
        return f"LPP to {step.n_components} components ({step.n_neighbors} neighbors)"
    if name == "vsrp" and isinstance(step, VerySparseRandomProjection):
        return f"VSRP to {step.n_components} components"
    if name == "dbn" and isinstance(step, DeepBeliefFeatures):
        layers = "x".join(str(size) for size in step.hidden_layers)
        return f"DBN features ({layers})"
    return type(step).__name__


_BRANCH_LETTERS: dict[str, str] = {
    "aez": "A",
    "context": "A",
    "month": "B",
    "albedo": "C",
    "indices": "D",
    "spectra": "E",
    "aez_month": "F",
    "principal_components": "G",
}


def describe_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    r"""List the active preprocessing steps, grouped by independent branch.

    Reads the built transformer, so optional steps appear only when enabled. Each
    branch has a fixed letter (A: AEZ, B: Month, C: Albedo, D: Indices, E: Spectra,
    F: AEZ x month, G: PC_1/PC_2)
    and steps are numbered per branch from 1, so the labels stay comparable across
    runs. Print the lines in the notebook to log which preprocessing a run used.

    Args:
        preprocessor: Built by ``build_preprocessor`` (fitting is not required).

    Returns:
        One labeled description per active step, e.g. ``"E3. PCA to 20 components"``.

    Raises:
        ValueError: If a branch has no letter yet (add one to ``_BRANCH_LETTERS``).

    Example:
        ```python
        print("\\n".join(describe_preprocessor(build_preprocessor(
            PreprocessingConfig(use_meta=True, use_spectral_snv=True)
        ))))
        ```
    """
    lines: list[str] = []
    for branch_name, transformer, _columns in preprocessor.transformers:
        if branch_name not in _BRANCH_LETTERS:
            raise ValueError(f"describe_preprocessor needs a letter for branch: {branch_name}")
        letter = _BRANCH_LETTERS[branch_name]
        if isinstance(transformer, Pipeline):
            steps: list[tuple[str, BaseEstimator | str]] = list(transformer.steps)
        else:
            steps = [(branch_name, transformer)]
        for number, (step_name, step) in enumerate(steps, start=1):
            lines.append(f"{letter}{number}. {_describe_step(step_name, step)}")
    return lines


def pipeline_diagram(config: PreprocessingConfig | None = None) -> str:
    """Mermaid flowchart of the built preprocessor, one node per step.

    Mechanical dump of ``build_preprocessor``: parallel branches as subgraphs, steps
    chained in order, labels shared with ``describe_preprocessor``. Inspects one
    rendered config; for the full option map see ``pipeline_options_diagram``.

    Args:
        config: Options; the defaults of ``PreprocessingConfig`` if omitted.

    Returns:
        Complete mermaid code block for the docs.
    """
    preprocessor = build_preprocessor(config or PreprocessingConfig())
    diagram: list[str] = ["```mermaid", "flowchart TD"]
    diagram.append('    X["Eingang: Metadaten + Bänder"]')
    edges: list[str] = []
    for branch_name, transformer, _columns in preprocessor.transformers:
        steps: list[tuple[str, BaseEstimator | str]]
        if isinstance(transformer, Pipeline):
            steps = list(transformer.steps)
        else:
            steps = [(branch_name, transformer)]
        diagram.append(f"    subgraph {branch_name}")
        node_ids: list[str] = []
        for position, (step_name, step) in enumerate(steps):
            node_id = f"n_{branch_name}_{position}"
            label = _describe_step(step_name, step).replace('"', "'")
            diagram.append(f'        {node_id}["{label}"]')
            node_ids.append(node_id)
        for source, target in zip(node_ids, node_ids[1:], strict=False):
            diagram.append(f"        {source} --> {target}")
        diagram.append("    end")
        edges.append(f"    X --> {node_ids[0]}")
        edges.append(f"    {node_ids[-1]} --> FEATURES")
    diagram.extend(edges)
    diagram.append('    FEATURES["Modellmerkmale (verkettet)"]')
    diagram.append("```")
    return "\n".join(diagram)


@dataclass(frozen=True)
class _OptionNode:
    """One box in the options diagram, addressed in built-preprocessor coordinates."""

    branch: str
    step: str
    label: str
    optional: bool
    diamond: bool = False


def _config_fields(*names: str) -> str:
    unknown = sorted(name for name in names if name not in PreprocessingConfig.model_fields)
    if unknown:
        raise ValueError(f"unknown PreprocessingConfig field: {unknown}")
    return " + ".join(names)


_META_NODES: tuple[_OptionNode, ...] = (
    _OptionNode("aez", "aez", f"AEZ one-hot [{_config_fields('use_meta')}]", True),
    _OptionNode("month", "cyclic", f"Monat sin/cos [{_config_fields('use_cyclic_month')}]", False),
    _OptionNode("month", "month", "Monat numerisch [use_cyclic_month = False]", False),
    _OptionNode(
        "aez_month", "aez_month", f"AEZ × Monat one-hot [{_config_fields('use_aez_month')}]", True
    ),
)

_ALBEDO_NODES: tuple[_OptionNode, ...] = (
    _OptionNode("albedo", "drop_empty", "Leere Bänder entfernen (immer)", False),
    _OptionNode(
        "albedo",
        "interpolate",
        f"Lücken interpolieren (immer, {_config_fields('max_interpolation_gap_nm')})",
        False,
    ),
    _OptionNode("albedo", "albedo", f"Mean Albedo [{_config_fields('use_mean_albedo')}]", True),
)

_ALBEDO_SCALER_NODES: tuple[_OptionNode, ...] = (
    _OptionNode("albedo", "standard_scale", "Standard (wie Spektren)", False),
    _OptionNode("albedo", "minmax_scale", "Min-Max (wie Spektren)", False),
    _OptionNode("albedo", "centering", "Zentrierung (wie Spektren)", False),
    _OptionNode("albedo", "no_scaler", "keine Skalierung (wie Spektren)", False),
)

_SPECTRA_PREFIX_NODES: tuple[_OptionNode, ...] = (
    _OptionNode(
        "spectra",
        "acquisition_centering",
        f"Pro Aufnahme zentrieren [{_config_fields('use_acquisition_centering')}]",
        True,
    ),
    _OptionNode(
        "spectra",
        "drop_empty",
        f"Leere Bänder entfernen [{_config_fields('use_spectra')}, Standard]",
        True,
    ),
    _OptionNode(
        "spectra",
        "interpolate",
        f"Lücken interpolieren (immer, {_config_fields('max_interpolation_gap_nm')})",
        False,
    ),
    _OptionNode(
        "spectra",
        "sensor",
        f"Sentinel-2 simulieren [{_config_fields('spectral_sensor')} = sentinel2]",
        True,
    ),
    _OptionNode(
        "spectra",
        "savgol",
        "Savitzky-Golay ["
        + _config_fields("use_savgol", "savgol_window_length", "savgol_polyorder")
        + "]",
        True,
    ),
    _OptionNode("spectra", "snv", f"SNV [{_config_fields('use_spectral_snv')}]", True),
    _OptionNode(
        "spectra",
        "band_step",
        f"Jedes n-te Band [{_config_fields('spectral_band_step')}]",
        True,
    ),
)

_INDICES_NODES: tuple[_OptionNode, ...] = (
    _OptionNode("indices", "drop_empty", "Leere Bänder entfernen (immer)", False),
    _OptionNode(
        "indices",
        "interpolate",
        f"Lücken interpolieren (immer, {_config_fields('max_interpolation_gap_nm')})",
        False,
    ),
    _OptionNode(
        "indices",
        "indices",
        "Vegetationsindizes [" + _config_fields("use_vegetation_indices") + "]",
        True,
    ),
    _OptionNode("indices", "select", "Nur Indizes behalten", False),
    _OptionNode("indices", "impute", "Fehlende Indizes: Median (Training)", False),
)

_INDEX_SCALER_NODES: tuple[_OptionNode, ...] = (
    _OptionNode(
        "indices",
        "standard_scale",
        f"Standard [{_config_fields('use_spectral_standard_scale')}]",
        True,
    ),
    _OptionNode(
        "indices",
        "minmax_scale",
        f"Min-Max [{_config_fields('use_spectral_minmax_scale')}]",
        True,
    ),
    _OptionNode(
        "indices",
        "centering",
        f"Zentrierung [{_config_fields('use_spectral_centering')}]",
        True,
    ),
    _OptionNode("indices", "no_scaler", "keine Skalierung [Standard]", False),
)

_PRINCIPAL_COMPONENT_NODES: tuple[_OptionNode, ...] = (
    _OptionNode("principal_components", "drop_empty", "Leere Bänder entfernen (immer)", True),
    _OptionNode(
        "principal_components",
        "interpolate",
        f"Lücken interpolieren (immer, {_config_fields('max_interpolation_gap_nm')})",
        True,
    ),
    _OptionNode("principal_components", "standard_scale", "Standard (immer)", True),
    _OptionNode(
        "principal_components",
        "pca",
        "PC_1 + PC_2 [" + _config_fields("use_principal_components") + "]",
        True,
    ),
)

_SCALER_NODES: tuple[_OptionNode, ...] = (
    _OptionNode(
        "spectra",
        "standard_scale",
        f"Standard [{_config_fields('use_spectral_standard_scale')}]",
        False,
    ),
    _OptionNode(
        "spectra", "minmax_scale", f"Min-Max [{_config_fields('use_spectral_minmax_scale')}]", False
    ),
    _OptionNode(
        "spectra", "centering", f"Zentrierung [{_config_fields('use_spectral_centering')}]", False
    ),
    _OptionNode("spectra", "no_scaler", "keine Skalierung [Standard]", False),
)

_REDUCTION_NODES: tuple[_OptionNode, ...] = (
    _OptionNode(
        "spectra",
        "pca",
        f"PCA [{_config_fields('spectral_pca_components')} | braucht Standard/Center]",
        False,
    ),
    _OptionNode(
        "spectra",
        "select_k",
        "Bandauswahl ["
        + _config_fields("spectral_select_k", "spectral_select_score")
        + " | überwacht, je Fold]",
        False,
    ),
    _OptionNode(
        "spectra",
        "ica",
        f"ICA [{_config_fields('spectral_ica_components')} | braucht Standard/Center]",
        False,
    ),
    _OptionNode(
        "spectra",
        "nmf",
        "NMF ["
        + _config_fields("spectral_nmf_components")
        + " | nur roh: ohne SG/SNV/Skalierer/Indizes]",
        False,
    ),
    _OptionNode(
        "spectra",
        "osp",
        f"OSP [{_config_fields('spectral_osp_components')} | nur roh: ohne SNV/Skalierer/Indizes]",
        False,
    ),
    _OptionNode(
        "spectra",
        "lpp",
        f"LPP [{_config_fields('spectral_lpp_components', 'spectral_lpp_neighbors')}]",
        False,
    ),
    _OptionNode("spectra", "vsrp", f"VSRP [{_config_fields('spectral_vsrp_components')}]", False),
    _OptionNode(
        "spectra",
        "dbn",
        f"DBN [{_config_fields('spectral_dbn_layers')} | braucht Min-Max]",
        False,
    ),
    _OptionNode("spectra", "no_reduction", "keine Reduktion [Standard]", False),
)

_DIAMONDS: tuple[_OptionNode, ...] = (
    _OptionNode("", "month", "Monat [use_meta]", True, True),
    _OptionNode("", "scaler", "Skalierung (XOR)", False, True),
    _OptionNode("", "reduction", "Reduktion (max. eine)", False, True),
)


def _diagram_steps() -> set[tuple[str, str]]:
    """Built-preprocessor coordinates shown in the options diagram (CI coverage probe).

    Includes the "keine" placeholders, which are diagram connectors the builder never
    produces; the legacy ``scale`` path is intentionally absent (deprecated).

    Returns:
        (branch, step) pairs covered by the options diagram.
    """
    lanes = (
        _META_NODES,
        _ALBEDO_NODES,
        _ALBEDO_SCALER_NODES,
        _INDICES_NODES,
        _INDEX_SCALER_NODES,
        _PRINCIPAL_COMPONENT_NODES,
        _SPECTRA_PREFIX_NODES,
        _SCALER_NODES,
        _REDUCTION_NODES,
    )
    return {(node.branch, node.step) for lane in lanes for node in lane}


def _node_id(node: _OptionNode) -> str:
    return f"n_{node.branch}_{node.step}" if node.branch else f"d_{node.step}"


def _node_line(node: _OptionNode) -> str:
    label = node.label.replace('"', "'")
    shape = ("{", "}") if node.diamond else ("[", "]")
    return f'        {_node_id(node)}{shape[0]}"{label}"{shape[1]}'


def pipeline_options_diagram() -> str:
    """Mermaid option map of the preprocessing: data-flow order, fields, XOR splits.

    Shows every step ``build_preprocessor`` can produce (not one rendered config):
    dashed edges mark optional steps, diamonds mark exclusive choices, and each box
    names its ``PreprocessingConfig`` fields. Paste the output into
    ``docs/daten/pipeline.md`` (between the ``pipeline-options-diagram`` markers);
    ``tools/check_pipeline_diagram.py`` guards freshness and step coverage.

    Returns:
        Complete mermaid code block for the docs.
    """
    aez, cyclic, numeric, aez_month = _META_NODES
    a_drop, a_inter, albedo = _ALBEDO_NODES
    i_drop, i_inter, indices, select, impute = _INDICES_NODES
    pc_drop, pc_inter, pc_scale, pc_pca = _PRINCIPAL_COMPONENT_NODES
    acquisition, s_drop, s_inter, sensor, savgol, snv, band_step = _SPECTRA_PREFIX_NODES
    month_diamond, scaler_diamond, reduction_diamond = _DIAMONDS

    diagram: list[str] = ["```mermaid", "flowchart TD"]
    diagram.append('    X["Rohdaten (Meta + Bänder)"]')
    diagram.append("    subgraph meta")
    diagram.append(_node_line(aez))
    diagram.append(_node_line(month_diamond))
    diagram.append(_node_line(cyclic))
    diagram.append(_node_line(numeric))
    diagram.append(_node_line(aez_month))
    diagram.append("    end")
    diagram.append("    subgraph albedo")
    for node in (*_ALBEDO_NODES, *_ALBEDO_SCALER_NODES):
        diagram.append(_node_line(node))
    diagram.append("    end")
    diagram.append("    subgraph indizes")
    for node in _INDICES_NODES:
        diagram.append(_node_line(node))
    for node in _INDEX_SCALER_NODES:
        diagram.append(_node_line(node))
    diagram.append("    end")
    diagram.append("    subgraph hauptkomponenten")
    for node in _PRINCIPAL_COMPONENT_NODES:
        diagram.append(_node_line(node))
    diagram.append("    end")
    diagram.append("    subgraph spektren")
    for node in _SPECTRA_PREFIX_NODES:
        diagram.append(_node_line(node))
    diagram.append("    end")
    diagram.append("    subgraph skalierung")
    diagram.append(_node_line(scaler_diamond))
    for node in _SCALER_NODES:
        diagram.append(_node_line(node))
    diagram.append("    end")
    diagram.append("    subgraph reduktion")
    diagram.append(_node_line(reduction_diamond))
    for node in _REDUCTION_NODES:
        diagram.append(_node_line(node))
    diagram.append("    end")
    concat = _OptionNode("", "concat", "Merkmale (verkettet)", False)
    diagram.append(_node_line(concat))

    edges: list[str] = []

    def link(source: str, target: _OptionNode, text: str | None = None) -> None:
        arrow = "-.->" if target.optional else "-->"
        head = f'|"{text}"|' if text else ""
        edges.append(f"    {source} {arrow}{head} {_node_id(target)}")

    link("X", aez)
    link("X", month_diamond)
    link(_node_id(month_diamond), cyclic, "Standard")
    link(_node_id(month_diamond), numeric, "False")
    link("X", a_drop)
    link("X", i_drop)
    link("X", acquisition)
    link(_node_id(acquisition), s_drop)
    link("X", aez_month)
    link(_node_id(a_drop), a_inter)
    link(_node_id(a_inter), albedo)
    for node in _ALBEDO_SCALER_NODES:
        link(_node_id(albedo), node)
        link(_node_id(node), concat)
    link(_node_id(i_drop), i_inter)
    link(_node_id(i_inter), indices)
    link(_node_id(indices), select)
    link(_node_id(select), impute)
    for node in _INDEX_SCALER_NODES:
        link(_node_id(impute), node)
        link(_node_id(node), concat)
    link("X", pc_drop)
    link(_node_id(pc_drop), pc_inter)
    link(_node_id(pc_inter), pc_scale)
    link(_node_id(pc_scale), pc_pca)
    link(_node_id(pc_pca), concat)
    link(_node_id(s_drop), s_inter)
    link(_node_id(s_inter), sensor)
    link(_node_id(sensor), savgol)
    link(_node_id(savgol), snv)
    link(_node_id(snv), band_step)
    link(_node_id(band_step), scaler_diamond)
    for node in _SCALER_NODES:
        link(_node_id(scaler_diamond), node)
        link(_node_id(node), reduction_diamond)
    for node in _REDUCTION_NODES:
        link(_node_id(reduction_diamond), node)
        link(_node_id(node), concat)
    for node in (aez, cyclic, numeric, albedo, aez_month, select):
        link(_node_id(node), concat)
    diagram.extend(edges)
    diagram.append("```")
    return "\n".join(diagram)
