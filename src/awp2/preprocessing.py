"""Preprocessing steps as sklearn transformers – always fitted on the training split only."""

import re
import warnings
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.linalg import eigh
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.neighbors import kneighbors_graph
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from awp2.config import (
    AEZ_COL,
    BAND_PATTERN,
    BAND_PREFIX,
    MONTH_COL,
    MONTH_COS_COL,
    MONTH_SIN_COL,
    NDRE_RED_EDGE_BAND,
    NDVI_NIR_BAND,
    NDVI_RED_BAND,
    NDWI_SWIR_BAND,
    PRI_FIRST_BAND,
    PRI_SECOND_BAND,
    SAVGOL_SPLIT_GAP_NM,
    SEED,
    VEGETATION_INDEX_BANDS,
    VEGETATION_INDEX_NAMES,
)
from awp2.data import wavelengths


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
                    values[:, start:end], self.window_length, self.polyorder, axis=1,
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
    """Append NDVI, NDRE, PRI and NDWI to a frame of spectral bands."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate the required bands and record output feature names.

        Args:
            X: Band columns after missing-value processing.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If a required band is unavailable.
        """
        X = self._check(X)
        self._set_input(X)
        missing_bands = sorted(set(VEGETATION_INDEX_BANDS) - set(X.columns))
        if missing_bands:
            raise ValueError(f"Vegetation indices require unavailable bands: {missing_bands}")
        self.feature_names_out_ = [*X.columns, *VEGETATION_INDEX_NAMES]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the four index columns.

        Args:
            X: Band columns with the same columns as in ``fit``.

        Returns:
            The input bands plus the vegetation indices.
        """
        X = self._check(X)
        indices = pd.DataFrame(
            {
                "NDVI": (X[NDVI_NIR_BAND] - X[NDVI_RED_BAND])
                / (X[NDVI_NIR_BAND] + X[NDVI_RED_BAND]),
                "NDRE": (X[NDVI_NIR_BAND] - X[NDRE_RED_EDGE_BAND])
                / (X[NDVI_NIR_BAND] + X[NDRE_RED_EDGE_BAND]),
                "PRI": (X[PRI_FIRST_BAND] - X[PRI_SECOND_BAND])
                / (X[PRI_FIRST_BAND] + X[PRI_SECOND_BAND]),
                "NDWI": (X[NDVI_NIR_BAND] - X[NDWI_SWIR_BAND])
                / (X[NDVI_NIR_BAND] + X[NDWI_SWIR_BAND]),
            },
            index=X.index,
        )
        return pd.concat([X, indices], axis=1)


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


class PreprocessingConfig(BaseModel):
    """Options of the standard preprocessing.

    New preprocessing steps get a field here (and a transformer in this module) instead of
    ad-hoc code in notebooks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    use_meta: bool = Field(
        default=True, description="Use AEZ (one-hot) and Month as features next to the spectrum."
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
        description="Append EDA-defined NDVI, NDRE, PRI and NDWI to the spectral bands.",
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
            ValueError: If more than one reduction method is selected; if centering,
                scalers, PCA, ICA or the deprecated ``scale`` flag are combined ambiguously;
                if NMF, OSP or DBN get unsuitable input; or if Savitzky-Golay or LPP
                arguments are set without their method or missing with it.
        """
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
            self.spectral_ica_components,
            self.spectral_nmf_components,
            self.spectral_osp_components,
            self.spectral_lpp_components,
            self.spectral_vsrp_components,
        ]
        if sum(selected is not None for selected in reductions) + (
            self.spectral_dbn_layers is not None
        ) > 1:
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
    if config.scale:
        warnings.warn(
            "PreprocessingConfig(scale=...) is deprecated and will be removed soon; "
            "use use_spectral_standard_scale instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    spectra: list[tuple[str, BaseEstimator]] = [
        ("drop_empty", DropEmptyBands()),
        ("interpolate", InterpolateBands(config.max_interpolation_gap_nm)),
    ]
    if config.use_savgol:
        assert config.savgol_window_length is not None and config.savgol_polyorder is not None
        spectra.append(
            ("savgol", SavitzkyGolaySmoothing(
                config.savgol_window_length, config.savgol_polyorder
            ))
        )
    if config.use_spectral_snv:
        spectra.append(("snv", StandardNormalVariate()))
    if config.use_vegetation_indices:
        spectra.append(("vegetation_indices", AddVegetationIndices()))
    if config.use_spectral_standard_scale or config.scale:
        spectra.append(("standard_scale", StandardScaler()))
    elif config.use_spectral_minmax_scale:
        spectra.append(("minmax_scale", MinMaxScaler()))
    elif config.use_spectral_centering:
        spectra.append(("centering", StandardScaler(with_mean=True, with_std=False)))
    pca_components = config.spectral_pca_components
    ica_components = config.spectral_ica_components
    nmf_components = config.spectral_nmf_components
    osp_components = config.spectral_osp_components
    lpp_components = config.spectral_lpp_components
    vsrp_components = config.spectral_vsrp_components
    dbn_layers = config.spectral_dbn_layers
    if pca_components is not None:
        spectra.append(("pca", PCA(n_components=pca_components, random_state=SEED)))
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
        transformers.append(
            ("aez", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [AEZ_COL])
        )
        if config.use_cyclic_month:
            month_transformers: list[tuple[str, BaseEstimator]] = [("cyclic", CyclicMonthEncoder())]
            if config.scale:
                month_transformers.append(("scale", StandardScaler()))
            transformers.append(("month", Pipeline(month_transformers), [MONTH_COL]))
        else:
            transformers.append(
                ("month", StandardScaler() if config.scale else "passthrough", [MONTH_COL])
            )
    transformers.append(("spectra", Pipeline(spectra), make_column_selector(pattern=BAND_PATTERN)))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")
