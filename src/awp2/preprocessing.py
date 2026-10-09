"""Preprocessing steps as sklearn transformers – always fitted on the training split only."""

import re
import warnings
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
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
    spectral_pca_components: int | None = Field(
        default=None,
        gt=0,
        description="Reduce the (standardised) spectral bands to this many principal components; "
        "None keeps all bands. Requires use_spectral_standard_scale.",
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
    def _check_scaler_options(self) -> Self:
        """Reject ambiguous scaler combinations.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If both spectral scalers, a spectral scaler together with the
                deprecated ``scale`` flag, or PCA without the spectral standard scaler
                are requested.
        """
        if self.use_spectral_standard_scale and self.use_spectral_minmax_scale:
            raise ValueError("use only one of use_spectral_standard_scale/minmax_scale")
        if self.scale and (self.use_spectral_standard_scale or self.use_spectral_minmax_scale):
            raise ValueError("deprecated 'scale' must not be combined with the spectral scalers")
        if self.spectral_pca_components is not None and not self.use_spectral_standard_scale:
            raise ValueError("spectral_pca_components requires use_spectral_standard_scale=True")
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
    if config.use_vegetation_indices:
        spectra.append(("vegetation_indices", AddVegetationIndices()))
    if config.use_spectral_standard_scale or config.scale:
        spectra.append(("standard_scale", StandardScaler()))
    elif config.use_spectral_minmax_scale:
        spectra.append(("minmax_scale", MinMaxScaler()))
    if config.spectral_pca_components is not None:
        spectra.append(
            ("pca", PCA(n_components=config.spectral_pca_components, random_state=SEED))
        )

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
