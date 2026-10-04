"""Preprocessing steps as sklearn transformers – always fitted on the training split only."""

import re
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from awp2.config import (
    AEZ_COL,
    BAND_PATTERN,
    BAND_PREFIX,
    MONTH_COL,
    NDRE_RED_EDGE_BAND,
    NDVI_NIR_BAND,
    NDVI_RED_BAND,
    NDWI_SWIR_BAND,
    PRI_FIRST_BAND,
    PRI_SECOND_BAND,
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

    - Between two measured bands: linear interpolation by wavelength (neighbouring bands are
      strongly correlated, so this beats a column mean).
    - At the edges of a spectrum: value of the nearest measured band (spectra differ a lot in
      brightness, so the neighbour is a better estimate than a global median).
    - Spectrum without any value: training median per band.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Store the training medians as last-resort fallback.

        Args:
            X: Band columns of the training part.
            y: Ignored; accepted for sklearn compatibility.

        Returns:
            The fitted transformer.
        """
        X = self._check(X)
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
        return _interpolate_along_wavelength(X).fillna(self.medians_)


def _interpolate_along_wavelength(spectra: pd.DataFrame) -> pd.DataFrame:
    by_wavelength = spectra.set_axis(wavelengths(spectra), axis=1)
    filled = by_wavelength.T.interpolate(method="index", limit_direction="both").T
    return filled.set_axis(spectra.columns, axis=1)


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
        description="Standardise spectra and month – on for SVM, logistic regression, MLP; "
        "off for tree ensembles.",
    )
    use_vegetation_indices: bool = Field(
        default=False,
        description="Append EDA-defined NDVI, NDRE, PRI and NDWI to the spectral bands.",
    )


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
            ("preprocess", build_preprocessor(PreprocessingConfig(scale=True))),
            ("model", SVC()),
        ])
        ```
    """
    config = config or PreprocessingConfig()
    spectra: list[tuple[str, BaseEstimator]] = [
        ("drop_empty", DropEmptyBands()),
        ("interpolate", InterpolateBands()),
    ]
    if config.use_vegetation_indices:
        spectra.append(("vegetation_indices", AddVegetationIndices()))
    if config.scale:
        spectra.append(("scale", StandardScaler()))

    transformers: list[tuple[str, BaseEstimator | str, object]] = [
        ("spectra", Pipeline(spectra), make_column_selector(pattern=BAND_PATTERN))
    ]
    if config.use_meta:
        transformers += [
            ("aez", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [AEZ_COL]),
            ("month", StandardScaler() if config.scale else "passthrough", [MONTH_COL]),
        ]
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")
