"""Preprocessing steps as sklearn transformers – always fitted on the training split only."""

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from awp2.config import BAND_PATTERN, BAND_PREFIX
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

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the output columns."""
        return np.asarray(self.feature_names_out_, dtype=object)


class DropEmptyBands(_BandTransformer):
    """Drop bands that contain no values at all in the training data."""

    def fit(self, X: pd.DataFrame, y=None) -> "DropEmptyBands":
        """Learn which bands are completely empty."""
        X = self._check(X)
        self._set_input(X)
        self.empty_bands_ = [c for c in X.columns if X[c].isna().all()]
        self.feature_names_out_ = [c for c in X.columns if c not in self.empty_bands_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` without the empty bands."""
        return self._check(X)[self.feature_names_out_]


class InterpolateBands(_BandTransformer):
    """Fill missing band values per spectrum along the wavelength.

    - Between two measured bands: linear interpolation by wavelength (neighbouring bands are
      strongly correlated, so this beats a column mean).
    - At the edges of a spectrum: value of the nearest measured band (spectra differ a lot in
      brightness, so the neighbour is a better estimate than a global median).
    - Spectrum without any value: training median per band.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "InterpolateBands":
        """Store the training medians as last-resort fallback."""
        X = self._check(X)
        self._set_input(X)
        self.medians_ = X.median()
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` without missing values."""
        X = self._check(X)
        if not X.isna().any().any():
            return X
        return _interpolate_along_wavelength(X).fillna(self.medians_)


def _interpolate_along_wavelength(spectra: pd.DataFrame) -> pd.DataFrame:
    by_wavelength = spectra.set_axis(wavelengths(spectra), axis=1)
    filled = by_wavelength.T.interpolate(method="index", limit_direction="both").T
    return filled.set_axis(spectra.columns, axis=1)


class PreprocessingConfig(BaseModel):
    """Options of the standard preprocessing; new steps get a field here, not ad-hoc code."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    use_meta: bool = Field(
        default=True, description="Use AEZ (one-hot) and Month as features next to the spectrum."
    )
    scale: bool = Field(
        default=False,
        description="Standardise spectra and month – on for SVM, logistic regression, MLP; "
        "off for tree ensembles.",
    )


def build_preprocessor(config: PreprocessingConfig | None = None) -> ColumnTransformer:
    """Build the standard preprocessing described by ``config`` (defaults if omitted)."""
    config = config or PreprocessingConfig()
    spectra: list[tuple[str, BaseEstimator]] = [
        ("drop_empty", DropEmptyBands()),
        ("interpolate", InterpolateBands()),
    ]
    if config.scale:
        spectra.append(("scale", StandardScaler()))

    transformers: list[tuple[str, BaseEstimator | str, object]] = [
        ("spectra", Pipeline(spectra), make_column_selector(pattern=BAND_PATTERN))
    ]
    if config.use_meta:
        transformers += [
            ("aez", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["AEZ"]),
            ("month", StandardScaler() if config.scale else "passthrough", ["Month"]),
        ]
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")
