"""Indices for the shared preprocessing pipeline."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from awp2.config import (
    INDEX_WAVELENGTH_TOLERANCE_NM,
    MCARI_GREEN_WEIGHT,
    NDVI_NIR_BAND,
    NDVI_RED_BAND,
    OSAVI_SOIL_ADJUSTMENT,
    PCA_FEATURE_NAMES,
    REFLECTANCE_PERCENT_SCALE,
    REIP_LAGRANGIAN_MIN_BANDS,
    REIP_LAGRANGIAN_WINDOW_NM,
    REIP_POLYNOMIAL_DEGREE,
    REIP_POLYNOMIAL_WINDOW_NM,
    TCARI_MULTIPLIER,
    VEGETATION_INDEX_NAMES,
    VEGETATION_INDEX_SPECS,
)
from awp2.data import wavelength
from awp2.preprocessing.bands import _BandTransformer


def ndvi(bands: pd.DataFrame) -> pd.Series:
    """Return diagnostic NDVI from the fixed 854 nm and 671 nm bands.

    This diagnostic is separate from the wavelength-mapped vegetation-index catalogue.

    Args:
        bands: Reflectance containing ``NDVI_NIR_BAND`` and ``NDVI_RED_BAND``.

    Returns:
        NDVI with the input index, without imputation or clipping. Missing inputs and
            zero denominators retain pandas' NaN and infinity behavior.

    Raises:
        KeyError: If a required band column is missing.
    """
    nir, red = bands[NDVI_NIR_BAND], bands[NDVI_RED_BAND]
    return (nir - red) / (nir + red)


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
        third_derivative = np.polynomial.polynomial.polyder(second_derivative)
        roots = np.polynomial.polynomial.polyroots(second_derivative)
        candidates = [
            float(root.real)
            for root in roots
            if np.isclose(root.imag, 0.0)
            and centered_wavelengths[0] <= root.real <= centered_wavelengths[-1]
            and np.polynomial.polynomial.polyval(root.real, third_derivative) < 0
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
