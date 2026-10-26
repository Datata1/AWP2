"""Rule-based flags for implausible acquisitions and labels (diagnosis only, nothing is removed)."""

from collections.abc import Sequence

import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from awp2.config import (
    ANY_FLAG_COL,
    CLASS_OUTLIER_FLAG,
    CROP_COL,
    DARK_FLAG,
    DARK_NIR_NM,
    HAZE_BLUE_NM,
    HAZE_FLAG,
    HAZE_RED_NM,
    HAZE_RED_WEIGHT,
    MIN_CONSENSUS_MODELS,
    MONTH_COL,
    NDVI_INDEX,
    NDVI_OUTLIER_FLAG,
    OUTLIER_MIN_CLASS_ROWS,
    OUTLIER_PCA_COMPONENTS,
    RARE_CONTEXT_FLAG,
    RARE_CONTEXT_MAX_COUNT,
    SEED,
    STAGE_COL,
    SUPPORT_COL,
    TUKEY_FAR_OUT_FACTOR,
    VEGETATION_INDEX_SPECS,
)
from awp2.data import band_columns, combined_label
from awp2.preprocessing import DropEmptyBands, InterpolateBands, nearest_band


def quality_flags(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose spectrum or label looks implausible.

    The flags are candidates for a closer look, not proven errors. Spectral rules flag only
    Tukey's far-out values, globally or within the row's own crop/stage class; the reasons are
    documented with the constants in ``awp2.config``.

    Args:
        X: Raw features (metadata + bands), e.g. ``load_split().X_train``.
        y: Targets with ``Crop`` and ``Stage`` for the same rows.

    Returns:
        One boolean column per flag (``haze``, ``dark``, ``ndvi_outlier``, ``rare_context``,
            ``class_outlier``) plus ``any_flag``, indexed like ``X``.
    """
    spectra = _interpolated(X)
    labels = combined_label(y)
    haze_transform = _reflectance(spectra, HAZE_BLUE_NM) - HAZE_RED_WEIGHT * _reflectance(
        spectra, HAZE_RED_NM
    )
    flags = pd.DataFrame(index=X.index)
    flags[HAZE_FLAG] = _far_out(haze_transform, high=True)
    flags[DARK_FLAG] = _far_out(_reflectance(spectra, DARK_NIR_NM), high=False)
    flags[NDVI_OUTLIER_FLAG] = _ndvi(spectra).groupby(labels).transform(_far_out)
    flags[RARE_CONTEXT_FLAG] = _rare_context(X, y)
    flags[CLASS_OUTLIER_FLAG] = _class_outliers(spectra, labels)
    flags[ANY_FLAG_COL] = flags.any(axis=1)
    return flags


def flag_shares(flags: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Share of flagged rows per group, e.g. per crop/stage class.

    Args:
        flags: Output of ``quality_flags``.
        labels: Group of every row, same index as ``flags``.

    Returns:
        One row per group with the share of each flag and the group size ``support``.
    """
    shares = flags.groupby(labels).mean()
    shares[SUPPORT_COL] = labels.value_counts()
    return shares.sort_values(ANY_FLAG_COL, ascending=False)


def _far_out(values: pd.Series, high: bool | None = None) -> pd.Series:
    q1, q3 = values.quantile([0.25, 0.75])
    fence = TUKEY_FAR_OUT_FACTOR * (q3 - q1)
    too_low, too_high = values < q1 - fence, values > q3 + fence
    if high is None:
        return too_low | too_high
    return too_high if high else too_low


def _interpolated(X: pd.DataFrame) -> pd.DataFrame:
    bands = X[band_columns(X)]
    return InterpolateBands().fit_transform(DropEmptyBands().fit_transform(bands))


def _reflectance(spectra: pd.DataFrame, wavelength_nm: int) -> pd.Series:
    return spectra[nearest_band(wavelength_nm, list(spectra.columns))]


def _ndvi(spectra: pd.DataFrame) -> pd.Series:
    spec = next(spec for spec in VEGETATION_INDEX_SPECS if spec.name == NDVI_INDEX)
    nir_nm, red_nm = spec.target_wavelengths_nm
    nir, red = _reflectance(spectra, nir_nm), _reflectance(spectra, red_nm)
    return (nir - red) / (nir + red)


def _rare_context(X: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    keys = pd.concat([X[MONTH_COL], y[CROP_COL], y[STAGE_COL]], axis=1)
    counts = keys.groupby(list(keys.columns))[MONTH_COL].transform("size")
    return counts <= RARE_CONTEXT_MAX_COUNT


def _class_outliers(spectra: pd.DataFrame, labels: pd.Series) -> pd.Series:
    outlier = pd.Series(False, index=spectra.index)
    for _label, rows in spectra.groupby(labels):
        # too few rows give no stable covariance; these classes stay unflagged
        if len(rows) < OUTLIER_MIN_CLASS_ROWS:
            continue
        scores = make_pipeline(
            StandardScaler(), PCA(n_components=OUTLIER_PCA_COMPONENTS)
        ).fit_transform(rows)
        # a robust covariance keeps a few extreme rows from hiding themselves (masking)
        distance = MinCovDet(random_state=SEED).fit(scores).mahalanobis(scores)
        outlier[rows.index] = _far_out(pd.Series(distance, index=rows.index), high=True)
    return outlier


def consensus_label_suspects(predictions: Sequence[pd.DataFrame], y: pd.DataFrame) -> pd.Series:
    """Rows whose label all models contradict with the same other crop/stage pair.

    Consensus filter after Brodley & Friedl (1999): if different classifiers, each predicting a
    row it was not trained on, agree on another class, the label itself is suspect.

    Args:
        predictions: Out-of-fold predictions with ``Crop`` and ``Stage`` of at least two
            different models, each indexed like ``y`` (e.g. from ``out_of_fold_predictions``).
        y: True ``Crop``/``Stage`` labels.

    Returns:
        ``True`` for every row where all models predict the same pair and it differs from the
            label.

    Raises:
        ValueError: If fewer than two predictions are given.
    """
    if len(predictions) < MIN_CONSENSUS_MODELS:
        raise ValueError(f"need at least {MIN_CONSENSUS_MODELS} models, got {len(predictions)}")
    pairs = [combined_label(prediction.loc[y.index]) for prediction in predictions]
    agree = pd.concat(pairs, axis=1).nunique(axis=1).eq(1)
    return agree & pairs[0].ne(combined_label(y))
