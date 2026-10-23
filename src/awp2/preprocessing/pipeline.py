"""Pipeline for the shared preprocessing pipeline."""

import re
import warnings
from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from awp2.config import (
    AEZ_COL,
    BAND_PATTERN,
    MONTH_COL,
    SEED,
    SPECTRAL_REGIONS,
)
from awp2.data import wavelength
from awp2.preprocessing.bands import (
    DropEmptyBands,
    InterpolateBands,
    MeanAlbedo,
    SavitzkyGolaySmoothing,
    StandardNormalVariate,
)
from awp2.preprocessing.decomposition import (
    DeepBeliefFeatures,
    LocalityPreservingProjection,
    OrthogonalSubspaceProjection,
    PrincipalComponentFeatures,
    VerySparseRandomProjection,
)
from awp2.preprocessing.indices import AddVegetationIndices, SelectVegetationIndices
from awp2.preprocessing.metadata import AezMonthEncoder, CyclicMonthEncoder
from awp2.preprocessing.options import PreprocessingConfig


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
    spectra = _clean_band_steps(config)
    if config.use_savgol:
        assert config.savgol_window_length is not None and config.savgol_polyorder is not None
        spectra.append(
            ("savgol", SavitzkyGolaySmoothing(config.savgol_window_length, config.savgol_polyorder))
        )
    if config.use_spectral_snv:
        spectra.append(("snv", StandardNormalVariate()))
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
                        *_clean_band_steps(config),
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
            *_clean_band_steps(config),
            ("indices", AddVegetationIndices()),
            ("select", SelectVegetationIndices()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            *_spectral_scaler(config),
        ]
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
                        *_clean_band_steps(config),
                        ("standard_scale", StandardScaler()),
                        ("pca", PrincipalComponentFeatures()),
                    ]
                ),
                band_selector,
            )
        )
    if config.use_aez_month:
        transformers.append(("aez_month", AezMonthEncoder(), [AEZ_COL, MONTH_COL]))
    transformers.append(("spectra", Pipeline(spectra), band_selector))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")


def _clean_band_steps(config: PreprocessingConfig) -> list[tuple[str, BaseEstimator]]:
    return [
        ("drop_empty", DropEmptyBands()),
        ("interpolate", InterpolateBands(config.max_interpolation_gap_nm)),
    ]


def _spectral_scaler(config: PreprocessingConfig) -> list[tuple[str, BaseEstimator]]:
    if config.use_spectral_standard_scale or config.scale:
        return [("standard_scale", StandardScaler())]
    if config.use_spectral_minmax_scale:
        return [("minmax_scale", MinMaxScaler())]
    if config.use_spectral_centering:
        return [("centering", StandardScaler(with_mean=True, with_std=False))]
    return []
