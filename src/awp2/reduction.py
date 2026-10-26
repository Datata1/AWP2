"""Data-reduction study: how much spectral input does the classification quality need?

Every variant is tuned by cross-validation on the training part only, so the resulting curve
can be used to choose a reduced setup before the single validation run.
"""

from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from awp2.config import (
    BAND_PATTERN,
    FIT_TIME_COL,
    MIN_CV_GAIN,
    N_COMPONENTS_COL,
    N_INPUT_FEATURES_COL,
    N_MEASURED_BANDS_COL,
    PCA_MIN_SPEEDUP,
    PREDICT_TIME_COL,
    REDUCTION_BAND_STEPS,
    REDUCTION_BEST_PARAMS_COL,
    REDUCTION_FAMILY_COL,
    REDUCTION_MAX_CV_LOSS,
    REDUCTION_PCA_COMPONENTS,
    REDUCTION_SELECT_K,
    REDUCTION_TUNING_RUN_COL,
    REDUCTION_VARIANT_COL,
    SEED,
    SENTINEL2_SENSOR,
    STD_SUFFIX,
    ReductionFamily,
)
from awp2.data import load_split
from awp2.evaluation import MetricName, evaluate
from awp2.experiment import (
    RunConfig,
    TuneConfig,
    cv_folds,
    timed_out_of_fold_predictions,
    tune,
)
from awp2.preprocessing import PreprocessingConfig, build_preprocessor, input_size

CURVE_METRICS: tuple[MetricName, ...] = ("bacc_crop", "bacc_stage", "bacc_combined")


class ReductionVariant(BaseModel):
    """One input setting of the study."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(description="Short slug, appended to the tuning run name.")
    family: ReductionFamily = Field(description="Kind of reduction, one curve per family.")
    preprocessing: PreprocessingConfig = Field(description="Preprocessing of this variant.")


def reduction_variants(base: PreprocessingConfig) -> list[ReductionVariant]:
    """Build the standard variants of the study on top of a base preprocessing.

    Args:
        base: Preprocessing of the reference model with all bands; it must allow PCA, i.e.
            standardise or centre the spectra.

    Returns:
        All bands, PCA, band selection, every n-th band, Sentinel-2, indices only and
            metadata only, with the sizes from ``awp2.config``.

    Raises:
        ValueError: If ``base`` cannot be combined with one of the variants.
    """
    variants = [ReductionVariant(name="all_bands", family="all_bands", preprocessing=base)]
    variants += [
        _variant(f"pca_{k}", "pca", base, spectral_pca_components=k)
        for k in REDUCTION_PCA_COMPONENTS
    ]
    variants += [
        _variant(f"select_{k}", "band_selection", base, spectral_select_k=k)
        for k in REDUCTION_SELECT_K
    ]
    variants += [
        _variant(f"step_{step}", "band_step", base, spectral_band_step=step)
        for step in REDUCTION_BAND_STEPS
    ]
    variants.append(_variant("sentinel2", "sentinel2", base, spectral_sensor=SENTINEL2_SENSOR))
    variants.append(
        _variant(
            "indices_only", "indices_only", base, use_spectra=False, use_vegetation_indices=True
        )
    )
    variants.append(
        _variant(
            "meta_only",
            "meta_only",
            base,
            use_spectra=False,
            use_vegetation_indices=False,
            use_mean_albedo=False,
            use_principal_components=False,
        )
    )
    return variants


def reduction_curve(
    model: BaseEstimator,
    variants: Sequence[ReductionVariant],
    template: TuneConfig,
    metrics: Sequence[MetricName] = CURVE_METRICS,
) -> pd.DataFrame:
    """Tune ``model`` for every variant and score its best setting per training fold.

    The search optimises ``template.metric``; the curve then reports every metric in
    ``metrics`` from out-of-fold predictions of the best setting, so crop and stage can be
    compared. The validation part is never touched.

    Args:
        model: Unfitted estimator that predicts ``Crop`` and ``Stage``.
        variants: Input settings, e.g. from ``reduction_variants``.
        template: Search settings shared by all variants; ``name`` becomes the prefix of the
            tuning runs and ``preprocessing`` is replaced per variant.
        metrics: Metrics to report as mean and std over the folds.

    Returns:
        One row per variant: variant, family, ``n_input_features``, ``n_measured_bands``,
            tuning run id, best hyperparameters, per metric its mean and ``<metric>_std``,
            and the mean and std of ``fit_s`` and ``predict_s`` over the folds.
    """
    rows = []
    for variant in variants:
        config = TuneConfig.model_validate(
            {
                **dict(template),
                "name": f"{template.name}_{variant.name}",
                "preprocessing": variant.preprocessing,
            }
        )
        tuned = tune(model, config)
        timed = timed_out_of_fold_predictions(tuned.best_model, config)
        scores = fold_scores(timed.predictions, config, metrics)
        rows.append(
            {
                REDUCTION_VARIANT_COL: variant.name,
                REDUCTION_FAMILY_COL: variant.family,
                **tuned.input_size._asdict(),
                REDUCTION_TUNING_RUN_COL: tuned.run_id,
                REDUCTION_BEST_PARAMS_COL: tuned.best_params,
                **scores.mean().to_dict(),
                **scores.std().add_suffix(STD_SUFFIX).to_dict(),
                **timed.timings.mean().to_dict(),
                **timed.timings.std().add_suffix(STD_SUFFIX).to_dict(),
            }
        )
    return pd.DataFrame(rows)


def pca_curve(
    model: BaseEstimator,
    reference: ReductionVariant,
    components: Sequence[int],
    template: TuneConfig,
    metrics: Sequence[MetricName] = CURVE_METRICS,
) -> pd.DataFrame:
    """Reduction curve of ``reference`` and of PCA with each number of components on top of it.

    PCA is fitted inside every training fold, after the spectral steps of ``reference``, and
    every variant gets its own search (``template.param_grid``).

    Args:
        model: Unfitted estimator that predicts ``Crop`` and ``Stage``.
        reference: Input without PCA, e.g. all Hyperion bands or the Sentinel-2 bands.
        components: Numbers of principal components to try, each below the number of bands
            of ``reference``.
        template: Search settings, as for ``reduction_curve``.
        metrics: Metrics to report as mean and std over the folds.

    Returns:
        The rows of ``reduction_curve`` with the reference first, plus ``n_components``: the
            spectral features the model receives (all bands for the reference).

    Raises:
        ValueError: If ``reference`` already reduces with PCA, has fewer bands than a ``k``,
            or cannot be combined with PCA (it must standardise or centre the spectra).
    """
    if reference.preprocessing.spectral_pca_components is not None:
        raise ValueError("reference must not use PCA itself")
    n_bands = input_size(
        build_preprocessor(reference.preprocessing).fit(load_split().X_train)
    ).n_measured_bands
    if max(components) >= n_bands:
        raise ValueError(f"every k must be below the {n_bands} bands of the reference")
    variants = [reference] + [
        _variant(
            f"{reference.name}_pca_{k}", "pca", reference.preprocessing, spectral_pca_components=k
        )
        for k in components
    ]
    curve = reduction_curve(model, variants, template, metrics)
    measured_bands = curve[N_MEASURED_BANDS_COL].iloc[0]
    curve[N_COMPONENTS_COL] = [measured_bands, *components]
    return curve


class SpectralPCA(NamedTuple):
    """A PCA of the preprocessed training spectra, for inspection only.

    Attributes:
        loadings: One row per spectral feature (index: band name ``X<nm>``), one column per
            component (``PC1``, ``PC2``, ...).
        explained_variance_ratio: Share of the total spectral variance per component, indexed
            like the columns of ``loadings``.
        scores: Component values of every training row, indexed like ``load_split().X_train``.
    """

    loadings: pd.DataFrame
    explained_variance_ratio: pd.Series
    scores: pd.DataFrame


def spectral_pca(preprocessing: PreprocessingConfig) -> SpectralPCA:
    """Fit a PCA with all components on the training part's preprocessed spectra.

    The spectra pass the same steps as in the model (drop empty bands, interpolation, sensor
    simulation, scaling) but no metadata. The fit uses the whole training part and serves
    figures and explained-variance tables; models fit their PCA inside the folds.

    Args:
        preprocessing: Spectral settings without PCA, e.g. those of the reference model.

    Returns:
        Loadings, explained-variance ratio and training scores of every component.

    Raises:
        ValueError: If ``preprocessing`` uses PCA, drops the spectra or adds non-band features.
    """
    if preprocessing.spectral_pca_components is not None or not preprocessing.use_spectra:
        raise ValueError("spectral_pca needs the spectra without PCA")
    spectra_only = PreprocessingConfig.model_validate({**dict(preprocessing), "use_meta": False})
    X = load_split().X_train
    features = build_preprocessor(spectra_only).fit_transform(X)
    bands = features.filter(regex=BAND_PATTERN)
    if bands.shape[1] != features.shape[1]:
        raise ValueError("spectral_pca needs a preprocessing that yields only spectral bands")
    pca = PCA(random_state=SEED).fit(bands)
    names = [f"PC{number}" for number in range(1, pca.n_components_ + 1)]
    return SpectralPCA(
        loadings=pd.DataFrame(pca.components_.T, index=bands.columns, columns=names),
        explained_variance_ratio=pd.Series(pca.explained_variance_ratio_, index=names),
        scores=pd.DataFrame(pca.transform(bands), index=bands.index, columns=names),
    )


def smallest_equivalent(
    curve: pd.DataFrame,
    reference: str,
    metric: MetricName = "bacc_combined",
    size: str = N_INPUT_FEATURES_COL,
    max_loss: float = REDUCTION_MAX_CV_LOSS,
) -> pd.Series | None:
    """Find the smallest variant that loses at most ``max_loss`` against the reference.

    Args:
        curve: Output of ``reduction_curve`` or ``pca_curve``.
        reference: Name of the reference variant in ``curve``.
        metric: CV score the loss is measured on.
        size: Column that defines "smallest", e.g. ``n_input_features`` or
            ``n_measured_bands``.
        max_loss: Largest tolerated drop of ``metric``.

    Returns:
        The row of the chosen variant, or ``None`` if no other variant is within ``max_loss``.
    """
    by_name = curve.set_index(REDUCTION_VARIANT_COL)
    threshold = by_name.loc[reference, metric] - max_loss
    others = by_name.drop(index=reference)
    equivalent = others[others[metric] >= threshold]
    if equivalent.empty:
        return None
    return equivalent.sort_values(size, kind="stable").iloc[0]


def explained_variance_at(curve: pd.DataFrame, explained_variance_ratio: pd.Series) -> pd.Series:
    """Cumulative explained variance for every row of a ``pca_curve``.

    Args:
        curve: Output of ``pca_curve``; its first row is the reference without PCA.
        explained_variance_ratio: Variance share per component, e.g. from ``spectral_pca``.

    Returns:
        Share of the spectral variance the model keeps, indexed by ``n_components``; 1 for the
            reference.
    """
    cumulative = explained_variance_ratio.cumsum().to_numpy()
    components = curve[N_COMPONENTS_COL].to_numpy()
    shares = [1.0] + [float(cumulative[k - 1]) for k in components[1:]]
    return pd.Series(shares, index=components, name="explained_variance")


class PcaDecision(BaseModel):
    """Whether the knee of a PCA curve is worth a validation run."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    knee: str | None = Field(description="Variant at the knee, None if no k is within the loss.")
    cv_gain: float | None = Field(description="CV score of the knee minus the reference.")
    speedup: float | None = Field(
        description="Fit + predict time of the reference divided by that of the knee."
    )
    recommended: bool = Field(
        description="Knee is better by MIN_CV_GAIN or PCA_MIN_SPEEDUP x faster."
    )


def pca_recommendation(curve: pd.DataFrame, metric: MetricName = "bacc_combined") -> PcaDecision:
    """Apply the PCA rule: the knee must beat the reference clearly or save clear time.

    The knee is the fewest components within ``REDUCTION_MAX_CV_LOSS`` of the reference
    (``smallest_equivalent``). Measured bands never drop with PCA, so it only counts if it
    gains at least ``MIN_CV_GAIN`` or is at least ``PCA_MIN_SPEEDUP`` times faster.

    Args:
        curve: Output of ``pca_curve``; its first row is the reference without PCA.
        metric: CV score of the rule.

    Returns:
        Knee, its gain and speedup, and whether it is recommended.
    """
    reference = curve.iloc[0]
    knee = smallest_equivalent(curve, reference[REDUCTION_VARIANT_COL], metric)
    if knee is None:
        return PcaDecision(knee=None, cv_gain=None, speedup=None, recommended=False)
    cv_gain = float(knee[metric] - reference[metric])
    speedup = float(
        (reference[FIT_TIME_COL] + reference[PREDICT_TIME_COL])
        / (knee[FIT_TIME_COL] + knee[PREDICT_TIME_COL])
    )
    return PcaDecision(
        knee=str(knee.name),
        cv_gain=cv_gain,
        speedup=speedup,
        recommended=cv_gain >= MIN_CV_GAIN or speedup >= PCA_MIN_SPEEDUP,
    )


def _variant(
    name: str, family: ReductionFamily, base: PreprocessingConfig, **changes: object
) -> ReductionVariant:
    preprocessing = PreprocessingConfig.model_validate({**dict(base), **changes})
    return ReductionVariant(name=name, family=family, preprocessing=preprocessing)


def fold_scores(
    predictions: pd.DataFrame,
    config: TuneConfig | RunConfig,
    metrics: Sequence[MetricName] = CURVE_METRICS,
) -> pd.DataFrame:
    """Score out-of-fold predictions separately on every held-out fold.

    Args:
        predictions: Output of ``out_of_fold_predictions`` for ``config``.
        config: Configuration whose ``folds`` produced the predictions.
        metrics: Metrics to report.

    Returns:
        One row per fold, one column per metric.
    """
    y = load_split().y_train
    return pd.DataFrame(
        [
            evaluate(y.iloc[fold.val], predictions.iloc[fold.val]).model_dump()
            for fold in cv_folds(config.folds)
        ]
    )[list(metrics)]
