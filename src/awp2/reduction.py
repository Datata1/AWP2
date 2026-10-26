"""Data-reduction study: how much spectral input does the classification quality need?

Every variant is tuned by cross-validation on the training part only, so the resulting curve
can be used to choose a reduced setup before the single validation run.
"""

from collections.abc import Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator

from awp2.config import (
    REDUCTION_BAND_STEPS,
    REDUCTION_BEST_PARAMS_COL,
    REDUCTION_FAMILY_COL,
    REDUCTION_PCA_COMPONENTS,
    REDUCTION_SELECT_K,
    REDUCTION_TUNING_RUN_COL,
    REDUCTION_VARIANT_COL,
    SENTINEL2_SENSOR,
    STD_SUFFIX,
    ReductionFamily,
)
from awp2.data import load_split
from awp2.evaluation import MetricName, evaluate
from awp2.experiment import RunConfig, TuneConfig, cv_folds, out_of_fold_predictions, tune
from awp2.preprocessing import PreprocessingConfig

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
            tuning run id, best hyperparameters, and per metric its mean and
            ``<metric>_std``.
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
        scores = fold_scores(out_of_fold_predictions(tuned.best_model, config), config, metrics)
        rows.append(
            {
                REDUCTION_VARIANT_COL: variant.name,
                REDUCTION_FAMILY_COL: variant.family,
                **tuned.input_size._asdict(),
                REDUCTION_TUNING_RUN_COL: tuned.run_id,
                REDUCTION_BEST_PARAMS_COL: tuned.best_params,
                **scores.mean().to_dict(),
                **scores.std().add_suffix(STD_SUFFIX).to_dict(),
            }
        )
    return pd.DataFrame(rows)


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
