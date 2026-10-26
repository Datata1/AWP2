"""Retrain the reference model on or without single spectral regions to compare their value.

Permutation importance of the full model is blurred by redundancy: when a region is shuffled,
neighbouring regions carry the same information. Retraining answers two separate questions:
how much a region achieves **alone** (``only``) and what the model loses **without** it
(``without``). Everything runs on the cross-validation folds of the training part.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator

from awp2.config import (
    ALL_BANDS_NO_META_VARIANT,
    ALL_BANDS_VARIANT,
    ALL_GROUPS_KIND,
    BAND_PATTERN,
    COMBINED_METRIC,
    GROUP_STUDY_APPROACH,
    GROUP_STUDY_C_GRID,
    GROUP_STUDY_C_PARAM,
    GROUP_STUDY_EXPERIMENT,
    GROUP_STUDY_GAMMA_GRID,
    GROUP_STUDY_GAMMA_PARAM,
    GROUP_STUDY_N_JOBS,
    GROUP_STUDY_RUN_PREFIX,
    META_ONLY_KIND,
    META_ONLY_VARIANT,
    NO_META_VARIANT_SUFFIX,
    ONLY_GROUP_KIND,
    ONLY_GROUP_NO_META_KIND,
    ONLY_VARIANT_PREFIX,
    SLUG_PATTERN,
    SPECTRAL_REGIONS,
    WITHOUT_GROUP_KIND,
    WITHOUT_VARIANT_PREFIX,
    GroupVariantKind,
    SpectralRegion,
)
from awp2.data import load_folds, load_split, valid_combinations
from awp2.data.split import Fold
from awp2.evaluation import METRIC_LABELS, as_target_frame, evaluate
from awp2.experiment import TuneConfig, _pipeline, tune
from awp2.preprocessing import PreprocessingConfig, build_preprocessor

STUDY_METRICS = tuple(METRIC_LABELS)


class GroupVariant(BaseModel):
    """One way to restrict the reference model to or exclude spectral regions."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(pattern=SLUG_PATTERN, description="Slug used for the MLflow run name.")
    kind: GroupVariantKind = Field(description="Which question the variant answers.")
    group: str | None = Field(
        description="Region the variant is about; None if it covers all or no region."
    )
    spectral_regions: tuple[str, ...] | None = Field(
        description="Regions the model sees; None for all bands, empty for none."
    )
    use_meta: bool = Field(description="Whether AEZ and Month are features.")


class CvScores(NamedTuple):
    """Cross-validated scores of one configuration.

    Attributes:
        mean: Mean over the folds per metric of ``METRIC_LABELS``.
        std: Standard deviation over the folds.
        folds: Score per fold (rows) and metric (columns).
    """

    mean: pd.Series
    std: pd.Series
    folds: pd.DataFrame


class VariantResult(NamedTuple):
    """Outcome of tuning and cross-validating one variant.

    Attributes:
        variant: The variant that was trained.
        n_bands: Spectral bands the model sees after preprocessing the whole training part.
        n_features: All model-ready features, including metadata, counted the same way.
        best_params: Hyperparameters chosen by the cross-validation score. The score is the
            maximum over the searched candidates, so it is slightly optimistic in absolute terms.
        tuning_run: MLflow run id of the search, ``None`` if untracked.
        cv: Cross-validated scores of the chosen hyperparameters.
    """

    variant: GroupVariant
    n_bands: int
    n_features: int
    best_params: dict[str, float]
    tuning_run: str | None
    cv: CvScores


class PairedDifference(NamedTuple):
    """Fold-wise difference of one metric between two variants.

    Both variants are scored on the same folds, so the difference removes the fold-to-fold
    variation that both share.

    Attributes:
        mean: Mean difference ``first - second`` over the folds.
        std: Standard deviation of the fold-wise differences.
        folds_first_better: Number of folds in which the first variant scores higher.
        n_folds: Number of folds.
    """

    mean: float
    std: float
    folds_first_better: int
    n_folds: int


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def group_variants(regions: Sequence[SpectralRegion] = SPECTRAL_REGIONS) -> list[GroupVariant]:
    """List the variants of the study.

    Args:
        regions: Spectral regions to take apart.

    Returns:
        The reference (all bands and metadata), all bands without metadata, metadata alone and,
            per region, the region alone with and without metadata plus all other regions with
            metadata.
    """
    names = [region.name for region in regions]
    variants = [
        GroupVariant(
            name=ALL_BANDS_VARIANT,
            kind=ALL_GROUPS_KIND,
            group=None,
            spectral_regions=None,
            use_meta=True,
        ),
        GroupVariant(
            name=ALL_BANDS_NO_META_VARIANT,
            kind=ALL_GROUPS_KIND,
            group=None,
            spectral_regions=None,
            use_meta=False,
        ),
        GroupVariant(
            name=META_ONLY_VARIANT,
            kind=META_ONLY_KIND,
            group=None,
            spectral_regions=(),
            use_meta=True,
        ),
    ]
    for name in names:
        slug = _slug(name)
        others = tuple(other for other in names if other != name)
        variants += [
            GroupVariant(
                name=f"{ONLY_VARIANT_PREFIX}-{slug}",
                kind=ONLY_GROUP_KIND,
                group=name,
                spectral_regions=(name,),
                use_meta=True,
            ),
            GroupVariant(
                name=f"{ONLY_VARIANT_PREFIX}-{slug}-{NO_META_VARIANT_SUFFIX}",
                kind=ONLY_GROUP_NO_META_KIND,
                group=name,
                spectral_regions=(name,),
                use_meta=False,
            ),
            GroupVariant(
                name=f"{WITHOUT_VARIANT_PREFIX}-{slug}",
                kind=WITHOUT_GROUP_KIND,
                group=name,
                spectral_regions=others,
                use_meta=True,
            ),
        ]
    return variants


def variant_preprocessing(base: PreprocessingConfig, variant: GroupVariant) -> PreprocessingConfig:
    """Apply a variant to the preprocessing of the reference.

    Args:
        base: Preprocessing of the reference model.
        variant: Regions and metadata to use.

    Returns:
        A validated copy of ``base`` with the variant's regions and metadata setting.
    """
    options = {
        **base.model_dump(),
        "spectral_regions": variant.spectral_regions,
        "use_meta": variant.use_meta,
    }
    return PreprocessingConfig.model_validate(options)


def _fold_scores(
    model: BaseEstimator, preprocessing: PreprocessingConfig, fold: Fold
) -> dict[str, float]:
    split = load_split()
    y_fit = split.y_train.iloc[fold.train]
    pipeline = _pipeline(model, preprocessing)
    pipeline.fit(split.X_train.iloc[fold.train], y_fit)
    y_val = split.y_train.iloc[fold.val]
    y_pred = as_target_frame(pipeline.predict(split.X_train.iloc[fold.val]), y_val.index)
    metrics = evaluate(y_val, y_pred, valid_combinations(y_fit))
    return {metric: getattr(metrics, metric) for metric in STUDY_METRICS}


def cv_scores(model: BaseEstimator, preprocessing: PreprocessingConfig) -> CvScores:
    """Score ``model`` on the shared cross-validation folds of the training part.

    Args:
        model: Estimator that predicts ``Crop`` **and** ``Stage``; it is cloned per fold.
        preprocessing: Preprocessing fitted inside every fold.

    Returns:
        All metrics of :func:`awp2.evaluation.evaluate` per fold and aggregated.
    """
    rows = Parallel(n_jobs=GROUP_STUDY_N_JOBS)(
        delayed(_fold_scores)(model, preprocessing, fold) for fold in load_folds()
    )
    folds = pd.DataFrame(rows)
    return CvScores(folds.mean(), folds.std(), folds)


def svm_param_grid() -> dict[str, list[float]]:
    """Hyperparameter grid of the RBF-SVM of the combined approach.

    Returns:
        ``C`` and ``gamma`` values in the naming of ``make_combined_svm()``.
    """
    return {
        GROUP_STUDY_C_PARAM: list(GROUP_STUDY_C_GRID),
        GROUP_STUDY_GAMMA_PARAM: list(GROUP_STUDY_GAMMA_GRID),
    }


def paired_difference(
    first: VariantResult, second: VariantResult, metric: str = COMBINED_METRIC
) -> PairedDifference:
    """Compare two variants fold by fold.

    Args:
        first: Variant on the left of the difference.
        second: Variant on the right of the difference.
        metric: Metric of ``METRIC_LABELS`` to compare.

    Returns:
        Mean and spread of ``first - second`` and in how many folds ``first`` is better.
    """
    difference = first.cv.folds[metric] - second.cv.folds[metric]
    return PairedDifference(
        mean=float(difference.mean()),
        std=float(difference.std()),
        folds_first_better=int((difference > 0).sum()),
        n_folds=len(difference),
    )


def _tune_variant(
    model: BaseEstimator,
    base: PreprocessingConfig,
    variant: GroupVariant,
    param_grid: dict[str, list[float]],
    track: bool,
) -> VariantResult:
    preprocessing = variant_preprocessing(base, variant)
    tuned = tune(
        model,
        TuneConfig(
            name=f"{GROUP_STUDY_RUN_PREFIX}_{variant.name.replace('-', '_')}_search",
            approach=GROUP_STUDY_APPROACH,
            experiment=GROUP_STUDY_EXPERIMENT,
            description=f"Spektralgruppen-Studie: {variant.kind}, Regionen "
            f"{variant.spectral_regions}, Metadaten {variant.use_meta}",
            param_grid=param_grid,
            preprocessing=preprocessing,
            track=track,
        ),
    )
    features = build_preprocessor(preprocessing).fit_transform(load_split().X_train)
    n_bands = sum(bool(re.fullmatch(BAND_PATTERN, column)) for column in features.columns)
    return VariantResult(
        variant=variant,
        n_bands=n_bands,
        n_features=features.shape[1],
        best_params=tuned.best_params,
        tuning_run=tuned.run_id,
        cv=cv_scores(tuned.best_model, preprocessing),
    )


def run_group_study(
    model: BaseEstimator,
    base_preprocessing: PreprocessingConfig,
    param_grid: dict[str, list[float]],
    variants: Sequence[GroupVariant] | None = None,
    track: bool = True,
) -> list[VariantResult]:
    """Tune and cross-validate the model for every variant on the training part.

    Every variant gets its own small hyperparameter search, because a model with few bands needs
    a different kernel width than the full spectrum. Only the cross-validation score decides;
    the validation part of the shared split is never used.

    Args:
        model: Unfitted estimator that predicts ``Crop`` **and** ``Stage``.
        base_preprocessing: Preprocessing of the reference model.
        param_grid: Hyperparameters to search per variant, e.g. :func:`svm_param_grid`.
        variants: Variants to train; all of :func:`group_variants` if omitted.
        track: Record each search in MLflow.

    Returns:
        One result per variant, in the order of ``variants``.
    """
    return [
        _tune_variant(model, base_preprocessing, variant, param_grid, track)
        for variant in (variants if variants is not None else group_variants())
    ]


def results_table(results: Sequence[VariantResult]) -> pd.DataFrame:
    """Summarise the study in one row per variant.

    Args:
        results: Output of :func:`run_group_study`.

    Returns:
        Variant name, kind, region, number of bands and features, chosen ``C`` and ``gamma``,
            and mean and standard deviation of each cross-validated metric.
    """
    rows = []
    for result in results:
        row: dict[str, object] = {
            "variant": result.variant.name,
            "kind": result.variant.kind,
            "group": result.variant.group,
            "use_meta": result.variant.use_meta,
            "n_bands": result.n_bands,
            "n_features": result.n_features,
            "C": result.best_params[GROUP_STUDY_C_PARAM],
            "gamma": result.best_params[GROUP_STUDY_GAMMA_PARAM],
            "tuning_run": result.tuning_run,
        }
        for metric in STUDY_METRICS:
            row[metric] = result.cv.mean[metric]
            row[f"{metric}_std"] = result.cv.std[metric]
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")
