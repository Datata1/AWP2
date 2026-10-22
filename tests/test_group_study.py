"""Tests for the spectral group study variants."""

import pandas as pd

from awp2.config import (
    ALL_BANDS_VARIANT,
    ALL_GROUPS_KIND,
    META_ONLY_KIND,
    META_ONLY_VARIANT,
    ONLY_GROUP_KIND,
    ONLY_GROUP_NO_META_KIND,
    SPECTRAL_REGIONS,
    WITHOUT_GROUP_KIND,
)
from awp2.evaluation import METRIC_LABELS
from awp2.group_study import (
    CvScores,
    VariantResult,
    group_variants,
    paired_difference,
    results_table,
    svm_param_grid,
    variant_preprocessing,
)
from awp2.preprocessing import PreprocessingConfig


def _result(name: str, combined_folds: list[float]) -> VariantResult:
    variant = next(v for v in group_variants() if v.name == name)
    folds = pd.DataFrame({metric: combined_folds for metric in METRIC_LABELS})
    best = {key: values[0] for key, values in svm_param_grid().items()}
    return VariantResult(variant, 7, 16, best, None, CvScores(folds.mean(), folds.std(), folds))


def test_variants_cover_every_region_in_three_ways() -> None:
    variants = group_variants()
    kinds = [variant.kind for variant in variants]

    assert kinds.count(ALL_GROUPS_KIND) == 2
    assert kinds.count(META_ONLY_KIND) == 1
    for kind in (ONLY_GROUP_KIND, ONLY_GROUP_NO_META_KIND, WITHOUT_GROUP_KIND):
        assert kinds.count(kind) == len(SPECTRAL_REGIONS)
    assert len({variant.name for variant in variants}) == len(variants)


def test_without_variant_keeps_all_other_regions_and_metadata() -> None:
    variant = next(
        v for v in group_variants() if v.kind == WITHOUT_GROUP_KIND and v.group == "Red Edge"
    )

    assert "Red Edge" not in (variant.spectral_regions or ())
    assert len(variant.spectral_regions or ()) == len(SPECTRAL_REGIONS) - 1
    assert variant.use_meta


def test_variant_preprocessing_keeps_the_reference_options() -> None:
    base = PreprocessingConfig(use_spectral_standard_scale=True)
    variant = next(v for v in group_variants() if v.name == "only-red-edge-no-meta")

    config = variant_preprocessing(base, variant)

    assert config.spectral_regions == ("Red Edge",)
    assert not config.use_meta
    assert config.use_spectral_standard_scale


def test_paired_difference_counts_folds_and_averages_the_gap() -> None:
    first = _result(ALL_BANDS_VARIANT, [0.9, 0.8, 0.7, 0.6, 0.5])
    second = _result(META_ONLY_VARIANT, [0.8, 0.8, 0.8, 0.5, 0.4])

    difference = paired_difference(first, second)

    assert abs(difference.mean - 0.04) < 1e-9
    assert difference.folds_first_better == 3
    assert difference.n_folds == 5


def test_results_table_has_mean_and_std_of_every_metric_per_variant() -> None:
    table = results_table(
        [_result(ALL_BANDS_VARIANT, [0.9, 0.7]), _result(META_ONLY_VARIANT, [0.5, 0.5])]
    )

    assert list(table.index) == [ALL_BANDS_VARIANT, META_ONLY_VARIANT]
    assert abs(table.loc[ALL_BANDS_VARIANT, "bacc_combined"] - 0.8) < 1e-9
    assert table.loc[META_ONLY_VARIANT, "bacc_combined_std"] == 0
    assert all(f"{metric}_std" in table for metric in METRIC_LABELS)
