"""Tests for the data-reduction study helpers."""

import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from awp2.config import (
    FIT_TIME_COL,
    N_COMPONENTS_COL,
    N_INPUT_FEATURES_COL,
    PREDICT_TIME_COL,
    REDUCTION_FAMILY_COL,
    REDUCTION_VARIANT_COL,
    STD_SUFFIX,
)
from awp2.experiment import TuneConfig
from awp2.models import make_combined_svm
from awp2.plots import plot_fold_timings, plot_pca_curve, plot_pca_loadings, plot_tradeoff
from awp2.preprocessing import PreprocessingConfig
from awp2.reduction import (
    ReductionVariant,
    explained_variance_at,
    pca_curve,
    pca_recommendation,
    reduction_variants,
    smallest_equivalent,
    spectral_pca,
)


class ReductionVariantsTest(unittest.TestCase):
    def test_variants_are_valid_and_uniquely_named(self) -> None:
        variants = reduction_variants(PreprocessingConfig(use_spectral_standard_scale=True))
        names = [variant.name for variant in variants]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(variants[0].family, "all_bands")
        self.assertIn("meta_only", {variant.family for variant in variants})

    def test_base_without_scaling_rejects_pca(self) -> None:
        with self.assertRaisesRegex(ValueError, "pca"):
            reduction_variants(PreprocessingConfig())


class TradeoffPlotTest(unittest.TestCase):
    def test_one_line_per_family(self) -> None:
        curve = pd.DataFrame(
            {
                REDUCTION_FAMILY_COL: ["all_bands", "pca", "pca", "meta_only"],
                "n_measured_bands": [131, 131, 131, 0],
                "bacc_combined": [0.85, 0.7, 0.8, 0.4],
                f"bacc_combined{STD_SUFFIX}": [0.02, 0.03, 0.02, 0.05],
            }
        )

        ax = plot_tradeoff(curve)

        self.assertEqual(len(ax.containers), 3)


def _pca_curve_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            REDUCTION_VARIANT_COL: ["all_bands", "pca_2", "pca_10", "pca_20"],
            N_INPUT_FEATURES_COL: [140, 11, 19, 29],
            N_COMPONENTS_COL: [131, 2, 10, 20],
            "bacc_combined": [0.86, 0.76, 0.851, 0.855],
            f"bacc_combined{STD_SUFFIX}": [0.01, 0.02, 0.01, 0.01],
            FIT_TIME_COL: [1.0, 0.2, 0.3, 0.4],
            PREDICT_TIME_COL: [0.2, 0.1, 0.1, 0.1],
        }
    )


class SmallestEquivalentTest(unittest.TestCase):
    def test_picks_fewest_features_within_the_loss(self) -> None:
        chosen = smallest_equivalent(_pca_curve_frame(), "all_bands")

        assert chosen is not None
        self.assertEqual(chosen.name, "pca_10")

    def test_none_if_every_variant_loses_too_much(self) -> None:
        self.assertIsNone(smallest_equivalent(_pca_curve_frame(), "all_bands", max_loss=0.001))


class PcaRecommendationTest(unittest.TestCase):
    def test_knee_that_is_much_faster_is_recommended(self) -> None:
        decision = pca_recommendation(_pca_curve_frame())

        self.assertEqual(decision.knee, "pca_10")
        self.assertAlmostEqual(decision.speedup, 1.2 / 0.4)
        self.assertTrue(decision.recommended)

    def test_knee_without_gain_or_speedup_is_not_recommended(self) -> None:
        curve = _pca_curve_frame().assign(**{FIT_TIME_COL: 1.0, PREDICT_TIME_COL: 0.2})

        self.assertFalse(pca_recommendation(curve).recommended)


class ExplainedVarianceTest(unittest.TestCase):
    def test_reference_keeps_everything(self) -> None:
        ratio = pd.Series([0.5] + [0.5 / 30] * 30)

        shares = explained_variance_at(_pca_curve_frame(), ratio)

        self.assertEqual(shares.loc[131], 1.0)
        self.assertAlmostEqual(shares.loc[2], 0.5 + 0.5 / 30)


class SpectralPcaTest(unittest.TestCase):
    def test_rejects_pca_in_the_preprocessing(self) -> None:
        with self.assertRaisesRegex(ValueError, "without PCA"):
            spectral_pca(
                PreprocessingConfig(use_spectral_standard_scale=True, spectral_pca_components=3)
            )


class PcaCurveTest(unittest.TestCase):
    def test_rejects_a_reference_with_pca(self) -> None:
        reference = ReductionVariant(
            name="pca",
            family="pca",
            preprocessing=PreprocessingConfig(
                use_spectral_standard_scale=True, spectral_pca_components=5
            ),
        )
        template = TuneConfig(
            name="pca", approach="combined", param_grid={"estimator__C": [1.0]}, track=False
        )

        with self.assertRaisesRegex(ValueError, "PCA"):
            pca_curve(make_combined_svm(), reference, [2], template)


class PcaPlotsTest(unittest.TestCase):
    def test_curve_with_variance_axis_and_relative_scale(self) -> None:
        variance = pd.Series([0.6, 0.3, 0.1], index=["PC1", "PC2", "PC3"])

        ax = plot_pca_curve({"SVM": _pca_curve_frame()}, variance, relative=True)

        self.assertEqual(len(ax.figure.axes), 2)
        self.assertAlmostEqual(ax.lines[0].get_ydata()[0], 0.76 - 0.86)

    def test_relative_curves_need_the_same_reference_size(self) -> None:
        sentinel2 = _pca_curve_frame().assign(**{N_COMPONENTS_COL: [10, 2, 5, 8]})

        with self.assertRaisesRegex(ValueError, "same number of bands"):
            plot_pca_curve({"a": _pca_curve_frame(), "b": sentinel2}, relative=True)

    def test_timings_have_two_panels(self) -> None:
        fig = plot_fold_timings({"SVM": _pca_curve_frame()})

        self.assertEqual(len(fig.axes), 2)

    def test_loadings_break_at_band_gaps(self) -> None:
        loadings = pd.DataFrame(
            {"PC1": [0.1, 0.2, 0.3], "PC2": [0.3, 0.2, 0.1]}, index=["X427", "X437", "X1500"]
        )
        variance = pd.Series([0.7, 0.3], index=["PC1", "PC2"])

        ax = plot_pca_loadings(loadings, variance, n_components=2)

        self.assertEqual(len(ax.lines[0].get_xdata()), 4)


if __name__ == "__main__":
    unittest.main()
