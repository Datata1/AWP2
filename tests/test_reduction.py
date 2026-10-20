"""Tests for the data-reduction study helpers."""

import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from awp2.config import REDUCTION_FAMILY_COL, STD_SUFFIX
from awp2.plots import plot_tradeoff
from awp2.preprocessing import PreprocessingConfig
from awp2.reduction import reduction_variants


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


if __name__ == "__main__":
    unittest.main()
