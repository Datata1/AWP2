"""Tests for the rule-based data-quality flags."""

import unittest

import numpy as np
import pandas as pd

from awp2.config import (
    AEZ_COL,
    ANY_FLAG_COL,
    CLASS_OUTLIER_FLAG,
    CROP_COL,
    DARK_FLAG,
    HAZE_FLAG,
    MONTH_COL,
    NDVI_OUTLIER_FLAG,
    RARE_CONTEXT_FLAG,
    STAGE_COL,
)
from awp2.quality import consensus_label_suspects, flag_shares, quality_flags

WAVELENGTHS_NM = range(430, 1000, 10)


def _vegetation(rows: int, generator: np.random.Generator) -> np.ndarray:
    # low red, high NIR: a healthy, closed canopy
    base = np.array([5.0 if nm < 700 else 40.0 for nm in WAVELENGTHS_NM])
    return base + generator.normal(0, 0.5, (rows, len(base)))


class QualityFlagsTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(0)
        rows = 60
        spectra = _vegetation(rows, generator)
        spectra[0] = 30.0  # bright and flat: haze candidate
        spectra[1] = 2.0  # very dark: shadow or water
        spectra[2, :] = 12.0  # flat soil-like spectrum labelled as closed canopy
        self.X = pd.DataFrame(spectra, columns=[f"X{nm}" for nm in WAVELENGTHS_NM])
        self.X[AEZ_COL] = 5
        self.X[MONTH_COL] = 7
        self.X.loc[3, MONTH_COL] = 5
        self.y = pd.DataFrame({CROP_COL: ["corn"] * rows, STAGE_COL: ["Late"] * rows})
        self.flags = quality_flags(self.X, self.y)

    def test_haze_and_dark_rows_are_flagged(self) -> None:
        self.assertTrue(self.flags.loc[0, HAZE_FLAG])
        self.assertTrue(self.flags.loc[1, DARK_FLAG])
        self.assertFalse(self.flags.loc[10, HAZE_FLAG] or self.flags.loc[10, DARK_FLAG])

    def test_soil_ndvi_in_a_closed_canopy_class_is_flagged(self) -> None:
        self.assertTrue(self.flags.loc[2, NDVI_OUTLIER_FLAG])
        self.assertFalse(self.flags.loc[10, NDVI_OUTLIER_FLAG])

    def test_rare_month_for_the_class_is_flagged(self) -> None:
        self.assertTrue(self.flags.loc[3, RARE_CONTEXT_FLAG])
        self.assertFalse(self.flags.loc[10, RARE_CONTEXT_FLAG])

    def test_class_outliers_and_any_flag(self) -> None:
        self.assertTrue(self.flags.loc[[0, 1, 2], CLASS_OUTLIER_FLAG].all())
        self.assertEqual(
            self.flags[ANY_FLAG_COL].sum(), self.flags.drop(columns=ANY_FLAG_COL).any(axis=1).sum()
        )

    def test_shares_per_group(self) -> None:
        shares = flag_shares(self.flags, self.y[CROP_COL])

        self.assertEqual(shares.loc["corn", "support"], 60)


class ConsensusLabelSuspectsTest(unittest.TestCase):
    def test_flags_rows_where_all_models_agree_on_another_pair(self) -> None:
        y = pd.DataFrame({CROP_COL: ["corn", "corn", "rice"], STAGE_COL: ["Late"] * 3})
        svm = pd.DataFrame({CROP_COL: ["rice", "corn", "corn"], STAGE_COL: ["Late"] * 3})
        trees = pd.DataFrame({CROP_COL: ["rice", "rice", "corn"], STAGE_COL: ["Late"] * 3})

        suspects = consensus_label_suspects([svm, trees], y)

        self.assertEqual(list(suspects), [True, False, True])

    def test_needs_two_models(self) -> None:
        y = pd.DataFrame({CROP_COL: ["corn"], STAGE_COL: ["Late"]})
        with self.assertRaisesRegex(ValueError, "at least 2"):
            consensus_label_suspects([y], y)


if __name__ == "__main__":
    unittest.main()
