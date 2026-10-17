"""Tests for shared preprocessing behavior."""

import unittest

import pandas as pd

from awp2.config import AEZ_COL, MEAN_ALBEDO_COL, MONTH_COL
from awp2.preprocessing import PreprocessingConfig, build_preprocessor


class CategoricalContextPreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2],
                MONTH_COL: [3, 9],
                "X427": [0.1, 0.2],
                "X437": [0.3, 0.4],
            }
        )

    def test_categorical_context_preserves_raw_metadata_columns(self) -> None:
        transformed = build_preprocessor(
            PreprocessingConfig(use_categorical_context=True)
        ).fit_transform(self.features)

        self.assertEqual(list(transformed.columns), [AEZ_COL, MONTH_COL, "X427", "X437"])
        pd.testing.assert_frame_equal(
            transformed[[AEZ_COL, MONTH_COL]], self.features[[AEZ_COL, MONTH_COL]]
        )


class MeanAlbedoScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2, 3],
                MONTH_COL: [5, 7, 9],
                "X427": [10.0, 20.0, 45.0],
                "X437": [12.0, 25.0, 50.0],
            }
        )

    def _albedo(self, config: PreprocessingConfig) -> pd.Series:
        return build_preprocessor(config).fit_transform(self.features)[MEAN_ALBEDO_COL]

    def test_albedo_follows_the_standard_scaling_of_the_spectra(self) -> None:
        albedo = self._albedo(
            PreprocessingConfig(use_mean_albedo=True, use_spectral_standard_scale=True)
        )

        self.assertAlmostEqual(albedo.mean(), 0.0)
        self.assertAlmostEqual(albedo.std(ddof=0), 1.0)

    def test_albedo_follows_the_minmax_scaling_of_the_spectra(self) -> None:
        albedo = self._albedo(
            PreprocessingConfig(use_mean_albedo=True, use_spectral_minmax_scale=True)
        )

        self.assertAlmostEqual(albedo.min(), 0.0)
        self.assertAlmostEqual(albedo.max(), 1.0)

    def test_albedo_stays_in_reflectance_units_without_spectral_scaling(self) -> None:
        albedo = self._albedo(PreprocessingConfig(use_mean_albedo=True))

        pd.testing.assert_series_equal(
            albedo, self.features[["X427", "X437"]].mean(axis=1), check_names=False
        )
