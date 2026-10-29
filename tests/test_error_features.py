"""Tests for the features and models that target the main confusions (dry matter, soil, slope)."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from awp2.config import (
    AEZ_COL,
    CROP_COL,
    DRY_MATTER_INDEX_NAMES,
    MONTH_COL,
    SOIL_INDEX_NAMES,
    STAGE_COL,
)
from awp2.models.ensemble import make_combined_stacking
from awp2.models.specialist import PairSpecialistClassifier, specialist_upper_bound
from awp2.preprocessing import PreprocessingConfig, build_preprocessor, describe_preprocessor
from awp2.preprocessing.bands import FirstDerivative
from awp2.preprocessing.indices import DryMatterIndices, SoilIndices


class FirstDerivativeTest(unittest.TestCase):
    def test_slope_per_nm_within_each_segment(self) -> None:
        # X600 sits behind a gap of deleted bands and forms its own one-band segment
        bands = pd.DataFrame({"X400": [1.0], "X410": [3.0], "X420": [5.0], "X600": [9.0]})

        slopes = FirstDerivative().fit_transform(bands)

        self.assertEqual(list(slopes.columns), ["dX400", "dX410", "dX420", "dX600"])
        np.testing.assert_allclose(slopes.iloc[0], [0.2, 0.2, 0.2, 0.0])

    def test_ignores_a_constant_offset(self) -> None:
        bands = pd.DataFrame({"X400": [1.0, 11.0], "X410": [2.0, 12.0], "X420": [4.0, 14.0]})

        slopes = FirstDerivative().fit_transform(bands)

        np.testing.assert_allclose(slopes.iloc[0], slopes.iloc[1])


class DryMatterIndicesTest(unittest.TestCase):
    def test_cai_is_the_absorption_depth_at_the_centre(self) -> None:
        bands = pd.DataFrame(
            {"X1649": [30.0], "X2063": [20.0], "X2123": [15.0], "X2184": [24.0], "X2214": [10.0]}
        )

        indices = DryMatterIndices().fit_transform(bands)

        cai, ndti = DRY_MATTER_INDEX_NAMES
        self.assertAlmostEqual(indices.loc[0, cai], (0.20 + 0.24) / 2 - 0.15)
        self.assertAlmostEqual(indices.loc[0, ndti], (0.30 - 0.10) / (0.30 + 0.10))


class SoilIndicesTest(unittest.TestCase):
    def test_bare_soil_lies_on_the_fitted_soil_line(self) -> None:
        red = np.linspace(5.0, 30.0, 40)
        soil_nir = 1.2 * red + 2.0
        # the five reddest-to-NIR rows are bare soil; the rest has vegetation on top
        nir = np.where(np.arange(40) < 2, soil_nir, soil_nir + 25.0)
        bands = pd.DataFrame({"X671": red, "X854": nir})

        indices = SoilIndices().fit_transform(bands)

        self.assertEqual(list(indices.columns), list(SOIL_INDEX_NAMES))
        np.testing.assert_allclose(indices["PVI"].iloc[:2], 0.0, atol=1e-9)
        self.assertTrue((indices["PVI"].iloc[2:] > 0).all())


class NewFeatureGroupsTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = 6
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2] * 3,
                MONTH_COL: [5, 7, 9] * 2,
                **{
                    f"X{nm}": np.linspace(10.0, 40.0, rows) + nm / 100
                    for nm in (671, 854, 1649, 2063, 2123, 2184, 2214)
                },
            }
        )

    def test_each_option_adds_its_columns(self) -> None:
        config = PreprocessingConfig(
            use_dry_matter_indices=True, use_soil_indices=True, use_spectral_derivative=True
        )

        columns = build_preprocessor(config).fit_transform(self.features).columns

        for name in (*DRY_MATTER_INDEX_NAMES, *SOIL_INDEX_NAMES, "dX854", "X854"):
            self.assertIn(name, columns)

    def test_every_new_branch_is_described(self) -> None:
        config = PreprocessingConfig(
            use_dry_matter_indices=True, use_soil_indices=True, use_spectral_derivative=True
        )

        lines = describe_preprocessor(build_preprocessor(config))

        self.assertTrue(any(line.startswith(("H", "I", "J")) for line in lines))

    def test_derivative_can_replace_the_spectra(self) -> None:
        config = PreprocessingConfig(use_spectra=False, use_spectral_derivative=True)

        columns = build_preprocessor(config).fit_transform(self.features).columns

        self.assertIn("dX854", columns)
        self.assertNotIn("X854", columns)

    def test_smoothing_may_feed_the_derivative_without_spectra(self) -> None:
        config = PreprocessingConfig(
            use_spectra=False,
            use_spectral_derivative=True,
            use_savgol=True,
            savgol_window_length=3,
            savgol_polyorder=1,
        )

        columns = build_preprocessor(config).fit_transform(self.features).columns

        self.assertIn("dX854", columns)


class _AlwaysEarlyCorn(ClassifierMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "_AlwaysEarlyCorn":
        self.classes_ = np.array(["corn|Early_Mid"])
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({CROP_COL: "corn", STAGE_COL: "Early_Mid"}, index=X.index)


class PairSpecialistTest(unittest.TestCase):
    def setUp(self) -> None:
        signal = np.repeat([0.0, 1.0], 10)
        self.X = pd.DataFrame({"x": signal})
        self.y = pd.DataFrame(
            {CROP_COL: np.where(signal == 0, "corn", "soybean"), STAGE_COL: "Early_Mid"}
        )

    def test_specialist_re_decides_the_routed_rows(self) -> None:
        model = PairSpecialistClassifier(_AlwaysEarlyCorn(), KNeighborsClassifier(1))

        predicted = model.fit(self.X, self.y).predict(self.X)

        pd.testing.assert_frame_equal(predicted, self.y)

    def test_upper_bound_fixes_only_rows_inside_the_group(self) -> None:
        y_true = pd.DataFrame({CROP_COL: ["soybean", "cotton"], STAGE_COL: ["Early_Mid", "Late"]})
        y_pred = pd.DataFrame({CROP_COL: ["corn", "corn"], STAGE_COL: ["Early_Mid", "Early_Mid"]})

        corrected = specialist_upper_bound(y_true, y_pred)

        self.assertEqual(list(corrected[CROP_COL]), ["soybean", "corn"])


class StackingTest(unittest.TestCase):
    def test_fits_and_predicts_valid_pairs(self) -> None:
        generator = np.random.default_rng(0)
        signal = np.repeat([0.0, 1.0], 20)
        X = pd.DataFrame(
            {"a": signal + generator.normal(0, 0.1, 40), "b": generator.normal(size=40)}
        )
        y = pd.DataFrame({CROP_COL: np.where(signal == 0, "corn", "rice"), STAGE_COL: "Late"})

        predicted = make_combined_stacking().fit(X, y).predict(X)

        self.assertTrue(set(predicted[CROP_COL]) <= {"corn", "rice"})
        self.assertGreater((predicted[CROP_COL] == y[CROP_COL]).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
