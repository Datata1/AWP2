"""Tests for the grouped permutation importance."""

import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from awp2.config import AEZ_COL, CROP_COL, META_GROUP, MONTH_COL, STAGE_COL
from awp2.importance import (
    ImportanceConfig,
    band_window_groups,
    permutation_importance,
    region_groups,
)
from awp2.plots import plot_band_importance, plot_group_importance


class _CropFromBand(BaseEstimator):
    """Predicts the crop from the sign of one band and always the same stage."""

    def __init__(self, band: str) -> None:
        self.band = band

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "_CropFromBand":
        self.classes_ = np.array(["corn", "rice"])
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        crops = np.where(X[self.band] > 0, "corn", "rice")
        return pd.DataFrame({CROP_COL: crops, STAGE_COL: "Late"}, index=X.index)


class GroupingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame(
            {
                AEZ_COL: [1, 2],
                MONTH_COL: [3, 4],
                "X427": [1.0, 2.0],
                "X702": [1.0, 2.0],
                "X712": [np.nan, np.nan],
                "X722": [1.0, 2.0],
                "X2203": [1.0, 2.0],
            }
        )

    def test_regions_hold_the_measured_bands_and_meta(self) -> None:
        groups = region_groups(self.X)

        self.assertEqual(groups["VIS"], ["X427"])
        self.assertEqual(groups["Red Edge"], ["X702", "X722"])
        self.assertEqual(groups["SWIR2"], ["X2203"])
        self.assertEqual(groups[META_GROUP], [AEZ_COL, MONTH_COL])
        self.assertNotIn("NIR", groups)

    def test_band_windows_skip_empty_bands(self) -> None:
        groups = band_window_groups(self.X, half_width=1)

        self.assertEqual(groups["X702"], ["X427", "X702", "X722"])
        self.assertEqual(groups["X427"], ["X427", "X702"])
        self.assertNotIn("X712", groups)


class PermutationImportanceTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        signal = np.tile([1.0, -1.0], 20)
        self.X = pd.DataFrame({"X427": signal, "X2203": rng.normal(size=signal.size)})
        self.y = pd.DataFrame({CROP_COL: np.where(signal > 0, "corn", "rice"), STAGE_COL: "Late"})
        self.model = Pipeline([("model", _CropFromBand("X427"))]).fit(self.X, self.y)
        self.result = permutation_importance(
            self.model,
            self.X,
            self.y,
            {"VIS": ["X427"], "SWIR2": ["X2203"]},
            ImportanceConfig(n_repeats=3, metrics=("bacc_crop", "bacc_stage")),
        )

    def _mean(self, group: str, metric: str) -> float:
        table = self.result.to_frame().set_index(["group", "metric"])
        return float(table.loc[(group, metric), "mean"])

    def test_only_the_informative_group_matters_for_the_crop(self) -> None:
        self.assertGreater(self._mean("VIS", "bacc_crop"), 0.3)
        self.assertEqual(self._mean("SWIR2", "bacc_crop"), 0.0)

    def test_baseline_is_the_unpermuted_score(self) -> None:
        self.assertEqual(self.result.baseline["bacc_crop"], 1.0)

    def test_plots_draw_band_curves_and_group_bars(self) -> None:
        curve = plot_band_importance(self.result, metrics=("bacc_crop",))
        bars = plot_group_importance(self.result, metrics=("bacc_crop", "bacc_stage"))

        self.assertEqual(len(curve.get_lines()), 2)  # curve + zero line
        self.assertEqual(len(bars.patches), 4)


if __name__ == "__main__":
    unittest.main()
