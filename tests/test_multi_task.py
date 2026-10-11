"""Tests for the tabular multi-task MLP classifier."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from awp2.config import AEZ_COL, CROP_COL, MONTH_COL, STAGE_COL, TARGET_COLS
from awp2.evaluation import scorer
from awp2.models import MultiTaskMLPClassifier
from awp2.preprocessing import PreprocessingConfig, build_preprocessor


class MultiTaskMLPClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        n_rows = 40
        self.X = pd.DataFrame(
            {
                AEZ_COL: np.tile([1, 2], n_rows // 2),
                MONTH_COL: np.tile([3, 9], n_rows // 2),
                "X427": generator.random(n_rows),
                "X437": generator.random(n_rows),
                "X447": generator.random(n_rows),
            }
        )
        self.y = pd.DataFrame(
            {
                CROP_COL: np.tile(["corn", "rice"], n_rows // 2),
                STAGE_COL: np.tile(["Late", "Harvest"], n_rows // 2),
            },
            index=self.X.index,
        )

    @staticmethod
    def _model() -> MultiTaskMLPClassifier:
        return MultiTaskMLPClassifier(
            hidden_units=8,
            hidden_layers=1,
            dropout=0.0,
            max_epochs=3,
            batch_size=4,
        )

    def test_fit_predict_returns_project_target_frame(self) -> None:
        model = self._model().fit(self.X[["X427", "X437", "X447"]], self.y)

        predicted = model.predict(self.X[["X427", "X437", "X447"]])

        self.assertEqual(list(predicted.columns), list(TARGET_COLS))
        self.assertTrue(predicted.index.equals(self.X.index))
        self.assertEqual(len(predicted), len(self.X))

    def test_pipeline_preserves_multi_output_prediction_contract(self) -> None:
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(PreprocessingConfig(use_spectral_standard_scale=True)),
                ),
                ("model", self._model()),
            ]
        )
        pipeline.fit(self.X, self.y)

        predicted = pipeline.predict(self.X)

        self.assertIsInstance(predicted, pd.DataFrame)
        self.assertEqual(list(predicted.columns), list(TARGET_COLS))

    def test_clone_exposes_hyperparameters(self) -> None:
        cloned = clone(self._model()).set_params(hidden_units=16, dropout=0.2, weight_decay=0.0001)

        self.assertEqual(cloned.hidden_units, 16)
        self.assertEqual(cloned.dropout, 0.2)
        self.assertEqual(cloned.weight_decay, 0.0001)

    def test_grid_search_scores_multi_output_predictions(self) -> None:
        search = GridSearchCV(
            self._model(),
            param_grid={"hidden_units": [8]},
            scoring=scorer(),
            cv=2,
            error_score="raise",
        )
        search.fit(self.X[["X427", "X437", "X447"]], self.y)

        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"][0]))

    def test_rejects_wrong_target_columns(self) -> None:
        invalid_targets = self.y.rename(columns={CROP_COL: "crop"})

        with self.assertRaisesRegex(ValueError, "Expected target columns"):
            self._model().fit(self.X[["X427", "X437", "X447"]], invalid_targets)


if __name__ == "__main__":
    unittest.main()