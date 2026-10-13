"""Tests for independently trained crop and growth-stage classifiers."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from awp2.config import AEZ_COL, CROP_COL, MONTH_COL, STAGE_COL, TARGET_COLS
from awp2.evaluation import scorer
from awp2.models import IndependentClassifier, make_independent_svm
from awp2.preprocessing import PreprocessingConfig, build_preprocessor


class IndependentClassifierTest(unittest.TestCase):
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

    def test_fit_predict_returns_project_target_frame(self) -> None:
        model = make_independent_svm().fit(self.X[["X427", "X437", "X447"]], self.y)

        predicted = model.predict(self.X[["X427", "X437", "X447"]])

        self.assertEqual(list(predicted.columns), list(TARGET_COLS))
        self.assertTrue(predicted.index.equals(self.X.index))
        self.assertEqual(len(predicted), len(self.X))

    def test_predict_allows_unseen_crop_stage_pairs(self) -> None:
        model = IndependentClassifier(
            DummyClassifier(strategy="constant", constant="rice"),
            DummyClassifier(strategy="constant", constant="Late"),
        ).fit(self.X[["X427", "X437", "X447"]], self.y)

        predicted = model.predict(self.X[["X427", "X437", "X447"]])
        observed_pairs = set(map(tuple, self.y[[CROP_COL, STAGE_COL]].to_numpy()))

        self.assertNotIn(("rice", "Late"), observed_pairs)
        self.assertEqual(set(map(tuple, predicted.to_numpy())), {("rice", "Late")})

    def test_pipeline_preserves_multi_output_prediction_contract(self) -> None:
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(PreprocessingConfig(use_spectral_standard_scale=True)),
                ),
                ("model", make_independent_svm()),
            ]
        )
        pipeline.fit(self.X, self.y)

        predicted = pipeline.predict(self.X)

        self.assertIsInstance(predicted, pd.DataFrame)
        self.assertEqual(list(predicted.columns), list(TARGET_COLS))

    def test_clone_exposes_estimator_hyperparameters(self) -> None:
        cloned = clone(make_independent_svm()).set_params(
            crop_estimator__C=30.0,
            stage_estimator__gamma=0.003,
        )

        self.assertEqual(cloned.crop_estimator.C, 30.0)
        self.assertEqual(cloned.stage_estimator.gamma, 0.003)

    def test_grid_search_scores_multi_output_predictions(self) -> None:
        search = GridSearchCV(
            IndependentClassifier(SVC(class_weight="balanced"), SVC(class_weight="balanced")),
            param_grid={"crop_estimator__C": [1.0], "stage_estimator__C": [1.0]},
            scoring=scorer(),
            cv=2,
            error_score="raise",
        )

        search.fit(self.X[["X427", "X437", "X447"]], self.y)

        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"][0]))

    def test_rejects_wrong_target_columns(self) -> None:
        invalid_targets = self.y.rename(columns={CROP_COL: "crop"})

        with self.assertRaisesRegex(ValueError, "Expected target columns"):
            make_independent_svm().fit(self.X[["X427", "X437", "X447"]], invalid_targets)


if __name__ == "__main__":
    unittest.main()