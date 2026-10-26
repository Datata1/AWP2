"""Tests for classifiers that select their own bands."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from awp2.config import AEZ_COL, CROP_COL, STAGE_COL
from awp2.models.reduced import BandReducedClassifier, make_hierarchical_reduced_svm


class BandReducedClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        n_rows = 40
        generator = np.random.default_rng(0)
        crop_signal = np.repeat([0.0, 1.0], n_rows // 2)
        stage_signal = np.tile([0.0, 1.0], n_rows // 2)
        self.X = pd.DataFrame(
            {
                "X427": crop_signal + generator.normal(0, 0.05, n_rows),
                "X437": generator.normal(size=n_rows),
                "X447": stage_signal + generator.normal(0, 0.05, n_rows),
                f"{AEZ_COL}_1": generator.integers(0, 2, n_rows).astype(float),
            }
        )
        self.y = pd.DataFrame(
            {
                CROP_COL: np.where(crop_signal == 0, "corn", "rice"),
                STAGE_COL: np.where(stage_signal == 0, "Early_Mid", "Late"),
            }
        )

    def test_keeps_selected_bands_and_context_columns(self) -> None:
        model = BandReducedClassifier(LogisticRegression(), k=1).fit(self.X, self.y[CROP_COL])

        self.assertEqual(model.selected_wavelengths_, [427])
        self.assertEqual(list(model._reduce(self.X).columns), ["X427", f"{AEZ_COL}_1"])

    def test_each_hierarchy_level_selects_its_own_bands(self) -> None:
        hierarchy = make_hierarchical_reduced_svm(k=1).fit(self.X, self.y)

        self.assertEqual(hierarchy.crop_estimator_.selected_wavelengths_, [427])
        for stage_model in hierarchy.stage_estimators_.values():
            self.assertEqual(stage_model.selected_wavelengths_, [447])

    def test_passes_sample_weight_to_the_classifier(self) -> None:
        weights = np.ones(len(self.X))

        model = BandReducedClassifier(SVC(), k=2).fit(self.X, self.y[CROP_COL], weights)

        self.assertEqual(len(model.predict(self.X)), len(self.X))

    def test_predict_proba_only_with_a_probabilistic_classifier(self) -> None:
        self.assertTrue(hasattr(BandReducedClassifier(LogisticRegression()), "predict_proba"))
        self.assertFalse(hasattr(BandReducedClassifier(SVC()), "predict_proba"))

    def test_clone_keeps_settings(self) -> None:
        cloned = clone(BandReducedClassifier(SVC(C=3.0), k=7, band_score="mutual_info"))

        settings = (cloned.k, cloned.band_score, cloned.classifier.C)
        self.assertEqual(settings, (7, "mutual_info", 3.0))


if __name__ == "__main__":
    unittest.main()
