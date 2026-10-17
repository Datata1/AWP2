"""Tests for train-only spectral jitter around the combined SVM reference."""

import unittest

import numpy as np
import pandas as pd

from awp2.config import CROP_COL, STAGE_COL, TARGET_COLS
from awp2.experiment import TuneConfig
from awp2.models import SpectralJitterSVC, make_combined_spectral_jitter_svm


class SpectralJitterSVCTest(unittest.TestCase):
    @staticmethod
    def _features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
        features = pd.DataFrame(
            {
                "AEZ_1": [1.0, 0.0, 1.0, 0.0],
                "Month_sin": [0.5, -0.5, 0.5, -0.5],
                "X427": [0.1, 0.2, 0.3, 0.4],
                "X437": [0.4, 0.3, 0.2, 0.1],
            }
        )
        labels = pd.Series(["corn|Late", "rice|Harvest", "corn|Late", "rice|Harvest"])
        return features, labels

    def test_jitter_perturbs_only_spectral_bands(self) -> None:
        features, _labels = self._features_and_labels()
        classifier = SpectralJitterSVC(augmentation_multiplier=1, noise_std=0.1)
        classifier.band_columns_ = ["X427", "X437"]

        augmented = classifier._augment_features(features)

        self.assertEqual(len(augmented), len(features) * 2)
        pd.testing.assert_frame_equal(
            augmented.loc[len(features) :, ["AEZ_1", "Month_sin"]].reset_index(drop=True),
            features[["AEZ_1", "Month_sin"]],
        )
        self.assertFalse(
            np.array_equal(
                augmented.loc[len(features) :, classifier.band_columns_].to_numpy(),
                features[classifier.band_columns_].to_numpy(),
            )
        )

    def test_fit_augments_training_rows_and_predicts_unmodified_features(self) -> None:
        features, labels = self._features_and_labels()
        classifier = SpectralJitterSVC(augmentation_multiplier=1, noise_std=0.01)

        predicted = classifier.fit(features, labels).predict(features)

        self.assertEqual(classifier.n_training_samples_, len(features) * 2)
        self.assertEqual(predicted.shape, (len(features),))
        self.assertTrue(set(predicted).issubset(set(labels)))

    def test_combined_factory_uses_the_jitter_svm(self) -> None:
        features, labels = self._features_and_labels()
        targets = pd.DataFrame(
            [label.split("|") for label in labels],
            columns=list(TARGET_COLS),
            index=features.index,
        )
        classifier = make_combined_spectral_jitter_svm().set_params(
            estimator__augmentation_multiplier=1,
            estimator__noise_std=0.01,
        )

        predicted = classifier.fit(features, targets).predict(features)

        self.assertEqual(list(predicted.columns), [CROP_COL, STAGE_COL])
        self.assertTrue(predicted.index.equals(features.index))

    def test_tune_config_accepts_conditional_jitter_grids(self) -> None:
        config = TuneConfig(
            name="spectral_jitter_svm_search",
            approach="combined",
            param_grid=[
                {"estimator__augmentation_multiplier": [0]},
                {
                    "estimator__augmentation_multiplier": [1],
                    "estimator__noise_std": [0.01, 0.03],
                },
            ],
        )

        self.assertEqual(len(config.param_grid), 2)