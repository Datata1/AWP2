"""Tests for sequential LSTM components."""

import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from awp2.config import (
    AEZ_COL,
    CROP_COL,
    LSTM_EMBEDDINGS_CONTEXT,
    LSTM_STANDARD_CONTEXT,
    MONTH_COL,
    STAGE_COL,
    TARGET_COLS,
)
from awp2.models.combined import CombinedLabelClassifier, make_combined_spectral_lstm
from awp2.models.hierarchical import HierarchicalClassifier, make_hierarchical_spectral_lstm
from awp2.models.lstm import (
    SpectralLSTMClassifier,
    _CategoricalContextEncoder,
    _EmbeddingFusionLSTMClassifier,
    _LateFusionLSTMClassifier,
    _LSTMInputAdapter,
    _SpectralLSTMEncoder,
)


class SpectralLSTMEncoderTest(unittest.TestCase):
    def test_returns_one_representation_per_spectrum(self) -> None:
        batch_size = 3
        n_bands = 4
        hidden_size = 5
        encoder = _SpectralLSTMEncoder(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
        )

        representation = encoder(torch.rand(batch_size, n_bands, 1))

        self.assertEqual(tuple(representation.shape), (batch_size, hidden_size))


class CategoricalContextEncoderTest(unittest.TestCase):
    def test_concatenates_one_embedding_per_context_category(self) -> None:
        batch_size = 3
        aez_embedding_dim = 4
        month_embedding_dim = 3
        encoder = _CategoricalContextEncoder(
            n_aez_categories=5,
            n_month_categories=4,
            aez_embedding_dim=aez_embedding_dim,
            month_embedding_dim=month_embedding_dim,
        )

        context = encoder(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))

        self.assertEqual(
            tuple(context.shape),
            (batch_size, aez_embedding_dim + month_embedding_dim),
        )


class LateFusionLSTMClassifierTest(unittest.TestCase):
    def test_fuses_static_context_after_spectral_encoding(self) -> None:
        batch_size = 3
        n_bands = 4
        hidden_size = 5
        n_context_features = 2
        n_classes = 6
        classifier = _LateFusionLSTMClassifier(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            n_context_features=n_context_features,
            n_classes=n_classes,
        )

        logits = classifier(
            torch.rand(batch_size, n_bands, 1), torch.rand(batch_size, n_context_features)
        )

        self.assertEqual(classifier.classifier.in_features, hidden_size + n_context_features)
        self.assertEqual(tuple(logits.shape), (batch_size, n_classes))


class EmbeddingFusionLSTMClassifierTest(unittest.TestCase):
    def test_fuses_context_embeddings_after_spectral_encoding(self) -> None:
        batch_size = 3
        n_bands = 4
        hidden_size = 5
        aez_embedding_dim = 4
        month_embedding_dim = 3
        n_classes = 6
        classifier = _EmbeddingFusionLSTMClassifier(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            n_aez_categories=5,
            n_month_categories=4,
            aez_embedding_dim=aez_embedding_dim,
            month_embedding_dim=month_embedding_dim,
            n_classes=n_classes,
        )

        logits = classifier(
            torch.rand(batch_size, n_bands, 1),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
        )

        self.assertEqual(
            classifier.classifier.in_features,
            hidden_size + aez_embedding_dim + month_embedding_dim,
        )
        self.assertEqual(tuple(logits.shape), (batch_size, n_classes))


class LSTMInputAdapterTest(unittest.TestCase):
    def test_separates_spectrum_and_categorical_context_codes(self) -> None:
        features = pd.DataFrame(
            {
                AEZ_COL: [1, 2, 3],
                MONTH_COL: [3, 9, 12],
                "X427": [0.1, 0.2, 0.3],
                "X437": [0.4, 0.5, 0.6],
            }
        )
        adapter = _LSTMInputAdapter(LSTM_EMBEDDINGS_CONTEXT).fit(features)

        inputs = adapter.transform(features)

        self.assertEqual(tuple(inputs["spectrum"].shape), (3, 2, 1))
        self.assertEqual(inputs["aez_codes"].tolist(), [1, 2, 3])
        self.assertEqual(inputs["month_codes"].tolist(), [1, 2, 3])


class SpectralLSTMClassifierTest(unittest.TestCase):
    @staticmethod
    def _paired_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        n_repeats = 10
        pairs = [
            ("corn", "Late"),
            ("corn", "Harvest"),
            ("rice", "Late"),
            ("rice", "Harvest"),
        ]
        n_rows = n_repeats * len(pairs)
        generator = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                "X427": generator.random(n_rows),
                "X437": generator.random(n_rows),
                "X447": generator.random(n_rows),
            }
        )
        targets = pd.DataFrame(
            np.tile(pairs, (n_repeats, 1)),
            columns=list(TARGET_COLS),
            index=features.index,
        )
        return features, targets

    def test_combined_lstm_fits_and_returns_target_frame(self) -> None:
        features, targets = self._paired_training_data()
        classifier = make_combined_spectral_lstm("none").set_params(
            estimator__hidden_size=4,
            estimator__num_layers=1,
            estimator__dropout=0.0,
            estimator__max_epochs=1,
            estimator__batch_size=4,
        )

        predicted = classifier.fit(features, targets).predict(features)

        self.assertEqual(list(predicted.columns), list(TARGET_COLS))
        self.assertTrue(predicted.index.equals(features.index))
        self.assertTrue(set(map(tuple, predicted.to_numpy())).issubset(set(map(tuple, targets.to_numpy()))))

    def test_hierarchical_lstm_fits_and_returns_target_frame(self) -> None:
        features, targets = self._paired_training_data()
        classifier = make_hierarchical_spectral_lstm("none").set_params(
            crop_estimator__hidden_size=4,
            crop_estimator__num_layers=1,
            crop_estimator__dropout=0.0,
            crop_estimator__max_epochs=1,
            crop_estimator__batch_size=4,
            stage_estimator__hidden_size=4,
            stage_estimator__num_layers=1,
            stage_estimator__dropout=0.0,
            stage_estimator__max_epochs=1,
            stage_estimator__batch_size=4,
        )

        predicted = classifier.fit(features, targets).predict(features)

        self.assertEqual(list(predicted.columns), list(TARGET_COLS))
        self.assertTrue(predicted.index.equals(features.index))
        self.assertTrue(set(map(tuple, predicted.to_numpy())).issubset(set(map(tuple, targets.to_numpy()))))

    def test_hierarchical_factory_configures_independent_lstm_prototypes(self) -> None:
        classifier = make_hierarchical_spectral_lstm(LSTM_EMBEDDINGS_CONTEXT)

        self.assertIsInstance(classifier, HierarchicalClassifier)
        self.assertIsInstance(classifier.crop_estimator, SpectralLSTMClassifier)
        self.assertIsInstance(classifier.stage_estimator, SpectralLSTMClassifier)
        self.assertIsNot(classifier.crop_estimator, classifier.stage_estimator)
        self.assertEqual(classifier.crop_estimator.context_mode, LSTM_EMBEDDINGS_CONTEXT)

    def test_combined_factory_wraps_lstm_with_requested_context_mode(self) -> None:
        classifier = make_combined_spectral_lstm(LSTM_EMBEDDINGS_CONTEXT)

        self.assertIsInstance(classifier, CombinedLabelClassifier)
        self.assertIsInstance(classifier.estimator, SpectralLSTMClassifier)
        self.assertEqual(classifier.estimator.context_mode, LSTM_EMBEDDINGS_CONTEXT)

    def test_clone_preserves_configured_hyperparameters(self) -> None:
        classifier = SpectralLSTMClassifier(hidden_size=8, context_mode=LSTM_EMBEDDINGS_CONTEXT)

        cloned = clone(classifier)

        self.assertEqual(cloned.hidden_size, 8)
        self.assertEqual(cloned.context_mode, LSTM_EMBEDDINGS_CONTEXT)

    def test_rejects_invalid_dropout(self) -> None:
        classifier = SpectralLSTMClassifier(dropout=1.0)

        with self.assertRaisesRegex(ValueError, "dropout"):
            classifier._validate_hyperparameters()

    def test_builds_embedding_module_for_embedding_context(self) -> None:
        features = pd.DataFrame(
            {
                AEZ_COL: [1, 2, 3],
                MONTH_COL: [3, 9, 12],
                "X427": [0.1, 0.2, 0.3],
                "X437": [0.4, 0.5, 0.6],
            }
        )
        adapter = _LSTMInputAdapter(LSTM_EMBEDDINGS_CONTEXT).fit(features)

        module = SpectralLSTMClassifier(context_mode=LSTM_EMBEDDINGS_CONTEXT)._build_module(
            adapter, n_classes=2
        )

        self.assertIsInstance(module, _EmbeddingFusionLSTMClassifier)

    def test_fit_predict_returns_one_label_per_spectrum(self) -> None:
        n_rows = 20
        generator = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                "X427": generator.random(n_rows),
                "X437": generator.random(n_rows),
                "X447": generator.random(n_rows),
            }
        )
        targets = pd.Series(np.tile(["first", "second"], n_rows // 2), index=features.index)
        classifier = SpectralLSTMClassifier(
            context_mode="none",
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            max_epochs=2,
            batch_size=4,
        )

        predicted = classifier.fit(features, targets).predict(features)

        self.assertEqual(predicted.shape, (n_rows,))
        self.assertTrue(set(predicted).issubset(set(targets)))

    def test_fit_predict_supports_standard_context(self) -> None:
        n_rows = 20
        generator = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                "AEZ_1": np.tile([1.0, 0.0], n_rows // 2),
                "Month_sin": np.tile([0.5, -0.5], n_rows // 2),
                "Month_cos": np.tile([0.5, -0.5], n_rows // 2),
                "X427": generator.random(n_rows),
                "X437": generator.random(n_rows),
            }
        )
        targets = pd.Series(np.tile(["first", "second"], n_rows // 2), index=features.index)
        classifier = SpectralLSTMClassifier(
            context_mode=LSTM_STANDARD_CONTEXT,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            max_epochs=1,
            batch_size=4,
        )

        predicted = classifier.fit(features, targets).predict(features)

        self.assertEqual(predicted.shape, (n_rows,))
        self.assertTrue(set(predicted).issubset(set(targets)))

    def test_fit_predict_supports_context_embeddings(self) -> None:
        n_rows = 20
        generator = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                AEZ_COL: np.tile([1, 2], n_rows // 2),
                MONTH_COL: np.tile([3, 9], n_rows // 2),
                "X427": generator.random(n_rows),
                "X437": generator.random(n_rows),
            }
        )
        targets = pd.Series(np.tile(["first", "second"], n_rows // 2), index=features.index)
        classifier = SpectralLSTMClassifier(
            context_mode=LSTM_EMBEDDINGS_CONTEXT,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            max_epochs=1,
            batch_size=4,
        )

        predicted = classifier.fit(features, targets).predict(features)

        self.assertEqual(predicted.shape, (n_rows,))
        self.assertTrue(set(predicted).issubset(set(targets)))