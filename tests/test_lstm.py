"""Tests for sequential LSTM components."""

import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from awp2.config import AEZ_COL, LSTM_EMBEDDINGS_CONTEXT, MONTH_COL
from awp2.models.lstm import (
    _CategoricalContextEncoder,
    _EmbeddingFusionLSTMClassifier,
    _LSTMInputAdapter,
    _LateFusionLSTMClassifier,
    _SpectralLSTMEncoder,
    SpectralLSTMClassifier,
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

        self.assertEqual(tuple(context.shape), (batch_size, aez_embedding_dim + month_embedding_dim))


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