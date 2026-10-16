"""Tests for sequential LSTM components."""

import unittest

import torch

from awp2.models.lstm import (
    _CategoricalContextEncoder,
    _LateFusionLSTMClassifier,
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