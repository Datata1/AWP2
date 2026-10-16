"""Tests for sequential LSTM components."""

import unittest

import torch

from awp2.models.lstm import _SpectralLSTMEncoder


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