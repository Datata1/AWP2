"""Sequential LSTM components for hyperspectral signatures."""

from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import Tensor, nn

from awp2.config import (
    AEZ_COL,
    LSTM_DEFAULT_AEZ_EMBEDDING_DIM,
    LSTM_DEFAULT_BATCH_SIZE,
    LSTM_DEFAULT_DROPOUT,
    LSTM_DEFAULT_HIDDEN_SIZE,
    LSTM_DEFAULT_LEARNING_RATE,
    LSTM_DEFAULT_MAX_EPOCHS,
    LSTM_DEFAULT_MONTH_EMBEDDING_DIM,
    LSTM_DEFAULT_NUM_LAYERS,
    LSTM_DEFAULT_WEIGHT_DECAY,
    LSTM_EMBEDDINGS_CONTEXT,
    LSTM_NO_CONTEXT,
    LSTM_STANDARD_CONTEXT,
    LSTMContextMode,
    MONTH_COL,
    SEED,
)
from awp2.data import band_columns


class _SpectralLSTMEncoder(nn.Module):
    """Encode an ordered spectral sequence into one fixed-size representation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, spectrum: Tensor) -> Tensor:
        """Return the final hidden representation for each spectral sequence."""
        _outputs, (hidden_state, _cell_state) = self.lstm(spectrum)
        return hidden_state[-1]


class _CategoricalContextEncoder(nn.Module):
    """Encode AEZ and Month categories as one learned context representation."""

    def __init__(
        self,
        n_aez_categories: int,
        n_month_categories: int,
        aez_embedding_dim: int,
        month_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.aez_embedding = nn.Embedding(n_aez_categories, aez_embedding_dim)
        self.month_embedding = nn.Embedding(n_month_categories, month_embedding_dim)

    def forward(self, aez_codes: Tensor, month_codes: Tensor) -> Tensor:
        """Return concatenated AEZ and Month embeddings per sample."""
        return torch.cat((self.aez_embedding(aez_codes), self.month_embedding(month_codes)), dim=1)


class _LateFusionLSTMClassifier(nn.Module):
    """Predict one target from an LSTM spectrum representation and static context."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_context_features: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.spectral_encoder = _SpectralLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size + n_context_features, n_classes)

    def forward(self, spectrum: Tensor, context: Tensor) -> Tensor:
        """Return class logits after fusing the encoded spectrum and context."""
        features = torch.cat((self.spectral_encoder(spectrum), context), dim=1)
        return self.classifier(features)


class _EmbeddingFusionLSTMClassifier(nn.Module):
    """Predict one target from an LSTM spectrum representation and learned context."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_aez_categories: int,
        n_month_categories: int,
        aez_embedding_dim: int,
        month_embedding_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.spectral_encoder = _SpectralLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.context_encoder = _CategoricalContextEncoder(
            n_aez_categories=n_aez_categories,
            n_month_categories=n_month_categories,
            aez_embedding_dim=aez_embedding_dim,
            month_embedding_dim=month_embedding_dim,
        )
        self.classifier = nn.Linear(
            hidden_size + aez_embedding_dim + month_embedding_dim, n_classes
        )

    def forward(self, spectrum: Tensor, aez_codes: Tensor, month_codes: Tensor) -> Tensor:
        """Return class logits after fusing the encoded spectrum and context embeddings."""
        context = self.context_encoder(aez_codes, month_codes)
        features = torch.cat((self.spectral_encoder(spectrum), context), dim=1)
        return self.classifier(features)


class _LSTMInputAdapter:
    """Separate spectral sequences from the configured static context representation."""

    def __init__(self, context_mode: LSTMContextMode) -> None:
        self.context_mode = context_mode

    def fit(self, X: pd.DataFrame) -> Self:
        """Learn the fitted input columns and categorical context codes."""
        self._validate_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.band_columns_ = band_columns(X)
        if not self.band_columns_:
            raise ValueError("X must contain at least one spectral band")
        self.context_columns_ = [column for column in X.columns if column not in self.band_columns_]
        if self.context_mode == LSTM_EMBEDDINGS_CONTEXT:
            if self.context_columns_ != [AEZ_COL, MONTH_COL]:
                raise ValueError("Embedding context requires exactly AEZ and Month columns")
            self.aez_codes_ = self._category_codes(X[AEZ_COL])
            self.month_codes_ = self._category_codes(X[MONTH_COL])
        elif self.context_mode == LSTM_NO_CONTEXT and self.context_columns_:
            raise ValueError("No-context mode requires spectral bands only")
        return self

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return network inputs in the representation selected during fitting."""
        self._validate_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("X must have the same columns in the same order as during fitting")
        inputs = {
            "spectrum": X[self.band_columns_].to_numpy(dtype=np.float32, copy=True)[..., None]
        }
        if self.context_mode == LSTM_NO_CONTEXT:
            return inputs
        if self.context_mode == LSTM_STANDARD_CONTEXT:
            inputs["context"] = X[self.context_columns_].to_numpy(dtype=np.float32, copy=True)
            return inputs
        inputs["aez_codes"] = self._encode_categories(X[AEZ_COL], self.aez_codes_)
        inputs["month_codes"] = self._encode_categories(X[MONTH_COL], self.month_codes_)
        return inputs

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X must be a non-empty pandas DataFrame")

    @staticmethod
    def _category_codes(values: pd.Series) -> dict[int, int]:
        categories = sorted(set(values.astype(int)))
        return {category: code for code, category in enumerate(categories, start=1)}

    @staticmethod
    def _encode_categories(values: pd.Series, codes: dict[int, int]) -> np.ndarray:
        return np.asarray([codes.get(int(value), 0) for value in values], dtype=np.int64)


class SpectralLSTMClassifier(ClassifierMixin, BaseEstimator):
    """Classify one target from a spectral sequence and optional static context.

    Args:
        context_mode: Whether to omit context, use preprocessed numeric context, or learn
            AEZ and Month embeddings.
        hidden_size: Width of the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between stacked LSTM layers.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 regularization strength.
        aez_embedding_dim: Width of the AEZ embedding.
        month_embedding_dim: Width of the Month embedding.
        max_epochs: Maximum training epochs before early stopping.
        batch_size: Number of rows per optimization batch.
        random_state: Seed for training randomness and the inner validation split.
    """

    def __init__(
        self,
        context_mode: LSTMContextMode = LSTM_STANDARD_CONTEXT,
        hidden_size: int = LSTM_DEFAULT_HIDDEN_SIZE,
        num_layers: int = LSTM_DEFAULT_NUM_LAYERS,
        dropout: float = LSTM_DEFAULT_DROPOUT,
        learning_rate: float = LSTM_DEFAULT_LEARNING_RATE,
        weight_decay: float = LSTM_DEFAULT_WEIGHT_DECAY,
        aez_embedding_dim: int = LSTM_DEFAULT_AEZ_EMBEDDING_DIM,
        month_embedding_dim: int = LSTM_DEFAULT_MONTH_EMBEDDING_DIM,
        max_epochs: int = LSTM_DEFAULT_MAX_EPOCHS,
        batch_size: int = LSTM_DEFAULT_BATCH_SIZE,
        random_state: int = SEED,
    ) -> None:
        """Store unfitted hyperparameters for sklearn cloning."""
        self.context_mode = context_mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.aez_embedding_dim = aez_embedding_dim
        self.month_embedding_dim = month_embedding_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _validate_hyperparameters(self) -> None:
        if self.context_mode not in (
            LSTM_NO_CONTEXT,
            LSTM_STANDARD_CONTEXT,
            LSTM_EMBEDDINGS_CONTEXT,
        ):
            raise ValueError(f"Unknown context mode {self.context_mode!r}")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.aez_embedding_dim < 1 or self.month_embedding_dim < 1:
            raise ValueError("embedding dimensions must be positive")
        if self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("max_epochs and batch_size must be positive")