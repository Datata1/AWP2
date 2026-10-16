"""Sequential LSTM components for hyperspectral signatures."""

from typing import Self

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from awp2.config import (
    AEZ_COL,
    LSTM_EMBEDDINGS_CONTEXT,
    LSTM_NO_CONTEXT,
    LSTM_STANDARD_CONTEXT,
    LSTMContextMode,
    MONTH_COL,
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