"""Sequential LSTM components for hyperspectral signatures."""

import torch
from torch import Tensor, nn


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