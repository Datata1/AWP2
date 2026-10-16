"""Sequential LSTM components for hyperspectral signatures."""

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