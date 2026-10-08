"""Sequential 1D-CNN classifiers for hyperspectral signatures."""

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn

from awp2.config import (
    CNN_DEFAULT_BATCH_SIZE,
    CNN_DEFAULT_DROPOUT,
    CNN_DEFAULT_FILTERS,
    CNN_DEFAULT_KERNEL_SIZE,
    CNN_DEFAULT_LEARNING_RATE,
    CNN_DEFAULT_MAX_EPOCHS,
    CNN_EARLY_STOPPING_PATIENCE,
    CNN_INNER_VALIDATION_FRACTION,
    SEED,
)
from awp2.data import band_columns, wavelengths
from awp2.models.combined import CombinedLabelClassifier


class _SpectralCNNModule(nn.Module):
    def __init__(
        self,
        n_context_features: int,
        n_classes: int,
        filters: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral_features = nn.Sequential(
            nn.Conv1d(2, filters, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(filters + n_context_features, n_classes)

    def forward(self, spectrum: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return class logits from spectral and contextual inputs."""
        features = torch.cat((self.spectral_features(spectrum), context), dim=1)
        return self.classifier(features)


class SpectralCNNClassifier(ClassifierMixin, BaseEstimator):
    """Classify a spectral sequence while keeping metadata outside the wavelength axis.

    The network receives a scaled reflectance channel and a normalized wavelength-position
    channel. The latter exposes gaps left after fully empty bands are removed. All metadata
    columns are supplied through a separate context branch.

    Args:
        filters: Number of convolutional filters in both convolution layers.
        kernel_size: Width of the local spectral convolution.
        dropout: Fraction of learned spectral features dropped before classification.
        learning_rate: Adam learning rate.
        max_epochs: Upper bound on training epochs; early stopping can stop sooner.
        batch_size: Number of samples per optimization batch.
        random_state: Seed for NumPy and PyTorch training randomness.
    """

    def __init__(
        self,
        filters: int = CNN_DEFAULT_FILTERS,
        kernel_size: int = CNN_DEFAULT_KERNEL_SIZE,
        dropout: float = CNN_DEFAULT_DROPOUT,
        learning_rate: float = CNN_DEFAULT_LEARNING_RATE,
        max_epochs: int = CNN_DEFAULT_MAX_EPOCHS,
        batch_size: int = CNN_DEFAULT_BATCH_SIZE,
        random_state: int = SEED,
    ) -> None:
        """Store CNN hyperparameters for sklearn cloning."""
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "SpectralCNNClassifier":
        """Fit a weighted CNN for one combined crop/stage class per sample.

        Args:
            X: Scaled spectral bands plus optional encoded metadata columns.
            y: One combined crop/stage label per sample.
            sample_weight: Unsupported; this classifier balances classes internally.

        Returns:
            This fitted classifier.

        Raises:
            TypeError: If features are not supplied as a DataFrame.
            ValueError: If external sample weights are passed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X)}.")
        if sample_weight is not None:
            raise ValueError("SpectralCNNClassifier balances combined classes internally.")

        self.band_columns_ = band_columns(X)
        self.context_columns_ = [column for column in X.columns if column not in self.band_columns_]
        self.wavelength_positions_ = self._normalized_wavelengths(X[self.band_columns_])
        self.classes_, encoded_y = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        self.n_features_in_ = X.shape[1]

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        class_weights = self._balanced_class_weights(encoded_y)
        self.net_ = NeuralNetClassifier(
            _SpectralCNNModule,
            module__n_context_features=len(self.context_columns_),
            module__n_classes=len(self.classes_),
            module__filters=self.filters,
            module__kernel_size=self.kernel_size,
            module__dropout=self.dropout,
            criterion=nn.CrossEntropyLoss,
            criterion__weight=torch.tensor(class_weights, dtype=torch.float32),
            optimizer=torch.optim.Adam,
            lr=self.learning_rate,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=ValidSplit(
                cv=CNN_INNER_VALIDATION_FRACTION,
                stratified=True,
                random_state=self.random_state,
            ),
            callbacks=[EarlyStopping(patience=CNN_EARLY_STOPPING_PATIENCE, load_best=True)],
            verbose=0,
        )
        self.net_.fit(self._network_input(X), encoded_y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the combined crop/stage label for each sample.

        Args:
            X: Scaled spectral bands plus optional encoded metadata columns.

        Returns:
            One observed combined label per sample.
        """
        check_is_fitted(self, "net_")
        predicted_codes = self.net_.predict(self._network_input(X))
        return self.classes_[predicted_codes]

    def _network_input(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        spectrum = X[self.band_columns_].to_numpy(dtype=np.float32)
        positions = np.broadcast_to(self.wavelength_positions_, spectrum.shape)
        context = X[self.context_columns_].to_numpy(dtype=np.float32)
        return {
            "spectrum": np.stack((spectrum, positions), axis=1),
            "context": context,
        }

    @staticmethod
    def _balanced_class_weights(encoded_y: np.ndarray) -> np.ndarray:
        counts = np.bincount(encoded_y)
        return len(encoded_y) / (len(counts) * counts)

    @staticmethod
    def _normalized_wavelengths(spectra: pd.DataFrame) -> np.ndarray:
        positions = np.asarray(wavelengths(spectra), dtype=np.float32)
        return (positions - positions.min()) / (positions.max() - positions.min())


def make_combined_spectral_cnn() -> CombinedLabelClassifier:
    """Create the 1D-CNN classifier for the combined-label approach.

    Returns:
        Unfitted CNN that balances observed combined classes internally.
    """
    return CombinedLabelClassifier(SpectralCNNClassifier())