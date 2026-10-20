"""Sequential LSTM components for hyperspectral signatures."""

import copy
import warnings
from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn

from awp2.config import (
    AEZ_COL,
    GRU_CELL,
    LSTM_CELL,
    LSTM_DEFAULT_AEZ_EMBEDDING_DIM,
    LSTM_DEFAULT_BANDS_PER_STEP,
    LSTM_DEFAULT_BATCH_SIZE,
    LSTM_DEFAULT_DROPOUT,
    LSTM_DEFAULT_HEAD_HIDDEN_SIZE,
    LSTM_DEFAULT_HIDDEN_SIZE,
    LSTM_DEFAULT_LEARNING_RATE,
    LSTM_DEFAULT_MAX_EPOCHS,
    LSTM_DEFAULT_MONTH_EMBEDDING_DIM,
    LSTM_DEFAULT_NUM_LAYERS,
    LSTM_DEFAULT_WEIGHT_DECAY,
    LSTM_EARLY_STOPPING_PATIENCE,
    LSTM_EMBEDDINGS_CONTEXT,
    LSTM_INNER_VALIDATION_FRACTION,
    LSTM_NO_CONTEXT,
    LSTM_STANDARD_CONTEXT,
    MONTH_COL,
    SAVGOL_SPLIT_GAP_NM,
    SEED,
    LSTMContextMode,
    SequenceCell,
)
from awp2.data import band_columns, wavelength


class _SpectralLSTMEncoder(nn.Module):
    """Encode an ordered spectral sequence into one fixed-size representation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell: SequenceCell = LSTM_CELL,
    ) -> None:
        super().__init__()
        recurrent = nn.GRU if cell == GRU_CELL else nn.LSTM
        # the attribute keeps its name so that pickled LSTM models still load
        self.lstm = recurrent(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, spectrum: Tensor) -> Tensor:
        """Return the final hidden representation for each spectral sequence."""
        _outputs, state = self.lstm(spectrum)
        hidden_state = state[0] if isinstance(state, tuple) else state
        return hidden_state[-1]


def _classifier_head(
    n_features: int, head_hidden_size: int, dropout: float, n_classes: int
) -> nn.Module:
    if head_hidden_size == 0:
        return nn.Linear(n_features, n_classes)
    # a hidden layer lets spectrum and context interact non-linearly after the fusion
    return nn.Sequential(
        nn.Linear(n_features, head_hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(head_hidden_size, n_classes),
    )


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
        cell: SequenceCell = LSTM_CELL,
        head_hidden_size: int = LSTM_DEFAULT_HEAD_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.spectral_encoder = _SpectralLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell=cell,
        )
        self.classifier = _classifier_head(
            hidden_size + n_context_features, head_hidden_size, dropout, n_classes
        )

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
        cell: SequenceCell = LSTM_CELL,
        head_hidden_size: int = LSTM_DEFAULT_HEAD_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.spectral_encoder = _SpectralLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell=cell,
        )
        self.context_encoder = _CategoricalContextEncoder(
            n_aez_categories=n_aez_categories,
            n_month_categories=n_month_categories,
            aez_embedding_dim=aez_embedding_dim,
            month_embedding_dim=month_embedding_dim,
        )
        self.classifier = _classifier_head(
            hidden_size + aez_embedding_dim + month_embedding_dim,
            head_hidden_size,
            dropout,
            n_classes,
        )

    def forward(self, spectrum: Tensor, aez_codes: Tensor, month_codes: Tensor) -> Tensor:
        """Return class logits after fusing the encoded spectrum and context embeddings."""
        context = self.context_encoder(aez_codes, month_codes)
        features = torch.cat((self.spectral_encoder(spectrum), context), dim=1)
        return self.classifier(features)


class _LSTMInputAdapter:
    """Separate spectral sequences from the configured static context representation.

    Each sequence step holds ``bands_per_step`` neighbouring bands; steps never span a
    water-absorption gap, the last step of a segment repeats its edge band. The optional
    wavelength channel tells the network where a step lies, since the gaps make equal steps
    cover very different wavelength distances.
    """

    def __init__(
        self,
        context_mode: LSTMContextMode,
        bands_per_step: int = LSTM_DEFAULT_BANDS_PER_STEP,
        wavelength_channel: bool = False,
    ) -> None:
        self.context_mode = context_mode
        self.bands_per_step = bands_per_step
        self.wavelength_channel = wavelength_channel

    def fit(self, X: pd.DataFrame) -> Self:
        """Learn the fitted input columns and categorical context codes.

        Args:
            X: Preprocessed spectral bands and context columns.

        Returns:
            This fitted input adapter.

        Raises:
            ValueError: If features are empty, lack spectral bands or violate the context mode.
        """
        self._validate_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.band_columns_ = band_columns(X)
        if not self.band_columns_:
            raise ValueError("X must contain at least one spectral band")
        wavelengths_nm = np.asarray([wavelength(band) for band in self.band_columns_])
        self.step_layout_ = _step_layout(wavelengths_nm, self.bands_per_step)
        centers = wavelengths_nm[self.step_layout_].mean(axis=1)
        span = centers.max() - centers.min()
        relative = (centers - centers.min()) / span if span else np.zeros_like(centers)
        self.step_wavelengths_ = relative.astype(np.float32)
        self.input_size_ = self.bands_per_step + int(self.wavelength_channel)
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
        """Return network inputs in the representation selected during fitting.

        Args:
            X: Preprocessed features with the fitted column order.

        Returns:
            Arrays for the spectral sequence and configured context representation.

        Raises:
            ValueError: If features are empty or have different columns than during fitting.
        """
        self._validate_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("X must have the same columns in the same order as during fitting")
        bands = X[self.band_columns_].to_numpy(dtype=np.float32, copy=True)
        spectrum = bands[:, self.step_layout_]
        if self.wavelength_channel:
            position = np.broadcast_to(
                self.step_wavelengths_[None, :, None], (len(X), len(self.step_wavelengths_), 1)
            )
            spectrum = np.concatenate([spectrum, position], axis=2)
        inputs = {"spectrum": np.ascontiguousarray(spectrum)}
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


def _step_layout(wavelengths_nm: np.ndarray, bands_per_step: int) -> np.ndarray:
    gaps = np.flatnonzero(np.diff(wavelengths_nm) > SAVGOL_SPLIT_GAP_NM) + 1
    steps = []
    for segment in np.split(np.arange(len(wavelengths_nm)), gaps):
        for start in range(0, len(segment), bands_per_step):
            block = segment[start : start + bands_per_step]
            steps.append(np.pad(block, (0, bands_per_step - len(block)), mode="edge"))
    return np.stack(steps)


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
        cell: Recurrent cell, ``"lstm"`` or ``"gru"`` (fewer parameters).
        bands_per_step: Neighbouring bands fed together per sequence step; larger values
            shorten the sequence.
        wavelength_channel: Add the normalised wavelength of each step as extra input.
        head_hidden_size: Width of a hidden layer after fusing spectrum and context; 0 keeps a
            single linear layer.
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
        cell: SequenceCell = LSTM_CELL,
        bands_per_step: int = LSTM_DEFAULT_BANDS_PER_STEP,
        wavelength_channel: bool = False,
        head_hidden_size: int = LSTM_DEFAULT_HEAD_HIDDEN_SIZE,
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
        self.cell = cell
        self.bands_per_step = bands_per_step
        self.wavelength_channel = wavelength_channel
        self.head_hidden_size = head_hidden_size

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled classifier, e.g. one loaded from MLflow.

        Args:
            state: Attributes of the pickled object.
        """
        # models stored before these options existed used their default behaviour
        state.setdefault("cell", LSTM_CELL)
        state.setdefault("bands_per_step", LSTM_DEFAULT_BANDS_PER_STEP)
        state.setdefault("wavelength_channel", False)
        state.setdefault("head_hidden_size", LSTM_DEFAULT_HEAD_HIDDEN_SIZE)
        super().__setstate__(state)

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
        if self.cell not in (LSTM_CELL, GRU_CELL):
            raise ValueError(f"Unknown cell {self.cell!r}")
        if self.bands_per_step < 1 or self.head_hidden_size < 0:
            raise ValueError("bands_per_step must be positive and head_hidden_size non-negative")

    def _build_module(self, adapter: _LSTMInputAdapter, n_classes: int) -> nn.Module:
        if self.context_mode == LSTM_EMBEDDINGS_CONTEXT:
            return _EmbeddingFusionLSTMClassifier(
                input_size=adapter.input_size_,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                n_aez_categories=len(adapter.aez_codes_) + 1,
                n_month_categories=len(adapter.month_codes_) + 1,
                aez_embedding_dim=self.aez_embedding_dim,
                month_embedding_dim=self.month_embedding_dim,
                n_classes=n_classes,
                cell=self.cell,
                head_hidden_size=self.head_hidden_size,
            )
        n_context_features = (
            0 if self.context_mode == LSTM_NO_CONTEXT else len(adapter.context_columns_)
        )
        return _LateFusionLSTMClassifier(
            input_size=adapter.input_size_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_context_features=n_context_features,
            n_classes=n_classes,
            cell=self.cell,
            head_hidden_size=self.head_hidden_size,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit one class label per spectrum with balanced weighted cross-entropy.

        Args:
            X: Preprocessed spectral bands and context columns.
            y: One target label per row of ``X``.
            sample_weight: Optional non-negative weight per training row.

        Returns:
            This fitted classifier.

        Raises:
            TypeError: If targets are neither a pandas Series nor a one-dimensional NumPy array.
            ValueError: If features, targets or sample weights are invalid.
        """
        self._validate_hyperparameters()
        if isinstance(y, pd.Series):
            if not y.index.equals(X.index):
                raise ValueError("X and y must have the same index")
            target_values = y.to_numpy(dtype=str)
        elif isinstance(y, np.ndarray) and y.ndim == 1:
            target_values = y.astype(str, copy=False)
        else:
            raise TypeError(
                f"Expected y as a pandas Series or one-dimensional ndarray, got {type(y)}."
            )
        if len(target_values) != len(X):
            raise ValueError("X and y must have the same index")
        if len(target_values) < 2:
            raise ValueError("At least two rows are required for the inner validation split")
        weights = self._validate_sample_weight(sample_weight, len(X))
        adapter = _LSTMInputAdapter(
            self.context_mode, self.bands_per_step, self.wavelength_channel
        ).fit(X)
        inputs = adapter.transform(X)
        self._validate_inputs(inputs)
        self.classes_, target_codes = np.unique(target_values, return_inverse=True)
        train_indices, validation_indices = train_test_split(
            np.arange(len(X)),
            test_size=LSTM_INNER_VALIDATION_FRACTION,
            stratify=target_codes,
            random_state=self.random_state,
        )
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.input_adapter_ = adapter
        self.feature_names_in_ = adapter.feature_names_in_
        self.n_features_in_ = X.shape[1]
        self.module_ = self._build_module(adapter, len(self.classes_))
        class_weights = self._class_weights(target_codes[train_indices], len(self.classes_))
        optimizer = torch.optim.Adam(
            self.module_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        best_state: dict[str, Tensor] | None = None
        best_loss = float("inf")
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            self.module_.train()
            shuffled_indices = np.random.default_rng(self.random_state + epoch).permutation(
                train_indices
            )
            for start in range(0, len(shuffled_indices), self.batch_size):
                batch_indices = shuffled_indices[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = self._logits(inputs, batch_indices)
                loss = self._loss(
                    logits,
                    target_codes[batch_indices],
                    weights[batch_indices],
                    class_weights,
                )
                loss.backward()
                optimizer.step()

            validation_loss = self._loss(
                self._logits(inputs, validation_indices),
                target_codes[validation_indices],
                np.ones(len(validation_indices), dtype=np.float32),
                class_weights,
            ).item()
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.module_.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= LSTM_EARLY_STOPPING_PATIENCE:
                    self.n_epochs_ = epoch + 1
                    break
        else:
            self.n_epochs_ = self.max_epochs
            warnings.warn(
                f"Training stopped at max_epochs={self.max_epochs} while the inner validation "
                "loss was still improving; raise max_epochs.",
                stacklevel=2,
            )
        self.stopped_early_ = self.n_epochs_ < self.max_epochs

        assert best_state is not None
        self.module_.load_state_dict(best_state)
        self.best_validation_loss_ = best_loss
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict one target label per spectrum.

        Args:
            X: Preprocessed features with the fitted column order.

        Returns:
            One predicted class label per row of ``X``.
        """
        check_is_fitted(self, ["input_adapter_", "module_"])
        inputs = self.input_adapter_.transform(X)
        self._validate_inputs(inputs)
        self.module_.eval()
        with torch.no_grad():
            prediction_codes = self._logits(inputs, np.arange(len(X))).argmax(dim=1).numpy()
        return self.classes_[prediction_codes]

    @staticmethod
    def _validate_sample_weight(sample_weight: np.ndarray | None, n_samples: int) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n_samples, dtype=np.float32)
        weights = np.asarray(sample_weight, dtype=np.float32)
        if weights.shape != (n_samples,):
            raise ValueError(f"sample_weight must have shape ({n_samples},), got {weights.shape}")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() == 0:
            raise ValueError("sample_weight must be finite, non-negative and not all zero")
        return weights

    @staticmethod
    def _validate_inputs(inputs: dict[str, np.ndarray]) -> None:
        if not np.isfinite(inputs["spectrum"]).all():
            raise ValueError("Spectral inputs must be finite")
        if "context" in inputs and not np.isfinite(inputs["context"]).all():
            raise ValueError("Context inputs must be finite")

    @staticmethod
    def _class_weights(target_codes: np.ndarray, n_classes: int) -> Tensor:
        counts = np.bincount(target_codes, minlength=n_classes)
        weights = np.zeros(n_classes, dtype=np.float32)
        present = counts > 0
        weights[present] = counts.sum() / (present.sum() * counts[present])
        return torch.from_numpy(weights)

    def _logits(self, inputs: dict[str, np.ndarray], indices: np.ndarray) -> Tensor:
        spectrum = torch.from_numpy(inputs["spectrum"][indices])
        if self.context_mode == LSTM_EMBEDDINGS_CONTEXT:
            return self.module_(
                spectrum,
                torch.from_numpy(inputs["aez_codes"][indices]),
                torch.from_numpy(inputs["month_codes"][indices]),
            )
        context = torch.from_numpy(
            inputs.get("context", np.empty((len(indices), 0), dtype=np.float32))[indices]
            if "context" in inputs
            else np.empty((len(indices), 0), dtype=np.float32)
        )
        return self.module_(spectrum, context)

    @staticmethod
    def _loss(
        logits: Tensor,
        target_codes: np.ndarray,
        sample_weights: np.ndarray,
        class_weights: Tensor,
    ) -> Tensor:
        losses = nn.functional.cross_entropy(
            logits,
            torch.from_numpy(target_codes),
            weight=class_weights,
            reduction="none",
        )
        weights = torch.from_numpy(sample_weights)
        return (losses * weights).sum() / weights.sum()
