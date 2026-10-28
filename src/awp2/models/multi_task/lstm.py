"""Multi-task LSTM: a spectral sequence encoder with a crop and a stage head."""

from __future__ import annotations

import copy
from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn
from torch.nn import functional as functional

from awp2.config import (
    CROP_COL,
    CROPS,
    LSTM_DEFAULT_AEZ_EMBEDDING_DIM,
    LSTM_DEFAULT_BATCH_SIZE,
    LSTM_DEFAULT_DROPOUT,
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
    SEED,
    STAGE_COL,
    STAGES,
    TARGET_COLS,
    LSTMContextMode,
)
from awp2.data import combined_label
from awp2.models.lstm import (
    _CategoricalContextEncoder,
    _LSTMInputAdapter,
    _SpectralLSTMEncoder,
)


class _MultiTaskLateFusionLSTMModule(nn.Module):
    """Predict crop and stage from shared LSTM features and static context."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_context_features: int,
    ) -> None:
        super().__init__()
        self.spectral_encoder = _SpectralLSTMEncoder(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        n_fused_features = hidden_size + n_context_features
        self.crop_head = nn.Linear(n_fused_features, len(CROPS))
        self.stage_head = nn.Linear(n_fused_features, len(STAGES))

    def forward(self, spectrum: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """Return crop and stage logits after fusing context with spectral features."""
        features = torch.cat((self.spectral_encoder(spectrum), context), dim=1)
        return self.crop_head(features), self.stage_head(features)


class _MultiTaskEmbeddingFusionLSTMModule(nn.Module):
    """Predict crop and stage from shared LSTM features and learned context."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_aez_categories: int,
        n_month_categories: int,
        aez_embedding_dim: int,
        month_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.spectral_encoder = _SpectralLSTMEncoder(
            input_size=1,
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
        n_fused_features = hidden_size + aez_embedding_dim + month_embedding_dim
        self.crop_head = nn.Linear(n_fused_features, len(CROPS))
        self.stage_head = nn.Linear(n_fused_features, len(STAGES))

    def forward(
        self, spectrum: Tensor, aez_codes: Tensor, month_codes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return crop and stage logits after fusing context embeddings."""
        context = self.context_encoder(aez_codes, month_codes)
        features = torch.cat((self.spectral_encoder(spectrum), context), dim=1)
        return self.crop_head(features), self.stage_head(features)


class MultiTaskLSTMClassifier(ClassifierMixin, BaseEstimator):
    """Learn shared LSTM features with separate crop and stage prediction heads.

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
        crop_loss_weight: Weight of the crop loss; the stage loss gets the remaining weight.
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
        crop_loss_weight: float = 0.5,
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
        self.crop_loss_weight = crop_loss_weight
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
        if not 0 < self.crop_loss_weight < 1:
            raise ValueError("crop_loss_weight must be in (0, 1)")

    def _build_module(self, adapter: _LSTMInputAdapter) -> nn.Module:
        if self.context_mode == LSTM_EMBEDDINGS_CONTEXT:
            return _MultiTaskEmbeddingFusionLSTMModule(
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                n_aez_categories=len(adapter.aez_codes_) + 1,
                n_month_categories=len(adapter.month_codes_) + 1,
                aez_embedding_dim=self.aez_embedding_dim,
                month_embedding_dim=self.month_embedding_dim,
            )
        n_context_features = (
            0 if self.context_mode == LSTM_NO_CONTEXT else len(adapter.context_columns_)
        )
        return _MultiTaskLateFusionLSTMModule(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_context_features=n_context_features,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit shared LSTM features and separate crop and stage heads.

        Args:
            X: Preprocessed spectral bands and context columns.
            y: Target frame containing the project ``Crop`` and ``Stage`` columns.
            sample_weight: Optional non-negative weight for each training row.

        Returns:
            This fitted classifier.

        Raises:
            TypeError: If features or targets are not pandas DataFrames.
            ValueError: If inputs, target labels or sample weights are invalid.
        """
        self._validate_hyperparameters()
        if not isinstance(y, pd.DataFrame):
            raise TypeError(f"Expected y as a pandas DataFrame, got {type(y)}.")
        if list(y.columns) != list(TARGET_COLS):
            raise ValueError(f"Expected target columns {list(TARGET_COLS)}, got {list(y.columns)}.")
        if not y.index.equals(X.index):
            raise ValueError("X and y must have the same index")
        if len(y) < 2:
            raise ValueError("At least two rows are required for the inner validation split")
        weights = self._validate_sample_weight(sample_weight, len(X))
        adapter = _LSTMInputAdapter(self.context_mode).fit(X)
        inputs = adapter.transform(X)
        self._validate_inputs(inputs)
        crop_codes = self._encode_targets(y[CROP_COL], CROPS, CROP_COL)
        stage_codes = self._encode_targets(y[STAGE_COL], STAGES, STAGE_COL)
        train_indices, validation_indices = train_test_split(
            np.arange(len(X)),
            test_size=LSTM_INNER_VALIDATION_FRACTION,
            stratify=combined_label(y),
            random_state=self.random_state,
        )
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.input_adapter_ = adapter
        self.feature_names_in_ = adapter.feature_names_in_
        self.n_features_in_ = X.shape[1]
        self.crop_classes_ = np.asarray(CROPS, dtype=object)
        self.stage_classes_ = np.asarray(STAGES, dtype=object)
        self.classes_ = [self.crop_classes_, self.stage_classes_]
        self.allowed_pair_codes_ = np.unique(
            np.column_stack((crop_codes, stage_codes)),
            axis=0,
        )
        self.module_ = self._build_module(adapter)
        crop_weights = self._class_weights(crop_codes[train_indices], len(CROPS))
        stage_weights = self._class_weights(stage_codes[train_indices], len(STAGES))
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
                crop_logits, stage_logits = self._logits(inputs, batch_indices)
                loss = self._loss(
                    crop_logits,
                    stage_logits,
                    crop_codes[batch_indices],
                    stage_codes[batch_indices],
                    weights[batch_indices],
                    crop_weights,
                    stage_weights,
                )
                loss.backward()
                optimizer.step()

            crop_logits, stage_logits = self._logits(inputs, validation_indices)
            validation_loss = self._loss(
                crop_logits,
                stage_logits,
                crop_codes[validation_indices],
                stage_codes[validation_indices],
                np.ones(len(validation_indices), dtype=np.float32),
                crop_weights,
                stage_weights,
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

        assert best_state is not None
        self.module_.load_state_dict(best_state)
        self.best_validation_loss_ = best_loss
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict crop and stage labels from preprocessed spectral and context features.

        Args:
            X: Preprocessed features with the fitted column order.

        Returns:
            Target frame with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, ["input_adapter_", "module_", "allowed_pair_codes_"])
        inputs = self.input_adapter_.transform(X)
        self._validate_inputs(inputs)
        self.module_.eval()
        with torch.no_grad():
            crop_logits, stage_logits = self._logits(inputs, np.arange(len(X)))
        allowed_pair_codes = torch.from_numpy(self.allowed_pair_codes_)
        pair_log_probabilities = (
            functional.log_softmax(crop_logits, dim=1)[:, allowed_pair_codes[:, 0]]
            + functional.log_softmax(stage_logits, dim=1)[:, allowed_pair_codes[:, 1]]
        )
        prediction_codes = self.allowed_pair_codes_[pair_log_probabilities.argmax(dim=1).numpy()]
        return pd.DataFrame(
            {
                CROP_COL: self.crop_classes_[prediction_codes[:, 0]],
                STAGE_COL: self.stage_classes_[prediction_codes[:, 1]],
            },
            index=X.index,
        )

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
    def _encode_targets(target: pd.Series, labels: tuple[str, ...], name: str) -> np.ndarray:
        codes = {label: code for code, label in enumerate(labels)}
        unknown = sorted(set(target) - set(codes))
        if unknown:
            raise ValueError(f"{name} contains unknown labels: {unknown}")
        return np.asarray([codes[label] for label in target], dtype=np.int64)

    @staticmethod
    def _class_weights(codes: np.ndarray, n_classes: int) -> Tensor:
        counts = np.bincount(codes, minlength=n_classes)
        weights = np.zeros(n_classes, dtype=np.float32)
        present = counts > 0
        weights[present] = counts.sum() / (present.sum() * counts[present])
        return torch.from_numpy(weights)

    def _logits(self, inputs: dict[str, np.ndarray], indices: np.ndarray) -> tuple[Tensor, Tensor]:
        spectrum = torch.from_numpy(inputs["spectrum"][indices])
        if self.context_mode == LSTM_EMBEDDINGS_CONTEXT:
            return self.module_(
                spectrum,
                torch.from_numpy(inputs["aez_codes"][indices]),
                torch.from_numpy(inputs["month_codes"][indices]),
            )
        context = torch.from_numpy(
            inputs["context"][indices]
            if "context" in inputs
            else np.empty((len(indices), 0), dtype=np.float32)
        )
        return self.module_(spectrum, context)

    def _loss(
        self,
        crop_logits: Tensor,
        stage_logits: Tensor,
        crop_codes: np.ndarray,
        stage_codes: np.ndarray,
        sample_weights: np.ndarray,
        crop_class_weights: Tensor,
        stage_class_weights: Tensor,
    ) -> Tensor:
        crop_loss = functional.cross_entropy(
            crop_logits,
            torch.from_numpy(crop_codes),
            weight=crop_class_weights,
            reduction="none",
        )
        stage_loss = functional.cross_entropy(
            stage_logits,
            torch.from_numpy(stage_codes),
            weight=stage_class_weights,
            reduction="none",
        )
        losses = self.crop_loss_weight * crop_loss + (1 - self.crop_loss_weight) * stage_loss
        weights = torch.from_numpy(sample_weights)
        return (losses * weights).sum() / weights.sum()
