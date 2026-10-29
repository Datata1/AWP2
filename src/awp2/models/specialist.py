"""A second model that re-decides the rows of one confused crop group."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    CROP_COL,
    EARLY_STAGES,
    LABEL_SEP,
    SPECIALIST_CROPS,
    STAGE_COL,
    TARGET_COLS,
    Crop,
    Stage,
)
from awp2.data import combined_label
from awp2.evaluation import as_target_frame


class PairSpecialistClassifier(ClassifierMixin, BaseEstimator):
    """Base model for all rows; a specialist re-decides the rows it predicts into one group.

    The group is every pair of ``crops`` × ``stages`` (by default early maize and soybean). The
    specialist is trained only on the true rows of the group, so it can spend its capacity on
    the small differences inside it; rows the base model wrongly sends elsewhere stay as they are.

    Args:
        base: Unfitted classifier that predicts ``Crop`` and ``Stage`` for all rows.
        specialist: Unfitted single-label classifier for the combined ``Crop|Stage`` label.
        crops: Crops of the group.
        stages: Stages of the group.
    """

    def __init__(
        self,
        base: BaseEstimator,
        specialist: ClassifierMixin,
        crops: tuple[Crop, ...] = SPECIALIST_CROPS,
        stages: tuple[Stage, ...] = EARLY_STAGES,
    ) -> None:
        """Initialise from two unfitted classifiers and the group.

        Args:
            base: Classifier for all rows.
            specialist: Classifier for the rows of the group.
            crops: Crops of the group.
            stages: Stages of the group.
        """
        self.base = base
        self.specialist = specialist
        self.crops = crops
        self.stages = stages

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None = None
    ) -> "PairSpecialistClassifier":
        """Fit the base model on all rows and the specialist on the true rows of the group.

        Args:
            X: Model features.
            y: Targets with the columns ``Crop`` and ``Stage``.
            sample_weight: Optional weight for each sample, passed to both models.

        Returns:
            This fitted classifier.

        Raises:
            ValueError: If the target columns do not match the project contract or the group
                has fewer than two classes in ``y``.
        """
        if list(y.columns) != list(TARGET_COLS):
            raise ValueError(f"Expected target columns {list(TARGET_COLS)}, got {list(y.columns)}.")
        in_group = self._in_group(y).to_numpy()
        if combined_label(y[in_group]).nunique() < 2:
            raise ValueError("The specialist group needs at least two classes in y.")
        weights = {} if sample_weight is None else {"sample_weight": sample_weight}
        group_weights = {} if sample_weight is None else {"sample_weight": sample_weight[in_group]}
        self.base_ = clone(self.base).fit(X, y, **weights)
        self.specialist_ = clone(self.specialist).fit(
            X[in_group], combined_label(y[in_group]), **group_weights
        )
        self.classes_ = np.unique(combined_label(y).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict with the base model, then let the specialist re-decide its group.

        Args:
            X: Model features.

        Returns:
            Predicted targets with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, "specialist_")
        predicted = as_target_frame(self.base_.predict(X), X.index)
        routed = self._in_group(predicted).to_numpy()
        if routed.any():
            pairs = pd.Series(self.specialist_.predict(X[routed]), index=X.index[routed])
            predicted.loc[routed, list(TARGET_COLS)] = pairs.str.split(
                LABEL_SEP, n=1, expand=True
            ).to_numpy()
        return predicted

    def _in_group(self, y: pd.DataFrame) -> pd.Series:
        return y[CROP_COL].isin(self.crops) & y[STAGE_COL].isin(self.stages)


def specialist_upper_bound(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    crops: tuple[Crop, ...] = SPECIALIST_CROPS,
    stages: tuple[Stage, ...] = EARLY_STAGES,
) -> pd.DataFrame:
    """Predictions as a perfect specialist would leave them – the most it can gain.

    A row is corrected if the prediction falls into the group and the true pair does too; the
    specialist never sees rows outside its group, so those errors remain.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predictions of the base model, same index as ``y_true``.
        crops: Crops of the group.
        stages: Stages of the group.

    Returns:
        ``y_pred`` with the correctable rows set to the truth.
    """

    def in_group(y: pd.DataFrame) -> pd.Series:
        return y[CROP_COL].isin(crops) & y[STAGE_COL].isin(stages)

    corrected = y_pred.copy()
    fixable = in_group(y_pred) & in_group(y_true)
    corrected.loc[fixable] = y_true.loc[fixable].to_numpy()
    return corrected
