"""Standard evaluation for crop and growth-stage predictions (matches the grading metrics)."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, make_scorer

from awp2.config import CROP_COL, FIGURE_DPI, SCORE_DECIMALS, STAGE_COL, TARGET_COLS
from awp2.data.split import CropStage, combined_label

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A score between 0 and 1."""


class Metrics(BaseModel):
    """Scores of one evaluation, each between 0 and 1.

    Balanced accuracy (``bacc_*``) is the primary grading metric; ``*_combined`` treats each
    crop/stage pair as one class.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    bacc_crop: Score = Field(description="Balanced accuracy of the crop.")
    f1_macro_crop: Score = Field(description="Macro-F1 of the crop.")
    bacc_stage: Score = Field(description="Balanced accuracy of the stage.")
    f1_macro_stage: Score = Field(description="Macro-F1 of the stage.")
    bacc_combined: Score = Field(description="Balanced accuracy of crop+stage.")
    f1_macro_combined: Score = Field(description="Macro-F1 of crop+stage.")
    f1_samples: Score = Field(description="Mean share of correct labels per sample.")
    invalid_combinations: Score | None = Field(
        default=None,
        description="Share of predicted crop/stage pairs that never occur in training.",
    )


MetricName = Literal[
    "bacc_crop",
    "f1_macro_crop",
    "bacc_stage",
    "f1_macro_stage",
    "bacc_combined",
    "f1_macro_combined",
    "f1_samples",
]

METRIC_LABELS: dict[str, str] = {
    "bacc_crop": "BAcc Kultur",
    "bacc_stage": "BAcc Stadium",
    "bacc_combined": "BAcc kombiniert",
    "f1_macro_crop": "Macro-F1 Kultur",
    "f1_macro_stage": "Macro-F1 Stadium",
    "f1_macro_combined": "Macro-F1 kombiniert",
    "f1_samples": "Samples-F1",
}
"""Readable (German) names of the metrics for plots and tables."""


def as_target_frame(pred: pd.DataFrame | np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Bring model predictions into the ``Crop``/``Stage`` form that ``evaluate`` expects.

    Args:
        pred: Output of ``predict`` – an array of shape ``(n, 2)`` or a two-column DataFrame.
        index: Index of the true labels, so rows line up.

    Returns:
        Predictions with the columns ``Crop`` and ``Stage``.
    """
    if isinstance(pred, pd.DataFrame):
        return pred.set_axis(list(TARGET_COLS), axis=1).set_axis(index)
    return pd.DataFrame(np.asarray(pred), columns=list(TARGET_COLS), index=index)


def scorer(metric: MetricName = "bacc_combined") -> Callable[..., float]:
    """Create an sklearn scorer from one of the ``Metrics`` fields.

    Needed for ``GridSearchCV``/``cross_validate`` because sklearn's default ``score()`` cannot
    handle two target columns.

    Args:
        metric: Name of the metric to optimise.

    Returns:
        Scorer to pass as ``scoring=``.
    """

    def _score(y_true: pd.DataFrame, y_pred: pd.DataFrame | np.ndarray) -> float:
        return getattr(evaluate(y_true, as_target_frame(y_pred, y_true.index)), metric)

    return make_scorer(_score)


def evaluate(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    valid_combinations: frozenset[CropStage] | None = None,
) -> Metrics:
    """Compute all grading metrics for crop, stage and their combination.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order as ``y_true``.
        valid_combinations: Crop/stage pairs seen in training; if given, the share of predicted
            pairs outside this set is reported as ``invalid_combinations``.

    Returns:
        All metrics, rounded to ``SCORE_DECIMALS``.
    """
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    true_pair, pred_pair = combined_label(y_true), combined_label(y_pred)
    # Each sample has exactly one crop and one stage label, so the per-sample F1 equals
    # the fraction of the two labels that were predicted correctly.
    correct = np.column_stack([y_true[col] == y_pred[col] for col in TARGET_COLS])

    return Metrics(
        bacc_crop=_bacc(y_true[CROP_COL], y_pred[CROP_COL]),
        f1_macro_crop=_f1_macro(y_true[CROP_COL], y_pred[CROP_COL]),
        bacc_stage=_bacc(y_true[STAGE_COL], y_pred[STAGE_COL]),
        f1_macro_stage=_f1_macro(y_true[STAGE_COL], y_pred[STAGE_COL]),
        bacc_combined=_bacc(true_pair, pred_pair),
        f1_macro_combined=_f1_macro(true_pair, pred_pair),
        f1_samples=_rounded(correct.mean()),
        invalid_combinations=(
            None if valid_combinations is None else _invalid_share(y_pred, valid_combinations)
        ),
    )


def _rounded(value: float) -> float:
    return round(float(value), SCORE_DECIMALS)


def _bacc(y_true: pd.Series, y_pred: pd.Series) -> float:
    return _rounded(balanced_accuracy_score(y_true, y_pred))


def _f1_macro(y_true: pd.Series, y_pred: pd.Series) -> float:
    return _rounded(f1_score(y_true, y_pred, average="macro"))


def _invalid_share(y_pred: pd.DataFrame, valid: frozenset[CropStage]) -> float:
    valid_pairs = {(pair.crop, pair.stage) for pair in valid}
    predicted = y_pred[[CROP_COL, STAGE_COL]].itertuples(index=False, name=None)
    return _rounded(np.mean([pair not in valid_pairs for pair in predicted]))


def plot_confusion_matrices(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Plot row-normalised confusion matrices for crop and stage side by side.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order as ``y_true``.
        save_path: Also save the figure there if given.

    Returns:
        The figure with one confusion matrix per target.
    """
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(14, 6))
    for ax, col in zip(axes, TARGET_COLS, strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_true[col], y_pred[col], normalize="true", values_format=".2f", ax=ax, colorbar=False
        )
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig
