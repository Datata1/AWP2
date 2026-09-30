"""Standard evaluation for crop and growth-stage predictions (matches the grading metrics)."""

from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, make_scorer

from awp2.config import TARGET_COLS
from awp2.data.split import combined_label

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A score between 0 and 1."""


class Metrics(BaseModel):
    """Scores of one evaluation. BAcc = balanced accuracy (primary grading metric)."""

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


def as_target_frame(pred: pd.DataFrame | np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Bring predictions (array of shape (n, 2) or DataFrame) into ``Crop``/``Stage`` form."""
    if isinstance(pred, pd.DataFrame):
        return pred.set_axis(list(TARGET_COLS), axis=1).set_axis(index)
    return pd.DataFrame(np.asarray(pred), columns=list(TARGET_COLS), index=index)


def scorer(metric: MetricName = "bacc_combined"):
    """sklearn scorer for ``GridSearchCV`` / ``cross_validate`` based on :func:`evaluate`.

    Needed because sklearn's default ``score()`` cannot handle two target columns.
    """

    def _score(y_true: pd.DataFrame, y_pred: pd.DataFrame | np.ndarray) -> float:
        return getattr(evaluate(y_true, as_target_frame(y_pred, y_true.index)), metric)

    return make_scorer(_score)


def evaluate(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    valid_combinations: set[tuple[str, str]] | None = None,
) -> Metrics:
    """Balanced accuracy and macro-F1 for crop, stage and their combination, plus samples-F1.

    Both frames need the columns ``Crop`` and ``Stage`` and the same row order. Pass the
    crop/stage pairs seen in training to also get ``invalid_combinations``.
    """
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    scores: dict[str, float] = {}
    for col in TARGET_COLS:
        name = col.lower()
        scores[f"bacc_{name}"] = balanced_accuracy_score(y_true[col], y_pred[col])
        scores[f"f1_macro_{name}"] = f1_score(y_true[col], y_pred[col], average="macro")

    true_comb, pred_comb = combined_label(y_true), combined_label(y_pred)
    scores["bacc_combined"] = balanced_accuracy_score(true_comb, pred_comb)
    scores["f1_macro_combined"] = f1_score(true_comb, pred_comb, average="macro")

    # Each sample has exactly one crop and one stage label, so the per-sample F1 equals
    # the fraction of the two labels that were predicted correctly.
    correct = np.column_stack([y_true[col] == y_pred[col] for col in TARGET_COLS])
    scores["f1_samples"] = float(correct.mean())

    if valid_combinations is not None:
        pairs = y_pred.itertuples(index=False, name=None)
        scores["invalid_combinations"] = float(
            np.mean([pair not in valid_combinations for pair in pairs])
        )
    return Metrics(**{k: round(float(v), 4) for k, v in scores.items()})


def plot_confusion_matrices(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Plot row-normalized confusion matrices for crop and stage side by side."""
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(14, 6))
    for ax, col in zip(axes, TARGET_COLS, strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_true[col], y_pred[col], normalize="true", values_format=".2f", ax=ax, colorbar=False
        )
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
