"""Standard evaluation for crop and growth-stage predictions (matches the grading metrics)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score

from awp2.config import TARGET_COLS


def _combined(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLS[0]].astype(str) + "_" + df[TARGET_COLS[1]].astype(str)


def evaluate(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    """Compute the grading metrics for crop, stage and their combination.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order as ``y_true``.

    Returns:
        Balanced accuracy and macro-F1 for crop, stage and the combined label, plus samples-F1
            (mean share of correctly predicted labels per sample), each rounded to 4 decimals.
    """
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    scores: dict[str, float] = {}
    targets = {col.lower(): col for col in TARGET_COLS}
    for name, col in targets.items():
        scores[f"bacc_{name}"] = balanced_accuracy_score(y_true[col], y_pred[col])
        scores[f"f1_macro_{name}"] = f1_score(y_true[col], y_pred[col], average="macro")

    true_comb, pred_comb = _combined(y_true), _combined(y_pred)
    scores["bacc_combined"] = balanced_accuracy_score(true_comb, pred_comb)
    scores["f1_macro_combined"] = f1_score(true_comb, pred_comb, average="macro")

    # Each sample has exactly one crop and one stage label, so the per-sample F1 equals
    # the fraction of the two labels that were predicted correctly.
    correct = np.column_stack([y_true[col] == y_pred[col] for col in TARGET_COLS])
    scores["f1_samples"] = float(correct.mean())
    return {k: round(float(v), 4) for k, v in scores.items()}


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
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
