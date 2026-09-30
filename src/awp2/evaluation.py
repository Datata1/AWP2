"""Standard evaluation for crop and growth-stage predictions (matches the grading metrics)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, make_scorer

from awp2.config import TARGET_COLS
from awp2.data.split import combined_label


def as_target_frame(pred: pd.DataFrame | np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Bring predictions (array of shape (n, 2) or DataFrame) into ``Crop``/``Stage`` form."""
    if isinstance(pred, pd.DataFrame):
        return pred.set_axis(list(TARGET_COLS), axis=1).set_axis(index)
    return pd.DataFrame(np.asarray(pred), columns=list(TARGET_COLS), index=index)


def scorer(metric: str = "bacc_combined"):
    """sklearn scorer for ``GridSearchCV`` / ``cross_validate`` based on :func:`evaluate`.

    Needed because sklearn's default ``score()`` cannot handle two target columns.
    """

    def _score(y_true: pd.DataFrame, y_pred) -> float:
        return evaluate(y_true, as_target_frame(y_pred, y_true.index))[metric]

    return make_scorer(_score)


def evaluate(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    """Compute balanced accuracy and macro-F1 for crop, stage and their combination,
    plus samples-F1 (mean share of correctly predicted labels per sample).

    Both frames need the columns ``Crop`` and ``Stage`` and the same row order.
    """
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    scores: dict[str, float] = {}
    targets = {col.lower(): col for col in TARGET_COLS}
    for name, col in targets.items():
        scores[f"bacc_{name}"] = balanced_accuracy_score(y_true[col], y_pred[col])
        scores[f"f1_macro_{name}"] = f1_score(y_true[col], y_pred[col], average="macro")

    true_comb, pred_comb = combined_label(y_true), combined_label(y_pred)
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
