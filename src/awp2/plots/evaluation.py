"""Plots of model results: metrics, tuning, recall, confusions and errors."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import BarContainer

from awp2.config import (
    RECALL_COL,
    SUPPORT_COL,
    TARGET_COLS,
)
from awp2.evaluation import METRIC_LABELS, ErrorBreakdown, MetricName, Metrics


def plot_metric_comparison(
    results: Mapping[str, Metrics],
    metrics: Sequence[MetricName] = ("bacc_crop", "bacc_stage", "bacc_combined"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare several models on a few metrics as grouped bars.

    Args:
        results: Metrics per model, keyed by the label to show (e.g. ``{"Dummy": ..., "RF": ...}``).
        metrics: Which metrics to show, one bar group each.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one bar per model and metric, values written on the bars.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    table = pd.DataFrame({label: m.model_dump() for label, m in results.items()}).loc[list(metrics)]
    table.index = [METRIC_LABELS.get(metric, metric) for metric in metrics]
    table.plot.bar(ax=ax, rot=0, width=0.75)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.2f", fontsize="small", padding=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (Validierung)")
    ax.legend(title="Modell", fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_tuning_candidates(
    candidates: pd.DataFrame,
    color_by: str | None = None,
    validation_score: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Show all candidates of a hyperparameter search with their cross-validation score.

    Args:
        candidates: ``TuneResult.candidates`` (columns ``params``, ``mean``, ``std``, ``rank``).
        color_by: Hyperparameter whose values colour the bars, to make its effect visible.
        validation_score: Score of the best candidate on the validation part, drawn as a line to
            compare it with the cross-validation estimate.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one horizontal bar (mean ± std over the folds) per candidate, best on top.
    """
    ordered = candidates.sort_values("mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.45 * len(ordered) + 1.5))
    labels = [", ".join(f"{k}={v}" for k, v in params.items()) for params in ordered["params"]]
    values = (
        [params.get(color_by) for params in ordered["params"]]
        if color_by
        else [None] * len(ordered)
    )
    palette = {value: f"C{number}" for number, value in enumerate(dict.fromkeys(values))}
    ax.barh(
        labels, ordered["mean"], xerr=ordered["std"], color=[palette[v] for v in values], capsize=3
    )
    if color_by:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in palette.values()]
        ax.legend(
            handles, [f"{color_by}={v}" for v in palette], fontsize="small", loc="lower right"
        )
    if validation_score is not None:
        ax.axvline(validation_score, color="black", linestyle="--", linewidth=1)
        ax.text(validation_score, len(ordered) - 0.4, " Validierung", fontsize="small", va="top")
    ax.set_xlabel("CV-Score (Mittelwert ± Std über die Folds)")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_cv_scores(
    candidates: Mapping[str, pd.DataFrame],
    reference: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare the best cross-validation score of several searches, e.g. preprocessing variants.

    Args:
        candidates: ``TuneResult.candidates`` per search, keyed by the label to show.
        reference: CV score drawn as a dashed line, e.g. the variant everything is compared to.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one horizontal bar (best mean ± its std over the folds) per search, best on
            top and the value written next to each bar.
    """
    best = pd.DataFrame(
        {
            label: table.loc[table["rank"].idxmin(), ["mean", "std"]]
            for label, table in candidates.items()
        }
    ).T.sort_values("mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.5 * len(best) + 1.5))
    bars = ax.barh(best.index, best["mean"], xerr=best["std"], color="C0", capsize=3)
    ax.bar_label(bars, fmt="%.3f", fontsize="small", padding=12)
    if reference is not None:
        ax.axvline(reference, color="black", linestyle="--", linewidth=1)
    lower = (best["mean"] - best["std"]).min()
    ax.set_xlim(max(0.0, lower - 0.02), min(1.0, (best["mean"] + best["std"]).max() + 0.03))
    ax.set_xlabel("Bester CV-Score (Mittelwert ± Std über die Folds)")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_class_recall(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> plt.Figure:
    """Recall per class for crop and stage – which classes the model finds worst.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order.

    Returns:
        Figure with one horizontal bar chart per target, weakest class at the bottom.
    """
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(12, 4))
    for ax, col in zip(axes, TARGET_COLS, strict=True):
        hit = y_true[col].to_numpy() == y_pred[col].to_numpy()
        recall = pd.Series(hit, index=y_true[col].to_numpy()).groupby(level=0).agg(["mean", "size"])
        recall = recall.sort_values("mean")
        labels = [f"{cls} (n={n})" for cls, n in zip(recall.index, recall["size"], strict=True)]
        bars = ax.barh(labels, recall["mean"], color="C0")
        ax.bar_label(bars, fmt="%.2f", fontsize="small", padding=2)
        ax.set_xlim(0, 1.1)
        ax.set_title(col)
        ax.set_xlabel("Recall")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_confusions(y_true: pd.DataFrame, y_pred: pd.DataFrame, top: int = 5) -> plt.Figure:
    """The most frequent mix-ups (true → predicted) for crop and stage.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order.
        top: Number of mix-ups to show per target.

    Returns:
        Figure with one horizontal bar chart per target, most frequent mix-up on top.
    """
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(12, 0.5 * top + 1.5))
    for ax, col in zip(axes, TARGET_COLS, strict=True):
        pairs = pd.DataFrame({"true": y_true[col].to_numpy(), "pred": y_pred[col].to_numpy()})
        wrong = pairs[pairs["true"] != pairs["pred"]]
        counts = wrong.value_counts().head(top).iloc[::-1]
        labels = [f"{t} → {p}" for t, p in counts.index]
        bars = ax.barh(labels, counts.to_numpy(), color="C3")
        ax.bar_label(bars, fontsize="small", padding=2)
        ax.set_title(f"{col}: häufigste Verwechslungen")
        ax.set_xlabel("Anzahl (Validierung)")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


_ERROR_SHARE_LABELS = {
    "both_correct": "beides richtig",
    "stage_only_wrong": "nur Stadium falsch",
    "crop_wrong": "Kultur falsch",
}

_ERROR_SHARE_COLORS = ("C2", "C1", "C3")

_MIN_LABELLED_SHARE = 0.05  # narrower segments are too small for a readable label


def plot_error_breakdown(
    breakdowns: Mapping[str, ErrorBreakdown], ax: plt.Axes | None = None
) -> plt.Axes:
    """Compare where the errors of several models come from as stacked bars.

    Args:
        breakdowns: ``error_breakdown()`` per model, keyed by the label to show.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one bar per model, split into fully correct, stage-only and crop errors.
    """
    table = pd.DataFrame(
        {label: breakdown.model_dump() for label, breakdown in breakdowns.items()}
    ).T[list(_ERROR_SHARE_LABELS)]
    table.columns = list(_ERROR_SHARE_LABELS.values())
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.6 * len(table) + 1.5))
    table.plot.barh(stacked=True, ax=ax, color=_ERROR_SHARE_COLORS, width=0.7)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = [
                f"{bar.get_width():.2f}" if bar.get_width() >= _MIN_LABELLED_SHARE else ""
                for bar in container
            ]
            ax.bar_label(container, labels=labels, label_type="center", fontsize="small")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Anteil der Zeilen")
    ax.legend(
        fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(table.columns)
    )
    ax.invert_yaxis()
    return ax


def plot_recall_comparison(
    recalls: Mapping[str, pd.DataFrame], top: int | None = None
) -> plt.Figure:
    """Compare the recall per combined crop/stage class of several models.

    Args:
        recalls: ``recall_by_class()`` per model, keyed by the label to show; all for the same
            samples.
        top: Show only the classes where the models differ most (recommended with many
            classes); all classes if omitted.

    Returns:
        Figure with one group of horizontal bars per class (support in the label), classes
            ordered by the largest recall difference between the models.

    Raises:
        ValueError: If the tables do not cover the same samples or ``top`` is not positive.
    """
    if top is not None and top < 1:
        raise ValueError(f"top must be positive, got {top}.")
    supports = [recall[SUPPORT_COL].sort_index() for recall in recalls.values()]
    if not all(support.equals(supports[0]) for support in supports):
        raise ValueError("All recall tables must come from the same samples.")
    support = supports[0]
    table = pd.DataFrame({label: recall[RECALL_COL] for label, recall in recalls.items()})
    spread = table.max(axis=1) - table.min(axis=1)
    table = table.loc[spread.sort_values(ascending=False).index[:top]].iloc[::-1]
    table.index = [f"{pair} (n={support[pair]})" for pair in table.index]
    bar_height = 0.225
    fig, ax = plt.subplots(figsize=(9, bar_height * len(table) * len(recalls) + 1.5))
    table.plot.barh(ax=ax, width=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_rate_by(
    errors: Mapping[str, pd.Series],
    groups: pd.Series,
    xlabel: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare how the error rate of several models changes over a context such as month or AEZ.

    Args:
        errors: Per model a boolean Series (``True`` = wrong), aligned with ``groups``.
        groups: Context value per sample, e.g. ``X_val["Month"]``.
        xlabel: Label of the x axis; the name of ``groups`` if omitted.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one group of bars per context value, its sample count in the tick label.
    """
    table = pd.DataFrame({label: error.groupby(groups).mean() for label, error in errors.items()})
    counts = groups.value_counts()
    table.index = [f"{value}\n(n={counts[value]})" for value in table.index]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.9 * len(table) + 2), 4))
    table.plot.bar(ax=ax, rot=0, width=0.8)
    ax.set_xlabel(xlabel or str(groups.name))
    ax.set_ylabel("Fehlerrate")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return ax
