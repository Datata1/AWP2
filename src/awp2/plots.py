"""Shared plotting helpers for EDA, model analysis and reports."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import BarContainer

from awp2.config import DOCS_FIGURES_DIR, FIGURE_DPI, TARGET_COLS
from awp2.data import band_columns, wavelengths
from awp2.evaluation import METRIC_LABELS, MetricName, Metrics


def plot_spectra(
    df: pd.DataFrame,
    by: str | None = None,
    std: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot the mean reflectance spectrum (± one standard deviation) per group.

    Bands that are entirely missing show up as gaps in the curve.

    Args:
        df: Data with band columns, e.g. from ``load_train()``.
        by: Column to group by (e.g. ``"Crop"``); ``None`` plots all rows as one curve.
        std: Draw a shaded ± one standard deviation band.
        ax: Axes to draw on; a new figure is created if omitted.
        title: Optional plot title.

    Returns:
        The axes the spectra were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    bands = band_columns(df)
    x = wavelengths(df)
    groups = df.groupby(by, observed=True) if by else [("all", df)]

    for name, group in groups:
        mean = group[bands].mean()
        (line,) = ax.plot(x, mean.values, label=f"{name} (n={len(group)})", linewidth=1.5)
        if std:
            sd = group[bands].std()
            ax.fill_between(
                x, (mean - sd).values, (mean + sd).values, color=line.get_color(), alpha=0.15
            )

    ax.set_xlabel("Wellenlänge [nm]")
    ax.set_ylabel("Reflektanz [%]")
    if title:
        ax.set_title(title)
    if by:
        ax.legend(title=by, fontsize="small")
    ax.grid(alpha=0.3)
    return ax


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


def save_doc_figure(fig: plt.Figure, name: str, directory: Path = DOCS_FIGURES_DIR) -> None:
    """Save a figure as PNG for the documentation (the folder is versioned in git).

    Args:
        fig: Figure to save.
        name: File name without extension.
        directory: ``DOCS_FIGURES_DIR`` (data/EDA pages) or ``MODEL_DOCS_FIGURES_DIR`` (models).
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
