"""Plots of the data-reduction studies: trade-off, PCA curve, loadings, timings."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from awp2.config import (
    BAND_GAP_NM,
    FIT_TIME_COL,
    MIN_CV_GAIN,
    N_COMPONENTS_COL,
    N_MEASURED_BANDS_COL,
    PREDICT_TIME_COL,
    REDUCTION_FAMILY_COL,
    REDUCTION_FAMILY_LABELS,
    REDUCTION_MAX_CV_LOSS,
    REDUCTION_REFERENCE_FAMILY,
    STD_SUFFIX,
)
from awp2.data import wavelength
from awp2.evaluation import METRIC_LABELS, MetricName
from awp2.plots.common import _shade_regions


def plot_tradeoff(
    curve: pd.DataFrame,
    metric: MetricName = "bacc_combined",
    x: str = N_MEASURED_BANDS_COL,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot quality against input size, one line per reduction family.

    Args:
        curve: Output of ``awp2.reduction.reduction_curve``.
        metric: Score on the y-axis, drawn as mean ± std over the folds.
        x: ``"n_measured_bands"`` (what a sensor must measure) or ``"n_input_features"``
            (what the model receives).
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes; a dashed line marks the reference with all bands.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    is_reference = curve[REDUCTION_FAMILY_COL] == REDUCTION_REFERENCE_FAMILY
    # the reference shares its band count with every PCA variant, so it goes on top
    for family, rows in curve[~is_reference].groupby(REDUCTION_FAMILY_COL, sort=False):
        rows = rows.sort_values(x)
        ax.errorbar(
            rows[x],
            rows[metric],
            yerr=rows[f"{metric}{STD_SUFFIX}"],
            marker="o",
            capsize=3,
            alpha=0.8,
            label=REDUCTION_FAMILY_LABELS.get(str(family), str(family)),
        )
    reference_rows = curve[is_reference]
    ax.errorbar(
        reference_rows[x],
        reference_rows[metric],
        yerr=reference_rows[f"{metric}{STD_SUFFIX}"],
        marker="D",
        markersize=8,
        color="black",
        capsize=3,
        zorder=5,
        label=REDUCTION_FAMILY_LABELS[REDUCTION_REFERENCE_FAMILY],
    )
    reference = reference_rows[metric]
    if not reference.empty:
        ax.axhline(reference.iloc[0], color="grey", linestyle="--", linewidth=1)
    # metadata-only has zero bands, which a plain log axis cannot show
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, curve[x].max() * 1.5)
    ax.set_xlabel("Gemessene Bänder" if x == N_MEASURED_BANDS_COL else "Merkmale im Modell")
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (CV)")
    ax.legend(fontsize="small", loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def plot_pca_curve(
    curves: Mapping[str, pd.DataFrame],
    explained_variance_ratio: pd.Series | None = None,
    metric: MetricName = "bacc_combined",
    relative: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot quality against the number of principal components, one line per model.

    The first row of every curve is its reference without PCA: drawn as a dashed line with a
    band of ± ``MIN_CV_GAIN`` (the smallest gain the project counts) and as an open marker at
    its band count; with ``relative`` one shared reference at 0.

    Args:
        curves: Output of ``awp2.reduction.pca_curve`` per model label.
        explained_variance_ratio: Optional variance share per component; its cumulative sum
            is drawn on a second y-axis.
        metric: Score on the y-axis, drawn as mean with ± std over the folds as a band.
        relative: Plot the difference to each reference instead of the score, so models of
            different quality share one scale; adds the loss limit ``REDUCTION_MAX_CV_LOSS``.
            All curves must then start from references with the same number of bands.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with the scores (the variance axis is its twin).

    Raises:
        ValueError: If ``relative`` and the references differ in their number of bands.
    """
    if relative and len({curve[N_COMPONENTS_COL].iloc[0] for curve in curves.values()}) > 1:
        raise ValueError("relative curves need references with the same number of bands")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for number, (label, curve) in enumerate(curves.items()):
        color = f"C{number}"
        reference, pca = curve.iloc[0], curve.iloc[1:]
        offset = reference[metric] if relative else 0.0
        mean = pca[metric] - offset
        std = pca[f"{metric}{STD_SUFFIX}"]
        ax.plot(pca[N_COMPONENTS_COL], mean, marker="o", color=color, label=label)
        ax.fill_between(pca[N_COMPONENTS_COL], mean - std, mean + std, color=color, alpha=0.1)
        if not relative:
            _reference_level(ax, reference[metric], reference[N_COMPONENTS_COL], color)
    if relative:
        # every curve is shifted onto its own reference, so one shared line at 0 suffices
        first = next(iter(curves.values())).iloc[0]
        _reference_level(ax, 0.0, first[N_COMPONENTS_COL], "black")
        ax.axhline(-REDUCTION_MAX_CV_LOSS, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Hauptkomponenten (offener Punkt: ohne PCA)")
    label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(f"Δ {label} zur Referenz (CV)" if relative else f"{label} (CV)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small", loc="lower right")
    if explained_variance_ratio is not None:
        cumulative = explained_variance_ratio.cumsum().to_numpy() * 100
        variance_ax = ax.twinx()
        variance_ax.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            color="grey",
            drawstyle="steps-post",
            linewidth=1,
        )
        variance_ax.set_ylabel("Kumulierte erklärte Varianz [%] (grau)")
        variance_ax.set_ylim(top=100.5)
    return ax


def plot_fold_timings(curves: Mapping[str, pd.DataFrame]) -> plt.Figure:
    """Plot fit and prediction time per fold against the number of principal components.

    Args:
        curves: Output of ``awp2.reduction.pca_curve`` per model label; the first row (no PCA)
            is drawn as a dashed line.

    Returns:
        Figure with one panel for training and one for prediction, both log-log.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    for column, ax, title in zip(
        (FIT_TIME_COL, PREDICT_TIME_COL), axes, ("Training", "Vorhersage"), strict=True
    ):
        for number, (label, curve) in enumerate(curves.items()):
            reference, pca = curve.iloc[0], curve.iloc[1:]
            color = f"C{number}"
            ax.plot(pca[N_COMPONENTS_COL], pca[column], marker="o", color=color, label=label)
            ax.axhline(reference[column], color=color, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Hauptkomponenten (gestrichelt: ohne PCA)")
        ax.set_ylabel("Sekunden je Fold")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize="small")
    fig.tight_layout()
    return fig


def _reference_level(ax: plt.Axes, level: float, n_bands: float, color: str) -> None:
    ax.axhline(level, color=color, linestyle="--", linewidth=1)
    ax.axhspan(level - MIN_CV_GAIN, level + MIN_CV_GAIN, color=color, alpha=0.08)
    ax.plot(n_bands, level, marker="o", mfc="white", color=color)


def plot_pca_loadings(
    loadings: pd.DataFrame,
    explained_variance_ratio: pd.Series,
    n_components: int = 4,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the loadings of the first components over the wavelength, spectral regions shaded.

    Args:
        loadings: From ``awp2.reduction.spectral_pca``: one row per band (``X<nm>``), one
            column per component.
        explained_variance_ratio: Variance share per component, shown in the legend.
        n_components: How many leading components to draw.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one curve per component; removed bands stay gaps.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    x = np.array([wavelength(band) for band in loadings.index], dtype=float)
    breaks = np.flatnonzero(np.diff(x) > BAND_GAP_NM) + 1
    x_plot = np.insert(x, breaks, np.nan)
    for component in loadings.columns[:n_components]:
        values = np.insert(loadings[component].to_numpy(dtype=float), breaks, np.nan)
        share = explained_variance_ratio[component]
        ax.plot(x_plot, values, marker=".", linewidth=1.5, label=f"{component} ({share:.1%})")
    _shade_regions(ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Wellenlänge [nm]")
    ax.set_ylabel("Ladung")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return ax
