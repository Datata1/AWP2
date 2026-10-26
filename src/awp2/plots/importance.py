"""Plots of the feature-importance analyses: spectrum, heatmap, group studies."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from awp2.config import (
    ALL_GROUPS_KIND,
    BAND_GAP_NM,
    META_ONLY_KIND,
    ONLY_GROUP_KIND,
    ONLY_GROUP_NO_META_KIND,
    SPECTRAL_REGIONS,
    WITHOUT_GROUP_KIND,
    SpectralRegion,
)
from awp2.data import wavelength
from awp2.evaluation import METRIC_LABELS


def plot_group_study(
    table: pd.DataFrame,
    metric: str = "bacc_combined",
    regions: Sequence[SpectralRegion] = SPECTRAL_REGIONS,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare retrained models that see only one region, or all but one, per region.

    Args:
        table: Output of :func:`awp2.group_study.results_table`.
        metric: Cross-validated metric to plot; its ``<metric>_std`` column gives the error bars.
        regions: Regions in plotting order.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with three bars per region and reference lines for the full model, the model
            without metadata and the metadata alone.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.8))
    names = [region.name for region in regions]
    series = (
        (ONLY_GROUP_NO_META_KIND, "nur diese Region (ohne Metadaten)"),
        (ONLY_GROUP_KIND, "nur diese Region + Metadaten"),
        (WITHOUT_GROUP_KIND, "alle anderen Regionen + Metadaten"),
    )
    width = 0.8 / len(series)
    positions = np.arange(len(names))
    for number, (kind, label) in enumerate(series):
        rows = table[table["kind"] == kind].set_index("group").loc[names]
        ax.bar(
            positions + (number - 1) * width,
            rows[metric],
            width,
            yerr=rows[f"{metric}_std"],
            capsize=2,
            label=label,
        )
    references = (
        (table[(table["kind"] == ALL_GROUPS_KIND) & table["use_meta"]], "alle Bänder + Metadaten"),
        (
            table[(table["kind"] == ALL_GROUPS_KIND) & ~table["use_meta"]],
            "alle Bänder, keine Metadaten",
        ),
        (table[table["kind"] == META_ONLY_KIND], "nur Metadaten"),
    )
    for number, (rows, label) in enumerate(references):
        ax.axhline(
            rows[metric].iloc[0],
            color=f"C{3 + number}",
            linestyle=("-", "--", ":")[number],
            linewidth=1.2,
            label=label,
        )
    ax.set_xticks(positions, names)
    ax.set_ylabel(f"CV-{METRIC_LABELS[metric]}")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize="x-small", ncols=3, loc="upper center")
    return ax


def plot_group_against_null(
    observed: pd.Series, null: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """Compare the importance of each region with random band groups of the same size.

    Args:
        observed: Importance per region (index = region name).
        null: Importance of random groups of the region's size, one column per region, one row
            per random group.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with the null distribution as box plot and the region as red dot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    names = list(observed.index)
    ax.boxplot([null[name].dropna() for name in names], tick_labels=names, showfliers=False)
    ax.scatter(range(1, len(names) + 1), observed[names], color="C3", zorder=3, label="Region")
    ax.set_ylabel("Abfall der BAcc beim Mischen")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize="small")
    return ax


def plot_importance_spectrum(
    importance: pd.Series,
    std: pd.Series | None = None,
    regions: Sequence[SpectralRegion] = SPECTRAL_REGIONS,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot the importance of every band over the wavelength with the spectral regions behind it.

    Args:
        importance: Importance per band, indexed by band name (``X<nm>``) – bands only.
        std: Spread per band, drawn as a shaded band; omitted if ``None``.
        regions: Spectral regions to shade and label.
        ax: Axes to draw on; a new figure is created if omitted.
        title: Optional plot title.

    Returns:
        The axes the importance was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    x = [wavelength(band) for band in importance.index]
    after_gap = np.flatnonzero(np.diff(x) > BAND_GAP_NM) + 1

    def with_breaks(values: np.ndarray) -> np.ndarray:
        return np.insert(values.astype(float), after_gap, np.nan)

    x_plot = with_breaks(np.asarray(x))
    for number, region in enumerate(regions):
        ax.axvspan(region.start_nm, region.end_nm, color="0.93" if number % 2 else "1", zorder=0)
        ax.text(
            (region.start_nm + region.end_nm) / 2,
            1.01 + 0.05 * (number % 2),
            region.name,
            ha="center",
            fontsize="small",
            transform=ax.get_xaxis_transform(),
        )
    ax.plot(
        x_plot,
        with_breaks(importance.to_numpy()),
        marker="o",
        markersize=3,
        linewidth=1,
        color="C0",
    )
    if std is not None:
        ax.fill_between(
            x_plot,
            with_breaks((importance - std).to_numpy()),
            with_breaks((importance + std).to_numpy()),
            color="C0",
            alpha=0.2,
        )
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlim(min(x), max(x))
    ax.set_xlabel("Wellenlänge [nm]")
    ax.set_ylabel("Abfall der BAcc beim Mischen")
    if title:
        ax.set_title(title, pad=34)
    ax.grid(alpha=0.3)
    return ax


def plot_importance_heatmap(
    importance: pd.DataFrame, title: str | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """Show the importance of each feature group (rows) for each class (columns) as a heatmap.

    Args:
        importance: Drop in recall per feature group (index) and class (columns).
        title: Optional plot title.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with the annotated heatmap.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(1.1 * len(importance.columns) + 2, 0.55 * len(importance) + 1)
        )
    image = ax.imshow(importance.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(importance.columns)), importance.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(importance)), importance.index)
    for row, values in enumerate(importance.to_numpy()):
        for column, value in enumerate(values):
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.figure.colorbar(image, ax=ax, label="Abfall des Recalls")
    if title:
        ax.set_title(title)
    return ax
