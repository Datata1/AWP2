"""Plots of the raw data: spectra and label counts by group."""

import matplotlib.pyplot as plt
import pandas as pd

from awp2.config import CROP_COL, CROPS, MONTH_COL, STAGE_COL, STAGES
from awp2.data import band_columns, wavelengths


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


def plot_stage_by_month(df: pd.DataFrame) -> plt.Figure:
    """Stacked stage counts per month, one panel per crop, on a shared month axis.

    Every panel shows all months that occur in ``df`` (empty months stay as gaps) and every
    stage keeps the same colour, so panels can be compared directly.

    Args:
        df: Labelled data with ``Crop``, ``Stage`` and ``Month``, e.g. from ``load_train()``.

    Returns:
        The figure with one panel per crop; unused panels are hidden.
    """
    months = sorted(df[MONTH_COL].unique())
    palette = plt.get_cmap("tab10")
    colors = {stage: palette(position) for position, stage in enumerate(STAGES)}
    n_cols = 3
    n_rows = -(-len(CROPS) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=True, sharey=True)
    for ax, crop in zip(axes.flat, CROPS, strict=False):
        counts = pd.crosstab(
            df.loc[df[CROP_COL] == crop, MONTH_COL], df.loc[df[CROP_COL] == crop, STAGE_COL]
        ).reindex(index=months, columns=list(STAGES), fill_value=0)
        bottom = pd.Series(0, index=months)
        for stage in STAGES:
            ax.bar(counts.index, counts[stage], bottom=bottom, color=colors[stage], label=stage)
            bottom = bottom + counts[stage]
        ax.set_title(crop)
        ax.set_xticks(months)
        ax.grid(axis="y", alpha=0.3)
    for position, ax in enumerate(axes.flat):
        if position >= len(CROPS):
            ax.set_visible(False)
        elif position + n_cols >= len(CROPS):
            # the panel below is hidden, so this panel carries the shared month axis
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("Monat")
    for ax in axes[:, 0]:
        ax.set_ylabel("Anzahl")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=STAGE_COL, loc="lower right", bbox_to_anchor=(0.97, 0.12))
    fig.suptitle("Stadien je Monat innerhalb jeder Kultur")
    fig.tight_layout()
    return fig
