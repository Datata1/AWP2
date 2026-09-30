"""Shared plotting helpers for EDA and reports."""

import matplotlib.pyplot as plt
import pandas as pd

from awp2.config import DOCS_FIGURES_DIR, FIGURE_DPI
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


def save_doc_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG to ``docs/daten/img/<name>.png`` for use in the documentation."""
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
