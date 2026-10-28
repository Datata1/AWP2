"""Figure helpers shared by all plot modules."""

from pathlib import Path

import matplotlib.pyplot as plt

from awp2.config import (
    DOCS_FIGURES_DIR,
    FIGURE_DPI,
    SPECTRAL_REGIONS_NM,
)


def save_doc_figure(fig: plt.Figure, name: str, directory: Path = DOCS_FIGURES_DIR) -> None:
    """Save a figure as PNG for the documentation (the folder is versioned in git).

    Args:
        fig: Figure to save.
        name: File name without extension.
        directory: ``DOCS_FIGURES_DIR`` (data/EDA pages) or ``MODEL_DOCS_FIGURES_DIR`` (models).
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")


def _shade_regions(ax: plt.Axes) -> None:
    for position, (region, (low, high)) in enumerate(SPECTRAL_REGIONS_NM.items()):
        ax.axvspan(low, high, color="grey", alpha=0.08 if position % 2 else 0.16, linewidth=0)
        ax.text(
            (low + high) / 2,
            1.0,
            region,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize="small",
        )
