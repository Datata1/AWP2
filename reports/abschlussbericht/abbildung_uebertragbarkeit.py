"""Draw the combined CV BAcc of random vs. group folds for the final report (80/20 protocol)."""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt

from awp2.config import FIGURE_DPI, PROJECT_ROOT

OUTPUT = PROJECT_ROOT / "reports" / "abschlussbericht" / "img" / "uebertragbarkeit.png"
FIGSIZE = (6.4, 3.0)
RANDOM_COLOR = "tab:blue"
GROUP_COLOR = "tab:red"


class FoldScheme(NamedTuple):
    """Combined CV BAcc of the combined SVM on all bands, from docs/modelle/experimente.md."""

    label: str
    bacc: float
    held_out_group: bool


# Runs c0417d33, 482cd374, cb600d5c and 1f6323a2 (experiment split-leakage, #170).
SCHEMES = (
    FoldScheme("zufällige Folds", 0.859, False),
    FoldScheme("neue Aufnahme\n(AEZ × Monat)", 0.378, True),
    FoldScheme("neuer Monat", 0.421, True),
    FoldScheme("neue Region (AEZ)", 0.284, True),
)


def main(output: Path = OUTPUT) -> None:
    """Create the figure.

    Args:
        output: Target PNG path.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    positions = range(len(SCHEMES))
    bars = ax.barh(
        positions,
        [scheme.bacc for scheme in SCHEMES],
        color=[GROUP_COLOR if scheme.held_out_group else RANDOM_COLOR for scheme in SCHEMES],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_yticks(positions, [scheme.label for scheme in SCHEMES])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced Accuracy kombiniert (CV)")
    ax.grid(axis="x", alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")


if __name__ == "__main__":
    main()
