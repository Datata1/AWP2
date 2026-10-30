"""Draw CV and holdout BAcc of the main candidates for the final report (70/30 protocol)."""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt

from awp2.config import FIGURE_DPI, PROJECT_ROOT

OUTPUT = PROJECT_ROOT / "reports" / "abschlussbericht" / "img" / "modellvergleich.png"
FIGSIZE = (6.4, 3.4)
CV_COLOR = "tab:blue"
HOLDOUT_COLOR = "tab:orange"


class Candidate(NamedTuple):
    """Combined BAcc of one model, copied from docs/modelle/experimente.md."""

    label: str
    cv: float
    holdout: float


# Values of the 70/30 runs in docs/modelle/experimente.md (CV of the tuning run, holdout of run()).
CANDIDATES = (
    Candidate("Baseline: Random Forest", 0.750, 0.743),
    Candidate("Kombiniert: Random Forest", 0.788, 0.779),
    Candidate("Kombiniert: Extra Trees", 0.812, 0.804),
    Candidate("Kombiniert: HistGradientBoosting", 0.781, 0.777),
    Candidate("Kombiniert: RBF-SVM", 0.847, 0.866),
    Candidate("Hierarchisch: SVM \u2192 SVM", 0.839, 0.844),
    Candidate("Multi-Task-MLP (PCA-20)", 0.728, 0.765),
    Candidate("Kombiniert: LSTM", 0.711, 0.700),
    Candidate("Kombiniert: 1D-CNN", 0.611, 0.626),
)


def main(output: Path = OUTPUT) -> None:
    """Create the figure.

    Args:
        output: Target PNG path.
    """
    ranked = sorted(CANDIDATES, key=lambda candidate: candidate.holdout)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    positions = range(len(ranked))
    ax.scatter([c.cv for c in ranked], positions, color=CV_COLOR, label="CV (Trainingsfolds)")
    ax.scatter([c.holdout for c in ranked], positions, color=HOLDOUT_COLOR, label="Holdout")
    for position, candidate in zip(positions, ranked, strict=True):
        ax.plot(
            [candidate.cv, candidate.holdout], [position, position], color="grey", linewidth=0.8
        )
    ax.set_yticks(positions, [c.label for c in ranked])
    ax.set_xlabel("Balanced Accuracy kombiniert")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")


if __name__ == "__main__":
    main()
