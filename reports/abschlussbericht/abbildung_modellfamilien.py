"""Draw the CV comparison of the AutoML model families for the final report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from awp2.config import FIGURE_DPI, PROJECT_ROOT

CANDIDATES_CSV = PROJECT_ROOT / "models" / "automl_final" / "provenance" / "candidates.csv"
OUTPUT = PROJECT_ROOT / "reports" / "abschlussbericht" / "img" / "modellfamilien_cv.png"
LABELS = {
    "combined_svm": "RBF-SVM",
    "combined_et": "Extra Trees",
    "combined_rf": "Random Forest",
    "combined_hgb": "HistGradientBoosting",
    "hierarchical_svm_et": "SVM \u2192 Extra Trees",
    "hierarchical_svm_svm": "SVM \u2192 SVM",
    "hierarchical_svm_lda": "SVM \u2192 LDA",
    "hierarchical_et_hgb": "Extra Trees \u2192 HGB",
}
APPROACH_COLORS = {"Kombinierte Klasse": "tab:blue", "Hierarchisch": "tab:orange"}
FIGSIZE = (6.4, 3.0)


def best_per_configuration(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the best candidate (by CV) of every model configuration.

    Args:
        candidates: AutoML candidate table with ``cv_bacc_combined`` and ``details_json``.

    Returns:
        One row per configuration with label, approach, CV mean and fold standard deviation.
    """
    details = candidates["details_json"].apply(json.loads)
    table = candidates.assign(
        name=details.apply(lambda entry: entry["name"]),
        std=details.apply(lambda entry: entry["std"]),
    )
    best = table.loc[table.groupby("name")["cv_bacc_combined"].idxmax()].copy()
    best["label"] = best["name"].map(LABELS)
    best["approach"] = best["name"].map(
        lambda name: "Hierarchisch" if name.startswith("hierarchical") else "Kombinierte Klasse"
    )
    return best.sort_values("cv_bacc_combined")


def main(output: Path = OUTPUT) -> None:
    """Create the figure.

    Args:
        output: Target PNG path.
    """
    best = best_per_configuration(pd.read_csv(CANDIDATES_CSV))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for position, row in enumerate(best.itertuples()):
        ax.errorbar(
            row.cv_bacc_combined,
            position,
            xerr=row.std,
            fmt="o",
            color=APPROACH_COLORS[row.approach],
            capsize=3,
        )
        ax.text(row.cv_bacc_combined, position + 0.28, f"{row.cv_bacc_combined:.3f}", ha="center")
    ax.set_yticks(range(len(best)), best["label"])
    ax.set_xlabel("CV-BAcc kombiniert (Mittel \u00b1 Std. \u00fcber 5 Folds)")
    ax.grid(axis="x", alpha=0.3)
    handles = [
        ax.plot([], [], "o", color=color, label=approach)[0]
        for approach, color in APPROACH_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")


if __name__ == "__main__":
    main()
