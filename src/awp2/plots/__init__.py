"""Shared plotting helpers for EDA, model analysis and reports."""

from awp2.plots.common import save_doc_figure
from awp2.plots.data import plot_spectra
from awp2.plots.evaluation import (
    plot_class_recall,
    plot_cv_scores,
    plot_error_breakdown,
    plot_error_rate_by,
    plot_metric_comparison,
    plot_recall_comparison,
    plot_top_confusions,
    plot_tuning_candidates,
)
from awp2.plots.importance import (
    plot_group_against_null,
    plot_group_study,
    plot_importance_heatmap,
    plot_importance_spectrum,
)
from awp2.plots.reduction import plot_fold_timings, plot_pca_curve, plot_pca_loadings, plot_tradeoff

__all__ = [
    "plot_class_recall",
    "plot_cv_scores",
    "plot_error_breakdown",
    "plot_error_rate_by",
    "plot_fold_timings",
    "plot_group_against_null",
    "plot_group_study",
    "plot_importance_heatmap",
    "plot_importance_spectrum",
    "plot_metric_comparison",
    "plot_pca_curve",
    "plot_pca_loadings",
    "plot_recall_comparison",
    "plot_spectra",
    "plot_top_confusions",
    "plot_tradeoff",
    "plot_tuning_candidates",
    "save_doc_figure",
]
