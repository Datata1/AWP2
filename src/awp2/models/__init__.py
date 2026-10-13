"""Reusable modelling approaches for crop and growth-stage classification."""

from awp2.models.cnn import SpectralCNNClassifier, make_combined_spectral_cnn
from awp2.models.combined import (
    CombinedLabelClassifier,
    make_combined_extra_trees,
    make_combined_hist_gradient_boosting,
    make_combined_random_forest,
    make_combined_svm,
)
from awp2.models.hierarchical import (
    HierarchicalClassifier,
    make_hierarchical_extra_trees,
    make_hierarchical_extra_trees_svm,
    make_hierarchical_svm,
    oracle_crop_metrics,
)
from awp2.models.independent import IndependentClassifier, make_independent_svm
from awp2.models.linear import BalancedLinearDiscriminantAnalysis
from awp2.models.multi_task import MultiTaskMLPClassifier
from awp2.models.plsda import PLSDAClassifier

__all__ = [
    "BalancedLinearDiscriminantAnalysis",
    "PLSDAClassifier",
    "CombinedLabelClassifier",
    "IndependentClassifier",
    "MultiTaskMLPClassifier",
    "HierarchicalClassifier",
    "SpectralCNNClassifier",
    "make_combined_extra_trees",
    "make_combined_hist_gradient_boosting",
    "make_combined_random_forest",
    "make_combined_spectral_cnn",
    "make_combined_svm",
    "make_hierarchical_extra_trees",
    "make_hierarchical_extra_trees_svm",
    "make_hierarchical_svm",
    "make_independent_svm",
    "oracle_crop_metrics",
]
