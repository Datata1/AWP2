"""Hierarchical classifiers with the hyperparameters found by ``tune()`` (#87, #98–#103).

The values are results of the logged searches (the ``tuning_run`` of each factory); keeping them
in code lets every notebook rebuild the same models without access to the local MLflow store.
Re-tuning a classifier means updating its factory here.
"""

from collections.abc import Callable, Mapping
from typing import NamedTuple

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from awp2.config import (
    EXTRA_TREES_N_ESTIMATORS,
    LOGISTIC_REGRESSION_MAX_ITER,
    MLP_MAX_ITER,
    SEED,
)
from awp2.experiment import RunConfig, RunResult, run
from awp2.models.hierarchical import HierarchicalClassifier
from awp2.models.linear import BalancedLinearDiscriminantAnalysis
from awp2.models.plsda import PLSDAClassifier
from awp2.preprocessing import PreprocessingConfig


class TunedHierarchy(NamedTuple):
    """A tuned hierarchical classifier and the settings it was tuned with.

    Attributes:
        model: Unfitted classifier with the best hyperparameters.
        preprocessing: Preprocessing used in tuning and evaluation.
        balance_samples: Whether balanced sample weights were passed to ``fit``.
        tuning_run: MLflow run id (first 8 characters) of the search the values come from.
        cv_score: Mean cross-validation ``bacc_combined`` of these hyperparameters in
            ``tuning_run``.
    """

    model: HierarchicalClassifier
    preprocessing: PreprocessingConfig
    balance_samples: bool
    tuning_run: str
    cv_score: float


def _extra_trees(**params: float | int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=EXTRA_TREES_N_ESTIMATORS,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )


def _standardised() -> PreprocessingConfig:
    return PreprocessingConfig(use_spectral_standard_scale=True)


def tuned_hierarchical_svm() -> TunedHierarchy:
    """RBF-SVM on both levels – the reference of the hierarchical approach (#87).

    Returns:
        The tuned classifier and its settings.
    """
    return TunedHierarchy(
        model=HierarchicalClassifier(
            SVC(class_weight="balanced", C=30.0, gamma=0.01),
            SVC(class_weight="balanced", C=100.0, gamma=0.003),
        ),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="2b851b88",
        cv_score=0.83934,
    )


def tuned_hierarchical_extra_trees_svm() -> TunedHierarchy:
    """Extra Trees for the crop, RBF-SVM per crop for the stage (#87).

    Returns:
        The tuned classifier and its settings.
    """
    crop = _extra_trees(max_features=0.3)
    return TunedHierarchy(
        model=HierarchicalClassifier(crop, SVC(class_weight="balanced", C=100.0, gamma=0.003)),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="c3f13a6e",
        cv_score=0.81108,
    )


def tuned_hierarchical_extra_trees() -> TunedHierarchy:
    """Extra Trees on both levels (#87).

    Returns:
        The tuned classifier and its settings.
    """
    return TunedHierarchy(
        model=HierarchicalClassifier(
            _extra_trees(max_features=0.3),
            _extra_trees(max_features=0.3, min_samples_leaf=3),
        ),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="95c88504",
        cv_score=0.79728,
    )


def tuned_hierarchical_mlp() -> TunedHierarchy:
    """MLP on both levels with early stopping and balanced sample weights (#102).

    Returns:
        The tuned classifier and its settings.
    """

    def mlp(alpha: float) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=alpha,
            early_stopping=True,
            max_iter=MLP_MAX_ITER,
            random_state=SEED,
        )

    return TunedHierarchy(
        model=HierarchicalClassifier(mlp(0.01), mlp(0.1)),
        preprocessing=_standardised(),
        balance_samples=True,
        tuning_run="777c0509",
        cv_score=0.789,
    )


def tuned_hierarchical_hist_gradient_boosting() -> TunedHierarchy:
    """HistGradientBoosting on both levels with balanced sample weights (#103).

    Returns:
        The tuned classifier and its settings.
    """

    def boosting() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(learning_rate=0.05, random_state=SEED)

    return TunedHierarchy(
        model=HierarchicalClassifier(boosting(), boosting()),
        preprocessing=_standardised(),
        balance_samples=True,
        tuning_run="3aa6e548",
        cv_score=0.78312,
    )


def tuned_hierarchical_logistic_regression() -> TunedHierarchy:
    """Logistic regression on both levels with balanced class weights (#99).

    Returns:
        The tuned classifier and its settings.
    """

    def logistic_regression(strength: float) -> LogisticRegression:
        return LogisticRegression(
            class_weight="balanced", C=strength, max_iter=LOGISTIC_REGRESSION_MAX_ITER
        )

    return TunedHierarchy(
        model=HierarchicalClassifier(logistic_regression(10.0), logistic_regression(100.0)),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="f94ee4d2",
        cv_score=0.74692,
    )


def tuned_hierarchical_knn() -> TunedHierarchy:
    """One nearest neighbour by cosine distance on both levels (#101).

    Returns:
        The tuned classifier and its settings.
    """

    def knn() -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=1, metric="cosine", weights="distance")

    return TunedHierarchy(
        model=HierarchicalClassifier(knn(), knn()),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="49726021",
        cv_score=0.7417,
    )


def tuned_hierarchical_plsda() -> TunedHierarchy:
    """PLS-DA on both levels (#100).

    Returns:
        The tuned classifier and its settings.
    """
    return TunedHierarchy(
        model=HierarchicalClassifier(PLSDAClassifier(120), PLSDAClassifier(20)),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="679e9f30",
        cv_score=0.67768,
    )


def tuned_hierarchical_lda() -> TunedHierarchy:
    """LDA with equal priors on both levels (#98).

    Returns:
        The tuned classifier and its settings.
    """
    return TunedHierarchy(
        model=HierarchicalClassifier(
            BalancedLinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            BalancedLinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.01),
        ),
        preprocessing=_standardised(),
        balance_samples=False,
        tuning_run="35a238ff",
        cv_score=0.6571,
    )


TUNED_HIERARCHIES: Mapping[str, Callable[[], TunedHierarchy]] = {
    "svm": tuned_hierarchical_svm,
    "extra_trees_svm": tuned_hierarchical_extra_trees_svm,
    "extra_trees": tuned_hierarchical_extra_trees,
    "mlp": tuned_hierarchical_mlp,
    "hist_gradient_boosting": tuned_hierarchical_hist_gradient_boosting,
    "logistic_regression": tuned_hierarchical_logistic_regression,
    "knn": tuned_hierarchical_knn,
    "plsda": tuned_hierarchical_plsda,
    "lda": tuned_hierarchical_lda,
}
"""All tuned hierarchies by short name, best by CV first."""


def refit_untracked(tuned: TunedHierarchy, name: str) -> RunResult:
    """Fit a tuned hierarchy on the training part and predict the validation part, without MLflow.

    For comparisons and analysis only – the logged evaluation run of the model stays the
    reference for its scores.

    Args:
        tuned: Classifier and settings, e.g. ``tuned_hierarchical_svm()``.
        name: Short name, used for warnings only.

    Returns:
        Metrics, fitted pipeline and validation predictions (``run_id`` is ``None``).
    """
    return run(
        tuned.model,
        RunConfig(
            name=name,
            approach="hierarchical",
            preprocessing=tuned.preprocessing,
            balance_samples=tuned.balance_samples,
            track=False,
            log_model=False,
        ),
    )
