"""Ensembles of crop/stage classifiers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    EXTRA_TREES_N_ESTIMATORS,
    LABEL_SEP,
    LOGISTIC_REGRESSION_MAX_ITER,
    SEED,
    STACKING_INNER_FOLDS,
    STACKING_MEMBERS,
    TARGET_COLS,
    StackMember,
)
from awp2.data import combined_label
from awp2.evaluation import as_target_frame
from awp2.models.combined import CombinedLabelClassifier


class CropStageVotingClassifier(ClassifierMixin, BaseEstimator):
    """Weighted majority vote over the crop/stage pairs predicted by several classifiers.

    Every member predicts a full ``Crop``/``Stage`` pair; the pair with the largest summed weight
    wins. A tie goes to the earliest member whose pair reaches the top score. ``first_weight``
    lets a strong first member (e.g. the reference model) be overruled only when enough others
    agree. Only pairs some member
    predicted can win, so members that predict valid pairs only (e.g. hierarchical classifiers)
    keep that guarantee.

    Args:
        estimators: ``(name, unfitted classifier)`` pairs; each predicts ``Crop`` and ``Stage``.
        first_weight: Vote weight of the first member; every other member counts 1.
    """

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        first_weight: float = 1.0,
    ) -> None:
        """Initialise the ensemble.

        Args:
            estimators: ``(name, unfitted classifier)`` pairs.
            first_weight: Vote weight of the first member.
        """
        self.estimators = estimators
        self.first_weight = first_weight

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "CropStageVotingClassifier":
        """Fit every member on the same data.

        Args:
            X: Model features.
            y: Targets with the columns ``Crop`` and ``Stage``.

        Returns:
            This fitted classifier.

        Raises:
            ValueError: If member names repeat or ``first_weight`` is not positive.
        """
        names = [name for name, _ in self.estimators]
        if len(set(names)) != len(names):
            raise ValueError(f"Member names must be unique, got {names}.")
        if self.first_weight <= 0:
            raise ValueError(f"first_weight must be positive, got {self.first_weight}.")
        self.estimators_ = [(name, clone(model).fit(X, y)) for name, model in self.estimators]
        self.classes_ = np.unique(combined_label(y).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the pair with the most weighted votes per sample.

        Args:
            X: Model features.

        Returns:
            Predicted targets with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, "estimators_")
        weights = [self.first_weight] + [1.0] * (len(self.estimators_) - 1)
        votes = pd.DataFrame(
            {
                name: combined_label(as_target_frame(model.predict(X), X.index))
                for name, model in self.estimators_
            }
        )
        weighted = pd.concat(
            [
                pd.get_dummies(votes[name]).astype(float) * weight
                for name, weight in zip(votes.columns, weights, strict=True)
            ],
            axis=1,
        )
        totals = weighted.T.groupby(level=0).sum().T
        pairs = _earliest_top_pair(votes, totals).str.split(LABEL_SEP, n=1, expand=True)
        return pairs.set_axis(list(TARGET_COLS), axis=1)


def _earliest_top_pair(votes: pd.DataFrame, totals: pd.DataFrame) -> pd.Series:
    """Pick per row the pair of the earliest member (column order) that has the top score.

    Args:
        votes: Predicted ``Crop|Stage`` pair per member (columns) and sample (rows).
        totals: Summed vote weight per pair (columns) and sample.

    Returns:
        The winning pair per sample.
    """
    best = totals.max(axis=1).to_numpy()
    rows = np.arange(len(votes))
    winner = pd.Series(pd.NA, index=votes.index, dtype=object)
    for name in votes.columns:
        score = totals.to_numpy()[rows, totals.columns.get_indexer(votes[name])]
        takes = np.isclose(score, best) & winner.isna().to_numpy()
        winner[takes] = votes.loc[takes, name]
    return winner


def _stack_member(member: StackMember) -> ClassifierMixin:
    # settings of the tuned reference models: combined SVM (d93cea89), AutoML Extra Trees
    # (30c75139), hierarchical logistic regression and kNN (#104)
    if member == "svm":
        return SVC(class_weight="balanced", C=300.0, gamma=0.003)
    if member == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=EXTRA_TREES_N_ESTIMATORS,
            class_weight="balanced",
            max_features=0.56,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )
    if member == "logreg":
        return LogisticRegression(
            class_weight="balanced", C=10.0, max_iter=LOGISTIC_REGRESSION_MAX_ITER
        )
    return KNeighborsClassifier(n_neighbors=1, metric="cosine", weights="distance")


def make_combined_stacking(
    members: tuple[StackMember, ...] = STACKING_MEMBERS,
) -> CombinedLabelClassifier:
    """Create a stacked ensemble on the combined label.

    Unlike :class:`CropStageVotingClassifier`, a logistic regression learns from the members'
    class scores (SVM decision values, probabilities of the others) when to trust which member;
    the scores come from inner cross-validation, so the meta-model never sees fitted rows.

    Args:
        members: Base models; the first is the tuned reference SVM.

    Returns:
        Unfitted classifier; tune the meta-model as ``estimator__final_estimator__C``.
    """
    return CombinedLabelClassifier(
        StackingClassifier(
            estimators=[(member, _stack_member(member)) for member in members],
            final_estimator=LogisticRegression(
                class_weight="balanced", max_iter=LOGISTIC_REGRESSION_MAX_ITER
            ),
            cv=StratifiedKFold(STACKING_INNER_FOLDS, shuffle=True, random_state=SEED),
            n_jobs=1,
        )
    )
