"""Classifiers that predict the crop first and then the growth stage with one model per crop."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from awp2.config import (
    CROP_COL,
    EXTRA_TREES_N_ESTIMATORS,
    HARD_ROUTING,
    LSTM_STANDARD_CONTEXT,
    LSTMContextMode,
    SEED,
    SOFT_ROUTING,
    STAGE_COL,
    TARGET_COLS,
    Routing,
)
from awp2.data import combined_label
from awp2.evaluation import (
    ErrorBreakdown,
    Metrics,
    as_target_frame,
    error_breakdown,
    evaluate,
)
from awp2.models.lstm import SpectralLSTMClassifier


class HierarchicalClassifier(ClassifierMixin, BaseEstimator):
    """Predict the crop, then the stage with a stage model trained only on that crop.

    Each stage model only knows the stages observed for its crop, so every prediction is a
    crop/stage pair that occurs in training. Hyperparameters of both levels are reachable for
    ``tune()`` as ``crop_estimator__<name>`` and ``stage_estimator__<name>``; all stage models
    share the same settings.

    Args:
        crop_estimator: Unfitted classifier for the crop.
        stage_estimator: Unfitted classifier for the stage, cloned once per crop.
        routing: ``"hard"`` lets the predicted crop pick the stage model; ``"soft"`` picks the
            pair with the highest P(crop) · P(stage | crop), so a confident stage model can
            overrule an uncertain crop. ``"soft"`` needs ``predict_proba`` on both levels.
    """

    def __init__(
        self,
        crop_estimator: ClassifierMixin,
        stage_estimator: ClassifierMixin,
        routing: Routing = HARD_ROUTING,
    ) -> None:
        """Initialise the hierarchy from two unfitted single-output classifiers.

        Args:
            crop_estimator: Classifier for the crop.
            stage_estimator: Classifier for the stage, cloned once per crop.
            routing: How the crop decision reaches the stage, ``"hard"`` or ``"soft"``.
        """
        self.crop_estimator = crop_estimator
        self.stage_estimator = stage_estimator
        self.routing = routing

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled classifier, e.g. one loaded from MLflow.

        Args:
            state: Attributes of the pickled object.
        """
        # models stored before routing existed were all fitted with hard routing
        state.setdefault("routing", HARD_ROUTING)
        super().__setstate__(state)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "HierarchicalClassifier":
        """Fit the crop model on all samples and one stage model per crop.

        The stage models are trained on the true crop, so they learn the stage given a correct
        crop; crop errors at prediction time propagate into the stage.

        Args:
            X: Model features.
            y: Targets with the columns ``Crop`` and ``Stage``.
            sample_weight: Optional weight for each sample. It is rescaled per level: the crop
                model sees the same total weight for every crop, each stage model sees weights
                with mean 1, so shared hyperparameters such as ``C`` act alike for all crops.

        Returns:
            This fitted classifier.

        Raises:
            ValueError: If the target columns do not match the project contract, ``routing`` is
                unknown, or soft routing meets a classifier without ``predict_proba``.
        """
        if self.routing not in (HARD_ROUTING, SOFT_ROUTING):
            raise ValueError(f"Unknown routing {self.routing!r}.")
        if self.routing == SOFT_ROUTING and not all(
            hasattr(model, "predict_proba") for model in (self.crop_estimator, self.stage_estimator)
        ):
            raise ValueError("Soft routing needs predict_proba, e.g. CalibratedClassifierCV.")
        if list(y.columns) != list(TARGET_COLS):
            raise ValueError(f"Expected target columns {list(TARGET_COLS)}, got {list(y.columns)}.")
        crops = y[CROP_COL].to_numpy()
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.crop_estimator_ = clone(self.crop_estimator).fit(
            X, crops, **_weight_params(None if weights is None else _equal_per_crop(weights, crops))
        )
        self.stage_estimators_ = {}
        for crop in self.crop_estimator_.classes_:
            is_crop = crops == crop
            crop_weights = None if weights is None else weights[is_crop] / weights[is_crop].mean()
            self.stage_estimators_[crop] = clone(self.stage_estimator).fit(
                X[is_crop], y.loc[is_crop, STAGE_COL], **_weight_params(crop_weights)
            )
        self.classes_ = np.unique(combined_label(y).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict crop and stage according to ``routing``.

        Args:
            X: Model features.

        Returns:
            Predicted targets with the columns ``Crop`` and ``Stage``.
        """
        check_is_fitted(self, "crop_estimator_")
        if self.routing == SOFT_ROUTING:
            return self._predict_soft(X)
        crop = pd.Series(self.crop_estimator_.predict(X), index=X.index, name=CROP_COL)
        return pd.concat([crop, self.predict_stage(X, crop)], axis=1)

    def _predict_soft(self, X: pd.DataFrame) -> pd.DataFrame:
        crop_proba = self.crop_estimator_.predict_proba(X)
        pairs: list[tuple[str, str]] = []
        pair_scores: list[np.ndarray] = []
        for column, crop in enumerate(self.crop_estimator_.classes_):
            stage_model = self.stage_estimators_[crop]
            pair_scores.append(crop_proba[:, [column]] * stage_model.predict_proba(X))
            pairs.extend((crop, stage) for stage in stage_model.classes_)
        best = np.hstack(pair_scores).argmax(axis=1)
        return pd.DataFrame([pairs[i] for i in best], index=X.index, columns=list(TARGET_COLS))

    def predict_stage(self, X: pd.DataFrame, crop: pd.Series | np.ndarray) -> pd.Series:
        """Predict the stage for a given crop per sample.

        Passing the true crop measures the stage error without crop errors (oracle crop). ``X``
        must be preprocessed like inside the pipeline, i.e. ``result.pipeline[:-1].transform(X)``.

        Args:
            X: Model features.
            crop: Crop per sample, matched to ``X`` by position.

        Returns:
            Predicted stage per sample, named ``Stage``, with the index of ``X``.

        Raises:
            ValueError: If ``crop`` has another length than ``X`` or contains a crop that was not
                seen in training.
        """
        check_is_fitted(self, "stage_estimators_")
        crops = np.asarray(crop)
        if len(crops) != len(X):
            raise ValueError(f"Got {len(crops)} crops for {len(X)} samples.")
        unknown = set(crops) - set(self.stage_estimators_)
        if unknown:
            raise ValueError(f"No stage model for crop(s) {sorted(unknown)}.")
        stage = np.empty(len(X), dtype=object)
        for value in np.unique(crops):
            is_crop = crops == value
            stage[is_crop] = self.stage_estimators_[value].predict(X[is_crop])
        return pd.Series(stage, index=X.index, name=STAGE_COL)


def oracle_crop_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame) -> Metrics:
    """Score the stage models of a fitted hierarchy as if the crop were always correct.

    The gap to the normal metrics is what crop errors cost the stage and the combination.

    Args:
        pipeline: Fitted pipeline whose last step is a :class:`HierarchicalClassifier`, e.g.
            ``RunResult.pipeline``.
        X: Raw features, e.g. ``load_split().X_val``.
        y: True targets with the columns ``Crop`` and ``Stage``.

    Returns:
        Metrics of the true crop combined with the stage predicted for it.

    Raises:
        TypeError: If the last pipeline step is not a :class:`HierarchicalClassifier`.
    """
    model = pipeline[-1]
    if not isinstance(model, HierarchicalClassifier):
        raise TypeError(f"Expected a HierarchicalClassifier, got {type(model).__name__}.")
    stage = model.predict_stage(pipeline[:-1].transform(X), y[CROP_COL])
    return evaluate(y, pd.concat([y[CROP_COL], stage], axis=1))


def _weight_params(sample_weight: np.ndarray | None) -> dict[str, np.ndarray]:
    return {} if sample_weight is None else {"sample_weight": sample_weight}


def _equal_per_crop(weights: np.ndarray, crops: np.ndarray) -> np.ndarray:
    # balanced_sample_weight balances crop/stage pairs, which favours crops with many stages
    _, crop_index = np.unique(crops, return_inverse=True)
    crop_totals = np.bincount(crop_index, weights=weights)
    return weights * len(weights) / (len(crop_totals) * crop_totals[crop_index])


def make_hierarchical_svm(routing: Routing = HARD_ROUTING) -> HierarchicalClassifier:
    """Create the hierarchy with an RBF-SVM on both levels.

    Args:
        routing: ``"soft"`` wraps each SVM in a sigmoid ``CalibratedClassifierCV`` for
            probabilities (internal cross-validation, slower to fit); tune its SVM as
            ``crop_estimator__estimator__C`` etc. The calibrated models also *predict* by the
            highest probability, so compare soft against a separately tuned hard hierarchy.

    Returns:
        Unfitted classifier with balanced class weights, so keep ``balance_samples=False``.
            Scale its features through
            ``PreprocessingConfig(use_spectral_standard_scale=True)`` before fitting.
    """
    return HierarchicalClassifier(_svm(routing), _svm(routing), routing=routing)


def make_hierarchical_spectral_lstm(
    context_mode: LSTMContextMode = LSTM_STANDARD_CONTEXT,
) -> HierarchicalClassifier:
    """Create the hard-routed LSTM hierarchy.

    Args:
        context_mode: How each LSTM receives AEZ and Month context.

    Returns:
        Unfitted hard-routed hierarchy with separate LSTMs for crop and stage models.
    """
    return HierarchicalClassifier(
        SpectralLSTMClassifier(context_mode=context_mode),
        SpectralLSTMClassifier(context_mode=context_mode),
    )


def make_hierarchical_extra_trees() -> HierarchicalClassifier:
    """Create the hierarchy with Extra Trees on both levels.

    Returns:
        Unfitted classifier with balanced class weights and the project seed, so keep
            ``balance_samples=False``.
    """
    return HierarchicalClassifier(_extra_trees(), _extra_trees())


def make_hierarchical_extra_trees_svm() -> HierarchicalClassifier:
    """Create the hierarchy with Extra Trees for the crop and an RBF-SVM per crop for the stage.

    Returns:
        Unfitted classifier with balanced class weights and the project seed, so keep
            ``balance_samples=False``. Scale its features through
            ``PreprocessingConfig(use_spectral_standard_scale=True)`` for the stage SVMs.
    """
    return HierarchicalClassifier(_extra_trees(), SVC(class_weight="balanced"))


def _svm(routing: Routing) -> SVC | CalibratedClassifierCV:
    svm = SVC(class_weight="balanced")
    if routing == SOFT_ROUTING:
        return CalibratedClassifierCV(svm, method="sigmoid", ensemble=False)
    return svm


def _extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=EXTRA_TREES_N_ESTIMATORS,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


class HierarchicalSummary(BaseModel):
    """Scores of a fitted hierarchy on labelled data, incl. where its errors come from."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    metrics: Metrics = Field(description="Grading metrics of the predictions.")
    breakdown: ErrorBreakdown = Field(description="Crop errors vs. stage-only errors.")
    oracle: Metrics = Field(description="Metrics if the crop were always correct.")


def summarize_hierarchical(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame
) -> HierarchicalSummary:
    """Score a fitted hierarchy, split its errors and add the oracle-crop scores.

    Args:
        pipeline: Fitted pipeline whose last step is a ``HierarchicalClassifier``.
        X: Raw features, e.g. ``load_split().X_val``.
        y: True targets with the columns ``Crop`` and ``Stage``.

    Returns:
        Metrics, error breakdown and oracle-crop metrics.
    """
    predicted = as_target_frame(pipeline.predict(X), y.index)
    return HierarchicalSummary(
        metrics=evaluate(y, predicted),
        breakdown=error_breakdown(y, predicted),
        oracle=oracle_crop_metrics(pipeline, X, y),
    )


def summary_table(summaries: Mapping[str, HierarchicalSummary]) -> pd.DataFrame:
    """Put several summaries side by side, one column per model.

    Args:
        summaries: ``summarize_hierarchical()`` per model, keyed by the label to show.

    Returns:
        Rows: BAcc crop/stage/combined, oracle-crop combined and the error breakdown.
    """
    return pd.DataFrame(
        {
            label: {
                "BAcc Kultur": summary.metrics.bacc_crop,
                "BAcc Stadium": summary.metrics.bacc_stage,
                "BAcc kombiniert": summary.metrics.bacc_combined,
                "Oracle-Kultur kombiniert": summary.oracle.bacc_combined,
                **summary.breakdown.model_dump(),
            }
            for label, summary in summaries.items()
        }
    )
