"""Search model family, hyperparameters and preprocessing in one Optuna study (AutoML)."""

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from awp2.config import (
    AUTOML_APPROACH,
    AUTOML_ET_MAX_FEATURES,
    AUTOML_ET_MIN_SAMPLES_LEAF,
    AUTOML_HGB_L2,
    AUTOML_HGB_LEARNING_RATE,
    AUTOML_HGB_MAX_LEAF_NODES,
    AUTOML_LOGREG_C,
    AUTOML_MLP_ALPHA,
    AUTOML_MLP_LAYERS,
    AUTOML_N_TRIALS,
    AUTOML_PCA_COMPONENTS,
    AUTOML_PRUNER_WARMUP_FOLDS,
    AUTOML_SAVGOL_POLYORDER,
    AUTOML_SAVGOL_WINDOWS,
    AUTOML_STARTUP_TRIALS,
    AUTOML_STORAGE,
    AUTOML_SVM_C,
    AUTOML_SVM_GAMMA,
    AUTOML_TIMEOUT_S,
    COMBINED_STRUCTURE,
    EXTRA_TREES_FAMILY,
    EXTRA_TREES_N_ESTIMATORS,
    HGB_FAMILY,
    HIERARCHICAL_STRUCTURE,
    LOGISTIC_REGRESSION_MAX_ITER,
    LOGREG_FAMILY,
    MINMAX_SCALER,
    MLFLOW_EXPERIMENT,
    MLP_FAMILY,
    MLP_MAX_ITER,
    RANDOM_FOLDS,
    SEED,
    SLUG_PATTERN,
    STANDARD_SCALER,
    SVM_FAMILY,
    FoldGrouping,
    ModelFamily,
    ModelStructure,
    SpectralScaler,
)
from awp2.data import load_split, load_split_config
from awp2.evaluation import MetricName, as_target_frame, evaluate
from awp2.experiment import build_pipeline, cv_folds, pipeline_fit_params
from awp2.models.combined import CombinedLabelClassifier
from awp2.models.hierarchical import HierarchicalClassifier
from awp2.models.specialist import PairSpecialistClassifier
from awp2.preprocessing import PreprocessingConfig
from awp2.tracking import log_tuning

_FAMILY_PARAM = "family"
_STRUCTURE_PARAM = "structure"
# the two levels of a hierarchy get their own parameters, so the search can tune them apart
_CROP_PREFIX, _STAGE_PREFIX = "crop_", "stage_"
_FOLD_SCORES_ATTR = "fold_scores"
# HistGradientBoosting and MLP have no class_weight option
_SAMPLE_WEIGHTED_FAMILIES = frozenset({HGB_FAMILY, MLP_FAMILY})


class AutoMLConfig(BaseModel):
    """Everything that defines an AutoML search; the search space itself is fixed in code."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        pattern=SLUG_PATTERN,
        description="Name of the search and of the Optuna study; the same name resumes it.",
    )
    approach: str = Field(
        default=AUTOML_APPROACH, pattern=SLUG_PATTERN, description="Approach tag in MLflow."
    )
    description: str = Field(default="", description="What was searched and why (free text).")
    experiment: str = Field(
        default=MLFLOW_EXPERIMENT, pattern=SLUG_PATTERN, description="MLflow experiment."
    )
    folds: FoldGrouping = Field(default=RANDOM_FOLDS, description="Cross-validation folds.")
    metric: MetricName = Field(default="bacc_combined", description="Score to maximise.")
    families: tuple[ModelFamily, ...] = Field(
        default=(SVM_FAMILY, EXTRA_TREES_FAMILY, HGB_FAMILY, LOGREG_FAMILY, MLP_FAMILY),
        min_length=1,
        description="Model families the search may choose from.",
    )
    structures: tuple[ModelStructure, ...] = Field(
        default=(COMBINED_STRUCTURE,),
        min_length=1,
        description="How crop and stage are predicted: one combined label, a hierarchy (crop, "
        "then stage per crop – each level picks its own family) or two separate models.",
    )
    extra_features: bool = Field(
        default=False,
        description="Also switch the dry-matter indices, soil indices and first derivative of "
        "the spectra on or off.",
    )
    specialist: bool = Field(
        default=False,
        description="Also try an RBF-SVM specialist that re-decides early maize and soybean "
        "(PairSpecialistClassifier) on top of the drawn model.",
    )
    n_trials: int = Field(default=AUTOML_N_TRIALS, gt=0, description="Maximum number of trials.")
    timeout_s: int | None = Field(
        default=AUTOML_TIMEOUT_S, gt=0, description="Stop starting new trials after this time."
    )
    persist: bool = Field(
        default=True,
        description="Keep the study in AUTOML_STORAGE so a stopped search can be resumed; "
        "False keeps it in memory, e.g. for tests.",
    )
    track: bool = Field(default=True, description="Record the search as tuning run in MLflow.")


class Candidate(NamedTuple):
    """Model and fit settings one set of search parameters stands for.

    Attributes:
        model: Unfitted classifier that predicts ``Crop`` and ``Stage``.
        preprocessing: Preprocessing to fit before the model.
        balance_samples: Whether the model gets balanced sample weights.
    """

    model: BaseEstimator
    preprocessing: PreprocessingConfig
    balance_samples: bool


@dataclass(frozen=True)
class AutoMLResult:
    """Outcome of :func:`automl`.

    Attributes:
        config: The configuration the search was started with.
        best: Model, preprocessing and sample weighting of the best trial – pass them to
            ``run()``.
        best_params: Search parameters of the best trial.
        best_score: Its mean cross-validation score.
        candidates: One row per complete trial: ``params``, ``mean``, ``std``, ``rank``,
            ``fold_<i>``.
        study: The Optuna study, e.g. for ``optuna.visualization``.
        run_id: MLflow run id of the search, ``None`` if it was not tracked.
    """

    config: AutoMLConfig
    best: Candidate
    best_params: dict[str, Any]
    best_score: float
    candidates: pd.DataFrame
    study: optuna.Study
    run_id: str | None


def suggest_candidate(
    trial: optuna.trial.BaseTrial,
    families: tuple[ModelFamily, ...],
    structures: tuple[ModelStructure, ...] = (COMBINED_STRUCTURE,),
    extra_features: bool = False,
    specialist: bool = False,
) -> Candidate:
    """Draw structure, model families, their hyperparameters and the preprocessing for one trial.

    Pass an ``optuna.trial.FixedTrial`` to rebuild the candidate of known parameters. The
    optional dimensions are only drawn when switched on, so studies without them stay
    reproducible.

    Args:
        trial: Optuna trial that suggests the values.
        families: Model families to choose from.
        structures: Ways to predict crop and stage to choose from; with a single one it is not
            a search parameter.
        extra_features: Also draw the dry-matter, soil and derivative switches.
        specialist: Also draw whether an early maize/soybean specialist re-decides its rows.

    Returns:
        The candidate the suggested values stand for.
    """
    structure = (
        structures[0]
        if len(structures) == 1
        else cast(ModelStructure, trial.suggest_categorical(_STRUCTURE_PARAM, structures))
    )
    if structure == HIERARCHICAL_STRUCTURE:
        crop_family = _suggest_family(trial, families, _CROP_PREFIX)
        stage_family = _suggest_family(trial, families, _STAGE_PREFIX)
        balance_samples = bool({crop_family, stage_family} & _SAMPLE_WEIGHTED_FAMILIES)
        model: BaseEstimator = HierarchicalClassifier(
            _suggest_estimator(trial, crop_family, _CROP_PREFIX, balance_samples),
            _suggest_estimator(trial, stage_family, _STAGE_PREFIX, balance_samples),
        )
    else:
        family = _suggest_family(trial, families, "")
        balance_samples = family in _SAMPLE_WEIGHTED_FAMILIES
        estimator = _suggest_estimator(trial, family, "", balance_samples)
        model = (
            CombinedLabelClassifier(estimator)
            if structure == COMBINED_STRUCTURE
            else MultiOutputClassifier(estimator)
        )
    if specialist and trial.suggest_categorical("use_specialist", (False, True)):
        model = PairSpecialistClassifier(
            model, _suggest_estimator(trial, SVM_FAMILY, "specialist_", balance_samples)
        )
    return Candidate(
        model=model,
        preprocessing=_suggest_preprocessing(trial, extra_features),
        balance_samples=balance_samples,
    )


def _suggest_family(
    trial: optuna.trial.BaseTrial, families: tuple[ModelFamily, ...], prefix: str
) -> ModelFamily:
    return cast(ModelFamily, trial.suggest_categorical(f"{prefix}{_FAMILY_PARAM}", families))


def _suggest_estimator(
    trial: optuna.trial.BaseTrial, family: ModelFamily, prefix: str, balance_samples: bool
) -> ClassifierMixin:
    # sample weights already balance the classes; class weights on top would do it twice
    class_weight = None if balance_samples else "balanced"
    if family == SVM_FAMILY:
        return SVC(
            class_weight=class_weight,
            C=trial.suggest_float(f"{prefix}svm_c", *AUTOML_SVM_C, log=True),
            gamma=trial.suggest_float(f"{prefix}svm_gamma", *AUTOML_SVM_GAMMA, log=True),
        )
    if family == EXTRA_TREES_FAMILY:
        return ExtraTreesClassifier(
            n_estimators=EXTRA_TREES_N_ESTIMATORS,
            class_weight=class_weight,
            max_features=trial.suggest_float(f"{prefix}et_max_features", *AUTOML_ET_MAX_FEATURES),
            min_samples_leaf=trial.suggest_int(
                f"{prefix}et_min_samples_leaf", *AUTOML_ET_MIN_SAMPLES_LEAF
            ),
            random_state=SEED,
            n_jobs=-1,
        )
    if family == HGB_FAMILY:
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float(
                f"{prefix}hgb_learning_rate", *AUTOML_HGB_LEARNING_RATE, log=True
            ),
            max_leaf_nodes=trial.suggest_int(
                f"{prefix}hgb_max_leaf_nodes", *AUTOML_HGB_MAX_LEAF_NODES, log=True
            ),
            l2_regularization=trial.suggest_float(f"{prefix}hgb_l2", *AUTOML_HGB_L2, log=True),
            random_state=SEED,
        )
    if family == LOGREG_FAMILY:
        return LogisticRegression(
            class_weight=class_weight,
            C=trial.suggest_float(f"{prefix}logreg_c", *AUTOML_LOGREG_C, log=True),
            max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        )
    layers = trial.suggest_categorical(f"{prefix}mlp_layers", tuple(AUTOML_MLP_LAYERS))
    return MLPClassifier(
        hidden_layer_sizes=AUTOML_MLP_LAYERS[layers],
        alpha=trial.suggest_float(f"{prefix}mlp_alpha", *AUTOML_MLP_ALPHA, log=True),
        early_stopping=True,
        max_iter=MLP_MAX_ITER,
        random_state=SEED,
    )


def _suggest_preprocessing(
    trial: optuna.trial.BaseTrial, extra_features: bool
) -> PreprocessingConfig:
    scaler: SpectralScaler = trial.suggest_categorical("scaler", (STANDARD_SCALER, MINMAX_SCALER))
    # PCA needs standardised bands (PreprocessingConfig rejects it otherwise)
    pca_components = (
        cast(
            int | None, trial.suggest_categorical("pca_components", (None, *AUTOML_PCA_COMPONENTS))
        )
        if scaler == STANDARD_SCALER
        else None
    )
    use_savgol = trial.suggest_categorical("use_savgol", (False, True))
    savgol_window = (
        trial.suggest_categorical("savgol_window", AUTOML_SAVGOL_WINDOWS) if use_savgol else None
    )
    extra = (
        _suggest_extra_features(trial) if extra_features else _ExtraFeatures(False, False, False)
    )
    return PreprocessingConfig(
        use_spectral_standard_scale=scaler == STANDARD_SCALER,
        use_spectral_minmax_scale=scaler != STANDARD_SCALER,
        use_spectral_snv=trial.suggest_categorical("use_snv", (False, True)),
        use_savgol=use_savgol,
        savgol_window_length=savgol_window,
        savgol_polyorder=AUTOML_SAVGOL_POLYORDER if use_savgol else None,
        spectral_pca_components=pca_components,
        use_vegetation_indices=trial.suggest_categorical("use_indices", (False, True)),
        use_aez_month=trial.suggest_categorical("use_aez_month", (False, True)),
        use_cyclic_month=trial.suggest_categorical("use_cyclic_month", (False, True)),
        use_dry_matter_indices=extra.dry_matter,
        use_soil_indices=extra.soil,
        use_spectral_derivative=extra.derivative,
    )


class _ExtraFeatures(NamedTuple):
    dry_matter: bool
    soil: bool
    derivative: bool


def _suggest_extra_features(trial: optuna.trial.BaseTrial) -> _ExtraFeatures:
    return _ExtraFeatures(
        dry_matter=bool(trial.suggest_categorical("use_dry_matter_indices", (False, True))),
        soil=bool(trial.suggest_categorical("use_soil_indices", (False, True))),
        derivative=bool(trial.suggest_categorical("use_spectral_derivative", (False, True))),
    )


def automl(config: AutoMLConfig) -> AutoMLResult:
    """Search model family, hyperparameters and preprocessing by cross-validation.

    TPE proposes the next trial from the scores of the previous ones; after every fold the
    running mean is compared with the other trials at the same fold, and a trial below their
    median is stopped early. Like ``tune()`` it only uses the training part, so evaluate the
    winner once with ``run()``. With many trials the best CV score is optimistically biased –
    compare it with the reference before trusting a small gain.

    Args:
        config: Name, budget and model families of the search.

    Returns:
        Best candidate and score, all complete trials and the Optuna study.

    Example:
        ```python
        result = automl(AutoMLConfig(name="automl_all", n_trials=200))
        best = result.best
        run(
            best.model,
            RunConfig(
                name="automl_best",
                approach=AUTOML_APPROACH,
                preprocessing=best.preprocessing,
                balance_samples=best.balance_samples,
                tuning_run=result.run_id,
            ),
        )
        ```
    """
    split = load_split()
    folds = cv_folds(config.folds)

    def objective(trial: optuna.Trial) -> float:
        candidate = suggest_candidate(
            trial, config.families, config.structures, config.extra_features, config.specialist
        )
        scores: list[float] = []
        for step, fold in enumerate(folds):
            X_fold, y_fold = split.X_train.iloc[fold.train], split.y_train.iloc[fold.train]
            y_held_out = split.y_train.iloc[fold.val]
            pipeline = build_pipeline(candidate.model, candidate.preprocessing)
            pipeline.fit(X_fold, y_fold, **pipeline_fit_params(candidate.balance_samples, y_fold))
            predicted = pipeline.predict(split.X_train.iloc[fold.val])
            metrics = evaluate(y_held_out, as_target_frame(predicted, y_held_out.index))
            scores.append(getattr(metrics, config.metric))
            trial.set_user_attr(_FOLD_SCORES_ATTR, scores)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=config.name,
        storage=f"sqlite:///{AUTOML_STORAGE}" if config.persist else None,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=AUTOML_STARTUP_TRIALS),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=AUTOML_STARTUP_TRIALS, n_warmup_steps=AUTOML_PRUNER_WARMUP_FOLDS
        ),
    )
    # an invalid combination or a diverging fit fails one trial, not the whole search
    study.optimize(
        objective, n_trials=config.n_trials, timeout=config.timeout_s, catch=(ValueError,)
    )

    candidates = _candidates(study)
    best = suggest_candidate(
        optuna.trial.FixedTrial(study.best_params),
        config.families,
        config.structures,
        config.extra_features,
        config.specialist,
    )
    run_id = _track(config, study, candidates) if config.track else None
    return AutoMLResult(
        config=config,
        best=best,
        best_params=study.best_params,
        best_score=float(study.best_value),
        candidates=candidates,
        study=study,
        run_id=run_id,
    )


def _candidates(study: optuna.Study) -> pd.DataFrame:
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    rows = [
        {
            "params": trial.params,
            "mean": trial.value,
            "std": float(np.std(trial.user_attrs[_FOLD_SCORES_ATTR])),
            **{
                f"fold_{number}": score
                for number, score in enumerate(trial.user_attrs[_FOLD_SCORES_ATTR])
            },
        }
        for trial in complete
    ]
    candidates = pd.DataFrame(rows)
    candidates["rank"] = candidates["mean"].rank(ascending=False, method="min").astype(int)
    return candidates


def _track(config: AutoMLConfig, study: optuna.Study, candidates: pd.DataFrame) -> str | None:
    split = load_split()
    states = Counter(trial.state.name for trial in study.trials)
    try:
        return log_tuning(
            experiment=config.experiment,
            name=config.name,
            approach=config.approach,
            description=config.description,
            params={
                "automl.families": ",".join(config.families),
                "automl.structures": ",".join(config.structures),
                "automl.extra_features": config.extra_features,
                "automl.specialist": config.specialist,
                "automl.n_trials": config.n_trials,
                "automl.timeout_s": config.timeout_s or 0,
                "split.validation_size": load_split_config().validation_size,
                "cv.folds": config.folds,
            },
            metric=config.metric,
            candidates=candidates,
            train=pd.concat([split.X_train, split.y_train], axis=1),
            extra_metrics={
                f"automl_{state.lower()}_trials": float(count) for state, count in states.items()
            },
            nested_candidates=False,
        )
    except Exception as error:  # noqa: BLE001 – a tracking problem must not discard the search
        warnings.warn(f"{config.name}: search could not be tracked ({error}).", stacklevel=3)
        return None
