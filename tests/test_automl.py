"""Tests for the Optuna search over model family, hyperparameters and preprocessing."""

import unittest
from typing import get_args

import optuna
from sklearn.multioutput import MultiOutputClassifier

from awp2.automl import AutoMLConfig, automl, suggest_candidate
from awp2.config import (
    COMBINED_STRUCTURE,
    EXTRA_TREES_FAMILY,
    HGB_FAMILY,
    HIERARCHICAL_STRUCTURE,
    MLP_FAMILY,
    SEED,
    SEPARATE_STRUCTURE,
    SPLIT_FILE,
    SVM_FAMILY,
    ModelFamily,
    ModelStructure,
)
from awp2.models.combined import CombinedLabelClassifier
from awp2.models.hierarchical import HierarchicalClassifier
from awp2.models.specialist import PairSpecialistClassifier

N_DRAWS = 25
MODEL_TYPES = {
    COMBINED_STRUCTURE: CombinedLabelClassifier,
    HIERARCHICAL_STRUCTURE: HierarchicalClassifier,
    SEPARATE_STRUCTURE: MultiOutputClassifier,
}


class SuggestCandidateTest(unittest.TestCase):
    def test_every_draw_is_a_valid_candidate(self) -> None:
        for family in get_args(ModelFamily):
            with self.subTest(family=family):
                study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED))
                for _ in range(N_DRAWS):
                    candidate = suggest_candidate(study.ask(), (family,))

                    self.assertIsInstance(candidate.model, CombinedLabelClassifier)
                    self.assertEqual(candidate.balance_samples, family in {HGB_FAMILY, MLP_FAMILY})

    def test_every_structure_builds_its_model_type(self) -> None:
        for structure in get_args(ModelStructure):
            with self.subTest(structure=structure):
                study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED))
                for _ in range(N_DRAWS):
                    candidate = suggest_candidate(study.ask(), get_args(ModelFamily), (structure,))

                    self.assertIsInstance(candidate.model, MODEL_TYPES[structure])

    def test_extra_features_and_specialist_build_valid_candidates(self) -> None:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED))
        drawn_specialist = False
        for _ in range(N_DRAWS):
            candidate = suggest_candidate(
                study.ask(), get_args(ModelFamily), extra_features=True, specialist=True
            )

            drawn_specialist |= isinstance(candidate.model, PairSpecialistClassifier)
        self.assertTrue(drawn_specialist)

    def test_hierarchy_levels_are_drawn_independently(self) -> None:
        params = {
            "structure": HIERARCHICAL_STRUCTURE,
            "crop_family": EXTRA_TREES_FAMILY,
            "crop_et_max_features": 0.3,
            "crop_et_min_samples_leaf": 1,
            "stage_family": MLP_FAMILY,
            "stage_mlp_layers": "128",
            "stage_mlp_alpha": 0.01,
            "scaler": "standard",
            "pca_components": None,
            "use_savgol": False,
            "use_snv": False,
            "use_indices": False,
            "use_aez_month": False,
            "use_cyclic_month": True,
        }

        candidate = suggest_candidate(
            optuna.trial.FixedTrial(params),
            (EXTRA_TREES_FAMILY, MLP_FAMILY),
            (COMBINED_STRUCTURE, HIERARCHICAL_STRUCTURE),
        )

        self.assertEqual(candidate.model.crop_estimator.max_features, 0.3)
        self.assertEqual(candidate.model.stage_estimator.alpha, 0.01)
        # the MLP level needs sample weights, so the trees must not balance a second time
        self.assertTrue(candidate.balance_samples)
        self.assertIsNone(candidate.model.crop_estimator.class_weight)

    def test_fixed_params_rebuild_the_candidate(self) -> None:
        params = {
            "family": SVM_FAMILY,
            "svm_c": 300.0,
            "svm_gamma": 0.003,
            "scaler": "standard",
            "pca_components": None,
            "use_savgol": False,
            "use_snv": False,
            "use_indices": False,
            "use_aez_month": True,
            "use_cyclic_month": True,
        }

        candidate = suggest_candidate(optuna.trial.FixedTrial(params), (SVM_FAMILY,))

        self.assertEqual(candidate.model.estimator.C, 300.0)
        self.assertTrue(candidate.preprocessing.use_aez_month)
        self.assertTrue(candidate.preprocessing.use_spectral_standard_scale)


# the search runs on the shared split, which CI does not have (data/ is not in git)
@unittest.skipUnless(SPLIT_FILE.exists(), "needs the shared split from `make data`")
class AutoMLTest(unittest.TestCase):
    def test_short_search_returns_the_best_complete_trial(self) -> None:
        config = AutoMLConfig(
            name="automl_test",
            families=(SVM_FAMILY,),
            n_trials=2,
            timeout_s=None,
            persist=False,
            track=False,
        )

        result = automl(config)

        self.assertEqual(result.best_score, result.candidates["mean"].max())
        self.assertIsInstance(result.best.model, CombinedLabelClassifier)
        self.assertIsNone(result.run_id)


if __name__ == "__main__":
    unittest.main()
