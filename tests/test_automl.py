"""Search orchestration checks without model fitting or data access."""

import itertools
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from awp2 import automl
from awp2.config import (
    AUTOML_FOCUSED_DIR,
    AUTOML_NO_SAMPLE_WEIGHT_FAMILIES,
    AUTOML_SAMPLE_WEIGHT_FAMILIES,
)
from awp2.preprocessing import PreprocessingConfig


class AutoMLTests(unittest.TestCase):
    def test_focused_profile_has_a_finite_balanced_search_space(self) -> None:
        config = automl.SearchConfig(profile="focused", region_subsets=False)
        plan = automl._plan(config)
        self.assertEqual(plan["preprocessing_variants"], 6)
        self.assertEqual(plan["model_spaces"], 8)
        self.assertEqual(plan["candidates_all_shards"], 384)
        self.assertEqual(plan["cv_pipeline_fits_all_shards"], 1920)
        tasks = list(automl._tasks(config))
        self.assertEqual(len(tasks), 384)
        self.assertEqual(len({task for task, _, _ in tasks}), 384)
        names = {payload["name"] for _, _, payload in tasks}
        for name in names:
            self.assertEqual(sum(payload["name"] == name for _, _, payload in tasks), 48)
        self.assertEqual(len({payload["name"] for _, _, payload in tasks[:8]}), 8)
        for name, model, grid, _ in automl._model_spaces(config):
            with self.subTest(name=name):
                self.assertFalse(set(grid) - set(model.get_params(deep=True)))

    def test_focused_profile_keeps_both_screening_winners(self) -> None:
        tasks = list(automl._tasks(automl.SearchConfig(profile="focused", region_subsets=False)))
        combined_preprocessing = PreprocessingConfig(
            use_spectral_standard_scale=True,
            use_savgol=True,
            savgol_window_length=11,
            savgol_polyorder=3,
            spectral_lpp_components=40,
            spectral_lpp_neighbors=20,
            use_principal_components=True,
        ).model_dump(mode="json")
        hierarchy_preprocessing = PreprocessingConfig(
            max_interpolation_gap_nm=10,
            use_spectral_minmax_scale=True,
            use_savgol=True,
            savgol_window_length=11,
            savgol_polyorder=3,
            use_mean_albedo=True,
            use_aez_month=True,
            use_vegetation_indices=True,
        ).model_dump(mode="json")
        expected = (
            (
                "combined_et",
                combined_preprocessing,
                {
                    "estimator__max_depth": None,
                    "estimator__max_features": 0.5,
                    "estimator__min_samples_leaf": 5,
                    "estimator__n_estimators": 600,
                },
            ),
            (
                "hierarchical_svm_lda",
                hierarchy_preprocessing,
                {
                    "crop_estimator__C": 100.0,
                    "crop_estimator__gamma": 0.1,
                    "crop_estimator__kernel": "rbf",
                    "stage_estimator__shrinkage": 0.01,
                },
            ),
        )
        for name, preprocessing, params in expected:
            with self.subTest(name=name):
                self.assertTrue(
                    any(
                        payload["name"] == name
                        and payload["preprocessing"] == preprocessing
                        and payload["params"] == params
                        for _, _, payload in tasks
                    )
                )

    def test_focused_shards_are_disjoint_and_cover_the_complete_grid(self) -> None:
        config = automl.SearchConfig(profile="focused", region_subsets=False)
        full = {task for task, _, _ in automl._tasks(config)}
        shards = [
            {
                task
                for task, _, _ in automl._tasks(
                    config.model_copy(
                        update={
                            "shard": index,
                            "shards": 3,
                        }
                    )
                )
            }
            for index in range(3)
        ]
        self.assertEqual(set.union(*shards), full)
        for first, second in itertools.combinations(shards, 2):
            self.assertFalse(first & second)

    def test_focused_cli_uses_a_separate_output_directory(self) -> None:
        with (
            patch("sys.argv", ["automl", "search", "--profile", "focused"]),
            patch.object(automl, "_search") as search,
        ):
            automl.main()
        directory, config, limit = search.call_args.args
        self.assertEqual(directory, AUTOML_FOCUSED_DIR)
        self.assertEqual(config.profile, "focused")
        self.assertFalse(config.region_subsets)
        self.assertIsNone(limit)

    def test_focused_profile_rejects_unsupported_family_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "focused"):
            automl.SearchConfig(profile="focused", models=("gnb",))

    def test_preprocessing_is_valid_and_has_no_duplicate_configurations(self) -> None:
        configurations = list(automl.preprocessing_candidates())
        serialized = {item.model_dump_json() for item in configurations}
        self.assertEqual(len(configurations), len(serialized))
        self.assertTrue(any(item.spectral_pca_components for item in configurations))
        self.assertTrue(any(item.spectral_dbn_layers for item in configurations))
        metadata = [item for item in configurations if item.spectral_regions == ()]
        self.assertEqual(len(metadata), 4)
        self.assertTrue(all(item.use_meta for item in metadata))

    def test_all_model_grids_name_existing_parameters(self) -> None:
        spaces = list(automl._model_spaces(automl.SearchConfig()))
        for name, model, grid, weighted in spaces:
            with self.subTest(name=name):
                self.assertFalse(set(grid) - set(model.get_params(deep=True)))
                if weighted and name.startswith("hierarchical_"):
                    self.assertFalse(
                        any(family in name for family in AUTOML_NO_SAMPLE_WEIGHT_FAMILIES)
                    )
        names = {name for name, _, _, _ in spaces}
        for family in AUTOML_SAMPLE_WEIGHT_FAMILIES:
            self.assertIn(f"combined_{family}", names)

    def test_permutation_and_shards_cover_the_plan_exactly_once(self) -> None:
        candidates = [PreprocessingConfig(spectral_regions=()), PreprocessingConfig()]
        spaces = [("combined_svm", object(), {"estimator__C": [1.0, 10.0]}, False)]
        with (
            patch.object(automl, "preprocessing_candidates", side_effect=lambda: iter(candidates)),
            patch.object(automl, "_model_spaces", side_effect=lambda config: iter(spaces)),
            patch.object(automl, "_region_choices", return_value=[None]),
            patch.object(automl, "AUTOML_INTERPOLATION_GAPS", (15,)),
        ):
            full = {task for task, _, _ in automl._tasks(automl.SearchConfig())}
            shards = [
                {task for task, _, _ in automl._tasks(automl.SearchConfig(shard=index, shards=3))}
                for index in range(3)
            ]
            self.assertEqual(len(full), 4)
            self.assertEqual(
                len(full), automl._plan(automl.SearchConfig())["candidates_all_shards"]
            )
            self.assertEqual(full, set.union(*shards))
            for first, second in itertools.combinations(shards, 2):
                self.assertFalse(first & second)

    def test_rejects_a_non_80_20_split_before_loading_data(self) -> None:
        with (
            patch.object(
                automl, "load_split_config", return_value=SimpleNamespace(validation_size=0.3)
            ),
            patch.object(automl, "load_split") as loader,
        ):
            with self.assertRaises(ValueError):
                automl._manifest(automl.SearchConfig())
            loader.assert_not_called()

    def test_manifest_mismatch_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(automl, "_manifest", return_value={"version": 1}):
                automl._connect(directory, automl.SearchConfig()).close()
            with patch.object(automl, "_manifest", return_value={"version": 2}):
                with self.assertRaises(ValueError):
                    automl._connect(directory, automl.SearchConfig())

    def test_checkpoint_resume_and_single_holdout_evaluation(self) -> None:
        config = automl.SearchConfig(models=("svm",), approaches=("combined",))
        payload = {
            "name": "combined_svm",
            "preprocessing": PreprocessingConfig().model_dump(mode="json"),
            "params": {"estimator__C": 1.0},
            "balance_samples": False,
        }
        tuned = SimpleNamespace(
            best_score=0.8, run_id="tuning-id", candidates=pd.DataFrame({"std": [0.02]})
        )
        evaluated = SimpleNamespace(
            run_id="holdout-id", metrics=SimpleNamespace(model_dump=lambda: {"bacc_combined": 0.7})
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(automl, "_manifest", return_value={"test": True}),
            patch.object(
                automl,
                "_tasks",
                side_effect=lambda config: iter(
                    [
                        ("task-id", object(), dict(payload)),
                    ]
                ),
            ),
            patch.object(automl, "tune", return_value=tuned) as tuning,
            patch.object(automl, "run", return_value=evaluated) as evaluation,
            patch.object(automl, "_plan", return_value={"candidates_all_shards": 1}),
        ):
            directory = Path(temporary)
            automl._search(directory, config, max_tasks=1)
            automl._search(directory, config, max_tasks=1)
            self.assertEqual(tuning.call_count, 1)
            evaluation.assert_not_called()
            automl._evaluate(directory, config, allow_partial=False)
            self.assertEqual(evaluation.call_count, 1)
            with self.assertRaises(FileExistsError):
                automl._evaluate(directory, config, allow_partial=False)
            with self.assertRaises(ValueError):
                automl._search(directory, config, max_tasks=1)
            self.assertEqual(evaluation.call_count, 1)
            record = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "completed")
            automl._report(directory)
            self.assertTrue((directory / "candidates.csv").exists())
            self.assertTrue((directory / "report.md").exists())

    def test_failed_candidate_is_recorded_and_not_ranked(self) -> None:
        payload = {
            "name": "combined_svm",
            "preprocessing": PreprocessingConfig().model_dump(mode="json"),
            "params": {"estimator__C": 1.0},
            "balance_samples": False,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(automl, "_manifest", return_value={"test": True}),
            patch.object(automl, "_tasks", return_value=iter([("failed", object(), payload)])),
            patch.object(automl, "tune", side_effect=ValueError("Invalid fold dimensions")),
        ):
            directory = Path(temporary)
            automl._search(directory, automl.SearchConfig(), max_tasks=1)
            connection = sqlite3.connect(directory / "search.sqlite")
            try:
                self.assertEqual(
                    connection.execute("SELECT status FROM results").fetchone()[0], "failed"
                )
                with self.assertRaises(ValueError):
                    automl._winner(connection)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
