"""Tests for shared preprocessing behavior."""

import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from awp2 import preprocessing
from awp2.config import (
    AEZ_COL,
    MEAN_ALBEDO_COL,
    MONTH_COL,
    SEED,
    VEGETATION_INDEX_NAMES,
)
from awp2.preprocessing import (
    AddVegetationIndices,
    AezMonthEncoder,
    InterpolateBands,
    PreprocessingConfig,
    StandardNormalVariate,
    build_preprocessor,
    describe_preprocessor,
    nearest_band,
)


class PreprocessingApiTest(unittest.TestCase):
    def test_public_api_keeps_existing_imports(self) -> None:
        expected = {
            "nearest_band",
            "DropEmptyBands",
            "InterpolateBands",
            "SavitzkyGolaySmoothing",
            "StandardNormalVariate",
            "OrthogonalSubspaceProjection",
            "VerySparseRandomProjection",
            "LocalityPreservingProjection",
            "DeepBeliefFeatures",
            "AddVegetationIndices",
            "SelectVegetationIndices",
            "PrincipalComponentFeatures",
            "CyclicMonthEncoder",
            "MeanAlbedo",
            "AezMonthEncoder",
            "PreprocessingConfig",
            "build_preprocessor",
            "describe_preprocessor",
        }
        self.assertEqual(set(preprocessing.__all__), expected)
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(preprocessing, name)))

    def test_region_selector_keeps_legacy_pickle_import_path(self) -> None:
        from awp2.preprocessing.pipeline import _region_columns

        self.assertIs(preprocessing._region_columns, _region_columns)


class SpectralReductionContractTest(unittest.TestCase):
    def test_reductions_are_cloneable_and_preserve_feature_names(self) -> None:
        random = np.random.default_rng(SEED)
        features = pd.DataFrame(
            random.uniform(5, 90, (32, 6)),
            columns=["X681", "X691", "X702", "X712", "X722", "X732"],
        )
        choices = [
            {"use_spectral_standard_scale": True, "spectral_pca_components": 2},
            {"use_spectral_centering": True, "spectral_ica_components": 2},
            {"spectral_nmf_components": 2},
            {"spectral_osp_components": 2},
            {"spectral_lpp_components": 2, "spectral_lpp_neighbors": 3},
            {"spectral_vsrp_components": 2},
            {"use_spectral_minmax_scale": True, "spectral_dbn_layers": (3, 2)},
        ]
        for choice in choices:
            with self.subTest(choice=choice):
                config = PreprocessingConfig(use_meta=False, **choice)
                preprocessor = build_preprocessor(config)
                cloned = clone(preprocessor)
                transformed = preprocessor.fit_transform(features)
                pd.testing.assert_frame_equal(transformed, cloned.fit_transform(features))
                np.testing.assert_array_equal(
                    transformed.columns, preprocessor.get_feature_names_out()
                )
                self.assertEqual(transformed.shape, (len(features), 2))
                self.assertTrue(np.isfinite(transformed.to_numpy()).all())
                self.assertEqual(
                    len(describe_preprocessor(preprocessor)),
                    len(preprocessor.named_transformers_["spectra"].steps),
                )


class PreprocessingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2, 1],
                MONTH_COL: [3, 6, 9],
                "X681": [10.0, 20.0, 30.0],
                "X691": [12.0, 24.0, 36.0],
                "X702": [14.0, 28.0, 42.0],
            },
            index=[4, 8, 12],
        )

    def test_scaling_uses_only_training_statistics(self) -> None:
        preprocessor = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_spectral_standard_scale=True)
        ).fit(self.features)
        validation = self.features.iloc[[0]].assign(X681=40.0)

        transformed = preprocessor.transform(validation)

        self.assertAlmostEqual(transformed.iloc[0]["X681"], 20.0 / np.std([10, 20, 30]))
        self.assertEqual(list(transformed.columns), ["X681", "X691", "X702"])
        pd.testing.assert_index_equal(transformed.index, validation.index)
        pd.testing.assert_frame_equal(
            preprocessor.transform(self.features),
            build_preprocessor(
                PreprocessingConfig(use_meta=False, use_spectral_standard_scale=True)
            ).fit_transform(self.features),
        )

    def test_missing_spectrum_uses_training_medians(self) -> None:
        bands = self.features[["X681", "X691", "X702"]]
        transformer = InterpolateBands().fit(bands)
        missing = bands.iloc[[0]] * np.nan

        transformed = transformer.transform(missing)

        np.testing.assert_allclose(transformed.iloc[0], bands.median())
        pd.testing.assert_index_equal(transformed.index, missing.index)

    def test_interpolation_respects_wide_gaps(self) -> None:
        bands = pd.DataFrame({"X681": [10.0], "X691": [np.nan], "X702": [31.0]})

        nearest = InterpolateBands().fit_transform(bands)
        linear = InterpolateBands(max_interpolation_gap_nm=30).fit_transform(bands)

        self.assertEqual(nearest.iloc[0]["X691"], 10.0)
        self.assertEqual(linear.iloc[0]["X691"], 20.0)

    def test_parallel_branches_have_independent_cleaning_steps(self) -> None:
        preprocessor = build_preprocessor(
            PreprocessingConfig(use_mean_albedo=True, use_principal_components=True)
        )
        branches = dict((name, transformer) for name, transformer, _ in preprocessor.transformers)
        for step_name in ("drop_empty", "interpolate"):
            steps = [
                branches[name].named_steps[step_name]
                for name in ("spectra", "albedo", "principal_components")
            ]
            self.assertEqual(len({id(step) for step in steps}), len(steps))

    def test_legacy_scale_preserves_month_scaling(self) -> None:
        with self.assertWarns(DeprecationWarning):
            legacy = build_preprocessor(PreprocessingConfig(scale=True, use_cyclic_month=False))
        spectral_only = build_preprocessor(
            PreprocessingConfig(use_spectral_standard_scale=True, use_cyclic_month=False)
        )

        legacy_output = legacy.fit_transform(self.features)
        spectral_output = spectral_only.fit_transform(self.features)

        np.testing.assert_allclose(legacy_output[MONTH_COL], [-np.sqrt(1.5), 0, np.sqrt(1.5)])
        pd.testing.assert_series_equal(spectral_output[MONTH_COL], self.features[MONTH_COL])
        pd.testing.assert_frame_equal(
            legacy_output.filter(regex="^X"), spectral_output.filter(regex="^X")
        )

    def test_snv_handles_constant_spectra_without_changing_index(self) -> None:
        bands = self.features[["X681", "X691", "X702"]].copy()
        bands.iloc[0] = 1.0

        transformed = StandardNormalVariate().fit_transform(bands)

        np.testing.assert_allclose(transformed.iloc[0], 0.0)
        np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed.iloc[1:].std(axis=1, ddof=0), 1.0)
        pd.testing.assert_index_equal(transformed.index, bands.index)

    def test_description_preserves_branch_and_step_order(self) -> None:
        preprocessor = build_preprocessor(PreprocessingConfig(use_meta=False))

        self.assertEqual(
            describe_preprocessor(preprocessor),
            [
                "E1. Drop empty bands",
                "E2. Interpolate band gaps (max 15 nm)",
            ],
        )

    def test_description_includes_principal_component_branch(self) -> None:
        preprocessor = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_principal_components=True)
        )

        self.assertEqual(
            describe_preprocessor(preprocessor),
            [
                "G1. Drop empty bands",
                "G2. Interpolate band gaps (max 15 nm)",
                "G3. Standardize bands (per-band mean and std)",
                "G4. PCA to PC_1 and PC_2",
                "E1. Drop empty bands",
                "E2. Interpolate band gaps (max 15 nm)",
            ],
        )

    def test_region_selected_pipeline_survives_pickle_round_trip(self) -> None:
        preprocessor = build_preprocessor(
            PreprocessingConfig(spectral_regions=("Red Edge",), use_mean_albedo=True)
        ).fit(self.features)

        restored = pickle.loads(pickle.dumps(preprocessor))

        pd.testing.assert_frame_equal(
            preprocessor.transform(self.features), restored.transform(self.features)
        )

    def test_empty_bands_are_determined_by_training_only(self) -> None:
        training = self.features.assign(X712=np.nan)
        preprocessor = build_preprocessor(PreprocessingConfig(use_meta=False)).fit(training)

        transformed = preprocessor.transform(training.assign(X712=100.0))

        self.assertEqual(list(transformed.columns), ["X681", "X691", "X702"])


class NearestBandTest(unittest.TestCase):
    def test_returns_closest_band_within_tolerance(self) -> None:
        self.assertEqual(nearest_band(445, ["X437", "X447", "X498"]), "X447")

    def test_rejects_closest_band_outside_tolerance(self) -> None:
        with self.assertRaisesRegex(ValueError, "420 nm; closest is X437"):
            nearest_band(420, ["X437", "X447", "X498"])


class SpectralRegionSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                AEZ_COL: [1, 2],
                MONTH_COL: [3, 4],
                "X671": [999.0, 999.0],
                "X681": [10.0, 12.0],
                "X691": [float("nan"), 16.0],
                "X742": [30.0, 32.0],
                "X752": [999.0, 999.0],
            }
        )

    def test_selection_precedes_interpolation(self) -> None:
        config = PreprocessingConfig(spectral_regions=("Red Edge",), use_meta=False)
        transformer = build_preprocessor(config)
        result = transformer.fit_transform(self.frame)
        self.assertEqual(list(result.columns), ["X681", "X691", "X742"])
        changed = self.frame.assign(X671=-999.0, X752=-999.0)
        pd.testing.assert_frame_equal(result, transformer.transform(changed))
        self.assertTrue(result.notna().all().all())

    def test_empty_selection_is_metadata_only(self) -> None:
        result = build_preprocessor(PreprocessingConfig(spectral_regions=())).fit_transform(
            self.frame
        )
        self.assertTrue(all(not column.startswith("X") for column in result.columns))
        self.assertIn("Month_sin", result)
        self.assertTrue(any(column.startswith("AEZ_") for column in result.columns))

    def test_default_retains_all_bands(self) -> None:
        result = build_preprocessor(PreprocessingConfig(use_meta=False)).fit_transform(self.frame)
        self.assertEqual(list(result.columns), ["X671", "X681", "X691", "X742", "X752"])

    def test_invalid_configs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "known names"):
            PreprocessingConfig(spectral_regions=("unknown",))
        with self.assertRaisesRegex(ValueError, "unique names"):
            PreprocessingConfig(spectral_regions=("Red Edge", "Red Edge"))
        with self.assertRaisesRegex(ValueError, "requires use_meta"):
            PreprocessingConfig(spectral_regions=(), use_meta=False)
        with self.assertRaisesRegex(ValueError, "nonempty spectral_regions"):
            PreprocessingConfig(spectral_regions=(), use_mean_albedo=True)


class VegetationIndexCatalogueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bands = pd.DataFrame(
            {
                "X447": [25.0],
                "X498": [30.0],
                "X529": [32.0],
                "X549": [35.0],
                "X569": [30.0],
                "X630": [40.0],
                "X651": [45.0],
                "X661": [48.0],
                "X671": [50.0],
                "X681": [55.0],
                "X691": [60.0],
                "X702": [65.0],
                "X712": [67.0],
                "X722": [68.0],
                "X732": [69.0],
                "X742": [70.0],
                "X752": [70.0],
                "X763": [72.0],
                "X773": [74.0],
                "X803": [80.0],
                "X824": [82.0],
                "X854": [84.0],
                "X864": [86.0],
                "X1069": [75.0],
                "X1205": [70.0],
                "X1235": [65.0],
                "X1599": [50.0],
                "X1649": [45.0],
                "X1659": [44.0],
            }
        )

    def test_catalogue_uses_mapped_bands_for_representative_formulas(self) -> None:
        transformer = AddVegetationIndices().fit(self.bands)
        transformed = transformer.transform(self.bands)

        self.assertAlmostEqual(transformed.loc[0, "SR_750_705"], 70.0 / 65.0)
        self.assertAlmostEqual(transformed.loc[0, "ND_750_660"], (70.0 - 48.0) / (70.0 + 48.0))
        self.assertAlmostEqual(transformed.loc[0, "MSR_750_445"], (70.0 - 25.0) / (65.0 - 25.0))
        self.assertAlmostEqual(transformed.loc[0, "PRI"], (32.0 - 30.0) / (32.0 + 30.0))
        self.assertAlmostEqual(transformed.loc[0, "NDWI_1070_1200"], (75.0 - 70.0) / (75.0 + 70.0))
        self.assertAlmostEqual(transformed.loc[0, "PSRI"], (55.0 - 30.0) / 70.0)
        self.assertAlmostEqual(
            transformed.loc[0, "OSAVI"],
            1.16 * (0.8 - 0.5) / (0.8 + 0.5 + 0.16),
        )
        self.assertAlmostEqual(
            transformed.loc[0, "TCARI_OSAVI"],
            transformed.loc[0, "TCARI"] / transformed.loc[0, "OSAVI"],
        )
        self.assertEqual(list(transformed.columns), list(transformer.get_feature_names_out()))

    def test_reip_algorithms_locate_the_maximum_red_edge_slope(self) -> None:
        red_edge_bands = [
            column for column in self.bands.columns if 650 <= int(column.removeprefix("X")) <= 763
        ]
        for band in red_edge_bands:
            centered_wavelength = int(band.removeprefix("X")) - 702
            self.bands.loc[0, band] = 50 + centered_wavelength - centered_wavelength**3 / 10000

        transformed = AddVegetationIndices().fit_transform(self.bands)

        self.assertAlmostEqual(transformed.loc[0, "REIP_LAGRANGIAN"], 702.0, delta=5.0)
        self.assertAlmostEqual(transformed.loc[0, "REIP_POLYNOMIAL"], 702.0, delta=5.0)

    def test_preprocessing_config_adds_only_implemented_catalogue_indices(self) -> None:
        features = self.bands.assign(**{AEZ_COL: 1, MONTH_COL: 3})
        transformed = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_vegetation_indices=True)
        ).fit_transform(features)

        self.assertTrue(set(VEGETATION_INDEX_NAMES).issubset(transformed.columns))

    def test_index_branch_uses_the_configured_standard_scaler(self) -> None:
        features = pd.concat([self.bands, self.bands * 2], ignore_index=True).assign(
            **{AEZ_COL: [1, 1], MONTH_COL: [3, 3]}
        )
        transformed = build_preprocessor(
            PreprocessingConfig(
                use_meta=False,
                use_spectral_standard_scale=True,
                use_vegetation_indices=True,
            )
        ).fit_transform(features)

        self.assertAlmostEqual(transformed["OSAVI"].mean(), 0.0)

    def test_snv_does_not_change_physical_reflectance_indices(self) -> None:
        baseline = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_vegetation_indices=True)
        ).fit_transform(self.bands)
        corrected = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_vegetation_indices=True, use_spectral_snv=True)
        ).fit_transform(self.bands)

        pd.testing.assert_frame_equal(
            baseline[list(VEGETATION_INDEX_NAMES)], corrected[list(VEGETATION_INDEX_NAMES)]
        )

    def test_index_branch_imputes_undefined_reip_values(self) -> None:
        features = pd.concat([self.bands, self.bands], ignore_index=True)
        red_edge_bands = [
            column for column in self.bands.columns if 650 <= int(column.removeprefix("X")) <= 763
        ]
        for band in red_edge_bands:
            centered_wavelength = int(band.removeprefix("X")) - 702
            features.loc[0, band] = 50 + centered_wavelength - centered_wavelength**3 / 10000
            features.loc[1, band] = 50 + centered_wavelength
        features = features.assign(**{AEZ_COL: [1, 1], MONTH_COL: [3, 3]})

        transformed = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_vegetation_indices=True)
        ).fit_transform(features)

        self.assertFalse(transformed[list(VEGETATION_INDEX_NAMES)].isna().any().any())

    def test_principal_component_branch_appends_named_components(self) -> None:
        features = pd.concat(
            [self.bands, self.bands * 2, self.bands * 3], ignore_index=True
        ).assign(**{AEZ_COL: [1, 1, 1], MONTH_COL: [3, 3, 3]})
        preprocessor = build_preprocessor(
            PreprocessingConfig(use_meta=False, use_principal_components=True)
        )

        transformed = preprocessor.fit_transform(features)
        component_pipeline = next(
            transformer
            for name, transformer, _columns in preprocessor.transformers_
            if name == "principal_components"
        )

        self.assertTrue({"PC_1", "PC_2"}.issubset(transformed.columns))
        self.assertEqual(
            [name for name, _step in component_pipeline.steps],
            [
                "drop_empty",
                "interpolate",
                "standard_scale",
                "pca",
            ],
        )


class CategoricalContextPreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2],
                MONTH_COL: [3, 9],
                "X427": [0.1, 0.2],
                "X437": [0.3, 0.4],
            }
        )

    def test_categorical_context_preserves_raw_metadata_columns(self) -> None:
        transformed = build_preprocessor(
            PreprocessingConfig(use_categorical_context=True)
        ).fit_transform(self.features)

        self.assertEqual(list(transformed.columns), [AEZ_COL, MONTH_COL, "X427", "X437"])
        pd.testing.assert_frame_equal(
            transformed[[AEZ_COL, MONTH_COL]], self.features[[AEZ_COL, MONTH_COL]]
        )


class MeanAlbedoScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2, 3],
                MONTH_COL: [5, 7, 9],
                "X427": [10.0, 20.0, 45.0],
                "X437": [12.0, 25.0, 50.0],
            }
        )

    def _albedo(self, config: PreprocessingConfig) -> pd.Series:
        return build_preprocessor(config).fit_transform(self.features)[MEAN_ALBEDO_COL]

    def test_albedo_follows_the_standard_scaling_of_the_spectra(self) -> None:
        albedo = self._albedo(
            PreprocessingConfig(use_mean_albedo=True, use_spectral_standard_scale=True)
        )

        self.assertAlmostEqual(albedo.mean(), 0.0)
        self.assertAlmostEqual(albedo.std(ddof=0), 1.0)

    def test_albedo_follows_the_minmax_scaling_of_the_spectra(self) -> None:
        albedo = self._albedo(
            PreprocessingConfig(use_mean_albedo=True, use_spectral_minmax_scale=True)
        )

        self.assertAlmostEqual(albedo.min(), 0.0)
        self.assertAlmostEqual(albedo.max(), 1.0)

    def test_albedo_stays_in_reflectance_units_without_spectral_scaling(self) -> None:
        albedo = self._albedo(PreprocessingConfig(use_mean_albedo=True))

        pd.testing.assert_series_equal(
            albedo, self.features[["X427", "X437"]].mean(axis=1), check_names=False
        )


class AezMonthEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame({AEZ_COL: [5, 5, 9], MONTH_COL: [5, 10, 5]})

    def test_one_indicator_per_training_combination(self) -> None:
        encoded = AezMonthEncoder().fit_transform(self.train)

        self.assertEqual(
            list(encoded.columns), ["aez_month_5_10", "aez_month_5_5", "aez_month_9_5"]
        )
        self.assertEqual(encoded.sum(axis=1).tolist(), [1.0, 1.0, 1.0])

    def test_unseen_combination_gives_all_zeros(self) -> None:
        encoder = AezMonthEncoder().fit(self.train)

        encoded = encoder.transform(pd.DataFrame({AEZ_COL: [9], MONTH_COL: [10]}))

        self.assertEqual(encoded.sum(axis=1).tolist(), [0.0])

    def test_requires_numeric_context(self) -> None:
        with self.assertRaises(ValueError):
            PreprocessingConfig(use_aez_month=True, use_meta=False)
        with self.assertRaises(ValueError):
            PreprocessingConfig(use_aez_month=True, use_categorical_context=True)
