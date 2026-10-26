"""Tests for shared preprocessing behavior."""

import unittest

import numpy as np
import pandas as pd

from awp2.config import (
    AEZ_COL,
    CROP_COL,
    MEAN_ALBEDO_COL,
    MONTH_COL,
    SENTINEL2_BANDS,
    STAGE_COL,
    VEGETATION_INDEX_NAMES,
    SensorBand,
)
from awp2.preprocessing import (
    AddVegetationIndices,
    AezMonthEncoder,
    PreprocessingConfig,
    SelectBands,
    SimulateSensor,
    build_preprocessor,
    nearest_band,
)


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
        self.assertAlmostEqual(
            transformed.loc[0, "NDWI_1070_1200"], (75.0 - 70.0) / (75.0 + 70.0)
        )
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
            column
            for column in self.bands.columns
            if 650 <= int(column.removeprefix("X")) <= 763
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

    def test_index_branch_imputes_undefined_reip_values(self) -> None:
        features = pd.concat([self.bands, self.bands], ignore_index=True)
        red_edge_bands = [
            column
            for column in self.bands.columns
            if 650 <= int(column.removeprefix("X")) <= 763
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
        ).assign(
            **{AEZ_COL: [1, 1, 1], MONTH_COL: [3, 3, 3]}
        )
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
        self.assertEqual([name for name, _step in component_pipeline.steps], [
            "drop_empty",
            "interpolate",
            "standard_scale",
            "pca",
        ])


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


class FeatureSetSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2, 3],
                MONTH_COL: [5, 7, 9],
                "X427": [10.0, 20.0, 45.0],
                "X437": [np.nan, np.nan, np.nan],
                "X447": [12.0, 25.0, 50.0],
                "X457": [13.0, 26.0, 51.0],
                "X467": [14.0, 27.0, 52.0],
            }
        )

    def _columns(self, config: PreprocessingConfig) -> list[str]:
        return list(build_preprocessor(config).fit_transform(self.features).columns)

    def test_band_step_counts_only_usable_bands(self) -> None:
        columns = self._columns(PreprocessingConfig(use_meta=False, spectral_band_step=2))

        self.assertEqual(columns, ["X427", "X457"])

    def test_without_spectra_keeps_only_the_other_groups(self) -> None:
        columns = self._columns(PreprocessingConfig(use_spectra=False))

        self.assertFalse(any(column.startswith("X") for column in columns))
        self.assertIn(AEZ_COL + "_1", columns)

    def test_without_spectra_rejects_spectral_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "spectral_band_step"):
            PreprocessingConfig(use_spectra=False, spectral_band_step=2)

    def test_without_spectra_needs_another_feature_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "another feature group"):
            PreprocessingConfig(use_spectra=False, use_meta=False)


class SelectBandsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bands = pd.DataFrame(
            {
                "X427": [1.0, 1.1, 5.0, 5.1],
                "X437": [3.0, 1.0, 3.0, 1.0],
                "X447": [9.0, 9.2, 1.0, 1.2],
            }
        )
        self.targets = pd.DataFrame(
            {CROP_COL: ["corn", "corn", "rice", "rice"], STAGE_COL: ["Late"] * 4}
        )

    def test_keeps_the_most_separating_bands_in_wavelength_order(self) -> None:
        selector = SelectBands(k=2).fit(self.bands, self.targets)

        self.assertEqual(selector.selected_wavelengths_, [427, 447])

    def test_transform_reuses_the_bands_chosen_on_the_training_part(self) -> None:
        selector = SelectBands(k=1).fit(self.bands, self.targets)
        unseen = pd.DataFrame({"X427": [0.0], "X437": [100.0], "X447": [0.0]})

        self.assertEqual(list(selector.transform(unseen).columns), selector.feature_names_out_)

    def test_rejects_more_bands_than_available(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 3"):
            SelectBands(k=4).fit(self.bands, self.targets)

    def test_counts_as_a_reduction_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one"):
            PreprocessingConfig(
                use_spectral_standard_scale=True,
                spectral_pca_components=5,
                spectral_select_k=5,
            )


class SimulateSensorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bands = pd.DataFrame({"X650": [1.0, 4.0], "X660": [3.0, 6.0], "X700": [9.0, 9.0]})
        self.sensor = (SensorBand("red", 655, 20), SensorBand("edge", 700, 4))

    def test_averages_the_bands_inside_each_bandwidth(self) -> None:
        simulated = SimulateSensor(self.sensor).fit_transform(self.bands)

        self.assertEqual(list(simulated.columns), ["X655", "X700"])
        self.assertEqual(list(simulated["X655"]), [2.0, 5.0])

    def test_rejects_a_sensor_band_without_input_bands(self) -> None:
        with self.assertRaisesRegex(ValueError, "nir"):
            SimulateSensor((SensorBand("nir", 850, 10),)).fit(self.bands)

    def test_sentinel2_config_yields_ten_surface_bands(self) -> None:
        wavelengths_nm = range(430, 2400, 10)
        spectra = pd.DataFrame(
            [[0.3] * len(wavelengths_nm)], columns=[f"X{nm}" for nm in wavelengths_nm]
        )
        simulated = build_preprocessor(
            PreprocessingConfig(use_meta=False, spectral_sensor="sentinel2")
        ).fit_transform(spectra)

        expected = [f"X{band.center_nm}" for band in SENTINEL2_BANDS]
        self.assertEqual(list(simulated.columns), expected)
