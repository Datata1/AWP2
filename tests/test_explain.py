import numpy as np
import pandas as pd

from awp2.config import BAND_GROUPING, CROP_KIND, METRIC_KIND, REGION_GROUPING
from awp2.explain import _aggregate, _permutation_drops, feature_groups

COLUMNS = ["AEZ_2", "AEZ_5", "Month_sin", "Month_cos", "X437", "X700", "X720", "X1000", "X1356"]


def test_band_groups_keep_metadata_together() -> None:
    groups = feature_groups(COLUMNS, BAND_GROUPING)

    assert groups["AEZ"] == ["AEZ_2", "AEZ_5"]
    assert groups["Month"] == ["Month_sin", "Month_cos"]
    assert groups["X700"] == ["X700"]


def test_region_groups_collect_bands_by_wavelength() -> None:
    groups = feature_groups(COLUMNS, REGION_GROUPING)

    assert groups["Red Edge"] == ["X700", "X720"]
    assert groups["NIR"] == ["X1000"]
    assert groups["SWIR 1"] == ["X1356"]
    assert "X700" not in groups


def test_shuffling_the_informative_band_costs_score_but_noise_does_not() -> None:
    rng = np.random.default_rng(0)
    informative = np.repeat([-1.0, 1.0], 100)
    X = pd.DataFrame({"X700": informative, "X720": rng.normal(size=200)})
    y = pd.DataFrame(
        {
            "Crop": np.where(informative > 0, "corn", "cotton"),
            "Stage": "Late",
        }
    )

    def predict(features: pd.DataFrame) -> pd.DataFrame:
        crop = np.where(features["X700"] > 0, "corn", "cotton")
        return pd.DataFrame({"Crop": crop, "Stage": "Late"}, index=features.index)

    baseline, drops = _permutation_drops(
        predict, X, y, feature_groups(list(X.columns), BAND_GROUPING), 5, rng
    )
    records = [(group, 0, repeat, drop) for group, repeat, drop in drops]
    result = _aggregate(pd.Series(baseline), records)
    crop_drop = result.mean[(METRIC_KIND, "bacc_crop")]

    assert baseline[(METRIC_KIND, "bacc_crop")] == 1.0
    assert crop_drop["X700"] > 0.3
    assert crop_drop["X720"] == 0.0
    assert (CROP_KIND, "corn") in result.mean.columns
