import numpy as np
import pandas as pd

from awp2.config import BAND_GROUPING, CROP_KIND, METRIC_KIND, REGION_GROUPING
from awp2.explain import (
    _aggregate,
    _permutation_drops,
    feature_groups,
    random_band_groups,
    size_matched_windows,
)

COLUMNS = ["AEZ_2", "AEZ_5", "Month_sin", "Month_cos", "X437", "X700", "X720", "X1000", "X1356"]


def test_band_groups_keep_metadata_together() -> None:
    groups = feature_groups(COLUMNS, BAND_GROUPING)

    assert groups["AEZ"] == ["AEZ_2", "AEZ_5"]
    assert groups["Month"] == ["Month_sin", "Month_cos"]
    assert groups["X700"] == ["X700"]


def test_random_windows_are_neighbouring_bands_of_the_requested_size() -> None:
    columns = [f"X{nm}" for nm in range(400, 500, 10)] + ["AEZ_1", "Month_sin"]
    groups = random_band_groups(columns, size=4, n_groups=15, contiguous=True)
    bands = [c for c in columns if c.startswith("X")]

    assert len(groups) == len(bands) - 4 + 1
    for members in groups.values():
        start = bands.index(members[0])
        assert members == bands[start : start + 4]
    assert groups == random_band_groups(columns, size=4, n_groups=15, contiguous=True)


def test_scattered_groups_have_distinct_bands_and_ignore_metadata() -> None:
    columns = [f"X{nm}" for nm in range(400, 500, 10)] + ["AEZ_1", "Month_sin"]
    groups = random_band_groups(columns, size=4, n_groups=15, contiguous=False)

    for members in groups.values():
        assert len(set(members)) == 4
        assert all(member.startswith("X") for member in members)


def test_windows_avoid_excluded_bands_and_are_distinct() -> None:
    columns = [f"X{nm}" for nm in range(400, 500, 10)]
    excluded = ["X430", "X440", "X450"]

    groups = random_band_groups(columns, size=3, n_groups=50, exclude=excluded)

    assert 0 < len(groups) < 50
    assert len({tuple(members) for members in groups.values()}) == len(groups)
    assert all(set(members).isdisjoint(excluded) for members in groups.values())


def test_regions_of_equal_size_get_different_windows() -> None:
    columns = [f"X{nm}" for nm in range(400, 700, 10)]

    first = random_band_groups(columns, size=3, n_groups=5, exclude=["X400", "X410", "X420"])
    second = random_band_groups(columns, size=3, n_groups=5, exclude=["X500", "X510", "X520"])

    assert first != second


def test_size_matched_windows_match_the_region_size_and_miss_it() -> None:
    columns = [f"X{nm}" for nm in range(400, 1300, 10)]
    red_edge = feature_groups(columns, REGION_GROUPING)["Red Edge"]

    groups = size_matched_windows(columns, "Red Edge", n_groups=10)

    assert all(len(members) == len(red_edge) for members in groups.values())
    assert all(set(members).isdisjoint(red_edge) for members in groups.values())


def test_no_window_fitting_outside_the_excluded_bands_is_an_error() -> None:
    columns = [f"X{nm}" for nm in range(400, 440, 10)]
    try:
        random_band_groups(columns, size=3, n_groups=5, exclude=["X410"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


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
