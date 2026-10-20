"""Model-agnostic permutation importance on groups of input columns.

Groups are permuted jointly on the raw input (before preprocessing), so the importance refers
to wavelengths and metadata whatever reduction the pipeline applies afterwards.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from awp2.config import (
    AEZ_COL,
    IMPORTANCE_BASELINE_COL,
    IMPORTANCE_DROP_COL,
    IMPORTANCE_GROUP_COL,
    IMPORTANCE_MEAN_COL,
    IMPORTANCE_METRIC_COL,
    IMPORTANCE_REPEAT_COL,
    IMPORTANCE_STD_COL,
    META_GROUP,
    MONTH_COL,
    SEED,
    SPECTRAL_REGIONS_NM,
)
from awp2.data import band_columns, wavelength
from awp2.evaluation import MetricName, as_target_frame, evaluate

Groups = Mapping[str, Sequence[str]]
"""Columns permuted together, keyed by the group name."""


class ImportanceConfig(BaseModel):
    """Options of the permutation importance."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    n_repeats: int = Field(
        default=5, gt=0, description="Random permutations per group; more give a smaller std."
    )
    metrics: tuple[MetricName, ...] = Field(
        default=("bacc_crop", "bacc_stage", "bacc_combined"),
        description="Metrics whose drop is reported; crop and stage are kept apart because "
        "different bands may carry them.",
    )


class GroupImportance(BaseModel):
    """Drop of one metric when one group is permuted."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    group: str = Field(description="Name of the permuted group, e.g. a region or a band.")
    metric: MetricName = Field(description="Metric the drop refers to.")
    mean: float = Field(description="Mean drop (baseline minus permuted score).")
    std: float = Field(description="Standard deviation of the drop over repeats and folds.")


class ImportanceResult(BaseModel):
    """Permutation importance of all groups, one entry per group and metric."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    baseline: dict[str, float] = Field(description="Unpermuted score per metric.")
    importances: tuple[GroupImportance, ...] = Field(description="Drop per group and metric.")

    def to_frame(self) -> pd.DataFrame:
        """Return the importances as a table.

        Returns:
            One row per group and metric with the columns group, metric, mean and std.
        """
        return pd.DataFrame(
            [entry.model_dump() for entry in self.importances],
            columns=[
                IMPORTANCE_BASELINE_COL,
                IMPORTANCE_DROP_COL,
                IMPORTANCE_GROUP_COL,
                IMPORTANCE_METRIC_COL,
                IMPORTANCE_MEAN_COL,
                IMPORTANCE_STD_COL,
            ],
        )


def region_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Group the measured bands by spectral region and the metadata into ``Meta``.

    Args:
        X: Raw features (metadata + bands); bands without any value are left out.

    Returns:
        Columns per region of ``SPECTRAL_REGIONS_NM`` plus ``Meta``; empty regions are dropped.
    """
    bands = _measured_bands(X)
    groups = {
        region: [band for band in bands if low <= wavelength(band) < high]
        for region, (low, high) in SPECTRAL_REGIONS_NM.items()
    }
    groups[META_GROUP] = [column for column in (AEZ_COL, MONTH_COL) if column in X.columns]
    return {name: columns for name, columns in groups.items() if columns}


def band_window_groups(X: pd.DataFrame, half_width: int = 1) -> dict[str, list[str]]:
    """One group per measured band together with its neighbours.

    A lone band is replaced by its almost identical neighbour, so its permutation shows hardly
    any drop; the window gives a curve of importance over the wavelength instead.

    Args:
        X: Raw features (metadata + bands); bands without any value are left out.
        half_width: Number of measured neighbours on each side of the centre band.

    Returns:
        Columns per window, keyed by the centre band (e.g. ``"X702"``).
    """
    bands = _measured_bands(X)
    return {
        band: bands[max(0, position - half_width) : position + half_width + 1]
        for position, band in enumerate(bands)
    }


def permutation_drops(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: Groups,
    config: ImportanceConfig | None = None,
) -> pd.DataFrame:
    """Score drops of a fitted model for every group, metric and permutation.

    Args:
        model: Fitted pipeline (preprocessing + model) predicting ``Crop`` and ``Stage``.
        X: Held-out raw features the model was not fitted on.
        y: True ``Crop``/``Stage`` of ``X``.
        groups: Columns to permute together, e.g. from ``region_groups``.
        config: Repeats and metrics; the defaults of ``ImportanceConfig`` if omitted.

    Returns:
        Long table with the columns group, metric, repeat, drop and baseline.
    """
    config = config or ImportanceConfig()
    baseline = _scores(model, X, y, config.metrics)
    rng = np.random.default_rng(SEED)
    rows = []
    for repeat in range(config.n_repeats):
        order = rng.permutation(len(X))
        for group, columns in groups.items():
            permuted = X.copy()
            permuted[list(columns)] = X[list(columns)].to_numpy()[order]
            scores = _scores(model, permuted, y, config.metrics)
            rows.extend(
                {
                    IMPORTANCE_GROUP_COL: group,
                    IMPORTANCE_METRIC_COL: metric,
                    IMPORTANCE_REPEAT_COL: repeat,
                    IMPORTANCE_DROP_COL: baseline[metric] - scores[metric],
                    IMPORTANCE_BASELINE_COL: baseline[metric],
                }
                for metric in config.metrics
            )
    return pd.DataFrame(rows)


def summarize_drops(drops: pd.DataFrame) -> ImportanceResult:
    """Aggregate drops, e.g. from several folds, into mean and std per group and metric.

    Args:
        drops: Output of ``permutation_drops``, possibly concatenated over folds.

    Returns:
        Importance per group and metric, in the group order of ``drops``.
    """
    keys = [IMPORTANCE_GROUP_COL, IMPORTANCE_METRIC_COL]
    stats = (
        drops.groupby(keys, sort=False)[IMPORTANCE_DROP_COL]
        .agg(**{IMPORTANCE_MEAN_COL: "mean", IMPORTANCE_STD_COL: "std"})
        .fillna(0.0)
    )
    baseline = drops.groupby(IMPORTANCE_METRIC_COL, sort=False)[IMPORTANCE_BASELINE_COL].mean()
    return ImportanceResult(
        baseline={str(metric): float(score) for metric, score in baseline.items()},
        importances=tuple(
            GroupImportance(group=str(group), metric=metric, mean=float(mean), std=float(std))
            for (group, metric), mean, std in zip(
                stats.index, stats[IMPORTANCE_MEAN_COL], stats[IMPORTANCE_STD_COL], strict=True
            )
        ),
    )


def permutation_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: Groups,
    config: ImportanceConfig | None = None,
) -> ImportanceResult:
    """Permutation importance of column groups for a fitted model on held-out data.

    Args:
        model: Fitted pipeline (preprocessing + model) predicting ``Crop`` and ``Stage``.
        X: Held-out raw features the model was not fitted on.
        y: True ``Crop``/``Stage`` of ``X``.
        groups: Columns to permute together, e.g. from ``region_groups``.
        config: Repeats and metrics; the defaults of ``ImportanceConfig`` if omitted.

    Returns:
        Mean and std of the score drop per group and metric.
    """
    return summarize_drops(permutation_drops(model, X, y, groups, config))


def _measured_bands(X: pd.DataFrame) -> list[str]:
    return [band for band in band_columns(X) if X[band].notna().any()]


def _scores(
    model: Pipeline, X: pd.DataFrame, y: pd.DataFrame, metrics: Sequence[MetricName]
) -> dict[str, float]:
    result = evaluate(y, as_target_frame(model.predict(X), y.index))
    return {metric: getattr(result, metric) for metric in metrics}
