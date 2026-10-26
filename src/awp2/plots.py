"""Shared plotting helpers for EDA, model analysis and reports."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import BarContainer

from awp2.config import (
    ALL_GROUPS_KIND,
    BAND_GAP_NM,
    BAND_PATTERN,
    DOCS_FIGURES_DIR,
    FIGURE_DPI,
    IMPORTANCE_GROUP_COL,
    IMPORTANCE_MEAN_COL,
    IMPORTANCE_METRIC_COL,
    IMPORTANCE_STD_COL,
    META_ONLY_KIND,
    N_MEASURED_BANDS_COL,
    ONLY_GROUP_KIND,
    ONLY_GROUP_NO_META_KIND,
    RECALL_COL,
    REDUCTION_FAMILY_COL,
    REDUCTION_FAMILY_LABELS,
    SAVGOL_SPLIT_GAP_NM,
    SPECTRAL_REGIONS,
    SPECTRAL_REGIONS_NM,
    STD_SUFFIX,
    SUPPORT_COL,
    TARGET_COLS,
    WITHOUT_GROUP_KIND,
    SpectralRegion,
)
from awp2.data import band_columns, wavelength, wavelengths
from awp2.evaluation import METRIC_LABELS, ErrorBreakdown, MetricName, Metrics
from awp2.importance import ImportanceResult


def plot_spectra(
    df: pd.DataFrame,
    by: str | None = None,
    std: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot the mean reflectance spectrum (± one standard deviation) per group.

    Bands that are entirely missing show up as gaps in the curve.

    Args:
        df: Data with band columns, e.g. from ``load_train()``.
        by: Column to group by (e.g. ``"Crop"``); ``None`` plots all rows as one curve.
        std: Draw a shaded ± one standard deviation band.
        ax: Axes to draw on; a new figure is created if omitted.
        title: Optional plot title.

    Returns:
        The axes the spectra were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    bands = band_columns(df)
    x = wavelengths(df)
    groups = df.groupby(by, observed=True) if by else [("all", df)]

    for name, group in groups:
        mean = group[bands].mean()
        (line,) = ax.plot(x, mean.values, label=f"{name} (n={len(group)})", linewidth=1.5)
        if std:
            sd = group[bands].std()
            ax.fill_between(
                x, (mean - sd).values, (mean + sd).values, color=line.get_color(), alpha=0.15
            )

    ax.set_xlabel("Wellenlänge [nm]")
    ax.set_ylabel("Reflektanz [%]")
    if title:
        ax.set_title(title)
    if by:
        ax.legend(title=by, fontsize="small")
    ax.grid(alpha=0.3)
    return ax


def plot_metric_comparison(
    results: Mapping[str, Metrics],
    metrics: Sequence[MetricName] = ("bacc_crop", "bacc_stage", "bacc_combined"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare several models on a few metrics as grouped bars.

    Args:
        results: Metrics per model, keyed by the label to show (e.g. ``{"Dummy": ..., "RF": ...}``).
        metrics: Which metrics to show, one bar group each.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one bar per model and metric, values written on the bars.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    table = pd.DataFrame({label: m.model_dump() for label, m in results.items()}).loc[list(metrics)]
    table.index = [METRIC_LABELS.get(metric, metric) for metric in metrics]
    table.plot.bar(ax=ax, rot=0, width=0.75)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.2f", fontsize="small", padding=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (Validierung)")
    ax.legend(title="Modell", fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_tuning_candidates(
    candidates: pd.DataFrame,
    color_by: str | None = None,
    validation_score: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Show all candidates of a hyperparameter search with their cross-validation score.

    Args:
        candidates: ``TuneResult.candidates`` (columns ``params``, ``mean``, ``std``, ``rank``).
        color_by: Hyperparameter whose values colour the bars, to make its effect visible.
        validation_score: Score of the best candidate on the validation part, drawn as a line to
            compare it with the cross-validation estimate.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one horizontal bar (mean ± std over the folds) per candidate, best on top.
    """
    ordered = candidates.sort_values("mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.45 * len(ordered) + 1.5))
    labels = [", ".join(f"{k}={v}" for k, v in params.items()) for params in ordered["params"]]
    values = (
        [params.get(color_by) for params in ordered["params"]]
        if color_by
        else [None] * len(ordered)
    )
    palette = {value: f"C{number}" for number, value in enumerate(dict.fromkeys(values))}
    ax.barh(
        labels, ordered["mean"], xerr=ordered["std"], color=[palette[v] for v in values], capsize=3
    )
    if color_by:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in palette.values()]
        ax.legend(
            handles, [f"{color_by}={v}" for v in palette], fontsize="small", loc="lower right"
        )
    if validation_score is not None:
        ax.axvline(validation_score, color="black", linestyle="--", linewidth=1)
        ax.text(validation_score, len(ordered) - 0.4, " Validierung", fontsize="small", va="top")
    ax.set_xlabel("CV-Score (Mittelwert ± Std über die Folds)")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_cv_scores(
    candidates: Mapping[str, pd.DataFrame],
    reference: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare the best cross-validation score of several searches, e.g. preprocessing variants.

    Args:
        candidates: ``TuneResult.candidates`` per search, keyed by the label to show.
        reference: CV score drawn as a dashed line, e.g. the variant everything is compared to.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one horizontal bar (best mean ± its std over the folds) per search, best on
            top and the value written next to each bar.
    """
    best = pd.DataFrame(
        {
            label: table.loc[table["rank"].idxmin(), ["mean", "std"]]
            for label, table in candidates.items()
        }
    ).T.sort_values("mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.5 * len(best) + 1.5))
    bars = ax.barh(best.index, best["mean"], xerr=best["std"], color="C0", capsize=3)
    ax.bar_label(bars, fmt="%.3f", fontsize="small", padding=12)
    if reference is not None:
        ax.axvline(reference, color="black", linestyle="--", linewidth=1)
    lower = (best["mean"] - best["std"]).min()
    ax.set_xlim(max(0.0, lower - 0.02), min(1.0, (best["mean"] + best["std"]).max() + 0.03))
    ax.set_xlabel("Bester CV-Score (Mittelwert ± Std über die Folds)")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_class_recall(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> plt.Figure:
    """Recall per class for crop and stage – which classes the model finds worst.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order.

    Returns:
        Figure with one horizontal bar chart per target, weakest class at the bottom.
    """
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(12, 4))
    for ax, col in zip(axes, TARGET_COLS, strict=True):
        hit = y_true[col].to_numpy() == y_pred[col].to_numpy()
        recall = pd.Series(hit, index=y_true[col].to_numpy()).groupby(level=0).agg(["mean", "size"])
        recall = recall.sort_values("mean")
        labels = [f"{cls} (n={n})" for cls, n in zip(recall.index, recall["size"], strict=True)]
        bars = ax.barh(labels, recall["mean"], color="C0")
        ax.bar_label(bars, fmt="%.2f", fontsize="small", padding=2)
        ax.set_xlim(0, 1.1)
        ax.set_title(col)
        ax.set_xlabel("Recall")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_confusions(y_true: pd.DataFrame, y_pred: pd.DataFrame, top: int = 5) -> plt.Figure:
    """The most frequent mix-ups (true → predicted) for crop and stage.

    Args:
        y_true: True labels with the columns ``Crop`` and ``Stage``.
        y_pred: Predicted labels, same columns and row order.
        top: Number of mix-ups to show per target.

    Returns:
        Figure with one horizontal bar chart per target, most frequent mix-up on top.
    """
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(12, 0.5 * top + 1.5))
    for ax, col in zip(axes, TARGET_COLS, strict=True):
        pairs = pd.DataFrame({"true": y_true[col].to_numpy(), "pred": y_pred[col].to_numpy()})
        wrong = pairs[pairs["true"] != pairs["pred"]]
        counts = wrong.value_counts().head(top).iloc[::-1]
        labels = [f"{t} → {p}" for t, p in counts.index]
        bars = ax.barh(labels, counts.to_numpy(), color="C3")
        ax.bar_label(bars, fontsize="small", padding=2)
        ax.set_title(f"{col}: häufigste Verwechslungen")
        ax.set_xlabel("Anzahl (Validierung)")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def save_doc_figure(fig: plt.Figure, name: str, directory: Path = DOCS_FIGURES_DIR) -> None:
    """Save a figure as PNG for the documentation (the folder is versioned in git).

    Args:
        fig: Figure to save.
        name: File name without extension.
        directory: ``DOCS_FIGURES_DIR`` (data/EDA pages) or ``MODEL_DOCS_FIGURES_DIR`` (models).
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")


_ERROR_SHARE_LABELS = {
    "both_correct": "beides richtig",
    "stage_only_wrong": "nur Stadium falsch",
    "crop_wrong": "Kultur falsch",
}
_ERROR_SHARE_COLORS = ("C2", "C1", "C3")
_MIN_LABELLED_SHARE = 0.05  # narrower segments are too small for a readable label


def plot_error_breakdown(
    breakdowns: Mapping[str, ErrorBreakdown], ax: plt.Axes | None = None
) -> plt.Axes:
    """Compare where the errors of several models come from as stacked bars.

    Args:
        breakdowns: ``error_breakdown()`` per model, keyed by the label to show.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one bar per model, split into fully correct, stage-only and crop errors.
    """
    table = pd.DataFrame(
        {label: breakdown.model_dump() for label, breakdown in breakdowns.items()}
    ).T[list(_ERROR_SHARE_LABELS)]
    table.columns = list(_ERROR_SHARE_LABELS.values())
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.6 * len(table) + 1.5))
    table.plot.barh(stacked=True, ax=ax, color=_ERROR_SHARE_COLORS, width=0.7)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = [
                f"{bar.get_width():.2f}" if bar.get_width() >= _MIN_LABELLED_SHARE else ""
                for bar in container
            ]
            ax.bar_label(container, labels=labels, label_type="center", fontsize="small")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Anteil der Zeilen")
    ax.legend(
        fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(table.columns)
    )
    ax.invert_yaxis()
    return ax


def plot_recall_comparison(
    recalls: Mapping[str, pd.DataFrame], top: int | None = None
) -> plt.Figure:
    """Compare the recall per combined crop/stage class of several models.

    Args:
        recalls: ``recall_by_class()`` per model, keyed by the label to show; all for the same
            samples.
        top: Show only the classes where the models differ most (recommended with many
            classes); all classes if omitted.

    Returns:
        Figure with one group of horizontal bars per class (support in the label), classes
            ordered by the largest recall difference between the models.

    Raises:
        ValueError: If the tables do not cover the same samples or ``top`` is not positive.
    """
    if top is not None and top < 1:
        raise ValueError(f"top must be positive, got {top}.")
    supports = [recall[SUPPORT_COL].sort_index() for recall in recalls.values()]
    if not all(support.equals(supports[0]) for support in supports):
        raise ValueError("All recall tables must come from the same samples.")
    support = supports[0]
    table = pd.DataFrame({label: recall[RECALL_COL] for label, recall in recalls.items()})
    spread = table.max(axis=1) - table.min(axis=1)
    table = table.loc[spread.sort_values(ascending=False).index[:top]].iloc[::-1]
    table.index = [f"{pair} (n={support[pair]})" for pair in table.index]
    bar_height = 0.225
    fig, ax = plt.subplots(figsize=(9, bar_height * len(table) * len(recalls) + 1.5))
    table.plot.barh(ax=ax, width=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_rate_by(
    errors: Mapping[str, pd.Series],
    groups: pd.Series,
    xlabel: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare how the error rate of several models changes over a context such as month or AEZ.

    Args:
        errors: Per model a boolean Series (``True`` = wrong), aligned with ``groups``.
        groups: Context value per sample, e.g. ``X_val["Month"]``.
        xlabel: Label of the x axis; the name of ``groups`` if omitted.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one group of bars per context value, its sample count in the tick label.
    """
    table = pd.DataFrame({label: error.groupby(groups).mean() for label, error in errors.items()})
    counts = groups.value_counts()
    table.index = [f"{value}\n(n={counts[value]})" for value in table.index]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.9 * len(table) + 2), 4))
    table.plot.bar(ax=ax, rot=0, width=0.8)
    ax.set_xlabel(xlabel or str(groups.name))
    ax.set_ylabel("Fehlerrate")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_tradeoff(
    curve: pd.DataFrame,
    metric: MetricName = "bacc_combined",
    x: str = N_MEASURED_BANDS_COL,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot quality against input size, one line per reduction family.

    Args:
        curve: Output of ``awp2.reduction.reduction_curve``.
        metric: Score on the y-axis, drawn as mean ± std over the folds.
        x: ``"n_measured_bands"`` (what a sensor must measure) or ``"n_input_features"``
            (what the model receives).
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes; a dashed line marks the reference with all bands.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for family, rows in curve.groupby(REDUCTION_FAMILY_COL, sort=False):
        rows = rows.sort_values(x)
        ax.errorbar(
            rows[x],
            rows[metric],
            yerr=rows[f"{metric}{STD_SUFFIX}"],
            marker="o",
            capsize=3,
            label=REDUCTION_FAMILY_LABELS.get(str(family), str(family)),
        )
    reference = curve.loc[curve[REDUCTION_FAMILY_COL] == "all_bands", metric]
    if not reference.empty:
        ax.axhline(reference.iloc[0], color="grey", linestyle="--", linewidth=1)
    # metadata-only has zero bands, which a plain log axis cannot show
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, curve[x].max() * 1.5)
    ax.set_xlabel("Gemessene Bänder" if x == N_MEASURED_BANDS_COL else "Merkmale im Modell")
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (CV)")
    ax.legend(fontsize="small", loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def plot_band_importance(
    result: ImportanceResult,
    metrics: Sequence[MetricName] = ("bacc_crop", "bacc_stage"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the importance of band windows over the wavelength, spectral regions shaded.

    Args:
        result: Importance with one group per band, e.g. from ``band_window_groups``.
        metrics: One curve (mean ± std) per metric.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with the importance curves; non-band groups are left out.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    table = result.to_frame()
    table = table[table[IMPORTANCE_GROUP_COL].str.fullmatch(BAND_PATTERN)]
    for metric in metrics:
        rows = table[table[IMPORTANCE_METRIC_COL] == metric]
        x = np.array([wavelength(band) for band in rows[IMPORTANCE_GROUP_COL]], dtype=float)
        mean = rows[IMPORTANCE_MEAN_COL].to_numpy(dtype=float)
        std = rows[IMPORTANCE_STD_COL].to_numpy(dtype=float)
        # a NaN after each water-absorption gap keeps the curve from bridging it
        breaks = np.flatnonzero(np.diff(x) > SAVGOL_SPLIT_GAP_NM) + 1
        x, mean, std = (np.insert(values, breaks, np.nan) for values in (x, mean, std))
        (line,) = ax.plot(x, mean, label=METRIC_LABELS.get(metric, metric), linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    _shade_regions(ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Wellenlänge [nm]")
    ax.set_ylabel("Abfall des Scores bei Permutation")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return ax


def plot_group_study(
    table: pd.DataFrame,
    metric: str = "bacc_combined",
    regions: Sequence[SpectralRegion] = SPECTRAL_REGIONS,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare retrained models that see only one region, or all but one, per region.

    Args:
        table: Output of :func:`awp2.group_study.results_table`.
        metric: Cross-validated metric to plot; its ``<metric>_std`` column gives the error bars.
        regions: Regions in plotting order.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with three bars per region and reference lines for the full model, the model
            without metadata and the metadata alone.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.8))
    names = [region.name for region in regions]
    series = (
        (ONLY_GROUP_NO_META_KIND, "nur diese Region (ohne Metadaten)"),
        (ONLY_GROUP_KIND, "nur diese Region + Metadaten"),
        (WITHOUT_GROUP_KIND, "alle anderen Regionen + Metadaten"),
    )
    width = 0.8 / len(series)
    positions = np.arange(len(names))
    for number, (kind, label) in enumerate(series):
        rows = table[table["kind"] == kind].set_index("group").loc[names]
        ax.bar(
            positions + (number - 1) * width,
            rows[metric],
            width,
            yerr=rows[f"{metric}_std"],
            capsize=2,
            label=label,
        )
    references = (
        (table[(table["kind"] == ALL_GROUPS_KIND) & table["use_meta"]], "alle Bänder + Metadaten"),
        (
            table[(table["kind"] == ALL_GROUPS_KIND) & ~table["use_meta"]],
            "alle Bänder, keine Metadaten",
        ),
        (table[table["kind"] == META_ONLY_KIND], "nur Metadaten"),
    )
    for number, (rows, label) in enumerate(references):
        ax.axhline(
            rows[metric].iloc[0],
            color=f"C{3 + number}",
            linestyle=("-", "--", ":")[number],
            linewidth=1.2,
            label=label,
        )
    ax.set_xticks(positions, names)
    ax.set_ylabel(f"CV-{METRIC_LABELS[metric]}")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize="x-small", ncols=3, loc="upper center")
    return ax


def plot_group_against_null(
    observed: pd.Series, null: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """Compare the importance of each region with random band groups of the same size.

    Args:
        observed: Importance per region (index = region name).
        null: Importance of random groups of the region's size, one column per region, one row
            per random group.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with the null distribution as box plot and the region as red dot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    names = list(observed.index)
    ax.boxplot([null[name].dropna() for name in names], tick_labels=names, showfliers=False)
    ax.scatter(range(1, len(names) + 1), observed[names], color="C3", zorder=3, label="Region")
    ax.set_ylabel("Abfall der BAcc beim Mischen")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize="small")
    return ax


def plot_importance_spectrum(
    importance: pd.Series,
    std: pd.Series | None = None,
    regions: Sequence[SpectralRegion] = SPECTRAL_REGIONS,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot the importance of every band over the wavelength with the spectral regions behind it.

    Args:
        importance: Importance per band, indexed by band name (``X<nm>``) – bands only.
        std: Spread per band, drawn as a shaded band; omitted if ``None``.
        regions: Spectral regions to shade and label.
        ax: Axes to draw on; a new figure is created if omitted.
        title: Optional plot title.

    Returns:
        The axes the importance was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))
    x = [wavelength(band) for band in importance.index]
    after_gap = np.flatnonzero(np.diff(x) > BAND_GAP_NM) + 1

    def with_breaks(values: np.ndarray) -> np.ndarray:
        return np.insert(values.astype(float), after_gap, np.nan)

    x_plot = with_breaks(np.asarray(x))
    for number, region in enumerate(regions):
        ax.axvspan(region.start_nm, region.end_nm, color="0.93" if number % 2 else "1", zorder=0)
        ax.text(
            (region.start_nm + region.end_nm) / 2,
            1.01 + 0.05 * (number % 2),
            region.name,
            ha="center",
            fontsize="small",
            transform=ax.get_xaxis_transform(),
        )
    ax.plot(
        x_plot,
        with_breaks(importance.to_numpy()),
        marker="o",
        markersize=3,
        linewidth=1,
        color="C0",
    )
    if std is not None:
        ax.fill_between(
            x_plot,
            with_breaks((importance - std).to_numpy()),
            with_breaks((importance + std).to_numpy()),
            color="C0",
            alpha=0.2,
        )
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlim(min(x), max(x))
    ax.set_xlabel("Wellenlänge [nm]")
    ax.set_ylabel("Abfall der BAcc beim Mischen")
    if title:
        ax.set_title(title, pad=34)
    ax.grid(alpha=0.3)
    return ax


def plot_group_importance(
    result: ImportanceResult,
    metrics: Sequence[MetricName] = ("bacc_crop", "bacc_stage", "bacc_combined"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compare the importance of column groups (e.g. spectral regions) as grouped bars.

    Args:
        result: Importance with a few named groups, e.g. from ``region_groups``.
        metrics: One bar per metric and group, with the std as error bar.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with one bar group per column group, in the order of ``result``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    table = result.to_frame()
    table = table[table[IMPORTANCE_METRIC_COL].isin(metrics)]
    order = list(dict.fromkeys(table[IMPORTANCE_GROUP_COL]))
    means = table.pivot(
        index=IMPORTANCE_GROUP_COL, columns=IMPORTANCE_METRIC_COL, values=IMPORTANCE_MEAN_COL
    ).loc[order, list(metrics)]
    stds = table.pivot(
        index=IMPORTANCE_GROUP_COL, columns=IMPORTANCE_METRIC_COL, values=IMPORTANCE_STD_COL
    ).loc[order, list(metrics)]
    means.columns = [METRIC_LABELS.get(metric, metric) for metric in metrics]
    stds.columns = means.columns
    means.plot.bar(ax=ax, yerr=stds, rot=0, width=0.75, capsize=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Abfall des Scores bei Permutation")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return ax


def _shade_regions(ax: plt.Axes) -> None:
    for position, (region, (low, high)) in enumerate(SPECTRAL_REGIONS_NM.items()):
        ax.axvspan(low, high, color="grey", alpha=0.08 if position % 2 else 0.16, linewidth=0)
        ax.text(
            (low + high) / 2,
            1.0,
            region,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize="small",
        )


def plot_importance_heatmap(
    importance: pd.DataFrame, title: str | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """Show the importance of each feature group (rows) for each class (columns) as a heatmap.

    Args:
        importance: Drop in recall per feature group (index) and class (columns).
        title: Optional plot title.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes with the annotated heatmap.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(1.1 * len(importance.columns) + 2, 0.55 * len(importance) + 1)
        )
    image = ax.imshow(importance.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(importance.columns)), importance.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(importance)), importance.index)
    for row, values in enumerate(importance.to_numpy()):
        for column, value in enumerate(values):
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.figure.colorbar(image, ax=ax, label="Abfall des Recalls")
    if title:
        ax.set_title(title)
    return ax
