"""Exhaustive, resumable searches on the shared training folds."""

import argparse
import hashlib
import itertools
import json
import math
import random
import sqlite3
import subprocess
import time
from bisect import bisect_right
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from awp2.config import (
    AUTOML_COMPONENTS,
    AUTOML_DBN_LAYERS,
    AUTOML_DIR,
    AUTOML_EXPERIMENT,
    AUTOML_FOCUSED_DIR,
    AUTOML_FOCUSED_MODEL_GRIDS,
    AUTOML_FOCUSED_PREPROCESSING,
    AUTOML_INTERPOLATION_GAPS,
    AUTOML_LPP_NEIGHBORS,
    AUTOML_MODEL_GRIDS,
    AUTOML_NO_SAMPLE_WEIGHT_FAMILIES,
    AUTOML_SAMPLE_WEIGHT_FAMILIES,
    AUTOML_SAVGOL,
    AUTOML_VALIDATION_SIZE,
    CLEAN_TRAIN_FILE,
    CV_FOLDS,
    PROJECT_ROOT,
    SEED,
    SPECTRAL_REGIONS,
    SPLIT_CONFIG_FILE,
    SPLIT_FILE,
)
from awp2.data import load_folds, load_split, load_split_config
from awp2.experiment import RunConfig, TuneConfig, run, tune
from awp2.models.combined import CombinedLabelClassifier
from awp2.models.hierarchical import HierarchicalClassifier
from awp2.models.linear import BalancedLinearDiscriminantAnalysis
from awp2.models.plsda import PLSDAClassifier
from awp2.preprocessing import PreprocessingConfig


class SearchConfig(BaseModel):
    """Select a reproducible subset of the exhaustive search.

    Attributes:
        models: Model families included at each prediction level.
        approaches: Combined labels and/or hard hierarchical routing.
        shard: Zero-based shard for independent workers.
        shards: Total number of disjoint workers.
        region_subsets: Include every nonempty subset of the spectral regions.
        profile: Broad screening or the finite, balanced follow-up grid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    models: tuple[str, ...] = Field(
        default=tuple(AUTOML_MODEL_GRIDS),
        description="Registered classical model families, used at both hierarchy levels.",
    )
    approaches: tuple[Literal["combined", "hierarchical"], ...] = Field(
        default=("combined", "hierarchical"), description="Prediction strategies to enumerate."
    )
    shard: int = Field(default=0, ge=0, description="Worker index, starting at zero.")
    shards: int = Field(default=1, ge=1, description="Number of independent workers.")
    region_subsets: bool = Field(
        default=True, description="Enumerate every nonempty subset of SPECTRAL_REGIONS."
    )
    profile: Literal["broad", "focused"] = Field(
        default="broad", description="Broad screening or a finite balanced follow-up search."
    )

    @model_validator(mode="after")
    def _check_search(self) -> Self:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Select nonempty, unique model families.")
        if not self.approaches or len(set(self.approaches)) != len(self.approaches):
            raise ValueError("Select nonempty, unique approaches.")
        if set(self.models) - set(AUTOML_MODEL_GRIDS):
            raise ValueError("Unknown model family.")
        if self.shard >= self.shards:
            raise ValueError("Shard index must be smaller than shard count.")
        if self.profile == "focused":
            allowed = {
                *(f"combined_{name}" for name in self.models if "combined" in self.approaches),
                *(
                    f"hierarchical_{crop}_{stage}"
                    for crop, stage in itertools.product(self.models, repeat=2)
                    if "hierarchical" in self.approaches
                ),
            }
            if not allowed.intersection(AUTOML_FOCUSED_MODEL_GRIDS):
                raise ValueError("The selected families/approaches have no focused model space.")
        return self


def preprocessing_candidates() -> Iterator[PreprocessingConfig]:
    """Yield every valid, nonredundant configuration in the declared finite grid.

    Yields:
        Validated preprocessing configurations.
    """
    metadata = (
        {"use_meta": False, "use_cyclic_month": False},
        *(
            {"use_meta": True, "use_cyclic_month": cyclic, "use_aez_month": interaction}
            for cyclic, interaction in itertools.product((False, True), repeat=2)
        ),
    )
    scalers = (
        {},
        {"use_spectral_standard_scale": True},
        {"use_spectral_minmax_scale": True},
        {"use_spectral_centering": True},
    )
    reductions = [{}]
    for method in ("pca", "ica", "nmf", "osp", "vsrp"):
        reductions.extend({f"spectral_{method}_components": count} for count in AUTOML_COMPONENTS)
    reductions.extend(
        {"spectral_lpp_components": count, "spectral_lpp_neighbors": neighbors}
        for count, neighbors in itertools.product(AUTOML_COMPONENTS, AUTOML_LPP_NEIGHBORS)
    )
    reductions.extend({"spectral_dbn_layers": layers} for layers in AUTOML_DBN_LAYERS)
    for meta in metadata:
        if meta["use_meta"]:
            yield PreprocessingConfig.model_validate({**meta, "spectral_regions": ()})
    for meta, scaler, reduction, smoothing, flags in itertools.product(
        metadata, scalers, reductions, AUTOML_SAVGOL, itertools.product((False, True), repeat=4)
    ):
        snv, albedo, indices, principal = flags
        window, order = smoothing
        try:
            yield PreprocessingConfig.model_validate(
                {
                    **meta,
                    **scaler,
                    **reduction,
                    "use_spectral_snv": snv,
                    "use_mean_albedo": albedo,
                    "use_vegetation_indices": indices,
                    "use_principal_components": principal,
                    "use_savgol": window is not None,
                    "savgol_window_length": window,
                    "savgol_polyorder": order,
                }
            )
        except ValidationError:
            continue


def _families() -> dict[str, tuple[ClassifierMixin, dict[str, list[Any]]]]:
    estimators = {
        "svm": SVC(class_weight="balanced", random_state=SEED),
        "rf": RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        "et": ExtraTreesClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        "logreg": LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=3000),
        "knn": KNeighborsClassifier(),
        "mlp": MLPClassifier(random_state=SEED, early_stopping=True, max_iter=500),
        "lda": BalancedLinearDiscriminantAnalysis(solver="lsqr"),
        "plsda": PLSDAClassifier(),
        "gnb": GaussianNB(),
    }
    return {name: (model, AUTOML_MODEL_GRIDS[name]) for name, model in estimators.items()}


def _broad_model_spaces(
    config: SearchConfig,
) -> Iterator[tuple[str, Any, dict[str, list[Any]], bool]]:
    families = _families()
    for name in config.models:
        if name not in families:
            raise ValueError(f"Unknown model family: {name}")
    if "combined" in config.approaches:
        for name in config.models:
            estimator, grid = families[name]
            yield (
                f"combined_{name}",
                CombinedLabelClassifier(estimator),
                {f"estimator__{key}": values for key, values in grid.items()},
                name in AUTOML_SAMPLE_WEIGHT_FAMILIES,
            )
    if "hierarchical" in config.approaches:
        for crop, stage in itertools.product(config.models, repeat=2):
            weighted = any(name in AUTOML_SAMPLE_WEIGHT_FAMILIES for name in (crop, stage))
            if weighted and any(name in AUTOML_NO_SAMPLE_WEIGHT_FAMILIES for name in (crop, stage)):
                continue
            crop_model, crop_grid = families[crop]
            stage_model, stage_grid = families[stage]
            if weighted:
                crop_model, stage_model = clone(crop_model), clone(stage_model)
                for estimator in (crop_model, stage_model):
                    if "class_weight" in estimator.get_params():
                        estimator.set_params(class_weight=None)
            yield (
                f"hierarchical_{crop}_{stage}",
                HierarchicalClassifier(crop_model, stage_model),
                {
                    **{f"crop_estimator__{key}": values for key, values in crop_grid.items()},
                    **{f"stage_estimator__{key}": values for key, values in stage_grid.items()},
                },
                weighted,
            )


def _model_spaces(config: SearchConfig) -> Iterator[tuple[str, Any, dict[str, list[Any]], bool]]:
    for name, model, grid, weighted in _broad_model_spaces(config):
        if config.profile == "focused":
            if name not in AUTOML_FOCUSED_MODEL_GRIDS:
                continue
            grid = AUTOML_FOCUSED_MODEL_GRIDS[name]
        yield name, model, grid, weighted


def _focused_preprocessing() -> list[PreprocessingConfig]:
    return [PreprocessingConfig.model_validate(options) for options in AUTOML_FOCUSED_PREPROCESSING]


def _focused_tasks(config: SearchConfig) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    spaces = list(_model_spaces(config))
    preprocessors = _focused_preprocessing()
    grids = [list(ParameterGrid(grid)) for _, _, grid, _ in spaces]
    positions = itertools.product(range(max(map(len, grids))), preprocessors, range(len(spaces)))
    for position, (parameter_index, preprocessing, space_index) in enumerate(positions):
        if parameter_index >= len(grids[space_index]) or position % config.shards != config.shard:
            continue
        name, model, _, weighted = spaces[space_index]
        payload = {
            "name": name,
            "preprocessing": preprocessing.model_dump(mode="json"),
            "params": grids[space_index][parameter_index],
            "balance_samples": weighted,
        }
        task = hashlib.sha256(_json(payload).encode()).hexdigest()
        yield task, model, payload


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _manifest(config: SearchConfig) -> dict[str, Any]:
    if load_split_config().validation_size != AUTOML_VALIDATION_SIZE:
        raise ValueError("Expected shared 80/20 artifacts; rebuild them explicitly first.")
    split = load_split()
    folds = load_folds()
    if len(folds) != CV_FOLDS:
        raise ValueError("Unexpected number of shared CV folds.")
    return {
        "config": config.model_dump(exclude={"shard", "shards"}),
        "seed": SEED,
        "folds": len(folds),
        "train_rows": len(split.y_train),
        "validation_rows": len(split.y_val),
        "plan": {key: value for key, value in _plan(config).items() if key != "shards"},
        "data": {
            str(path.relative_to(PROJECT_ROOT)): _digest(path)
            for path in (
                CLEAN_TRAIN_FILE,
                SPLIT_FILE,
                SPLIT_CONFIG_FILE,
            )
        },
        "source": {
            str(path.relative_to(PROJECT_ROOT)): _digest(path)
            for path in sorted((PROJECT_ROOT / "src" / "awp2").rglob("*.py"))
        },
        "environment": _digest(PROJECT_ROOT / "uv.lock"),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
    }


def _connect(directory: Path, config: SearchConfig) -> sqlite3.Connection:
    manifest = _json(_manifest(config))
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "search.sqlite", timeout=60)
    connection.execute("CREATE TABLE IF NOT EXISTS manifest (content TEXT PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS results "
        "(task TEXT PRIMARY KEY, status TEXT, score REAL, payload TEXT)"
    )
    existing = connection.execute("SELECT content FROM manifest").fetchone()
    if existing is not None and existing[0] != manifest:
        connection.close()
        raise ValueError("Source, environment, data or search space changed; use a new directory.")
    if existing is None:
        connection.execute("INSERT INTO manifest VALUES (?)", (manifest,))
    connection.commit()
    return connection


def _region_choices(config: SearchConfig) -> list[tuple[str, ...] | None]:
    if not config.region_subsets:
        return [None]
    names = tuple(region.name for region in SPECTRAL_REGIONS)
    return [
        subset
        for size in range(1, len(names) + 1)
        for subset in itertools.combinations(names, size)
    ]


def _tasks(config: SearchConfig) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    if config.profile == "focused":
        yield from _focused_tasks(config)
        return
    base = list(preprocessing_candidates())
    metadata = [item for item in base if item.spectral_regions == ()]
    spectral = [item for item in base if item.spectral_regions != ()]
    regions = _region_choices(config)
    expansions = len(regions) * len(AUTOML_INTERPOLATION_GAPS)
    spaces = list(_model_spaces(config))
    ends = list(itertools.accumulate(len(ParameterGrid(grid)) for _, _, grid, _ in spaces))
    per_preprocessing = ends[-1]
    total = (len(metadata) + len(spectral) * expansions) * per_preprocessing
    generator = random.Random(SEED)
    stride = generator.randrange(1, total)
    while math.gcd(stride, total) != 1:
        stride += 1
    start = generator.randrange(total)
    for position in range(config.shard, total, config.shards):
        index = (position * stride + start) % total
        prep_index, model_index = divmod(index, per_preprocessing)
        if prep_index < len(metadata):
            preprocessing = metadata[prep_index]
        else:
            base_index, expansion = divmod(prep_index - len(metadata), expansions)
            region_index, gap_index = divmod(expansion, len(AUTOML_INTERPOLATION_GAPS))
            preprocessing = spectral[base_index].model_copy(
                update={
                    "spectral_regions": regions[region_index],
                    "max_interpolation_gap_nm": AUTOML_INTERPOLATION_GAPS[gap_index],
                }
            )
        space_index = bisect_right(ends, model_index)
        name, model, grid, weighted = spaces[space_index]
        offset = ends[space_index - 1] if space_index else 0
        params = ParameterGrid(grid)[model_index - offset]
        payload = {
            "name": name,
            "preprocessing": preprocessing.model_dump(mode="json"),
            "params": params,
            "balance_samples": weighted,
        }
        task = hashlib.sha256(_json(payload).encode()).hexdigest()
        yield task, model, payload


def _plan(config: SearchConfig) -> dict[str, int]:
    if config.profile == "focused":
        spaces = list(_model_spaces(config))
        preprocessing_count = len(_focused_preprocessing())
        candidates = preprocessing_count * sum(len(ParameterGrid(grid)) for _, _, grid, _ in spaces)
        return {
            "preprocessing_variants": preprocessing_count,
            "model_spaces": len(spaces),
            "candidates_all_shards": candidates,
            "cv_pipeline_fits_all_shards": candidates * CV_FOLDS,
            "spectral_region_choices": 1,
            "shards": config.shards,
        }
    base = list(preprocessing_candidates())
    metadata_count = sum(preprocessing.spectral_regions == () for preprocessing in base)
    preprocessing_count = metadata_count + (
        (len(base) - metadata_count) * len(_region_choices(config)) * len(AUTOML_INTERPOLATION_GAPS)
    )
    spaces = list(_model_spaces(config))
    candidates = preprocessing_count * sum(len(ParameterGrid(grid)) for _, _, grid, _ in spaces)
    return {
        "preprocessing_variants": preprocessing_count,
        "model_spaces": len(spaces),
        "candidates_all_shards": candidates,
        "cv_pipeline_fits_all_shards": candidates * CV_FOLDS,
        "spectral_region_choices": len(_region_choices(config)),
        "shards": config.shards,
    }


def _search(directory: Path, config: SearchConfig, max_tasks: int | None) -> None:
    connection = _connect(directory, config)
    completed = 0
    lock = directory / f"search-{config.shard}.lock"
    try:
        with lock.open("x", encoding="utf-8") as output:
            output.write(_json(config.model_dump()))
    except FileExistsError:
        connection.close()
        raise RuntimeError(
            "Worker lock exists; stop the worker before removing a stale lock."
        ) from None
    try:
        if (directory / "validation.json").exists():
            raise ValueError("Holdout evaluation started; this search is frozen.")
        for task, model, payload in _tasks(config):
            if (directory / "validation.json").exists():
                raise ValueError("Holdout evaluation started; this search is frozen.")
            if connection.execute("SELECT 1 FROM results WHERE task = ?", (task,)).fetchone():
                continue
            if max_tasks is not None and completed >= max_tasks:
                break
            started = time.monotonic()
            print(f"Starting {task[:12]}: {payload['name']}", flush=True)
            try:
                preprocessing = PreprocessingConfig.model_validate_json(
                    _json(payload["preprocessing"])
                )
                tuned = tune(
                    model,
                    TuneConfig(
                        name=f"automl_{task}",
                        approach="automl",
                        experiment=AUTOML_EXPERIMENT,
                        description=f"Exhaustive 80/20 search: {payload['name']}; task {task}",
                        preprocessing=preprocessing,
                        balance_samples=payload["balance_samples"],
                        param_grid={key: [value] for key, value in payload["params"].items()},
                    ),
                )
                if not math.isfinite(tuned.best_score):
                    raise ValueError("Non-finite CV score; candidate cannot be ranked.")
                if tuned.run_id is None:
                    raise RuntimeError("MLflow tracking failed; inspect tracking before resuming.")
                payload.update(
                    run_id=tuned.run_id,
                    score=tuned.best_score,
                    std=float(tuned.candidates.iloc[0]["std"]),
                    seconds=time.monotonic() - started,
                )
                status, score = "ok", tuned.best_score
            except (ValueError, ArithmeticError) as error:
                payload.update(
                    error=f"{type(error).__name__}: {error}", seconds=time.monotonic() - started
                )
                status, score = "failed", None
            connection.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?)", (task, status, score, _json(payload))
            )
            connection.commit()
            completed += 1
            print(f"Finished {task[:12]}: {status}, CV={score}", flush=True)
    finally:
        connection.close()
        lock.unlink()


def _winner(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT payload FROM results WHERE status = 'ok' ORDER BY score DESC, task LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("No successful candidate exists.")
    return json.loads(row[0])


def _evaluate(directory: Path, config: SearchConfig, allow_partial: bool) -> None:
    connection = _connect(directory, config)
    try:
        if list(directory.glob("search-*.lock")):
            raise ValueError("Stop all search workers before evaluating the CV winner.")
        total = connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        planned = _plan(config)["candidates_all_shards"]
        if total != planned and not allow_partial:
            raise ValueError(
                f"Only {total}/{planned} candidates attempted. Finish the search or explicitly "
                "freeze this partial search with --allow-partial."
            )
        winner = _winner(connection)
        model = next(model for name, model, _, _ in _model_spaces(config) if name == winner["name"])
        params = {
            key: tuple(value) if key.endswith("hidden_layer_sizes") else value
            for key, value in winner["params"].items()
        }
        model.set_params(**params)
        preprocessing = PreprocessingConfig.model_validate_json(_json(winner["preprocessing"]))
        record = {"status": "started", "winner": winner, "attempted": total, "planned": planned}
        marker = directory / "validation.json"
        with marker.open("x", encoding="utf-8") as output:
            output.write(_json(record))
        result = run(
            model,
            RunConfig(
                name="automl_final",
                approach="automl",
                experiment=AUTOML_EXPERIMENT,
                description=f"Single holdout evaluation of CV winner: {winner['name']}",
                preprocessing=preprocessing,
                balance_samples=winner["balance_samples"],
                tuning_run=winner["run_id"],
            ),
        )
        record.update(status="completed", metrics=result.metrics.model_dump(), run_id=result.run_id)
        marker.write_text(_json(record) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
    finally:
        connection.close()


def _report(directory: Path) -> None:
    import csv

    database = directory / "search.sqlite"
    if not database.exists():
        raise ValueError("No search database exists.")
    connection = sqlite3.connect(database)
    try:
        counts = dict(connection.execute("SELECT status, COUNT(*) FROM results GROUP BY status"))
        manifest = json.loads(connection.execute("SELECT content FROM manifest").fetchone()[0])
        planned = manifest.get("plan", {}).get("candidates_all_shards", "unbekannt")
        with (directory / "candidates.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("task", "status", "cv_bacc_combined", "details_json"))
            writer.writerows(
                connection.execute(
                    "SELECT task, status, score, payload FROM results ORDER BY score DESC, task"
                )
            )
        lines = [
            "# AutoML: Ergebnisbericht",
            "",
            f"Abgeschlossene Kandidaten: `{counts}`",
            f"Geplante Kandidaten: {planned}",
            "",
            "Ranking ausschliesslich nach CV-BAcc kombiniert; keine Holdout-Auswahl.",
            "",
            "| Modell | CV-BAcc | Standardabweichung | MLflow |",
            "| --- | --- | --- | --- |",
        ]
        for (payload,) in connection.execute(
            "SELECT payload FROM results WHERE status = 'ok' ORDER BY score DESC, task LIMIT 20"
        ):
            candidate = json.loads(payload)
            lines.append(
                f"| {candidate['name']} | {candidate['score']:.4f} | "
                f"{candidate['std']:.4f} | {candidate['run_id']} |"
            )
        if counts.get("ok", 0):
            lines.extend(
                (
                    "",
                    "## CV-Gewinner",
                    "",
                    "```json",
                    json.dumps(_winner(connection), indent=2),
                    "```",
                )
            )
        marker = directory / "validation.json"
        if marker.exists():
            lines.extend(
                (
                    "",
                    "## Einmalige Validierung",
                    "",
                    "```json",
                    json.dumps(json.loads(marker.read_text(encoding="utf-8")), indent=2),
                    "```",
                )
            )
        else:
            lines.extend(("", "Validierung noch nicht ausgefuehrt."))
        (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Report: {directory / 'report.md'}")
    finally:
        connection.close()


def main() -> None:
    """Parse commands without starting a search unless explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "search", "evaluate", "report"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=("broad", "focused"), default="broad")
    parser.add_argument("--models", nargs="+", default=list(SearchConfig().models))
    parser.add_argument(
        "--approaches",
        nargs="+",
        choices=("combined", "hierarchical"),
        default=["combined", "hierarchical"],
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--full-spectrum-only", action="store_true")
    args = parser.parse_args()
    config = SearchConfig(
        models=tuple(args.models),
        approaches=tuple(args.approaches),
        shard=args.shard,
        shards=args.shards,
        region_subsets=args.profile == "broad" and not args.full_spectrum_only,
        profile=args.profile,
    )
    directory = args.output or (AUTOML_FOCUSED_DIR if args.profile == "focused" else AUTOML_DIR)
    if config.shard >= config.shards:
        parser.error("--shard must be smaller than --shards")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    if args.command == "plan":
        print(json.dumps(_plan(config), indent=2))
    elif args.command == "search":
        _search(directory, config, args.max_tasks)
    elif args.command == "evaluate":
        _evaluate(directory, config, args.allow_partial)
    else:
        _report(directory)


if __name__ == "__main__":
    main()
