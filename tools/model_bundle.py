"""Export and verify a trusted fitted AutoML pipeline without training or tracking writes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUNTIME_PACKAGES = ("mlflow", "scikit-learn", "numpy", "scipy", "pandas", "cloudpickle")


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _checksums(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "checksums.json"
    }


def _check_bundle(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    for relative, expected in _read_json(directory / "checksums.json").items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Bundle file missing, changed or outside bundle: {relative}")
    metadata = _read_json(directory / "bundle.json")
    if platform.python_version() != metadata["python"]:
        raise ValueError("Use the recorded Python version to load this trusted pickle.")
    for name, expected in metadata["runtime_packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Different dependency version: {name}; expected {expected}")
    sys.dont_write_bytecode = True
    source = directory / "project" / "src"
    sys.path.insert(0, str(source))
    import awp2

    if awp2.__file__ is None or not Path(awp2.__file__).resolve().is_relative_to(source):
        raise ValueError("The model must use the bundled awp2 code, not an editable checkout.")
    return metadata


def _verify(directory: Path) -> dict[str, Any]:
    metadata = _check_bundle(directory)
    import mlflow.sklearn
    import pandas as pd

    from awp2.data.split import CropStage
    from awp2.evaluation import as_target_frame, evaluate

    model = mlflow.sklearn.load_model(str(directory / "pipeline"))
    features = pd.read_parquet(directory / "reference" / "features.parquet")
    expected = pd.read_parquet(directory / "reference" / "predictions.parquet")
    truth = pd.read_parquet(directory / "reference" / "targets.parquet")
    actual = as_target_frame(model.predict(features), features.index)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    pairs = frozenset(CropStage.model_validate(pair) for pair in metadata["valid_combinations"])
    metrics = evaluate(truth, actual, pairs).model_dump()
    if metrics != metadata["metrics"]:
        raise ValueError("The exported pipeline does not reproduce all original metrics.")
    return {
        "status": "passed",
        "rows": len(features),
        "identical_predictions": True,
        "metrics": metrics,
        "python": platform.python_version(),
        "bundled_code": True,
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
    }


def _predict(directory: Path, input_path: Path, output_path: Path) -> None:
    metadata = _check_bundle(directory)
    import mlflow.sklearn
    import pandas as pd

    from awp2.evaluation import as_target_frame

    features = pd.read_parquet(input_path)
    columns = [item["name"] for item in metadata["input_schema"]]
    if set(features.columns) != set(columns):
        raise ValueError("Input must contain exactly the feature columns in bundle.json.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {output_path}")
    model = mlflow.sklearn.load_model(str(directory / "pipeline"))
    predictions = as_target_frame(model.predict(features[columns]), features.index)
    predictions.to_parquet(output_path)
    print(f"Predictions: {output_path} ({len(predictions)} rows)")


def _write_readme(directory: Path, run_id: str) -> None:
    (directory / "README.md").write_text(
        f"""# Gesicherte AutoML-Pipeline

MLflow-Run: `{run_id}`. Vollstaendige, bereits trainierte Vorverarbeitung und SVM.
Dieses Paket benoetigt weder die urspruengliche MLflow-Datenbank noch Trainingsdaten.
`project/src/awp2/` ist der tatsaechliche Code-Snapshot inklusive uncommitteter Aenderungen.
Git-HEAD allein beschreibt diesen Stand nicht; `bundle.json` und `checksums.json` tun es.

## Wiederherstellen

Archiv in einen eigenen Ordner entpacken und darin ausfuehren (uv erforderlich):

```powershell
uv sync --project project --frozen --no-default-groups
uv run --project project --no-sync python model_bundle.py verify .
```

Die genaue Python-Version steht in `project/.python-version` und `bundle.json`.
Abhaengigkeiten sind in `project/uv.lock` gesperrt. Ohne lokalen uv-Cache ist fuer die
Installation Internetzugang erforderlich. Das Paket enthaelt keine Python-Installation.

## Vorhersagen

```powershell
uv run --project project --no-sync python model_bundle.py predict . `
    --input neue_features.parquet --output vorhersagen.parquet
```

Eingabe: pandas-DataFrame als Parquet mit genau den Features aus `bundle.json/input_schema`,
einschliesslich Bandspalten, AEZ und Month. Numerische Werte und fehlende Werte wie im
Referenzdatensatz; keine Zielspalten. Vorverarbeitung nicht vorher selbst fitten oder anwenden.
Ausgabe: Parquet mit Crop und Stage, unter Erhalt des Eingabeindex. Ein vorhandenes Ziel
wird nicht ueberschrieben. Es wird ausschliesslich `predict()`, nie `fit()`, ausgefuehrt.

## Pruefung und Grenzen

`verify` prueft Datei-Hashes, Python- und Kernpaketversionen, den Code-Snapshot sowie
alle Referenzvorhersagen und acht Metriken. `verification.json` dokumentiert den beim
Export ausgefuehrten isolierten Prozesstest. `reference/` enthaelt den Holdout und seine
Referenzvorhersagen; nicht zur erneuten Modellwahl verwenden und nicht oeffentlich committen.
`provenance/` enthaelt das Original-Suchmanifest, Konfiguration und Ergebnis des finalen Laufs.

Pickle/Cloudpickle kann beim Laden Code ausfuehren: Nur dieses vertrauenswuerdige Paket laden.
Pruefsummen erkennen Aenderungen, beweisen aber keine vertrauenswuerdige Herkunft.
Die `.sha256`-Datei neben dem Archiv prueft dessen Integritaet. Archiv samt Pruefsumme
zusaetzlich auf einem anderen Geraet oder freigegebenen Backup-Speicher sichern.
""",
        encoding="utf-8",
    )


def _export(directory: Path, destination: Path | None) -> Path:
    import mlflow
    import mlflow.sklearn
    from analyze_automl_final import _check_manifest

    from awp2.config import MLFLOW_TRACKING_URI, MODELS_DIR, PROJECT_ROOT
    from awp2.data import load_split, valid_combinations
    from awp2.evaluation import as_target_frame, evaluate

    record = _read_json(directory / "validation.json")
    if record["status"] != "completed":
        raise ValueError("Only a completed frozen evaluation can be exported.")
    _check_manifest(directory)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    run = client.get_run(record["run_id"])
    if run.outputs is None or len(run.outputs.model_outputs) != 1:
        raise ValueError("The final run must contain exactly one saved fitted pipeline.")
    model_id = run.outputs.model_outputs[0].model_id
    logged = client.get_logged_model(model_id)
    if logged.source_run_id != run.info.run_id:
        raise ValueError("The saved model does not belong to the final run.")
    destination = (destination or MODELS_DIR / run.data.tags["mlflow.runName"]).resolve()
    archive = destination.with_suffix(".zip")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    if any(path.exists() for path in (destination, archive, checksum)):
        raise FileExistsError("Refusing to overwrite an existing bundle, archive or checksum.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    split = load_split()
    fitted = mlflow.sklearn.load_model(f"models:/{model_id}")
    expected = as_target_frame(fitted.predict(split.X_val), split.y_val.index)
    pairs = valid_combinations(split.y_train)
    metrics = evaluate(split.y_val, expected, pairs).model_dump()
    if metrics != record["metrics"] or any(
        run.data.metrics.get(name) != value for name, value in metrics.items()
    ):
        raise ValueError("The original saved pipeline does not reproduce the final metrics.")
    with tempfile.TemporaryDirectory(prefix="awp2-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        with tempfile.TemporaryDirectory(prefix="awp2-artifacts-") as downloaded:
            artifacts = mlflow.artifacts.download_artifacts(
                artifact_uri=logged.artifact_location, dst_path=downloaded
            )
            shutil.copytree(artifacts, staging / "pipeline")
        project = staging / "project"
        shutil.copytree(
            PROJECT_ROOT / "src" / "awp2",
            project / "src" / "awp2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in ("pyproject.toml", "uv.lock", "README.md"):
            shutil.copy2(PROJECT_ROOT / name, project / name)
        (project / ".python-version").write_text(platform.python_version() + "\n", encoding="ascii")
        shutil.copy2(Path(__file__), staging / "model_bundle.py")
        reference = staging / "reference"
        reference.mkdir()
        split.X_val.to_parquet(reference / "features.parquet")
        split.y_val.to_parquet(reference / "targets.parquet")
        expected.to_parquet(reference / "predictions.parquet")
        provenance = staging / "provenance"
        provenance.mkdir()
        shutil.copy2(directory / "validation.json", provenance / "validation.json")
        database = (directory / "search.sqlite").resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(database, uri=True) as connection:
            row = connection.execute("SELECT content FROM manifest").fetchone()
        if row is None:
            raise ValueError("The search manifest is missing.")
        _write_json(provenance / "search_manifest.json", json.loads(row[0]))
        for name in ("report.md", "candidates.csv"):
            if (directory / name).is_file():
                shutil.copy2(directory / name, provenance / name)
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        _write_json(
            staging / "bundle.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "run_id": run.info.run_id,
                "model_id": model_id,
                "git_head": git_head,
                "git_status": git_status,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "runtime_packages": {
                    name: importlib.metadata.version(name) for name in _RUNTIME_PACKAGES
                },
                "installed_packages": {
                    item.metadata["Name"]: item.version
                    for item in importlib.metadata.distributions()
                },
                "metrics": metrics,
                "valid_combinations": [
                    pair.model_dump()
                    for pair in sorted(pairs, key=lambda pair: (pair.crop, pair.stage))
                ],
                "input_schema": [
                    {"name": str(name), "dtype": str(dtype)}
                    for name, dtype in split.X_val.dtypes.items()
                ],
                "target_columns": list(split.y_val.columns),
            },
        )
        _write_readme(staging, run.info.run_id)
        _write_json(staging / "checksums.json", _checksums(staging))
        with tempfile.TemporaryDirectory(prefix="awp2-load-test-") as isolated:
            environment = dict(os.environ)
            environment["MLFLOW_TRACKING_URI"] = (
                f"sqlite:///{Path(isolated).as_posix()}/unused.sqlite"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(staging / "model_bundle.py"),
                    "verify",
                    str(staging),
                ],
                cwd=isolated,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"Isolated load test failed:\n{result.stderr}\n{result.stdout}")
            verification = json.loads(result.stdout)
            verification["isolated_process"] = True
            verification["fresh_environment"] = False
        _write_json(staging / "verification.json", verification)
        _write_json(staging / "checksums.json", _checksums(staging))
        staging.rename(destination)
    shutil.make_archive(str(archive.with_suffix("")), "zip", destination.parent, destination.name)
    checksum.write_text(f"{_digest(archive)}  {archive.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "bundle": str(destination),
                "archive": str(archive),
                "sha256": str(checksum),
                "verification": verification,
            },
            indent=2,
        )
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--directory", type=Path)
    export.add_argument("--output", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("bundle", type=Path)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        from awp2.config import AUTOML_FOCUSED_DIR

        _export(args.directory or AUTOML_FOCUSED_DIR, args.output)
    elif args.command == "verify":
        print(json.dumps(_verify(args.bundle.resolve()), indent=2))
    else:
        _predict(args.bundle.resolve(), args.input, args.output)


if __name__ == "__main__":
    main()
