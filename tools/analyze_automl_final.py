"""Describe a frozen AutoML model without fitting or opening a new MLflow run."""

import argparse
import hashlib
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from pydantic import BaseModel, ConfigDict, Field

from awp2.config import (
    AEZ_COL,
    AUTOML_FOCUSED_DIR,
    CROP_COL,
    MLFLOW_TRACKING_URI,
    MODEL_DOCS_FIGURES_DIR,
    MONTH_COL,
    PROJECT_ROOT,
    RECALL_COL,
    STAGE_COL,
    SUPPORT_COL,
    TARGET_COLS,
)
from awp2.data import combined_label, load_split, valid_combinations
from awp2.evaluation import (
    as_target_frame,
    error_breakdown,
    evaluate,
    plot_confusion_matrices,
    recall_by_class,
)
from awp2.plots import (
    plot_class_recall,
    plot_recall_comparison,
    plot_top_confusions,
    save_doc_figure,
)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    directory: Path = Field(default=AUTOML_FOCUSED_DIR, description="Frozen search artifacts.")
    figure_directory: Path = Field(
        default=MODEL_DOCS_FIGURES_DIR, description="Versioned figures for the German model docs."
    )


def _check_manifest(directory: Path) -> None:
    database = directory / "search.sqlite"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT content FROM manifest").fetchone()
    if row is None:
        raise ValueError("The search manifest is missing.")
    manifest = json.loads(row[0])
    expected = {**manifest["source"], **manifest["data"], "uv.lock": manifest["environment"]}
    for relative, digest in expected.items():
        with (PROJECT_ROOT / relative).open("rb") as source:
            current = hashlib.file_digest(source, "sha256").hexdigest()
        if current != digest:
            raise ValueError(f"Artifact or source changed since evaluation: {relative}")


def _target_recall(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    hits = y_true.eq(y_pred)
    return (
        pd.DataFrame({RECALL_COL: hits, "class": y_true})
        .groupby("class")
        .agg(**{RECALL_COL: (RECALL_COL, "mean"), SUPPORT_COL: (RECALL_COL, "size")})
        .sort_values(RECALL_COL)
    )


def _confusions(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    pairs = pd.DataFrame({"true": y_true, "predicted": y_pred})
    return (
        pairs.loc[pairs["true"].ne(pairs["predicted"])].value_counts().rename("count").reset_index()
    )


def _save_figure(figure: plt.Figure, name: str, directory: Path) -> None:
    try:
        save_doc_figure(figure, name, directory)
    finally:
        plt.close(figure)


def _analyze(config: AnalysisConfig) -> dict[str, Any]:
    record = json.loads((config.directory / "validation.json").read_text(encoding="utf-8"))
    if record["status"] != "completed":
        raise ValueError("The final evaluation is not completed.")
    _check_manifest(config.directory)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    run = client.get_run(record["run_id"])
    if run.inputs is None:
        raise ValueError("The saved run has no dataset provenance; analysis is forbidden.")
    split = load_split()
    frames = {
        "training": pd.concat([split.X_train, split.y_train], axis=1),
        "validation": pd.concat([split.X_val, split.y_val], axis=1),
    }
    recorded_digests = {
        item.dataset.name: item.dataset.digest for item in run.inputs.dataset_inputs
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*can be interpreted in multiple ways.*")
        warnings.filterwarnings("ignore", message=".*Hint: Inferred schema.*")
        digests = {name: from_pandas(frame, name=name).digest for name, frame in frames.items()}
    if any(recorded_digests.get(name) != digest for name, digest in digests.items()):
        raise ValueError(
            "The current split does not match the logged training/validation datasets."
        )
    if run.outputs is None or len(run.outputs.model_outputs) != 1:
        raise ValueError("Exactly one saved fitted model is required; retraining is forbidden.")
    model_id = run.outputs.model_outputs[0].model_id
    model = mlflow.sklearn.load_model(f"models:/{model_id}")
    predictions = as_target_frame(model.predict(split.X_val), split.y_val.index)
    metrics = evaluate(split.y_val, predictions, valid_combinations(split.y_train)).model_dump()
    if metrics != record["metrics"] or any(
        run.data.metrics.get(name) != value for name, value in metrics.items()
    ):
        raise ValueError("Frozen-model predictions do not reproduce the recorded metrics.")

    output = config.directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    label = "AutoML-SVM (80/20)"
    prefix = run.data.tags["mlflow.runName"]
    breakdown = error_breakdown(split.y_val, predictions)
    crop_correct = split.y_val[CROP_COL].eq(predictions[CROP_COL])
    stage_correct = split.y_val[STAGE_COL].eq(predictions[STAGE_COL])
    pairs = recall_by_class(split.y_val, predictions)
    pairs.to_csv(output / "recall_combined.csv")
    for target in TARGET_COLS:
        _target_recall(split.y_val[target], predictions[target]).to_csv(
            output / f"recall_{target}.csv"
        )
        _confusions(split.y_val[target], predictions[target]).to_csv(
            output / f"confusions_{target}.csv", index=False
        )
    _confusions(combined_label(split.y_val), combined_label(predictions)).to_csv(
        output / "confusions_combined.csv", index=False
    )
    rows = pd.concat(
        [
            split.X_val[[AEZ_COL, MONTH_COL]],
            split.y_val.add_prefix("true_"),
            predictions.add_prefix("predicted_"),
        ],
        axis=1,
    )
    rows.to_csv(output / "predictions.csv")
    for column in (MONTH_COL, AEZ_COL):
        grouped = (
            pd.DataFrame(
                {
                    column: split.X_val[column],
                    "crop_wrong": ~crop_correct,
                    "stage_wrong": ~stage_correct,
                    "pair_wrong": ~(crop_correct & stage_correct),
                }
            )
            .groupby(column)
            .agg(
                support=("pair_wrong", "size"),
                crop_error_rate=("crop_wrong", "mean"),
                stage_error_rate=("stage_wrong", "mean"),
                pair_error_rate=("pair_wrong", "mean"),
            )
        )
        grouped.to_csv(output / f"errors_by_{column}.csv")
    _save_figure(
        plot_confusion_matrices(split.y_val, predictions),
        f"{prefix}_confusion",
        config.figure_directory,
    )
    _save_figure(
        plot_class_recall(split.y_val, predictions), f"{prefix}_recall", config.figure_directory
    )
    _save_figure(
        plot_recall_comparison({label: pairs}), f"{prefix}_combined_recall", config.figure_directory
    )
    _save_figure(
        plot_top_confusions(split.y_val, predictions),
        f"{prefix}_confusions",
        config.figure_directory,
    )
    summary = {
        "run_id": run.info.run_id,
        "model_id": model_id,
        "rows": len(predictions),
        "dataset_digests": digests,
        "metrics_reproduced": True,
        "metrics": metrics,
        "error_breakdown": breakdown.model_dump(),
        "counts": {
            "both_correct": int((crop_correct & stage_correct).sum()),
            "crop_wrong": int((~crop_correct).sum()),
            "stage_only_wrong": int((crop_correct & ~stage_correct).sum()),
            "both_wrong": int((~crop_correct & ~stage_correct).sum()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Analysis artifacts: {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=AUTOML_FOCUSED_DIR)
    parser.add_argument("--figure-directory", type=Path, default=MODEL_DOCS_FIGURES_DIR)
    args = parser.parse_args()
    _analyze(AnalysisConfig(directory=args.directory, figure_directory=args.figure_directory))


if __name__ == "__main__":
    main()
