"""Fit a finished model on all labelled data and predict the unlabelled test set."""

import argparse
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from awp2.config import ID_COL, PREDICTIONS_DIR, TEST_FILE
from awp2.data import load_test, load_train, prepare_dataset
from awp2.evaluation import as_target_frame
from awp2.preprocessing import PreprocessingConfig, build_preprocessor

_PREPROCESS_STEP = "preprocess"
_MODEL_STEP = "model"


class Submission(NamedTuple):
    """Result of :func:`fit_and_predict_test`.

    Attributes:
        pipeline: Preprocessing + model, fitted on all labelled rows.
        test: The test rows (``AEZ``, ``Month``, bands) with the predicted columns ``Crop`` and
            ``Stage`` appended, indexed by ``id``.
        train_rows: Number of labelled rows the pipeline was fitted on.
    """

    pipeline: Pipeline
    test: pd.DataFrame
    train_rows: int


def fit_and_predict_test(
    model: BaseEstimator, preprocessing: PreprocessingConfig, test_file: Path = TEST_FILE
) -> Submission:
    """Fit preprocessing + ``model`` on all labelled rows and predict the test file.

    No validation part is held back: hyperparameters must already be fixed. Exact duplicates are
    dropped like in the shared dataset.

    Args:
        model: Unfitted estimator that predicts ``Crop`` **and** ``Stage`` with fixed
            hyperparameters; it is fitted in place.
        preprocessing: Preprocessing options of the model.
        test_file: CSV file with the unlabelled rows to predict.

    Returns:
        The fitted pipeline, the test rows with the predicted columns and the number of training
            rows.
    """
    dataset = prepare_dataset(load_train())
    pipeline = Pipeline(
        [(_PREPROCESS_STEP, build_preprocessor(preprocessing)), (_MODEL_STEP, model)]
    )
    pipeline.fit(dataset.X, dataset.y)
    X_test = load_test(test_file)
    predictions = as_target_frame(pipeline.predict(X_test), X_test.index)
    return Submission(
        pipeline=pipeline,
        test=pd.concat([X_test, predictions], axis=1),
        train_rows=len(dataset.X),
    )


def create_prediction(test: pd.DataFrame, path: Path) -> Path:
    """Write the test rows with their predicted columns as CSV, ``id`` as first column.

    Args:
        test: Test rows with the appended columns ``Crop`` and ``Stage``, indexed by ``id``.
        path: Target file; missing parent directories are created.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    test.rename_axis(ID_COL).to_csv(path)
    return path


def main(name: str, model: BaseEstimator, preprocessing: PreprocessingConfig) -> None:
    """Command line entry point of a final-model script.

    Args:
        name: Name of the model; the default output is ``reports/predictions/<name>.csv``.
        model: Unfitted estimator with fixed hyperparameters.
        preprocessing: Preprocessing options of the model.
    """
    parser = argparse.ArgumentParser(description=f"Train {name} on all labelled data.")
    parser.add_argument("--test-file", type=Path, default=TEST_FILE, help="%(default)s")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / f"{name}.csv")
    args = parser.parse_args()

    result = fit_and_predict_test(model, preprocessing, args.test_file)
    path = create_prediction(result.test, args.output)
    print(f"{name}: fitted on {result.train_rows} rows, wrote {len(result.test)} to {path}")
