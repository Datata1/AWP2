"""Pipeline artifacts in ``data/``: the cleaned dataset and the shared split.

Build them once with ``make data``; everyone then works on identical files. The preprocessing
itself is not stored – it is fitted inside each model pipeline on the training part only.
"""

import pandas as pd

from awp2.config import (
    CLEAN_TRAIN_FILE,
    FOLD_COL,
    ID_COL,
    SPLIT_FILE,
    SUBSET_COL,
    TRAIN_SUBSET,
    VAL_SUBSET,
)
from awp2.data.load import load_train
from awp2.data.split import (
    Dataset,
    Fold,
    TrainValSplit,
    cv_splits,
    prepare_dataset,
    train_val_split,
)


class StaleArtifactsError(RuntimeError):
    """The artifacts are missing or do not match each other – run ``make data``."""


def build_artifacts() -> None:
    """Write the cleaned dataset (``data/interim``) and the split assignment (``data/processed``).

    Deterministic: rebuilding from the same raw data gives identical files.
    """
    dataset = prepare_dataset(load_train())
    split = train_val_split(dataset.X, dataset.y)

    assignment = pd.Series(VAL_SUBSET, index=dataset.X.index, name=SUBSET_COL)
    assignment[split.X_train.index] = TRAIN_SUBSET
    folds = pd.Series(pd.NA, index=dataset.X.index, name=FOLD_COL, dtype="Int64")
    for number, fold in enumerate(cv_splits(split.X_train, split.y_train)):
        folds[split.X_train.index[fold.val]] = number

    CLEAN_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([dataset.X, dataset.y], axis=1).to_parquet(CLEAN_TRAIN_FILE)
    pd.concat([assignment, folds], axis=1).rename_axis(ID_COL).to_csv(SPLIT_FILE)


def load_dataset() -> Dataset:
    """The cleaned, deduplicated labelled data from ``data/interim``."""
    if not CLEAN_TRAIN_FILE.exists():
        raise StaleArtifactsError(f"{CLEAN_TRAIN_FILE} missing – run `make data`.")
    df = pd.read_parquet(CLEAN_TRAIN_FILE)
    return prepare_dataset(df)


def _load_assignment(dataset: Dataset) -> pd.DataFrame:
    if not SPLIT_FILE.exists():
        raise StaleArtifactsError(f"{SPLIT_FILE} missing – run `make data`.")
    assignment = pd.read_csv(SPLIT_FILE, index_col=ID_COL, dtype={FOLD_COL: "Int64"})
    if not assignment.index.sort_values().equals(dataset.X.index.sort_values()):
        raise StaleArtifactsError("Split and dataset do not match – run `make data`.")
    return assignment.loc[dataset.X.index]


def load_split() -> TrainValSplit:
    """The shared 70/30 holdout split, read from the artifacts."""
    dataset = load_dataset()
    is_train = _load_assignment(dataset)[SUBSET_COL].eq(TRAIN_SUBSET).to_numpy()
    return TrainValSplit(
        X_train=dataset.X[is_train],
        X_val=dataset.X[~is_train],
        y_train=dataset.y[is_train],
        y_val=dataset.y[~is_train],
    )


def load_folds() -> list[Fold]:
    """Cross-validation folds of the training part, as row positions within ``X_train``."""
    dataset = load_dataset()
    assignment = _load_assignment(dataset)
    fold_ids = assignment.loc[assignment[SUBSET_COL].eq(TRAIN_SUBSET), FOLD_COL].to_numpy()
    positions = pd.RangeIndex(len(fold_ids)).to_numpy()
    return [
        Fold(train=positions[fold_ids != number], val=positions[fold_ids == number])
        for number in sorted(set(fold_ids))
    ]


if __name__ == "__main__":
    build_artifacts()
    print(f"Wrote {CLEAN_TRAIN_FILE} and {SPLIT_FILE}")
