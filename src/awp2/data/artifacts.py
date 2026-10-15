"""Pipeline artifacts in ``data/``: the cleaned dataset and the shared split.

Build them once with ``make data``; everyone then works on identical files. The preprocessing
itself is not stored – it is fitted inside each model pipeline on the training part only.
"""

import argparse
from hashlib import sha256

import pandas as pd

from awp2.config import (
    CLEAN_TRAIN_FILE,
    FOLD_COL,
    ID_COL,
    SPLIT_CONFIG_FILE,
    SPLIT_CONFIG_HASH_COL,
    SPLIT_FILE,
    SUBSET_COL,
    TRAIN_SUBSET,
    VAL_SUBSET,
)
from awp2.data.load import load_train
from awp2.data.split import (
    Dataset,
    Fold,
    SplitConfig,
    TrainValSplit,
    cv_splits,
    prepare_dataset,
    train_val_split,
)


class StaleArtifactsError(RuntimeError):
    """The artifacts are missing or do not match each other – run ``make data``."""


def build_artifacts(config: SplitConfig | None = None) -> None:
    """Write the cleaned dataset and the split assignment to ``data/`` (run via ``make data``).

    Creates ``data/interim/train_clean.parquet`` (validated, deduplicated training data),
    ``data/processed/split.csv`` (per ``id``: ``subset`` train/val and ``cv_fold``) and
    ``data/processed/split_config.json`` (the split settings).
    Deterministic: rebuilding from the same raw data gives identical files.
    """
    if config is None:
        config = SplitConfig()
    dataset = prepare_dataset(load_train())
    split = train_val_split(dataset.X, dataset.y, config)

    assignment = pd.Series(VAL_SUBSET, index=dataset.X.index, name=SUBSET_COL)
    assignment[split.X_train.index] = TRAIN_SUBSET
    folds = pd.Series(pd.NA, index=dataset.X.index, name=FOLD_COL, dtype="Int64")
    for number, fold in enumerate(cv_splits(split.X_train, split.y_train)):
        folds[split.X_train.index[fold.val]] = number

    CLEAN_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([dataset.X, dataset.y], axis=1).to_parquet(CLEAN_TRAIN_FILE)
    config_json = config.model_dump_json()
    config_hash = sha256(config_json.encode()).hexdigest()
    config_hashes = pd.Series(config_hash, index=dataset.X.index, name=SPLIT_CONFIG_HASH_COL)
    pd.concat([assignment, folds, config_hashes], axis=1).rename_axis(ID_COL).to_csv(SPLIT_FILE)
    SPLIT_CONFIG_FILE.write_text(config_json + "\n", encoding="utf-8")


def load_dataset() -> Dataset:
    """Load the cleaned, deduplicated labelled data, e.g. for the EDA.

    Returns:
        Features and targets of all labelled rows.

    Raises:
        StaleArtifactsError: If the artifacts are missing or do not match – run ``make data``.
    """
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
    if SPLIT_CONFIG_HASH_COL not in assignment:
        raise StaleArtifactsError("Split lacks its configuration fingerprint – run `make data`.")
    config_hash = sha256(load_split_config().model_dump_json().encode()).hexdigest()
    if not assignment[SPLIT_CONFIG_HASH_COL].eq(config_hash).all():
        raise StaleArtifactsError("Split and configuration do not match – run `make data`.")
    return assignment.loc[dataset.X.index]


def load_split_config() -> SplitConfig:
    """Load the configuration used to build the shared split.

    Returns:
        Validated settings persisted with the split artifact.

    Raises:
        StaleArtifactsError: If the split configuration is missing – run ``make data``.
    """
    if not SPLIT_CONFIG_FILE.exists():
        raise StaleArtifactsError(f"{SPLIT_CONFIG_FILE} missing – run `make data`.")
    return SplitConfig.model_validate_json(SPLIT_CONFIG_FILE.read_text(encoding="utf-8"))


def load_split() -> TrainValSplit:
    """Load the shared train/validation split – the same for everyone.

    Returns:
        Training and validation part, read from ``data/processed/split.csv``.

    Raises:
        StaleArtifactsError: If the artifacts are missing or do not match – run ``make data``.
    """
    dataset = load_dataset()
    is_train = _load_assignment(dataset)[SUBSET_COL].eq(TRAIN_SUBSET).to_numpy()
    return TrainValSplit(
        X_train=dataset.X[is_train],
        X_val=dataset.X[~is_train],
        y_train=dataset.y[is_train],
        y_val=dataset.y[~is_train],
    )


def load_folds() -> list[Fold]:
    """Load the cross-validation folds of the training part (for tuning).

    Returns:
        ``CV_FOLDS`` folds as row positions within ``load_split().X_train``; pass them as
            ``cv=`` to ``GridSearchCV`` or ``cross_validate``.

    Raises:
        StaleArtifactsError: If the artifacts are missing or do not match – run ``make data``.
    """
    dataset = load_dataset()
    assignment = _load_assignment(dataset)
    fold_ids = assignment.loc[assignment[SUBSET_COL].eq(TRAIN_SUBSET), FOLD_COL].to_numpy()
    positions = pd.RangeIndex(len(fold_ids)).to_numpy()
    return [
        Fold(train=positions[fold_ids != number], val=positions[fold_ids == number])
        for number in sorted(set(fold_ids))
    ]


def _parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(description="Build the shared data artifacts.")
    parser.add_argument(
        "--validation-size",
        type=float,
        default=SplitConfig.model_fields["validation_size"].default,
        help="Fraction of rows assigned to validation (default: %(default)s).",
    )
    args = parser.parse_args()
    return SplitConfig(validation_size=args.validation_size)


if __name__ == "__main__":
    config = _parse_args()
    build_artifacts(config)
    print(f"Wrote {CLEAN_TRAIN_FILE}, {SPLIT_FILE} and {SPLIT_CONFIG_FILE}")
