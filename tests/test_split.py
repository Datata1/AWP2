import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from awp2.config import AEZ_COL, CROP_COL, MONTH_COL, STAGE_COL
from awp2.data import artifacts
from awp2.data.split import SplitConfig, train_val_split


class SplitConfigTest(unittest.TestCase):
    def test_train_val_split_uses_configured_validation_size(self) -> None:
        rows_per_label = 10
        labels = [
            ("corn", "Early_Mid"),
            ("rice", "Late"),
        ]
        X = pd.DataFrame({"feature": range(rows_per_label * len(labels))})
        y = pd.DataFrame(
            [
                {CROP_COL: crop, STAGE_COL: stage}
                for crop, stage in labels
                for _ in range(rows_per_label)
            ]
        )

        split = train_val_split(X, y, SplitConfig(validation_size=0.2))

        self.assertEqual(len(split.X_train), 16)
        self.assertEqual(len(split.X_val), 4)
        self.assertTrue(split.y_val.groupby([CROP_COL, STAGE_COL]).size().eq(2).all())

    def test_split_config_rejects_invalid_validation_sizes(self) -> None:
        for validation_size in (0.0, 1.0):
            with self.assertRaises(ValidationError):
                SplitConfig(validation_size=validation_size)

    def test_build_artifacts_persists_configured_validation_size(self) -> None:
        rows_per_label = 10
        labels = [("corn", "Early_Mid"), ("rice", "Late")]
        dataset = pd.DataFrame(
            [
                {AEZ_COL: 1, MONTH_COL: 1, "X427": row, CROP_COL: crop, STAGE_COL: stage}
                for crop, stage in labels
                for row in range(rows_per_label)
            ]
        )
        dataset.index.name = "id"

        with TemporaryDirectory() as temporary_directory:
            processed_dir = Path(temporary_directory) / "processed"
            interim_dir = Path(temporary_directory) / "interim"
            with (
                patch.object(artifacts, "CLEAN_TRAIN_FILE", interim_dir / "train_clean.parquet"),
                patch.object(artifacts, "SPLIT_FILE", processed_dir / "split.csv"),
                patch.object(artifacts, "SPLIT_CONFIG_FILE", processed_dir / "split_config.json"),
                patch.object(artifacts, "load_train", return_value=dataset),
            ):
                artifacts.build_artifacts(SplitConfig(validation_size=0.2))
                config = artifacts.load_split_config()
                split = artifacts.load_split()

        self.assertEqual(config.validation_size, 0.2)
        self.assertEqual(len(split.X_train), 16)
        self.assertEqual(len(split.X_val), 4)