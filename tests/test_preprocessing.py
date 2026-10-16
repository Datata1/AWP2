"""Tests for shared preprocessing behavior."""

import unittest

import pandas as pd

from awp2.config import AEZ_COL, MONTH_COL
from awp2.preprocessing import PreprocessingConfig, build_preprocessor


class CategoricalContextPreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                AEZ_COL: [1, 2],
                MONTH_COL: [3, 9],
                "X427": [0.1, 0.2],
                "X437": [0.3, 0.4],
            }
        )

    def test_categorical_context_preserves_raw_metadata_columns(self) -> None:
        transformed = build_preprocessor(
            PreprocessingConfig(use_categorical_context=True)
        ).fit_transform(self.features)

        self.assertEqual(list(transformed.columns), [AEZ_COL, MONTH_COL, "X427", "X437"])
        pd.testing.assert_frame_equal(
            transformed[[AEZ_COL, MONTH_COL]], self.features[[AEZ_COL, MONTH_COL]]
        )