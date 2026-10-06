import unittest

import pandas as pd
from edahelper import fill_spectral_nans


class TestFillSpectralNans(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            [
                {
                    "Other": "a",
                    "X780": None,
                    "X800": None,
                    "X813": 10,
                    "X824": None,
                    "X834": 20,
                    "X850": None,
                    "X870": None,
                },
                {
                    "Other": "a",
                    "X780": None,
                    "X800": None,
                    "X813": 7,
                    "X824": None,
                    "X834": None,
                    "X850": None,
                    "X870": None,
                },
                {
                    "Other": "a",
                    "X780": None,
                    "X800": None,
                    "X813": None,
                    "X824": None,
                    "X834": 9,
                    "X850": None,
                    "X870": None,
                },
                {
                    "Other": "a",
                    "X780": None,
                    "X800": 12,
                    "X813": None,
                    "X824": None,
                    "X834": None,
                    "X850": 20,
                    "X870": None,
                },
                {
                    "Other": "a",
                    "X780": 2,
                    "X800": None,
                    "X813": None,
                    "X824": None,
                    "X834": None,
                    "X850": None,
                    "X870": 40,
                },
                {
                    "Other": "a",
                    "X780": None,
                    "X800": None,
                    "X813": None,
                    "X824": 40,
                    "X834": None,
                    "X850": None,
                    "X870": None,
                },
                {
                    "Other": "a",
                    "X780": None,
                    "X800": None,
                    "X813": None,
                    "X824": 60,
                    "X834": None,
                    "X850": None,
                    "X870": None,
                },
            ],
            index=[
                "both_sides",
                "left_only",
                "right_only",
                "expanded_range",
                "column_mean",
                "mean_a",
                "mean_b",
            ],
        )
        self.df.index.name = "id"

    def test_averages_values_from_both_neighbors(self) -> None:
        result = fill_spectral_nans(self.df)
        self.assertEqual(result.loc["both_sides", "X824"], 15)

    def test_uses_left_neighbor_when_right_is_missing(self) -> None:
        result = fill_spectral_nans(self.df)
        self.assertEqual(result.loc["left_only", "X824"], 7)

    def test_uses_right_neighbor_when_left_is_missing(self) -> None:
        result = fill_spectral_nans(self.df)
        self.assertEqual(result.loc["right_only", "X824"], 9)

    def test_widens_search_range(self) -> None:
        result = fill_spectral_nans(self.df)
        self.assertEqual(result.loc["expanded_range", "X824"], 16)

    def test_uses_column_mean_when_no_neighbor_is_in_range(self) -> None:
        result = fill_spectral_nans(self.df)
        self.assertEqual(result.loc["column_mean", "X824"], 50)

    def test_does_not_modify_input_dataframe(self) -> None:
        original = self.df.copy(deep=True)
        fill_spectral_nans(self.df)
        pd.testing.assert_frame_equal(self.df, original)


unittest.main(argv=[""], exit=False)
