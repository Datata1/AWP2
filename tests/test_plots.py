"""Tests for the data plots."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from awp2.config import CROP_COL, CROPS, MONTH_COL, STAGE_COL  # noqa: E402
from awp2.plots import plot_stage_by_month  # noqa: E402


class StageByMonthTest(unittest.TestCase):
    def setUp(self) -> None:
        # corn has no row in month 6 and rice only appears in month 8, as in the real data
        self.data = pd.DataFrame(
            {
                CROP_COL: ["corn", "corn", "corn", "rice", "cotton"],
                STAGE_COL: ["Emerge_VEarly", "Critical", "Critical", "Late", "Early_Mid"],
                MONTH_COL: [5, 8, 8, 8, 6],
            }
        )

    def tearDown(self) -> None:
        plt.close("all")

    def test_bars_sit_at_their_month_in_every_panel(self) -> None:
        fig = plot_stage_by_month(self.data)
        corn, rice = (ax for ax in fig.axes if ax.get_title() in {"corn", "rice"})

        def filled_months(ax: plt.Axes) -> set[float]:
            return {round(p.get_x() + p.get_width() / 2) for p in ax.patches if p.get_height() > 0}

        self.assertEqual(filled_months(corn), {5, 8})
        self.assertEqual(filled_months(rice), {8})

    def test_one_panel_per_crop(self) -> None:
        fig = plot_stage_by_month(self.data)

        visible = [ax.get_title() for ax in fig.axes if ax.get_visible()]

        self.assertEqual(visible, list(CROPS))
