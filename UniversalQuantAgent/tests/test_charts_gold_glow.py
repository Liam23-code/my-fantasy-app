"""Contract tests for the shared gold-glow chart data adapter."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.style import CIRCUIT_CYAN, MIDNIGHT_NAVY, SIGNAL_GOLD, gold_glow_chart


class GoldGlowChartContractTests(unittest.TestCase):
    def test_sequence_uses_smooth_thin_gold_line_and_cyan_peak_dots(self):
        figure = gold_glow_chart([10, 14, 12, 14], title="Weekly projection")

        self.assertEqual(len(figure.data), 3)
        self.assertEqual(figure.data[1].line.color, SIGNAL_GOLD)
        self.assertEqual(figure.data[1].line.width, 2)
        self.assertEqual(figure.data[1].line.shape, "spline")
        self.assertEqual(figure.data[2].marker.color, CIRCUIT_CYAN)
        self.assertEqual(list(figure.data[2].x), [2, 4])
        self.assertEqual(list(figure.data[2].y), [14.0, 14.0])

    def test_glow_opacity_is_between_ten_and_fifteen_percent(self):
        figure = gold_glow_chart([8, 11, 9])
        color = figure.data[0].line.color
        match = re.fullmatch(r"rgba\(245,197,66,([0-9.]+)\)", color)

        self.assertIsNotNone(match)
        self.assertGreaterEqual(float(match.group(1)), 0.10)
        self.assertLessEqual(float(match.group(1)), 0.15)
        self.assertGreater(figure.data[0].line.width, figure.data[1].line.width)

    def test_chart_uses_navy_canvas_and_minimal_grid(self):
        figure = gold_glow_chart([4, 7, 5])

        self.assertEqual(figure.layout.paper_bgcolor, MIDNIGHT_NAVY)
        self.assertEqual(figure.layout.plot_bgcolor, MIDNIGHT_NAVY)
        self.assertEqual(figure.layout.xaxis.gridcolor, "rgba(242,244,247,.06)")
        self.assertEqual(figure.layout.yaxis.gridcolor, "rgba(242,244,247,.06)")

    def test_series_preserves_index_and_name(self):
        series = pd.Series([12.5, 13.2], index=["W1", "W2"], name="Ceiling")
        figure = gold_glow_chart(series)

        self.assertEqual(list(figure.data[1].x), ["W1", "W2"])
        self.assertEqual(list(figure.data[1].y), [12.5, 13.2])
        self.assertEqual(figure.data[1].name, "Ceiling")

    def test_dataframe_infers_week_and_projected_points_columns(self):
        frame = pd.DataFrame(
            {
                "Week": [1, 2, 3],
                "Opponent": ["DEN", "KC", "LV"],
                "Projected points": [18.2, 21.6, 19.4],
            }
        )
        figure = gold_glow_chart(frame)

        self.assertEqual(list(figure.data[1].x), [1, 2, 3])
        self.assertEqual(list(figure.data[1].y), [18.2, 21.6, 19.4])
        self.assertEqual(figure.data[1].name, "Projected points")

    def test_single_column_dataframe_preserves_meaningful_index(self):
        frame = pd.DataFrame(
            {"Projection": [16.2, 19.1]},
            index=pd.Index(["Week 4", "Week 5"], name="Schedule"),
        )
        figure = gold_glow_chart(frame)

        self.assertEqual(list(figure.data[1].x), ["Week 4", "Week 5"])
        self.assertEqual(list(figure.data[1].y), [16.2, 19.1])

    def test_dataframe_allows_explicit_confidence_column(self):
        frame = pd.DataFrame(
            {
                "Week": [1, 2],
                "Projected points": [20.1, 18.4],
                "Confidence": [0.81, 0.76],
            }
        )
        figure = gold_glow_chart(frame, x="week", y="confidence", y_suffix="%")

        self.assertEqual(list(figure.data[1].x), [1, 2])
        self.assertEqual(list(figure.data[1].y), [0.81, 0.76])
        self.assertEqual(figure.data[1].name, "Confidence")

    def test_label_value_mapping_preserves_insertion_order(self):
        figure = gold_glow_chart({"W1": 15.0, "W2": 16.4, "W3": 14.9})

        self.assertEqual(list(figure.data[1].x), ["W1", "W2", "W3"])
        self.assertEqual(list(figure.data[1].y), [15.0, 16.4, 14.9])

    def test_plural_column_mapping_names_are_inferred(self):
        figure = gold_glow_chart({"weeks": [1, 2], "values": [8.5, 10.1]})

        self.assertEqual(list(figure.data[1].x), [1, 2])
        self.assertEqual(list(figure.data[1].y), [8.5, 10.1])

    def test_weekly_projection_mapping_selects_points(self):
        weekly = {
            1: {"points": 17.1, "confidence": 0.72},
            2: {"points": 20.3, "confidence": 0.78},
        }
        figure = gold_glow_chart(weekly)

        self.assertEqual(list(figure.data[1].x), [1, 2])
        self.assertEqual(list(figure.data[1].y), [17.1, 20.3])
        self.assertEqual(figure.data[1].name, "points")

    def test_record_and_pair_sequences_are_supported(self):
        records = [{"week": 1, "score": 9.4}, {"week": 2, "score": 12.8}]
        record_figure = gold_glow_chart(records)
        pair_figure = gold_glow_chart([("W1", 9.4), ("W2", 12.8)])

        self.assertEqual(list(record_figure.data[1].x), [1, 2])
        self.assertEqual(list(record_figure.data[1].y), [9.4, 12.8])
        self.assertEqual(list(pair_figure.data[1].x), ["W1", "W2"])
        self.assertEqual(list(pair_figure.data[1].y), [9.4, 12.8])

    def test_bad_axis_name_and_non_numeric_data_fail_clearly(self):
        frame = pd.DataFrame({"Week": [1], "Points": [10]})
        with self.assertRaisesRegex(ValueError, "x field"):
            gold_glow_chart(frame, x="missing")
        with self.assertRaisesRegex(ValueError, "numeric"):
            gold_glow_chart(["high", "low"])


if __name__ == "__main__":
    unittest.main()
