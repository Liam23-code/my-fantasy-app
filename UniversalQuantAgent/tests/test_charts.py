"""Visual identity contracts for gold-line and cyan-peak Plotly charts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.style import (
    CIRCUIT_CYAN,
    MIDNIGHT_NAVY,
    SIGNAL_GOLD,
    SOFT_WHITE,
    apply_gold_glow_theme,
    gold_glow_line_chart,
)


class ChartAppearanceContracts(unittest.TestCase):
    def test_weekly_chart_has_gold_signal_glow_and_all_cyan_peaks(self):
        figure = gold_glow_line_chart([10, 14, 12, 14], [1, 2, 3, 4], title="Weekly")
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(figure.data[0].line.color, "rgba(245,197,66,.16)")
        self.assertEqual(figure.data[1].line.color, SIGNAL_GOLD)
        self.assertLessEqual(figure.data[1].line.width, 3)
        self.assertEqual(figure.data[2].marker.color, CIRCUIT_CYAN)
        self.assertEqual(list(figure.data[2].x), [2, 4])
        self.assertEqual(list(figure.data[2].y), [14.0, 14.0])

    def test_chart_theme_uses_midnight_canvas_and_soft_white_type(self):
        figure = gold_glow_line_chart([1, 2, 3])
        self.assertEqual(figure.layout.plot_bgcolor, MIDNIGHT_NAVY)
        self.assertEqual(figure.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(figure.layout.font.color, SOFT_WHITE)
        self.assertEqual(figure.layout.hovermode, "x unified")

    def test_theme_can_style_non_line_figures_without_changing_data(self):
        figure = go.Figure(go.Bar(x=["A", "B"], y=[2, 4]))
        styled = apply_gold_glow_theme(figure, height=410)
        self.assertIs(styled, figure)
        self.assertEqual(styled.layout.height, 410)
        self.assertEqual(list(styled.data[0].y), [2, 4])
        self.assertEqual(styled.layout.colorway[0], SIGNAL_GOLD)

    def test_mismatched_axes_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            gold_glow_line_chart([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
