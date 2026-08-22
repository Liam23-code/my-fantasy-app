"""Offline contract tests for the price-aware NBA prop evaluation layer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_prop_model import (
    index_props_by_player_and_category,
    price_aware_evaluations,
    price_prop_comparison,
)


def _comparison_row(**overrides) -> dict:
    row = {
        "player": "Nikola Jokic",
        "team": "DEN",
        "category": "points",
        "projection": 27.0,
        "minutes_adjusted_projection": 27.0,
        "sportsbook_line": 25.5,
        "edge": 1.5,
        "confidence_low": 23.0,
        "confidence_high": 31.0,
        "best_sportsbook": "default",
        "lean": "over",
        "key_drivers": ["Stable role."],
        "sportsbook": "default",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    row.update(overrides)
    return row


class PricePropComparisonTests(unittest.TestCase):
    def test_projection_above_line_favors_over(self):
        result = price_prop_comparison(_comparison_row(), {"over_price": -110.0, "under_price": -110.0})
        self.assertEqual(result["recommended_priced_side"], "over")
        self.assertGreater(result["model_probability_over"], 0.5)
        self.assertGreater(result["probability_edge_over"], 0.0)

    def test_projection_below_line_favors_under(self):
        row = _comparison_row(sportsbook_line=30.0)
        result = price_prop_comparison(row, {"over_price": -110.0, "under_price": -110.0})
        self.assertEqual(result["recommended_priced_side"], "under")

    def test_market_fair_probabilities_sum_to_one(self):
        result = price_prop_comparison(_comparison_row(), {"over_price": -130.0, "under_price": 110.0})
        total = result["market_fair_probability_over"] + result["market_fair_probability_under"]
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_missing_prices_default_to_minus_110(self):
        result = price_prop_comparison(_comparison_row(), {})
        self.assertEqual(result["over_price"], -110.0)
        self.assertEqual(result["under_price"], -110.0)

    def test_original_comparison_fields_pass_through(self):
        result = price_prop_comparison(_comparison_row(), {"over_price": -110.0, "under_price": -110.0})
        self.assertEqual(result["player"], "Nikola Jokic")
        self.assertEqual(result["lean"], "over")

    def test_zero_width_confidence_band_does_not_explode_edge(self):
        # A degenerate zero-width band shouldn't produce infinite/NaN edges.
        row = _comparison_row(confidence_low=27.0, confidence_high=27.0)
        result = price_prop_comparison(row, {"over_price": -110.0, "under_price": -110.0})
        self.assertTrue(0.0 <= result["model_probability_over"] <= 1.0)


class PriceAwareEvaluationsTests(unittest.TestCase):
    def test_matches_by_normalized_player_name_and_category(self):
        # comparison row carries the accented, nba_api-style raw name;
        # props_by_key was built from the already-normalized loader output.
        comparison_rows = [_comparison_row(player="Nikola Jokić")]
        props = [{"player_name": "Nikola Jokic", "team": "DEN", "category": "points", "line": 25.5, "over_price": -110.0, "under_price": -110.0}]
        props_by_key = index_props_by_player_and_category(props)
        priced = price_aware_evaluations(comparison_rows, props_by_key)
        self.assertEqual(len(priced), 1)
        self.assertEqual(priced[0]["recommended_priced_side"], "over")

    def test_unmatched_rows_are_skipped_not_errored(self):
        comparison_rows = [_comparison_row(player="Someone Unpriced", category="assists")]
        priced = price_aware_evaluations(comparison_rows, {})
        self.assertEqual(priced, [])

    def test_sorted_by_largest_absolute_ev_first(self):
        comparison_rows = [
            _comparison_row(player="Small Edge", category="points", sportsbook_line=26.9),
            _comparison_row(player="Big Edge", category="points", sportsbook_line=20.0),
        ]
        props = [
            {"player_name": "Small Edge", "category": "points", "line": 26.9, "over_price": -110.0, "under_price": -110.0},
            {"player_name": "Big Edge", "category": "points", "line": 20.0, "over_price": -110.0, "under_price": -110.0},
        ]
        priced = price_aware_evaluations(comparison_rows, index_props_by_player_and_category(props))
        self.assertEqual(priced[0]["player"], "Big Edge")


if __name__ == "__main__":
    unittest.main()
