"""Tests for the MLB moneyline model: composite team ratings, win probability, edges."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mlb_moneyline_model import evaluate_game, evaluate_games, fair_moneyline, win_probability_from_composite


class WinProbabilityFromCompositeTests(unittest.TestCase):
    def test_even_teams_favor_home_by_the_home_field_constant(self):
        probability = win_probability_from_composite(1.0, 1.0, home_field_advantage=0.54)
        self.assertAlmostEqual(probability, 0.54, places=2)

    def test_stronger_home_composite_raises_win_probability(self):
        even = win_probability_from_composite(1.0, 1.0)
        favored = win_probability_from_composite(1.3, 1.0)
        self.assertGreater(favored, even)

    def test_probability_stays_within_bounds_for_extreme_ratings(self):
        probability = win_probability_from_composite(1.8, 0.3)
        self.assertTrue(0.0 < probability <= 0.98)


class FairMoneylineTests(unittest.TestCase):
    def test_no_context_is_a_pick_em_favoring_home_field_only(self):
        fair = fair_moneyline("NYY", "BOS")
        self.assertAlmostEqual(fair["home_win_probability"], 0.54, places=2)

    def test_stronger_pitching_context_favors_that_team(self):
        fair = fair_moneyline("NYY", "BOS", home_context={"pitcher_quality": 1.3, "bullpen_strength": 1.2})
        self.assertGreater(fair["home_win_probability"], 0.54)
        self.assertLess(fair["fair_home_moneyline"], 0)  # a favorite prices negative

    def test_team_names_are_normalized_to_uppercase(self):
        fair = fair_moneyline("nyy", "bos")
        self.assertEqual(fair["home_team"], "NYY")
        self.assertEqual(fair["away_team"], "BOS")


class EvaluateGameTests(unittest.TestCase):
    def test_no_moneyline_odds_still_returns_the_model_fair_line(self):
        result = evaluate_game({"home_team": "NYY", "away_team": "BOS"})
        self.assertIn("model", result)
        self.assertNotIn("moneyline", result)

    def test_moneyline_odds_produce_an_edge_and_recommended_side(self):
        result = evaluate_game({"home_team": "NYY", "away_team": "BOS", "moneyline": {"home": -150.0, "away": 130.0}})
        self.assertIn("recommended_side", result["moneyline"])
        self.assertIn(result["moneyline"]["recommended_side"], {"home", "away"})


class EvaluateGamesTests(unittest.TestCase):
    def test_skips_games_missing_a_team(self):
        games = [{"home_team": "NYY"}, {"home_team": "NYY", "away_team": "BOS"}]
        rows = evaluate_games(games)
        self.assertEqual(len(rows), 1)

    def test_context_by_team_is_applied_per_team(self):
        games = [{"home_team": "NYY", "away_team": "BOS", "moneyline": {"home": -110.0, "away": -110.0}}]
        context = {"NYY": {"pitcher_quality": 1.3}, "BOS": {"pitcher_quality": 0.8}}
        rows = evaluate_games(games, context_by_team=context)
        self.assertGreater(rows[0]["model"]["home_win_probability"], 0.5)

    def test_results_sorted_by_moneyline_edge_descending(self):
        games = [
            {"home_team": "NYY", "away_team": "BOS", "moneyline": {"home": -110.0, "away": -110.0}},
            {"home_team": "LAD", "away_team": "SF", "moneyline": {"home": 300.0, "away": -400.0}},
        ]
        rows = evaluate_games(games)
        edges = [row["moneyline"]["recommended_edge"] for row in rows]
        self.assertEqual(edges, sorted(edges, reverse=True))


if __name__ == "__main__":
    unittest.main()
