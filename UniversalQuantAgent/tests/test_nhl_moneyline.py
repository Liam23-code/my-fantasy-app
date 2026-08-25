"""Tests for the NHL moneyline model: composite team ratings, win probability, edges."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nhl_moneyline_model import evaluate_game, evaluate_games, fair_moneyline, win_probability_from_composite


class WinProbabilityFromCompositeTests(unittest.TestCase):
    def test_even_teams_favor_home_by_the_home_ice_constant(self):
        probability = win_probability_from_composite(1.0, 1.0, home_ice_advantage=0.55)
        self.assertAlmostEqual(probability, 0.55, places=2)

    def test_stronger_home_composite_raises_win_probability(self):
        even = win_probability_from_composite(1.0, 1.0)
        favored = win_probability_from_composite(1.3, 1.0)
        self.assertGreater(favored, even)


class FairMoneylineTests(unittest.TestCase):
    def test_no_context_is_a_pick_em_favoring_home_ice_only(self):
        fair = fair_moneyline("TOR", "MTL")
        self.assertAlmostEqual(fair["home_win_probability"], 0.55, places=2)

    def test_stronger_goaltending_context_favors_that_team(self):
        fair = fair_moneyline("TOR", "MTL", home_context={"goaltending_strength": 1.3})
        self.assertGreater(fair["home_win_probability"], 0.55)


class EvaluateGameTests(unittest.TestCase):
    def test_no_moneyline_odds_still_returns_the_model_fair_line(self):
        result = evaluate_game({"home_team": "TOR", "away_team": "MTL"})
        self.assertIn("model", result)
        self.assertNotIn("moneyline", result)

    def test_moneyline_odds_produce_an_edge_and_recommended_side(self):
        result = evaluate_game({"home_team": "TOR", "away_team": "MTL", "moneyline": {"home": -150.0, "away": 130.0}})
        self.assertIn(result["moneyline"]["recommended_side"], {"home", "away"})


class EvaluateGamesTests(unittest.TestCase):
    def test_skips_games_missing_a_team(self):
        games = [{"home_team": "TOR"}, {"home_team": "TOR", "away_team": "MTL"}]
        rows = evaluate_games(games)
        self.assertEqual(len(rows), 1)

    def test_results_sorted_by_moneyline_edge_descending(self):
        games = [
            {"home_team": "TOR", "away_team": "MTL", "moneyline": {"home": -110.0, "away": -110.0}},
            {"home_team": "EDM", "away_team": "CGY", "moneyline": {"home": 300.0, "away": -400.0}},
        ]
        rows = evaluate_games(games)
        edges = [row["moneyline"]["recommended_edge"] for row in rows]
        self.assertEqual(edges, sorted(edges, reverse=True))


if __name__ == "__main__":
    unittest.main()
