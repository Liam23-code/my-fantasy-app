"""Tests for the NHL parlay engine: goal/assist overlap and teammate goal-stack correlation patterns."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nhl_parlay_engine import evaluate_parlay, make_leg, nhl_detect_correlations, rank_parlays


def _leg(**overrides):
    base = {"description": "leg", "model_probability": 0.55, "price": -110.0}
    base.update(overrides)
    return make_leg(**base)


class GoalsAndAssistsCorrelationTests(unittest.TestCase):
    def test_fires_for_same_player_same_direction(self):
        goals = _leg(description="g", player_id="p1", team="TOR", market="goals", side="over")
        assists = _leg(description="a", player_id="p1", team="TOR", market="assists", side="over")
        findings = nhl_detect_correlations([goals, assists])
        self.assertEqual(findings[0]["kind"], "goals_and_assists")

    def test_does_not_fire_for_different_players(self):
        goals = _leg(description="g", player_id="p1", team="TOR", market="goals", side="over")
        assists = _leg(description="a", player_id="p2", team="TOR", market="assists", side="over")
        self.assertEqual(nhl_detect_correlations([goals, assists]), [])


class TeammateGoalStackCorrelationTests(unittest.TestCase):
    def test_fires_for_two_teammates_goals(self):
        a = _leg(description="a", player_id="p1", team="TOR", market="goals", side="over")
        b = _leg(description="b", player_id="p2", team="TOR", market="goals", side="over")
        findings = nhl_detect_correlations([a, b])
        self.assertEqual(findings[0]["kind"], "teammate_goal_stack")

    def test_does_not_fire_for_different_teams(self):
        a = _leg(description="a", player_id="p1", team="TOR", market="goals", side="over")
        b = _leg(description="b", player_id="p2", team="MTL", market="goals", side="over")
        self.assertEqual(nhl_detect_correlations([a, b]), [])


class EvaluateParlayTests(unittest.TestCase):
    def test_requires_at_least_two_legs(self):
        with self.assertRaises(ValueError):
            evaluate_parlay([_leg()])

    def test_correlated_legs_raise_adjusted_probability_above_naive(self):
        goals = _leg(description="g", player_id="p1", team="TOR", market="goals", side="over", model_probability=0.3)
        assists = _leg(description="a", player_id="p1", team="TOR", market="assists", side="over", model_probability=0.4)
        result = evaluate_parlay([goals, assists])
        self.assertGreater(result["adjusted_hit_probability"], result["naive_hit_probability"])


class RankParlaysTests(unittest.TestCase):
    def test_ranks_by_adjusted_ev_descending(self):
        low_ev = [_leg(description="a", model_probability=0.2, price=-110.0), _leg(description="b", model_probability=0.2, price=-110.0)]
        high_ev = [_leg(description="c", model_probability=0.8, price=150.0), _leg(description="d", model_probability=0.8, price=150.0)]
        ranked = rank_parlays([low_ev, high_ev])
        self.assertGreaterEqual(ranked[0]["adjusted_ev"], ranked[1]["adjusted_ev"])


if __name__ == "__main__":
    unittest.main()
