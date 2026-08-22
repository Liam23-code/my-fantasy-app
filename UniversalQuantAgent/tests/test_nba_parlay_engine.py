"""Offline contract tests for the NBA parlay engine."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_parlay_engine import evaluate_parlay, make_leg, nba_detect_correlations, rank_parlays


def _leg(**overrides) -> dict:
    base = dict(
        description="leg", model_probability=0.55, price=-110.0,
        player_id="p1", team="DEN", game_id="g1", market="points", side="over", confidence=0.7,
    )
    base.update(overrides)
    return make_leg(**base)


class NbaDetectCorrelationsTests(unittest.TestCase):
    def test_same_player_points_and_pra_over_is_detected(self):
        legs = [
            _leg(description="A", market="points", side="over", player_id="p1"),
            _leg(description="B", market="PRA", side="over", player_id="p1"),
        ]
        findings = nba_detect_correlations(legs)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "overlapping_stat_categories")

    def test_same_player_points_and_rebounds_not_flagged(self):
        # points and rebounds don't structurally overlap the way PRA does.
        legs = [
            _leg(description="A", market="points", side="over", player_id="p1"),
            _leg(description="B", market="rebounds", side="over", player_id="p1"),
        ]
        self.assertEqual(nba_detect_correlations(legs), [])

    def test_teammate_scoring_stack_detected(self):
        legs = [
            _leg(description="A", market="points", side="over", player_id="p1", team="DEN"),
            _leg(description="B", market="points", side="over", player_id="p2", team="DEN"),
        ]
        findings = nba_detect_correlations(legs)
        self.assertEqual(findings[0]["kind"], "teammate_scoring_stack")

    def test_different_teams_no_stack(self):
        legs = [
            _leg(description="A", market="points", side="over", player_id="p1", team="DEN"),
            _leg(description="B", market="points", side="over", player_id="p2", team="BOS"),
        ]
        self.assertEqual(nba_detect_correlations(legs), [])

    def test_generic_moneyline_total_pattern_still_fires_for_nba(self):
        legs = [
            make_leg(description="DEN ML", model_probability=0.65, price=-150.0, game_id="g1", market="moneyline", side="home"),
            make_leg(description="Over 230.5", model_probability=0.5, price=-110.0, game_id="g1", market="total", side="over"),
        ]
        findings = nba_detect_correlations(legs)
        self.assertEqual(findings[0]["kind"], "favorite_and_game_total_over")


class NbaEvaluateParlayTests(unittest.TestCase):
    def test_requires_at_least_two_legs(self):
        with self.assertRaises(ValueError):
            evaluate_parlay([_leg()])

    def test_correlated_legs_raise_adjusted_probability_above_naive(self):
        legs = [
            _leg(description="A", market="points", side="over", player_id="p1", model_probability=0.6),
            _leg(description="B", market="PRA", side="over", player_id="p1", model_probability=0.6),
        ]
        result = evaluate_parlay(legs)
        self.assertGreater(result["adjusted_hit_probability"], result["naive_hit_probability"])
        self.assertEqual(len(result["correlations_detected"]), 1)

    def test_uncorrelated_legs_naive_equals_adjusted(self):
        legs = [
            _leg(description="A", market="points", side="over", player_id="p1", team="DEN", model_probability=0.6),
            _leg(description="B", market="assists", side="over", player_id="p2", team="BOS", model_probability=0.55),
        ]
        result = evaluate_parlay(legs)
        self.assertAlmostEqual(result["naive_hit_probability"], result["adjusted_hit_probability"])

    def test_rank_parlays_orders_by_adjusted_ev(self):
        good = [
            _leg(description="A", model_probability=0.7, price=150.0, player_id="p1"),
            _leg(description="B", model_probability=0.7, price=150.0, player_id="p2"),
        ]
        bad = [
            _leg(description="C", model_probability=0.3, price=-200.0, player_id="p3"),
            _leg(description="D", model_probability=0.3, price=-200.0, player_id="p4"),
        ]
        ranked = rank_parlays([bad, good])
        self.assertEqual(ranked[0]["legs"], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
