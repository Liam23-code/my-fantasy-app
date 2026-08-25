"""Tests for the MLB parlay engine: the four real baseball correlation patterns plus generic parlay math."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mlb_parlay_engine import evaluate_parlay, make_leg, mlb_detect_correlations, rank_parlays


def _leg(**overrides):
    base = {"description": "leg", "model_probability": 0.55, "price": -110.0}
    base.update(overrides)
    return make_leg(**base)


class HrAndTotalBasesCorrelationTests(unittest.TestCase):
    def test_fires_for_same_player_same_game_same_direction(self):
        hr = _leg(description="hr", player_id="p1", team="NYY", game_id="g1", market="home_runs", side="over")
        tb = _leg(description="tb", player_id="p1", team="NYY", game_id="g1", market="total_bases", side="over")
        findings = mlb_detect_correlations([hr, tb])
        self.assertEqual(findings[0]["kind"], "hr_and_total_bases")

    def test_does_not_fire_for_different_players(self):
        hr = _leg(description="hr", player_id="p1", team="NYY", game_id="g1", market="home_runs", side="over")
        tb = _leg(description="tb", player_id="p2", team="NYY", game_id="g1", market="total_bases", side="over")
        self.assertEqual(mlb_detect_correlations([hr, tb]), [])


class HitsAndRbiCorrelationTests(unittest.TestCase):
    def test_fires_for_same_player_same_game_same_direction(self):
        hits = _leg(description="h", player_id="p1", team="NYY", game_id="g1", market="hits", side="over")
        rbi = _leg(description="rbi", player_id="p1", team="NYY", game_id="g1", market="rbi", side="over")
        findings = mlb_detect_correlations([hits, rbi])
        self.assertEqual(findings[0]["kind"], "hits_and_rbi")

    def test_does_not_fire_for_opposite_directions(self):
        hits = _leg(description="h", player_id="p1", team="NYY", game_id="g1", market="hits", side="over")
        rbi = _leg(description="rbi", player_id="p1", team="NYY", game_id="g1", market="rbi", side="under")
        self.assertEqual(mlb_detect_correlations([hits, rbi]), [])


class PitcherVsHighStrikeoutLineupCorrelationTests(unittest.TestCase):
    def test_fires_for_opposing_strikeout_legs_same_game(self):
        pitcher_ks = _leg(description="pk", player_id="pitcher", team="NYY", game_id="g1", market="strikeouts", side="over")
        batter_ks = _leg(description="bk", player_id="batter", team="BOS", game_id="g1", market="strikeouts", side="over")
        findings = mlb_detect_correlations([pitcher_ks, batter_ks])
        self.assertEqual(findings[0]["kind"], "pitcher_vs_high_strikeout_lineup")

    def test_does_not_fire_for_same_team(self):
        a = _leg(description="a", player_id="p1", team="NYY", game_id="g1", market="strikeouts", side="over")
        b = _leg(description="b", player_id="p2", team="NYY", game_id="g1", market="strikeouts", side="over")
        self.assertEqual(mlb_detect_correlations([a, b]), [])


class TeammateStolenBaseCorrelationTests(unittest.TestCase):
    def test_fires_for_two_teammates_stolen_bases(self):
        a = _leg(description="a", player_id="p1", team="NYY", game_id="g1", market="stolen_bases", side="over")
        b = _leg(description="b", player_id="p2", team="NYY", game_id="g1", market="stolen_bases", side="over")
        findings = mlb_detect_correlations([a, b])
        self.assertEqual(findings[0]["kind"], "teammate_stolen_base_environment")

    def test_does_not_fire_for_the_same_player(self):
        a = _leg(description="a", player_id="p1", team="NYY", game_id="g1", market="stolen_bases", side="over")
        b = _leg(description="b", player_id="p1", team="NYY", game_id="g1", market="stolen_bases", side="over")
        self.assertEqual(mlb_detect_correlations([a, b]), [])


class EvaluateParlayTests(unittest.TestCase):
    def test_requires_at_least_two_legs(self):
        with self.assertRaises(ValueError):
            evaluate_parlay([_leg()])

    def test_correlated_legs_raise_adjusted_probability_above_naive(self):
        hr = _leg(description="hr", player_id="p1", team="NYY", game_id="g1", market="home_runs", side="over", model_probability=0.3)
        tb = _leg(description="tb", player_id="p1", team="NYY", game_id="g1", market="total_bases", side="over", model_probability=0.5)
        result = evaluate_parlay([hr, tb])
        self.assertGreater(result["adjusted_hit_probability"], result["naive_hit_probability"])
        self.assertEqual(len(result["correlations_detected"]), 1)

    def test_uncorrelated_legs_have_no_adjustment(self):
        a = _leg(description="a", player_id="p1", team="NYY", game_id="g1", market="hits", side="over")
        b = _leg(description="b", player_id="p2", team="BOS", game_id="g2", market="walks", side="under")
        result = evaluate_parlay([a, b])
        self.assertEqual(result["adjusted_hit_probability"], result["naive_hit_probability"])


class RankParlaysTests(unittest.TestCase):
    def test_ranks_by_adjusted_ev_descending(self):
        low_ev = [_leg(description="a", model_probability=0.2, price=-110.0), _leg(description="b", model_probability=0.2, price=-110.0)]
        high_ev = [_leg(description="c", model_probability=0.8, price=150.0), _leg(description="d", model_probability=0.8, price=150.0)]
        ranked = rank_parlays([low_ev, high_ev])
        self.assertGreaterEqual(ranked[0]["adjusted_ev"], ranked[1]["adjusted_ev"])


if __name__ == "__main__":
    unittest.main()
