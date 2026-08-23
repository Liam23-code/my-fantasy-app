"""Offline contract tests for the cross-sport (NFL + NBA) parlay engine."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.unified_parlay_engine import (
    detect_cross_sport_correlations,
    evaluate_cross_sport_parlay,
    make_unified_leg,
)


def _nfl_leg(**overrides):
    base = dict(description="nfl leg", model_probability=0.55, price=-110.0, team="KC", market="passing_yards", side="over", confidence=0.7)
    base.update(overrides)
    return make_unified_leg("NFL", **base)


def _nba_leg(**overrides):
    base = dict(description="nba leg", model_probability=0.55, price=-110.0, player_id="p1", team="DEN", market="points", side="over", confidence=0.7)
    base.update(overrides)
    return make_unified_leg("NBA", **base)


class MakeUnifiedLegTests(unittest.TestCase):
    def test_tags_the_leg_with_its_sport(self):
        leg = _nfl_leg()
        self.assertEqual(leg["sport"], "NFL")

    def test_rejects_an_unknown_sport(self):
        with self.assertRaises(ValueError):
            make_unified_leg("MLB", description="x", model_probability=0.5, price=-110.0)


class DetectCrossSportCorrelationsTests(unittest.TestCase):
    def test_nfl_pattern_still_fires_within_the_nfl_subset(self):
        qb = _nfl_leg(team="KC", market="passing_yards", side="over")
        wr = _nfl_leg(team="KC", market="receiving_yards", side="over")
        nba_leg = _nba_leg()
        findings = detect_cross_sport_correlations([qb, wr, nba_leg])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "qb_pass_catcher_stack")
        self.assertEqual(findings[0]["legs"], (0, 1))  # indices in the original mixed list

    def test_nba_pattern_still_fires_within_the_nba_subset(self):
        nfl_leg = _nfl_leg()
        points = _nba_leg(player_id="p1", market="points", side="over")
        pra = _nba_leg(player_id="p1", market="PRA", side="over")
        findings = detect_cross_sport_correlations([nfl_leg, points, pra])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "overlapping_stat_categories")
        self.assertEqual(findings[0]["legs"], (1, 2))

    def test_an_nfl_leg_and_an_nba_leg_are_never_correlated(self):
        nfl_leg = _nfl_leg(market="passing_yards", side="over")
        nba_leg = _nba_leg(market="points", side="over")  # same "over" side, unrelated sports
        self.assertEqual(detect_cross_sport_correlations([nfl_leg, nba_leg]), [])

    def test_both_sports_patterns_found_simultaneously_with_correct_indices(self):
        qb = _nfl_leg(team="KC", market="passing_yards", side="over")
        wr = _nfl_leg(team="KC", market="receiving_yards", side="over")
        points = _nba_leg(player_id="p1", market="points", side="over")
        pra = _nba_leg(player_id="p1", market="PRA", side="over")
        findings = detect_cross_sport_correlations([qb, wr, points, pra])
        kinds = {finding["kind"] for finding in findings}
        self.assertEqual(kinds, {"qb_pass_catcher_stack", "overlapping_stat_categories"})


class EvaluateCrossSportParlayTests(unittest.TestCase):
    def test_requires_at_least_two_legs(self):
        with self.assertRaises(ValueError):
            evaluate_cross_sport_parlay([_nfl_leg()])

    def test_rejects_a_leg_missing_a_valid_sport_tag(self):
        legs = [_nfl_leg(), {**_nba_leg(), "sport": "MLB"}]
        with self.assertRaises(ValueError):
            evaluate_cross_sport_parlay(legs)

    def test_mixed_sport_parlay_evaluates_and_reports_both_sports(self):
        result = evaluate_cross_sport_parlay([_nfl_leg(), _nba_leg()])
        self.assertEqual(result["sports"], ["NBA", "NFL"])
        self.assertEqual(result["num_legs"], 2)
        self.assertIn("adjusted_hit_probability", result)
        self.assertIn("risk_tier", result)

    def test_uncorrelated_cross_sport_legs_naive_equals_adjusted(self):
        result = evaluate_cross_sport_parlay([_nfl_leg(), _nba_leg()])
        self.assertAlmostEqual(result["naive_hit_probability"], result["adjusted_hit_probability"])

    def test_same_sport_correlation_still_adjusts_within_a_mixed_parlay(self):
        points = _nba_leg(player_id="p1", market="points", side="over")
        pra = _nba_leg(player_id="p1", market="PRA", side="over")
        result = evaluate_cross_sport_parlay([_nfl_leg(), points, pra])
        self.assertGreater(result["adjusted_hit_probability"], result["naive_hit_probability"])
        self.assertEqual(len(result["correlations_detected"]), 1)


if __name__ == "__main__":
    unittest.main()
