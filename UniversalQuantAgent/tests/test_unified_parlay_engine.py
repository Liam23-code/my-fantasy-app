"""Offline contract tests for the cross-sport (NFL + NBA + CFB + CBB + MLB + NHL) parlay engine."""
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


def _cfb_leg(**overrides):
    base = dict(description="cfb leg", model_probability=0.55, price=-110.0, team="OSU", market="passing_yards", side="over", confidence=0.7)
    base.update(overrides)
    return make_unified_leg("CFB", **base)


def _cbb_leg(**overrides):
    base = dict(description="cbb leg", model_probability=0.55, price=-110.0, player_id="p1", team="DUKE", market="points", side="over", confidence=0.7)
    base.update(overrides)
    return make_unified_leg("CBB", **base)


def _mlb_leg(**overrides):
    base = dict(description="mlb leg", model_probability=0.55, price=-110.0, player_id="p1", team="NYY", game_id="g1", market="hits", side="over", confidence=0.7)
    base.update(overrides)
    return make_unified_leg("MLB", **base)


def _nhl_leg(**overrides):
    base = dict(description="nhl leg", model_probability=0.55, price=-110.0, player_id="p1", team="TOR", market="shots", side="over", confidence=0.7)
    base.update(overrides)
    return make_unified_leg("NHL", **base)


class MakeUnifiedLegTests(unittest.TestCase):
    def test_tags_the_leg_with_its_sport(self):
        leg = _nfl_leg()
        self.assertEqual(leg["sport"], "NFL")

    def test_rejects_an_unknown_sport(self):
        # Not "MLB" -- MLB joined the six supported sports this cycle and
        # is a real, valid sport now.
        with self.assertRaises(ValueError):
            make_unified_leg("XFL", description="x", model_probability=0.5, price=-110.0)


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

    def test_cfb_reuses_nfl_pattern_and_cbb_reuses_nba_pattern(self):
        cfb_qb = _cfb_leg(team="OSU", market="passing_yards", side="over")
        cfb_wr = _cfb_leg(team="OSU", market="receiving_yards", side="over")
        cbb_pts = _cbb_leg(player_id="p1", market="points", side="over")
        cbb_pra = _cbb_leg(player_id="p1", market="PRA", side="over")
        findings = detect_cross_sport_correlations([cfb_qb, cfb_wr, cbb_pts, cbb_pra])
        kinds = {finding["kind"] for finding in findings}
        self.assertEqual(kinds, {"qb_pass_catcher_stack", "overlapping_stat_categories"})

    def test_all_six_sports_pairwise_never_cross_correlated(self):
        legs = [_nfl_leg(), _nba_leg(), _cfb_leg(), _cbb_leg(), _mlb_leg(), _nhl_leg()]
        # Each sport appears only once -- no same-sport pair exists, so no
        # detector has >= 2 legs to check; nothing should ever fire.
        self.assertEqual(detect_cross_sport_correlations(legs), [])

    def test_mlb_pattern_fires_within_the_mlb_subset(self):
        hr = _mlb_leg(player_id="p1", team="NYY", game_id="g1", market="home_runs", side="over")
        tb = _mlb_leg(player_id="p1", team="NYY", game_id="g1", market="total_bases", side="over")
        nhl_leg = _nhl_leg()
        findings = detect_cross_sport_correlations([hr, tb, nhl_leg])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "hr_and_total_bases")
        self.assertEqual(findings[0]["legs"], (0, 1))

    def test_nhl_pattern_fires_within_the_nhl_subset(self):
        goals = _nhl_leg(player_id="p1", team="TOR", market="goals", side="over")
        assists = _nhl_leg(player_id="p1", team="TOR", market="assists", side="over")
        mlb_leg = _mlb_leg()
        findings = detect_cross_sport_correlations([goals, assists, mlb_leg])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["kind"], "goals_and_assists")
        self.assertEqual(findings[0]["legs"], (0, 1))

    def test_an_mlb_leg_and_an_nhl_leg_are_never_correlated(self):
        mlb_leg = _mlb_leg(market="hits", side="over")
        nhl_leg = _nhl_leg(market="shots", side="over")
        self.assertEqual(detect_cross_sport_correlations([mlb_leg, nhl_leg]), [])


class EvaluateCrossSportParlayTests(unittest.TestCase):
    def test_requires_at_least_two_legs(self):
        with self.assertRaises(ValueError):
            evaluate_cross_sport_parlay([_nfl_leg()])

    def test_rejects_a_leg_missing_a_valid_sport_tag(self):
        # Not "MLB" -- see MakeUnifiedLegTests.test_rejects_an_unknown_sport.
        legs = [_nfl_leg(), {**_nba_leg(), "sport": "XFL"}]
        with self.assertRaises(ValueError):
            evaluate_cross_sport_parlay(legs)

    def test_mixed_sport_parlay_evaluates_and_reports_both_sports(self):
        result = evaluate_cross_sport_parlay([_nfl_leg(), _nba_leg()])
        self.assertEqual(result["sports"], ["NBA", "NFL"])
        self.assertEqual(result["num_legs"], 2)
        self.assertIn("adjusted_hit_probability", result)
        self.assertIn("risk_tier", result)

    def test_all_six_sports_in_one_parlay(self):
        result = evaluate_cross_sport_parlay([_nfl_leg(), _nba_leg(), _cfb_leg(), _cbb_leg(), _mlb_leg(), _nhl_leg()])
        self.assertEqual(result["sports"], ["CBB", "CFB", "MLB", "NBA", "NFL", "NHL"])
        self.assertEqual(result["num_legs"], 6)

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
