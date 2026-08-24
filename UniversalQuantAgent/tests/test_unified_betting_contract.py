"""Tests for the unified betting-engine contract dispatch layer: load_odds/compute_ev/compute_confidence/build_parlays."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.unified_betting_contract import VALID_SPORTS, build_parlays, compute_confidence, compute_ev, load_odds
from modules.cfb_parlay_engine import make_leg as cfb_make_leg
from modules.cbb_parlay_engine import make_leg as cbb_make_leg
from betting.parlay_engine import make_leg as nfl_make_leg
from modules.nba_parlay_engine import make_leg as nba_make_leg


class LoadOddsTests(unittest.TestCase):
    def test_rejects_unknown_sport(self):
        with self.assertRaises(ValueError):
            load_odds("MLB")

    def test_every_sport_returns_the_same_top_level_shape(self):
        for sport in VALID_SPORTS:
            result = load_odds(sport)
            self.assertEqual(set(result), {"props", "games"})
            self.assertIsInstance(result["props"], list)
            self.assertIsInstance(result["games"], dict)

    def test_nfl_and_nba_and_cbb_have_real_default_props(self):
        # CFB's default is empty by design (no CFBD_API_KEY configured yet)
        # -- see offline_data_contract.md; not asserted here.
        for sport in ("NFL", "NBA", "CBB"):
            result = load_odds(sport)
            self.assertGreater(len(result["props"]), 0, f"{sport} should have real default props")


class ComputeEvTests(unittest.TestCase):
    def test_rejects_unknown_sport(self):
        with self.assertRaises(ValueError):
            compute_ev("MLB", [])

    def test_nfl_requires_players_by_id_context(self):
        with self.assertRaises(KeyError):
            compute_ev("NFL", [])

    def test_nba_requires_comparison_rows_context(self):
        with self.assertRaises(KeyError):
            compute_ev("NBA", [])

    def test_cfb_needs_no_extra_context(self):
        props = [{"player_name": "Test QB", "team": "OSU", "category": "passing_yards", "line": 275.5, "over_price": -110.0, "under_price": -110.0}]
        rows = compute_ev("CFB", props)
        self.assertEqual(len(rows), 1)
        self.assertIn("recommended_edge", rows[0])

    def test_cbb_works_with_or_without_minutes_context(self):
        props = [{"player_name": "Test Guard", "team": "DUKE", "category": "points", "line": 18.5, "over_price": -110.0, "under_price": -110.0}]
        rows_no_context = compute_ev("CBB", props)
        rows_with_context = compute_ev("CBB", props, minutes_by_player={"Test Guard": 30.0})
        self.assertEqual(len(rows_no_context), 1)
        self.assertEqual(len(rows_with_context), 1)


class ComputeConfidenceTests(unittest.TestCase):
    def test_rejects_unknown_sport(self):
        with self.assertRaises(ValueError):
            compute_confidence("MLB", [])

    def test_every_sport_returns_the_same_row_shape(self):
        cfb_rows = compute_ev("CFB", [{"player_name": "A", "team": "OSU", "category": "passing_yards", "line": 275.5, "over_price": -110.0, "under_price": -110.0}])
        result = compute_confidence("CFB", cfb_rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]), {"description", "confidence", "risk_tier"})
        self.assertIsInstance(result[0]["confidence"], float)

    def test_cfb_confidence_derives_from_risk_tier_proxy(self):
        cfb_rows = compute_ev("CFB", [{"player_name": "A", "team": "OSU", "category": "passing_yards", "line": 275.5, "over_price": -110.0, "under_price": -110.0}])
        result = compute_confidence("CFB", cfb_rows)
        expected = {"low": 0.8, "medium": 0.5, "high": 0.25}[cfb_rows[0]["risk_tier"]]
        self.assertEqual(result[0]["confidence"], expected)


class BuildParlaysTests(unittest.TestCase):
    def test_rejects_unknown_sport(self):
        with self.assertRaises(ValueError):
            build_parlays("MLB", [])

    def test_every_sport_returns_the_same_output_shape(self):
        expected_keys = {"legs", "num_legs", "decimal_odds", "payout_per_100_stake", "naive_hit_probability", "adjusted_hit_probability", "correlations_detected", "naive_ev", "adjusted_ev", "confidence", "risk_tier"}
        cases = {
            "NFL": [nfl_make_leg(description="a", model_probability=0.6, price=120.0), nfl_make_leg(description="b", model_probability=0.6, price=120.0)],
            "NBA": [nba_make_leg(description="a", model_probability=0.6, price=120.0), nba_make_leg(description="b", model_probability=0.6, price=120.0)],
            "CFB": [cfb_make_leg(description="a", model_probability=0.6, price=120.0), cfb_make_leg(description="b", model_probability=0.6, price=120.0)],
            "CBB": [cbb_make_leg(description="a", model_probability=0.6, price=120.0), cbb_make_leg(description="b", model_probability=0.6, price=120.0)],
        }
        for sport, legs in cases.items():
            result = build_parlays(sport, legs)
            self.assertEqual(set(result), expected_keys, f"{sport} output shape mismatch")


if __name__ == "__main__":
    unittest.main()
