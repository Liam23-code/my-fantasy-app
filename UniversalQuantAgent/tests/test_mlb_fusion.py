"""Tests for the MLB fusion model: blending the season baseline with the matchup layer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mlb_fusion_model import WEIGHTS, combined_matchup_adjustment, fuse_projection, sample_size_confidence
from modules.mlb_season_model import project_season_baseline
from modules.mlb_common import STAT_CATEGORIES


class WeightsTests(unittest.TestCase):
    def test_every_named_weight_from_the_spec_is_present(self):
        for name in ("reliability", "sample_size", "matchup_difficulty", "park_factor", "pitcher_quality", "lineup_protection"):
            self.assertIn(name, WEIGHTS)

    def test_confidence_weights_sum_to_one(self):
        self.assertAlmostEqual(WEIGHTS["reliability"] + WEIGHTS["sample_size"], 1.0, places=6)

    def test_matchup_weights_sum_to_one(self):
        total = WEIGHTS["matchup_difficulty"] + WEIGHTS["park_factor"] + WEIGHTS["pitcher_quality"] + WEIGHTS["lineup_protection"]
        self.assertAlmostEqual(total, 1.0, places=6)


class SampleSizeConfidenceTests(unittest.TestCase):
    def test_zero_games_is_zero_confidence(self):
        self.assertEqual(sample_size_confidence(0), 0.0)

    def test_full_sample_caps_at_one(self):
        self.assertEqual(sample_size_confidence(200), 1.0)


class CombinedMatchupAdjustmentTests(unittest.TestCase):
    def test_all_neutral_inputs_produce_zero_adjustment(self):
        self.assertEqual(combined_matchup_adjustment(), 0.0)

    def test_favorable_matchup_produces_a_positive_delta(self):
        delta = combined_matchup_adjustment(matchup_difficulty_multiplier=1.2, park_factor_multiplier=1.1, pitcher_quality_multiplier=1.1, lineup_protection_multiplier=1.05)
        self.assertGreater(delta, 0.0)

    def test_unfavorable_matchup_produces_a_negative_delta(self):
        delta = combined_matchup_adjustment(matchup_difficulty_multiplier=0.8, park_factor_multiplier=0.9, pitcher_quality_multiplier=0.85, lineup_protection_multiplier=0.95)
        self.assertLess(delta, 0.0)


class FuseProjectionTests(unittest.TestCase):
    def setUp(self):
        game_log = [{"hits": 1.0, "home_runs": 0.3, "rbi": 0.6, "total_bases": 1.7, "strikeouts": 0.0, "walks": 0.4, "stolen_bases": 0.1} for _ in range(80)]
        self.baseline = project_season_baseline({"game_log": game_log})

    def test_neutral_matchup_leaves_projection_close_to_baseline(self):
        fused = fuse_projection(self.baseline)
        for category in STAT_CATEGORIES:
            self.assertAlmostEqual(fused[category]["projection"], fused[category]["season_baseline"], places=4)
            self.assertEqual(fused[category]["adjustment_delta"], 0.0)

    def test_favorable_matchup_raises_the_projection_above_baseline(self):
        fused = fuse_projection(
            self.baseline,
            matchup_difficulty_multiplier=1.3,
            park_factor_by_category={category: 1.15 for category in STAT_CATEGORIES},
            pitcher_quality_multiplier=1.2,
            lineup_protection_multiplier=1.1,
        )
        for category in STAT_CATEGORIES:
            self.assertGreaterEqual(fused[category]["projection"], fused[category]["season_baseline"])

    def test_every_category_has_a_risk_tier_and_confidence(self):
        fused = fuse_projection(self.baseline)
        for category in STAT_CATEGORIES:
            self.assertIn(fused[category]["risk_tier"], {"low", "medium", "high"})
            self.assertTrue(0.0 <= fused[category]["confidence"] <= 1.0)

    def test_low_reliability_player_is_damped_more_than_high_reliability_player(self):
        thin_log = [{"hits": 1.0} for _ in range(2)]
        thin_baseline = project_season_baseline({"game_log": thin_log})
        thick_log = [{"hits": 1.0} for _ in range(150)]
        thick_baseline = project_season_baseline({"game_log": thick_log})

        thin_fused = fuse_projection(thin_baseline, matchup_difficulty_multiplier=1.3)
        thick_fused = fuse_projection(thick_baseline, matchup_difficulty_multiplier=1.3)
        self.assertLess(abs(thin_fused["hits"]["adjustment_delta"]), abs(thick_fused["hits"]["adjustment_delta"]))

    def test_missing_category_in_baseline_falls_back_to_a_neutral_zero_projection(self):
        fused = fuse_projection({})
        for category in STAT_CATEGORIES:
            self.assertEqual(fused[category]["projection"], 0.0)


if __name__ == "__main__":
    unittest.main()
