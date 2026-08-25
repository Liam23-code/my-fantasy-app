"""Tests for the five MLB DFS matchup modules: batter-vs-pitcher, ballpark, lineup, bullpen, defense."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mlb_batter_vs_pitcher import (
    career_bvp_adjustment,
    handedness_matchup_multiplier,
    hard_hit_barrel_index,
    matchup_multiplier,
    pitch_type_effectiveness,
)
from modules.mlb_ballpark_model import PARK_FACTORS, altitude_and_foul_territory_note, park_adjustment, team_park_factors
from modules.mlb_lineup_model import (
    expected_plate_appearances,
    lineup_context,
    lineup_protection_multiplier,
    platoon_usage_probability,
    stolen_base_environment,
)
from modules.mlb_bullpen_model import bullpen_era_rating, bullpen_fatigue_index, bullpen_strength, leverage_usage_score
from modules.mlb_defense_model import composite_defense_rating, defensive_efficiency_rating, outfield_arm_rating
from modules.mlb_common import STAT_CATEGORIES


class HandednessMatchupTests(unittest.TestCase):
    def test_opposite_handed_is_an_advantage(self):
        self.assertGreater(handedness_matchup_multiplier("L", "R"), 1.0)

    def test_same_handed_is_a_disadvantage(self):
        self.assertLess(handedness_matchup_multiplier("R", "R"), 1.0)

    def test_switch_hitter_is_neutral(self):
        self.assertEqual(handedness_matchup_multiplier("S", "R"), 1.0)

    def test_unknown_hand_is_neutral(self):
        self.assertEqual(handedness_matchup_multiplier("", "R"), 1.0)


class PitchTypeEffectivenessTests(unittest.TestCase):
    def test_weighted_by_real_pitch_usage(self):
        result = pitch_type_effectiveness({"fastball": 0.8, "slider": 0.2}, {"fastball": 0.20, "slider": 0.40})
        self.assertAlmostEqual(result, 0.8 * 0.20 + 0.2 * 0.40, places=4)

    def test_missing_pitch_type_defaults_to_a_neutral_whiff_rate(self):
        result = pitch_type_effectiveness({"curveball": 1.0}, {})
        self.assertEqual(result, 0.25)


class CareerBvpAdjustmentTests(unittest.TestCase):
    def test_no_at_bats_is_neutral(self):
        self.assertEqual(career_bvp_adjustment({}), 1.0)

    def test_small_sample_is_heavily_regressed(self):
        hot_streak = career_bvp_adjustment({"at_bats": 8, "hits": 6})  # a real .750 in a tiny sample
        naive_ratio = 0.750 / 0.250  # what an unregressed multiplier would be
        self.assertLess(hot_streak, naive_ratio)  # nowhere near a full, naive .750/.250 swing
        self.assertLessEqual(hot_streak, 1.15)  # and never past the disclosed cap


class HardHitBarrelIndexTests(unittest.TestCase):
    def test_league_average_inputs_are_neutral(self):
        self.assertAlmostEqual(hard_hit_barrel_index(0.35, 0.08), 1.0, places=2)

    def test_above_average_contact_quality_raises_the_index(self):
        self.assertGreater(hard_hit_barrel_index(0.45, 0.12), 1.0)


class BatterVsPitcherMatchupMultiplierTests(unittest.TestCase):
    def test_returns_bounded_combined_multiplier_and_components(self):
        result = matchup_multiplier(
            {"hand": "L", "whiff_rates": {"fastball": 0.22}, "hard_hit_rate": 0.38, "barrel_rate": 0.09, "bvp": {"at_bats": 20, "hits": 6}},
            {"hand": "R", "pitch_mix": {"fastball": 1.0}},
        )
        self.assertTrue(0.6 <= result["combined_multiplier"] <= 1.6)
        self.assertIn("handedness_multiplier", result)

    def test_missing_profile_data_still_returns_a_neutral_ish_result(self):
        result = matchup_multiplier({}, {})
        self.assertTrue(0.6 <= result["combined_multiplier"] <= 1.6)


class BallparkModelTests(unittest.TestCase):
    def test_every_team_has_real_park_factors(self):
        self.assertEqual(len(PARK_FACTORS), 30)
        for team, park in PARK_FACTORS.items():
            self.assertIn("hr", park)
            self.assertIn("altitude_ft", park)

    def test_unrecognized_team_gets_a_neutral_default(self):
        park = team_park_factors("ZZZ")
        self.assertEqual(park["hr"], 100)

    def test_coors_field_favors_home_runs(self):
        self.assertGreater(park_adjustment("COL", "home_runs"), 1.0)

    def test_park_adjustment_rejects_unmodeled_category(self):
        with self.assertRaises(ValueError):
            park_adjustment("COL", "not_a_real_category")

    def test_every_modeled_category_has_a_park_adjustment(self):
        for category in STAT_CATEGORIES:
            multiplier = park_adjustment("COL", category)
            self.assertTrue(0.75 <= multiplier <= 1.35)

    def test_altitude_note_reflects_real_static_data(self):
        note = altitude_and_foul_territory_note("COL")
        self.assertEqual(note["altitude_ft"], 5280)


class LineupModelTests(unittest.TestCase):
    def test_leadoff_gets_more_plate_appearances_than_ninth(self):
        self.assertGreater(expected_plate_appearances(1), expected_plate_appearances(9))

    def test_protection_multiplier_rises_with_next_hitter_obp(self):
        weak = lineup_protection_multiplier(3, 0.280)
        strong = lineup_protection_multiplier(3, 0.400)
        self.assertGreater(strong, weak)

    def test_platoon_usage_defaults_to_starts_every_game(self):
        self.assertEqual(platoon_usage_probability("L", "R", None), 1.0)

    def test_platoon_usage_reads_real_historical_split(self):
        result = platoon_usage_probability("L", "L", {"vs_left": 0.2, "vs_right": 0.95})
        self.assertEqual(result, 0.2)

    def test_stolen_base_environment_above_league_average(self):
        # A 2x-league-average raw rate is capped at the disclosed 1.6 ceiling.
        result = stolen_base_environment({"stolen_base_attempts_per_game": 1.1, "league_average_attempts_per_game": 0.55})
        self.assertEqual(result, 1.6)

    def test_lineup_context_combines_every_signal(self):
        context = lineup_context({"batting_order_position": 1, "hand": "L", "on_base_pct_next_hitter": 0.35, "opposing_pitcher_hand": "R"})
        self.assertIn("expected_plate_appearances", context)
        self.assertIn("stolen_base_environment", context)


class BullpenModelTests(unittest.TestCase):
    def test_no_recent_appearances_is_zero_fatigue(self):
        self.assertEqual(bullpen_fatigue_index([]), 0.0)

    def test_heavy_usage_raises_fatigue(self):
        fatigue = bullpen_fatigue_index([{"pitches_last_3_days": 50, "appearances_last_3_days": 3}])
        self.assertEqual(fatigue, 1.0)

    def test_era_rating_above_one_for_a_strong_bullpen(self):
        self.assertGreater(bullpen_era_rating(3.0), 1.0)

    def test_leverage_usage_above_league_average_share(self):
        self.assertGreater(leverage_usage_score(0.45), 1.0)

    def test_composite_strength_discounts_a_fatigued_bullpen(self):
        fresh = bullpen_strength({"recent_appearances": [], "bullpen_era": 3.0, "high_leverage_innings_pct": 0.4})
        tired = bullpen_strength({"recent_appearances": [{"pitches_last_3_days": 50, "appearances_last_3_days": 3}], "bullpen_era": 3.0, "high_leverage_innings_pct": 0.4})
        self.assertGreater(fresh["composite_strength"], tired["composite_strength"])


class DefenseModelTests(unittest.TestCase):
    def test_above_average_efficiency_raises_rating(self):
        self.assertGreater(defensive_efficiency_rating(0.72), 1.0)

    def test_strong_arm_raises_outfield_rating(self):
        self.assertGreater(outfield_arm_rating(60.0), 1.0)

    def test_composite_rating_is_bounded_and_has_every_component(self):
        result = composite_defense_rating({"balls_in_play_converted_pct": 0.70, "outfield_assists": 45.0, "catcher_framing_runs": 5.0, "infield_range_factor": 2.9})
        self.assertTrue(0.5 <= result["composite_rating"] <= 1.4)
        for key in ("defensive_efficiency_rating", "outfield_arm_rating", "catcher_framing_rating", "infield_range_rating"):
            self.assertIn(key, result)

    def test_missing_components_fall_back_to_neutral(self):
        result = composite_defense_rating({})
        self.assertAlmostEqual(result["composite_rating"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
