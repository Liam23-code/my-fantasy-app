"""Tests for the MLB season-average model: rolling averages, reliability, stabilization, park-neutral rates."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mlb_season_model import (
    STABILIZATION_GAMES,
    park_neutral_rate,
    project_season_baseline,
    reliability_score,
    rolling_average,
    stabilize,
)
from modules.mlb_common import STAT_CATEGORIES


class RollingAverageTests(unittest.TestCase):
    def test_empty_values_return_zero(self):
        self.assertEqual(rolling_average([]), 0.0)

    def test_averages_the_most_recent_window(self):
        values = [0.0] * 20 + [2.0] * 5
        self.assertEqual(rolling_average(values, window=5), 2.0)

    def test_uses_all_values_when_fewer_than_window(self):
        self.assertEqual(rolling_average([1.0, 2.0, 3.0], window=15), 2.0)


class ReliabilityScoreTests(unittest.TestCase):
    def test_zero_games_is_minimum_reliability(self):
        self.assertEqual(reliability_score(0, category="hits"), 0.05)

    def test_reliability_increases_with_games_played(self):
        low = reliability_score(5, category="hits")
        high = reliability_score(200, category="hits")
        self.assertLess(low, high)
        self.assertLessEqual(high, 0.98)

    def test_slow_stabilizing_category_is_less_reliable_at_the_same_sample_size(self):
        # home_runs (55-game stabilization point) is slower to stabilize
        # than hits (30-game point) -- the same real sample size should
        # leave it less reliable.
        hits_reliability = reliability_score(30, category="hits")
        hr_reliability = reliability_score(30, category="home_runs")
        self.assertGreater(hits_reliability, hr_reliability)

    def test_every_modeled_category_has_a_stabilization_point(self):
        for category in STAT_CATEGORIES:
            self.assertIn(category, STABILIZATION_GAMES)


class StabilizeTests(unittest.TestCase):
    def test_low_sample_leans_on_league_mean(self):
        blended = stabilize(raw_rate=3.0, league_mean=1.0, games_played=1, category="home_runs")
        self.assertLess(blended, 2.0)  # far from the raw 3.0, close to the 1.0 prior

    def test_high_sample_leans_on_observed_rate(self):
        blended = stabilize(raw_rate=1.5, league_mean=1.0, games_played=500, category="hits")
        self.assertAlmostEqual(blended, 1.5, delta=0.05)


class ParkNeutralRateTests(unittest.TestCase):
    def test_neutral_park_factor_is_unchanged(self):
        self.assertEqual(park_neutral_rate(1.2, 100.0), 1.2)

    def test_hitter_friendly_park_is_scaled_down(self):
        self.assertLess(park_neutral_rate(1.2, 116.0), 1.2)

    def test_pitcher_friendly_park_is_scaled_up(self):
        self.assertGreater(park_neutral_rate(1.2, 90.0), 1.2)


class ProjectSeasonBaselineTests(unittest.TestCase):
    def test_no_game_log_returns_zeroed_baseline_for_every_category(self):
        baseline = project_season_baseline({"game_log": []})
        for category in STAT_CATEGORIES:
            self.assertIn(category, baseline)
            self.assertEqual(baseline[category]["games_played"], 0)
            self.assertEqual(baseline[category]["reliability"], 0.05)

    def test_real_game_log_computes_a_raw_mean_per_category(self):
        game_log = [{"hits": 1.0, "home_runs": 0.0, "rbi": 1.0, "total_bases": 2.0, "strikeouts": 0.0, "walks": 1.0, "stolen_bases": 0.0} for _ in range(10)]
        baseline = project_season_baseline({"game_log": game_log})
        self.assertEqual(baseline["hits"]["raw_mean"], 1.0)
        self.assertEqual(baseline["total_bases"]["raw_mean"], 2.0)
        self.assertEqual(baseline["hits"]["games_played"], 10)

    def test_park_neutral_mean_reflects_home_park_factor(self):
        game_log = [{"home_runs": 1.0} for _ in range(60)]
        neutral = project_season_baseline({"game_log": game_log}, home_park_factor=100.0)
        hitter_friendly = project_season_baseline({"game_log": game_log}, home_park_factor=116.0)
        self.assertLess(hitter_friendly["home_runs"]["park_neutral_mean"], neutral["home_runs"]["park_neutral_mean"])


if __name__ == "__main__":
    unittest.main()
