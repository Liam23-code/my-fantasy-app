"""Offline contract tests for real pace/usage trend-delta signals."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_trend_signals import player_usage_trend, team_pace_trend


class TeamPaceTrendTests(unittest.TestCase):
    def setUp(self):
        # team_pace_trend is TTL-cached (see modules/nba_trend_signals.py);
        # several tests reuse the same team/season args with different mocks.
        team_pace_trend.cache_clear()

    def test_accelerating_pace_detected(self):
        season_table = pd.DataFrame([{"TEAM_ID": 1610612743, "PACE": 98.0}])
        recent_table = pd.DataFrame([{"TEAM_ID": 1610612743, "PACE": 101.0}])
        with patch("modules.nba_trend_signals.find_team", return_value={"id": 1610612743, "abbreviation": "DEN"}), \
             patch("modules.nba_trend_signals.fetch_league_team_stats", side_effect=[season_table, recent_table]):
            result = team_pace_trend("Denver Nuggets", "2025-26")
        self.assertEqual(result["pace_delta"], 3.0)
        self.assertEqual(result["trend"], "accelerating")

    def test_stable_pace_within_threshold(self):
        season_table = pd.DataFrame([{"TEAM_ID": 1, "PACE": 100.0}])
        recent_table = pd.DataFrame([{"TEAM_ID": 1, "PACE": 100.5}])
        with patch("modules.nba_trend_signals.find_team", return_value={"id": 1, "abbreviation": "BOS"}), \
             patch("modules.nba_trend_signals.fetch_league_team_stats", side_effect=[season_table, recent_table]):
            result = team_pace_trend("Boston Celtics", "2025-26")
        self.assertEqual(result["trend"], "stable")

    def test_falls_back_to_season_pace_when_recent_table_missing_team(self):
        season_table = pd.DataFrame([{"TEAM_ID": 1, "PACE": 100.0}])
        recent_table = pd.DataFrame([{"TEAM_ID": 2, "PACE": 105.0}])  # different team -- no row for team 1
        with patch("modules.nba_trend_signals.find_team", return_value={"id": 1, "abbreviation": "BOS"}), \
             patch("modules.nba_trend_signals.fetch_league_team_stats", side_effect=[season_table, recent_table]):
            result = team_pace_trend("Boston Celtics", "2025-26")
        self.assertEqual(result["pace_delta"], 0.0)

    def test_raises_when_no_season_data_at_all(self):
        empty = pd.DataFrame(columns=["TEAM_ID", "PACE"])
        with patch("modules.nba_trend_signals.find_team", return_value={"id": 1, "abbreviation": "BOS"}), \
             patch("modules.nba_trend_signals.fetch_league_team_stats", return_value=empty):
            with self.assertRaises(ValueError):
                team_pace_trend("Boston Celtics", "2025-26")


class PlayerUsageTrendTests(unittest.TestCase):
    def setUp(self):
        player_usage_trend.cache_clear()

    def test_falling_usage_detected(self):
        season_frame = pd.DataFrame([{"PLAYER_ID": 1, "USG_PCT": 0.30}])
        recent_frame = pd.DataFrame([{"PLAYER_ID": 1, "USG_PCT": 0.26}])
        season_endpoint, recent_endpoint = MagicMock(), MagicMock()
        season_endpoint.get_data_frames.return_value = [season_frame]
        recent_endpoint.get_data_frames.return_value = [recent_frame]
        with patch("modules.projections._find_player", return_value={"id": 1, "full_name": "Test Player"}), \
             patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", side_effect=[season_endpoint, recent_endpoint]):
            result = player_usage_trend("Test Player", "2025-26")
        self.assertEqual(result["season_usage_pct"], 30.0)
        self.assertEqual(result["recent_10_usage_pct"], 26.0)
        self.assertEqual(result["trend"], "falling")

    def test_recent_endpoint_failure_falls_back_to_season_value(self):
        season_frame = pd.DataFrame([{"PLAYER_ID": 1, "USG_PCT": 0.30}])
        season_endpoint = MagicMock()
        season_endpoint.get_data_frames.return_value = [season_frame]
        with patch("modules.projections._find_player", return_value={"id": 1, "full_name": "Test Player"}), \
             patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", side_effect=[season_endpoint, RuntimeError("down")]):
            result = player_usage_trend("Test Player", "2025-26")
        self.assertEqual(result["trend"], "stable")
        self.assertEqual(result["usage_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
