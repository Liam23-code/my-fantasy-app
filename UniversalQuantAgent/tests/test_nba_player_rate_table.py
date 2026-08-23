"""Offline contract tests for the precomputed NBA player-rate table."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_player_rate_table import _fetch_advanced_player_stats, build_player_rate_table
from modules.nba_props_generator import _fetch_base_player_stats


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PLAYER_NAME": "Fast Pace Player", "TEAM_ABBREVIATION": "DEN", "GP": 40, "MIN": 1200.0,
             "PTS": 800.0, "REB": 200.0, "AST": 200.0, "FG3M": 80.0},
            {"PLAYER_ID": 2, "PLAYER_NAME": "Slow Pace Player", "TEAM_ABBREVIATION": "MIA", "GP": 40, "MIN": 1200.0,
             "PTS": 800.0, "REB": 200.0, "AST": 200.0, "FG3M": 80.0},
        ]
    )


def _advanced_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PLAYER_ID": 1, "GP": 40, "USG_PCT": 0.25, "PACE": 105.0},  # faster than league average
            {"PLAYER_ID": 2, "GP": 40, "USG_PCT": 0.25, "PACE": 95.0},  # slower than league average
        ]
    )


class PlayerRateTableTests(unittest.TestCase):
    def setUp(self):
        _fetch_base_player_stats.cache_clear()
        _fetch_advanced_player_stats.cache_clear()
        build_player_rate_table.cache_clear()

    def test_real_per_game_rates_computed_from_season_totals(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_base_frame()]
        adv_endpoint = MagicMock()
        adv_endpoint.get_data_frames.return_value = [_advanced_frame()]
        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", side_effect=[endpoint, adv_endpoint]):
            rows = build_player_rate_table("2025-26", pool_size=10)
        by_name = {row["player_name"]: row for row in rows}
        self.assertEqual(by_name["Fast Pace Player"]["points_per_game"], 20.0)  # 800/40
        self.assertEqual(by_name["Fast Pace Player"]["pra_per_game"], 30.0)  # (800+200+200)/40

    def test_pace_adjusted_scoring_normalizes_toward_league_average(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_base_frame()]
        adv_endpoint = MagicMock()
        adv_endpoint.get_data_frames.return_value = [_advanced_frame()]
        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", side_effect=[endpoint, adv_endpoint]):
            rows = build_player_rate_table("2025-26", pool_size=10)
        by_name = {row["player_name"]: row for row in rows}
        # Same raw points-per-game, but the fast-pace player's pace-adjusted
        # figure should be lower and the slow-pace player's higher, since
        # each is normalized toward the league-average pace (100.0 here).
        self.assertLess(by_name["Fast Pace Player"]["pace_adjusted_points_per_game"], by_name["Fast Pace Player"]["points_per_game"])
        self.assertGreater(by_name["Slow Pace Player"]["pace_adjusted_points_per_game"], by_name["Slow Pace Player"]["points_per_game"])

    def test_missing_advanced_row_leaves_usage_and_pace_none(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_base_frame()]
        adv_endpoint = MagicMock()
        adv_endpoint.get_data_frames.return_value = [pd.DataFrame(columns=["PLAYER_ID", "GP", "USG_PCT", "PACE"])]
        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", side_effect=[endpoint, adv_endpoint]):
            rows = build_player_rate_table("2025-26", pool_size=10)
        self.assertTrue(all(row["usage_pct"] is None for row in rows))
        self.assertTrue(all(row["pace_adjusted_points_per_game"] is None for row in rows))

    def test_every_row_discloses_real_basis(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_base_frame()]
        adv_endpoint = MagicMock()
        adv_endpoint.get_data_frames.return_value = [_advanced_frame()]
        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", side_effect=[endpoint, adv_endpoint]):
            rows = build_player_rate_table("2025-26", pool_size=10)
        for row in rows:
            self.assertIn("real per-game rate", row["basis"])


if __name__ == "__main__":
    unittest.main()
