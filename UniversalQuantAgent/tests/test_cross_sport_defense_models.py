"""Offline (mocked) contract tests for the NFL/CFB/CBB defensive-difficulty extensions of the matchup-aware engine.

See tests/test_nba_matchup_engine.py for the NBA-first versions these
mirror -- same "shared contract, separate code" pattern (see
matchup_engine.md): each sport computes its own real percentile from its
own already-verified real data source, no shared implementation.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FANTASY_ENGINE_ROOT = ROOT.parent / "fantasy_engine"
if str(FANTASY_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(FANTASY_ENGINE_ROOT))


class NflDefenseModelTests(unittest.TestCase):
    def _team_stats_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"team": "KC", "opponent_team": "BAL", "passing_yards": 180, "rushing_yards": 140},
                {"team": "CIN", "opponent_team": "BAL", "passing_yards": 190, "rushing_yards": 150},
                {"team": "BAL", "opponent_team": "KC", "passing_yards": 320, "rushing_yards": 90},
                {"team": "BAL", "opponent_team": "CIN", "passing_yards": 300, "rushing_yards": 100},
            ]
        )

    @patch("betting.defense_model._fetch_team_stats")
    def test_real_pass_defense_percentile_uses_yards_allowed(self, fetch_stats):
        fetch_stats.return_value = self._team_stats_frame()
        from betting.defense_model import matchup_difficulty_score

        result = matchup_difficulty_score("passer", "BAL", 2025)
        self.assertEqual(result["basis"], "real_yards_allowed")
        # BAL allows fewer real passing yards (180, 190) than KC/CIN allow
        # (320, 300) -- BAL should rank as the tougher pass defense.
        self.assertGreater(result["pass_defense_percentile"], 50.0)

    @patch("betting.defense_model._fetch_team_stats", side_effect=Exception("provider down"))
    def test_fails_soft_to_neutral_on_provider_error(self, fetch_stats):
        from betting.defense_model import matchup_difficulty_score

        result = matchup_difficulty_score("rusher", "BAL", 2025)
        self.assertEqual(result["difficulty_score"], 50.0)
        self.assertEqual(result["basis"], "no_real_data")


class CfbDefenseModelTests(unittest.TestCase):
    @patch("modules.cfb_defense_model.team_scoring_averages")
    def test_fewer_real_points_allowed_ranks_tougher(self, team_averages):
        team_averages.return_value = {
            "OSU": {"points_allowed_avg": 14.0, "games_played": 10},
            "MICH": {"points_allowed_avg": 28.0, "games_played": 10},
        }
        from modules.cfb_defense_model import matchup_difficulty_score

        result = matchup_difficulty_score("OSU", 2025)
        self.assertEqual(result["basis"], "real_points_allowed")
        self.assertEqual(result["difficulty_score"], 100.0)

    @patch("modules.cfb_defense_model.team_scoring_averages", return_value={})
    def test_no_cfbd_api_key_fails_soft_to_neutral(self, team_averages):
        from modules.cfb_defense_model import matchup_difficulty_score

        result = matchup_difficulty_score("OSU", 2025)
        self.assertEqual(result["difficulty_score"], 50.0)
        self.assertEqual(result["basis"], "no_real_data")


class CbbDefenseModelTests(unittest.TestCase):
    def test_takes_already_fetched_averages_rather_than_fetching_again(self):
        from modules.cbb_defense_model import matchup_difficulty_score

        averages = {
            "DUKE": {"points_allowed_avg": 60.0, "games_played": 20},
            "UNC": {"points_allowed_avg": 75.0, "games_played": 20},
        }
        result = matchup_difficulty_score("DUKE", averages)
        self.assertEqual(result["basis"], "real_points_allowed")
        self.assertEqual(result["difficulty_score"], 100.0)

    def test_unknown_team_fails_soft_to_neutral(self):
        from modules.cbb_defense_model import matchup_difficulty_score

        result = matchup_difficulty_score("NOT_A_REAL_TEAM", {"DUKE": {"points_allowed_avg": 60.0, "games_played": 20}})
        self.assertEqual(result["difficulty_score"], 50.0)
        self.assertEqual(result["basis"], "no_real_data")


if __name__ == "__main__":
    unittest.main()
