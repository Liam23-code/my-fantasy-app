"""Offline (mocked) contract tests for the four new matchup-aware NBA modules and their aggregator.

Mirrors tests/test_cfb_engine.py's pattern of mocking the live-data call
site with hand-built response shapes matching the real, live-verified
columns discovered during this cycle's build (see matchup_engine.md) --
these tests check the real computation/aggregation logic, not live
connectivity.
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

_TEAM = {"id": 1610612743, "abbreviation": "DEN", "full_name": "Denver Nuggets"}
_OPPONENT = {"id": 1610612738, "abbreviation": "BOS", "full_name": "Boston Celtics"}
_PLAYER = {"id": 203999, "full_name": "Nikola Jokic"}


def _game_log_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"GAME_DATE": "Feb 25, 2026", "MATCHUP": "DEN vs. BOS", "PTS": 30, "REB": 12, "AST": 6, "MIN": 33},
            {"GAME_DATE": "Jan 10, 2026", "MATCHUP": "DEN @ LAL", "PTS": 25, "REB": 13, "AST": 11, "MIN": 35},
            {"GAME_DATE": "Dec 01, 2025", "MATCHUP": "DEN vs. LAL", "PTS": 28, "REB": 12, "AST": 9, "MIN": 34},
        ]
    )


class NbaMatchupHistoryTests(unittest.TestCase):
    @patch("modules.nba_matchup_history.find_team", return_value=_OPPONENT)
    @patch("modules.nba_matchup_history._find_player", return_value=_PLAYER)
    @patch("modules.nba_matchup_history._game_log_with_fuzzy_fallback")
    def test_filters_to_real_games_against_the_named_opponent(self, game_log, find_player, find_team):
        game_log.return_value = (_PLAYER, _game_log_frame(), [])
        from modules.nba_matchup_history import games_vs_opponent

        result = games_vs_opponent("Nikola Jokic", "BOS", "2025-26")
        self.assertEqual(result["games_sampled"], 1)
        self.assertEqual(result["games"][0]["pts"], 30.0)
        self.assertIn("pts", result["season_averages"])

    @patch("modules.nba_matchup_history.find_team", return_value=_OPPONENT)
    @patch("modules.nba_matchup_history._find_player", return_value=_PLAYER)
    @patch("modules.nba_matchup_history._game_log_with_fuzzy_fallback")
    def test_delta_vs_season_is_matchup_average_minus_season_average(self, game_log, find_player, find_team):
        game_log.return_value = (_PLAYER, _game_log_frame(), [])
        from modules.nba_matchup_history import games_vs_opponent

        result = games_vs_opponent("Nikola Jokic", "BOS", "2025-26")
        expected_season_avg = round((30 + 25 + 28) / 3, 2)
        self.assertEqual(result["season_averages"]["pts"], expected_season_avg)
        self.assertAlmostEqual(result["delta_vs_season"]["pts"], round(30.0 - expected_season_avg, 2), places=2)


class NbaDefenseModelTests(unittest.TestCase):
    def _rim_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"PLAYER_NAME": "Rudy Gobert", "PLAYER_LAST_TEAM_ABBREVIATION": "MIN", "FGA_LT_06": 500, "LT_06_PCT": 0.45},
                {"PLAYER_NAME": "Weak Defender", "PLAYER_LAST_TEAM_ABBREVIATION": "GSW", "FGA_LT_06": 300, "LT_06_PCT": 0.65},
            ]
        )

    def _perimeter_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"PLAYER_NAME": "Rudy Gobert", "PLAYER_LAST_TEAM_ABBREVIATION": "MIN", "FG3A": 100, "FG3_PCT": 0.34},
                {"PLAYER_NAME": "Weak Defender", "PLAYER_LAST_TEAM_ABBREVIATION": "GSW", "FG3A": 100, "FG3_PCT": 0.40},
            ]
        )

    @patch("modules.nba_defense_model.fetch_league_team_stats")
    @patch("modules.nba_defense_model.find_team")
    @patch("modules.nba_defense_model._fetch_pt_defend")
    def test_tougher_real_rim_defense_scores_a_higher_difficulty(self, fetch_defend, find_team, league_stats):
        find_team.side_effect = lambda name: {"id": 1, "abbreviation": name} if name == "MIN" else {"id": 2, "abbreviation": name}
        fetch_defend.side_effect = lambda season, category: self._rim_frame() if category == "Less Than 6Ft" else self._perimeter_frame()
        league_stats.return_value = pd.DataFrame([{"TEAM_ID": 1, "PACE": 100.0}, {"TEAM_ID": 2, "PACE": 100.0}])
        from modules.nba_defense_model import matchup_difficulty_score

        tough = matchup_difficulty_score("interior", "MIN", "2025-26")
        weak = matchup_difficulty_score("interior", "GSW", "2025-26")
        self.assertGreater(tough["difficulty_score"], weak["difficulty_score"])
        self.assertEqual(tough["basis"], "real_pt_defend_data")

    @patch("modules.nba_defense_model.fetch_league_team_stats", return_value=pd.DataFrame())
    @patch("modules.nba_defense_model.find_team", return_value={"id": 1, "abbreviation": "MIN"})
    @patch("modules.nba_defense_model._fetch_pt_defend", side_effect=Exception("provider down"))
    def test_fails_soft_to_neutral_difficulty_on_provider_error(self, fetch_defend, find_team, league_stats):
        from modules.nba_defense_model import matchup_difficulty_score

        result = matchup_difficulty_score("wing", "MIN", "2025-26")
        self.assertEqual(result["difficulty_score"], 50.0)
        self.assertEqual(result["basis"], "no_pt_defend_data")


class NbaInjuryImpactTests(unittest.TestCase):
    @patch("modules.nba_injury_impact._fetch_pt_defend")
    @patch("modules.nba_injury_impact.load_injury_data_from_file")
    def test_flags_a_real_meaningful_rim_defender_as_out(self, injuries, fetch_defend):
        injuries.return_value = [{"team": "MIN", "player": "Rudy Gobert", "status": "OUT"}]
        fetch_defend.return_value = pd.DataFrame(
            [{"PLAYER_NAME": "Rudy Gobert", "FGA_LT_06": 500, "LT_06_PCT": 0.45}, {"PLAYER_NAME": "Bench Guy", "FGA_LT_06": 5, "LT_06_PCT": 0.5}]
        )
        from modules.nba_injury_impact import opponent_injury_context

        result = opponent_injury_context("MIN", "2025-26")
        self.assertEqual(result["severity"], 1.0)
        self.assertIsNotNone(result["rim_defender_out"])
        self.assertEqual(result["rim_defender_out"]["player"], "Rudy Gobert")

    @patch("modules.nba_injury_impact._fetch_pt_defend")
    @patch("modules.nba_injury_impact.load_injury_data_from_file")
    def test_a_low_volume_absence_does_not_trigger_the_rim_defender_flag(self, injuries, fetch_defend):
        injuries.return_value = [{"team": "MIN", "player": "Bench Guy", "status": "OUT"}]
        fetch_defend.return_value = pd.DataFrame(
            [{"PLAYER_NAME": "Rudy Gobert", "FGA_LT_06": 500, "LT_06_PCT": 0.45}, {"PLAYER_NAME": "Bench Guy", "FGA_LT_06": 5, "LT_06_PCT": 0.5}]
        )
        from modules.nba_injury_impact import opponent_injury_context

        result = opponent_injury_context("MIN", "2025-26")
        self.assertIsNone(result["rim_defender_out"])

    @patch("modules.nba_injury_impact.load_injury_data_from_file", return_value=[])
    def test_empty_default_injury_file_is_a_normal_state(self, _injuries):
        from modules.nba_injury_impact import opponent_injury_context

        result = opponent_injury_context("MIN", "2025-26")
        self.assertEqual(result["absences"], [])
        self.assertEqual(result["severity"], 0.0)

    @patch("modules.nba_injury_impact._find_player")
    @patch("modules.nba_injury_impact._game_log_with_fuzzy_fallback")
    def test_teammate_absence_split_partitions_by_real_game_date(self, game_log, find_player):
        find_player.side_effect = lambda name: {"id": 1, "full_name": name}

        def _log(name, player, season):
            if name == "Nikola Jokic":
                frame = pd.DataFrame(
                    [
                        {"GAME_DATE": "Feb 25, 2026", "PTS": 24, "REB": 12, "AST": 6, "MIN": 33},
                        {"GAME_DATE": "Feb 26, 2026", "PTS": 35, "REB": 14, "AST": 12, "MIN": 40},
                        {"GAME_DATE": "Feb 27, 2026", "PTS": 33, "REB": 13, "AST": 11, "MIN": 38},
                        {"GAME_DATE": "Feb 28, 2026", "PTS": 30, "REB": 12, "AST": 10, "MIN": 36},
                    ]
                )
            else:
                frame = pd.DataFrame([{"GAME_DATE": "Feb 25, 2026", "PTS": 20, "REB": 4, "AST": 5, "MIN": 30}])
            return {"full_name": name}, frame, []

        game_log.side_effect = _log
        from modules.nba_injury_impact import teammate_absence_split

        result = teammate_absence_split("Nikola Jokic", "Jamal Murray", "2025-26", min_games=1)
        self.assertEqual(result["games_with_teammate"], 1)
        self.assertEqual(result["games_without_teammate"], 3)
        self.assertFalse(result["insufficient_sample"])
        self.assertIn("ast", result["delta_without_minus_with"])


class NbaLineupModelTests(unittest.TestCase):
    def _on_off_tables(self) -> list[pd.DataFrame]:
        overall = pd.DataFrame([{"GROUP_SET": "Overall", "TEAM_ID": 1, "PTS": 120.0}])
        on = pd.DataFrame(
            [{"VS_PLAYER_ID": 1, "VS_PLAYER_NAME": "Jokic, Nikola", "COURT_STATUS": "On", "NET_RATING": 10.8}]
        )
        off = pd.DataFrame(
            [{"VS_PLAYER_ID": 1, "VS_PLAYER_NAME": "Jokic, Nikola", "COURT_STATUS": "Off", "NET_RATING": -2.9}]
        )
        return [overall, on, off]

    @patch("modules.nba_lineup_model.fetch_on_off_splits")
    @patch("modules.nba_lineup_model.find_team", return_value=_TEAM)
    def test_skips_the_leading_team_overall_table_before_summarizing(self, find_team, fetch_splits):
        fetch_splits.return_value = self._on_off_tables()
        from modules.nba_lineup_model import team_on_off_impact

        result = team_on_off_impact("DEN", "2025-26")
        self.assertEqual(len(result["players"]), 1)
        self.assertEqual(result["players"][0]["player"], "Jokic, Nikola")
        self.assertAlmostEqual(result["players"][0]["on_off_swing"], 13.7, places=1)

    @patch("modules.nba_lineup_model.fetch_on_off_splits")
    @patch("modules.nba_lineup_model.find_team", return_value=_TEAM)
    def test_matches_a_first_last_name_against_the_real_last_comma_first_row(self, find_team, fetch_splits):
        fetch_splits.return_value = self._on_off_tables()
        from modules.nba_lineup_model import teammate_on_off_swing

        result = teammate_on_off_swing("DEN", "Nikola Jokic", "2025-26")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["on_off_swing"], 13.7, places=1)

    @patch("modules.nba_lineup_model.fetch_on_off_splits", return_value=[])
    @patch("modules.nba_lineup_model.find_team", return_value=_TEAM)
    def test_fails_soft_to_an_empty_list_when_fewer_than_three_tables_come_back(self, find_team, fetch_splits):
        from modules.nba_lineup_model import team_on_off_impact

        result = team_on_off_impact("DEN", "2025-26")
        self.assertEqual(result["players"], [])


class NbaMatchupEngineIntegrationTests(unittest.TestCase):
    def test_matchup_adjusted_evaluation_reuses_price_prop_comparison_directly(self):
        from modules.nba_matchup_engine import matchup_adjusted_evaluation

        comparison_row = {
            "player": "Nikola Jokic", "team": "DEN", "category": "points",
            "minutes_adjusted_projection": 27.0, "sportsbook_line": 25.5,
            "confidence_low": 23.0, "confidence_high": 31.0,
        }
        prop_odds = {"over_price": -110.0, "under_price": -110.0}
        context = {"context_multiplier": 1.05, "history_shift": 0.5}

        result = matchup_adjusted_evaluation(comparison_row, prop_odds, context)
        self.assertAlmostEqual(result["pre_matchup_projection"], 27.0)
        self.assertAlmostEqual(result["minutes_adjusted_projection"], 27.0 * 1.05 + 0.5, places=2)
        self.assertIn("recommended_priced_side", result)
        self.assertIn("risk_tier", result)


if __name__ == "__main__":
    unittest.main()
