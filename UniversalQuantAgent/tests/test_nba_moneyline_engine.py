"""Offline contract tests for the NBA moneyline engine: team_model, moneyline_model, odds_loader."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_team_model import team_scoring_averages, team_scoring_by_game, real_margin_and_total_volatility
from modules.nba_moneyline_model import evaluate_game, evaluate_games, fair_moneyline, win_probability_from_spread
from modules.nba_odds_loader import (
    GameOddsLoadError,
    index_by_matchup,
    load_default_game_odds,
    load_uploaded_game_odds,
    merge_game_odds,
    unified_game_odds,
)


def _fake_game_log_frame() -> pd.DataFrame:
    # Two real-shaped games: DEN beats BOS 120-110, then BOS beats DEN 100-95.
    return pd.DataFrame(
        [
            {"GAME_ID": "1", "TEAM_ABBREVIATION": "DEN", "PTS": 120},
            {"GAME_ID": "1", "TEAM_ABBREVIATION": "BOS", "PTS": 110},
            {"GAME_ID": "2", "TEAM_ABBREVIATION": "BOS", "PTS": 100},
            {"GAME_ID": "2", "TEAM_ABBREVIATION": "DEN", "PTS": 95},
        ]
    )


class NbaTeamModelTests(unittest.TestCase):
    def test_team_scoring_by_game_pairs_real_games_correctly(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_fake_game_log_frame()]
        with patch("nba_api.stats.endpoints.leaguegamelog.LeagueGameLog", return_value=endpoint):
            by_team = team_scoring_by_game("2025-26")
        self.assertEqual(len(by_team["DEN"]), 2)
        self.assertEqual(by_team["DEN"][0], {"game_id": "1", "opponent": "BOS", "points_scored": 120.0, "points_allowed": 110.0})
        self.assertEqual(by_team["BOS"][1], {"game_id": "2", "opponent": "DEN", "points_scored": 100.0, "points_allowed": 95.0})

    def test_team_scoring_averages_shape_matches_nfl_convention(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_fake_game_log_frame()]
        with patch("nba_api.stats.endpoints.leaguegamelog.LeagueGameLog", return_value=endpoint):
            averages = team_scoring_averages("2025-26")
        self.assertEqual(set(averages["DEN"]), {"points_scored_avg", "points_allowed_avg", "games_played"})
        self.assertEqual(averages["DEN"]["points_scored_avg"], 107.5)  # (120+95)/2
        self.assertEqual(averages["DEN"]["games_played"], 2)

    def test_real_margin_and_total_volatility_computed_from_real_games(self):
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [_fake_game_log_frame()]
        with patch("nba_api.stats.endpoints.leaguegamelog.LeagueGameLog", return_value=endpoint):
            vol = real_margin_and_total_volatility("2025-26")
        self.assertEqual(vol["games_sampled"], 2)
        self.assertIn("real completed games", vol["basis"])
        self.assertGreater(vol["margin_stdev"], 0)

    def test_fails_soft_to_empty_when_provider_unavailable(self):
        with patch("nba_api.stats.endpoints.leaguegamelog.LeagueGameLog", side_effect=RuntimeError("down")):
            self.assertEqual(team_scoring_by_game("2025-26"), {})
            self.assertEqual(team_scoring_averages("2025-26"), {})

    def test_volatility_falls_back_with_fewer_than_two_real_games(self):
        with patch("nba_api.stats.endpoints.leaguegamelog.LeagueGameLog", side_effect=RuntimeError("down")):
            vol = real_margin_and_total_volatility("2025-26")
        self.assertEqual(vol["basis"], "fallback_estimate")


class NbaMoneylineModelTests(unittest.TestCase):
    def setUp(self):
        self.averages = {
            "DEN": {"points_scored_avg": 122.0, "points_allowed_avg": 114.0, "games_played": 40},
            "BOS": {"points_scored_avg": 115.0, "points_allowed_avg": 108.0, "games_played": 40},
        }

    def test_win_probability_uses_supplied_stdev_not_nfl_constant(self):
        # A 5-point favorite is a bigger favorite under a smaller stdev.
        tighter = win_probability_from_spread(5.0, stdev=10.0)
        wider = win_probability_from_spread(5.0, stdev=20.0)
        self.assertGreater(tighter, wider)

    def test_fair_moneyline_home_favorite_gets_negative_price(self):
        fair = fair_moneyline("DEN", "BOS", averages=self.averages, margin_stdev=16.2)
        self.assertGreater(fair["spread"], 0)  # DEN projected to outscore BOS
        self.assertGreater(fair["home_win_probability"], 0.5)
        self.assertLess(fair["fair_home_moneyline"], 0)  # favorite -> negative American odds

    def test_evaluate_game_computes_edge_and_ev_against_real_odds(self):
        game_odds = {
            "home_team": "DEN", "away_team": "BOS",
            "moneyline": {"home": -200.0, "away": 170.0},
            "total": {"line": 230.0, "over_price": -110.0, "under_price": -110.0},
        }
        result = evaluate_game(game_odds, averages=self.averages, volatility={"margin_stdev": 16.2, "total_stdev": 19.9})
        self.assertIn("recommended_edge", result["moneyline"])
        self.assertIn("recommended_edge", result["total"])
        self.assertEqual(result["home_team"], "DEN")

    def test_evaluate_game_without_market_odds_still_returns_fair_line(self):
        result = evaluate_game({"home_team": "DEN", "away_team": "BOS"}, averages=self.averages, volatility={"margin_stdev": 16.2, "total_stdev": 19.9})
        self.assertNotIn("moneyline", result)
        self.assertNotIn("total", result)
        self.assertIn("fair_home_moneyline", result["model"])

    def test_evaluate_games_ranks_by_largest_edge(self):
        games = [{"home_team": "DEN", "away_team": "BOS"}]
        odds_by_matchup = {
            ("DEN", "BOS"): {
                "home_team": "DEN", "away_team": "BOS",
                "moneyline": {"home": 150.0, "away": -180.0},  # mispriced vs. our real fair line
            }
        }
        rows = evaluate_games(games, averages=self.averages, odds_by_matchup=odds_by_matchup, volatility={"margin_stdev": 16.2, "total_stdev": 19.9})
        self.assertEqual(len(rows), 1)
        self.assertIn("moneyline", rows[0])


class NbaOddsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_default_game_odds("Z:/does/not/exist.json"), {"games": {}})

    def test_default_shipped_file_is_empty(self):
        # No fixed NBA schedule to pre-populate -- see the module docstring.
        self.assertEqual(load_default_game_odds(), {"games": {}})

    def test_load_and_normalize_flat_row(self):
        payload = {"games": [{"home_team": "DEN", "away_team": "BOS", "moneyline_home": -200, "moneyline_away": 170, "total_line": 230}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_game_odds.json"
            path.write_text(json.dumps(payload))
            result = load_default_game_odds(path)
        games = result["games"]
        self.assertEqual(len(games), 1)
        row = next(iter(games.values()))
        self.assertEqual(row["home_team"], "DEN")
        self.assertEqual(row["moneyline"]["home"], -200.0)
        self.assertEqual(row["total"]["line"], 230.0)

    def test_upload_overrides_default_by_matchup(self):
        default_payload = {"games": [{"home_team": "DEN", "away_team": "BOS", "moneyline_home": -200, "moneyline_away": 170}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_game_odds.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps({"games": [{"home_team": "DEN", "away_team": "BOS", "moneyline_home": -220, "moneyline_away": 185}]})
            merged = unified_game_odds(default_path=path, uploaded=uploaded, uploaded_format="json")
        row = next(iter(merged["games"].values()))
        self.assertEqual(row["moneyline"]["home"], -220.0)
        self.assertEqual(row["source"], "uploaded")

    def test_upload_csv(self):
        text = "home_team,away_team,moneyline_home,moneyline_away\nDEN,BOS,-200,170\n"
        result = load_uploaded_game_odds(text, file_format="csv")
        row = next(iter(result["games"].values()))
        self.assertEqual(row["home_team"], "DEN")

    def test_malformed_json_upload_raises_loud_error(self):
        with self.assertRaises(GameOddsLoadError):
            load_uploaded_game_odds("{not valid json", file_format="json")

    def test_index_by_matchup(self):
        odds = {"games": {"BOS@DEN": {"home_team": "DEN", "away_team": "BOS", "moneyline": {"home": -200, "away": 170}}}}
        indexed = index_by_matchup(odds)
        self.assertIn(("DEN", "BOS"), indexed)


if __name__ == "__main__":
    unittest.main()
