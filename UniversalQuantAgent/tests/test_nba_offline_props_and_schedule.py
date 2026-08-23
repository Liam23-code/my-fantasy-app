"""Offline contract tests for the NBA props/schedule replacements.

These cover the two modules that unblocked modules.daily_slate,
modules.props, and modules.recommendations after
modules.sportsbook_scraper_disabled was hard-disabled: modules.nba_props_loader
(default file + user upload, never a sportsbook fetch) and
modules.nba_schedule (live, non-odds schedule data).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_props_loader import (
    load_props_from_file,
    load_props_from_user_upload,
    unified_props,
)
from modules.nba_props_generator import _round_to_half, generate_default_props
from modules.nba_schedule import fetch_todays_games


class NbaPropsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_props_from_file("Z:/does/not/exist.json"), [])

    def test_load_props_from_file_flat_schema(self):
        payload = {
            "props": [
                {"player_name": "Nikola Jokic", "team": "DEN", "category": "points", "line": 25.5, "sportsbook": "default"}
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_props.json"
            path.write_text(json.dumps(payload))
            result = load_props_from_file(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["player_name"], "Nikola Jokic")
        self.assertEqual(result[0]["team"], "DEN")
        self.assertEqual(result[0]["category"], "points")
        self.assertEqual(result[0]["line"], 25.5)

    def test_load_props_from_file_accepts_category_alias_keys(self):
        # A flat row using "market" instead of "category" (a common upstream
        # export shape) must still normalize correctly.
        payload = {"props": [{"player": "Jayson Tatum", "market": "rebounds", "value": 8.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_props.json"
            path.write_text(json.dumps(payload))
            result = load_props_from_file(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "rebounds")
        self.assertEqual(result[0]["line"], 8.5)

    def test_default_shipped_file_loads_real_generated_rows(self):
        # data/nba_props.json is seeded by modules.nba_props_generator from
        # real NBA per-game rates -- never from a sportsbook -- and every
        # row must carry that provenance.
        rows = load_props_from_file()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn(row["category"], {"points", "rebounds", "assists", "PRA", "3PM"})
            self.assertGreater(row["line"], 0)
            self.assertIn("basis", row)
            self.assertIn("real per-game rate", row["basis"])

    def test_upload_json_list(self):
        text = json.dumps([{"player_name": "Luka Doncic", "category": "assists", "line": 9.5}])
        result = load_props_from_user_upload(text, file_format="json")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "assists")

    def test_upload_csv(self):
        text = "player_name,team,category,line\nStephen Curry,GSW,3PM,4.5\n"
        result = load_props_from_user_upload(text, file_format="csv")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "3PM")
        self.assertEqual(result[0]["team"], "GSW")

    def test_upload_malformed_json_returns_empty_not_error(self):
        self.assertEqual(load_props_from_user_upload("{not valid json", file_format="json"), [])

    def test_unified_props_upload_overrides_default_by_player_and_category(self):
        default_payload = {
            "props": [{"player_name": "Nikola Jokic", "team": "DEN", "category": "points", "line": 25.5, "sportsbook": "default"}]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_props.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps([{"player_name": "Nikola Jokic", "category": "points", "line": 27.0, "sportsbook": "my_book"}])
            merged = unified_props(default_path=path, uploaded=uploaded, uploaded_format="json")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["line"], 27.0)
        self.assertEqual(merged[0]["sportsbook"], "my_book")

    def test_unified_props_without_upload_returns_default_only(self):
        default_payload = {"props": [{"player_name": "Anthony Edwards", "category": "points", "line": 27.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_props.json"
            path.write_text(json.dumps(default_payload))
            result = unified_props(default_path=path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["player_name"], "Anthony Edwards")


class NbaScheduleTests(unittest.TestCase):
    def setUp(self):
        # fetch_todays_games is TTL-cached (see modules/nba_schedule.py); each
        # test needs a clean cache or it may see a previous test's mocked result.
        fetch_todays_games.cache_clear()

    def test_fetch_todays_games_returns_empty_list_when_provider_unavailable(self):
        with patch("modules.nba_schedule._from_live_scoreboard", side_effect=RuntimeError("network unavailable")):
            self.assertEqual(fetch_todays_games(), [])

    def test_fetch_todays_games_normalizes_live_scoreboard_payload(self):
        fake_games = [
            {
                "homeTeam": {"teamTricode": "DEN"},
                "awayTeam": {"teamTricode": "BOS"},
                "gameTimeUTC": "2026-01-01T00:00:00Z",
            }
        ]
        with patch("modules.nba_schedule._from_live_scoreboard", return_value=[
            {"home_team": "DEN", "away_team": "BOS", "start_time": "2026-01-01T00:00:00Z"}
        ]):
            result = fetch_todays_games()
        self.assertEqual(result, [{"home_team": "DEN", "away_team": "BOS", "start_time": "2026-01-01T00:00:00Z"}])

    def test_fetch_todays_games_for_past_date_uses_stats_scoreboard(self):
        with patch("modules.nba_schedule._from_stats_scoreboard", return_value=[{"home_team": "LAL", "away_team": "GSW", "start_time": ""}]) as mocked:
            result = fetch_todays_games(date(2025, 1, 1))
        mocked.assert_called_once_with(date(2025, 1, 1))
        self.assertEqual(result[0]["home_team"], "LAL")

    def test_no_scraping_or_odds_libraries_imported(self):
        import ast

        source = (ROOT / "modules" / "nba_schedule.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        banned = imported & {"requests", "bs4", "selenium", "playwright", "sportsbook_scraper_disabled"}
        self.assertEqual(banned, set())


class NbaPropsGeneratorTests(unittest.TestCase):
    def setUp(self):
        # _fetch_base_player_stats is TTL-cached (see modules/nba_props_generator.py);
        # each test needs a clean cache or it may see a previous test's mocked frame.
        from modules.nba_props_generator import _fetch_base_player_stats

        _fetch_base_player_stats.cache_clear()

    def test_round_to_half_never_returns_a_whole_number(self):
        for value, expected in ((18.2, 18.5), (18.6, 18.5), (19.1, 19.5), (0.4, 0.5), (0.0, 0.5)):
            self.assertEqual(_round_to_half(value), expected)

    def test_generate_default_props_computes_real_per_game_rates(self):
        frame = pd.DataFrame(
            [
                {"PLAYER_ID": 1, "PLAYER_NAME": "Test Star", "TEAM_ABBREVIATION": "DEN", "GP": 20, "MIN": 700.0,
                 "PTS": 500.0, "REB": 200.0, "AST": 100.0, "FG3M": 40.0},
                {"PLAYER_ID": 2, "PLAYER_NAME": "Bench Guy", "TEAM_ABBREVIATION": "DEN", "GP": 20, "MIN": 100.0,
                 "PTS": 40.0, "REB": 20.0, "AST": 10.0, "FG3M": 2.0},
            ]
        )
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [frame]
        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", return_value=endpoint):
            rows = generate_default_props("2025-26", pool_size=10)

        by_key = {(row["player_name"], row["category"]): row for row in rows}
        self.assertEqual(by_key[("Test Star", "points")]["line"], _round_to_half(500.0 / 20))
        self.assertEqual(by_key[("Test Star", "points")]["team"], "DEN")
        self.assertIn("2025-26 real per-game rate (20 games)", by_key[("Test Star", "points")]["basis"])
        # Bench Guy's 100 total minutes over 20 games is 5.0 min/game --
        # below the rotation-minutes floor, so no lines at all.
        self.assertNotIn(("Bench Guy", "points"), by_key)

    def test_generate_default_props_dedupes_traded_player_to_combined_row(self):
        frame = pd.DataFrame(
            [
                {"PLAYER_ID": 1, "PLAYER_NAME": "Traded Player", "TEAM_ABBREVIATION": "TOT", "GP": 40, "MIN": 1200.0,
                 "PTS": 800.0, "REB": 200.0, "AST": 150.0, "FG3M": 60.0},
                {"PLAYER_ID": 1, "PLAYER_NAME": "Traded Player", "TEAM_ABBREVIATION": "DEN", "GP": 15, "MIN": 450.0,
                 "PTS": 300.0, "REB": 75.0, "AST": 55.0, "FG3M": 22.0},
                {"PLAYER_ID": 1, "PLAYER_NAME": "Traded Player", "TEAM_ABBREVIATION": "BOS", "GP": 25, "MIN": 750.0,
                 "PTS": 500.0, "REB": 125.0, "AST": 95.0, "FG3M": 38.0},
            ]
        )
        endpoint = MagicMock()
        endpoint.get_data_frames.return_value = [frame]
        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", return_value=endpoint):
            rows = generate_default_props("2025-26", pool_size=10)

        points_rows = [row for row in rows if row["category"] == "points"]
        self.assertEqual(len(points_rows), 1)  # one combined line, not one per team stint
        self.assertEqual(points_rows[0]["line"], _round_to_half(800.0 / 40))
        self.assertEqual(points_rows[0]["team"], "")  # TOT is not a real team; consumers fall back to live lookup


if __name__ == "__main__":
    unittest.main()
