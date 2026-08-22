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
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nba_props_loader import (
    load_props_from_file,
    load_props_from_user_upload,
    unified_props,
)
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

    def test_default_shipped_file_is_empty(self):
        # data/nba_props.json ships empty by default -- this app never
        # fetches prop lines from a sportsbook.
        self.assertEqual(load_props_from_file(), [])

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


if __name__ == "__main__":
    unittest.main()
