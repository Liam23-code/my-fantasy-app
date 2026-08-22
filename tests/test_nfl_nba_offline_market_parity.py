"""Cross-package integration tests: NFL and NBA offline market pipelines.

fantasy_engine (NFL props/moneylines: betting/odds_loader.py,
betting/odds_generator.py) and UniversalQuantAgent (NBA props/schedule:
modules/nba_props_loader.py, modules/nba_props_generator.py,
modules/nba_schedule.py) are two independent projects, built at different
times, that both had to satisfy the same rule: sportsbook odds and injury
data come ONLY from our own file or a user upload, real per-player rates
are computed from real stats (never fabricated), and any live-data
ingestion (schedules) must fail closed instead of crashing a page.

These tests don't re-test either project's own unit-level behavior (that's
what fantasy_engine/tests and UniversalQuantAgent/tests are for) -- they
assert the two independently-built pipelines actually satisfy the *same*
contract, so a future change that quietly weakens one sport's guarantees
relative to the other gets caught here.

Run from the repo root: pytest tests/
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NBA_ROOT = ROOT / "UniversalQuantAgent"
if str(NBA_ROOT) not in sys.path:
    sys.path.insert(0, str(NBA_ROOT))

# fantasy_engine's `betting`/`fantasy` packages are installed editable into
# this venv (see fantasy_engine/pyproject.toml) and importable from any cwd.
from betting.odds_generator import generate_default_games as nfl_generate_default_games
from betting.odds_loader import load_default_odds, merge_odds

from modules.nba_props_generator import DEFAULT_PROPS_PATH as NBA_DEFAULT_PROPS_PATH
from modules.nba_props_loader import load_props_from_file, unified_props
from modules.nba_schedule import fetch_todays_games

_BANNED_LIBRARIES = {"requests", "httpx", "aiohttp", "urllib3", "selenium", "playwright", "bs4"}

# Files whose entire purpose is delivering "real odds/schedule data, never a
# sportsbook, never fabricated" for one sport or the other.
_MARKET_PIPELINE_FILES = (
    ROOT / "fantasy_engine" / "betting" / "odds_loader.py",
    ROOT / "fantasy_engine" / "betting" / "odds_generator.py",
    NBA_ROOT / "modules" / "nba_props_loader.py",
    NBA_ROOT / "modules" / "nba_props_generator.py",
    NBA_ROOT / "modules" / "nba_schedule.py",
)


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class MarketPipelineImportParityTests(unittest.TestCase):
    """Neither sport's market pipeline may import a scraping/HTTP library directly."""

    def test_no_market_pipeline_file_imports_a_banned_library(self):
        offenders = {}
        for path in _MARKET_PIPELINE_FILES:
            banned = _imported_top_level_modules(path) & _BANNED_LIBRARIES
            if banned:
                offenders[str(path)] = banned
        self.assertEqual(offenders, {}, f"banned network/scraping imports found: {offenders}")


class DefaultFilePropsParityTests(unittest.TestCase):
    """Both sports must treat 'no default file configured' as empty, not an error."""

    def test_missing_default_file_is_empty_not_an_error_for_both_sports(self):
        missing = Path("Z:/does/not/exist.json")
        self.assertEqual(load_default_odds(missing), {"games": {}, "player_props": {}})
        self.assertEqual(load_props_from_file(missing), [])

    def test_shipped_default_files_exist_and_are_labeled_real_not_fabricated(self):
        """Every row NFL/NBA ship by default must disclose its real-data basis."""
        nfl_odds = load_default_odds()
        self.assertGreater(len(nfl_odds["games"]) + len(nfl_odds["player_props"]), 0)
        for row in list(nfl_odds["games"].values()) + list(nfl_odds["player_props"].values()):
            self.assertIn("basis", row)
            self.assertIn("real", row["basis"].lower())

        self.assertTrue(NBA_DEFAULT_PROPS_PATH.is_file())
        nba_props = load_props_from_file()
        self.assertGreater(len(nba_props), 0)
        for row in nba_props:
            self.assertIn("basis", row)
            self.assertIn("real", row["basis"].lower())


class UploadOverrideParityTests(unittest.TestCase):
    """An uploaded/overriding entry must replace only the matching default entry, for both sports."""

    def test_nfl_merge_odds_upload_overrides_by_key_leaves_rest_untouched(self):
        default = {
            "games": {"g1": {"game_id": "g1", "home_team": "KC", "away_team": "BUF", "moneyline_home": -150}},
            "player_props": {"p1:points": {"player_id": "p1", "market": "points", "line": 20.5}},
        }
        uploaded = {"games": {"g1": {"game_id": "g1", "home_team": "KC", "away_team": "BUF", "moneyline_home": -200}}}
        merged = merge_odds(default, uploaded)
        self.assertEqual(merged["games"]["g1"]["moneyline_home"], -200)
        self.assertEqual(merged["player_props"]["p1:points"]["line"], 20.5)  # untouched

    def test_nba_unified_props_upload_overrides_by_key_leaves_rest_untouched(self):
        default_payload = {
            "props": [
                {"player_name": "Player A", "team": "DEN", "category": "points", "line": 20.5, "sportsbook": "default"},
                {"player_name": "Player B", "team": "BOS", "category": "assists", "line": 5.5, "sportsbook": "default"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nba_props.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps([{"player_name": "Player A", "category": "points", "line": 24.5}])
            merged = unified_props(default_path=path, uploaded=uploaded, uploaded_format="json")

        by_key = {(row["player_name"], row["category"]): row for row in merged}
        self.assertEqual(by_key[("Player A", "points")]["line"], 24.5)
        self.assertEqual(by_key[("Player B", "assists")]["line"], 5.5)  # untouched


class LiveScheduleIngestionParityTests(unittest.TestCase):
    """Live schedule ingestion is legal for both sports, but must fail closed, never raise."""

    def test_nfl_generate_default_games_returns_empty_list_not_exception_without_nflreadpy(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "nflreadpy":
                raise ImportError("simulated: nflreadpy not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = nfl_generate_default_games(2025)
        self.assertEqual(result, [])

    def test_nba_fetch_todays_games_returns_empty_list_not_exception_when_provider_fails(self):
        with patch("modules.nba_schedule._from_live_scoreboard", side_effect=RuntimeError("simulated: nba_api unreachable")):
            result = fetch_todays_games()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
