"""Data-contract hardening tests, spanning both projects.

Complements tests/test_nfl_nba_offline_market_parity.py (which asserts the
two sports satisfy the *same* contract) with deeper, per-loader coverage of
the contract itself (see offline_data_contract.md):

1. Every loader fails closed on a malformed *field* within an otherwise
   well-formed row, not just on a missing file.
2. Every default file that ships real data discloses its provenance.
3. Every upload path replaces only the matching default entry, by key.
4. No banned network/scraping import has crept into any market-pipeline
   file, across the full, current file list (not just the files that
   existed when the parity test was first written).
5. No live sportsbook-odds ingestion exists anywhere in either project --
   specifically, nba_api's dedicated live-odds client is never imported.

Run from the repo root: pytest tests/
"""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NBA_ROOT = ROOT / "UniversalQuantAgent"
if str(NBA_ROOT) not in sys.path:
    sys.path.insert(0, str(NBA_ROOT))

from betting.odds_loader import load_uploaded_odds
from modules.nba_odds_loader import load_uploaded_game_odds
from modules.nba_props_loader import load_props_from_user_upload
from modules.injury_parser import load_injury_data_from_user_upload

_BANNED_LIBRARIES = {"requests", "httpx", "aiohttp", "urllib3", "selenium", "playwright", "bs4"}

# Every .py file whose job is loading/generating/serving real odds, props,
# schedule, or injury data for either sport -- the full current set, not a
# frozen snapshot. If a new file joins this pipeline, add it here.
_ALL_MARKET_PIPELINE_FILES = (
    ROOT / "fantasy_engine" / "betting" / "odds_loader.py",
    ROOT / "fantasy_engine" / "betting" / "odds_generator.py",
    ROOT / "fantasy_engine" / "betting" / "odds_math.py",
    ROOT / "fantasy_engine" / "betting" / "team_model.py",
    ROOT / "fantasy_engine" / "betting" / "moneyline_model.py",
    ROOT / "fantasy_engine" / "betting" / "prop_model.py",
    ROOT / "fantasy_engine" / "betting" / "parlay_engine.py",
    ROOT / "fantasy_engine" / "betting" / "cache_utils.py",
    ROOT / "fantasy_engine" / "betting" / "parallel_utils.py",
    NBA_ROOT / "modules" / "nba_props_loader.py",
    NBA_ROOT / "modules" / "nba_props_generator.py",
    NBA_ROOT / "modules" / "nba_schedule.py",
    NBA_ROOT / "modules" / "nba_odds_loader.py",
    NBA_ROOT / "modules" / "nba_team_model.py",
    NBA_ROOT / "modules" / "nba_moneyline_model.py",
    NBA_ROOT / "modules" / "nba_prop_model.py",
    NBA_ROOT / "modules" / "nba_parlay_engine.py",
    NBA_ROOT / "modules" / "nba_trend_signals.py",
    NBA_ROOT / "modules" / "nba_player_rate_table.py",
    NBA_ROOT / "modules" / "unified_parlay_engine.py",
    NBA_ROOT / "modules" / "async_upload.py",
    NBA_ROOT / "modules" / "parallel_utils.py",
    NBA_ROOT / "modules" / "injury_parser.py",
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


class FailsClosedOnMalformedFieldsTests(unittest.TestCase):
    """A malformed *field* in an otherwise-parseable row is skipped, not an error and not a partial/garbage row."""

    def test_nba_props_upload_skips_row_missing_line(self):
        text = json.dumps([{"player_name": "Test Player", "category": "points"}])
        self.assertEqual(load_props_from_user_upload(text, file_format="json"), [])

    def test_nba_props_upload_skips_row_missing_player_name(self):
        text = json.dumps([{"category": "points", "line": 20.5}])
        self.assertEqual(load_props_from_user_upload(text, file_format="json"), [])

    def test_nba_props_upload_skips_row_with_unrecognized_category(self):
        text = json.dumps([{"player_name": "Test Player", "category": "not_a_real_stat", "line": 20.5}])
        self.assertEqual(load_props_from_user_upload(text, file_format="json"), [])

    def test_nba_props_upload_skips_row_with_non_numeric_line(self):
        text = json.dumps([{"player_name": "Test Player", "category": "points", "line": "not-a-number"}])
        self.assertEqual(load_props_from_user_upload(text, file_format="json"), [])

    def test_nba_game_odds_upload_skips_row_missing_teams(self):
        result = load_uploaded_game_odds(json.dumps([{"moneyline_home": -110, "moneyline_away": 105}]), file_format="json")
        self.assertEqual(result["games"], {})

    def test_nba_game_odds_upload_skips_row_with_teams_but_no_market_data(self):
        # Real team names, but no moneyline/spread/total at all -- nothing to evaluate.
        result = load_uploaded_game_odds(json.dumps([{"home_team": "DEN", "away_team": "BOS"}]), file_format="json")
        self.assertEqual(result["games"], {})

    def test_nfl_odds_upload_skips_prop_row_missing_line(self):
        payload = {"player_props": [{"player_id": "p1", "market": "points"}]}
        result = load_uploaded_odds(json.dumps(payload), file_format="json")
        self.assertEqual(result["player_props"], {})

    def test_nfl_odds_upload_skips_game_row_with_no_market_data(self):
        payload = {"games": [{"home_team": "KC", "away_team": "BUF"}]}  # no moneyline/spread/total
        result = load_uploaded_odds(json.dumps(payload), file_format="json")
        self.assertEqual(result["games"], {})

    def test_injury_upload_skips_record_with_no_player_name(self):
        payload = [{"team": "DEN", "status": "OUT"}]  # no player identified
        self.assertEqual(load_injury_data_from_user_upload(json.dumps(payload)), [])


class ProvenanceDisclosureTests(unittest.TestCase):
    """Every real row a default file ships discloses why it's real; an empty-by-design file says why it's empty."""

    def test_nba_default_props_file_every_row_has_basis(self):
        from modules.nba_props_loader import load_props_from_file

        rows = load_props_from_file()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("basis", row)
            self.assertIn("real", row["basis"].lower())

    def test_nfl_default_odds_file_every_row_has_basis(self):
        from betting.odds_loader import load_default_odds

        odds = load_default_odds()
        rows = list(odds["games"].values()) + list(odds["player_props"].values())
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("basis", row)
            self.assertIn("real", row["basis"].lower())

    def test_nba_game_odds_file_ships_empty_and_explains_why(self):
        # No fixed default here by design (see nba_odds_loader.py's module
        # docstring) -- the *file itself* must still say why, for a human
        # reading data/nba_game_odds.json without the source code handy.
        payload = json.loads((NBA_ROOT / "data" / "nba_game_odds.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["games"], [])
        self.assertIn("note", payload)
        self.assertTrue(payload["note"])

    def test_nba_injuries_file_ships_empty_and_explains_why(self):
        payload = json.loads((NBA_ROOT / "data" / "injuries.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["injuries"], [])
        self.assertIn("note", payload)
        self.assertTrue(payload["note"])


class UploadOverrideKeyingTests(unittest.TestCase):
    """An upload never adds a stray, un-keyed entry -- every normalized row round-trips through the same key scheme the default file uses."""

    def test_nba_props_upload_key_matches_player_and_category(self):
        rows = load_props_from_user_upload(json.dumps([{"player_name": "Test Player", "category": "points", "line": 20.5}]), file_format="json")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["player_name"], rows[0]["category"]), ("Test Player", "points"))

    def test_nba_game_odds_upload_key_is_matchup_pair(self):
        from modules.nba_odds_loader import index_by_matchup

        result = load_uploaded_game_odds(json.dumps([{"home_team": "DEN", "away_team": "BOS", "moneyline_home": -110, "moneyline_away": 100}]), file_format="json")
        indexed = index_by_matchup(result)
        self.assertIn(("DEN", "BOS"), indexed)

    def test_duplicate_player_category_rows_in_one_upload_last_one_wins_deterministically(self):
        payload = [
            {"player_name": "Test Player", "category": "points", "line": 20.5},
            {"player_name": "Test Player", "category": "points", "line": 25.5},
        ]
        rows = load_props_from_user_upload(json.dumps(payload), file_format="json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["line"], 25.5)


class ComprehensiveBannedImportSweepTests(unittest.TestCase):
    def test_no_market_pipeline_file_imports_a_banned_library(self):
        offenders = {}
        for path in _ALL_MARKET_PIPELINE_FILES:
            self.assertTrue(path.is_file(), f"expected file not found: {path}")
            banned = _imported_top_level_modules(path) & _BANNED_LIBRARIES
            if banned:
                offenders[str(path)] = banned
        self.assertEqual(offenders, {}, f"banned network/scraping imports found: {offenders}")


class NoLiveOddsIngestionTests(unittest.TestCase):
    """nba_api ships a dedicated live sportsbook-odds client; it must never be imported anywhere in either project."""

    def test_nba_api_live_odds_endpoint_never_imported(self):
        offenders = []
        for root in (NBA_ROOT, ROOT / "fantasy_engine"):
            for path in root.rglob("*.py"):
                if any(part in {".venv", "__pycache__"} for part in path.parts):
                    continue
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and "nba_api.live.nba.endpoints" in node.module:
                        if any(alias.name == "odds" for alias in node.names):
                            offenders.append(str(path))
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "nba_api.live.nba.endpoints.odds":
                                offenders.append(str(path))
        self.assertEqual(offenders, [], f"live sportsbook-odds client imported in: {offenders}")

    def test_no_source_file_references_a_named_sportsbook_odds_domain(self):
        # A blunter, string-level check for the sportsbooks this project
        # will never fetch from -- catches a stray URL/comment that an
        # import-based AST scan wouldn't (e.g. a hardcoded API base URL).
        banned_terms = ("draftkings.com", "fanduel.com", "betmgm.com", "caesars.com", "espnbet.com")
        offenders = []
        for root in (NBA_ROOT, ROOT / "fantasy_engine"):
            for path in root.rglob("*.py"):
                if any(part in {".venv", "__pycache__"} for part in path.parts):
                    continue
                if "scraper_disabled" in path.name or path.name.startswith("test_"):
                    continue  # the disabled stubs' docstrings and tests discuss these names by design
                try:
                    text = path.read_text(encoding="utf-8").lower()
                except UnicodeDecodeError:
                    continue
                if any(term in text for term in banned_terms):
                    offenders.append(str(path))
        self.assertEqual(offenders, [], f"sportsbook domain referenced in: {offenders}")


if __name__ == "__main__":
    unittest.main()
