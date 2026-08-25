"""Tests for the MLB prop model and its four loaders (props, odds, injuries, lineups)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.mlb_common import normalize_category, normalize_team_name
from modules.mlb_props_loader import load_props_from_file, load_props_from_user_upload, unified_props
from modules.mlb_odds_loader import GameOddsLoadError, index_by_matchup, load_default_game_odds, load_uploaded_game_odds
from modules.mlb_injuries_loader import load_injury_data_from_file, load_injury_data_from_user_upload
from modules.mlb_lineups_loader import get_team_lineup, load_lineups_from_file, load_lineups_from_user_upload, unified_lineups
from modules.mlb_prop_model import evaluate_prop, evaluate_props, over_under_probability


class MlbCommonNormalizationTests(unittest.TestCase):
    def test_every_team_normalizes_to_a_real_code(self):
        self.assertEqual(normalize_team_name("New York Yankees"), "NYY")
        self.assertEqual(normalize_team_name("Colorado Rockies"), "COL")

    def test_ambiguous_sox_alias_does_not_crash_and_resolves_to_one_team(self):
        # Whichever of Red Sox/White Sox is inserted first keeps the bare
        # "SOX" alias -- both are still reachable by their full name.
        self.assertIn(normalize_team_name("SOX"), {"BOS", "CWS"})
        self.assertEqual(normalize_team_name("Boston Red Sox"), "BOS")
        self.assertEqual(normalize_team_name("Chicago White Sox"), "CWS")

    def test_category_normalization_covers_all_seven_categories(self):
        self.assertEqual(normalize_category("HR"), "home_runs")
        self.assertEqual(normalize_category("RBIs"), "rbi")
        self.assertEqual(normalize_category("Total Bases"), "total_bases")
        self.assertEqual(normalize_category("Ks"), "strikeouts")
        self.assertEqual(normalize_category("not a real stat"), None)


class MlbPropsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_props_from_file("Z:/nope.json"), [])

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_props_from_file(), [])

    def test_upload_normalizes_team_and_category(self):
        text = json.dumps([{"player_name": "Test Batter", "team": "Yankees", "category": "HR", "line": 0.5}])
        rows = load_props_from_user_upload(text, file_format="json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "NYY")
        self.assertEqual(rows[0]["category"], "home_runs")

    def test_upload_skips_row_missing_line(self):
        text = json.dumps([{"player_name": "Test Batter", "category": "hits"}])
        self.assertEqual(load_props_from_user_upload(text, file_format="json"), [])

    def test_unified_props_upload_overrides_default_by_player_and_category(self):
        default_payload = {"props": [{"player_name": "Test Batter", "team": "NYY", "category": "hits", "line": 1.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mlb_props.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps([{"player_name": "Test Batter", "category": "hits", "line": 2.5}])
            merged = unified_props(default_path=path, uploaded=uploaded, uploaded_format="json")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["line"], 2.5)


class MlbOddsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_games_not_error(self):
        self.assertEqual(load_default_game_odds("Z:/nope.json"), {"games": {}})

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_default_game_odds(), {"games": {}})

    def test_upload_and_matchup_index(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "Yankees", "away_team": "Boston Red Sox", "moneyline_home": -150, "moneyline_away": 130}]), file_format="json")
        indexed = index_by_matchup(result)
        self.assertIn(("NYY", "BOS"), indexed)

    def test_upload_skips_row_with_teams_but_no_market_data(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "NYY", "away_team": "BOS"}]), file_format="json")
        self.assertEqual(result["games"], {})

    def test_malformed_json_upload_raises_loud_error(self):
        with self.assertRaises(GameOddsLoadError):
            load_uploaded_game_odds("{not valid", file_format="json")


class MlbInjuriesLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_injury_data_from_file("Z:/nope.json"), [])

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_injury_data_from_file(), [])

    def test_upload_reads_team_from_flat_record(self):
        payload = [{"player": "Test Pitcher", "team": "Yankees", "status": "OUT"}]
        rows = load_injury_data_from_user_upload(json.dumps(payload))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "NYY")

    def test_upload_skips_record_with_no_player_name(self):
        payload = [{"team": "NYY", "status": "OUT"}]
        self.assertEqual(load_injury_data_from_user_upload(json.dumps(payload)), [])


class MlbLineupsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_lineups_from_file("Z:/nope.json"), [])

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_lineups_from_file(), [])

    def test_upload_normalizes_team_pitcher_and_batting_order(self):
        payload = [
            {
                "team": "Yankees",
                "opponent": "Boston Red Sox",
                "starting_pitcher": {"player_name": "Test Pitcher", "hand": "R"},
                "batting_order": [{"position": 1, "player_name": "Test Leadoff", "hand": "L"}, {"position": 2, "player_name": "Test Two Hole", "hand": "R"}],
            }
        ]
        rows = load_lineups_from_user_upload(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "NYY")
        self.assertEqual(rows[0]["opponent"], "BOS")
        self.assertEqual(rows[0]["starting_pitcher"]["hand"], "R")
        self.assertEqual(len(rows[0]["batting_order"]), 2)
        self.assertEqual(rows[0]["batting_order"][0]["position"], 1)

    def test_row_with_neither_pitcher_nor_batting_order_is_skipped(self):
        rows = load_lineups_from_user_upload([{"team": "NYY"}])
        self.assertEqual(rows, [])

    def test_unified_lineups_upload_overrides_by_team(self):
        default_payload = {"lineups": [{"team": "NYY", "starting_pitcher": {"player_name": "Old Pitcher", "hand": "R"}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mlb_lineups.json"
            path.write_text(json.dumps(default_payload))
            uploaded = [{"team": "NYY", "starting_pitcher": {"player_name": "New Pitcher", "hand": "L"}}]
            merged = unified_lineups(default_path=path, uploaded=uploaded)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["starting_pitcher"]["player_name"], "New Pitcher")

    def test_get_team_lineup_looks_up_by_normalized_team(self):
        lineups = [{"team": "NYY", "starting_pitcher": {"player_name": "Test Pitcher", "hand": "R"}}]
        self.assertIsNotNone(get_team_lineup("Yankees", lineups))
        self.assertIsNone(get_team_lineup("Dodgers", lineups))


class MlbPropModelTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        distribution = over_under_probability(1.5, 1.2)
        self.assertAlmostEqual(distribution["probability_over"] + distribution["probability_under"], 1.0, places=4)

    def test_evaluate_prop_returns_a_complete_row(self):
        row = evaluate_prop({"player_name": "Test Batter", "team": "NYY", "category": "hits", "line": 1.5, "over_price": -110.0, "under_price": -110.0})
        self.assertIn(row["risk_tier"], {"low", "medium", "high"})
        self.assertIn(row["recommended_side"], {"over", "under"})

    def test_matchup_multiplier_shifts_the_priced_mean(self):
        neutral = evaluate_prop({"player_name": "A", "team": "NYY", "category": "hits", "line": 1.5, "over_price": -110.0, "under_price": -110.0})
        boosted = evaluate_prop({"player_name": "A", "team": "NYY", "category": "hits", "line": 1.5, "over_price": -110.0, "under_price": -110.0}, matchup_multiplier=1.2)
        self.assertGreater(boosted["line"], neutral["line"])

    def test_evaluate_props_applies_per_player_matchup_multipliers(self):
        props = [{"player_name": "Boosted Batter", "team": "NYY", "category": "hits", "line": 1.0, "over_price": -110.0, "under_price": -110.0}]
        rows = evaluate_props(props, matchup_multipliers={"Boosted Batter": 1.5})
        self.assertEqual(rows[0]["line"], 1.5)

    def test_evaluate_props_sorts_by_absolute_edge_descending(self):
        props = [
            {"player_name": "Low Edge", "team": "NYY", "category": "hits", "line": 1.0, "over_price": -110.0, "under_price": -110.0},
            {"player_name": "High Edge", "team": "NYY", "category": "home_runs", "line": 0.2, "over_price": 400.0, "under_price": -600.0},
        ]
        rows = evaluate_props(props)
        self.assertGreaterEqual(abs(rows[0]["recommended_edge"]), abs(rows[1]["recommended_edge"]))


if __name__ == "__main__":
    unittest.main()
