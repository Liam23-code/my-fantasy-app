"""Tests for the NHL prop model and its three loaders (props, odds, injuries)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.nhl_common import normalize_category, normalize_team_name
from modules.nhl_props_loader import load_props_from_file, load_props_from_user_upload, unified_props
from modules.nhl_odds_loader import GameOddsLoadError, index_by_matchup, load_default_game_odds, load_uploaded_game_odds
from modules.nhl_injuries_loader import load_injury_data_from_file, load_injury_data_from_user_upload
from modules.nhl_prop_model import evaluate_prop, evaluate_props, over_under_probability


class NhlCommonNormalizationTests(unittest.TestCase):
    def test_every_team_normalizes_to_a_real_code(self):
        self.assertEqual(normalize_team_name("Toronto Maple Leafs"), "TOR")
        self.assertEqual(normalize_team_name("Utah Mammoth"), "UTA")

    def test_category_normalization_covers_all_four_categories(self):
        self.assertEqual(normalize_category("SOG"), "shots")
        self.assertEqual(normalize_category("Goalie Saves"), "saves")
        self.assertEqual(normalize_category("not a real stat"), None)


class NhlPropsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_props_from_file("Z:/nope.json"), [])

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_props_from_file(), [])

    def test_upload_normalizes_team_and_category(self):
        text = json.dumps([{"player_name": "Test Winger", "team": "Toronto Maple Leafs", "category": "SOG", "line": 2.5}])
        rows = load_props_from_user_upload(text, file_format="json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "TOR")
        self.assertEqual(rows[0]["category"], "shots")

    def test_unified_props_upload_overrides_default_by_player_and_category(self):
        default_payload = {"props": [{"player_name": "Test Winger", "team": "TOR", "category": "goals", "line": 0.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nhl_props.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps([{"player_name": "Test Winger", "category": "goals", "line": 0.75}])
            merged = unified_props(default_path=path, uploaded=uploaded, uploaded_format="json")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["line"], 0.75)


class NhlOddsLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_games_not_error(self):
        self.assertEqual(load_default_game_odds("Z:/nope.json"), {"games": {}})

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_default_game_odds(), {"games": {}})

    def test_upload_and_matchup_index(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "Toronto Maple Leafs", "away_team": "Montreal Canadiens", "moneyline_home": -150, "moneyline_away": 130}]), file_format="json")
        indexed = index_by_matchup(result)
        self.assertIn(("TOR", "MTL"), indexed)

    def test_malformed_json_upload_raises_loud_error(self):
        with self.assertRaises(GameOddsLoadError):
            load_uploaded_game_odds("{not valid", file_format="json")


class NhlInjuriesLoaderTests(unittest.TestCase):
    def test_missing_default_file_returns_empty_not_error(self):
        self.assertEqual(load_injury_data_from_file("Z:/nope.json"), [])

    def test_default_shipped_file_ships_empty_by_design(self):
        self.assertEqual(load_injury_data_from_file(), [])

    def test_upload_reads_team_from_flat_record(self):
        payload = [{"player": "Test Forward", "team": "Toronto Maple Leafs", "status": "OUT"}]
        rows = load_injury_data_from_user_upload(json.dumps(payload))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "TOR")


class NhlPropModelTests(unittest.TestCase):
    def test_shots_probabilities_sum_to_one(self):
        distribution = over_under_probability(2.5, 2.5, category="shots")
        self.assertAlmostEqual(distribution["probability_over"] + distribution["probability_under"], 1.0, places=4)
        self.assertEqual(distribution["family"], "gaussian")

    def test_goals_uses_poisson_family(self):
        distribution = over_under_probability(0.5, 0.5, category="goals")
        self.assertEqual(distribution["family"], "poisson")

    def test_evaluate_prop_returns_a_complete_row(self):
        row = evaluate_prop({"player_name": "Test Winger", "team": "TOR", "category": "shots", "line": 2.5, "over_price": -110.0, "under_price": -110.0})
        self.assertIn(row["risk_tier"], {"low", "medium", "high"})
        self.assertIn(row["recommended_side"], {"over", "under"})

    def test_evaluate_props_sorts_by_absolute_edge_descending(self):
        props = [
            {"player_name": "Low Edge", "team": "TOR", "category": "shots", "line": 2.5, "over_price": -110.0, "under_price": -110.0},
            {"player_name": "High Edge", "team": "TOR", "category": "goals", "line": 0.5, "over_price": 400.0, "under_price": -600.0},
        ]
        rows = evaluate_props(props)
        self.assertGreaterEqual(abs(rows[0]["recommended_edge"]), abs(rows[1]["recommended_edge"]))


if __name__ == "__main__":
    unittest.main()
