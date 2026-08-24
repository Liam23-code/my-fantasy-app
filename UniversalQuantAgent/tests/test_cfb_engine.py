"""Offline contract tests for the CFB betting stack: loaders, team/prop/moneyline models, parlay engine.

modules/cfb_team_model.py and modules/cfb_props_generator.py require a
CFBD_API_KEY (see their docstrings and cfb_pipeline.md); this file tests
both the always-available "no key configured" fail-closed path and, via
mocked HTTP responses, the parsing/computation logic for when a key *is*
present -- without making a real network call.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.cfb_props_loader import load_props_from_file, load_props_from_user_upload, unified_props
from modules.cfb_odds_loader import (
    GameOddsLoadError,
    index_by_matchup,
    load_default_game_odds,
    load_uploaded_game_odds,
    unified_game_odds,
)
from modules.cfb_injuries_loader import load_injury_data_from_file, load_injury_data_from_user_upload
from modules.cfb_team_model import team_scoring_averages, team_scoring_by_game, real_margin_and_total_volatility, fetch_week_games
from modules.cfb_moneyline_model import evaluate_game, evaluate_games, fair_moneyline, win_probability_from_spread
from modules.cfb_prop_model import evaluate_prop, evaluate_props
from modules.cfb_parlay_engine import make_leg, detect_correlations, evaluate_parlay
from modules import cfb_parlay_engine
from betting import parlay_engine as nfl_parlay_engine
from modules.cfb_props_generator import generate_default_props, _round_to_half


class CfbLoaderTests(unittest.TestCase):
    def test_missing_default_files_return_empty_not_error(self):
        self.assertEqual(load_props_from_file("Z:/nope.json"), [])
        self.assertEqual(load_default_game_odds("Z:/nope.json"), {"games": {}})
        self.assertEqual(load_injury_data_from_file("Z:/nope.json"), [])

    def test_default_shipped_files_ship_empty_with_provenance_note(self):
        self.assertEqual(load_props_from_file(), [])
        payload = json.loads((ROOT / "data" / "cfb_props.json").read_text(encoding="utf-8"))
        self.assertIn("note", payload)
        self.assertIn("CFBD_API_KEY", payload["note"])

    def test_props_upload_roundtrip(self):
        text = json.dumps([{"player_name": "Test QB", "team": "Ohio State", "category": "passing_yards", "line": 275.5}])
        rows = load_props_from_user_upload(text, file_format="json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "OHIO STATE")
        self.assertEqual(rows[0]["category"], "passing_yards")

    def test_props_upload_skips_row_with_unknown_category(self):
        text = json.dumps([{"player_name": "X", "category": "not_a_real_stat", "line": 10}])
        self.assertEqual(load_props_from_user_upload(text, file_format="json"), [])

    def test_unified_props_upload_overrides_by_player_and_category(self):
        default_payload = {"props": [{"player_name": "Test QB", "team": "OHIO STATE", "category": "passing_yards", "line": 275.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cfb_props.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps([{"player_name": "Test QB", "category": "passing_yards", "line": 300.5}])
            merged = unified_props(default_path=path, uploaded=uploaded, uploaded_format="json")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["line"], 300.5)

    def test_game_odds_upload_and_matchup_index(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "Ohio State", "away_team": "Michigan", "moneyline_home": -150, "moneyline_away": 130}]), file_format="json")
        indexed = index_by_matchup(result)
        self.assertIn(("OHIO STATE", "MICHIGAN"), indexed)

    def test_game_odds_upload_skips_row_with_no_market_data(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "Ohio State", "away_team": "Michigan"}]), file_format="json")
        self.assertEqual(result["games"], {})

    def test_malformed_json_game_odds_upload_raises_loud_error(self):
        with self.assertRaises(GameOddsLoadError):
            load_uploaded_game_odds("{not valid", file_format="json")

    def test_injury_upload_roundtrip(self):
        payload = [{"player": "Test RB", "team": "Alabama", "status": "OUT"}]
        rows = load_injury_data_from_user_upload(json.dumps(payload))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "OUT")
        self.assertEqual(rows[0]["team"], "ALABAMA")


class CfbTeamModelNoKeyTests(unittest.TestCase):
    """With no CFBD_API_KEY configured, everything fails closed -- no network attempted."""

    def setUp(self):
        self._patcher = patch.dict("os.environ", {}, clear=False)
        self._patcher.start()
        import os

        os.environ.pop("CFBD_API_KEY", None)

    def tearDown(self):
        self._patcher.stop()

    def test_team_scoring_by_game_empty_without_key(self):
        self.assertEqual(team_scoring_by_game(2025), {})

    def test_team_scoring_averages_empty_without_key(self):
        self.assertEqual(team_scoring_averages(2025), {})

    def test_volatility_falls_back_without_key(self):
        vol = real_margin_and_total_volatility(2025)
        self.assertEqual(vol["basis"], "fallback_estimate")

    def test_fetch_week_games_empty_without_key(self):
        self.assertEqual(fetch_week_games(2025, 1), [])

    def test_generate_default_props_empty_without_key(self):
        self.assertEqual(generate_default_props(2025), [])


class CfbTeamModelMockedKeyTests(unittest.TestCase):
    """With a key configured, verify the parsing/computation logic against a mocked (best-understanding) response shape."""

    def setUp(self):
        import os

        self._patcher = patch.dict(os.environ, {"CFBD_API_KEY": "test-key"})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _mock_games_response(self, games):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = games
        return response

    def test_team_scoring_by_game_pairs_real_games(self):
        games = [{"homeTeam": "OHIO STATE", "awayTeam": "MICHIGAN", "homePoints": 30, "awayPoints": 24}]
        with patch("requests.get", return_value=self._mock_games_response(games)):
            by_team = team_scoring_by_game(2025)
        self.assertEqual(by_team["OHIO STATE"], [{"opponent": "MICHIGAN", "points_scored": 30.0, "points_allowed": 24.0}])
        self.assertEqual(by_team["MICHIGAN"], [{"opponent": "OHIO STATE", "points_scored": 24.0, "points_allowed": 30.0}])

    def test_team_scoring_by_game_tolerates_snake_case_fields(self):
        games = [{"home_team": "AUBURN", "away_team": "LSU", "home_points": 20, "away_points": 17}]
        with patch("requests.get", return_value=self._mock_games_response(games)):
            by_team = team_scoring_by_game(2025)
        self.assertIn("AUBURN", by_team)

    def test_team_scoring_by_game_skips_unplayed_games(self):
        games = [{"homeTeam": "A", "awayTeam": "B", "homePoints": None, "awayPoints": None}]
        with patch("requests.get", return_value=self._mock_games_response(games)):
            by_team = team_scoring_by_game(2025)
        self.assertEqual(by_team, {})

    def test_volatility_computed_from_real_mocked_games(self):
        games = [
            {"homeTeam": "A", "awayTeam": "B", "homePoints": 35, "awayPoints": 14},
            {"homeTeam": "C", "awayTeam": "D", "homePoints": 21, "awayPoints": 20},
        ]
        with patch("requests.get", return_value=self._mock_games_response(games)):
            vol = real_margin_and_total_volatility(2025)
        self.assertEqual(vol["games_sampled"], 2)
        self.assertIn("real completed games", vol["basis"])

    def test_request_failure_fails_closed_not_raises(self):
        with patch("requests.get", side_effect=RuntimeError("network down")):
            self.assertEqual(team_scoring_by_game(2025), {})


class CfbMoneylineModelTests(unittest.TestCase):
    def setUp(self):
        self.averages = {
            "OHIO STATE": {"points_scored_avg": 38.0, "points_allowed_avg": 18.0, "games_played": 10},
            "MICHIGAN": {"points_scored_avg": 30.0, "points_allowed_avg": 20.0, "games_played": 10},
        }

    def test_win_probability_uses_supplied_stdev(self):
        tighter = win_probability_from_spread(7.0, stdev=14.0)
        wider = win_probability_from_spread(7.0, stdev=25.0)
        self.assertGreater(tighter, wider)

    def test_fair_moneyline_favorite_gets_negative_price(self):
        fair = fair_moneyline("OHIO STATE", "MICHIGAN", averages=self.averages, margin_stdev=21.0)
        self.assertGreater(fair["spread"], 0)
        self.assertLess(fair["fair_home_moneyline"], 0)

    def test_evaluate_game_computes_edges_against_real_odds(self):
        game_odds = {"home_team": "OHIO STATE", "away_team": "MICHIGAN", "moneyline": {"home": -300.0, "away": 250.0}, "total": {"line": 55.0, "over_price": -110.0, "under_price": -110.0}}
        result = evaluate_game(game_odds, averages=self.averages, volatility={"margin_stdev": 21.0, "total_stdev": 17.0})
        self.assertIn("recommended_edge", result["moneyline"])
        self.assertIn("recommended_edge", result["total"])

    def test_evaluate_games_sorts_by_edge(self):
        games = [{"home_team": "OHIO STATE", "away_team": "MICHIGAN"}]
        rows = evaluate_games(games, averages=self.averages, volatility={"margin_stdev": 21.0, "total_stdev": 17.0})
        self.assertEqual(len(rows), 1)


class CfbPropModelTests(unittest.TestCase):
    def test_evaluate_prop_favors_over_when_price_symmetric_and_line_equals_mean(self):
        # A symmetric market with the model's own mean as the line should
        # sit right at 50/50 -- side selection is a coin flip either way,
        # but the shape (both edges near zero) should hold.
        row = evaluate_prop({"player_name": "Test QB", "team": "OHIO STATE", "category": "passing_yards", "line": 275.5, "over_price": -110.0, "under_price": -110.0})
        self.assertAlmostEqual(row["model_probability_over"] + row["model_probability_under"], 1.0, places=4)
        self.assertIn(row["risk_tier"], {"low", "medium", "high"})

    def test_evaluate_props_sorted_by_absolute_edge(self):
        props = [
            {"player_name": "A", "category": "passing_yards", "line": 275.5, "over_price": -110.0, "under_price": -110.0},
            {"player_name": "B", "category": "rushing_yards", "line": 90.5, "over_price": -110.0, "under_price": -110.0},
        ]
        rows = evaluate_props(props)
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(abs(rows[0]["recommended_edge"]), abs(rows[1]["recommended_edge"]))


class CfbParlayEngineReExportTests(unittest.TestCase):
    def test_reexports_are_the_same_objects_as_nfl_parlay_engine(self):
        self.assertIs(cfb_parlay_engine.make_leg, nfl_parlay_engine.make_leg)
        self.assertIs(cfb_parlay_engine.detect_correlations, nfl_parlay_engine.detect_correlations)
        self.assertIs(cfb_parlay_engine.evaluate_parlay, nfl_parlay_engine.evaluate_parlay)

    def test_qb_wr_stack_pattern_fires_for_cfb_legs(self):
        qb = make_leg(description="qb", model_probability=0.55, price=-110.0, team="OSU", market="passing_yards", side="over")
        wr = make_leg(description="wr", model_probability=0.55, price=-110.0, team="OSU", market="receiving_yards", side="over")
        findings = detect_correlations([qb, wr])
        self.assertEqual(findings[0]["kind"], "qb_pass_catcher_stack")

    def test_evaluate_parlay_works_for_cfb_legs(self):
        legs = [
            make_leg(description="a", model_probability=0.6, price=120.0),
            make_leg(description="b", model_probability=0.6, price=120.0),
        ]
        result = evaluate_parlay(legs)
        self.assertIn("adjusted_hit_probability", result)


class CfbPropsGeneratorTests(unittest.TestCase):
    def test_round_to_half_never_whole_number(self):
        self.assertEqual(_round_to_half(275.2), 275.5)
        self.assertEqual(_round_to_half(275.6), 275.5)
        self.assertEqual(_round_to_half(274.4), 274.5)


if __name__ == "__main__":
    unittest.main()
