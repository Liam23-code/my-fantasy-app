"""Offline contract tests for the CBB betting stack: loaders, team/prop/moneyline models, parlay engine, schedule.

modules/cbb_team_model.py, modules/cbb_schedule.py, and
modules/cbb_props_generator.py call ESPN's public JSON API live (see their
docstrings for the legal basis). These tests mock the HTTP layer for
determinism -- see the live seed generation this cycle already ran
(cbb_pipeline.md) for confirmation the real integration works end to end.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.cbb_props_loader import load_props_from_file, load_props_from_user_upload, unified_props
from modules.cbb_odds_loader import GameOddsLoadError, index_by_matchup, load_default_game_odds, load_uploaded_game_odds
from modules.cbb_injuries_loader import load_injury_data_from_file, load_injury_data_from_user_upload
from modules.cbb_schedule import fetch_todays_games
from modules.cbb_team_model import build_team_scoring_averages, real_margin_and_total_volatility, team_pace_estimate
from modules.cbb_moneyline_model import evaluate_game, fair_moneyline, win_probability_from_spread
from modules.cbb_prop_model import evaluate_prop, evaluate_props, minutes_volatility_multiplier
from modules.cbb_parlay_engine import make_leg, detect_correlations, evaluate_parlay
from modules import cbb_parlay_engine, nba_parlay_engine


class CbbLoaderTests(unittest.TestCase):
    def test_missing_default_files_return_empty_not_error(self):
        self.assertEqual(load_props_from_file("Z:/nope.json"), [])
        self.assertEqual(load_default_game_odds("Z:/nope.json"), {"games": {}})
        self.assertEqual(load_injury_data_from_file("Z:/nope.json"), [])

    def test_default_shipped_props_file_has_real_rows_with_basis(self):
        rows = load_props_from_file()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("basis", row)
            self.assertIn("real", row["basis"].lower())
            self.assertIn(row["category"], {"points", "rebounds", "assists", "PRA", "3PM"})

    def test_props_upload_roundtrip_reuses_nba_basketball_categories(self):
        text = json.dumps([{"player_name": "Test Guard", "team": "Duke", "category": "PTS", "line": 18.5}])
        rows = load_props_from_user_upload(text, file_format="json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "points")  # "PTS" normalized via sportsbook_parser, shared with NBA
        self.assertEqual(rows[0]["team"], "DUKE")

    def test_unified_props_upload_overrides_by_player_and_category(self):
        default_payload = {"props": [{"player_name": "Test Guard", "team": "DUKE", "category": "points", "line": 18.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cbb_props.json"
            path.write_text(json.dumps(default_payload))
            uploaded = json.dumps([{"player_name": "Test Guard", "category": "points", "line": 20.5}])
            merged = unified_props(default_path=path, uploaded=uploaded, uploaded_format="json")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["line"], 20.5)

    def test_game_odds_upload_and_matchup_index(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "Duke", "away_team": "UNC", "moneyline_home": -150, "moneyline_away": 130}]), file_format="json")
        indexed = index_by_matchup(result)
        self.assertIn(("DUKE", "UNC"), indexed)

    def test_game_odds_upload_skips_row_with_no_market_data(self):
        result = load_uploaded_game_odds(json.dumps([{"home_team": "Duke", "away_team": "UNC"}]), file_format="json")
        self.assertEqual(result["games"], {})

    def test_malformed_json_game_odds_upload_raises_loud_error(self):
        with self.assertRaises(GameOddsLoadError):
            load_uploaded_game_odds("{not valid", file_format="json")

    def test_injury_upload_reads_team_from_flat_record(self):
        payload = [{"player": "Test Player", "team": "Duke", "status": "OUT"}]
        rows = load_injury_data_from_user_upload(json.dumps(payload))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team"], "DUKE")


class CbbTeamModelMockedTests(unittest.TestCase):
    def _mock_response(self, payload):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_build_team_scoring_averages_computes_real_shape(self):
        teams_payload = {"sports": [{"leagues": [{"teams": [
            {"team": {"id": "1", "abbreviation": "DUKE"}},
            {"team": {"id": "2", "abbreviation": "UNC"}},
        ]}]}]}
        schedule_payload = {"events": [
            {"competitions": [{"status": {"type": {"completed": True}}, "competitors": [
                {"team": {"id": "1"}, "score": {"value": 80}},
                {"team": {"id": "2"}, "score": {"value": 70}},
            ]}]}
        ]}

        def fake_get(url, params=None, timeout=None):
            if "/teams" in url and "statistics" not in url and "/1/" not in url and "/2/" not in url:
                return self._mock_response(teams_payload)
            return self._mock_response(schedule_payload)

        with patch("requests.get", side_effect=fake_get):
            averages, games_by_team = build_team_scoring_averages(2026, team_pool_size=2)
        self.assertIn("DUKE", averages)
        self.assertEqual(averages["DUKE"]["points_scored_avg"], 80.0)
        self.assertEqual(averages["DUKE"]["games_played"], 1)

    def test_team_pace_estimate_uses_real_possession_formula(self):
        stats_payload = {"results": {"stats": {"categories": [
            {"stats": [
                {"name": "avgFieldGoalsAttempted", "value": 60.0},
                {"name": "avgOffensiveRebounds", "value": 10.0},
                {"name": "avgTurnovers", "value": 12.0},
                {"name": "avgFreeThrowsAttempted", "value": 20.0},
            ]}
        ]}}}
        with patch("requests.get", return_value=self._mock_response(stats_payload)):
            pace = team_pace_estimate("150", 2026)
        # Poss = 60 - 10 + 12 + 0.44*20 = 70.8
        self.assertEqual(pace, 70.8)

    def test_team_pace_estimate_returns_none_on_incomplete_data(self):
        with patch("requests.get", return_value=self._mock_response({"results": {"stats": {"categories": []}}})):
            self.assertIsNone(team_pace_estimate("150", 2026))

    def test_network_failure_fails_closed(self):
        with patch("requests.get", side_effect=RuntimeError("network down")):
            averages, games = build_team_scoring_averages(2026, team_pool_size=3)
        self.assertEqual(averages, {})

    def test_volatility_computed_from_real_games_by_team(self):
        games_by_team = {
            "DUKE": [{"opponent": "UNC", "points_scored": 80.0, "points_allowed": 70.0}],
            "MICH": [{"opponent": "OSU", "points_scored": 65.0, "points_allowed": 60.0}],
        }
        vol = real_margin_and_total_volatility(games_by_team)
        self.assertEqual(vol["games_sampled"], 2)
        self.assertIn("real completed games", vol["basis"])

    def test_volatility_falls_back_with_insufficient_data(self):
        vol = real_margin_and_total_volatility({})
        self.assertEqual(vol["basis"], "fallback_estimate")


class CbbScheduleTests(unittest.TestCase):
    def setUp(self):
        fetch_todays_games.cache_clear()

    def test_fetch_todays_games_parses_real_shaped_payload(self):
        payload = {"events": [
            {"date": "2026-01-01T00:00Z", "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": "DUKE"}},
                {"homeAway": "away", "team": {"abbreviation": "UNC"}},
            ]}]}
        ]}
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        with patch("requests.get", return_value=response):
            games = fetch_todays_games()
        self.assertEqual(games, [{"home_team": "DUKE", "away_team": "UNC", "start_time": "2026-01-01T00:00Z"}])

    def test_fetch_todays_games_fails_closed_on_network_error(self):
        with patch("requests.get", side_effect=RuntimeError("down")):
            self.assertEqual(fetch_todays_games(date(2099, 1, 1)), [])


class CbbMoneylineModelTests(unittest.TestCase):
    def setUp(self):
        self.averages = {"DUKE": {"points_scored_avg": 82.0, "points_allowed_avg": 68.0, "games_played": 20}, "UNC": {"points_scored_avg": 75.0, "points_allowed_avg": 72.0, "games_played": 20}}

    def test_win_probability_uses_supplied_stdev(self):
        tighter = win_probability_from_spread(5.0, stdev=8.0)
        wider = win_probability_from_spread(5.0, stdev=15.0)
        self.assertGreater(tighter, wider)

    def test_fair_moneyline_favorite_gets_negative_price(self):
        fair = fair_moneyline("DUKE", "UNC", averages=self.averages, margin_stdev=12.0)
        self.assertGreater(fair["spread"], 0)
        self.assertLess(fair["fair_home_moneyline"], 0)

    def test_evaluate_game_with_real_odds(self):
        game_odds = {"home_team": "DUKE", "away_team": "UNC", "moneyline": {"home": -250.0, "away": 210.0}}
        result = evaluate_game(game_odds, averages=self.averages, volatility={"margin_stdev": 12.0, "total_stdev": 11.0})
        self.assertIn("recommended_edge", result["moneyline"])


class CbbPropModelTests(unittest.TestCase):
    def test_low_minutes_widens_assumed_variance(self):
        full_rotation = minutes_volatility_multiplier(30.0)
        bench = minutes_volatility_multiplier(6.0)
        self.assertLess(full_rotation, bench)
        self.assertEqual(full_rotation, 1.0)  # already at/above full-rotation minutes -> no widening

    def test_evaluate_prop_computes_valid_probabilities(self):
        row = evaluate_prop({"player_name": "Test Guard", "team": "DUKE", "category": "points", "line": 18.5, "over_price": -110.0, "under_price": -110.0}, minutes_per_game=28.0)
        self.assertAlmostEqual(row["model_probability_over"] + row["model_probability_under"], 1.0, places=4)
        self.assertIn(row["risk_tier"], {"low", "medium", "high"})

    def test_evaluate_props_uses_per_player_minutes(self):
        props = [{"player_name": "Bench Guy", "category": "points", "line": 4.5, "over_price": -110.0, "under_price": -110.0}]
        rows = evaluate_props(props, minutes_by_player={"Bench Guy": 6.0})
        self.assertEqual(len(rows), 1)


class CbbParlayEngineReExportTests(unittest.TestCase):
    def test_reexports_are_the_same_objects_as_nba_parlay_engine(self):
        self.assertIs(cbb_parlay_engine.make_leg, nba_parlay_engine.make_leg)
        self.assertIs(cbb_parlay_engine.detect_correlations, nba_parlay_engine.nba_detect_correlations)
        self.assertIs(cbb_parlay_engine.evaluate_parlay, nba_parlay_engine.evaluate_parlay)

    def test_pra_overlap_pattern_fires_for_cbb_legs(self):
        points = make_leg(description="pts", model_probability=0.55, price=-110.0, player_id="p1", market="points", side="over")
        pra = make_leg(description="pra", model_probability=0.55, price=-110.0, player_id="p1", market="PRA", side="over")
        findings = detect_correlations([points, pra])
        self.assertEqual(findings[0]["kind"], "overlapping_stat_categories")

    def test_evaluate_parlay_works_for_cbb_legs(self):
        legs = [make_leg(description="a", model_probability=0.6, price=120.0), make_leg(description="b", model_probability=0.6, price=120.0)]
        result = evaluate_parlay(legs)
        self.assertIn("adjusted_hit_probability", result)


if __name__ == "__main__":
    unittest.main()
