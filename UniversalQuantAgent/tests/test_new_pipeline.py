"""Offline contract tests for the strengthened quant pipeline."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from modules.data_quality import fuzzy_name_match, parse_minutes, rolling_average
from modules.daily_slate import get_daily_slate
from modules.injury_parser import normalize_status
from modules.model_performance import get_model_performance_summary
from modules.parlay import build_parlay
from modules.props import compare_props
from modules.sports import fetch_nba_player_stats

class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.line = {"player_name":"Nikola Jokic","team":"DEN","category":"points",
                     "line":25.5,"sportsbook":"DraftKings","timestamp":"2026-01-01T00:00:00Z"}
        context = {"team":"DEN","minutes_trend":{"last_10":36},
                   "opponent_matchup_trend":{"explanation":"Neutral matchup."},
                   "injury_impact":{"warning":""},
                   "components":{"minutes":{"rolling_minutes":{"last_10":36},"minutes_projection":36},
                                 "pace":{"pace_projection":100},
                                 "matchup":{"difficulty_score":50},
                                 "availability":{"impact_score":0}}}
        self.fusion = {"player":"Nikola Jokic","team":"DEN",
                       "final_projection":{"points":27.0,"rebounds":12.0,"assists":9.0,"pra":48.0},
                       "confidence_range":{"points":{"low":23.0,"high":31.0}},
                       "key_drivers":["Stable role."],"context":context,
                       "base_projection":{"player":"Nikola Jokic"}}

    @patch("modules.props.get_reliability_score",return_value={"score":82.0,"rating":"high"})
    @patch("modules.props.fuse_projection")
    @patch("modules.props.unified_props")
    def test_compare_props_schema(self,fetch_lines,fuse,_):
        fetch_lines.return_value = [self.line]
        fuse.return_value = self.fusion
        row = compare_props("Nikola Jokic","BOS",["points"],"2025-26")[0]
        expected = {"player","team","category","projection","minutes_adjusted_projection",
                    "sportsbook_line","edge","confidence_low","confidence_high","best_sportsbook",
                    "lean","key_drivers","sportsbook","timestamp"}
        self.assertEqual(set(row),expected)
        self.assertEqual(row["edge"],1.5)
        self.assertTrue(any("reliability" in driver.lower() for driver in row["key_drivers"]))

    def test_parlay_schema(self):
        prop = {"player":"A Player","team":"DEN","edge":2,"confidence_low":20,
                "confidence_high":28,"key_drivers":["Quant reliability: 80/100."]}
        result = build_parlay([prop])
        self.assertEqual(set(result),{"props","combined_edge","risk_score","correlation_warnings",
                                     "injury_warnings","pace_correlation","blowout_risk",
                                     "expected_value","summary"})
        self.assertTrue(0 <= result["risk_score"] <= 100)
        self.assertIn("reliability",result["summary"].lower())

    def test_data_quality_helpers(self):
        self.assertAlmostEqual(parse_minutes("31:30"),31.5)
        values = rolling_average(pd.Series([10.0,20.0]),5)
        self.assertEqual(values.iloc[-1],15.0)
        self.assertEqual(fuzzy_name_match("nikola jokc",["Nikola Jokic","Nikola Jovic"]),"Nikola Jokic")

    def test_injury_statuses(self):
        self.assertEqual(normalize_status("Game-time decision"),"QUESTIONABLE")
        self.assertEqual(normalize_status("Out"),"OUT")

    def test_performance_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"performance.csv"
            path.write_text("category,projection,actual,sportsbook_line,date\npoints,25,27,24.5,2026-01-01\n")
            result = get_model_performance_summary(path)
        self.assertEqual(result["sample_size"],1)
        self.assertEqual(result["categories"][0]["mae"],2.0)

    @patch("modules.daily_slate.fuse_projection")
    @patch("modules.daily_slate.unified_props")
    @patch("modules.daily_slate.fetch_todays_games")
    def test_daily_slate_schema(self,games,props,fuse):
        games.return_value = [{"home_team":"DEN","away_team":"BOS","start_time":"now"}]
        props.return_value = [self.line]
        fuse.return_value = self.fusion
        result = get_daily_slate()
        self.assertEqual(set(result),{"date","games","player_props","projections"})
        self.assertEqual(set(result["projections"][0]),{"player","team","projection","minutes","pace","matchup_difficulty"})
        self.assertEqual(result["projections"][0]["projection"]["points"],27.0)

    @patch("modules.sports.playergamelog.PlayerGameLog")
    @patch("modules.sports.leaguedashplayerstats.LeagueDashPlayerStats")
    def test_player_analysis_schema(self,league_endpoint,log_endpoint):
        base = pd.DataFrame([{"PLAYER_ID":1,"PLAYER_NAME":"Test Player","TEAM_ABBREVIATION":"DEN",
                              "GP":10,"MIN":30,"PTS":15,"REB":5,"AST":4,"FGA":12,"FGM":6,
                              "FTA":4,"FTM":3,"TOV":2,"STL":1,"BLK":1}])
        advanced = pd.DataFrame([{"PLAYER_ID":1,"USG_PCT":.24,"TS_PCT":.60}])
        base_call,advanced_call = MagicMock(),MagicMock()
        base_call.get_data_frames.return_value = [base]
        advanced_call.get_data_frames.return_value = [advanced]
        league_endpoint.side_effect = [base_call,advanced_call]
        logs = pd.DataFrame({"GAME_DATE":pd.date_range("2025-01-01",periods=10),
                             "MIN":["30:00"]*10,"PTS":range(1,11),"REB":[5]*10,"AST":[4]*10})
        log_call = MagicMock(); log_call.get_data_frames.return_value = [logs]
        log_endpoint.return_value = log_call
        result = fetch_nba_player_stats("Test Player","2025-26")
        self.assertEqual(set(result),{"player","team","season","season_avg","last5_avg",
                                     "last10_avg","season_totals","advanced","league_ranks",
                                     "trend_series"})
        self.assertEqual(result["season"],2025)
        self.assertEqual(result["last5_avg"]["ppg"],8.0)
        self.assertEqual(result["advanced"]["usage_pct"],24.0)
        self.assertEqual(result["league_ranks"]["ppg"]["rank"],1)

    @patch("modules.fusion_model.get_player_context")
    @patch("modules.fusion_model.project_player_statline")
    def test_fusion_contract(self,base_projection,player_context):
        base_projection.return_value = {
            "player":"Test Player","projected_statline":{"points":20,"rebounds":6,"assists":5,"pra":31},
            "confidence_ranges":{stat:{"low":10,"high":30} for stat in ("points","rebounds","assists","pra")},
            "recent_5_game_averages":{"points":22,"rebounds":7,"assists":6,"pra":35},
            "recent_10_game_averages":{"points":21,"rebounds":6,"assists":5,"pra":32}}
        player_context.return_value = {"team":"DEN","role_stability":80,
            "components":{"minutes":{"rolling_minutes":30,"minutes_projection":32},
                          "pace":{"pace_projection":102},"matchup":{"difficulty_score":45},
                          "availability":{"impact_score":0}}}
        from modules.fusion_model import fuse_projection
        result = fuse_projection("Test Player","BOS","2025-26")
        self.assertIn("final_projection",result)
        self.assertIn("confidence_range",result)
        self.assertIn("driver_breakdown",result)
        self.assertEqual(result["final_projection"]["pra"],
                         round(result["final_projection"]["points"]+result["final_projection"]["rebounds"]+result["final_projection"]["assists"],1))
    @patch("modules.context_engine.project_matchup_difficulty",return_value={"difficulty_score":45,"matchup_history_points":21,"explanation":"Favorable."})
    @patch("modules.context_engine.project_pace",return_value={"pace_projection":102,"recent_pace":101,"confidence":80})
    @patch("modules.context_engine.project_minutes",return_value={"minutes_projection":32,"blowout_risk":30,"confidence_low":29,"confidence_high":35,"coach_tendency_score":85,"rolling_minutes":{"last_10":31}})
    @patch("modules.context_engine.get_player_availability",return_value={"availability":"ACTIVE","impact_score":0,"warning":""})
    @patch("modules.context_engine._game_log_with_fuzzy_fallback")
    @patch("modules.context_engine._find_player",return_value={"id":1,"full_name":"Test Player"})
    def test_context_contract(self,_find,game_log,*_):
        logs = pd.DataFrame({"GAME_DATE":pd.date_range("2025-01-01",periods=12),
                             "MATCHUP":["DEN vs. BOS","DEN @ BOS"]*6,
                             "TEAM_ABBREVIATION":["DEN"]*12,"MIN":[30+i%3 for i in range(12)],
                             "PTS":[20+i%4 for i in range(12)],"REB":[6]*12,"AST":[5]*12,
                             "FGA":[15]*12,"FTA":[4]*12,"TOV":[2]*12})
        game_log.return_value = ({"id":1,"full_name":"Test Player"},logs,[])
        from modules.context_engine import get_player_context
        result = get_player_context("Test Player","BOS","2025-26")
        required = {"role_stability","usage_trend","minutes_trend","efficiency_trend",
                    "pace_trend","opponent_matchup_trend","injury_impact","blowout_risk",
                    "home_away_splits","back_to_back_fatigue","coach_rotation_tendencies"}
        self.assertTrue(required.issubset(result))
    def test_safe_ingestion_helpers(self):
        from modules.data_quality import (
            safe_dict,
            safe_get,
            safe_list,
            safe_scalar_to_dict,
        )

        self.assertIsNone(safe_get(7, "status"))
        self.assertEqual(safe_get({"status": "OUT"}, "status"), "OUT")
        self.assertEqual(safe_dict(["not", "a", "dict"]), {})
        self.assertEqual(safe_list({"not": "a list"}), [])
        self.assertEqual(safe_scalar_to_dict(42), {"value": 42})

    def test_injury_walker_handles_mixed_provider_types(self):
        from modules.injury_parser import walk_injury_records

        payload = {
            "displayName": "Denver Nuggets",
            "injuries": [
                {
                    "athlete": {"displayName": "Test Player"},
                    "type": 7,
                    "status": None,
                    "shortComment": "Provider omitted a status.",
                },
                "unexpected scalar",
                12,
                None,
                {"athlete": "malformed", "type": {"description": "Out"}},
            ],
        }
        results = []
        walk_injury_records(payload, "", results)
        self.assertEqual(len(results), 1)
        self.assertEqual(
            set(results[0]), {"team", "player", "status", "details"}
        )
        self.assertEqual(results[0]["player"], "Test Player")
        self.assertEqual(results[0]["status"], "ACTIVE")
        self.assertIsInstance(results[0]["details"], dict)

    def test_load_injury_data_from_file_enforces_clean_schema(self):
        from modules.injury_parser import load_injury_data_from_file

        payload = {
            "displayName": "Boston Celtics",
            "injuries": [
                {
                    "athlete": {"displayName": "Schema Player"},
                    "status": {"description": "Questionable"},
                    "details": "Knee soreness",
                },
                99,
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "injuries.json"
            path.write_text(json.dumps(payload))
            result = load_injury_data_from_file(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]), {"team", "player", "status", "details"})
        self.assertEqual(result[0]["status"], "QUESTIONABLE")
        self.assertIsInstance(result[0]["details"], dict)

    def test_load_injury_data_from_file_missing_path_returns_empty_not_error(self):
        from modules.injury_parser import load_injury_data_from_file

        self.assertEqual(load_injury_data_from_file("Z:/does/not/exist.json"), [])

    def test_sportsbook_fetch_functions_are_hard_disabled(self):
        from modules import sportsbook_scraper_disabled as sb

        for fn in (
            sb.fetch_draftkings_props,
            sb.fetch_fanduel_props,
            sb.fetch_betmgm_props,
            sb.fetch_caesars_props,
            sb.fetch_espnbet_props,
            sb.fetch_all_sportsbook_props,
            sb.fetch_daily_games,
        ):
            with self.assertRaises(RuntimeError):
                fn()

    def test_injury_fetch_functions_are_hard_disabled(self):
        from modules import injury_scraper_disabled as inj

        with self.assertRaises(RuntimeError):
            inj.fetch_injury_report()
        with self.assertRaises(RuntimeError):
            inj.fetch_injury_data_online()

    def test_no_network_libraries_imported_by_sportsbook_or_injury_modules(self):
        import ast

        for module_name in ("sportsbook_parser", "sportsbook_scraper_disabled", "injury_parser", "injury_scraper_disabled"):
            source = (ROOT / "modules" / f"{module_name}.py").read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            banned = imported & {"requests", "aiohttp", "urllib3", "httpx", "selenium", "playwright", "bs4"}
            self.assertEqual(banned, set(), msg=f"{module_name} imports network/scraping library: {banned}")

    def test_scalar_model_output_normalization(self):
        from modules.fusion_model import normalize_model_output

        result = normalize_model_output(31)
        self.assertEqual(result["value"], 31.0)
        self.assertEqual(result["confidence"], 50.0)
        self.assertEqual(result["details"], {})

    @patch("modules.minutes_model._blowout_risk", return_value=30.0)
    @patch("modules.minutes_model.get_player_availability", return_value=8)
    @patch("modules.minutes_model._find_team", return_value={"id": 2, "abbreviation": "BOS"})
    @patch("modules.minutes_model._game_data")
    def test_minutes_model_envelope(self, game_data, *_):
        games = pd.DataFrame(
            {
                "minutes": [30.0, 31.0, 32.0, 33.0, 34.0],
                "game_date": pd.date_range("2026-01-01", periods=5),
            }
        )
        game_data.return_value = ({"full_name": "Test Player"}, games, [])
        from modules.minutes_model import project_minutes

        result = project_minutes("Test Player", "BOS", "2025-26")
        self.assertTrue({"value", "confidence", "details"}.issubset(result))
        self.assertIsInstance(result["details"], dict)

    @patch("modules.pace_model.load_injury_data_from_file", return_value=[1, None, "bad"])
    @patch("modules.pace_model.fetch_league_team_stats")
    @patch("modules.pace_model.find_team")
    def test_pace_model_envelope(self, find_team, league_stats, _):
        find_team.side_effect = [
            {"id": 1, "abbreviation": "DEN"},
            {"id": 2, "abbreviation": "BOS"},
        ]
        league_stats.return_value = pd.DataFrame(
            [{"TEAM_ID": 1, "PACE": 101.0}, {"TEAM_ID": 2, "PACE": 99.0}]
        )
        from modules.pace_model import project_pace

        result = project_pace("DEN", "BOS", "2025-26")
        self.assertTrue({"value", "confidence", "details"}.issubset(result))
        self.assertIsInstance(result["details"], dict)

    @patch(
        "modules.matchup_model.load_injury_data_from_file",
        return_value=[3, {"team": "BOS", "status": 9}, {"team": "BOS", "status": "OUT"}],
    )
    @patch("modules.matchup_model._find_player", side_effect=ValueError("no log"))
    @patch("modules.matchup_model.fetch_league_team_stats")
    @patch(
        "modules.matchup_model.find_team",
        return_value={"id": 2, "abbreviation": "BOS"},
    )
    def test_matchup_model_envelope(self, _, league_stats, *_rest):
        league_stats.return_value = pd.DataFrame(
            [
                {"TEAM_ID": 1, "DEF_RATING": 111.0, "PACE": 101.0},
                {"TEAM_ID": 2, "DEF_RATING": 108.0, "PACE": 99.0},
            ]
        )
        from modules.matchup_model import project_matchup_difficulty

        result = project_matchup_difficulty(
            "Test Player", "BOS", "2025-26"
        )
        self.assertTrue({"value", "confidence", "details"}.issubset(result))
        self.assertIsInstance(result["details"], dict)


    def test_slate_correlations_handle_partial_records(self):
        from modules.correlation_engine import compute_slate_correlations

        result = compute_slate_correlations(
            [
                {
                    "player": "One",
                    "projection": {
                        "points": 20,
                        "rebounds": 8,
                        "assists": 5,
                        "pra": 33,
                    },
                    "minutes": 32,
                },
                {
                    "player": "Two",
                    "projection": {
                        "points": 28,
                        "rebounds": 5,
                        "assists": 8,
                        "pra": 41,
                    },
                    "minutes": 36,
                },
                {"player": "Malformed", "projection": 7},
                12,
            ]
        )
        self.assertEqual(result["observations"], 2)
        self.assertIn("points", result["available_stats"])
        self.assertIn("points", result["correlation_matrix"])
        self.assertTrue(
            all(
                isinstance(value, float)
                for row in result["correlation_matrix"].values()
                for value in row.values()
            )
        )

    def test_edge_heatmap_filters_and_sorts(self):
        from modules.edge_heatmap import prepare_edge_heatmap

        result = prepare_edge_heatmap(
            [
                {
                    "player": "One",
                    "category": "points",
                    "projection": 24,
                    "sportsbook_line": 21.5,
                    "edge": 2.5,
                    "reliability": 80,
                    "confidence": 75,
                    "sportsbook": "Book A",
                },
                {
                    "player": "Two",
                    "category": "points",
                    "projection": 17,
                    "sportsbook_line": 19.5,
                    "edge": -2.5,
                    "reliability": 60,
                    "confidence": 65,
                    "sportsbook": "Book B",
                },
                {"player": "Ignored", "category": "steals", "edge": 4},
            ],
            ["points"],
            "reliability",
        )
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["records"][0]["player"], "One")
        self.assertEqual(result["matrix"]["One"]["points"], 2.5)
        self.assertEqual(result["matrix"]["Two"]["points"], -2.5)

    def test_similarity_engine_returns_normalized_matches(self):
        from modules.similarity_engine import compute_player_similarity

        table = pd.DataFrame(
            [
                {
                    "PLAYER_ID": 1,
                    "PLAYER_NAME": "Alpha Player",
                    "TEAM_ABBREVIATION": "DEN",
                    "GP": 50,
                    "MIN": 34,
                    "PTS": 25,
                    "REB": 8,
                    "AST": 7,
                    "USG_PCT": .28,
                    "TS_PCT": .61,
                    "FGA": 18,
                    "FGM": 9,
                    "FG3A": 5,
                    "FTA": 6,
                    "FTM": 5,
                    "TOV": 3,
                    "STL": 1.2,
                    "BLK": .7,
                    "OREB": 2,
                    "DREB": 6,
                    "PACE": 100,
                    "DEF_RATING": 111,
                    "NET_RATING": 5,
                    "AST_PCT": .30,
                    "REB_PCT": .14,
                },
                {
                    "PLAYER_ID": 2,
                    "PLAYER_NAME": "Beta Player",
                    "TEAM_ABBREVIATION": "BOS",
                    "GP": 48,
                    "MIN": 33,
                    "PTS": 24,
                    "REB": 8,
                    "AST": 6.5,
                    "USG_PCT": .27,
                    "TS_PCT": .60,
                    "FGA": 17,
                    "FGM": 8.5,
                    "FG3A": 5,
                    "FTA": 6,
                    "FTM": 5,
                    "TOV": 3,
                    "STL": 1.1,
                    "BLK": .8,
                    "OREB": 2,
                    "DREB": 6,
                    "PACE": 101,
                    "DEF_RATING": 110,
                    "NET_RATING": 6,
                    "AST_PCT": .28,
                    "REB_PCT": .14,
                },
                {
                    "PLAYER_ID": 3,
                    "PLAYER_NAME": "Gamma Player",
                    "TEAM_ABBREVIATION": "MIA",
                    "GP": 40,
                    "MIN": 20,
                    "PTS": 9,
                    "REB": 2,
                    "AST": 1,
                    "USG_PCT": .15,
                    "TS_PCT": .52,
                    "FGA": 8,
                    "FGM": 3,
                    "FG3A": 3,
                    "FTA": 2,
                    "FTM": 1,
                    "TOV": 1,
                    "STL": .5,
                    "BLK": .1,
                    "OREB": .4,
                    "DREB": 1.6,
                    "PACE": 96,
                    "DEF_RATING": 118,
                    "NET_RATING": -5,
                    "AST_PCT": .08,
                    "REB_PCT": .06,
                },
            ]
        )
        result = compute_player_similarity(
            "Alpha Player",
            "2025-26",
            limit=2,
            player_table=table,
            recent_table=table,
        )
        self.assertEqual(result["similar_players"][0]["player"], "Beta Player")
        self.assertTrue(
            0 <= result["similar_players"][0]["similarity_score"] <= 100
        )
        self.assertIn(
            "scoring_profile",
            result["similar_players"][0]["dimension_scores"],
        )


if __name__ == "__main__":
    unittest.main()
