"""Offline contracts for the NFL analytics, projections, and Graph Lab."""
import unittest
from unittest.mock import patch

import pandas as pd

from modules import nfl_stats
from modules.nfl_analysis import analyze_nfl_player
from modules.nfl_graph_lab import (
    render_defensive_pressure_map, render_pace_play_volume,
    render_qb_passing_map, render_rb_usage_funnel, render_route_tree,
)
from modules.nfl_player import get_nfl_player
from modules.nfl_projections import project_nfl_player
from modules.nfl_slate import analyze_nfl_slate


class NFLEngineContracts(unittest.TestCase):
    def setUp(self):
        self.table = nfl_stats.fallback_player_table(2025)
        self.loader = patch("modules.nfl_stats.load_player_stats", return_value=self.table)
        self.loader.start()

    def tearDown(self):
        self.loader.stop()

    def test_position_schema_and_blended_percentiles(self):
        profile = nfl_stats.normalized_profile("Josh Allen", 2025, "Position")
        self.assertEqual(profile["position"], "QB")
        self.assertEqual(len(profile["attributes"]), 9)
        for attribute in profile["attributes"]:
            expected = round(.6 * attribute["league_percentile"] + .4 * attribute["position_percentile"], 1)
            self.assertEqual(attribute["blended_percentile"], expected)

    def test_raw_and_adjusted_modes_apply_sample_confidence(self):
        self.table.loc[self.table["player"] == "Justin Jefferson", ["targets", "plays"]] = [10, 100]
        raw = nfl_stats.normalized_profile("Justin Jefferson", 2025, "League", "Raw")
        adjusted = nfl_stats.normalized_profile("Justin Jefferson", 2025, "League", "Adjusted")
        self.assertEqual(raw["attributes"][0]["badge_value"], raw["attributes"][0]["raw_value"])
        self.assertLess(adjusted["attributes"][0]["badge_value"], raw["attributes"][0]["badge_value"])
        self.assertLess(adjusted["attributes"][0]["sample_confidence"], 1.0)

    def test_analysis_identity_tiers_and_volatility(self):
        result = analyze_nfl_player("Saquon Barkley", None, 2025)
        self.assertIn("RB1", result["identity_summary"])
        self.assertTrue(result["strengths"])
        self.assertGreaterEqual(result["volatility_score"], 0)
        self.assertLessEqual(result["volatility_score"], 100)

    def test_live_role_rows_merge_by_canonical_position(self):
        plays = pd.DataFrame([
            {"game_id":"g1","season_type":"REG","posteam":"BAL","defteam":"PIT","play_type":"pass","passer_player_id":"qb","passer_player_name":"Lamar Jackson","pass_attempt":1,"complete_pass":1,"yards_gained":20,"pass_touchdown":0,"epa":.2,"success":1,"yardline_100":50},
            {"game_id":"g1","season_type":"REG","posteam":"BAL","defteam":"PIT","play_type":"run","rusher_player_id":"qb","rusher_player_name":"Lamar Jackson","rush_attempt":1,"yards_gained":12,"rush_touchdown":0,"success":1,"yardline_100":45},
            {"game_id":"g2","season_type":"REG","posteam":"SF","defteam":"SEA","play_type":"run","rusher_player_id":"rb","rusher_player_name":"Christian McCaffrey","rush_attempt":1,"yards_gained":8,"rush_touchdown":0,"success":1,"yardline_100":40},
            {"game_id":"g2","season_type":"REG","posteam":"SF","defteam":"SEA","play_type":"pass","receiver_player_id":"rb","receiver_player_name":"Christian McCaffrey","complete_pass":1,"yards_gained":9,"pass_touchdown":0,"air_yards":3,"success":1,"yardline_100":35},
            {"game_id":"g3","season_type":"REG","posteam":"LA","defteam":"SF","play_type":"pass","receiver_player_id":"wr","receiver_player_name":"Puka Nacua","complete_pass":1,"yards_gained":18,"pass_touchdown":0,"air_yards":12,"success":1,"yardline_100":30},
            {"game_id":"g3","season_type":"REG","posteam":"LA","defteam":"SF","play_type":"run","rusher_player_id":"wr","rusher_player_name":"Puka Nacua","rush_attempt":1,"yards_gained":4,"rush_touchdown":0,"success":1,"yardline_100":25},
        ])
        table = nfl_stats._aggregate_live(plays, 2025, {"qb":"QB", "rb":"RB", "wr":"WR"})
        positions = table.set_index("player")["player_position"].to_dict()
        self.assertEqual(positions["Lamar Jackson"], "QB")
        self.assertEqual(positions["Christian McCaffrey"], "RB")
        self.assertEqual(positions["Puka Nacua"], "WR")
        cmc = table[table["player"] == "Christian McCaffrey"].iloc[0]
        self.assertGreater(cmc["rushing_yards"], 0)
        self.assertGreater(cmc["receiving_yards"], 0)
        self.assertGreater(cmc["rush_yards"], 0)
        self.assertGreater(cmc["rec_yards"], 0)

    def test_canonical_position_detection_uses_player_position(self):
        self.assertEqual(nfl_stats.detect_player_position({
            "player": "Puka Nacua", "player_position": "WR",
            "carries": 99, "targets": 1,
        }), "WR")
        self.assertEqual(nfl_stats.detect_player_position({
            "player": "Fallback Passer", "player_position": None,
            "attempts": 300, "targets": 0, "carries": 40,
        }), "QB")
        self.assertEqual(nfl_stats.resolve_player("Puka Nacua", 2025)[0]["player_position"], "WR")

    def test_qb_projection_routes_passing_and_rushing_only(self):
        result = project_nfl_player("Lamar Jackson", "PIT", 2025)
        projection = result["projection"]
        self.assertEqual(result["position"], "QB")
        self.assertGreater(projection["rushing_yards"], 0)
        self.assertEqual(projection["receiving_yards"], 0)
        self.assertEqual(projection["targets"], 0)
        self.assertEqual(projection["receptions"], 0)
        self.assertEqual(projection["receiving_targets"], 0)
        self.assertEqual(projection["receiving_receptions"], 0)

    def test_rb_projection_keeps_receiving_work(self):
        result = project_nfl_player("Christian McCaffrey", "SEA", 2025)
        projection = result["projection"]
        self.assertEqual(result["position"], "RB")
        self.assertGreater(projection["rushing_yards"], 0)
        self.assertGreater(projection["receiving_yards"], 0)
        self.assertGreater(projection["targets"], 0)
        self.assertGreater(projection["receptions"], 0)
        self.assertGreater(projection["receiving_targets"], 0)
        self.assertGreater(projection["receiving_receptions"], 0)

    def test_wr_projection_and_route_tree_are_position_correct(self):
        result = project_nfl_player("Puka Nacua", "SF", 2025)
        projection = result["projection"]
        self.assertEqual(result["position"], "WR")
        self.assertGreater(projection["receiving_yards"], 0)
        self.assertGreater(projection["targets"], 0)
        self.assertEqual(projection["rushing_yards"], 0)
        figure = render_route_tree("Puka Nacua", None, 2025)
        self.assertTrue(figure.layout.meta["route_data_exists"])
        figure.to_json()

    def test_te_projection_routes_receiving_only(self):
        result = project_nfl_player("Travis Kelce", "DEN", 2025)
        projection = result["projection"]
        self.assertEqual(result["position"], "TE")
        self.assertGreater(projection["receiving_yards"], 0)
        self.assertGreater(projection["targets"], 0)
        self.assertEqual(projection["rushing_yards"], 0)

    def test_missing_route_data_returns_message_not_position_error(self):
        mask = self.table["player"] == "Puka Nacua"
        self.table.loc[mask, ["targets", "target_share", "yards_per_route_run"]] = 0
        figure = render_route_tree("Puka Nacua", None, 2025)
        self.assertFalse(figure.layout.meta["route_data_exists"])
        self.assertEqual(figure.layout.annotations[0].text, "No route data available for this player yet.")

    def test_missing_aggregate_defaults_to_zero_and_marks_low_confidence(self):
        stats = nfl_stats.canonical_player_stats({
            "player": "Partial Back", "player_position": "RB",
            "games": 4, "rush_attempts": 40, "rush_yards": 180,
        })
        for field in nfl_stats.PROJECTION_FIELDS_BY_POSITION["RB"]:
            self.assertIn(field, stats)
            self.assertIsNotNone(stats[field])
        self.assertEqual(stats["rec_yards"], 0)
        self.assertTrue(stats["projection_low_confidence"])
        self.assertIn("rec_yards", stats["projection_missing_fields"])
    def test_public_player_contract_exposes_canonical_position_stats(self):
        expected = {
            "QB": {"pass_attempts", "pass_yards", "pass_tds", "ints", "rush_attempts", "rush_yards", "rush_tds"},
            "RB": {"rush_attempts", "rush_yards", "rush_tds", "targets", "receptions", "rec_yards", "rec_tds"},
            "WR": {"targets", "receptions", "rec_yards", "rec_tds", "air_yards", "routes_run", "red_zone_targets"},
            "TE": {"targets", "receptions", "rec_yards", "rec_tds", "air_yards", "routes_run", "red_zone_targets"},
        }
        for name, position in [("Josh Allen", "QB"), ("Saquon Barkley", "RB"), ("Puka Nacua", "WR"), ("Travis Kelce", "TE")]:
            player = get_nfl_player(name, 2025)
            self.assertEqual(player["position"], position)
            self.assertTrue(expected[position].issubset(player["stats"]))
            self.assertTrue(all(player["stats"][field] is not None for field in expected[position]))

    def test_named_player_projection_validation_matrix(self):
        lamar = project_nfl_player("Lamar Jackson", "PIT", 2025)["projection"]
        self.assertGreater(lamar["rushing_yards_projection"], 10)
        self.assertGreater(lamar["passing_tds_projection"], 0)
        self.assertEqual(lamar["receiving_yards_projection"], 0)

        allen = project_nfl_player("Josh Allen", "KC", 2025)["projection"]
        self.assertGreater(allen["passing_yards_projection"], 0)
        self.assertGreater(allen["passing_tds_projection"], 0)
        self.assertGreater(allen["rushing_yards_projection"], 10)
        self.assertGreater(allen["rushing_tds_projection"], 0)

        saquon = project_nfl_player("Saquon Barkley", "DAL", 2025)["projection"]
        self.assertGreater(saquon["rushing_yards_projection"], 0)
        self.assertGreaterEqual(saquon["receptions_projection"], 1)
        self.assertGreaterEqual(saquon["receiving_yards_projection"], 5)
        self.assertGreater(saquon["rushing_tds_projection"], 0)
        self.assertGreater(saquon["fantasy_points_projection"], 15)

        cmc = project_nfl_player("Christian McCaffrey", "SEA", 2025)["projection"]
        self.assertGreater(cmc["rushing_yards_projection"], 0)
        self.assertGreater(cmc["receiving_yards_projection"], 0)
        self.assertGreater(cmc["receiving_tds_projection"], 0)

        puka = project_nfl_player("Puka Nacua", "SF", 2025)
        self.assertEqual(puka["position"], "WR")
        self.assertGreaterEqual(puka["projection"]["targets_projection"], 3)
        self.assertGreater(puka["projection"]["receiving_tds_projection"], 0)
        self.assertEqual(puka["projection"]["rushing_yards_projection"], 0)

        kelce = project_nfl_player("Travis Kelce", "DEN", 2025)
        self.assertEqual(kelce["position"], "TE")
        self.assertGreater(kelce["projection"]["receiving_yards_projection"], 0)
        self.assertGreater(kelce["projection"]["receiving_tds_projection"], 0)
        self.assertEqual(kelce["projection"]["rushing_yards_projection"], 0)
    def test_projection_has_confidence_interval(self):
        result = project_nfl_player("Josh Allen", "KC", 2025)
        self.assertIn("expected_fantasy_points", result["projection"])
        self.assertLess(result["confidence"]["low"], result["confidence"]["high"])
        self.assertIn("Projected:", result["confidence"]["label"])

    def test_all_graph_lab_figures_serialize(self):
        figures = [
            render_qb_passing_map("Josh Allen", None, 2025),
            render_route_tree("Justin Jefferson", None, 2025),
            render_rb_usage_funnel("Saquon Barkley", None, 2025),
            render_defensive_pressure_map("Baltimore Ravens Defense", None, 2025),
            render_pace_play_volume("Josh Allen", None, 2025),
        ]
        for figure in figures:
            figure.to_json()
            self.assertIn("view", figure.layout.meta)

    def test_slate_contract_survives_provider_outage(self):
        with patch("modules.nfl_slate.compare_nfl_teams", side_effect=RuntimeError("offline")):
            slate = analyze_nfl_slate(season=2025)
        self.assertEqual(len(slate["games"]), 4)
        self.assertTrue(all(game["identity_summary"] for game in slate["games"]))


if __name__ == "__main__":
    unittest.main()
