"""Offline rendering contracts for Graph Lab figures."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import badge_graph, radar_chart, shot_chart, trend_graphs


class GraphLabContracts(unittest.TestCase):
    def setUp(self):
        self.player = {"id": 1, "full_name": "Test Player"}
        rows = []
        for pid, name, team, scale in [
            (1, "Test Player", "DEN", 1.0),
            (2, "Peer", "BOS", .8),
            (3, "Peer Two", "MIL", 1.2),
        ]:
            rows.append({
                "PLAYER_ID":pid,"PLAYER_NAME":name,"TEAM_ABBREVIATION":team,
                "MIN":32*scale,"PTS":20*scale,"REB":7*scale,"AST":6*scale,
                "FGA":15*scale,"FGM":8*scale,"FTA":5*scale,"TOV":2*scale,
                "STL":1.2*scale,"BLK":.8*scale,"FG3A":5*scale,"FG3M":2*scale,
                "FG3_PCT":.4,"FG_PCT":.53,"OREB":2*scale,"USG_PCT":.27,
                "TS_PCT":.62,"AST_PCT":.3,"REB_PCT":.13,"TM_TOV_PCT":.1,
                "DEF_RATING":109/scale,"PACE":100*scale,"PIE":.15,
                "NET_RATING":5*scale,
            })
        self.table = pd.DataFrame(rows)
        self.selection = (self.player, self.table.iloc[0], self.table, [])
        self.games = pd.DataFrame({
            "game_date":pd.date_range("2025-10-01",periods=12),
            "points":np.arange(12)+15,"rebounds":np.arange(12)/3+5,
            "assists":np.arange(12)/4+4,"minutes":np.arange(12)/5+30,
            "usage_pct":np.arange(12)/2+22,"ts_pct":np.arange(12)/3+55,
            "game_number":np.arange(1,13),
        })
        self.teams = pd.DataFrame({
            "TEAM_ABBREVIATION":["DEN","BOS","MIL"],
            "DEF_RATING":[110,106,112],"STL":[8,7,9],"BLK":[5,6,4],
            "REB":[44,46,43],"PACE":[99,101,103],"PTS":[116,112,118],
            "AST":[29,27,25],
        })

    def test_shot_chart_has_court_heatmap_and_hover_fields(self):
        shots = pd.DataFrame({
            "LOC_X":[0,10,-40,120,-180],"LOC_Y":[10,20,100,240,80],
            "SHOT_MADE_FLAG":[1,0,1,0,1],"SHOT_ATTEMPTED_FLAG":[1]*5,
            "SHOT_TYPE":["2PT Field Goal"]*3+["3PT Field Goal"]*2,
        })
        with patch("modules.shot_chart.player_row", return_value=self.selection):
            figure = shot_chart.render_shot_chart("Test Player","2025-26","Efficiency",shots)
        self.assertGreaterEqual(len(figure.data), 10)
        self.assertIn("Expected points", figure.data[0].hovertemplate)
        figure.to_json()

    def test_radar_and_badge_figures_are_valid(self):
        with patch("modules.radar_chart.player_row", return_value=self.selection):
            radar = radar_chart.render_efficiency_radar("Test Player","2025-26","Season")
        with patch("modules.badge_graph.player_row", return_value=self.selection):
            badge = badge_graph.render_badge_graph("Test Player","2025-26")
        self.assertEqual(len(radar.data[0].r), 8)
        self.assertEqual(len(badge.layout.meta["ratings"]), 8)
        self.assertNotIn("Dunking", badge.layout.meta["ratings"])
        radar.to_json(); badge.to_json()

    def test_badge_percentiles_samples_and_modes_are_transparent(self):
        table = self.table.copy()
        table["GP"] = [60, 60, 60]
        table["POSITION"] = ["C", "C", "G"]
        table.loc[0, ["FGA", "FG3A", "FG3M"]] = [10.0, .1, .08]
        row = table.iloc[0]
        raw = badge_graph.calculate_badge_profile(
            row, table, display_mode="Raw", comparison_mode="Entire league"
        )
        adjusted = badge_graph.calculate_badge_profile(
            row, table, display_mode="Adjusted", comparison_mode="Same position"
        )
        raw_three = next(item for item in raw["attributes"] if item["attribute"] == "3PT shooting")
        adjusted_three = next(item for item in adjusted["attributes"] if item["attribute"] == "3PT shooting")
        expected = min(
            99.0,
            round(.6 * raw_three["league_percentile"] + .4 * raw_three["position_percentile"], 1),
        )
        self.assertEqual(raw_three["rating"], expected)
        self.assertLess(adjusted_three["badge_value"], raw_three["badge_value"])
        self.assertLess(adjusted_three["sample_confidence"], 1.0)
        self.assertIn(adjusted_three["tier"], badge_graph.BADGE_COLORS)
        self.assertIn("Low sample size: ratings may be unstable.", adjusted["warnings"])
    def test_spider_badge_is_transparent_and_mode_aware(self):
        with patch("modules.badge_graph.player_row", return_value=self.selection):
            spider = badge_graph.render_spider_badge_graph(
                "Test Player", "Raw", "Same position", season="2025-26"
            )
        self.assertEqual(spider.layout.meta["view"], "Spider")
        self.assertEqual(spider.layout.meta["display_mode"], "Raw")
        self.assertEqual(spider.layout.meta["comparison_mode"], "Same position")
        fills = [trace.fillcolor for trace in spider.data if getattr(trace, "fill", None) == "toself"]
        self.assertTrue(any("rgba" in str(color) for color in fills))
        skill_trace = next(trace for trace in spider.data if trace.name == "Skill rating")
        self.assertEqual(len(skill_trace.r), 8)
        self.assertEqual(
            [item["attribute"] for item in spider.layout.meta["attributes"]],
            ["Finishing", "Mid-range", "3PT shooting", "Playmaking", "Defense", "Rebounding", "Efficiency", "Pace compatibility"],
        )
        self.assertNotIn("Dunking", spider.layout.meta["ratings"])
        self.assertIn("Dilution", skill_trace.hovertemplate)
        label_trace = next(trace for trace in spider.data if trace.name == "Skill labels")
        self.assertTrue(all(isinstance(value, (int, float, np.number)) for value in label_trace.theta))
        spider.to_json()

    def test_close_shot_dilution_uses_requested_weights(self):
        table = self.table.copy()
        table["GP"] = 60
        table["FGA"] = [600, 500, 700]
        table["FG3A"] = [200, 180, 260]
        table["FGM"] = [330, 250, 350]
        table["FG3M"] = [80, 60, 100]
        table["LAYUP_FGA"] = [100, 80, 120]
        table["LAYUP_FG_PCT"] = [.60, .55, .65]
        table["CLOSE_SHOT_FGA"] = [100, 90, 110]
        table["CLOSE_SHOT_FG_PCT"] = [.50, .48, .54]
        table["TRUE_MID_RANGE_FGA"] = [80, 70, 60]
        table["TRUE_MID_RANGE_FG_PCT"] = [.45, .42, .47]
        table["DUNK_FGA"] = [20, 12, 25]
        table["DUNK_FG_PCT"] = [.90, .85, .92]
        profile = badge_graph.calculate_badge_profile(
            table.iloc[0], table, display_mode="Raw"
        )
        attributes = {item["attribute"]: item for item in profile["attributes"]}
        self.assertAlmostEqual(attributes["Finishing"]["attempts"], 178.0)
        self.assertAlmostEqual(attributes["Mid-range"]["attempts"], 135.0)
        self.assertNotIn("Dunking", attributes)
        finishing = attributes["Finishing"]["dilution_contributions"]
        midrange = attributes["Mid-range"]["dilution_contributions"]
        self.assertAlmostEqual(finishing["Close shots × 0.60"], 30.0)
        self.assertAlmostEqual(finishing["Dunks × 0.90"], 16.2)
        self.assertAlmostEqual(midrange["Close shots × 0.55"], 27.5)
        self.assertAlmostEqual(midrange["True mid-range × efficiency"], 36.0)
    def test_player_and_league_trend_figures_are_valid(self):
        with patch("modules.trend_graphs.player_game_data", return_value=(self.player,self.games,[])), patch("modules.trend_graphs.load_player_table", return_value=self.table):
            momentum = trend_graphs.render_momentum_line("Test Player","2025-26")
            timeline = trend_graphs.render_performance_timeline("Test Player","2025-26")
        with patch("modules.trend_graphs.load_player_table", return_value=self.table), patch("modules.trend_graphs.resolve_player", return_value=self.player):
            scatter = trend_graphs.render_usage_efficiency_scatter("2025-26","Test Player")
        with patch("modules.trend_graphs.load_team_table", return_value=self.teams):
            difficulty = trend_graphs.render_opponent_difficulty_curve("Guards","2025-26")
            pace = trend_graphs.render_pace_impact_curve("2025-26")
        for figure in (momentum,timeline,scatter,difficulty,pace):
            figure.to_json()
        self.assertEqual(len(momentum.data),4)
        self.assertEqual(len(pace.data),6)


if __name__ == "__main__":
    unittest.main()