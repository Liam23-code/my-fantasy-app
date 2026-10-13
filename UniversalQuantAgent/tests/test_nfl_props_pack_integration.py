"""The NFL upgrade pack inside ``modules.nfl_prop_analysis`` (``project_stat`` / ``evaluate_line``). Offline.

Fixtures come from the engine's own test module. The default (no ``pack_context``) must be byte-identical.
"""
from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_nfl_prop_analysis as base  # noqa: E402

from modules import nfl_prop_analysis as npa  # noqa: E402
from modules import nfl_upgrade_pipeline  # noqa: E402

BAD_PROTECTION = {
    "offense": {"ol_pass_block_win_rate": 0.62, "ol_continuity": 0.65, "ol_injuries": 2.0, "ol_run_block_win_rate": 0.64},
    "opponent_defense": {"opp_pressure_rate": 0.38, "opp_blitz_rate": 0.34, "opp_pass_rush_win_rate": 0.36,
                         "opp_simulated_pressure_success": 0.33, "opp_run_defense_grade": 88.0, "opp_stacked_box_rate": 0.31},
    "qb": {"qb_time_to_throw": 3.1, "qb_pressure_epa": -0.8, "qb_sack_avoidance": 0.90, "qb_release_time": 2.6},
    "game": {"vegas_spread": 3.0, "vegas_total": 41.0},
}
GOOD_SPOT = {
    "offense": {"team_epa_per_play": 0.15, "explosive_pass_rate": 0.115, "explosive_run_rate": 0.15,
                "red_zone_efficiency": 0.68, "ol_pass_block_win_rate": 0.78, "ol_run_block_win_rate": 0.76,
                "ol_continuity": 0.95},
    "opponent_defense": {"opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_pressure_rate": 0.24,
                         "opp_coverage_grade": 8.0, "opp_explosive_play_prevention": 6.0, "opp_run_defense_grade": 20.0},
    "qb": {"qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96},
    "game": {"vegas_spread": -3.0, "vegas_total": 52.0},
}

PLAYERS = {
    "QB": base.qb_pocket,
    "RB": base.rb_one,
    "WR": base.wr_one,
}


class PackCase(base.IsolatedStatusCase):
    pass


class DefaultIsUnchangedTests(PackCase):
    def test_every_category_is_byte_identical_without_a_context(self):
        for make in (base.qb_pocket, base.rb_one, base.wr_one):
            player = make()
            for label in npa.CATEGORY_LABELS:
                try:
                    plain = npa.project_stat(player, label)
                except ValueError:
                    continue
                self.assertEqual(npa.project_stat(player, label, pack_context=None), plain, (player["name"], label))
                self.assertNotIn("pack", plain)
                self.assertNotIn("pack_applied", plain)

    def test_evaluate_line_is_identical_without_a_context(self):
        player = base.qb_pocket()
        label = "Passing Yards"
        plain = npa.evaluate_line(player, label, 249.5)
        self.assertEqual(npa.evaluate_line(player, label, 249.5, pack_context=None), plain)
        self.assertFalse(any(item["title"] == "Matchup upgrade pack" for item in plain["reasoning"]))


class DirectionTests(PackCase):
    def test_bad_protection_lowers_a_qbs_passing_yards_mean_and_raises_the_under(self):
        player = base.qb_pocket()
        plain = npa.project_stat(player, "Passing Yards")
        packed = npa.project_stat(player, "Passing Yards", pack_context=BAD_PROTECTION)
        self.assertTrue(packed["pack_applied"])
        self.assertLess(packed["mean"], plain["mean"])
        self.assertEqual(packed["mean_before_pack"], plain["mean"])
        # a Gaussian market keeps its coefficient of variation
        self.assertAlmostEqual(packed["cv"], plain["cv"], places=3)
        self.assertAlmostEqual(packed["stdev"], packed["mean"] * plain["cv"], delta=0.01 * packed["stdev"] + 0.02)
        line = round(plain["mean"]) + 0.5
        before = npa.evaluate_line(player, "Passing Yards", line)
        after = npa.evaluate_line(player, "Passing Yards", line, pack_context=BAD_PROTECTION)
        self.assertGreater(after["probability_under"], before["probability_under"])
        self.assertAlmostEqual(after["probability_over"] + after["probability_under"], 1.0, places=3)
        self.assertEqual(after["mean"], packed["mean"])
        self.assertEqual(after["distribution"]["mean"], packed["mean"])

    def test_a_good_spot_raises_the_over(self):
        player = base.qb_pocket()
        plain = npa.project_stat(player, "Passing Yards")
        line = round(plain["mean"]) + 0.5
        before = npa.evaluate_line(player, "Passing Yards", line)
        after = npa.evaluate_line(player, "Passing Yards", line, pack_context=GOOD_SPOT)
        self.assertGreater(after["mean"], before["mean"])
        self.assertGreater(after["probability_over"], before["probability_over"])

    def test_pass_game_collapse_cuts_a_wrs_receiving_yards_and_touchdown_rate(self):
        player = base.wr_one()
        for label in ("Receiving Yards", "Receiving TDs"):
            plain = npa.project_stat(player, label)
            packed = npa.project_stat(player, label, pack_context=BAD_PROTECTION)
            self.assertLess(packed["mean"], plain["mean"], label)

    def test_a_poisson_market_uses_sqrt_of_the_adjusted_mean(self):
        player = base.rb_one()
        packed = npa.project_stat(player, "Rushing TDs", pack_context=BAD_PROTECTION)
        self.assertEqual(packed["family"], "poisson")
        self.assertTrue(packed["pack_applied"])
        self.assertAlmostEqual(packed["stdev"], math.sqrt(packed["mean"]), places=2)

    def test_anytime_td_recomposes_the_rushing_and_receiving_rates(self):
        player = base.rb_one()
        plain = npa.project_stat(player, "Anytime TD (Rush + Rec)")
        packed = npa.project_stat(player, "Anytime TD (Rush + Rec)", pack_context=BAD_PROTECTION)
        self.assertTrue(packed["pack_applied"])
        self.assertLess(packed["mean"], plain["mean"])
        self.assertAlmostEqual(packed["stdev"], math.sqrt(packed["mean"]), places=2)
        priced = npa.evaluate_line(player, "Anytime TD (Rush + Rec)", 0.5, pack_context=BAD_PROTECTION)
        self.assertLess(priced["probability_over"], npa.evaluate_line(player, "Anytime TD (Rush + Rec)", 0.5)["probability_over"])

    def test_a_market_the_pack_does_not_touch_comes_back_unchanged(self):
        runner = base.qb_runner()
        plain = npa.project_stat(runner, "Rushing Yards")
        packed = npa.project_stat(runner, "Rushing Yards", pack_context={**BAD_PROTECTION, "game": {}})
        self.assertEqual(packed["mean"], plain["mean"])   # a QB's rushing yards are not in the brief's formulas
        self.assertFalse(packed["pack_applied"])


class SafetyTests(PackCase):
    def test_an_empty_context_is_a_no_op_with_a_report(self):
        player = base.qb_pocket()
        plain = npa.project_stat(player, "Passing Yards")
        packed = npa.project_stat(player, "Passing Yards", pack_context={})
        self.assertFalse(packed["pack_applied"])
        for key, value in plain.items():
            self.assertEqual(packed[key], value, key)
        self.assertTrue(packed["pack"]["explanations"])

    def test_status_overlay_is_applied_before_the_pack(self):
        player = base.qb_pocket()
        self.set_status(player["player_id"], "QUESTIONABLE")
        plain = npa.project_stat(player, "Passing Yards")
        packed = npa.project_stat(player, "Passing Yards", pack_context=GOOD_SPOT)
        self.assertEqual(packed["mean_before_pack"], plain["mean"])   # the status-trimmed mean is the pack's input
        self.assertEqual(packed["status"], plain["status"])

    def test_out_players_stay_at_zero(self):
        player = base.qb_pocket()
        self.set_status(player["player_id"], "OUT")
        packed = npa.project_stat(player, "Passing Yards", pack_context=GOOD_SPOT)
        self.assertEqual(packed["mean"], 0.0)

    def test_the_context_is_not_mutated(self):
        ctx = copy.deepcopy(BAD_PROTECTION)
        npa.project_stat(base.wr_one(), "Receiving Yards", pack_context=ctx)
        self.assertEqual(ctx, BAD_PROTECTION)

    def test_a_defect_inside_the_pack_cannot_break_pricing(self):
        player = base.qb_pocket()
        plain = npa.evaluate_line(player, "Passing Yards", 249.5)
        with mock.patch.object(nfl_upgrade_pipeline, "project_with_pack", side_effect=RuntimeError("boom")):
            result = npa.evaluate_line(player, "Passing Yards", 249.5, pack_context=BAD_PROTECTION)
        self.assertFalse(result["projection"]["pack_applied"])
        self.assertEqual(result["probability_over"], plain["probability_over"])
        self.assertTrue(any("could not be applied" in w for w in result["projection"]["pack"]["warnings"]))

    def test_a_kicker_is_reported_not_adjusted(self):
        kicker = base.make_player("00-KC-K1", "Kyle Kicker", "K", "KC", passing_yards=10.0)
        packed = npa.project_stat(kicker, "Passing Yards", pack_context=BAD_PROTECTION)
        self.assertFalse(packed["pack_applied"])
        self.assertTrue(packed["pack"]["warnings"])

    def test_everything_is_finite(self):
        player = base.rb_one()
        for label in npa.CATEGORY_LABELS:
            try:
                result = npa.evaluate_line(player, label, 0.5 if "TD" in label else 49.5, pack_context=GOOD_SPOT)
            except ValueError:
                continue
            for key in ("mean", "stdev", "probability_over", "probability_under"):
                self.assertTrue(math.isfinite(result[key]), (label, key))


class ReasoningTests(PackCase):
    def test_a_cited_pack_item_appears_only_when_the_pack_moved_the_mean(self):
        player = base.qb_pocket()
        applied = npa.evaluate_line(player, "Passing Yards", 249.5, pack_context=GOOD_SPOT)
        items = [i for i in applied["reasoning"] if i["title"] == "Matchup upgrade pack"]
        self.assertEqual(len(items), 1)
        self.assertIn("priors", items[0]["detail"])
        self.assertIn("->" if False else "from", items[0]["detail"])
        self.assertIn(items[0]["tone"], {"pro", "con"})
        neutral = npa.evaluate_line(player, "Passing Yards", 249.5, pack_context={})
        self.assertFalse(any(i["title"] == "Matchup upgrade pack" for i in neutral["reasoning"]))

    def test_peer_rank_and_headline_still_work_on_an_adjusted_projection(self):
        player = base.qb_pocket()
        result = npa.evaluate_line(player, "Passing Yards", 249.5, pool=base._skill_pool(), pack_context=BAD_PROTECTION)
        self.assertTrue(result["headline"])
        self.assertGreaterEqual(len(result["reasoning"]), 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
