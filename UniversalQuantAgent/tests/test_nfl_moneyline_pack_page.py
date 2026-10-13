"""MoneyLines page: the matchup upgrade pack path (toggle, contexts, card, fail-soft). Offline."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_moneylines_page as base  # noqa: E402  (page fixtures; its module-level pack patch is NOT inherited)

from modules import nfl_matchup_features as nmf  # noqa: E402

ELITE = {
    "offense": {
        "team_epa_per_play": 0.15, "qb_efficiency": 0.25, "explosive_pass_rate": 0.115, "explosive_run_rate": 0.15,
        "red_zone_efficiency": 0.68, "ol_pass_block_win_rate": 0.78, "pace_seconds_per_play": 31.0,
        "wr1_separation": 3.5, "rb_yards_after_contact": 2.7,
    },
    "opponent_defense": {"opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_pressure_rate": 0.24,
                         "opp_coverage_grade": 8.0, "opp_explosive_play_prevention": 6.0},
    "qb": {"qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96},
    "game": {},
    "warnings": [],
}
AVERAGE = {
    "offense": {"team_epa_per_play": 0.0, "qb_efficiency": 0.04, "explosive_pass_rate": 0.084, "explosive_run_rate": 0.116,
                "red_zone_efficiency": 0.57},
    "opponent_defense": {"opp_def_epa_per_play": 0.0, "opp_defense_rank": 16.5, "opp_pressure_rate": 0.30,
                         "opp_coverage_grade": 50.0, "opp_explosive_play_prevention": 50.0},
    "game": {}, "qb": {}, "warnings": [],
}
EMPTY = {"offense": {}, "opponent_defense": {}, "qb": {}, "game": {}, "player": {}, "warnings": ["no team features"]}


class PackPageCase(base.MoneyLinesPageCase):
    """The base fixture, with the context builder under the test's control."""

    def build_with(self, factory):
        self.calls = []

        def fake(team, opponent, **kwargs):
            self.calls.append({"team": team, "opponent": opponent, **kwargs})
            return copy.deepcopy(factory(team, opponent))

        patcher = mock.patch.object(nmf, "build_game_context", side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def default_game(self):
        return self.game(5, "NYJ", "BUF")


class PackOnTests(PackPageCase):
    def test_an_elite_home_offence_shows_the_card_and_moves_the_numbers(self):
        self.build_with(lambda team, opp: ELITE if team == "BUF" else AVERAGE)
        app = self.run_page()
        html = base.visible_text(app)
        self.assertIn("Matchup upgrade pack", html)
        self.assertIn("Explosion chance", html)
        self.assertIn("How the pack adjusts this, and its limits", [e.label for e in app.expander])
        plain = self.analysis(self.default_game())
        adjusted = self.analysis(self.default_game(), pack_contexts={"home": ELITE, "away": AVERAGE})
        self.assertGreater(adjusted["scoring"]["home_expected_points"], plain["scoring"]["home_expected_points"])
        tiles = base.stat_tiles(app)
        self.assertTrue(tiles)  # the page rendered its tiles from the adjusted analysis without a traceback

    def test_turning_the_toggle_off_restores_the_base_model_and_hides_the_card(self):
        self.build_with(lambda team, opp: ELITE if team == "BUF" else AVERAGE)
        on = self.run_page()
        self.assertIn("Matchup upgrade pack", base.visible_text(on))
        on.toggle(key="ml_use_pack").set_value(False)
        off = self.refresh(on)
        self.assertNotIn("Matchup upgrade pack", base.visible_text(off))
        plain_tiles = base.stat_tiles(off)
        # the same page with the builder unavailable renders identically
        with mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("down")):
            st_clear = __import__("streamlit").cache_data.clear
            st_clear()
            unavailable = self.run_page()
        self.assertEqual(plain_tiles, base.stat_tiles(unavailable))

    def test_the_toggle_defaults_on(self):
        self.build_with(lambda team, opp: AVERAGE)
        app = self.run_page()
        self.assertTrue(app.toggle(key="ml_use_pack").value)

    def test_a_league_average_matchup_says_so_and_leaves_the_numbers_alone(self):
        self.build_with(lambda team, opp: AVERAGE)
        app = self.run_page()
        self.assertIn("close to league average", base.visible_text(app))
        off_tiles = base.stat_tiles(app)
        with mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("down")):
            __import__("streamlit").cache_data.clear()
            self.assertEqual(off_tiles, base.stat_tiles(self.run_page()))

    def test_contexts_without_features_show_an_honest_empty_state(self):
        self.build_with(lambda team, opp: EMPTY)
        app = self.run_page()
        self.assertIn("No matchup features were available", base.markdown_html(app))

    def test_a_context_builder_failure_is_invisible(self):
        with mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("provider outage")):
            app = self.run_page()
        self.assertNotIn("Matchup upgrade pack", base.visible_text(app))
        self.assertTrue(base.stat_tiles(app))


class VegasSignTests(PackPageCase):
    def test_market_spread_is_converted_to_each_offences_sportsbook_sign(self):
        game = self.default_game()
        spread_home = game["market_spread_home"]  # nflverse sign: positive = HOME favored
        self.build_with(lambda team, opp: AVERAGE)
        self.run_page()
        by_team = {call["team"]: call for call in self.calls}
        self.assertEqual(set(by_team), {"BUF", "NYJ"})
        self.assertEqual(by_team["BUF"]["opponent"], "NYJ")
        self.assertIs(by_team["BUF"]["is_home"], True)
        self.assertIs(by_team["NYJ"]["is_home"], False)
        self.assertEqual(by_team["BUF"]["week"], 5)
        if spread_home is None:
            self.assertIsNone(by_team["BUF"]["vegas_spread"])
            self.assertIsNone(by_team["NYJ"]["vegas_spread"])
        else:
            self.assertAlmostEqual(by_team["BUF"]["vegas_spread"], -spread_home)
            self.assertAlmostEqual(by_team["NYJ"]["vegas_spread"], spread_home)
            self.assertAlmostEqual(by_team["BUF"]["vegas_spread"], -by_team["NYJ"]["vegas_spread"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
