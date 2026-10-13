"""Player Props page: the matchup upgrade pack path (opponent, toggle, signals card, fail-soft). Offline."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_playerprops_page as base  # noqa: E402  (page fixtures; its module-level pack patch is NOT inherited)

from modules import nfl_matchup_features as nmf  # noqa: E402
from modules import nfl_moneyline_analysis as nma  # noqa: E402

BAD_PROTECTION = {
    "offense": {"ol_pass_block_win_rate": 0.62, "ol_continuity": 0.65, "ol_injuries": 2.0},
    "opponent_defense": {"opp_pressure_rate": 0.38, "opp_blitz_rate": 0.34, "opp_pass_rush_win_rate": 0.36,
                         "opp_simulated_pressure_success": 0.33},
    "qb": {"qb_time_to_throw": 3.1, "qb_pressure_epa": -0.8, "qb_sack_avoidance": 0.90},
    "game": {}, "player": {}, "warnings": [], "has_features": True,
}
AVERAGE = {
    "offense": {"team_epa_per_play": 0.0, "qb_efficiency": 0.04, "explosive_pass_rate": 0.084, "explosive_run_rate": 0.116},
    "opponent_defense": {"opp_def_epa_per_play": 0.0, "opp_defense_rank": 16.5, "opp_pressure_rate": 0.30,
                         "opp_coverage_grade": 50.0, "opp_explosive_play_prevention": 50.0},
    "game": {}, "qb": {}, "player": {}, "warnings": [], "has_features": True,
}
EMPTY = {"offense": {}, "opponent_defense": {}, "qb": {}, "game": {}, "player": {}, "warnings": ["no team features"],
         "has_features": False}


def schedule_game(opponent="BUF", *, spread_home=None, total=None, home=False):
    """What nma.games_for_week would return for the selected team's game (the featured team is KC)."""
    return {"home_team": "KC" if home else opponent, "away_team": opponent if home else "KC", "week": 6, "season": 2026,
            "market_spread_home": spread_home, "market_total": total}


class PackPageCase(base.PlayerPropsPageCase):
    def with_schedule(self, game):
        for name, value in (
            ("current_season", 2026),
        ):
            patcher = mock.patch.object(nma, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (
            mock.patch.object(nma, "load_schedule", return_value=object()),
            mock.patch.object(nma, "default_week", return_value=6),
            mock.patch.object(nma, "games_for_week", return_value=[game] if game else []),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def with_context(self, factory):
        self.calls = []

        def fake(team, opponent, **kwargs):
            self.calls.append({"team": team, "opponent": opponent, **kwargs})
            return copy.deepcopy(factory(team, opponent))

        patcher = mock.patch.object(nmf, "build_game_context", side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)


class OpponentDefaultTests(PackPageCase):
    def test_the_opponent_defaults_to_the_teams_game_in_the_schedule(self):
        self.with_schedule(schedule_game("BUF"))
        self.with_context(lambda team, opp: AVERAGE)
        app = self.run_page()
        self.assertEqual(app.selectbox(key="pp_opponent").value, "BUF")
        self.assertTrue(app.toggle(key="pp_use_pack").value)

    def test_without_a_schedule_the_opponent_is_not_set_and_the_pack_says_so(self):
        with mock.patch.object(nma, "load_schedule", side_effect=nma.ScheduleUnavailable("offline")):
            app = self.run_page()
        self.assertEqual(app.selectbox(key="pp_opponent").value, "Not set")
        self.assertIn("Pick an opponent above", base.visible_text(app))

    def test_spread_is_converted_to_the_sportsbook_sign_for_the_selected_team(self):
        # nflverse: positive = HOME favored. KC (away) at BUF with BUF -3.0 -> market_spread_home +3.0 -> KC +3.0
        self.with_schedule(schedule_game("BUF", spread_home=3.0, total=47.5, home=False))
        self.with_context(lambda team, opp: AVERAGE)
        self.run_page()
        call = next(c for c in self.calls if c["team"] == "KC")
        self.assertEqual(call["opponent"], "BUF")
        self.assertEqual(call["vegas_spread"], 3.0)
        self.assertEqual(call["vegas_total"], 47.5)
        self.assertIs(call["is_home"], False)
        self.assertEqual(call["week"], 6)


class PackOnTests(PackPageCase):
    def test_bad_protection_lowers_the_projection_and_shows_the_signals_card(self):
        self.with_schedule(schedule_game("BUF"))
        self.with_context(lambda team, opp: BAD_PROTECTION)
        app = self.run_page()
        text = base.visible_text(app)
        self.assertIn("Matchup signals", " ".join(e.label for e in app.expander))
        self.assertIn("priors", text)
        # the same player/category without the pack renders a higher passing-yards projection
        plain_mean = self.pack_free_mean("Passing Yards")
        player = next(p for p in self.pool if p["player_id"] == base.QB_ID)
        from modules import nfl_prop_analysis as npa

        adjusted = npa.project_stat(player, "Passing Yards", pack_context=BAD_PROTECTION)
        self.assertLess(adjusted["mean"], plain_mean)

    def pack_free_mean(self, category):
        from modules import nfl_prop_analysis as npa

        player = next(p for p in self.pool if p["player_id"] == base.QB_ID)
        return npa.project_stat(player, category)["mean"]

    def test_turning_the_pack_off_shows_no_card_and_the_base_projection(self):
        self.with_schedule(schedule_game("BUF"))
        self.with_context(lambda team, opp: BAD_PROTECTION)
        on = self.run_page()
        self.set_line(on, "250.5")   # the user's own line survives the toggle, so compare on the same line
        on.toggle(key="pp_use_pack").set_value(False)
        off = self.rerun(on)
        self.assertNotIn("Matchup signals", " ".join(e.label for e in off.expander))
        with mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("down")):
            __import__("streamlit").cache_data.clear()
            unavailable = self.run_page()
            self.set_line(unavailable, "250.5")
        self.assertEqual(base.tile_values(off), base.tile_values(unavailable))

    def test_a_neutral_context_says_the_pack_changed_nothing(self):
        self.with_schedule(schedule_game("BUF"))
        self.with_context(lambda team, opp: {**EMPTY, "has_features": True})
        app = self.run_page()
        self.assertIn("close to league average", " ".join(str(m.value) for m in app.markdown))
        self.assertIn("Matchup signals", " ".join(e.label for e in app.expander))

    def test_a_context_with_no_features_is_reported_calmly(self):
        self.with_schedule(schedule_game("BUF"))
        self.with_context(lambda team, opp: EMPTY)
        app = self.run_page()
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertTrue(base.tile_values(app))

    def test_a_context_builder_failure_degrades_to_the_base_model(self):
        self.with_schedule(schedule_game("BUF"))
        with mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("provider outage")):
            app = self.run_page()
        self.assertIn("could not be loaded", base.visible_text(app))
        self.assertTrue(base.tile_values(app))

    def test_choosing_a_different_opponent_drops_the_scheduled_lines(self):
        self.with_schedule(schedule_game("BUF", spread_home=3.0, total=47.5))
        self.with_context(lambda team, opp: AVERAGE)
        app = self.run_page()
        app.selectbox(key="pp_opponent").set_value("NYJ")
        self.rerun(app)
        nyj_call = [c for c in self.calls if c["opponent"] == "NYJ"]
        self.assertTrue(nyj_call)
        self.assertIsNone(nyj_call[-1]["vegas_spread"])
        self.assertIsNone(nyj_call[-1]["week"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
