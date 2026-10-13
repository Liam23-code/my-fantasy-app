"""Parlays page: the matchup upgrade pack path (toggle, signals, TRAP / MEGA read, fail-soft). Offline."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_parlays_page as base  # noqa: E402  (page fixtures; its module-level pack patch is NOT inherited)

from modules import nfl_matchup_features as nmf  # noqa: E402
from modules import nfl_parlay_builder as pb  # noqa: E402

ADVERSE = {
    "offense": {"ol_pass_block_win_rate": 0.60, "ol_continuity": 0.60, "ol_injuries": 2.5},
    "opponent_defense": {"opp_pressure_rate": 0.40, "opp_blitz_rate": 0.36, "opp_pass_rush_win_rate": 0.38,
                         "opp_simulated_pressure_success": 0.35, "opp_def_epa_per_play": -0.10, "opp_defense_rank": 3},
    "qb": {"qb_time_to_throw": 3.2, "qb_pressure_epa": -0.9, "qb_sack_avoidance": 0.89},
    "game": {}, "player": {}, "warnings": [], "has_features": True,
}
FAVOURABLE = {
    "offense": {"ol_pass_block_win_rate": 0.80, "ol_continuity": 0.96, "ol_injuries": 0.0, "team_epa_per_play": 0.15,
                "explosive_pass_rate": 0.115, "red_zone_efficiency": 0.68},
    "opponent_defense": {"opp_pressure_rate": 0.22, "opp_blitz_rate": 0.18, "opp_pass_rush_win_rate": 0.24,
                         "opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_coverage_grade": 8.0},
    "qb": {"qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96},
    "game": {}, "player": {}, "warnings": [], "has_features": True,
}
EMPTY = {"offense": {}, "opponent_defense": {}, "qb": {}, "game": {}, "player": {}, "warnings": ["none"], "has_features": False}


def stack_legs():
    qb = base.make_leg_from(
        "prop", description="Pat Pocket over 250.5 passing yards", probability=0.5, price=-110, game_id=base.BUF_KC,
        team="KC", opponent="BUF", home_team="KC", away_team="BUF", side="over", line=250.5, market="passing_yards",
        player_name="Pat Pocket", position="QB",
    )
    wr = base.make_leg_from(
        "prop", description="Wes Receiver over 65.5 receiving yards", probability=0.5, price=-110, game_id=base.BUF_KC,
        team="KC", opponent="BUF", home_team="KC", away_team="BUF", side="over", line=65.5, market="receiving_yards",
        player_name="Wes Receiver", position="WR",
    )
    return [qb, wr]


class PackPageCase(base.ParlayPageCase):
    def with_context(self, factory):
        self.calls = []

        def fake(team, opponent, **kwargs):
            self.calls.append({"team": team, "opponent": opponent, **kwargs})
            return copy.deepcopy(factory(team, opponent))

        patcher = mock.patch.object(nmf, "build_game_context", side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_with_stack(self):
        app = self.run_page()
        app.session_state["pl_legs"] = stack_legs()
        return self.step(app)


class PackSectionTests(PackPageCase):
    def test_the_toggle_defaults_on(self):
        self.with_context(lambda team, opp: EMPTY)
        app = self.run_page()
        self.assertTrue(app.toggle(key="pl_use_pack").value)

    def test_an_adverse_stack_is_flagged_a_trap_with_a_strength_read(self):
        self.with_context(lambda team, opp: ADVERSE if team == "KC" else FAVOURABLE)
        app = self.run_with_stack()
        html = base.html_of(app)
        self.assertIn("Matchup upgrade pack", html)
        self.assertIn(">TRAP<", html)
        self.assertIn("/ 100", html)
        self.assertNotIn(">MEGA<", html)
        self.assert_no_missing_text(base.text_of(app))

    def test_a_favourable_stack_is_flagged_mega(self):
        self.with_context(lambda team, opp: FAVOURABLE)
        app = self.run_with_stack()
        html = base.html_of(app)
        self.assertIn(">MEGA<", html)
        self.assertNotIn(">TRAP<", html)

    def test_the_pack_changes_the_priced_correlation_and_the_off_toggle_restores_it(self):
        self.with_context(lambda team, opp: ADVERSE if team == "KC" else FAVOURABLE)
        on = self.run_with_stack()
        on_text = base.html_of(on)
        on.toggle(key="pl_use_pack").set_value(False)
        off = self.step(on)
        off_text = base.html_of(off)
        self.assertNotIn("Matchup upgrade pack", off_text)
        plain = pb.evaluate_parlay_legs(stack_legs())
        self.assertIn(f"{plain['adjusted_probability'] * 100:.1f}%", off_text)
        self.assertNotEqual(on_text, off_text)

    def test_signals_reach_the_engine_only_when_they_exist(self):
        seen = []
        real = pb.evaluate_parlay_legs

        def spy(legs, **kwargs):
            seen.append(sorted(kwargs))
            return real(legs, **kwargs)

        self.with_context(lambda team, opp: EMPTY)
        with mock.patch.object(pb, "evaluate_parlay_legs", side_effect=spy):
            self.run_with_stack()
        self.assertTrue(seen and all("pack_signals" not in kw for kw in seen))
        seen.clear()
        patcher = mock.patch.object(nmf, "build_game_context", side_effect=lambda team, opp, **kw: copy.deepcopy(ADVERSE))
        patcher.start()
        self.addCleanup(patcher.stop)
        st = __import__("streamlit")
        st.cache_data.clear()
        with mock.patch.object(pb, "evaluate_parlay_legs", side_effect=spy):
            self.run_with_stack()
        self.assertTrue(any("pack_signals" in kw for kw in seen))

    def test_no_features_gives_one_quiet_notice_and_the_unchanged_parlay(self):
        self.with_context(lambda team, opp: EMPTY)
        app = self.run_with_stack()
        self.assertIn("No matchup features were available", base.html_of(app))
        self.assertNotIn(">TRAP<", base.html_of(app))

    def test_a_context_builder_failure_is_invisible(self):
        with mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("provider outage")):
            app = self.run_with_stack()
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertIn("No matchup features were available", base.html_of(app))

    def test_a_game_id_without_season_and_week_is_tolerated(self):
        self.with_context(lambda team, opp: ADVERSE)
        app = self.run_page()
        legs = stack_legs()
        for leg in legs:
            leg["game_id"] = "custom-game"
        app.session_state["pl_legs"] = legs
        self.step(app)
        self.assertEqual([str(e.value) for e in app.exception], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
