"""Offline rendering contracts for the focused Fantasy workflow pages.

These run the real Streamlit scripts through ``AppTest`` with the player pool
stubbed out, so they catch the failure modes a unit test on the engine cannot:
a page that imports something that no longer exists, a widget key collision
between the pages that now share session state, or a label that still says the
wrong season.
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app import fantasy_shared
from fantasy.projections import projection_season_label, upcoming_season
from streamlit.testing.v1 import AppTest

PAGES = Path(_PROJECT_ROOT) / "app" / "pages"
TARGET_SEASON = upcoming_season()


def _pool():
    """A realistic, fully-projected pool -- deep enough for a 12-team, 15-round draft."""
    shape = [("RB", 60, 240.0), ("WR", 70, 230.0), ("QB", 25, 300.0), ("TE", 25, 180.0), ("K", 12, 130.0), ("DST", 12, 120.0)]
    players = []
    adp = 1.0
    for position, count, best in shape:
        for index in range(count):
            projection = round(best - index * (best / (count + 4)), 2)
            players.append(
                {
                    "player_id": f"{position.lower()}{index}",
                    "name": f"{position} Player {index}",
                    "position": position,
                    "team": ["SF", "KC", "BUF", "DAL"][index % 4],
                    "season": TARGET_SEASON,
                    "projection": projection,
                    "expected_fantasy_points": projection,
                    "points_per_game": round(projection / 16, 2),
                    "games_played": 16,
                    "expected_games": 15.6,
                    "floor": round(projection * 0.7, 2),
                    "median": round(projection * 0.95, 2),
                    "ceiling": round(projection * 1.35, 2),
                    "adp": adp,
                    "scoring_mode": "ppr",
                    "projection_season": TARGET_SEASON,
                    "prior_season": TARGET_SEASON - 1,
                    "projection_basis": f"{TARGET_SEASON} projection from {TARGET_SEASON - 1} actuals",
                }
            )
            adp += 1.0
    players.sort(key=lambda player: player["projection"], reverse=True)
    return players


class FantasyPageContracts(unittest.TestCase):
    """Each page must render clean, offline, from a cold session."""

    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()
        cls._real_load_pool = fantasy_shared.load_pool
        # Replaces the cached, network-backed loader outright, so no test here
        # ever reaches nflverse.
        fantasy_shared.load_pool = lambda *_args, **_kwargs: (list(cls.pool), "stubbed test pool")

    @classmethod
    def tearDownClass(cls):
        fantasy_shared.load_pool = cls._real_load_pool

    def _run(self, filename, timeout=90):
        app = AppTest.from_file(str(PAGES / filename), default_timeout=timeout)
        app.run()
        self._assert_clean(app, filename)
        return app

    def _assert_clean(self, app, label):
        """`AppTest.exception` is an (often empty) ElementList, never None."""
        raised = [str(element.value) for element in app.exception]
        self.assertEqual(raised, [], msg=f"{label} raised: {raised}")

    def _text(self, app):
        blocks = [str(element.value) for element in app.markdown] + [str(element.value) for element in app.caption]
        blocks += [str(element.value) for element in app.success] + [str(element.value) for element in app.info]
        return "\n".join(blocks)

    # --- each page renders ---------------------------------------------------

    def test_draft_room_renders_and_offers_a_draft(self):
        app = self._run("25_Fantasy_Draft_Room.py")
        self.assertTrue(any(button.key == "fantasy_start_live_draft" for button in app.button))
        self.assertIn("Mock Draft", self._text(app) + " ".join(h.value for h in app.header))

    def test_draft_assistant_renders(self):
        app = self._run("26_Fantasy_Draft_Assistant.py")
        self.assertTrue(any(widget.key == "fantasy_assistant_round" for widget in app.number_input))

    def test_season_tools_renders(self):
        app = self._run("27_Fantasy_Season_Tools.py")
        self.assertTrue(any(widget.key == "fantasy_scoring_mode" for widget in app.selectbox))

    # --- the 2026 labelling contract ----------------------------------------

    def test_every_page_labels_its_numbers_with_the_projected_season(self):
        label = projection_season_label(TARGET_SEASON)
        for filename in (
            "25_Fantasy_Draft_Room.py",
            "26_Fantasy_Draft_Assistant.py",
            "27_Fantasy_Season_Tools.py",
        ):
            with self.subTest(page=filename):
                self.assertIn(label, self._text(self._run(filename)))

    def test_the_season_input_is_labelled_as_the_source_of_actuals(self):
        app = self._run("25_Fantasy_Draft_Room.py")
        season_input = next(widget for widget in app.number_input if widget.key == "fantasy_source_season")
        self.assertIn("Source season", season_input.label)
        self.assertEqual(int(season_input.value) + 1, TARGET_SEASON)

    # --- the live draft actually starts, and grades -------------------------

    def test_starting_a_live_draft_produces_a_board_and_a_team_grade(self):
        app = self._run("25_Fantasy_Draft_Room.py")
        app.button(key="fantasy_start_live_draft").click().run(timeout=120)
        self._assert_clean(app, "starting a live draft")

        live = app.session_state["fantasy_live_draft"]
        self.assertTrue(live["awaiting_user_pick"] or live["is_complete"])
        text = self._text(app)
        self.assertIn("Team grade", text)
        self.assertIn("Positional strength", text)

    def test_a_finished_draft_renders_the_full_report(self):
        """The post-draft view: positional chart, best/worst pick, ADP value."""
        from fantasy.live_draft import (
            draft_for_user,
            start_live_draft,
            user_turn_context,
        )

        settings = dict(fantasy_shared.DEFAULT_LEAGUE_SETTINGS)
        state = start_live_draft(
            list(self.pool), settings, num_teams=12, num_rounds=3, user_draft_slot=1, seed=11
        )
        while not state["is_complete"]:
            context = user_turn_context(state)
            state = draft_for_user(state, context["board"][0]["player_id"])

        app = AppTest.from_file(str(PAGES / "25_Fantasy_Draft_Room.py"), default_timeout=120)
        app.session_state["fantasy_live_draft"] = state
        app.run()
        self._assert_clean(app, "a finished draft")

        text = self._text(app)
        self.assertIn("Final draft report", text)
        self.assertIn("Positional strength", text)
        self.assertIn("Best and worst picks", text)
        metric_labels = {metric.label for metric in app.metric}
        self.assertIn("Best pick", metric_labels)
        self.assertIn("Worst pick", metric_labels)
        self.assertIn("Value gained vs ADP", metric_labels)

    def test_the_override_control_is_offered_for_players_the_room_has_taken(self):
        app = self._run("25_Fantasy_Draft_Room.py")
        # Slot 4 so the room has already made three picks by the user's turn.
        app.number_input(key="fantasy_draft_pick").set_value(4).run(timeout=120)
        app.button(key="fantasy_start_live_draft").click().run(timeout=120)
        self._assert_clean(app, "starting a live draft on slot 4")

        live = app.session_state["fantasy_live_draft"]
        self.assertGreater(len(live["picks"]), 0)
        override_buttons = [button for button in app.button if str(button.key).startswith("fantasy_override_")]
        self.assertTrue(override_buttons, "no 'Override and Draft' control was rendered")
        self.assertTrue(all(button.label == "Override and Draft" for button in override_buttons))


if __name__ == "__main__":
    unittest.main()
