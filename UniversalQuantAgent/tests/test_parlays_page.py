"""Offline rendering and behaviour contract for the NFL Parlays page (app/pages/parlays.py).

Same pattern as tests/test_betting_engine_page.py: the real Streamlit script runs through
AppTest. The two network-backed loaders (``load_moneyline_inputs`` and ``load_player_pool``) are
monkeypatched on their module objects with a tiny hand-built league BEFORE ``app.run()``, so every
test is fast, deterministic and offline. Only ``LiveSmokeTests`` touches the network, and it skips
itself when the data cannot be loaded.

League: week 4 has one finished game (NE @ KC); week 5 has TB @ DAL (DAL laying 9.5), BUF @ KC,
DEN @ SEA (a neutral-site game) and NYJ @ MIA (no market line at all); week 6 has PHI @ NE
(no line). Every team has a QB / RB / WR in the player pool.
"""

import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FANTASY_ENGINE_ROOT = Path(_PROJECT_ROOT).parent / "fantasy_engine"
if _FANTASY_ENGINE_ROOT.is_dir() and str(_FANTASY_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FANTASY_ENGINE_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import betting_ui as bt
from fantasy import player_status
from modules import nfl_matchup_features as nmf
from modules import nfl_moneyline_analysis as ml
from modules import nfl_parlay_builder as pb
from modules import nfl_prop_analysis as pa

PAGE = Path(_PROJECT_ROOT) / "app" / "pages" / "parlays.py"


_PACK_PATCH = None


def setUpModule():
    """The page's matchup upgrade pack is ON by default and its context builder does I/O. These tests pin the BASE
    model, so the builder is made to fail: the page then runs exactly as it did before the pack existed.
    ``test_nfl_parlays_pack_page.py`` covers the pack path."""
    global _PACK_PATCH
    _PACK_PATCH = mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("pack disabled for base-model tests"))
    _PACK_PATCH.start()


def tearDownModule():
    if _PACK_PATCH is not None:
        _PACK_PATCH.stop()

TB_DAL = "2026_05_TB_DAL"
BUF_KC = "2026_05_BUF_KC"
NYJ_MIA = "2026_05_NYJ_MIA"
DEN_SEA = "2026_05_DEN_SEA"
NE_KC_FINAL = "2026_04_NE_KC"
PHI_NE = "2026_06_PHI_NE"

SHARED_WARNING = "Only ~4 games of 2026 data per team so far; ratings lean on 2025 (current-season weight 40%)."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_RAW_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team", "away_score",
    "home_team", "home_score", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_qb_name", "away_qb_name", "home_qb_id", "away_qb_id", "roof", "temp", "wind", "div_game",
    "home_rest", "away_rest", "location",
]
_TEAM_TAGS = {
    "DAL": "Dallas", "TB": "Tampa", "KC": "Kansas", "BUF": "Buffalo", "MIA": "Miami",
    "NYJ": "Jersey", "SEA": "Seattle", "DEN": "Denver", "NE": "England", "PHI": "Philly",
}


def _raw_game(week, away, home, *, gameday="2026-10-11", gametime="13:00", away_score=None, home_score=None,
              spread=None, total=None, home_ml=None, away_ml=None, home_qb=None, away_qb=None,
              location="Home", div=0, season=2026):
    nan = float("nan")
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "game_type": "REG", "week": week,
        "gameday": gameday, "weekday": "Sunday", "gametime": gametime, "away_team": away,
        "away_score": nan if away_score is None else away_score, "home_team": home,
        "home_score": nan if home_score is None else home_score,
        "spread_line": nan if spread is None else spread, "total_line": nan if total is None else total,
        "home_moneyline": nan if home_ml is None else home_ml, "away_moneyline": nan if away_ml is None else away_ml,
        "home_qb_name": home_qb, "away_qb_name": away_qb, "home_qb_id": None, "away_qb_id": None,
        "roof": "outdoors", "temp": 60.0, "wind": 5.0, "div_game": div, "home_rest": 7, "away_rest": 7,
        "location": location,
    }


def _team_row(*, pf=22.5, pa_=22.5, off=50.0, dfn=50.0, off_epa=0.0, def_epa=0.0, spp=30.4, games=12.0):
    return {"games": games, "n_points": games, "pf": pf, "pa": pa_, "off_score": off, "def_score": dfn,
            "off_epa": off_epa, "def_epa": def_epa, "spp": spp}


def _snapshot(teams):
    def mean(key):
        return float(np.mean([t[key] for t in teams.values()]))

    return {"teams": teams, "scoring_source": "schedule", "league": {
        "off_score": mean("off_score"), "def_score": mean("def_score"), "off_epa": mean("off_epa"),
        "def_epa": mean("def_epa"), "spp": mean("spp"), "pf": mean("pf"), "total_plays_pg": 121.0,
        "games_avg": mean("games"), "points_games_avg": mean("n_points")}}


def make_ratings():
    teams = {
        "BUF": _team_row(pf=30.0, pa_=17.0, off=85, dfn=75, off_epa=0.14, def_epa=-0.09),
        "NYJ": _team_row(pf=16.0, pa_=29.0, off=15, dfn=20, off_epa=-0.12, def_epa=0.10),
        "KC": _team_row(pf=24.0, pa_=21.0, off=65, dfn=55, off_epa=0.05, def_epa=-0.02, spp=27.0),
        "DEN": _team_row(pf=21.0, pa_=20.0, off=45, dfn=60, off_epa=0.0, def_epa=-0.03, spp=33.5),
        "DAL": _team_row(pf=26.0, pa_=20.0, off=70, dfn=55, off_epa=0.08, def_epa=-0.03),
        "TB": _team_row(pf=18.0, pa_=26.0, off=30, dfn=35, off_epa=-0.05, def_epa=0.05),
        "MIA": _team_row(spp=27.0),
        "SEA": _team_row(spp=33.6),
        "NE": _team_row(spp=27.2),
        "PHI": _team_row(pf=22.0, pa_=22.0),
    }
    return ml.blend_ratings(_snapshot(teams), None, season=2026, prior_season=2025)


def make_qb_context():
    qbs = {
        "BUF": ("00-BUF", "Josh Archer", 0.22), "NYJ": ("00-NYJ", "Gus Smith", -0.08), "KC": ("00-KC", "Pat Mahoney", 0.12),
        "DEN": ("00-DEN", "Bo Nixon", 0.05), "DAL": ("00-DAL", "Dan Dallas", 0.15), "TB": ("00-TB", "Tom Tampa", 0.02),
        "MIA": ("00-MIA", "Tua Tagov", 0.03), "SEA": ("00-SEA", "Sam Dar", 0.08), "NE": ("00-NE", "Drake May", 0.10),
        "PHI": ("00-PHI", "Jalen Hurt", 0.04),
    }
    frame = pd.DataFrame([
        {"player_id": pid, "player_display_name": name, "position": "QB", "season_type": "REG", "team": team,
         "week": week, "attempts": 34, "completions": 22, "passing_yards": 238, "passing_tds": 2,
         "passing_interceptions": 1, "sacks_suffered": 2, "passing_epa": epa * 36}
        for team, (pid, name, epa) in qbs.items()
        for week in range(1, 18)
    ])
    return ml.build_qb_context(None, frame, season=2026, prior_season=2025)


def make_inputs():
    rows = [
        _raw_game(4, "NE", "KC", gameday="2026-10-04", away_score=20, home_score=27, spread=7.0, total=44.5,
                  home_ml=-300, away_ml=240, home_qb="Pat Mahoney", away_qb="Drake May"),
        _raw_game(5, "TB", "DAL", spread=9.5, total=47.5, home_ml=-470, away_ml=360, home_qb="Dan Dallas", away_qb="Tom Tampa"),
        _raw_game(5, "BUF", "KC", gameday="2026-10-11", gametime="16:25", spread=2.5, total=51.0, home_ml=-140, away_ml=120,
                  home_qb="Pat Mahoney", away_qb="Josh Archer"),
        _raw_game(5, "NYJ", "MIA", gameday="2026-10-12", gametime="20:15"),
        _raw_game(5, "DEN", "SEA", gameday="2026-10-11", gametime="09:30", spread=1.0, total=41.0, home_ml=-115, away_ml=-105,
                  location="Neutral", home_qb="Sam Dar", away_qb="Bo Nixon"),
        _raw_game(6, "PHI", "NE", gameday="2026-10-18"),
    ]
    schedule = ml.normalize_schedule(pd.DataFrame(rows, columns=_RAW_COLUMNS), 2026)
    return {"season": 2026, "schedule": schedule, "ratings": make_ratings(), "qb_context": make_qb_context(),
            "warnings": [SHARED_WARNING]}


def _history(base):
    return [
        {"week": week, "points": round(base + ((week * 7) % 11 - 5) * 1.5, 1), "targets": 6.0, "carries": 12.0,
         "target_share": 0.2, "air_yards_share": 0.1, "opponent_defense_rank": (week * 5) % 32 + 1}
        for week in range(1, 18)
    ]


def _player(player_id, name, position, team, games=17, **stats):
    row = {
        "player_id": player_id, "name": name, "position": position, "team": team, "season": 2026,
        "prior_season": 2025, "projection_season": 2026, "carries": 0.0, "pass_attempts": 0.0, "passing_tds": 0.0,
        "passing_yards": 0.0, "interceptions": 0.0, "receiving_tds": 0.0, "receiving_yards": 0.0, "receptions": 0.0,
        "rushing_tds": 0.0, "rushing_yards": 0.0, "targets": 0.0, "projection": 240.0, "expected_fantasy_points": 240.0,
        "games_played": games, "prior_games_played": games, "points_per_game": 15.0, "floor": 170.0, "median": 235.0,
        "ceiling": 330.0, "expected_games": 13.5, "projection_confidence": 0.9, "drivers": ["1,000 rushing yards"],
        "projection_basis": "2026 projection from 2025 actuals", "scoring_mode": "ppr", "history": _history(15.0),
        "age": 27.0, "years_experience": 5, "adp": 40.0,
    }
    row.update(stats)
    return row


def make_pool():
    pool = []
    for code, tag in _TEAM_TAGS.items():
        pool.append(_player(f"00-{code}-QB1", f"Quinn {tag}", "QB", code, passing_yards=4100.0, passing_tds=28.0,
                            pass_attempts=540.0, rushing_yards=150.0, rushing_tds=1.0, carries=40.0))
        pool.append(_player(f"00-{code}-RB1", f"Rex {tag}", "RB", code, rushing_yards=1150.0, rushing_tds=8.0,
                            carries=250.0, receptions=45.0, receiving_yards=380.0, receiving_tds=2.0, targets=55.0))
        pool.append(_player(f"00-{code}-WR1", f"Wes {tag}", "WR", code, receiving_yards=1250.0, receiving_tds=9.0,
                            receptions=90.0, targets=135.0))
    return pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def money(amount: float) -> str:
    sign = "+" if amount > 0 else "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def html_of(app) -> str:
    return "\n".join(str(element.value) for element in app.markdown)


def text_of(app) -> str:
    parts = [html_of(app)]
    for group in (app.caption, app.warning, app.info, app.error, app.success, app.text):
        parts.extend(str(element.value) for element in group)
    for expander in app.expander:
        parts.append(str(expander.label))
    return "\n".join(parts)


def legs_of(app) -> list:
    return list(app.session_state["pl_legs"])


def make_leg_from(kind, **kwargs):
    leg = pb.make_parlay_leg(kind, **kwargs)
    leg["ui"] = {"kind": {"moneyline": "Moneyline", "spread": "Spread", "prop": "Prop"}[kind], "short": kwargs["description"],
                 "matchup": "Test game", "week": 5, "played": False, "price_source": "your sportsbook price"}
    return leg


class ParlayPageCase(unittest.TestCase):
    """Base: offline loaders, clean caches, helpers that always re-fetch widgets after a run."""

    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        self.inputs = make_inputs()
        self.pool = make_pool()
        for patcher in (
            mock.patch.object(ml, "load_moneyline_inputs", side_effect=lambda *a, **k: self.inputs),
            mock.patch.object(pa, "load_player_pool", side_effect=lambda *a, **k: self.pool),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_no_missing_text(self, text: str) -> None:
        self.assertEqual(re.findall(r"\b(?:nan|NaN|NAN)\b", text), [], "NaN leaked into the page")
        self.assertEqual(re.findall(r"\bNone\b", text), [], "None leaked into the page")
        self.assertNotIn("<NA>", text)
        self.assertNotIn("inf%", text)

    # -- driving the page ---------------------------------------------------
    def assert_clean(self, app) -> None:
        raised = [str(element.value) for element in app.exception]
        self.assertEqual(raised, [], f"page raised: {raised}")
        errors = [str(element.value) for element in app.error]
        self.assertEqual(errors, [], f"page showed st.error: {errors}")
        session_warnings = [str(w.value) for w in app.warning if "Session State API" in str(w.value)]
        self.assertEqual(session_warnings, [], "widget default / session-state clash")

    def run_page(self):
        app = AppTest.from_file(str(PAGE), default_timeout=120)
        app.run()
        self.assert_clean(app)
        return app

    def step(self, app):
        app.run()
        self.assert_clean(app)
        return app

    def pick_kind(self, app, kind):
        app.segmented_control(key="pl_kind").set_value(kind)
        return self.step(app)

    def pick_game(self, app, game_id, side=None):
        app.selectbox(key="pl_game").set_value(game_id)
        self.step(app)
        if side is not None:
            app.radio(key="pl_side").set_value(side)
            self.step(app)
        return app

    def click(self, app, key):
        app.button(key=key).click()
        return self.step(app)

    def type_into(self, app, key, value):
        app.text_input(key=key).set_value(value)
        return self.step(app)

    def add_moneyline(self, app, game_id, side="home"):
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, game_id, side)
        return self.click(app, "pl_add_ml")

    def add_spread(self, app, game_id, side="home", line=None):
        self.pick_kind(app, "Spread")
        self.pick_game(app, game_id, side)
        if line is not None:
            self.type_into(app, "pl_sp_line", line)
        return self.click(app, "pl_add_spread")

    def add_prop(self, app, team, category=None, side="over", player_id=None):
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value(team)
        self.step(app)
        if player_id is not None:
            app.selectbox(key="pl_pp_player").set_value(player_id)
            self.step(app)
        if category is not None:
            app.selectbox(key="pl_pp_cat").set_value(category)
            self.step(app)
        if side != "over":
            app.radio(key="pl_pp_side").set_value(side)
            self.step(app)
        return self.click(app, "pl_add_prop")

    def game(self, game_id):
        week = int(game_id.split("_")[1])
        return next(g for g in ml.games_for_week(self.inputs["schedule"], week) if g["game_id"] == game_id)

    def analysis(self, game_id, **kwargs):
        return ml.analyze_game(self.game(game_id), self.inputs["ratings"], self.inputs["qb_context"], **kwargs)


# ---------------------------------------------------------------------------
# Rendering, loading, empty states
# ---------------------------------------------------------------------------
class RenderingTests(ParlayPageCase):
    def test_renders_with_fixtures_and_no_exceptions(self):
        app = self.run_page()
        for key in ("pl_kind", "pl_week", "pl_game", "pl_side", "pl_ml_price", "pl_stake", "pl_book"):
            self.assertIsNotNone(
                next((el for group in (app.segmented_control, app.selectbox, app.radio, app.text_input, app.number_input)
                      for el in group if el.key == key), None),
                f"missing widget {key}",
            )
        for key in ("pl_add_ml", "pl_example", "pl_clear"):
            self.assertTrue(any(b.key == key for b in app.button), f"missing button {key}")
        self.assertEqual(app.selectbox(key="pl_week").value, 5, "default week is the first with an unplayed game")
        self.assertEqual(legs_of(app), [])

    def test_header_uses_the_betting_eyebrow_and_page_title(self):
        html = html_of(self.run_page())
        self.assertIn("Parlays", html)
        self.assertIn("Betting", html)
        self.assertIn("Nothing on this page places a bet", html)

    def test_shared_engine_warning_is_a_notice_card(self):
        html = html_of(self.run_page())
        self.assertIn("bt-notice", html)
        self.assertIn(SHARED_WARNING.split(";")[0], html)

    def test_empty_slip_shows_friendly_prompt_and_disables_clear(self):
        app = self.run_page()
        html = html_of(app)
        self.assertIn("Your slip is empty", html)
        self.assertTrue(app.button(key="pl_clear").disabled)
        self.assertNotIn("Risk and confidence", html)
        self.assertNotIn("aria-label=\"Correlation matrix\"", html)

    def test_disclaimer_and_method_expander_present(self):
        app = self.run_page()
        self.assertIn("not financial or wagering advice", html_of(app))
        labels = [str(expander.label) for expander in app.expander]
        self.assertTrue(any("How this is computed" in label for label in labels), labels)

    def test_clicking_the_active_leg_type_again_never_leaves_the_page_without_one(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        app.segmented_control(key="pl_kind").unselect("Spread")
        self.step(app)
        self.assertEqual(app.segmented_control(key="pl_kind").value, "Spread")
        self.assertIn("pl_add_spread", [b.key for b in app.button])

    def test_leg_type_and_game_survive_switching_between_builders(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, BUF_KC, "away")
        self.pick_kind(app, "Player prop")
        self.pick_kind(app, "Spread")
        self.assertEqual(app.selectbox(key="pl_game").value, BUF_KC)
        self.assertEqual(app.radio(key="pl_side").value, "away")

    def test_no_nan_or_none_strings_on_fresh_page(self):
        self.assert_no_missing_text(text_of(self.run_page()))

class OfflineTests(ParlayPageCase):
    def test_schedule_failure_shows_empty_card_and_retry_never_a_traceback(self):
        with mock.patch.object(ml, "load_moneyline_inputs", side_effect=ml.ScheduleUnavailable("offline")):
            app = self.run_page()
            html = html_of(app)
            self.assertIn("Game data is unavailable", html)
            self.assertTrue(any(b.key == "pl_retry_games" for b in app.button))
            self.assertEqual(len(app.selectbox), 0, "no game pickers without a schedule")

    def test_retry_clears_the_caches_and_recovers(self):
        outcomes = [ml.ScheduleUnavailable("offline")]

        def flaky(*_args, **_kwargs):
            if outcomes:
                raise outcomes.pop(0)
            return self.inputs

        with mock.patch.object(ml, "load_moneyline_inputs", side_effect=flaky), \
                mock.patch.object(ml, "clear_caches") as clear_ml, \
                mock.patch.object(pa, "clear_player_pool_cache") as clear_pool:
            app = self.run_page()
            self.assertIn("Game data is unavailable", html_of(app))
            self.click(app, "pl_retry_games")
            clear_ml.assert_called_once()
            clear_pool.assert_called_once()
            self.assertEqual(app.selectbox(key="pl_week").value, 5)
            self.assertNotIn("Game data is unavailable", html_of(app))

    def test_empty_player_pool_shows_retry_and_is_not_cached(self):
        self.pool = []
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        self.assertIn("Player props are unavailable", html_of(app))
        self.assertTrue(any(b.key == "pl_retry_pool" for b in app.button))
        self.assertTrue(app.button(key="pl_add_prop").disabled)
        self.pool = make_pool()  # the outage heals
        self.click(app, "pl_retry_pool")
        self.assertEqual(app.selectbox(key="pl_pp_team").value, "SEA")
        self.assertNotIn("Player props are unavailable", html_of(app))

    def test_game_data_failure_still_lets_props_be_priced_by_team(self):
        with mock.patch.object(ml, "load_moneyline_inputs", side_effect=ml.ScheduleUnavailable("offline")):
            app = self.run_page()
            self.pick_kind(app, "Player prop")
            app.selectbox(key="pl_pp_team").set_value("DAL")
            self.step(app)
            self.click(app, "pl_add_prop")
            leg = legs_of(app)[0]
            self.assertEqual(leg["team"], "DAL")
            self.assertIsNone(leg["game_id"])

    def test_example_without_schedule_explains_instead_of_crashing(self):
        with mock.patch.object(ml, "load_moneyline_inputs", side_effect=ml.ScheduleUnavailable("offline")):
            app = self.run_page()
            self.click(app, "pl_example")
            self.assertEqual(legs_of(app), [])
            # The message used to be popped and then dropped because no Add button (and so no feedback slot)
            # is drawn while the schedule is missing, so the click looked like it did nothing.
            self.assertIn("The example needs the NFL schedule", html_of(app))

    def test_unexpected_loader_failures_are_messages_not_tracebacks(self):
        for failure in (RuntimeError("boom"), KeyError("week"), ValueError("bad frame")):
            with self.subTest(failure=type(failure).__name__), \
                    mock.patch.object(ml, "load_moneyline_inputs", side_effect=failure), \
                    mock.patch.object(pa, "load_player_pool", side_effect=failure):
                st.cache_data.clear()
                app = self.run_page()
                self.assertIn("Game data is unavailable", html_of(app))
                for kind in ("Spread", "Player prop", "Moneyline"):
                    self.pick_kind(app, kind)
                self.pick_kind(app, "Player prop")
                self.assertIn("Player props are unavailable", html_of(app))
                self.click(app, "pl_example")
                self.assertEqual(legs_of(app), [])
                self.assert_no_missing_text(text_of(app))


# ---------------------------------------------------------------------------
# Moneyline legs
# ---------------------------------------------------------------------------
class MoneylineLegTests(ParlayPageCase):
    def test_default_game_card_has_neutral_site_chip_and_consensus_price(self):
        app = self.run_page()
        self.assertEqual(app.selectbox(key="pl_game").value, DEN_SEA)
        html = html_of(app)
        self.assertIn("Neutral site", html)
        self.assertIn("designated home", " ".join(app.radio(key="pl_side").options))
        self.assertIn("nflverse consensus moneyline", html)

    def test_adding_a_moneyline_leg_lands_in_the_slip_with_model_probability(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        legs = legs_of(app)
        self.assertEqual(len(legs), 1)
        leg = legs[0]
        self.assertEqual((leg["leg_type"], leg["team"], leg["game_id"], leg["side"]), ("moneyline", "DAL", TB_DAL, "home"))
        self.assertAlmostEqual(leg["probability"], self.analysis(TB_DAL)["probabilities"]["home_win"], places=6)
        self.assertEqual(leg["price"], -470.0)
        self.assertEqual(leg["ui"]["price_source"], "nflverse consensus moneyline")
        html = html_of(app)
        self.assertIn("DAL moneyline", html)
        self.assertIn("Price: nflverse consensus moneyline", html)
        self.assertIn("added", html.lower())

    def test_price_source_falls_back_pasted_then_consensus_then_model_fair(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, TB_DAL, "away")
        self.assertIn("nflverse consensus moneyline", html_of(app))
        self.type_into(app, "pl_ml_price", "+400")
        self.assertIn("Source: your sportsbook price", html_of(app))
        self.click(app, "pl_add_ml")
        pasted = legs_of(app)[0]
        self.assertEqual((pasted["price"], pasted["ui"]["price_source"]), (400.0, "your sportsbook price"))

        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, NYJ_MIA, "home")  # no market line posted
        html = html_of(app)
        self.assertIn("model fair price", html)
        self.assertIn("No line posted yet", html)
        self.click(app, "pl_add_ml")
        fair = legs_of(app)[0]
        expected = self.analysis(NYJ_MIA)["moneyline"]["home_fair_price"]
        self.assertEqual((fair["price"], fair["ui"]["price_source"]), (float(expected), "model fair price (no vig)"))
        self.assertAlmostEqual(fair["probability"], self.analysis(NYJ_MIA)["probabilities"]["home_win"], places=6)

    def test_selecting_a_game_or_side_changes_what_is_rendered(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, TB_DAL, "home")
        home_html = html_of(app)
        self.assertIn("Dallas Cowboys", home_html)
        self.assertIn("-470", home_html)
        self.pick_game(app, TB_DAL, "away")
        self.assertIn("+360", html_of(app))
        self.pick_game(app, BUF_KC, "home")
        self.assertIn("-140", html_of(app))

    def test_week_selection_changes_the_game_list(self):
        app = self.run_page()
        app.selectbox(key="pl_week").set_value(6)
        self.step(app)
        self.assertEqual(app.selectbox(key="pl_game").value, PHI_NE)
        self.assertEqual(len(app.selectbox(key="pl_game").options), 1)
        self.assertIn("No line posted yet", html_of(app))

    def test_assumed_starter_is_called_out(self):
        # NYJ @ MIA lists no quarterbacks at all, so the model falls back to each team's usual passer.
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, NYJ_MIA, "home")
        html = html_of(app)
        self.assertIn("(assumed starter)", html)
        self.assertIn("no starting quarterback is confirmed yet", html)
        # A game that names its starters carries no such note.
        self.pick_game(app, TB_DAL, "home")
        self.assertNotIn("(assumed starter)", html_of(app))

    def test_finished_game_is_allowed_but_noted(self):
        app = self.run_page()
        app.selectbox(key="pl_week").set_value(4)
        self.step(app)
        self.pick_kind(app, "Moneyline")
        html = html_of(app)
        self.assertIn("already final", html)
        self.click(app, "pl_add_ml")
        legs = legs_of(app)
        self.assertEqual(len(legs), 1)
        self.assertTrue(legs[0]["ui"]["played"])
        self.assertIn("Game already final", html_of(app))

    def test_invalid_price_shows_a_friendly_message_and_blocks_adding(self):
        app = self.run_page()
        self.type_into(app, "pl_ml_price", "abc")
        self.assertIn("Enter odds like -150 or +120", html_of(app))
        self.assertTrue(app.button(key="pl_add_ml").disabled)
        self.type_into(app, "pl_ml_price", "-50")
        self.assertIn("at least 100", html_of(app))
        self.assertTrue(app.button(key="pl_add_ml").disabled)
        self.type_into(app, "pl_ml_price", "1.91")  # decimal odds are accepted
        self.assertFalse(app.button(key="pl_add_ml").disabled)

    def test_game_without_market_line_renders_no_nan(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, NYJ_MIA)
        self.assertIn("No line posted yet", html_of(app))
        self.assert_no_missing_text(text_of(app))


# ---------------------------------------------------------------------------
# Spread legs
# ---------------------------------------------------------------------------
class SpreadLegTests(ParlayPageCase):
    def test_spread_defaults_to_the_market_line_from_the_backed_teams_view(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.assertEqual(app.text_input(key="pl_sp_line").value, "-9.5")
        self.assertEqual(app.text_input(key="pl_sp_price").value, "-110")
        self.pick_game(app, TB_DAL, "away")
        self.assertEqual(app.text_input(key="pl_sp_line").value, "+9.5")

    def test_spread_without_market_defaults_to_model_spread_rounded_to_half_point(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, NYJ_MIA, "home")
        text = app.text_input(key="pl_sp_line").value
        model = self.analysis(NYJ_MIA)["scoring"]["model_spread_home"]
        expected = -(round(model * 2) / 2)
        self.assertAlmostEqual(float(text), expected, places=6)
        self.assertEqual(round(float(text) * 2), float(text) * 2, "a half-point number")
        self.assertIn("No market spread is posted", text_of(app))

    def test_sign_conversion_home_and_away_lines_map_to_the_right_cover_probability(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")

        # DAL -3.5 (home lays 3.5)  ->  nflverse user_spread_home = +3.5.
        self.type_into(app, "pl_sp_line", "-3.5")
        self.click(app, "pl_add_spread")
        home_leg = legs_of(app)[0]
        self.assertEqual(home_leg["line"], -3.5)
        expected_home = self.analysis(TB_DAL, user_spread_home=3.5)["probabilities"]["home_cover"]
        self.assertAlmostEqual(home_leg["probability"], expected_home, places=6)

        # DAL -9.5 is a tougher number than -3.5, so it must be less likely to cover.
        self.type_into(app, "pl_sp_line", "-9.5")
        self.click(app, "pl_clear")
        self.click(app, "pl_add_spread")
        harder = legs_of(app)[0]
        self.assertLess(harder["probability"], home_leg["probability"])

        # TB +3.5 (away gets 3.5)  ->  user_spread_home = +3.5, and TB +9.5 covers more often than TB +3.5.
        self.click(app, "pl_clear")
        self.pick_game(app, TB_DAL, "away")
        self.type_into(app, "pl_sp_line", "+3.5")
        self.click(app, "pl_add_spread")
        away_leg = legs_of(app)[0]
        self.assertEqual((away_leg["team"], away_leg["line"]), ("TB", 3.5))
        expected_away = self.analysis(TB_DAL, user_spread_home=3.5)["probabilities"]["away_cover"]
        self.assertAlmostEqual(away_leg["probability"], expected_away, places=6)
        self.click(app, "pl_clear")
        self.type_into(app, "pl_sp_line", "+9.5")
        self.click(app, "pl_add_spread")
        self.assertGreater(legs_of(app)[0]["probability"], away_leg["probability"])

    def test_opposite_sides_of_the_same_number_cannot_both_be_added(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.click(app, "pl_add_spread")  # DAL -9.5
        self.pick_game(app, TB_DAL, "away")  # TB +9.5: the two can never both cover
        self.assertEqual(app.text_input(key="pl_sp_line").value, "+9.5")
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.assertIn("cannot both cover", html_of(app))
        self.assertEqual([(leg["team"], leg["line"]) for leg in legs_of(app)], [("DAL", -9.5)])

    def test_default_price_is_minus_110_and_blank_means_minus_110(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.type_into(app, "pl_sp_price", "")
        self.assertIn("standard -110", html_of(app))
        self.click(app, "pl_add_spread")
        self.assertEqual(legs_of(app)[0]["price"], -110.0)

    def test_pasted_spread_price_is_used_and_labelled(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.type_into(app, "pl_sp_price", "-105")
        self.assertIn("Source: your sportsbook price", html_of(app))
        self.click(app, "pl_add_spread")
        self.assertEqual(legs_of(app)[0]["price"], -105.0)

    def test_invalid_spread_and_price_are_friendly(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.type_into(app, "pl_sp_line", "abc")
        self.assertIn("Enter the spread as a number", html_of(app))
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.type_into(app, "pl_sp_line", "75")
        self.assertIn("at most 60", html_of(app))
        self.type_into(app, "pl_sp_line", "")
        self.assertIn("Enter the spread to price this leg", text_of(app))
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.type_into(app, "pl_sp_line", "-3.5")
        self.type_into(app, "pl_sp_price", "oops")
        self.assertIn("Enter odds like", html_of(app))
        self.assertTrue(app.button(key="pl_add_spread").disabled)

    def test_whole_number_spread_warns_about_pushes(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.type_into(app, "pl_sp_line", "-3")
        self.assertIn("can push", text_of(app))

    def test_pick_em_spread_is_labelled_pk(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.type_into(app, "pl_sp_line", "0")
        self.click(app, "pl_add_spread")
        self.assertIn("DAL PK", html_of(app))


# ---------------------------------------------------------------------------
# Prop legs
# ---------------------------------------------------------------------------
class PropLegTests(ParlayPageCase):
    def test_adding_a_prop_leg_links_it_to_the_teams_game(self):
        app = self.run_page()
        self.add_prop(app, "DAL", "Passing Yards")
        leg = legs_of(app)[0]
        self.assertEqual((leg["leg_type"], leg["market"], leg["team"], leg["game_id"]), ("prop", "passing_yards", "DAL", TB_DAL))
        self.assertEqual((leg["home_team"], leg["away_team"], leg["opponent"]), ("DAL", "TB", "TB"))
        self.assertEqual(leg["side"], "over")
        player = next(p for p in self.pool if p["player_id"] == "00-DAL-QB1")
        evaluation = pa.evaluate_line(player, "Passing Yards", leg["line"], -110, -110)
        self.assertAlmostEqual(leg["probability"], evaluation["probability_over"], places=6)
        self.assertEqual(leg["price"], -110.0)
        html = html_of(app)
        self.assertIn("Quinn Dallas Over", html)
        self.assertIn("Pass Yds", html)

    def test_under_side_uses_the_under_probability_and_price(self):
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value("DAL")
        self.step(app)
        self.type_into(app, "pl_pp_under", "-125")
        app.radio(key="pl_pp_side").set_value("under")
        self.step(app)
        self.click(app, "pl_add_prop")
        leg = legs_of(app)[0]
        self.assertEqual((leg["side"], leg["price"]), ("under", -125.0))
        player = next(p for p in self.pool if p["player_id"] == "00-DAL-QB1")
        evaluation = pa.evaluate_line(player, "Passing Yards", leg["line"], -110, -125)
        self.assertAlmostEqual(leg["probability"], evaluation["probability_under"], places=6)

    def test_team_player_and_category_choices_change_the_rendered_numbers(self):
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value("DAL")
        self.step(app)
        qb_line = app.text_input(key="pl_pp_line").value
        app.selectbox(key="pl_pp_player").set_value("00-DAL-RB1")
        self.step(app)
        self.assertEqual(app.selectbox(key="pl_pp_cat").value, "Rushing Yards")
        rb_line = app.text_input(key="pl_pp_line").value
        self.assertNotEqual(qb_line, rb_line, "the suggested line follows the player and stat")
        self.assertIn("Rex Dallas", " ".join(app.selectbox(key="pl_pp_player").options) + html_of(app))
        before = html_of(app)
        self.type_into(app, "pl_pp_line", "30.5")
        self.assertNotEqual(before, html_of(app))

    def test_typing_a_line_changes_the_probability(self):
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value("DAL")
        self.step(app)
        self.type_into(app, "pl_pp_line", "200.5")
        high = pa.evaluate_line(next(p for p in self.pool if p["player_id"] == "00-DAL-QB1"), "Passing Yards", 200.5, -110, -110)
        self.assertIn(f"{high['probability_over'] * 100:.1f}%", html_of(app))

    def test_invalid_line_and_price_are_friendly_and_block_adding(self):
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        self.type_into(app, "pl_pp_line", "abc")
        self.assertIn("as a number", html_of(app))
        self.assertTrue(app.button(key="pl_add_prop").disabled)
        self.type_into(app, "pl_pp_line", "250.3")
        self.assertIn("half-point", html_of(app))
        self.assertTrue(app.button(key="pl_add_prop").disabled)
        self.type_into(app, "pl_pp_line", "250.5")
        self.type_into(app, "pl_pp_over", "zzz")
        self.assertIn("Enter odds like", html_of(app))
        self.assertTrue(app.button(key="pl_add_prop").disabled)

    def test_integer_line_warns_about_pushes_but_can_be_added(self):
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        self.type_into(app, "pl_pp_line", "250")
        self.assertIn("Integer line", html_of(app))
        self.assertFalse(app.button(key="pl_add_prop").disabled)

    def test_qb_passing_yards_discloses_the_wide_spread(self):
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        self.assertIn("wider than quarterbacks show in practice", html_of(app))

    def test_player_listed_out_is_badged_and_cannot_be_added(self):
        player_status.STATUS_PATH.write_text(
            json.dumps({"00-SEA-QB1": {"status": "OUT", "last_updated": "2026-10-06T00:00:00Z", "source": "test"}}),
            encoding="utf-8",
        )
        player_status.clear_cache()
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        self.assertEqual(app.selectbox(key="pl_pp_team").value, "SEA")
        self.assertIn("Quinn Seattle (QB) - Out", app.selectbox(key="pl_pp_player").options)
        html = html_of(app)
        self.assertIn("Out", html)
        self.assertIn("is listed OUT", html)
        self.assertTrue(app.button(key="pl_add_prop").disabled)

    def test_questionable_player_is_badged_but_still_addable(self):
        player_status.STATUS_PATH.write_text(
            json.dumps({"00-SEA-QB1": {"status": "QUESTIONABLE", "last_updated": "2026-10-06T00:00:00Z", "source": "test"}}),
            encoding="utf-8",
        )
        player_status.clear_cache()
        app = self.run_page()
        self.pick_kind(app, "Player prop")
        self.assertIn("Questionable", html_of(app))
        self.assertFalse(app.button(key="pl_add_prop").disabled)

    def test_team_on_a_bye_week_cannot_be_added(self):
        app = self.run_page()
        app.selectbox(key="pl_week").set_value(6)
        self.step(app)
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value("KC")
        self.step(app)
        self.assertIn("no game in week 6", html_of(app))
        self.assertTrue(app.button(key="pl_add_prop").disabled)

    def test_anytime_td_is_described_truthfully_and_linked_to_the_team_result(self):
        # The parlay engine used to have no rule for Anytime TD and the page said so. It has one now
        # (team result + player scoring), so the old "no rule ... unrelated to your other legs" notice
        # would be a false statement about the very number the user is looking at.
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value("DAL")
        self.step(app)
        app.selectbox(key="pl_pp_player").set_value("00-DAL-RB1")
        self.step(app)
        app.selectbox(key="pl_pp_cat").set_value("Anytime TD (Rush + Rec)")
        self.step(app)
        html = html_of(app)
        self.assertNotIn("no rule in the correlation table", html)
        self.assertNotIn("unrelated to your other legs", html)
        self.assertIn("rushing and receiving touchdowns", html)
        self.click(app, "pl_add_prop")
        legs = legs_of(app)
        self.assertEqual([leg["market"] for leg in legs], ["moneyline", "anytime_td"])
        result = pb.evaluate_parlay_legs(legs, stake=100.0)
        self.assertEqual([pair["kind"] for pair in result["correlation_pairs"]], ["team_result_and_player_scoring"])
        self.assertGreater(result["adjusted_probability"], result["naive_probability"])
        self.assertIn("Team result and player scoring", html_of(app))


# ---------------------------------------------------------------------------
# Slip management and conflicts
# ---------------------------------------------------------------------------
class SlipTests(ParlayPageCase):
    def test_fewer_than_two_legs_shows_the_prompt_not_a_traceback(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        html = html_of(app)
        self.assertIn("Add at least two legs", html)
        self.assertNotIn("aria-label=\"Correlation matrix\"", html)
        self.assertNotIn("Risk and confidence", html)
        self.assertIn("1 leg", html)

    def test_contradictory_pair_is_blocked_with_a_message_and_stays_out_of_the_slip(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.pick_game(app, TB_DAL, "away")  # TB moneyline against DAL moneyline
        self.assertTrue(app.button(key="pl_add_ml").disabled)
        html = html_of(app)
        self.assertIn("can&#x27;t be combined", html)
        self.assertIn("cannot both win", html)
        self.assertEqual(len(legs_of(app)), 1)
        # Even a click that slips past the disabled state is re-checked by the callback.
        self.click(app, "pl_add_ml")
        self.assertEqual(len(legs_of(app)), 1)
        self.assertIn("can&#x27;t be combined", html_of(app))

    def test_duplicate_leg_is_blocked(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.assertTrue(app.button(key="pl_add_ml").disabled)
        just_added = html_of(app)
        self.assertIn("Added DAL moneyline", just_added)
        self.assertNotIn("already in your parlay", just_added, "the confirmation is not contradicted in the same breath")
        self.step(app)  # any later interaction
        self.assertTrue(app.button(key="pl_add_ml").disabled)
        self.assertIn("already in your parlay", html_of(app))
        self.click(app, "pl_add_ml")
        self.assertEqual(len(legs_of(app)), 1)

    def test_ml_and_opposing_spread_with_exclusive_numbers_is_blocked(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "away")
        self.type_into(app, "pl_sp_line", "-1.5")  # TB -1.5 needs TB to win; DAL must win the moneyline
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.assertEqual(len(legs_of(app)), 1)

    def test_nearly_same_bet_gets_a_heads_up_but_is_allowed(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.assertFalse(app.button(key="pl_add_spread").disabled)
        self.assertIn("nearly the same bet", html_of(app))
        self.click(app, "pl_add_spread")
        self.assertEqual(len(legs_of(app)), 2)

    def test_remove_one_leg_and_clear_the_slip(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_moneyline(app, BUF_KC, "home")
        self.assertEqual(len(legs_of(app)), 2)
        remove_keys = [b.key for b in app.button if str(b.key).startswith("pl_rm_")]
        self.assertEqual(len(remove_keys), 2)
        self.click(app, remove_keys[0])
        remaining = legs_of(app)
        self.assertEqual([leg["team"] for leg in remaining], ["KC"])
        self.assertIn("Add at least two legs", html_of(app))
        self.add_moneyline(app, TB_DAL, "home")
        self.assertEqual(len(legs_of(app)), 2)
        self.assertFalse(app.button(key="pl_clear").disabled)
        self.click(app, "pl_clear")
        self.assertEqual(legs_of(app), [])
        self.assertTrue(app.button(key="pl_clear").disabled)
        self.assertIn("Your slip is empty", html_of(app))

    def test_forty_one_legs_is_a_friendly_message_not_a_traceback(self):
        app = self.run_page()
        legs = [
            make_leg_from("moneyline", description=f"T{i} ML", probability=0.5, price=100, game_id=f"g{i}",
                          team=f"T{i:02d}", home_team=f"T{i:02d}", away_team=f"U{i:02d}")
            for i in range(pb.MAX_PARLAY_LEGS + 1)
        ]
        app.session_state["pl_legs"] = legs
        self.step(app)
        self.assertIn(f"at most {pb.MAX_PARLAY_LEGS} legs", html_of(app))
        self.assertTrue(app.button(key="pl_add_ml").disabled)


# ---------------------------------------------------------------------------
# Pricing the ticket
# ---------------------------------------------------------------------------
class PricingTests(ParlayPageCase):
    def two_independent_legs(self, app):
        self.add_moneyline(app, TB_DAL, "home")
        self.add_moneyline(app, BUF_KC, "home")
        return legs_of(app)

    def test_two_legs_show_probability_ev_correlation_grid_and_a_tier(self):
        app = self.run_page()
        legs = self.two_independent_legs(app)
        html = html_of(app)
        result = pb.evaluate_parlay_legs(legs, stake=100.0)
        for heading in ("Combined probability", "Risk and confidence", "Correlation between legs"):
            self.assertIn(heading, html)
        self.assertIn("aria-label=\"Correlation matrix\"", html)
        self.assertIn(f"{result['adjusted_probability'] * 100:.1f}%", html)
        self.assertIn(money(result["ev_adjusted"]), html)
        self.assertIn(money(result["payout"]), html)
        self.assertRegex(html, r'aria-label="Risk: (Low|Medium|High)"')
        self.assertIn(result["risk_tier"], html)
        self.assertIn("Hit probability", html)
        self.assertIn("Fair price", html)
        self.assertIn("Expected value", html)
        self.assertIn("Edge vs price", html)
        self.assertIn("Confidence", html)
        self.assertIn("Independent", html)
        self.assertIn("Correlation-adjusted", html)

    def test_risk_tier_is_always_low_medium_or_high(self):
        app = self.run_page()
        self.two_independent_legs(app)
        self.add_prop(app, "DAL", "Passing Yards")
        self.add_prop(app, "KC", "Rushing Yards")
        tiers = re.findall(r'aria-label="Risk: ([A-Za-z]+)"', html_of(app))
        self.assertEqual(len(tiers), 1)
        self.assertIn(tiers[0], ("Low", "Medium", "High"))

    def test_positively_correlated_pair_has_adjusted_above_naive(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_spread(app, TB_DAL, "home")  # DAL moneyline + DAL -9.5: covering implies winning
        legs = legs_of(app)
        self.assertEqual(len(legs), 2)
        result = pb.evaluate_parlay_legs(legs, stake=100.0)
        self.assertGreater(result["adjusted_probability"], result["naive_probability"])
        html = html_of(app)
        self.assertIn(f"{result['adjusted_probability'] * 100:.1f}%", html)
        self.assertIn(f"{result['naive_probability'] * 100:.1f}% if the legs were independent", html)
        self.assertIn(f"{result['correlation_lift'] * 100:+.2f} pts", html)
        self.assertIn("SUPPORTS".lower(), html.lower())
        self.assertIn("How your legs connect", html)
        self.assertIn("Same team ML and spread favorite", html)
        self.assertIn("+1.00", html)  # exact nested pair in the grid
        self.assertIn("nearly the same bet", html)

    def test_prop_and_same_team_moneyline_are_linked_by_game(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_prop(app, "DAL", "Rushing Yards", player_id="00-DAL-RB1")
        result = pb.evaluate_parlay_legs(legs_of(app), stake=100.0)
        kinds = [pair["kind"] for pair in result["correlation_pairs"]]
        self.assertEqual(kinds, ["team_result_and_rush_script"])
        self.assertIn("Team result and rush script", html_of(app))
        self.assertGreater(result["adjusted_probability"], result["naive_probability"])

    def test_legs_in_different_games_are_independent(self):
        app = self.run_page()
        self.two_independent_legs(app)
        html = html_of(app)
        self.assertIn("No linked pairs", html)
        result = pb.evaluate_parlay_legs(legs_of(app), stake=100.0)
        self.assertEqual(result["adjusted_probability"], result["naive_probability"])
        self.assertIn("exact product", text_of(app))

    def test_sportsbook_parlay_price_changes_the_ev(self):
        app = self.run_page()
        legs = self.two_independent_legs(app)
        before = pb.evaluate_parlay_legs(legs, stake=100.0)
        self.assertIn("product of your leg prices", html_of(app))
        self.type_into(app, "pl_book", "+300")
        after = pb.evaluate_parlay_legs(legs, stake=100.0, book_price=300)
        self.assertNotEqual(before["ev_adjusted"], after["ev_adjusted"])
        html = html_of(app)
        self.assertIn(money(after["ev_adjusted"]), html)
        self.assertIn(money(after["payout"]), html)
        self.assertIn("your sportsbook quote", html)
        self.assertIn("+300", html)
        # Clearing it goes back to the leg product.
        self.type_into(app, "pl_book", "")
        self.assertIn(money(before["ev_adjusted"]), html_of(app))

    def test_invalid_sportsbook_parlay_price_is_ignored_with_a_message(self):
        app = self.run_page()
        legs = self.two_independent_legs(app)
        self.type_into(app, "pl_book", "abc")
        html = html_of(app)
        self.assertIn("Sportsbook parlay price ignored", html)
        self.assertIn(money(pb.evaluate_parlay_legs(legs, stake=100.0)["ev_adjusted"]), html)
        self.type_into(app, "pl_book", "-50")
        self.assertIn("Sportsbook parlay price ignored", html_of(app))

    def test_stake_scales_ev_and_payout_proportionally(self):
        app = self.run_page()
        legs = self.two_independent_legs(app)
        base = pb.evaluate_parlay_legs(legs, stake=100.0)
        app.number_input(key="pl_stake").set_value(250.0)
        self.step(app)
        scaled = pb.evaluate_parlay_legs(legs, stake=250.0)
        self.assertAlmostEqual(scaled["payout"], base["payout"] * 2.5, places=1)
        self.assertAlmostEqual(scaled["ev_adjusted"], base["ev_adjusted"] * 2.5, delta=0.02)
        html = html_of(app)
        self.assertIn(money(scaled["payout"]), html)
        self.assertIn(money(scaled["ev_adjusted"]), html)
        self.assertIn("Expected value on $250", html)
        self.assertIn("Payout on $250", html)

    def test_adjusted_probability_is_shown_next_to_naive_with_delta_and_engine_cross_check(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_spread(app, TB_DAL, "home")
        result = pb.evaluate_parlay_legs(legs_of(app), stake=100.0)
        text = text_of(app)
        self.assertIn("original parlay engine", text)
        self.assertIn(f"{result['engine_adjusted_probability'] * 100:.1f}%", text)
        self.assertIn("No dependence structure can put this outside", text)

    def test_three_correlated_legs_mention_the_simulation_standard_error(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_prop(app, "DAL", "Rushing Yards", player_id="00-DAL-RB1")
        self.add_prop(app, "DAL", "Receiving Yards", player_id="00-DAL-WR1")
        result = pb.evaluate_parlay_legs(legs_of(app), stake=100.0)
        self.assertEqual(result["probability_method"], "monte_carlo")
        self.assertIn("fixed-seed simulation", text_of(app))
        self.assertIn("standard error", text_of(app))

    def test_engine_warnings_become_notice_cards(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_prop(app, "DAL", "Rushing Yards", player_id="00-DAL-RB1")
        html = html_of(app)
        self.assertIn("All 2 legs come from one game", html)
        self.assertIn("reprice correlated same-game legs", html)

    def test_contradictory_legs_inject_clamp_the_grid_and_flag_it(self):
        app = self.run_page()
        dal = make_leg_from("moneyline", description="DAL moneyline", probability=0.73, price=-470, game_id=TB_DAL,
                            team="DAL", home_team="DAL", away_team="TB")
        tb = make_leg_from("moneyline", description="TB moneyline", probability=0.27, price=360, game_id=TB_DAL,
                           team="TB", home_team="DAL", away_team="TB")
        app.session_state["pl_legs"] = [dal, tb]
        self.step(app)
        html = html_of(app)
        self.assertIn("Contradictory legs", html)
        self.assertIn("cannot both win", html)
        self.assertIn("-1.00", html)
        self.assertNotIn("-2.00", html)
        # The flag names the clashing pair (not every -1.00 cell is a contradiction: exactly opposed legs
        # can also sit at -1) and says what the grid did to the engine's internal -2.
        self.assertIn("DAL moneyline and TB moneyline cannot both win. The engine stores a pair like that as -2", html)
        self.assertIn("clamps it to -1", html)
        # A contradictory pair is not "a strong link at rho -0.95"; that is only the engine's display cap.
        self.assertNotIn("rho -0.95", html)
        self.assertRegex(html, r'aria-label="Risk: High"')

    def test_long_parlay_and_tiny_probability_are_flagged_without_zero_percent(self):
        app = self.run_page()
        legs = [
            make_leg_from("moneyline", description=f"T{i} ML", probability=0.2, price=400, game_id=f"g{i}",
                          team=f"T{i:02d}", home_team=f"T{i:02d}", away_team=f"U{i:02d}")
            for i in range(9)
        ]
        app.session_state["pl_legs"] = legs
        self.step(app)
        html = html_of(app)
        self.assertIn("very long parlay", html)
        self.assertIn("&lt;0.01%", html)
        self.assertNotIn("0.0% if the legs", html)
        self.assertRegex(html, r'aria-label="Risk: High"')

    def test_no_nan_or_none_strings_with_a_full_priced_slip(self):
        app = self.run_page()
        self.add_moneyline(app, NYJ_MIA, "home")  # no market line
        self.add_spread(app, NYJ_MIA, "home")
        self.add_moneyline(app, DEN_SEA, "away")
        self.add_prop(app, "SEA", "Passing Yards")
        self.assert_no_missing_text(text_of(app))


# ---------------------------------------------------------------------------
# Example parlay
# ---------------------------------------------------------------------------
class ExampleParlayTests(ParlayPageCase):
    def test_example_button_builds_a_realistic_three_leg_slip_from_the_data(self):
        app = self.run_page()
        self.click(app, "pl_example")
        legs = legs_of(app)
        self.assertEqual([leg["leg_type"] for leg in legs], ["moneyline", "prop", "spread"])
        self.assertEqual(pb.legs_conflict(legs), [])
        self.assertEqual(legs[0]["game_id"], legs[1]["game_id"], "the prop is a same-game prop of the favorite")
        self.assertNotEqual(legs[0]["game_id"], legs[2]["game_id"], "the spread is in another game")
        self.assertEqual(legs[1]["team"], legs[0]["team"])
        self.assertEqual(legs[1]["side"], "over")
        html = html_of(app)
        self.assertIn("Loaded an example with 3 legs", html)
        self.assertIn("aria-label=\"Correlation matrix\"", html)
        self.assertIn("Combined probability", html)
        self.assertRegex(html, r'aria-label="Risk: (Low|Medium|High)"')

    def test_example_replaces_the_slip_and_can_be_edited(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.click(app, "pl_example")
        self.assertEqual(len(legs_of(app)), 3)
        first = [b.key for b in app.button if str(b.key).startswith("pl_rm_")][0]
        self.click(app, first)
        self.assertEqual(len(legs_of(app)), 2)

    def test_example_without_a_player_pool_falls_back_to_game_legs(self):
        self.pool = []
        app = self.run_page()
        self.click(app, "pl_example")
        legs = legs_of(app)
        self.assertGreaterEqual(len(legs), 2)
        self.assertTrue(all(leg["leg_type"] in ("moneyline", "spread") for leg in legs))
        self.assertIn("game legs only", html_of(app))

    def test_example_has_no_nan_or_none_text(self):
        app = self.run_page()
        self.click(app, "pl_example")
        self.assert_no_missing_text(text_of(app))

    def test_example_skips_players_who_are_out(self):
        out = {
            player["player_id"]: {"status": "OUT", "last_updated": "2026-10-06T00:00:00Z", "source": "test"}
            for player in self.pool
            if player["position"] == "RB"
        }
        player_status.STATUS_PATH.write_text(json.dumps(out), encoding="utf-8")
        player_status.clear_cache()
        app = self.run_page()
        self.click(app, "pl_example")
        props = [leg for leg in legs_of(app) if leg["leg_type"] == "prop"]
        self.assertTrue(props, "a healthy receiver or quarterback stands in for the back who is out")
        for leg in props:
            self.assertNotEqual(leg["position"], "RB")


# ---------------------------------------------------------------------------
# Method notes and source hygiene
# ---------------------------------------------------------------------------
class MethodTests(ParlayPageCase):
    def test_method_expander_discloses_priors_copula_pushes_and_consensus_prices(self):
        text = html_of(self.run_page())
        for phrase in (
            "heuristic prior",
            "not a coefficient fitted",
            "Gaussian copula",
            "Monte Carlo",
            "push",
            "consensus references",
            "reprice correlated same-game parlays",
            f"{ml.WIN_PROB_STDEV:g}-point",
            f"{ml.MARGIN_STDEV:g}-point",
        ):
            self.assertIn(phrase, text, f"method notes should mention {phrase!r}")

    def test_rule_table_lists_every_correlation_kind(self):
        html = html_of(self.run_page())
        rows = pb.correlation_rule_table()
        self.assertEqual(len(rows), len(pb.CORRELATION_KINDS))
        for row in rows:
            self.assertIn(row["prior"].replace("+/-", "+/-"), html)
        self.assertIn("Prior correlation", html)

    def test_page_source_has_no_network_or_banned_layout_patterns(self):
        source = PAGE.read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "container-type"):
            self.assertNotIn(banned, source, msg=f"found {banned}")
        self.assertIn("compact=True", source, "matchup cards in a column must be compact")

    def test_page_does_not_recompute_engine_math(self):
        source = PAGE.read_text(encoding="utf-8")
        for helper in ("def _joint", "norm.cdf", "NormalDist", "erf(", "np.random", "numpy"):
            self.assertNotIn(helper, source, msg=f"page should delegate math, found {helper}")


# ---------------------------------------------------------------------------
# QA regressions: the add flow, typed numbers, the risk bar, honest copy
# ---------------------------------------------------------------------------
class AddFlowTests(ParlayPageCase):
    """Pressing Add must update the slip, the Add button and the results from ONE script pass."""

    def test_page_never_calls_st_rerun(self):
        # The page used to st.rerun() after an add. That drew the slip and every result twice, and Streamlit's
        # AppTest keeps the first pass's elements next to the second pass's (two "Clear slip" buttons, one of them
        # still disabled), so the very next click on Clear silently did nothing.
        self.assertNotIn("st.rerun(", PAGE.read_text(encoding="utf-8"))

    def test_after_an_add_the_whole_page_describes_the_new_slip(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, TB_DAL, "home")
        self.click(app, "pl_add_ml")
        keys = [button.key for button in app.button]
        for once in ("pl_clear", "pl_example", "pl_add_ml"):
            self.assertEqual(keys.count(once), 1, f"{once} drawn more than once: {keys}")
        self.assertEqual(len([key for key in keys if str(key).startswith("pl_rm_")]), 1)
        self.assertFalse(app.button(key="pl_clear").disabled, "Clear slip is live as soon as there is a leg")
        self.assertTrue(app.button(key="pl_add_ml").disabled, "the leg just added cannot be added again")
        self.assertIn("Added DAL moneyline", html_of(app))
        self.assertIn("1 leg", html_of(app))
        self.click(app, "pl_clear")  # works straight away, no intervening interaction needed
        self.assertEqual(legs_of(app), [])
        self.assertTrue(app.button(key="pl_clear").disabled)

    def test_second_add_prices_the_ticket_in_the_same_pass(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_moneyline(app, BUF_KC, "home")
        html = html_of(app)
        self.assertIn("Combined probability", html)
        self.assertIn("2 legs", html)
        self.assertEqual(len([b for b in app.button if str(b.key).startswith("pl_rm_")]), 2)

    def test_a_price_typed_just_before_the_click_is_the_price_that_is_added(self):
        # A text box only commits when it loses focus, so the typed value and the click arrive together.
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, TB_DAL, "away")
        app.text_input(key="pl_ml_price").set_value("+425")
        app.button(key="pl_add_ml").click()
        self.step(app)
        leg = legs_of(app)[0]
        self.assertEqual((leg["team"], leg["price"], leg["ui"]["price_source"]), ("TB", 425.0, "your sportsbook price"))

    def test_a_click_on_an_invalid_box_adds_nothing_and_says_so(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        app.text_input(key="pl_ml_price").set_value("abc")
        app.button(key="pl_add_ml").click()  # AppTest lets a disabled button be clicked; a phone never would
        self.step(app)
        self.assertEqual(legs_of(app), [])
        self.assertIn("Nothing was added", html_of(app))

    def test_remove_then_add_again_round_trips(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.click(app, [b.key for b in app.button if str(b.key).startswith("pl_rm_")][0])
        self.assertEqual(legs_of(app), [])
        self.assertFalse(app.button(key="pl_add_ml").disabled)
        self.click(app, "pl_add_ml")
        self.assertEqual(len(legs_of(app)), 1)


class SpreadEntryTests(ParlayPageCase):
    def setUp(self):
        super().setUp()
        self.app = self.run_page()
        self.pick_kind(self.app, "Spread")
        self.pick_game(self.app, TB_DAL, "home")

    def test_the_sign_of_the_typed_spread_is_spelled_out(self):
        app = self.app
        self.type_into(app, "pl_sp_line", "-3.5")
        self.assertIn("DAL -3.5 means DAL has to win by more than 3.5 points (4 or more)", text_of(app))
        self.type_into(app, "pl_sp_line", "+3.5")
        self.assertIn("DAL +3.5 means DAL covers by winning, or by losing by fewer than 3.5 points (3 or fewer)", text_of(app))
        self.type_into(app, "pl_sp_line", "-3")
        self.assertIn("a win by exactly 3 is a push", text_of(app))
        self.type_into(app, "pl_sp_line", "+7")
        self.assertIn("losing by exactly 7 is a push", text_of(app))
        self.type_into(app, "pl_sp_line", "PK")
        self.assertIn("DAL PK (pick'em): DAL has to win the game outright", text_of(app))
        self.pick_game(app, TB_DAL, "away")
        self.type_into(app, "pl_sp_line", "+9.5")
        self.assertIn("TB +9.5 means TB covers by winning, or by losing by fewer than 9.5 points (9 or fewer)", text_of(app))

    def test_a_number_that_is_not_a_half_point_is_refused_with_a_message(self):
        app = self.app
        for typed in ("-3.3", "-3.25", "+0.7"):
            self.type_into(app, "pl_sp_line", typed)
            self.assertIn("half-point steps", html_of(app), typed)
            self.assertTrue(app.button(key="pl_add_spread").disabled, typed)
            self.assertNotIn("means DAL", text_of(app), "no sentence about a number that cannot exist")
        for typed in ("-3", "-3.5", "+10", "0", "PK", "pick"):
            self.type_into(app, "pl_sp_line", typed)
            self.assertNotIn("half-point steps", html_of(app), typed)
            self.assertFalse(app.button(key="pl_add_spread").disabled, typed)

    def test_typographic_minus_and_pickem_words_are_understood(self):
        app = self.app
        self.type_into(app, "pl_sp_line", "−3.5")  # U+2212 MINUS SIGN, as pasted from a web page
        self.type_into(app, "pl_sp_price", "−105")
        self.click(app, "pl_add_spread")
        leg = legs_of(app)[0]
        self.assertEqual((leg["line"], leg["price"]), (-3.5, -105.0))

    def test_decimal_commas_are_refused_not_misread_as_thousands(self):
        # "-3,5" would parse as -35 and "1,91" as 191: a silent 10x error.
        app = self.app
        self.type_into(app, "pl_sp_line", "-3,5")
        self.assertIn("Use a period for the decimal point in the spread", html_of(app))
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.type_into(app, "pl_sp_line", "-3.5")
        self.type_into(app, "pl_sp_price", "1,91")
        self.assertIn("Use a period for the decimal point in the price", html_of(app))
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.pick_kind(app, "Moneyline")
        self.type_into(app, "pl_ml_price", "1,91")
        self.assertIn("Use a period for the decimal point", html_of(app))
        self.assertTrue(app.button(key="pl_add_ml").disabled)
        self.pick_kind(app, "Player prop")
        self.type_into(app, "pl_pp_line", "249,5")
        self.assertIn("Use a period for the decimal point in the line", html_of(app))
        self.assertTrue(app.button(key="pl_add_prop").disabled)

    def test_huge_prices_are_a_friendly_message(self):
        app = self.app
        self.type_into(app, "pl_sp_price", "+100000000")
        self.assertTrue(app.button(key="pl_add_spread").disabled)
        self.assertIn("no sportsbook posts", html_of(app))
        self.assert_no_missing_text(text_of(app))


class RiskBarTests(ParlayPageCase):
    """The engine's tiers start at 25 and 50; the kit's bar has three equal thirds. The marker must agree with the lit tier."""

    @staticmethod
    def independent_legs(count, probability):
        price = round(-100 * probability / (1 - probability)) if probability > 0.5 else round(100 * (1 - probability) / probability)
        return [
            make_leg_from("moneyline", description=f"T{i} ML", probability=probability, price=price, game_id=f"g{i}",
                          team=f"T{i:02d}", home_team=f"T{i:02d}", away_team=f"U{i:02d}")
            for i in range(count)
        ]

    def marker_and_tier(self, app):
        html = html_of(app)
        tier = re.search(r'aria-label="Risk: (Low|Medium|High)"', html).group(1)
        position = float(re.search(r"--bt-pos:([0-9.]+)%", html).group(1))
        return tier, position

    def test_marker_sits_inside_the_lit_segment_for_every_tier(self):
        thirds = {"Low": (0.0, 100 / 3), "Medium": (100 / 3, 200 / 3), "High": (200 / 3, 100.0)}
        seen = set()
        # (2 legs, 90%) is Low; (3, 60%) and (4, 65%) are Medium (the first sits exactly on the engine's 25.0
        # floor); (3, 50%) sits exactly on the High floor of 50.0; (5, 50%) and (8, 65%) are well into High.
        for count, probability in ((2, 0.9), (3, 0.6), (4, 0.65), (3, 0.5), (5, 0.5), (8, 0.65)):
            with self.subTest(legs=count, probability=probability):
                legs = self.independent_legs(count, probability)
                result = pb.evaluate_parlay_legs(legs, stake=100.0)
                app = self.run_page()
                app.session_state["pl_legs"] = legs
                self.step(app)
                tier, position = self.marker_and_tier(app)
                self.assertEqual(tier, result["risk_tier"])
                low, high = thirds[tier]
                self.assertTrue(low < position < high, f"{tier} tier (score {result['risk_score']}) has its marker at {position:.1f}%")
                seen.add(tier)
                captions = " ".join(str(caption.value) for caption in app.caption)
                self.assertIn(f"Risk score {result['risk_score']:.0f} / 100", captions)
        self.assertEqual(seen, {"Low", "Medium", "High"}, "the cases cover every tier")

    def test_the_true_score_and_tier_cut_offs_are_printed_and_the_kits_own_score_line_is_hidden(self):
        app = self.run_page()
        app.session_state["pl_legs"] = self.independent_legs(4, 0.65)
        self.step(app)
        text = text_of(app)
        self.assertIn(f"Low is below {pb._RISK_MEDIUM_AT:g}", text)
        self.assertIn(f"Medium starts at {pb._RISK_MEDIUM_AT:g}", text)
        self.assertIn(f"High starts at {pb._RISK_HIGH_AT:g}", text)
        # The kit prints "Risk score N / 100" from the stretched bar position, which is not the engine's score,
        # so the page hides that line (and the kit class it targets must still exist for the rule to bite).
        self.assertIn(".bt-risk .bt-risk-score{display:none}", html_of(app))
        self.assertIn("bt-risk-score", bt.BETTING_CSS)

    def test_a_contradictory_ticket_pins_the_marker_at_the_far_end_of_high(self):
        legs = self.independent_legs(2, 0.6)
        legs[1] = make_leg_from("moneyline", description="U00 ML", probability=0.4, price=150, game_id="g0",
                                team="U00", home_team="T00", away_team="U00")
        app = self.run_page()
        app.session_state["pl_legs"] = legs
        self.step(app)
        tier, position = self.marker_and_tier(app)
        self.assertEqual(tier, "High")
        self.assertGreater(position, 90.0)


class HonestCopyTests(ParlayPageCase):
    def test_a_fair_priced_leg_does_not_claim_the_whole_ticket_has_no_edge(self):
        # NYJ @ MIA has no posted line, so its leg is priced at the model's own fair price. Another leg
        # priced off a consensus moneyline can still carry the ticket's EV, so "comes from correlation alone" was wrong.
        app = self.run_page()
        self.add_moneyline(app, NYJ_MIA, "home")
        self.add_moneyline(app, TB_DAL, "home")
        html = html_of(app)
        self.assertIn("priced at the model", html)
        self.assertNotIn("comes from correlation alone", html)
        self.assertIn("not from that leg", html)

    def test_method_notes_describe_which_legs_are_exact_and_which_are_simulated(self):
        html = html_of(self.run_page())
        self.assertIn("Legs with no link multiply exactly", html)
        self.assertIn("a pair of linked legs uses an exact bivariate-normal calculation", html)
        self.assertIn("a group of three or more linked legs uses a fixed-seed Monte Carlo", html)

    def test_the_slip_says_legs_keep_the_numbers_they_were_added_with(self):
        self.assertIn("keeps the probability and price it had when you added it", html_of(self.run_page()))

    def test_contradictory_pairs_are_listed_without_a_made_up_correlation(self):
        app = self.run_page()
        dal = make_leg_from("moneyline", description="DAL moneyline", probability=0.73, price=-470, game_id=TB_DAL,
                            team="DAL", home_team="DAL", away_team="TB")
        tb = make_leg_from("moneyline", description="TB moneyline", probability=0.27, price=360, game_id=TB_DAL,
                           team="TB", home_team="DAL", away_team="TB")
        app.session_state["pl_legs"] = [dal, tb]
        self.step(app)
        html = html_of(app)
        self.assertIn("Cannot both win. Opposing moneylines:", html)
        self.assertNotIn("strong link, rho -0.95", html)


class StaleEntryTests(ParlayPageCase):
    """A number typed for one bet must not travel to another bet, or a quote for one slip to another slip."""

    def test_a_pasted_moneyline_price_does_not_follow_you_to_another_game_or_side(self):
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        self.pick_game(app, TB_DAL, "home")
        self.type_into(app, "pl_ml_price", "-470")
        self.assertIn("Source: your sportsbook price", html_of(app))
        self.pick_game(app, TB_DAL, "away")  # same game, other team: -470 is not TB's price
        self.assertEqual(app.text_input(key="pl_ml_price").value, "")
        self.assertIn("nflverse consensus moneyline", html_of(app))
        self.type_into(app, "pl_ml_price", "+380")
        self.pick_game(app, BUF_KC, "away")
        self.assertEqual(app.text_input(key="pl_ml_price").value, "")
        # ...but it stays put while you keep working on the same bet (adding it, then checking the slip).
        self.type_into(app, "pl_ml_price", "+110")
        self.click(app, "pl_add_ml")
        self.assertEqual(app.text_input(key="pl_ml_price").value, "+110")
        self.assertEqual(legs_of(app)[0]["price"], 110.0)

    def test_spread_and_prop_prices_reset_with_the_bet_they_were_typed_for(self):
        app = self.run_page()
        self.pick_kind(app, "Spread")
        self.pick_game(app, TB_DAL, "home")
        self.type_into(app, "pl_sp_price", "-125")
        self.pick_game(app, BUF_KC, "home")
        self.assertEqual(app.text_input(key="pl_sp_price").value, "-110")
        self.pick_kind(app, "Player prop")
        app.selectbox(key="pl_pp_team").set_value("DAL")
        self.step(app)
        self.type_into(app, "pl_pp_over", "-130")
        self.type_into(app, "pl_pp_under", "+105")
        app.selectbox(key="pl_pp_player").set_value("00-DAL-RB1")
        self.step(app)
        self.assertEqual((app.text_input(key="pl_pp_over").value, app.text_input(key="pl_pp_under").value), ("-110", "-110"))
        self.type_into(app, "pl_pp_over", "-120")
        self.click(app, "pl_add_prop")
        self.assertEqual(legs_of(app)[0]["price"], -120.0)

    def test_a_sportsbook_parlay_quote_is_forgotten_when_the_legs_change(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_moneyline(app, BUF_KC, "home")
        self.type_into(app, "pl_book", "+300")
        self.assertIn("x return, your sportsbook quote", html_of(app))
        self.assertNotIn("was cleared because the legs changed", html_of(app))
        # Other interactions (stake, a plain rerun) keep it: it is still the same slip.
        app.number_input(key="pl_stake").set_value(50.0)
        self.step(app)
        self.assertEqual(app.text_input(key="pl_book").value, "+300")
        # A third leg is a different ticket, so the old quote must not price it.
        self.add_moneyline(app, DEN_SEA, "away")
        self.assertEqual(app.text_input(key="pl_book").value, "")
        html = html_of(app)
        self.assertIn("x return, product of your leg prices", html)
        self.assertNotIn("x return, your sportsbook quote", html)
        self.assertIn("sportsbook parlay price was cleared because the legs changed", html)
        legs = legs_of(app)
        self.assertEqual(len(legs), 3)
        self.assertIn(money(pb.evaluate_parlay_legs(legs, stake=50.0)["ev_adjusted"]), html)
        self.step(app)  # the notice is shown once, not on every rerun
        self.assertNotIn("was cleared because the legs changed", html_of(app))

    def test_removing_a_leg_or_loading_the_example_also_forgets_the_quote(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_moneyline(app, BUF_KC, "home")
        self.type_into(app, "pl_book", "+300")
        self.click(app, [b.key for b in app.button if str(b.key).startswith("pl_rm_")][0])
        self.assertEqual(app.text_input(key="pl_book").value, "")
        self.add_moneyline(app, BUF_KC, "home")
        self.type_into(app, "pl_book", "+300")
        self.click(app, "pl_example")
        self.assertEqual(app.text_input(key="pl_book").value, "")

    def test_nothing_is_announced_when_there_was_no_quote_to_forget(self):
        app = self.run_page()
        self.add_moneyline(app, TB_DAL, "home")
        self.add_moneyline(app, BUF_KC, "home")
        self.assertNotIn("was cleared because the legs changed", html_of(app))


class SignedZeroTests(ParlayPageCase):
    def test_a_fair_priced_leg_never_shows_a_negative_zero_gap(self):
        # A leg priced at the model's own fair price has a value gap of about +/-0.00004, which the shared
        # formatter prints as "-0.0 pts" half the time: a shortfall that is not there.
        app = self.run_page()
        self.pick_kind(app, "Moneyline")
        for side in ("home", "away"):
            self.pick_game(app, NYJ_MIA, side)
            html = html_of(app)
            self.assertIn("model fair price", html)
            self.assertRegex(html, r"(?<![+-])0\.0 pts", side)  # a zero gap carries no sign
            self.assertNotRegex(html, r"-0\.0+ pts", side)

    def test_slip_and_tiles_never_print_minus_zero(self):
        app = self.run_page()
        self.add_moneyline(app, NYJ_MIA, "home")
        self.add_moneyline(app, DEN_SEA, "home")
        html = html_of(app)
        self.assertNotRegex(html, r"-0\.0+ pts|-0\.0+%|-\$0\.00")


class DefaultSelectionTests(ParlayPageCase):
    def test_props_open_on_a_team_that_still_has_a_game_to_play(self):
        # Like the game pickers, the prop team picker must not open on a finished game.
        schedule = self.inputs["schedule"]
        mask = schedule["game_id"] == DEN_SEA  # the earliest week-5 kickoff, so it would be the "first" game
        schedule.loc[mask, "away_score"] = 20.0
        schedule.loc[mask, "home_score"] = 24.0
        schedule.loc[mask, "played"] = True
        app = self.run_page()
        self.assertEqual(app.selectbox(key="pl_game").value, TB_DAL, "the game picker already skips finished games")
        self.pick_kind(app, "Player prop")
        self.assertNotIn(app.selectbox(key="pl_pp_team").value, ("SEA", "DEN"))
        self.assertEqual(app.selectbox(key="pl_pp_team").value, "DAL")


class RandomWalkTests(ParlayPageCase):
    """A seeded random walk over every control: nothing may raise, leak NaN/None, or leave a conflicting slip."""

    def test_random_interactions_keep_the_page_and_the_slip_consistent(self):
        import random

        rng = random.Random(20261008)
        app = self.run_page()

        def has(widget, key):
            return any(element.key == key for element in getattr(app, widget))

        def pick_option(widget, key):
            element = getattr(app, widget)(key=key)
            element.select_index(rng.randrange(len(element.options)))

        steps = {
            "kind": lambda: app.segmented_control(key="pl_kind").set_value(rng.choice(["Moneyline", "Spread", "Player prop"])),
            "week": lambda: has("selectbox", "pl_week") and app.selectbox(key="pl_week").set_value(rng.choice([4, 5, 6])),
            "game": lambda: has("selectbox", "pl_game") and pick_option("selectbox", "pl_game"),
            "side": lambda: has("radio", "pl_side") and app.radio(key="pl_side").set_value(rng.choice(["home", "away"])),
            "ml price": lambda: has("text_input", "pl_ml_price") and app.text_input(key="pl_ml_price").set_value(rng.choice(["", "+150", "-200", "abc", "even", "1.5"])),
            "spread": lambda: has("text_input", "pl_sp_line") and app.text_input(key="pl_sp_line").set_value(rng.choice(["", "-3.5", "+7", "PK", "3", "abc", "-3.3"])),
            "team": lambda: has("selectbox", "pl_pp_team") and app.selectbox(key="pl_pp_team").set_value(rng.choice(["DAL", "KC", "SEA", "MIA", "NE"])),
            "player": lambda: has("selectbox", "pl_pp_player") and pick_option("selectbox", "pl_pp_player"),
            "stat": lambda: has("selectbox", "pl_pp_cat") and pick_option("selectbox", "pl_pp_cat"),
            "prop line": lambda: has("text_input", "pl_pp_line") and app.text_input(key="pl_pp_line").set_value(rng.choice(["", "10.5", "250", "99999", "abc", "60.5"])),
            "prop side": lambda: has("radio", "pl_pp_side") and app.radio(key="pl_pp_side").set_value(rng.choice(["over", "under"])),
            "stake": lambda: app.number_input(key="pl_stake").set_value(rng.choice([1.0, 25.5, 100.0, 5000.0])),
            "book": lambda: app.text_input(key="pl_book").set_value(rng.choice(["", "+500", "-150", "abc", "+99999999999"])),
            "example": lambda: app.button(key="pl_example").click(),
            "clear": lambda: app.button(key="pl_clear").click(),
            "remove": lambda: [b for b in app.button if str(b.key).startswith("pl_rm_")][0].click(),
            "add": lambda: next(b for b in app.button if str(b.key).startswith("pl_add_")).click(),
        }
        names = sorted(steps)
        for number in range(90):
            name = rng.choice(names if number % 7 else ["add", "add", "kind", "game"])
            try:
                steps[name]()
            except IndexError:
                continue  # nothing to remove yet
            app.run()
            self.assert_clean(app)
            self.assert_no_missing_text(text_of(app))
            legs = legs_of(app)
            ids = [leg["leg_id"] for leg in legs]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate leg after step {number} ({name})")
            self.assertEqual(pb.legs_conflict(legs), [], f"conflicting legs after step {number} ({name})")
            self.assertLessEqual(len(legs), pb.MAX_PARLAY_LEGS)


# ---------------------------------------------------------------------------
# Live data (the only tests that touch the network)
# ---------------------------------------------------------------------------
class LiveSmokeTests(unittest.TestCase):
    def test_live_page_loads_and_example_parlay_prices(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        try:
            inputs = ml.load_moneyline_inputs()
            pool = pa.load_player_pool()
        except Exception as error:  # noqa: BLE001 - offline machines skip rather than fail
            self.skipTest(f"live NFL data unavailable: {error}")
        if not pool or inputs.get("schedule") is None or len(inputs["schedule"]) == 0:
            self.skipTest("live NFL data came back empty")
        app = AppTest.from_file(str(PAGE), default_timeout=300)
        app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertIn("Your slip is empty", html_of(app))
        app.button(key="pl_example").click()
        app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        legs = list(app.session_state["pl_legs"])
        self.assertGreaterEqual(len(legs), 2)
        html = html_of(app)
        self.assertIn("Combined probability", html)
        self.assertRegex(html, r'aria-label="Risk: (Low|Medium|High)"')
        self.assertIsNone(re.search(r"\bnan\b", html, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
