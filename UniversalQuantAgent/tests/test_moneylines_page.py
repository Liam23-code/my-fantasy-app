"""Page tests for the NFL MoneyLines page (``app/pages/moneylines.py``).

Same pattern as ``tests/test_betting_engine_page.py``: the real Streamlit script
runs through ``AppTest``, so a broken import, a widget-key collision or a crash
is caught here instead of in the browser. Everything is OFFLINE and fast: the
one network-backed loader (``nfl_moneyline_analysis.load_moneyline_inputs``) is
replaced with a hand-built, deterministic schedule / ratings / quarterback
fixture before each run. The page re-executes its imports on every run, so the
patched attribute is what it sees. ``st.cache_data`` is cleared around every
test so a cached result can never leak from one test (or fixture) to the next.

Exactly one test at the bottom (``LiveSmokeTest``) loads the real nflverse data
and skips itself when that is not possible.
"""
from __future__ import annotations

import html as html_lib
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FANTASY_ENGINE_ROOT = Path(_PROJECT_ROOT).resolve().parents[0] / "fantasy_engine"
if _FANTASY_ENGINE_ROOT.is_dir() and str(_FANTASY_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FANTASY_ENGINE_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from modules import nfl_matchup_features as nmf
from modules import nfl_moneyline_analysis as nma
from modules import odds_input as oi

PAGE = Path(_PROJECT_ROOT) / "app" / "pages" / "moneylines.py"

# --------------------------------------------------------------------------
# Fixtures: a small, hand-built 2026 season (no network anywhere)
# --------------------------------------------------------------------------
_RAW_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team", "away_score",
    "home_team", "home_score", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_qb_name", "away_qb_name", "home_qb_id", "away_qb_id", "roof", "temp", "wind", "div_game",
    "home_rest", "away_rest", "location",
]
_NAN = float("nan")



_PACK_PATCH = None


def setUpModule():
    """The page's matchup upgrade pack is ON by default; its context builder does I/O. These tests pin the base
    model, so the builder is made to fail (the page then quietly runs without the pack, exactly as it did before
    the pack existed). ``test_nfl_moneyline_pack_page.py`` exercises the pack path."""
    global _PACK_PATCH
    _PACK_PATCH = mock.patch.object(nmf, "build_game_context", side_effect=RuntimeError("pack disabled for base-model tests"))
    _PACK_PATCH.start()


def tearDownModule():
    if _PACK_PATCH is not None:
        _PACK_PATCH.stop()

def raw_game(week, away, home, *, gameday, gametime="13:00", away_score=None, home_score=None, spread=None,
             total=None, home_ml=None, away_ml=None, home_qb=None, away_qb=None, roof="outdoors", wind=5.0,
             temp=60.0, div=0, location="Home"):
    return {
        "game_id": f"2026_{week:02d}_{away}_{home}", "season": 2026, "game_type": "REG", "week": week,
        "gameday": gameday, "weekday": "Sunday", "gametime": gametime, "away_team": away,
        "away_score": _NAN if away_score is None else away_score, "home_team": home,
        "home_score": _NAN if home_score is None else home_score,
        "spread_line": _NAN if spread is None else spread, "total_line": _NAN if total is None else total,
        "home_moneyline": _NAN if home_ml is None else home_ml, "away_moneyline": _NAN if away_ml is None else away_ml,
        "home_qb_name": home_qb, "away_qb_name": away_qb, "home_qb_id": None, "away_qb_id": None,
        "roof": roof, "temp": temp, "wind": wind, "div_game": div, "home_rest": 7, "away_rest": 7,
        "location": location,
    }


def build_schedule() -> pd.DataFrame:
    """Weeks 4-7. Week 4 is fully played, week 5 has one final and three upcoming games, 6 has a neutral-site game."""
    rows = [
        raw_game(4, "MIA", "BUF", gameday="2026-10-04", away_score=17, home_score=30, spread=6.5, total=45.5,
                 home_ml=-280, away_ml=230, home_qb="Josh Allen", away_qb="Tua Tagovailoa"),
        raw_game(4, "PHI", "DAL", gameday="2026-10-04", gametime="16:25", away_score=21, home_score=24, spread=1.5,
                 total=47.0, home_ml=-125, away_ml=105, home_qb="Dak Prescott", away_qb="Jalen Hurts"),
        # Week 5: a played Thursday game first, then three upcoming games (one with no line, no listed starters).
        raw_game(5, "TEN", "NE", gameday="2026-10-08", gametime="20:15", away_score=17, home_score=27, spread=3.0,
                 total=41.5, home_ml=-160, away_ml=135, home_qb="Drake Maye", away_qb="Cam Ward"),
        raw_game(5, "NYJ", "BUF", gameday="2026-10-11", spread=9.5, total=44.5, home_ml=-450, away_ml=350,
                 home_qb="Josh Allen", away_qb="Geno Smith", div=1),
        raw_game(5, "KC", "DEN", gameday="2026-10-11", gametime="16:25", spread=-2.5, total=47.5, home_ml=120,
                 away_ml=-140, home_qb="Bo Nix", away_qb="Patrick Mahomes"),
        raw_game(5, "DAL", "PHI", gameday="2026-10-11", gametime="20:20", roof="dome"),
        # Week 6: a neutral-site game with a line and a home game with no line at all.
        raw_game(6, "SEA", "MIA", gameday="2026-10-18", gametime="09:30", spread=3.0, total=41.0, home_ml=-165,
                 away_ml=140, home_qb="Tua Tagovailoa", away_qb="Sam Darnold", location="Neutral"),
        raw_game(6, "DEN", "KC", gameday="2026-10-18", gametime="16:25", home_qb="Patrick Mahomes", away_qb="Bo Nix"),
        raw_game(7, "BUF", "NYJ", gameday="2026-10-25"),
    ]
    return nma.normalize_schedule(pd.DataFrame(rows, columns=_RAW_COLUMNS), 2026)


def _team_row(*, pf=22.5, pa=22.5, off=50.0, dfn=50.0, off_epa=0.0, def_epa=0.0, spp=30.4, games=12.0):
    return {"games": games, "n_points": games, "pf": pf, "pa": pa, "off_score": off, "def_score": dfn,
            "off_epa": off_epa, "def_epa": def_epa, "spp": spp}


def build_ratings() -> dict:
    teams = {
        "BUF": _team_row(pf=30.0, pa=17.0, off=85, dfn=75, off_epa=0.14, def_epa=-0.09),
        "NYJ": _team_row(pf=16.0, pa=29.0, off=15, dfn=20, off_epa=-0.12, def_epa=0.10),
        "KC": _team_row(pf=24.0, pa=21.0, off=65, dfn=55, off_epa=0.05, def_epa=-0.02, spp=27.0),
        "DEN": _team_row(pf=21.0, pa=20.0, off=45, dfn=60, off_epa=0.0, def_epa=-0.03, spp=33.5),
        "DAL": _team_row(pf=23.0, pa=23.0),
        "PHI": _team_row(pf=22.0, pa=22.0),
        "MIA": _team_row(spp=27.0),
        "NE": _team_row(spp=27.2),
        "TEN": _team_row(spp=33.8),
        "SEA": _team_row(spp=33.6),
    }

    def mean(key):
        return float(np.mean([t[key] for t in teams.values()]))

    snapshot = {
        "teams": teams, "scoring_source": "schedule",
        "league": {
            "off_score": mean("off_score"), "def_score": mean("def_score"), "off_epa": mean("off_epa"),
            "def_epa": mean("def_epa"), "spp": mean("spp"), "pf": mean("pf"), "total_plays_pg": 121.0,
            "games_avg": mean("games"), "points_games_avg": mean("n_points"),
        },
    }
    return nma.blend_ratings(snapshot, None, season=2026, prior_season=2025)


def build_qb_context() -> dict:
    qbs = {
        "BUF": ("00-BUF", "Josh Allen", 0.22), "NYJ": ("00-NYJ", "Geno Smith", -0.08),
        "KC": ("00-KC", "Patrick Mahomes", 0.12), "DEN": ("00-DEN", "Bo Nix", 0.05),
        "DAL": ("00-DAL", "Dak Prescott", 0.15), "PHI": ("00-PHI", "Jalen Hurts", 0.04),
        "MIA": ("00-MIA", "Tua Tagovailoa", 0.03), "NE": ("00-NE", "Drake Maye", 0.10),
        "TEN": ("00-TEN", "Cam Ward", -0.12), "SEA": ("00-SEA", "Sam Darnold", 0.08),
    }
    rows = []
    for team, (pid, name, epa_db) in qbs.items():
        for week in range(1, 18):
            rows.append({
                "player_id": pid, "player_display_name": name, "position": "QB", "season_type": "REG", "team": team,
                "week": week, "attempts": 34, "completions": 21, "passing_yards": 238, "passing_tds": 2,
                "passing_interceptions": 1, "sacks_suffered": 2, "passing_epa": epa_db * 36,
            })
    prior = pd.DataFrame(rows)
    return nma.build_qb_context(None, prior, season=2026, prior_season=2025)


def fixture_inputs() -> dict:
    return {
        "season": 2026, "schedule": build_schedule(), "ratings": build_ratings(), "qb_context": build_qb_context(),
        "warnings": [],
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_NAN_WORD = re.compile(r"(?<![\w.])-?nan(?![\w])", re.IGNORECASE)
_NONE_WORD = re.compile(r"\bNone\b")
_INF_WORD = re.compile(r"(?<![\w.])[+-]?(?:inf|infinity)(?![\w])", re.IGNORECASE)


def markdown_html(app: AppTest) -> str:
    """Every rendered markdown block EXCEPT the injected ``<style>`` sheets (which name every class on purpose)."""
    joined = "\n".join(str(e.value) for e in app.markdown if not str(e.value).lstrip().startswith("<style"))
    return html_lib.unescape(joined)  # the kit escapes quotes; assertions read the text the user sees


def visible_text(app: AppTest) -> str:
    """Every string the page renders through markdown, captions, metrics, text, warnings and errors."""
    parts = [markdown_html(app)]
    parts += [str(e.value) for e in app.caption]
    parts += [str(e.value) for e in app.text]
    parts += [str(e.value) for e in app.warning]
    parts += [str(e.value) for e in app.error]
    parts += [str(e.value) for e in app.info]
    parts += [str(e.value) for e in app.metric]
    return "\n".join(parts)


def tile_htmls(app: AppTest) -> list[str]:
    return [str(m.value) for m in app.markdown if str(m.value).startswith('<div class="bt bt-card bt-lift bt-tile')]


def selected_tile_codes(app: AppTest) -> list[tuple[str, str]]:
    found = []
    for html in tile_htmls(app):
        if "bt-selected" in html.split(">", 1)[0]:
            away, home = re.findall(r'bt-tile-code">([A-Z]+)<', html)[:2]
            found.append((away, home))
    return found


def hero_html(app: AppTest) -> str:
    return next(str(m.value) for m in app.markdown if str(m.value).startswith('<div class="bt bt-card bt-lift bt-matchup'))


def stat_tiles(app: AppTest) -> dict[str, str]:
    """``{label: value}`` of every stat tile on the page."""
    return {
        label: value
        for label, value in re.findall(r'bt-stat-label">([^<]*)</div><div class="bt-stat-value">([^<]*)<', markdown_html(app))
    }


def stat_subs(app: AppTest) -> dict[str, str]:
    """``{label: sub-caption}`` of every stat tile on the page."""
    return dict(
        re.findall(
            r'bt-stat-label">([^<]*)</div><div class="bt-stat-value">[^<]*</div><div class="bt-stat-sub">([^<]*)<',
            markdown_html(app),
        )
    )


def error_notice_texts(app: AppTest) -> list[str]:
    """Text of every red (error) notice card on the page."""
    return re.findall(r'class="bt bt-notice bt-t-bad" role="note"><div class="bt-notice-text">([^<]*)<', markdown_html(app))


def verdict_heads(app: AppTest) -> list[str]:
    return re.findall(r'bt-v-head">([^<]*)<', markdown_html(app))


class MoneyLinesPageCase(unittest.TestCase):
    """Base class: patches the loader with the fixture and keeps ``st.cache_data`` clean."""

    def setUp(self):
        st.cache_data.clear()
        self.inputs = fixture_inputs()
        patcher = mock.patch.object(nma, "load_moneyline_inputs", side_effect=lambda season=None, today=None: self.inputs)
        self.loader = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(st.cache_data.clear)
        self.schedule = self.inputs["schedule"]
        self.ratings = self.inputs["ratings"]
        self.qb_context = self.inputs["qb_context"]

    # -- running -----------------------------------------------------------
    def run_page(self) -> AppTest:
        app = AppTest.from_file(str(PAGE), default_timeout=120)
        app.run()
        self.assert_clean(app)
        return app

    def assert_clean(self, app: AppTest) -> None:
        self.assertEqual([str(e.value) for e in app.exception], [], "the page raised")
        # The page never uses st.warning / st.error itself, so any that appear are Streamlit's own complaints
        # (e.g. a widget given both a default value and a session-state value).
        self.assertEqual([str(w.value) for w in app.warning], [])
        self.assertEqual([str(e.value) for e in app.error], [])

    def refresh(self, app: AppTest) -> AppTest:
        app.run()
        self.assert_clean(app)
        return app

    # -- expected values straight from the engine ----------------------------
    def games(self, week: int) -> list[dict]:
        return nma.games_for_week(self.schedule, week)

    def game(self, week: int, away: str, home: str) -> dict:
        return next(g for g in self.games(week) if g["away_team"] == away and g["home_team"] == home)

    def analysis(self, game: dict, **kwargs) -> dict:
        return nma.analyze_game(game, self.ratings, self.qb_context, **kwargs)

    def pick(self, app: AppTest, game: dict) -> AppTest:
        app.button(key=f"ml_pick_{game['game_id']}").click()
        return self.refresh(app)

    def type_into(self, app: AppTest, key: str, text: str) -> AppTest:
        app.text_input(key=key).input(text)
        return self.refresh(app)


# --------------------------------------------------------------------------
# Rendering, week and game selection
# --------------------------------------------------------------------------
class RenderingTests(MoneyLinesPageCase):
    def test_renders_with_no_exceptions_and_every_section(self):
        app = self.run_page()
        html = markdown_html(app)
        for heading in ("Price check", "Model breakdown", "Why the model leans this way", "Model vs. market"):
            self.assertIn(heading, html)
        self.assertIn("Research and educational analysis only", html)  # the shared disclaimer ends the page
        self.assertEqual([e.value for e in app.error], [])

    def test_inputs_are_loaded_once_per_cache_window(self):
        app = self.run_page()
        self.assertEqual(self.loader.call_count, 1)
        self.type_into(app, "ml_price", "-150")
        self.assertEqual(self.loader.call_count, 1, "typing a price must not reload the schedule and ratings")

    def test_a_degraded_load_is_asked_for_again_instead_of_being_pinned_for_an_hour(self):
        # The engine remembers a fail-soft (warning-carrying) load for about a minute so a network blip heals itself;
        # st.cache_data(ttl=3600) must not freeze that degraded result for the whole hour.
        healthy = self.inputs
        self.inputs = {**healthy, "warnings": ["2026 play-by-play unavailable (URLError); only 2026 scoring is used."]}
        app = self.run_page()
        self.assertEqual(self.loader.call_count, 1)
        self.type_into(app, "ml_price", "-150")
        self.assertEqual(self.loader.call_count, 2, "a degraded load must be re-requested on the next run")
        self.type_into(app, "ml_price", "-140")
        self.assertEqual(self.loader.call_count, 3)
        self.inputs = healthy  # the data source recovered
        self.type_into(app, "ml_price", "-130")
        self.assertEqual(self.loader.call_count, 4)
        self.type_into(app, "ml_price", "-120")
        self.assertEqual(self.loader.call_count, 4, "once healthy, the result is cached again")

    def test_page_uses_the_midnight_circuit_kit_and_page_header(self):
        html = markdown_html(self.run_page())
        for marker in ("bt-matchup", "bt-grid", "bt-tablewrap", "bt-panel", "bt-meter", "page-head", "Betting · NFL"):
            self.assertIn(marker, html)

    def test_no_nan_or_none_text_anywhere_in_any_game_state(self):
        # Every distinct situation the fixture covers: lined, unlined, final, neutral-site, assumed starters.
        for week, away, home in (
            (5, "NYJ", "BUF"), (5, "TEN", "NE"), (5, "DAL", "PHI"), (6, "SEA", "MIA"), (6, "DEN", "KC"), (7, "BUF", "NYJ"),
        ):
            with self.subTest(game=f"{away}@{home}"):
                app = self.run_page()
                if week != 5:
                    app.selectbox(key="ml_week").set_value(week)
                    self.refresh(app)
                self.pick(app, self.game(week, away, home))
                self.type_into(app, "ml_price", "-150")
                self.type_into(app, "ml_price_other", "+130")
                self.type_into(app, "ml_spread", "-3.5")
                text = visible_text(app)
                for pattern in (_NAN_WORD, _NONE_WORD, _INF_WORD):
                    match = pattern.search(text)
                    self.assertIsNone(match, match and text[max(0, match.start() - 60) : match.end() + 40])

    def test_page_source_has_no_network_calls_at_import_time(self):
        source = PAGE.read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "nflreadpy", "sportsbook.draftkings"):
            self.assertNotIn(banned, source)


class WeekSelectionTests(MoneyLinesPageCase):
    def test_week_selectbox_lists_every_week_and_defaults_to_first_unplayed(self):
        app = self.run_page()
        box = app.selectbox(key="ml_week")
        self.assertEqual(list(box.options), ["Week 4", "Week 5", "Week 6", "Week 7"])
        self.assertEqual(box.value, 5)  # week 4 is entirely final, week 5 is the first with an unplayed game
        self.assertEqual(box.value, nma.default_week(self.schedule))

    def test_summary_chips_show_season_and_upcoming_vs_final_counts(self):
        html = markdown_html(self.run_page())
        for chip in ("2026 season", "Week 5", "4 games", "3 upcoming", "1 final"):
            self.assertIn(chip, html)

    def test_every_game_of_the_week_is_listed_including_the_final_one(self):
        app = self.run_page()
        expected = {f"ml_pick_{g['game_id']}" for g in self.games(5)}
        self.assertEqual({b.key for b in app.button if str(b.key).startswith("ml_pick_")}, expected)
        self.assertEqual(len(tile_htmls(app)), 4)
        finals = [html for html in tile_htmls(app) if "Final 17-27" in html]
        self.assertEqual(len(finals), 1)
        self.assertIn("TEN", finals[0])

    def test_changing_the_week_changes_the_displayed_games(self):
        app = self.run_page()
        before = {b.key for b in app.button if str(b.key).startswith("ml_pick_")}
        app.selectbox(key="ml_week").set_value(6)
        self.refresh(app)
        after = {b.key for b in app.button if str(b.key).startswith("ml_pick_")}
        self.assertNotEqual(before, after)
        self.assertEqual(after, {f"ml_pick_{g['game_id']}" for g in self.games(6)})
        html = markdown_html(app)
        self.assertIn("Week 6", html)
        self.assertIn("2 upcoming", html)
        self.assertIn("0 final", html)

    def test_week_change_resets_the_selection_to_the_new_weeks_first_upcoming_game(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        self.assertEqual(selected_tile_codes(app), [("KC", "DEN")])
        app.selectbox(key="ml_week").set_value(6)
        self.refresh(app)
        first = self.games(6)[0]
        self.assertEqual(selected_tile_codes(app), [(first["away_team"], first["home_team"])])
        app.selectbox(key="ml_week").set_value(5)
        self.refresh(app)
        self.assertEqual(selected_tile_codes(app), [("NYJ", "BUF")])  # back to the default, not the old KC pick

    def test_default_week_with_every_game_played_is_the_last_week(self):
        played = self.schedule.copy()
        played["away_score"], played["home_score"], played["played"] = 20, 17, True
        self.inputs = {**self.inputs, "schedule": played}
        self.schedule = played
        app = self.run_page()
        self.assertEqual(app.selectbox(key="ml_week").value, 7)
        self.assertIn("0 upcoming", markdown_html(app))

    def test_a_week_with_no_games_shows_an_empty_state_not_a_traceback(self):
        with mock.patch.object(nma, "games_for_week", return_value=[]):
            app = self.run_page()
        self.assertIn("No games in Week 5", markdown_html(app))
        self.assertEqual(selected_tile_codes(app), [])

    def test_empty_schedule_shows_an_empty_state_with_retry(self):
        self.inputs = {**self.inputs, "schedule": self.schedule.iloc[0:0]}
        app = self.run_page()
        self.assertIn("No NFL games found", markdown_html(app))
        self.assertTrue(any(b.key == "ml_retry" for b in app.button))


class GameSelectionTests(MoneyLinesPageCase):
    def test_default_selection_is_the_first_upcoming_game_not_the_final_one(self):
        app = self.run_page()
        first_upcoming = next(g for g in self.games(5) if not g["played"])
        self.assertEqual(selected_tile_codes(app), [(first_upcoming["away_team"], first_upcoming["home_team"])])
        self.assertEqual(("NYJ", "BUF"), (first_upcoming["away_team"], first_upcoming["home_team"]))
        self.assertIn("Buffalo Bills", hero_html(app))
        self.assertIn("New York Jets", hero_html(app))
        selected_button = app.button(key=f"ml_pick_{first_upcoming['game_id']}")
        self.assertEqual(selected_button.label, "Selected")

    def test_only_the_selected_tile_glows(self):
        app = self.run_page()
        glowing = [html for html in tile_htmls(app) if "bt-selected" in html.split(">", 1)[0]]
        self.assertEqual(len(glowing), 1)

    def test_picking_a_game_changes_the_matchup_card_and_the_numbers(self):
        app = self.run_page()
        before_hero, before_tiles = hero_html(app), stat_tiles(app)
        target = self.game(5, "KC", "DEN")
        self.pick(app, target)
        after_hero = hero_html(app)
        self.assertNotEqual(before_hero, after_hero)
        self.assertIn("Kansas City Chiefs", after_hero)
        self.assertIn("Denver Broncos", after_hero)
        self.assertEqual(selected_tile_codes(app), [("KC", "DEN")])
        self.assertEqual(app.button(key=f"ml_pick_{target['game_id']}").label, "Selected")
        analysis = self.analysis(target)
        tiles = stat_tiles(app)
        self.assertNotEqual(before_tiles["Model total"], tiles["Model total"])
        self.assertEqual(tiles["Model total"], f"{analysis['scoring']['model_total']:.1f}")
        self.assertEqual(tiles["Model favorite"], analysis["favored_team"])

    def test_the_selection_is_kept_when_only_the_price_changes(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        hero = hero_html(app)
        self.type_into(app, "ml_price", "-150")
        self.assertEqual(selected_tile_codes(app), [("KC", "DEN")])
        self.assertEqual(hero_html(app), hero)
        self.type_into(app, "ml_price", "+120")
        self.assertEqual(selected_tile_codes(app), [("KC", "DEN")])
        app.number_input(key="ml_stake").set_value(250.0)
        self.refresh(app)
        self.assertEqual(selected_tile_codes(app), [("KC", "DEN")])

    def test_a_new_game_starts_with_blank_price_boxes_and_the_models_favorite_selected(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-150")
        self.type_into(app, "ml_spread", "-3.5")
        self.assertEqual(verdict_heads(app) != [], True)
        target = self.game(5, "KC", "DEN")
        self.pick(app, target)
        self.assertEqual(app.text_input(key="ml_price").value, "")
        self.assertEqual(app.text_input(key="ml_spread").value, "")
        self.assertEqual(verdict_heads(app), [])
        self.assertEqual(app.radio(key="ml_side").value, self.analysis(target)["favored"])

    def test_the_hero_card_is_the_full_width_layout_and_lists_the_chips(self):
        app = self.run_page()
        hero = hero_html(app)
        self.assertNotIn("bt-compact", hero)  # one full-width card, not squeezed into a column
        self.assertIn("Divisional", hero)  # NYJ @ BUF is a division game in the fixture
        self.assertIn("Outdoors", hero)
        self.assertEqual(hero.count("MODEL FAVORS"), 1)
        analysis = self.analysis(self.game(5, "NYJ", "BUF"))
        self.assertIn(analysis["teams"][analysis["favored"]]["team_name"], hero)

    def test_neutral_site_game_shows_the_chip_and_the_no_home_field_notice(self):
        app = self.run_page()
        app.selectbox(key="ml_week").set_value(6)
        self.refresh(app)
        self.pick(app, self.game(6, "SEA", "MIA"))
        html = markdown_html(app)
        self.assertIn("Neutral site", hero_html(app))
        self.assertIn("Neutral-site game", html)
        self.assertIn("no home-field edge", html)
        analysis = self.analysis(self.game(6, "SEA", "MIA"))
        self.assertTrue(analysis["context"]["neutral_site"])

    def test_assumed_starters_are_flagged_when_the_schedule_names_nobody(self):
        app = self.run_page()
        target = self.game(5, "DAL", "PHI")
        self.assertEqual(target["home_qb_source"], "last_game")
        self.pick(app, target)
        hero = hero_html(app)
        self.assertIn("QB Jalen Hurts (assumed)", hero)
        self.assertIn("QB Dak Prescott (assumed)", hero)
        self.assertIn("is assumed", markdown_html(app))  # the engine's warning is shown as a notice card

    def test_a_finished_game_shows_the_final_score_and_is_still_analyzable(self):
        app = self.run_page()
        target = self.game(5, "TEN", "NE")
        self.pick(app, target)
        html = markdown_html(app)
        self.assertIn("Final: TEN 17, NE 27", html)
        self.assertIn("Final 17-27", hero_html(app))
        self.assertIn("already been played", html)
        self.assertEqual(html.count("This game has already been played"), 1, "the played note must not be duplicated")
        analysis = self.analysis(target)
        self.assertEqual(stat_tiles(app)["Model favorite"], analysis["favored_team"])
        self.type_into(app, "ml_price", "-150")
        self.assertEqual(len(verdict_heads(app)), 1)  # a finished game can still be priced

    def test_skip_link_jumps_to_the_analysis_while_the_slate_is_open_and_goes_away_once_it_folds(self):
        app = self.run_page()
        html = markdown_html(app)
        self.assertIn('href="#ml-analysis"', html)
        self.assertIn("Skip to the analysis for NYJ @ BUF", html)
        self.assertEqual(html.count('id="ml-analysis"'), 1)
        self.assertLess(html.index('id="ml-analysis"'), html.index("bt-matchup"), "the anchor must sit above the matchup card")
        self.pick(app, self.game(5, "KC", "DEN"))
        html = markdown_html(app)
        self.assertNotIn("Skip to the analysis", html, "the analysis is already right below a folded slate")
        self.assertEqual(html.count('id="ml-analysis"'), 1)
        app.selectbox(key="ml_week").set_value(6)  # a new week opens the slate again, and the link comes back
        self.refresh(app)
        self.assertIn("Skip to the analysis for", markdown_html(app))

    def test_market_lines_are_labelled_as_the_consensus_not_the_users_book(self):
        app = self.run_page()
        hero = hero_html(app)
        self.assertIn("Consensus: BUF -9.5", hero)
        self.assertNotIn("Line: BUF", hero)
        buf_tile = next(t for t in tile_htmls(app) if ">NYJ<" in t and ">BUF<" in t)
        self.assertIn("Consensus BUF -9.5 · O/U 44.5", buf_tile)

    def test_quarterback_edge_tile_flags_assumed_starters_like_the_hero_does(self):
        app = self.run_page()
        self.pick(app, self.game(5, "DAL", "PHI"))
        self.assertIn("QB Jalen Hurts (assumed)", hero_html(app))
        sub = stat_subs(app)["Quarterback edge"]
        self.assertIn("Jalen Hurts (assumed)", sub)
        self.assertIn("Dak Prescott (assumed)", sub)

    def test_tile_leans_come_from_the_engine_and_finished_games_have_none(self):
        app = self.run_page()
        tiles = tile_htmls(app)
        for game in self.games(5):
            tile = next(t for t in tiles if f">{game['away_team']}<" in t and f">{game['home_team']}<" in t)
            if game["played"]:
                self.assertNotIn("Model:", tile)
                continue
            analysis = self.analysis(game)
            expected = f"Model: {analysis['favored_team']} {oi.format_pct(analysis['teams'][analysis['favored']]['win_probability'], 0)}"
            self.assertIn(expected, tile)


# --------------------------------------------------------------------------
# QA regressions: slate layout, fold state, scroll after a pick
# --------------------------------------------------------------------------
def page_style(app: AppTest) -> str:
    """The page's own injected ``<style>`` block (not the kit's)."""
    return "\n".join(str(m.value) for m in app.markdown if str(m.value).lstrip().startswith("<style") and "st-key-ml_slate" in str(m.value))


def slate_expander(app: AppTest):
    return next(e for e in app.expander if " slate: " in str(e.label))


def scroll_scripts(app: AppTest) -> list[str]:
    """Every st.html element that carries the scroll-to-analysis script."""
    return [str(e.proto) for e in app.get("html") if "scrollIntoView" in str(e.proto)]


class SlateLayoutTests(MoneyLinesPageCase):
    """The slate must reflow with its CONTAINER's width.

    It used to be rows of three ``st.columns``. Streamlit only stacks those below 640px of BROWSER width, so beside an
    open sidebar (a 900px window leaves about 590px) the tiles were 120px wide: the kickoff text wrapped one letter per
    line and a team code dropped under its logo. A CSS grid with an auto-fill minimum tile width has no such band.
    """

    def test_tiles_are_a_container_width_css_grid_and_not_streamlit_columns(self):
        app = self.run_page()
        css = page_style(app)
        self.assertIn(".st-key-ml_slate", css)
        self.assertIn("grid-template-columns:repeat(auto-fill,minmax(min(100%,14rem),1fr))", css)
        for column in app.columns:
            stuck = [b.key for b in column.button if str(b.key).startswith("ml_pick_")]
            self.assertEqual(stuck, [], "a tile button sits in a fixed-width st.columns cell")
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn('st.container(key="ml_slate")', source)
        self.assertIn('st.container(key=f"ml_cell_{game[\'game_id\']}")', source)
        self.assertNotIn("TILE_COLUMNS", source)

    def test_every_tile_and_its_button_still_render_once_per_game(self):
        app = self.run_page()
        pick_keys = [str(b.key) for b in app.button if str(b.key).startswith("ml_pick_")]
        self.assertEqual(sorted(pick_keys), sorted(f"ml_pick_{g['game_id']}" for g in self.games(5)))
        self.assertEqual(len(tile_htmls(app)), len(pick_keys))

    def test_a_tile_and_its_button_share_a_flex_cell_so_unequal_tiles_still_line_up(self):
        # A finished game's tile has no "Model:" line, so it is shorter than its neighbours; without a stretching
        # flex cell its button sat 25px higher than the rest of the row.
        css = page_style(self.run_page())
        self.assertIn('[class*="st-key-ml_cell_"]', css)
        self.assertRegex(css, r'st-key-ml_cell_"\] \{display:flex !important;flex-direction:column;height:100%')
        self.assertIn(".bt-tile {flex:1 1 auto;}", css)

    def test_the_matchup_card_stacks_where_an_open_sidebar_squeezes_it(self):
        css = page_style(self.run_page())
        self.assertIn("@media (min-width:821px) and (max-width:960px)", css)
        self.assertIn(".bt-matchup:not(.bt-compact) .bt-mu-body", css)
        self.assertIn(".bt-matchup:not(.bt-compact) .bt-mu-mid", css)

    def test_stat_tiles_wait_for_a_readable_width_in_the_same_band(self):
        # Three ~130px tiles broke the projected score "24.9 - 19.4" over two lines at 900px beside an open sidebar.
        css = page_style(self.run_page())
        self.assertIn("@media (min-width:821px) and (max-width:1100px)", css)
        self.assertIn("minmax(max(11rem,calc((100% - (var(--bt-cols,4) - 1) * .7rem) / var(--bt-cols,4))),1fr)", css)

    def test_the_three_price_boxes_are_bottom_aligned_so_wrapped_labels_do_not_stagger_them(self):
        app = self.run_page()
        cells = [
            column for column in app.columns
            if any(str(w.key) in ("ml_price", "ml_price_other") for w in column.text_input) or any(str(w.key) == "ml_stake" for w in column.number_input)
        ]
        self.assertEqual(len(cells), 3)
        for column in cells:
            self.assertIn("vertical_alignment: BOTTOM", str(column.proto))


class SlateFoldTests(MoneyLinesPageCase):
    def test_the_slate_opens_with_a_pick_prompt_and_names_the_week(self):
        expander = slate_expander(self.run_page())
        self.assertTrue(expander.proto.expanded)
        self.assertEqual(expander.label, "Week 5 slate: 4 games. Pick one to analyze.")

    def test_a_pick_folds_the_slate_and_the_label_names_the_chosen_game(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        expander = slate_expander(app)
        self.assertFalse(expander.proto.expanded)
        self.assertEqual(expander.label, "Week 5 slate: 4 games. KC @ DEN is selected. Open to pick another.")
        self.pick(app, self.game(5, "NYJ", "BUF"))
        self.assertIn("NYJ @ BUF is selected", slate_expander(app).label)

    def test_editing_the_price_or_the_stake_does_not_reopen_a_folded_slate(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        self.type_into(app, "ml_price", "-150")
        app.number_input(key="ml_stake").set_value(40.0)
        self.refresh(app)
        self.assertFalse(slate_expander(app).proto.expanded)

    def test_a_new_week_reopens_the_slate_and_a_one_game_week_is_worded_in_the_singular(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        app.selectbox(key="ml_week").set_value(7)
        self.refresh(app)
        expander = slate_expander(app)
        self.assertTrue(expander.proto.expanded)
        self.assertEqual(expander.label, "Week 7 slate: 1 game. Pick one to analyze.")


class ScrollAfterPickTests(MoneyLinesPageCase):
    """Folding the slate shortens the page, so a tile tapped far down a phone's slate left the viewport in the middle
    of the price check. A static script scrolls to the matchup card once per pick."""

    def test_no_script_on_load_or_on_ordinary_edits(self):
        app = self.run_page()
        self.assertEqual(scroll_scripts(app), [])
        self.type_into(app, "ml_price", "-150")
        self.type_into(app, "ml_spread", "-3.5")
        app.radio(key="ml_side").set_value("away")
        self.refresh(app)
        app.number_input(key="ml_stake").set_value(30.0)
        self.refresh(app)
        self.assertEqual(scroll_scripts(app), [])

    def test_one_script_follows_each_pick_and_does_not_repeat_on_the_next_edit(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        first = scroll_scripts(app)
        self.assertEqual(len(first), 1)
        self.assertIn("ml-analysis", first[0])
        self.type_into(app, "ml_price", "-150")
        self.assertEqual(scroll_scripts(app), [], "typing a price must not yank the page back to the matchup card")
        self.pick(app, self.game(5, "NYJ", "BUF"))
        second = scroll_scripts(app)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first, second, "the script text must differ per pick or the browser would not run it again")

    def test_a_week_change_does_not_scroll(self):
        app = self.run_page()
        self.pick(app, self.game(5, "KC", "DEN"))
        app.selectbox(key="ml_week").set_value(6)
        self.refresh(app)
        self.assertEqual(scroll_scripts(app), [])

    def test_the_script_is_static_text_that_never_contains_user_input(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "<b>-150</b>")
        self.pick(app, self.game(5, "KC", "DEN"))
        script = scroll_scripts(app)[0]
        self.assertNotIn("<b>", script)
        self.assertNotIn("-150", script)
        self.assertNotIn("KC", script)

    def test_the_analysis_anchor_leaves_room_for_the_header_and_the_folded_slate_bar(self):
        self.assertIn('id="ml-analysis" style="scroll-margin-top:8rem"', markdown_html(self.run_page()))


# --------------------------------------------------------------------------
# What the model shows
# --------------------------------------------------------------------------
class ModelOutputTests(MoneyLinesPageCase):
    def setUp(self):
        super().setUp()
        self.target = self.game(5, "NYJ", "BUF")
        self.expected = self.analysis(self.target)

    def test_all_nine_measures_are_shown_for_both_teams(self):
        html = markdown_html(self.run_page())
        for label in (
            "Team strength rating", "Offensive efficiency", "Defensive efficiency", "QB impact", "Pace (seconds per play",
            "Scoring projection", "Win probability", "Spread probability", "Moneyline probability",
        ):
            self.assertIn(label, html)
        for team in ("NYJ", "BUF"):
            self.assertIn(f">{team}</th>", html)

    def test_breakdown_cells_match_the_engine_numbers(self):
        html = markdown_html(self.run_page())
        for side in ("away", "home"):
            team = self.expected["teams"][side]
            for value in (
                f"{team['strength_rating']:.1f}", f"{team['offensive_efficiency']:.1f}", f"{team['defensive_efficiency']:.1f}",
                f"{team['pace_spp']:.1f}", f"{team['expected_points']:.1f}", oi.format_pct(team["win_probability"]),
                oi.format_pct(team["spread_cover_probability"]), oi.format_american(team["fair_moneyline"]),
                f"{team['qb_impact_points']:+.1f}", team["qb_name"],
            ):
                self.assertIn(f">{value}<", html, msg=f"{side}: {value} missing from the breakdown")

    def test_headline_tiles_show_model_and_market_side_by_side(self):
        tiles = stat_tiles(self.run_page())
        scoring = self.expected["scoring"]
        self.assertEqual(tiles["Model spread"], "BUF -" + f"{round(scoring['model_spread_home'], 1):g}")
        self.assertEqual(tiles["Model total"], f"{scoring['model_total']:.1f}")
        self.assertEqual(tiles["Projected score"], f"{scoring['away_expected_points']:.1f} – {scoring['home_expected_points']:.1f}")
        self.assertEqual(tiles["Pace projection"], f"{self.expected['pace']['plays_projection']:.0f} plays")
        html = markdown_html(self.run_page())
        self.assertIn("Consensus line: BUF -9.5", html)
        self.assertIn("Consensus total 44.5", html)

    def test_a_game_with_no_line_says_so_instead_of_showing_a_blank(self):
        app = self.run_page()
        self.pick(app, self.game(5, "DAL", "PHI"))
        html = markdown_html(app)
        self.assertGreaterEqual(html.count("No line posted yet"), 3)  # tile, hero card and the headline tiles
        self.assertIn("nflverse leaves the market columns blank", html)
        self.assertIsNone(_NAN_WORD.search(visible_text(app)))

    def test_probability_rows_say_which_price_basis_they_use(self):
        app = self.run_page()
        html = markdown_html(app)
        self.assertIn(">Win probability (margin included)<", html)
        self.assertIn(">Win probability (margin removed)<", html)
        captions = " ".join(str(c.value) for c in app.caption)
        self.assertIn("Margin included is the break-even rate", captions)
        self.assertIn("margin removed is the market's own probability", captions)

    def test_qb_names_impact_and_note_are_shown(self):
        html = markdown_html(self.run_page())
        self.assertIn("QB Josh Allen", html)
        self.assertIn("QB Geno Smith", html)
        self.assertIn("Quarterback edge", html)

    def test_the_gap_between_the_qb_impact_row_and_the_net_edge_tile_is_explained(self):
        # Impacts of -0.4 and +0.5 in the table read as a +0.9 edge, yet the tile can say +0.2: part of each starter's
        # value is already in his team's rating. A user comparing the two numbers needs that sentence, so it must stay.
        app = self.run_page()
        captions = " ".join(str(c.value) for c in app.caption)
        self.assertIn("already built into his team's strength rating", captions)
        self.assertIn("net quarterback effect in the tile above", captions)

    def test_reasoning_panel_confidence_meter_and_engine_warnings_render(self):
        html = markdown_html(self.run_page())
        self.assertIn("bt-reasons", html)
        self.assertEqual(html.count('class="bt-reason '), len(self.expected["reasoning"]))
        for item in self.expected["reasoning"]:
            self.assertIn(item["title"].split(":")[0], html)
        label = self.expected["confidence_label"]
        self.assertIn(f"{label} · score {self.expected['confidence'] * 100:.0f}/100", html)
        for warning in self.expected["warnings"]:
            self.assertIn(warning.split(".")[0][:40], html)  # warnings become notice cards

    def test_methodology_panel_quotes_the_engine_constants(self):
        app = self.run_page()
        html = markdown_html(app).lower()
        # The expander itself, with the exact label the brief asks for.
        self.assertIn("How this is computed and its limits", [str(e.label) for e in app.expander])
        for fragment in (
            f"k = {nma.BLEND_K_GAMES:g} games", f"{nma.PRIOR_CARRYOVER:.0%}", f"{nma.WIN_PROB_STDEV:g} points",
            f"{nma.MARGIN_STDEV:g} points", "3,649", "0.6110", "0.6085", "74.5%", "69.4%", "only the quarterback",
            "consensus reference", "blank for games that have no line yet", "nothing here places a bet",
        ):
            self.assertIn(fragment.lower(), html, msg=f"methodology is missing: {fragment}")


# --------------------------------------------------------------------------
# The price check
# --------------------------------------------------------------------------
class PriceCheckTests(MoneyLinesPageCase):
    def setUp(self):
        super().setUp()
        self.target = self.game(5, "NYJ", "BUF")
        self.expected = self.analysis(self.target)  # BUF is the model's favorite

    def test_blank_price_shows_no_verdict_and_a_hint(self):
        app = self.run_page()
        self.assertEqual(verdict_heads(app), [])
        html = markdown_html(app)
        self.assertIn("Paste a sportsbook moneyline to price it", html)
        self.assertNotIn("Implied probability", stat_tiles(app))  # no price, so no price tiles (the methodology text still names it)
        self.assertEqual(app.radio(key="ml_side").value, self.expected["favored"])
        self.assertEqual(app.number_input(key="ml_stake").value, 100.0)

    def test_minus_150_matches_the_engine_exactly(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-150")
        ev = nma.evaluate_moneyline_price(self.expected, self.expected["favored"], -150.0, stake=100.0)
        reference = oi.evaluate_price(self.expected["teams"][self.expected["favored"]]["win_probability"], -150.0, stake=100.0)
        self.assertEqual(ev["implied_probability"], reference["implied_probability"])
        html = markdown_html(app)
        self.assertEqual(verdict_heads(app), [ev["verdict"]])
        tiles = stat_tiles(app)
        self.assertEqual(tiles["Implied probability"], oi.format_pct(ev["implied_probability"]))
        self.assertEqual(tiles["Implied probability"], "60.0%")  # -150 needs 60% to break even
        self.assertEqual(tiles["Model probability"], oi.format_pct(ev["model_probability"]))
        self.assertEqual(tiles["Value gap"], oi.format_signed_pct(ev["value_gap"]))
        self.assertEqual(tiles["Expected value per $100"], oi.format_money(ev["expected_value"]))
        self.assertEqual(tiles["Model fair price"], oi.format_american(ev["fair_price"]))
        self.assertIn(f"{ev['team']} at -150", html)

    def test_plus_120_on_the_other_team_matches_the_engine(self):
        app = self.run_page()
        app.radio(key="ml_side").set_value("away")
        self.refresh(app)
        self.type_into(app, "ml_price", "+120")
        ev = nma.evaluate_moneyline_price(self.expected, "away", 120.0, stake=100.0)
        tiles = stat_tiles(app)
        self.assertEqual(ev["team"], "NYJ")
        self.assertEqual(tiles["Implied probability"], oi.format_pct(ev["implied_probability"]))
        self.assertEqual(tiles["Implied probability"], "45.5%")  # +120 breaks even at 45.5%
        self.assertEqual(tiles["Expected value per $100"], oi.format_money(ev["expected_value"]))
        self.assertEqual(verdict_heads(app), [ev["verdict"]])
        self.assertIn("NYJ at +120", markdown_html(app))

    def test_each_verdict_label_is_reachable_and_matches_value_verdict(self):
        seen = set()
        for price in ("-1000", "-400", "-300", "-200", "-150", "+120"):
            for side in ("home", "away"):
                app = self.run_page()
                app.radio(key="ml_side").set_value(side)
                self.refresh(app)
                self.type_into(app, "ml_price", price)
                ev = nma.evaluate_moneyline_price(self.expected, side, float(price), stake=100.0)
                self.assertEqual(verdict_heads(app), [ev["verdict"]], msg=f"{side} {price}")
                self.assertEqual(ev["verdict"], oi.value_verdict(ev["value_gap"])[0])
                seen.add(ev["verdict"])
        self.assertGreaterEqual(len(seen), 3, f"fixture should reach several verdicts, saw {seen}")

    def test_decimal_and_even_money_prices_are_accepted(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "1.91")
        self.assertEqual(len(verdict_heads(app)), 1)
        ev = nma.evaluate_moneyline_price(self.expected, self.expected["favored"], oi.parse_odds("1.91").price)
        self.assertEqual(stat_tiles(app)["Implied probability"], oi.format_pct(ev["implied_probability"]))
        self.type_into(app, "ml_price", "EVEN")
        self.assertEqual(stat_tiles(app)["Implied probability"], "50.0%")

    def test_invalid_text_shows_a_friendly_error_and_no_verdict(self):
        app = self.run_page()
        for bad in ("abc", "-50", "12x", "1.0", "--150", "+0"):
            self.type_into(app, "ml_price", bad)
            html = markdown_html(app)
            self.assertEqual(verdict_heads(app), [], msg=bad)
            self.assertIn("Moneyline not understood", html, msg=bad)
            self.assertEqual(self.error_notices(app), 1, msg=bad)
        self.type_into(app, "ml_price", "-150")  # and it recovers
        self.assertEqual(len(verdict_heads(app)), 1)
        self.assertNotIn("Moneyline not understood", markdown_html(app))

    def error_notices(self, app: AppTest) -> int:
        return len(re.findall(r'class="bt bt-notice bt-t-bad"', markdown_html(app)))

    def test_a_decimal_comma_is_an_error_not_a_different_price(self):
        # modules.odds_input drops every comma, so a European "1,91" would silently become +191.
        app = self.run_page()
        for text in ("1,91", "1,9", "+1,5"):
            self.type_into(app, "ml_price", text)
            self.assertEqual(verdict_heads(app), [], msg=text)
            self.assertIn("Use a period for decimal odds", " ".join(error_notice_texts(app)), msg=text)
        self.type_into(app, "ml_price", "")
        self.type_into(app, "ml_price_other", "1,91")
        self.assertIn("Other team's moneyline not understood. Use a period", " ".join(error_notice_texts(app)))

    def test_a_thousands_separator_comma_is_still_fine(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-1,150")
        ev = nma.evaluate_moneyline_price(self.expected, self.expected["favored"], -1150.0)
        self.assertEqual(error_notice_texts(app), [])
        self.assertEqual(stat_tiles(app)["Implied probability"], oi.format_pct(ev["implied_probability"]))

    def test_an_impossible_price_is_a_friendly_error_not_a_twenty_digit_payout(self):
        app = self.run_page()
        for text in ("+99999999999999999999", "-150000", "+100001", "9" * 400):
            self.type_into(app, "ml_price", text)
            self.assertEqual(verdict_heads(app), [], msg=text[:12])
            self.assertIn("No sportsbook offers a price beyond", " ".join(error_notice_texts(app)), msg=text[:12])
            self.assertNotRegex(visible_text(app), r"\d{9,}", msg=f"a huge number leaked into the page for {text[:12]!r}")
        self.type_into(app, "ml_price", "+100000")  # the boundary itself is accepted
        self.assertEqual(len(verdict_heads(app)), 1)
        self.assertEqual(error_notice_texts(app), [])

    def test_blank_again_after_a_price_clears_the_verdict_silently(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-150")
        self.assertEqual(len(verdict_heads(app)), 1)
        self.type_into(app, "ml_price", "   ")
        self.assertEqual(verdict_heads(app), [])
        self.assertEqual(self.error_notices(app), 0)

    def test_stake_changes_the_expected_value(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "+110")
        side = self.expected["favored"]
        first = stat_tiles(app)["Expected value per $100"]
        self.assertEqual(first, oi.format_money(nma.evaluate_moneyline_price(self.expected, side, 110.0, stake=100.0)["expected_value"]))
        app.number_input(key="ml_stake").set_value(250.0)
        self.refresh(app)
        tiles = stat_tiles(app)
        ev = nma.evaluate_moneyline_price(self.expected, side, 110.0, stake=250.0)
        self.assertEqual(tiles["Expected value per $250"], oi.format_money(ev["expected_value"]))
        self.assertIn("$250 stake", markdown_html(app))

    def test_a_stake_with_cents_is_shown_with_cents(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "+110")
        app.number_input(key="ml_stake").set_value(25.5)
        self.refresh(app)
        ev = nma.evaluate_moneyline_price(self.expected, self.expected["favored"], 110.0, stake=25.5)
        self.assertEqual(stat_tiles(app)["Expected value per $25.50"], oi.format_money(ev["expected_value"]))
        self.assertIn("$25.50 stake", markdown_html(app))

    def test_other_side_price_adds_the_no_vig_market_probability(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-150")
        self.assertIn("Consensus, margin removed", stat_tiles(app))  # falls back to the nflverse consensus
        self.type_into(app, "ml_price_other", "+130")
        tiles = stat_tiles(app)
        fair_side, _ = oi.no_vig_probabilities(-150.0, 130.0)
        self.assertEqual(tiles["Your book, margin removed"], oi.format_pct(fair_side))
        self.assertNotIn("Consensus, margin removed", tiles)
        self.assertIn(oi.format_pct(fair_side), markdown_html(app))
        self.type_into(app, "ml_price_other", "garbage")
        self.assertIn("Other team's moneyline not understood", markdown_html(app))
        self.assertEqual(len(verdict_heads(app)), 1, "a bad second price must not hide the verdict")

    def test_two_prices_that_are_not_opposite_sides_get_a_warning(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "+150")
        self.type_into(app, "ml_price_other", "+150")
        self.assertIn("cannot be the two sides of one market", markdown_html(app))
        self.assertNotIn("Your book, margin removed", stat_tiles(app))

    def test_model_vs_market_table_lists_model_book_and_consensus(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-150")
        html = markdown_html(app)
        for header in ("Model", "Your sportsbook", "nflverse consensus"):
            self.assertIn(f">{header}</th>", html)
        self.assertIn(">-450<", html)  # the consensus moneyline for BUF
        self.assertIn(">-150<", html)  # the user's own price
        captions = " ".join(str(c.value) for c in app.caption)
        self.assertIn("consensus reference", captions)
        self.assertIn("not the price at your sportsbook", captions)

    def test_a_very_large_gap_shows_the_check_the_model_caution(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "+900")
        ev = nma.evaluate_moneyline_price(self.expected, self.expected["favored"], 900.0)
        self.assertGreaterEqual(abs(ev["value_gap"]), 0.10)
        self.assertIn("more often a sign the model is missing something", markdown_html(app))

    def test_no_market_line_game_can_still_be_priced(self):
        app = self.run_page()
        target = self.game(5, "DAL", "PHI")
        self.pick(app, target)
        self.type_into(app, "ml_price", "+105")
        analysis = self.analysis(target)
        ev = nma.evaluate_moneyline_price(analysis, analysis["favored"], 105.0)
        self.assertEqual(verdict_heads(app), [ev["verdict"]])
        self.assertIn("Model fair price", stat_tiles(app))
        self.assertNotIn("Consensus, margin removed", stat_tiles(app))  # there is no consensus to show
        self.assertIsNone(_NAN_WORD.search(visible_text(app)))


_NEGATIVE_ZERO = re.compile(r"(?<![\w.])-0\.0(?!\d)")
_CONSENSUS_NOTE = "nflverse consensus puts"


def expected_value_tone(app: AppTest) -> str:
    match = re.search(r'<div class="bt bt-stat bt-t-(\w+)"><div class="bt-stat-label">Expected value per \$', markdown_html(app))
    assert match, "no expected-value tile on the page"
    return match.group(1)


class PriceSanityTests(MoneyLinesPageCase):
    """QA regressions on the price check: tone consistency, signed zeros, and a price that cannot be this game's."""

    def setUp(self):
        super().setUp()
        self.target = self.game(5, "NYJ", "BUF")  # consensus BUF -450 / NYJ +350
        self.expected = self.analysis(self.target)
        self.favored = self.expected["favored"]

    def priced(self, price: int) -> dict:
        return nma.evaluate_moneyline_price(self.expected, self.favored, float(price))

    def test_a_fair_price_never_shows_a_red_expected_value(self):
        # "Fair price" beside a red -$0.07 read as a contradiction: inside the fair band the dollars are neutral.
        slightly_negative = next(
            (p for p in range(-1500, -100) if self.priced(p)["verdict"] == "Fair price" and self.priced(p)["expected_value"] < 0), None
        )
        self.assertIsNotNone(slightly_negative, "the fixture should have a fair price with a tiny negative expected value")
        app = self.run_page()
        self.type_into(app, "ml_price", str(slightly_negative))
        self.assertEqual(verdict_heads(app), ["Fair price"])
        self.assertEqual(expected_value_tone(app), "neutral")

    def test_overpriced_is_red_and_value_is_green(self):
        app = self.run_page()
        over = next(p for p in range(-1500, -100) if self.priced(p)["verdict"] == "Overpriced")
        self.type_into(app, "ml_price", str(over))
        self.assertEqual(expected_value_tone(app), "bad")
        under = next(p for p in range(-100, 1500, 5) if p not in range(-99, 100) and self.priced(p)["verdict"] == "Strong value")
        self.type_into(app, "ml_price", str(under))
        self.assertEqual(expected_value_tone(app), "good")

    def test_a_gap_that_rounds_to_nothing_never_prints_as_minus_zero(self):
        real = nma.evaluate_moneyline_price

        def barely_negative(analysis, side, price, *, stake=100.0):
            result = real(analysis, side, price, stake=stake)
            result.update(value_gap=-0.0003, ev_pct=-0.0002, expected_value=-0.02, verdict="Fair price", verdict_tone="neutral")
            return result

        with mock.patch.object(nma, "evaluate_moneyline_price", side_effect=barely_negative):
            app = self.run_page()
            self.type_into(app, "ml_price", "-150")
        self.assertEqual(stat_tiles(app)["Value gap"], "0.0 pts")  # no sign on zero, and never "-0.0 pts"
        text = visible_text(app)
        self.assertIn("(+0.0% of the stake)", text)
        self.assertIn("(0.0 pts)", text)  # the verdict sentence
        self.assertNotRegex(text, r"-0\.0+ pts")
        self.assertIsNone(_NEGATIVE_ZERO.search(text), _NEGATIVE_ZERO.search(text) and text[_NEGATIVE_ZERO.search(text).start() - 40 :][:90])

    def test_model_numbers_that_round_to_nothing_never_print_as_minus_zero(self):
        real = nma.analyze_game

        def nudged(game, ratings, qb_context, **kwargs):
            analysis = real(game, ratings, qb_context, **kwargs)
            analysis["teams"]["home"]["qb_impact_points"] = -0.02
            if analysis["scoring"]["market_total"] is not None:
                analysis["scoring"]["total_vs_market"] = -0.02
            return analysis

        with mock.patch.object(nma, "analyze_game", side_effect=nudged):
            app = self.run_page()
        self.assertIn("(model +0.0)", stat_subs(app)["Model total"])
        self.assertIn(">+0.0<", markdown_html(app))  # the QB impact cell of the breakdown table
        self.assertIsNone(_NEGATIVE_ZERO.search(visible_text(app)))

    def test_a_price_far_from_the_consensus_asks_you_to_check_the_team(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "+150")  # BUF is a -450 consensus favorite: +150 is the other side's kind of price
        html = markdown_html(app)
        self.assertIn(_CONSENSUS_NOTE, html)
        self.assertIn("Your BUF price implies 40.0%", html)
        self.assertIn("(NYJ)", html)
        self.assertEqual(len(verdict_heads(app)), 1, "the notice is a caution, the verdict is still shown")

    def test_a_price_near_the_consensus_gets_no_such_notice(self):
        app = self.run_page()
        for price in ("-450", "-400", "-500", "-380"):
            self.type_into(app, "ml_price", price)
            self.assertNotIn(_CONSENSUS_NOTE, markdown_html(app), msg=price)

    def test_with_both_prices_the_margin_removed_rate_is_compared_not_the_raw_one(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-450")
        self.type_into(app, "ml_price_other", "+350")
        self.assertNotIn(_CONSENSUS_NOTE, markdown_html(app))
        self.type_into(app, "ml_price", "+350")  # the same two prices, but now typed for the wrong team
        self.type_into(app, "ml_price_other", "-450")
        self.assertIn(_CONSENSUS_NOTE, markdown_html(app))

    def test_a_game_with_no_consensus_cannot_trigger_it(self):
        app = self.run_page()
        self.pick(app, self.game(5, "DAL", "PHI"))
        for price in ("+500", "-2000"):
            self.type_into(app, "ml_price", price)
            self.assertNotIn(_CONSENSUS_NOTE, markdown_html(app), msg=price)

    def test_model_vs_market_is_a_section_like_its_siblings_and_not_a_loose_markdown_heading(self):
        app = self.run_page()
        self.assertIn("The model's numbers beside your sportsbook's price", markdown_html(app))
        self.assertNotIn("Model vs. market", [str(m.value).strip() for m in app.markdown if str(m.value).strip().startswith("####")])
        self.assertEqual([m for m in app.markdown if str(m.value).lstrip().startswith("####")], [])


class SpreadInputTests(MoneyLinesPageCase):
    """Sportsbook spread notation (-3.5 = home favored) must reach ``analyze_game`` as nflverse notation (+3.5)."""

    def setUp(self):
        super().setUp()
        self.target = self.game(5, "NYJ", "BUF")
        self.real_analyze = nma.analyze_game
        self.calls: list[dict] = []

        def spy(game, ratings, qb_context, **kwargs):
            self.calls.append({"game_id": game["game_id"], **kwargs})
            return self.real_analyze(game, ratings, qb_context, **kwargs)

        patcher = mock.patch.object(nma, "analyze_game", side_effect=spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def spreads_for_target(self) -> list:
        return [c.get("user_spread_home") for c in self.calls if c["game_id"] == self.target["game_id"]]

    def test_no_spread_means_no_override(self):
        self.run_page()
        self.assertEqual(set(self.spreads_for_target()), {None})

    def test_home_minus_3_5_is_passed_as_plus_3_5(self):
        app = self.run_page()
        self.calls.clear()
        self.type_into(app, "ml_spread", "-3.5")
        self.assertEqual(self.spreads_for_target()[-1], 3.5)

    def test_home_plus_7_is_passed_as_minus_7(self):
        app = self.run_page()
        self.calls.clear()
        self.type_into(app, "ml_spread", "+7")
        self.assertEqual(self.spreads_for_target()[-1], -7.0)

    def test_pick_em_is_zero_and_not_negative_zero(self):
        app = self.run_page()
        for text in ("PK", "pick'em", "0"):
            self.calls.clear()
            self.type_into(app, "ml_spread", text)
            value = self.spreads_for_target()[-1]
            self.assertEqual(value, 0.0, msg=text)
            self.assertEqual(str(value), "0.0", msg=f"{text} must not become -0.0")

    def test_spread_probability_row_follows_the_typed_spread(self):
        app = self.run_page()
        base = markdown_html(app)
        self.type_into(app, "ml_spread", "-3.5")
        typed = markdown_html(app)
        analysis = self.real_analyze(self.target, self.ratings, self.qb_context, user_spread_home=3.5)
        self.assertEqual(analysis["probabilities"]["spread_source"], "user")
        for side in ("away", "home"):
            self.assertIn(f">{oi.format_pct(analysis['teams'][side]['spread_cover_probability'])}<", typed)
        self.assertNotEqual(base, typed)
        captions = " ".join(str(c.value) for c in app.caption)
        self.assertIn("your sportsbook spread, BUF -3.5", captions)
        self.assertIn("BUF -3.5", typed)  # also in the model-vs-market table's sportsbook column

    def test_only_pick_em_words_mean_zero_text_around_pk_is_an_error(self):
        # "PK -3" or "pick 99" used to be read as a pick'em and silently drop the number the user typed.
        app = self.run_page()
        for bad in ("pk 7", "PK -3", "pk-3", "pick 99", "pk7", "pkx"):
            self.calls.clear()
            self.type_into(app, "ml_spread", bad)
            self.assertIn("Enter the spread as a number", " ".join(error_notice_texts(app)), msg=bad)
            self.assertIsNone(self.spreads_for_target()[-1], msg=bad)
        for good in ("PK", "Pick'em", "pick-em", "P.K.", "pickem", "PICK", "  pk  "):
            self.calls.clear()
            self.type_into(app, "ml_spread", good)
            self.assertEqual(error_notice_texts(app), [], msg=good)
            self.assertEqual(str(self.spreads_for_target()[-1]), "0.0", msg=good)

    def test_a_decimal_comma_spread_is_an_error_not_minus_35(self):
        # modules.odds_input drops commas, so "-3,5" used to arrive as -35 (home favored by 35).
        app = self.run_page()
        for bad in ("-3,5", "3,5", "+7,0"):
            self.calls.clear()
            self.type_into(app, "ml_spread", bad)
            self.assertIn("Use a period for the decimal point", " ".join(error_notice_texts(app)), msg=bad)
            self.assertIsNone(self.spreads_for_target()[-1], msg=bad)

    def test_the_typed_spread_is_read_back_in_words_so_the_sign_cannot_be_misread(self):
        app = self.run_page()
        for text, expected in (
            ("-3.5", "Read as: BUF favored by 3.5."),
            ("+3.5", "Read as: NYJ favored by 3.5 (BUF +3.5)."),
            ("3.5", "Read as: NYJ favored by 3.5 (BUF +3.5)."),
            ("-7", "Read as: BUF favored by 7."),
            ("PK", "Read as: pick'em, neither team favored."),
        ):
            self.type_into(app, "ml_spread", text)
            captions = " ".join(str(c.value) for c in app.caption)
            self.assertIn(expected, captions, msg=text)
        self.type_into(app, "ml_spread", "abc")
        self.assertNotIn("Read as:", " ".join(str(c.value) for c in app.caption))
        self.type_into(app, "ml_spread", "")
        self.assertNotIn("Read as:", " ".join(str(c.value) for c in app.caption))

    def test_default_spread_source_is_the_nflverse_consensus(self):
        app = self.run_page()
        captions = " ".join(str(c.value) for c in app.caption)
        self.assertIn("the nflverse consensus spread, BUF -9.5", captions)

    def test_no_line_game_uses_the_model_spread_and_says_so(self):
        app = self.run_page()
        self.pick(app, self.game(5, "DAL", "PHI"))
        captions = " ".join(str(c.value) for c in app.caption)
        self.assertIn("the model's own spread rounded to the nearest half point", captions)

    def test_invalid_spread_is_a_friendly_error_and_no_override(self):
        app = self.run_page()
        for bad in ("abc", "BUF -3", "99", "-61"):
            self.calls.clear()
            self.type_into(app, "ml_spread", bad)
            self.assertIn("Enter the spread as a number", markdown_html(app), msg=bad)
            self.assertEqual(self.spreads_for_target()[-1], None, msg=bad)
        self.calls.clear()
        self.type_into(app, "ml_spread", "")
        self.assertNotIn("Enter the spread as a number", markdown_html(app))

    def test_spread_does_not_change_win_probability_or_the_price_verdict(self):
        app = self.run_page()
        self.type_into(app, "ml_price", "-150")
        verdict = verdict_heads(app)
        tiles = stat_tiles(app)
        self.type_into(app, "ml_spread", "-3.5")
        self.assertEqual(verdict_heads(app), verdict)
        self.assertEqual(stat_tiles(app)["Model probability"], tiles["Model probability"])


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------
class FailurePathTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def test_schedule_unavailable_shows_a_friendly_state_and_retry(self):
        message = "The 2026 NFL schedule could not be loaded (URLError: offline). Check your connection and try again."
        with mock.patch.object(nma, "load_moneyline_inputs", side_effect=nma.ScheduleUnavailable(message)):
            app = AppTest.from_file(str(PAGE), default_timeout=120)
            app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        html = markdown_html(app)
        self.assertIn("The NFL schedule is unavailable", html)
        self.assertIn("could not be loaded", html)
        self.assertIn("Research and educational analysis only", html)  # the page still ends cleanly
        self.assertTrue(any(b.key == "ml_retry" and b.label == "Retry" for b in app.button))
        self.assertEqual(len(app.selectbox), 0)

    def test_retry_clears_caches_and_loads_again(self):
        inputs = fixture_inputs()
        outcomes = [nma.ScheduleUnavailable("down"), inputs]

        def loader(season=None, today=None):
            outcome = outcomes[0] if len(outcomes) == 1 else outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with mock.patch.object(nma, "load_moneyline_inputs", side_effect=loader), mock.patch.object(nma, "clear_caches") as clear:
            app = AppTest.from_file(str(PAGE), default_timeout=120)
            app.run()
            self.assertIn("The NFL schedule is unavailable", markdown_html(app))
            app.button(key="ml_retry").click()
            app.run()
            self.assertEqual([str(e.value) for e in app.exception], [])
            clear.assert_called_once()
            self.assertEqual(app.selectbox(key="ml_week").value, 5)
            self.assertNotIn("The NFL schedule is unavailable", markdown_html(app))

    def test_an_unexpected_loader_error_is_not_a_traceback(self):
        with mock.patch.object(nma, "load_moneyline_inputs", side_effect=KeyError("boom")):
            app = AppTest.from_file(str(PAGE), default_timeout=120)
            app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        html = markdown_html(app)
        self.assertIn("The model inputs could not be loaded", html)
        self.assertIn("KeyError", html)
        self.assertTrue(any(b.key == "ml_retry" for b in app.button))

    def test_a_game_the_engine_cannot_analyze_is_an_empty_state_not_a_traceback(self):
        inputs = fixture_inputs()
        real = nma.analyze_game

        def flaky(game, ratings, qb_context, **kwargs):
            if game["home_team"] == "BUF" and game["week"] == 5:
                raise ValueError("home_team and away_team must be different teams")
            return real(game, ratings, qb_context, **kwargs)

        with mock.patch.object(nma, "load_moneyline_inputs", return_value=inputs), mock.patch.object(nma, "analyze_game", side_effect=flaky):
            app = AppTest.from_file(str(PAGE), default_timeout=120)
            app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertIn("This game could not be analyzed", markdown_html(app))

    def test_missing_ratings_and_quarterback_data_degrade_to_warnings(self):
        inputs = fixture_inputs()
        inputs["ratings"] = {"teams": {}, "league": {}, "meta": {}}
        inputs["qb_context"] = {}
        with mock.patch.object(nma, "load_moneyline_inputs", return_value=inputs):
            app = AppTest.from_file(str(PAGE), default_timeout=120)
            app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        html = markdown_html(app)
        self.assertIn("league-average", html)
        self.assertIn("Quarterback data unavailable", html)
        self.assertIsNone(_NAN_WORD.search(visible_text(app)))


# --------------------------------------------------------------------------
# One live-data smoke test (skipped when nflverse cannot be reached)
# --------------------------------------------------------------------------
class LiveSmokeTest(unittest.TestCase):
    def test_real_schedule_renders_and_prices_a_moneyline(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        try:
            live = nma.load_moneyline_inputs()
        except Exception as error:  # noqa: BLE001 - no network / nflverse down: nothing to smoke-test
            self.skipTest(f"live nflverse data unavailable: {type(error).__name__}")
        if not len(live["schedule"]):
            self.skipTest("live schedule is empty")
        app = AppTest.from_file(str(PAGE), default_timeout=180)
        app.run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        box = app.selectbox(key="ml_week")
        self.assertEqual(box.value, nma.default_week(live["schedule"]))
        games = nma.games_for_week(live["schedule"], box.value)
        self.assertEqual(len([b for b in app.button if str(b.key).startswith("ml_pick_")]), len(games))
        self.assertEqual(len(selected_tile_codes(app)), 1)
        app.text_input(key="ml_price").input("-150").run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertEqual(len(verdict_heads(app)), 1)
        self.assertIsNone(_NAN_WORD.search(visible_text(app)))
        self.assertIsNone(_NONE_WORD.search(visible_text(app)))


if __name__ == "__main__":
    unittest.main()
