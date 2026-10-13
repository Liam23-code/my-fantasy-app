"""Offline rendering contract for the NFL Player Props page (``app/pages/playerprops.py``).

Every test but the last runs the real Streamlit script through ``AppTest`` against a
hand-built player pool (the real forward-projection row shape) injected by patching
``modules.nfl_prop_analysis.load_player_pool``, so nothing touches the network. The
page re-executes its imports on every run, which is why patching the module attribute
before ``app.run()`` is enough. The single live-data smoke test skips itself when the
pool cannot be loaded.
"""

import json
import re
import sys
import tempfile
import unittest
from html import escape, unescape
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from fantasy import player_status
from streamlit.testing.v1 import AppTest

from app.style import SIGNAL_GOLD
from modules import nfl_matchup_features as nmf
from modules import nfl_moneyline_analysis as nma
from modules import nfl_prop_analysis as npa

PAGE = str(Path(_PROJECT_ROOT) / "app" / "pages" / "playerprops.py")


_PACK_PATCHES: list = []


def setUpModule():
    """The page's matchup upgrade pack is ON by default and its helpers do I/O (the schedule lookup that picks a
    default opponent, and the feature-context builder). These tests pin the BASE model and must not reach the
    network, so both are made unavailable: the page then runs exactly as it did before the pack existed.
    ``test_nfl_props_pack_page.py`` covers the pack path with its own fixtures."""
    for target, name in ((nma, "load_schedule"), (nmf, "build_game_context")):
        patcher = mock.patch.object(target, name, side_effect=RuntimeError("pack disabled for base-model tests"))
        patcher.start()
        _PACK_PATCHES.append(patcher)


def tearDownModule():
    while _PACK_PATCHES:
        _PACK_PATCHES.pop().stop()

# A printed non-finite number or a Python ``None`` leaking into a label. Whole words only:
# the disclaimer says "financial", which contains "nan".
LEAK = re.compile(r"(?<![\w.])(?:nan|NaN|None|inf|-inf|NaT|undefined)(?![\w])")


# ---------------------------------------------------------------------------
# Fixture pool
# ---------------------------------------------------------------------------
def _history(base: float) -> list[dict]:
    return [
        {
            "week": week,
            "points": round(base + ((week * 7) % 11 - 5) * 1.5, 1),
            "targets": 6.0,
            "carries": 12.0,
            "target_share": 0.2,
            "air_yards_share": 0.1,
            "opponent_defense_rank": (week * 5) % 32 + 1,
        }
        for week in range(1, 18)
    ]


def make_player(player_id, name, position, team, games=17, **stats) -> dict:
    row = {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": team,
        "season": 2026,
        "prior_season": 2025,
        "projection_season": 2026,
        "carries": 0.0,
        "pass_attempts": 0.0,
        "passing_tds": 0.0,
        "passing_yards": 0.0,
        "interceptions": 0.0,
        "receiving_tds": 0.0,
        "receiving_yards": 0.0,
        "receptions": 0.0,
        "rushing_tds": 0.0,
        "rushing_yards": 0.0,
        "targets": 0.0,
        "projection": 200.0,
        "expected_fantasy_points": 200.0,
        "games_played": games,
        "prior_games_played": games,
        "points_per_game": 15.0,
        "floor": 170.0,
        "median": 235.0,
        "ceiling": 330.0,
        "expected_games": 13.5,
        "projection_confidence": 0.9,
        "drivers": ["1,000 rushing yards", "60 receptions", "10 total TDs"],
        "projection_basis": (
            "2026 projection from 2025 actuals (17 games -> 13.5 expected); "
            "rate regressed to the position median, 35% market-reconciled"
        ),
        "scoring_mode": "ppr",
        "history": _history(15.0),
        "age": 27.0,
        "years_experience": 5,
        "adp": 40.0,
    }
    row.update(stats)
    return row


QB_ID = "00-KC-QB1"
RB_ID = "00-KC-RB1"
WR_ID = "00-KC-WR1"
TE_ID = "00-KC-TE1"
ROOKIE_ID = "00-KC-WR9"


def build_pool() -> list[dict]:
    """KC (QB / RB / WR / TE plus a thin-sample rookie and a kicker), BUF (running QB, RB, WR), NYJ (nobody eligible)."""
    return [
        make_player(QB_ID, "Pat Pocket", "QB", "KC", projection=320.0, passing_yards=4250.0, passing_tds=30.0,
                    pass_attempts=560.0, rushing_yards=180.0, rushing_tds=1.0, carries=45.0),
        make_player(RB_ID, "Rex Back", "RB", "KC", projection=280.0, rushing_yards=1190.0, rushing_tds=9.0, carries=260.0,
                    receptions=48.0, receiving_yards=410.0, receiving_tds=3.0, targets=60.0),
        make_player(WR_ID, "Wes Receiver", "WR", "KC", projection=270.0, receiving_yards=1300.0, receiving_tds=9.0,
                    receptions=95.0, targets=140.0),
        make_player(TE_ID, "Tex End", "TE", "KC", projection=210.0, receiving_yards=760.0, receiving_tds=6.0,
                    receptions=70.0, targets=95.0),
        make_player(ROOKIE_ID, "Rookie Wideout", "WR", "KC", games=3, projection=90.0, receiving_yards=150.0,
                    receiving_tds=1.0, receptions=12.0, targets=20.0),
        make_player("00-KC-K1", "Kyle Kicker", "K", "KC", projection=120.0),
        make_player("00-BUF-QB1", "Rob Runner", "QB", "BUF", games=16, projection=310.0, passing_yards=3700.0,
                    passing_tds=26.0, pass_attempts=480.0, rushing_yards=560.0, rushing_tds=12.0, carries=110.0),
        make_player("00-BUF-RB1", "Bill Back", "RB", "BUF", projection=230.0, rushing_yards=900.0, rushing_tds=7.0,
                    carries=210.0, receptions=30.0, receiving_yards=240.0, receiving_tds=1.0),
        make_player("00-BUF-WR1", "Buff Wideout", "WR", "BUF", projection=220.0, receiving_yards=980.0,
                    receiving_tds=6.0, receptions=70.0),
        make_player("00-NYJ-K1", "Nyj Kicker", "K", "NYJ"),
        {"player_id": "DEF-NYJ", "name": "NYJ Defense", "position": "DST", "team": "NYJ", "games_played": 17},
    ]


POOL = build_pool()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def html_of(app) -> str:
    return "\n".join(str(element.value) for element in app.markdown)


def visible_text(app) -> str:
    parts = [str(element.value) for element in app.markdown]
    parts += [str(element.value) for element in app.caption]
    parts += [str(element.value) for element in app.button]
    return "\n".join(parts)


def probabilities(app) -> tuple[float, float]:
    """``(over, under)`` read from the probability bar header."""
    found = re.findall(r'bt-prob-pct" style="[^"]*">([\d.]+)%<', html_of(app))
    assert len(found) == 2, f"expected an over/under probability bar, found {found}"
    return float(found[0]) / 100.0, float(found[1]) / 100.0


def side_table(app) -> dict[str, list[str]]:
    """Over / Under rows of the comparison table, keyed by side, as lists of cell text."""
    rows = {}
    for row in re.findall(r"<tr>(.*?)</tr>", html_of(app), flags=re.DOTALL):
        cells = [re.sub(r"<[^>]+>", "", cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)]
        if cells and cells[0].split()[0] in ("Over", "Under"):
            rows[cells[0].split()[0].lower()] = cells
    return rows


def money(text: str) -> float:
    return float(text.replace("$", "").replace(",", ""))


def verdict_heads(app) -> list[str]:
    return re.findall(r'bt-v-head">([^<]*)<', html_of(app))


def notices(app) -> list[str]:
    return re.findall(r'bt-notice-text">([^<]*)<', html_of(app))


def hero_html(app) -> str:
    """The player card's own markdown element ('' when it is not on the page)."""
    return next((str(m.value) for m in app.markdown if 'class="bt bt-card pp-hero"' in str(m.value)), "")


def reasoning_titles(app) -> list[str]:
    return re.findall(r'bt-reason-title">([^<]*)<', html_of(app))


def content_html(app) -> str:
    """The page's own markdown without the ``<style>`` blocks (whose CSS names every tone class)."""
    return "\n".join(str(m.value) for m in app.markdown if not str(m.value).lstrip().startswith("<style"))


def verdict_tone(app) -> str | None:
    """The kit tone class of the verdict banner (``steel`` neutral, ``bad`` danger, ``warn`` warning ...)."""
    found = re.findall(r'class="bt bt-verdict bt-t-(\w+)"', html_of(app))
    return found[0] if found else None


def tile_values(app) -> list[str]:
    return re.findall(r'bt-stat-value">([^<]*)<', html_of(app))


def gold_in_chart(app) -> bool:
    """Whether the distribution chart paints any region in the app's gold (the 'value' signal)."""
    charts = app.get("plotly_chart")
    if not charts:
        return False
    spec = charts[0].proto.spec
    red, green, blue = (int(SIGNAL_GOLD.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue}," in spec


class PlayerPropsPageCase(unittest.TestCase):
    """Base: offline pool, isolated availability feed, cold ``st.cache_data``."""

    pool = POOL

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self._status_path = Path(self._dir.name) / "player_status.json"
        self._status_path.write_text("{}", encoding="utf-8")
        for patcher in (
            mock.patch.object(player_status, "STATUS_PATH", self._status_path),
            mock.patch.object(npa, "load_player_pool", side_effect=lambda season=None: list(self.pool)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        player_status.clear_cache()
        self.addCleanup(player_status.clear_cache)
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def set_status(self, player_id: str, status: str) -> None:
        self._status_path.write_text(
            json.dumps({player_id: {"status": status, "last_updated": "2026-10-01T00:00:00Z", "source": "test"}}),
            encoding="utf-8",
        )
        player_status.clear_cache()

    def run_page(self) -> AppTest:
        app = AppTest.from_file(PAGE, default_timeout=120)
        app.run()
        self.assertEqual([str(e.value) for e in app.exception], [], "the page raised")
        return app

    def rerun(self, app: AppTest) -> AppTest:
        app.run()
        self.assertEqual([str(e.value) for e in app.exception], [], "the page raised")
        return app

    def set_line(self, app: AppTest, text: str, **prices) -> AppTest:
        app.text_input(key="pp_line").set_value(text)
        for key, value in prices.items():
            app.text_input(key=f"pp_{key}").set_value(value)
        return self.rerun(app)

    def assertNoLeaks(self, app: AppTest, where: str = "") -> None:
        text = visible_text(app)
        self.assertIsNone(LEAK.search(text), f"a missing-value token leaked into the page {where}: {LEAK.search(text)}")


# ---------------------------------------------------------------------------
# Render + the team -> player -> category chain
# ---------------------------------------------------------------------------
class RenderTests(PlayerPropsPageCase):
    def test_renders_with_the_fixture_pool_and_no_exceptions(self):
        app = self.run_page()
        self.assertEqual([s.key for s in app.selectbox], ["pp_team", "pp_player", "pp_category", "pp_opponent"])
        for key in ("pp_line", "pp_over_price", "pp_under_price"):
            app.text_input(key=key)
        app.number_input(key="pp_stake")
        self.assertEqual(app.selectbox(key="pp_team").value, "KC")
        self.assertEqual(app.selectbox(key="pp_player").value, QB_ID)
        self.assertEqual(app.selectbox(key="pp_category").value, "Passing Yards")
        text = html_of(app)
        self.assertIn("Pat Pocket", hero_html(app))
        self.assertIn("Model projection", text)
        self.assertIn("The model's projection".replace("'", "&#x27;"), text.replace("'", "&#x27;"))
        self.assertIn("Player Props", text)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertIn("Research and educational analysis only", text)
        self.assertNoLeaks(app)

    def test_availability_refresh_strip_matches_the_nfl_betting_page(self):
        app = self.run_page()
        self.assertEqual([b.key for b in app.button if b.key == "pp_refresh_player_status"], ["pp_refresh_player_status"])
        self.assertIn("Player status overlay", " ".join(c.value for c in app.caption))
        self.assertNoLeaks(app)

    def test_the_line_is_prefilled_with_the_model_suggestion_and_editable(self):
        app = self.run_page()
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Passing Yards")
        self.assertEqual(app.text_input(key="pp_line").value, f"{npa.suggest_line(projection):g}")
        self.set_line(app, "231.5")
        self.assertEqual(app.text_input(key="pp_line").value, "231.5")
        self.assertIn("Over 231.5", html_of(app))

    def test_team_options_come_from_the_pool_and_skip_teams_nobody_is_eligible_on(self):
        app = self.run_page()
        self.assertEqual(list(app.selectbox(key="pp_team").options), [npa.team_label(t) for t in npa.teams_in_pool(self.pool)])
        self.assertNotIn("NYJ", app.selectbox(key="pp_team").options)

    def test_changing_team_changes_the_player_options(self):
        app = self.run_page()
        kc_players = list(app.selectbox(key="pp_player").options)
        self.assertEqual(kc_players, [npa.player_label(p) for p in npa.players_for_team(self.pool, "KC")])
        app.selectbox(key="pp_team").select("BUF")
        self.rerun(app)
        buf_players = list(app.selectbox(key="pp_player").options)
        self.assertEqual(buf_players, [npa.player_label(p) for p in npa.players_for_team(self.pool, "BUF")])
        self.assertNotEqual(kc_players, buf_players)
        self.assertEqual(app.selectbox(key="pp_player").value, "00-BUF-QB1")
        self.assertIn("Rob Runner", hero_html(app))
        self.assertNotIn("Pat Pocket", hero_html(app))

    def test_category_options_match_eligible_categories_for_each_player(self):
        app = self.run_page()
        for player_id in (QB_ID, RB_ID, WR_ID, TE_ID, ROOKIE_ID):
            app.selectbox(key="pp_player").select(player_id)
            self.rerun(app)
            expected = npa.eligible_categories(npa.find_player(self.pool, player_id))
            self.assertEqual(list(app.selectbox(key="pp_category").options), expected, player_id)

    def test_default_category_follows_position(self):
        app = self.run_page()
        expected = {QB_ID: "Passing Yards", RB_ID: "Rushing Yards", WR_ID: "Receiving Yards", TE_ID: "Receiving Yards"}
        for player_id, category in expected.items():
            app.selectbox(key="pp_player").select(player_id)
            self.rerun(app)
            self.assertEqual(app.selectbox(key="pp_category").value, category, player_id)

    def test_switching_team_resets_player_and_category_to_sensible_defaults(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        app.selectbox(key="pp_team").select("BUF")
        self.rerun(app)
        self.assertEqual(app.selectbox(key="pp_player").value, "00-BUF-QB1")
        self.assertEqual(app.selectbox(key="pp_category").value, "Passing Yards")

    def test_changing_player_or_category_resets_the_line_to_the_new_suggestion(self):
        app = self.run_page()
        self.set_line(app, "199.5")
        app.selectbox(key="pp_category").select("Rushing Yards")
        self.rerun(app)
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Rushing Yards")
        self.assertEqual(app.text_input(key="pp_line").value, f"{npa.suggest_line(projection):g}")
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        projection = npa.project_stat(npa.find_player(self.pool, WR_ID), "Receiving Yards")
        self.assertEqual(app.text_input(key="pp_line").value, f"{npa.suggest_line(projection):g}")

    def test_use_suggested_line_button_restores_the_suggestion(self):
        app = self.run_page()
        suggestion = app.text_input(key="pp_line").value
        self.set_line(app, "12.5")
        next(b for b in app.button if b.key == "pp_use_suggested").click()
        self.rerun(app)
        self.assertEqual(app.text_input(key="pp_line").value, suggestion)

    def test_every_eligible_player_and_category_renders_cleanly(self):
        app = self.run_page()
        for team in npa.teams_in_pool(self.pool):
            app.selectbox(key="pp_team").select(team)
            self.rerun(app)
            for player in npa.players_for_team(self.pool, team):
                app.selectbox(key="pp_player").select(player["player_id"])
                self.rerun(app)
                for category in npa.eligible_categories(player):
                    app.selectbox(key="pp_category").select(category)
                    self.rerun(app)
                    where = f"({player['name']}, {category})"
                    self.assertTrue(verdict_heads(app), f"no verdict {where}")
                    probabilities(app)
                    self.assertNoLeaks(app, where)


# ---------------------------------------------------------------------------
# The projection card
# ---------------------------------------------------------------------------
class ProjectionCardTests(PlayerPropsPageCase):
    def test_hero_shows_the_models_projection_spread_range_and_sample(self):
        app = self.run_page()
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Passing Yards")
        hero = hero_html(app)
        self.assertIn(f"{projection['mean']:,.1f}", hero)
        text = html_of(app)
        self.assertIn("Spread (1 st. dev.)", text)
        self.assertIn(f"±{projection['stdev']:,.1f} yds", text)
        self.assertIn("Typical range, p10 to p90 (yds)", text)
        self.assertIn("Sample behind the rate", text)
        self.assertIn(f"{projection['games']:g} games", text)
        self.assertIn("Fantasy projection", text)
        self.assertIn(escape(projection["projection_basis"]), text)
        self.assertIn("KC - Kansas City Chiefs", hero)
        self.assertIn("pp-name", hero)
        self.assertIn("bt-logo", hero)  # the kit's team logo

    def test_healthy_player_gets_a_calm_chip_and_no_availability_notice(self):
        # A feed is loaded (it lists somebody else), and it says nothing about this player.
        self.set_status("nobody", "OUT")
        app = self.run_page()
        self.assertIn("No injury flag", hero_html(app))
        self.assertIn("bt-t-good", hero_html(app))
        self.assertFalse([n for n in notices(app) if n.startswith("Availability")])
        self.assertFalse([c for c in app.caption if "No live availability data is loaded" in c.value])

    def test_with_no_availability_feed_the_card_does_not_claim_a_clean_bill_of_health(self):
        # The feed is empty until Refresh Player Status runs: "No injury flag" in green would claim a check that
        # never happened (the engine's own reasoning already says "No live availability data is loaded").
        app = self.run_page()
        hero = hero_html(app)
        self.assertNotIn("No injury flag", hero)
        self.assertIn(">No status data<", hero)
        self.assertNotIn("bt-t-good", hero)
        self.assertFalse([n for n in notices(app) if n.startswith("Availability")])
        captions = [c.value for c in app.caption if "No live availability data is loaded" in c.value]
        self.assertEqual(len(captions), 1, [c.value for c in app.caption])
        self.assertIn("Refresh Player Status", captions[0])
        self.assertIn("top of this page", captions[0])
        self.assertTrue(verdict_heads(app))  # pricing goes on, with everyone treated as healthy

    def test_out_player_stays_selectable_with_a_warning_chip_and_a_what_if_banner(self):
        self.set_status(QB_ID, "OUT")
        app = self.run_page()
        options = list(app.selectbox(key="pp_player").options)
        self.assertIn("Pat Pocket (QB) · Out", options)  # still selectable, and tagged
        self.assertIn(">OUT<", hero_html(app))
        self.assertIn("bt-t-bad", hero_html(app))
        warning = [n for n in notices(app) if n.startswith("Availability")]
        self.assertEqual(len(warning), 1)
        self.assertIn("OUT", warning[0])
        self.assertTrue(any("what-if" in head for head in verdict_heads(app)), verdict_heads(app))
        over, _under = probabilities(app)
        self.assertLess(over, 0.01)  # the model gives the over essentially no chance
        self.assertNoLeaks(app)

    def test_questionable_player_shows_a_chip_a_warning_and_still_gets_a_normal_verdict(self):
        self.set_status(QB_ID, "QUESTIONABLE")
        app = self.run_page()
        self.assertIn(">QUESTIONABLE<", hero_html(app))
        self.assertIn("Raw 2025 rate", hero_html(app))
        warning = [n for n in notices(app) if n.startswith("Availability")]
        self.assertEqual(len(warning), 1)
        self.assertIn("88%", warning[0])
        self.assertFalse(any("what-if" in head for head in verdict_heads(app)))
        self.assertTrue(verdict_heads(app))

    def test_doubtful_option_label_is_tagged_in_the_dropdown(self):
        self.set_status(WR_ID, "DOUBTFUL")
        app = self.run_page()
        self.assertIn("Wes Receiver (WR) · Doubtful", list(app.selectbox(key="pp_player").options))

    def test_thin_sample_is_flagged_on_the_card(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(ROOKIE_ID)
        self.rerun(app)
        self.assertIn("Thin sample", hero_html(app))
        self.assertTrue(any("Thin sample" in n for n in notices(app)), notices(app))

    def test_count_markets_show_whole_number_ranges(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        text = html_of(app)
        self.assertIn("Typical range, p10 to p90 (rec)", text)
        self.assertRegex(text, r"\d+ to \d+")


# ---------------------------------------------------------------------------
# Pricing a line
# ---------------------------------------------------------------------------
class PricingTests(PlayerPropsPageCase):
    def test_a_higher_line_lowers_p_over_and_raises_p_under(self):
        app = self.run_page()
        previous = None
        for line in ("180.5", "230.5", "260.5", "300.5"):
            self.set_line(app, line)
            over, under = probabilities(app)
            self.assertAlmostEqual(over + under, 1.0, delta=0.002)
            if previous is not None:
                self.assertLess(over, previous[0], line)
                self.assertGreater(under, previous[1], line)
            previous = (over, under)
        self.assertGreater(probabilities(app)[1], probabilities(app)[0])

    def test_probabilities_match_the_engine_for_the_typed_line(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        expected = npa.evaluate_line(npa.find_player(self.pool, QB_ID), "Passing Yards", 240.5, pool=self.pool)
        over, under = probabilities(app)
        self.assertAlmostEqual(over, expected["probability_over"], delta=0.0006)
        self.assertAlmostEqual(under, expected["probability_under"], delta=0.0006)

    def test_prices_change_the_expected_value(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        base = side_table(app)
        self.assertEqual(base["over"][1], "-110")
        self.set_line(app, "240.5", over_price="+150", under_price="-200")
        priced = side_table(app)
        self.assertEqual(priced["over"][1], "+150")
        self.assertEqual(priced["under"][1], "-200")
        self.assertGreater(money(priced["over"][5]), money(base["over"][5]))
        self.assertLess(money(priced["under"][5]), money(base["under"][5]))
        # the break-even column moves with the price
        self.assertEqual(priced["over"][3], "40.0%")
        self.assertEqual(priced["under"][3], "66.7%")

    def test_stake_scales_expected_value(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        before = money(side_table(app)["over"][5])
        app.number_input(key="pp_stake").set_value(300)
        self.rerun(app)
        after = money(side_table(app)["over"][5])
        self.assertAlmostEqual(after, before * 3, delta=0.05)
        self.assertIn("EV on 300", html_of(app))

    def test_decimal_and_even_prices_are_accepted(self):
        app = self.run_page()
        self.set_line(app, "240.5", over_price="1.91", under_price="even")
        table = side_table(app)
        self.assertEqual(table["under"][1], "+100")
        self.assertIn(table["over"][1], ("-110", "-111"))

    def test_a_clearly_underpriced_side_gets_a_gold_lean_banner(self):
        app = self.run_page()
        self.set_line(app, "150.5")  # far below the projection: the over is a big favourite
        self.assertTrue(any(head.startswith("Lean OVER 150.5") for head in verdict_heads(app)), verdict_heads(app))
        self.assertIn("Over 150.5 (lean)", html_of(app))
        self.assertIn("bt-c-gold", html_of(app))

    def test_a_line_on_the_projection_has_no_edge_at_standard_juice(self):
        app = self.run_page()
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Passing Yards")
        self.set_line(app, f"{npa.suggest_line(projection):g}")
        self.assertIn("No edge at these prices", verdict_heads(app))

    def test_half_point_line_has_no_push_and_integer_line_shows_push_probability(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        text = html_of(app)
        self.assertNotIn("Push probability", text)
        self.assertIn("Model fair price", text)
        self.set_line(app, "240")
        text = html_of(app)
        expected = npa.evaluate_line(npa.find_player(self.pool, QB_ID), "Passing Yards", 240.0, pool=self.pool)
        self.assertGreater(expected["push_probability"], 0)
        self.assertIn("Push probability", text)
        self.assertIn(f"{expected['push_probability'] * 100:.1f}%", text)
        self.assertIn("if decided", text)
        self.assertTrue(any(n.startswith("Integer line") for n in notices(app)), notices(app))

    def test_poisson_market_prices_and_draws(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        self.set_line(app, "4.5")
        low_over, _ = probabilities(app)
        self.set_line(app, "7.5")
        high_over, _ = probabilities(app)
        self.assertGreater(low_over, high_over)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertIn("Why a Poisson count model", reasoning_titles(app))

    def test_anytime_td_is_pinned_to_half_a_point(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(RB_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Anytime TD (Rush + Rec)")
        self.rerun(app)
        self.assertEqual(app.text_input(key="pp_line").value, "0.5")
        probabilities(app)
        self.set_line(app, "1.5")
        self.assertTrue([n for n in notices(app) if "at most 0.5" in n], notices(app))
        self.assertFalse(verdict_heads(app))

    def test_extreme_lines_inside_the_allowed_range_price_without_nan(self):
        app = self.run_page()
        spec = npa.STAT_CATEGORIES["Passing Yards"]
        for line in (spec["min_line"], spec["max_line"]):
            self.set_line(app, f"{line:g}")
            self.assertTrue(verdict_heads(app), line)
            probabilities(app)
            self.assertNoLeaks(app, f"(line {line})")
        self.set_line(app, f"{spec['min_line']:g}")
        self.assertGreater(probabilities(app)[0], 0.9)
        self.set_line(app, f"{spec['max_line']:g}")
        self.assertLess(probabilities(app)[0], 0.2)

    def test_reasoning_warnings_and_methodology_are_present(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        titles = reasoning_titles(app)
        for expected in ("Per-game rate and sample", "Your line vs the projection", "Model vs market price", "What the price needs"):
            self.assertIn(expected, titles)
        self.assertGreaterEqual(len(titles), 7)
        expanders = [e for e in app.expander if e.label == "How this is computed and its limits"]
        self.assertEqual(len(expanders), 1)
        body = "\n".join(str(m.value) for m in expanders[0].markdown)
        for phrase in ("Gaussian", "Poisson", "regress", "opponent", "Availability", "Quarterback passing yards"):
            self.assertIn(phrase, body)
        # the QB passing-yards width caveat is surfaced as its own notice whenever the curve is wider than history
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Passing Yards")
        caveat = [n for n in notices(app) if n.startswith("Passing-yards spread")]
        self.assertEqual(bool(caveat), projection["cv"] > npa.OBSERVED_QB_PASSING_CV[2])
        self.assertIn("Why a Gaussian curve", titles)

    def test_the_reasoning_is_the_engines_not_the_pages(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        evaluation = npa.evaluate_line(npa.find_player(self.pool, QB_ID), "Passing Yards", 240.5, pool=self.pool)
        self.assertEqual(reasoning_titles(app), [item["title"] for item in evaluation["reasoning"]])


# ---------------------------------------------------------------------------
# Bad input and empty states
# ---------------------------------------------------------------------------
class InputAndEmptyStateTests(PlayerPropsPageCase):
    def test_blank_line_shows_the_projection_but_no_verdict(self):
        app = self.run_page()
        self.set_line(app, "")
        self.assertIn("Pat Pocket", hero_html(app))
        self.assertIn("Model projection", html_of(app))
        self.assertEqual(verdict_heads(app), [])
        self.assertNotIn('<span class="bt-prob-pct"', html_of(app))
        self.assertIn("Paste a line to see the verdict", html_of(app))
        self.assertEqual([n for n in notices(app) if "must" in n or "Enter" in n], [])
        self.assertEqual(len(app.get("plotly_chart")), 1)  # the curve is still drawn, with no line on it
        self.assertNoLeaks(app)

    def test_whitespace_only_line_counts_as_blank(self):
        app = self.run_page()
        self.set_line(app, "   ")
        self.assertEqual(verdict_heads(app), [])
        self.assertIn("Paste a line to see the verdict", html_of(app))

    def test_invalid_lines_show_a_friendly_error_and_no_verdict(self):
        app = self.run_page()
        cases = {
            "abc": "Enter the Passing Yards line as a number",
            "-5": "cannot be negative",
            "10": "must be at least 50.5",
            "999": "must be at most 450.5",
            "240.3": "half-point steps",
            "12..5": "Enter the Passing Yards line as a number",
        }
        for text, message in cases.items():
            self.set_line(app, text)
            self.assertTrue([n for n in notices(app) if message in n], f"{text!r}: {notices(app)}")
            self.assertEqual(verdict_heads(app), [], text)
            self.assertNotIn('<span class="bt-prob-pct"', html_of(app), text)
            self.assertNotIn("Traceback", visible_text(app))
            self.assertNoLeaks(app, f"({text!r})")

    def test_a_dollar_sign_comma_and_typographic_minus_are_forgiven(self):
        app = self.run_page()
        self.set_line(app, "$1,000")  # parsed as 1000, which is out of range -> friendly error, not a crash
        self.assertTrue([n for n in notices(app) if "must be at most" in n])
        self.set_line(app, "240.5", over_price="−110")
        self.assertTrue(verdict_heads(app))

    def test_invalid_prices_show_a_friendly_error_and_no_verdict(self):
        app = self.run_page()
        for field, text in (("over_price", "-50"), ("under_price", "abc"), ("over_price", "0"), ("under_price", "+99")):
            app.text_input(key="pp_over_price").set_value("")
            app.text_input(key="pp_under_price").set_value("")
            self.set_line(app, "240.5", **{field: text})
            label = "Over price" if field == "over_price" else "Under price"
            self.assertTrue([n for n in notices(app) if n.startswith(label)], f"{field}={text!r}: {notices(app)}")
            self.assertEqual(verdict_heads(app), [], f"{field}={text!r}")
            self.assertNotIn("Traceback", visible_text(app))

    def test_a_blank_price_is_silent_and_one_blank_price_gets_a_gentle_note(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        self.assertEqual([n for n in notices(app) if "price" in n.lower()], [])
        self.assertTrue(verdict_heads(app))
        self.assertTrue(any("-110" in c.value for c in app.caption))
        self.set_line(app, "240.5", over_price="-120")
        self.assertTrue(any("Under price left blank" in c.value for c in app.caption), [c.value for c in app.caption])
        self.assertEqual(side_table(app)["under"][1], "-110")

    def test_fixing_a_bad_line_brings_the_verdict_back(self):
        app = self.run_page()
        self.set_line(app, "nonsense")
        self.assertEqual(verdict_heads(app), [])
        self.set_line(app, "240.5")
        self.assertTrue(verdict_heads(app))
        self.assertEqual([n for n in notices(app) if n.startswith("Enter")], [])

    def test_empty_pool_shows_a_friendly_empty_state_with_retry(self):
        self.pool = []
        app = self.run_page()
        text = html_of(app)
        self.assertIn("Player data is unavailable", text)
        self.assertIn("Research and educational analysis only", text)
        retry = [b for b in app.button if b.key == "pp_retry"]
        self.assertEqual(len(retry), 1)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(verdict_heads(app), [])

    def test_retry_recovers_once_the_data_comes_back(self):
        self.pool = []
        app = self.run_page()
        self.assertIn("Player data is unavailable", html_of(app))
        self.pool = build_pool()
        next(b for b in app.button if b.key == "pp_retry").click()
        self.rerun(app)
        self.assertNotIn("Player data is unavailable", html_of(app))
        self.assertEqual(app.selectbox(key="pp_team").value, "KC")

    def test_an_empty_result_is_not_cached_by_the_page(self):
        self.pool = []
        self.run_page()
        self.pool = build_pool()
        app = self.run_page()  # a fresh session must see the recovered data without a manual cache clear
        self.assertEqual(app.selectbox(key="pp_team").value, "KC")

    def test_a_pool_with_nobody_eligible_is_an_empty_state(self):
        self.pool = [make_player("00-X-K1", "Only Kicker", "K", "NYJ"), {"player_id": "DEF-1", "name": "Defense", "position": "DST", "team": "NYJ", "games_played": 17}]
        app = self.run_page()
        self.assertIn("No props-eligible players found", html_of(app))
        self.assertEqual(len(app.selectbox), 0)

    def test_a_team_with_no_eligible_players_is_a_friendly_empty_state(self):
        with mock.patch.object(npa, "players_for_team", return_value=[]):
            app = self.run_page()
        self.assertIn("No props-eligible players on this team", html_of(app))
        self.assertEqual(verdict_heads(app), [])

    def test_a_player_with_no_games_gets_a_friendly_message_not_a_traceback(self):
        error = ValueError("Pat Pocket has no games played in the source season, so there is no per-game rate to project from.")
        with mock.patch.object(npa, "project_stat", side_effect=error):
            app = self.run_page()
        text = html_of(app)
        self.assertIn("No projection for this player", text)
        self.assertIn("no games played", text)
        self.assertEqual(verdict_heads(app), [])
        self.assertEqual(len(app.get("plotly_chart")), 0)

    def test_an_engine_error_while_pricing_degrades_to_a_message(self):
        real = npa.evaluate_line

        def flaky(player, category, line, *args, **kwargs):
            if abs(line - 240.5) < 1e-9:
                raise ValueError("line must be a positive number, got 240.5")
            return real(player, category, line, *args, **kwargs)

        with mock.patch.object(npa, "evaluate_line", side_effect=flaky):
            app = self.run_page()
            self.set_line(app, "240.5")
        self.assertTrue([n for n in notices(app) if "positive number" in n], notices(app))
        self.assertEqual(verdict_heads(app), [])

    def test_no_nan_or_none_text_for_messy_rows(self):
        messy = build_pool()
        messy[0]["passing_yards"] = float("nan")  # the engine treats it as missing and warns
        messy[2]["receiving_yards"] = None
        self.pool = messy
        app = self.run_page()
        self.set_line(app, "240.5")
        self.assertNoLeaks(app)
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        self.assertNoLeaks(app)


# ---------------------------------------------------------------------------
# Independent QA regressions (each one failed against the first build of the page)
# ---------------------------------------------------------------------------
class BrowserSyncTests(PlayerPropsPageCase):
    """The page must hand the browser the new Player / Stat values instead of silently deleting the old ones.

    AppTest has no browser, but each select box's proto carries ``set_value``: it is True only when the
    script assigned the value through session state, which is what makes the real select box follow. A
    deleted key leaves ``set_value`` False, and the browser keeps showing the previous team's player.
    """

    def test_changing_team_moves_the_player_and_stat_boxes(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        # The user's own pick is not something the page may re-assert.
        self.assertFalse(app.selectbox(key="pp_category").proto.set_value)
        app.selectbox(key="pp_team").select("BUF")
        self.rerun(app)
        player_box, category_box = app.selectbox(key="pp_player"), app.selectbox(key="pp_category")
        self.assertEqual((player_box.value, category_box.value), ("00-BUF-QB1", "Passing Yards"))
        self.assertTrue(player_box.proto.set_value, "the browser would keep showing the old team's player")
        self.assertTrue(category_box.proto.set_value, "the browser would keep showing the old stat")
        self.assertFalse(app.selectbox(key="pp_team").proto.set_value)
        self.assertIn("Rob Runner", hero_html(app))

    def test_changing_player_moves_the_stat_box(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(RB_ID)
        self.rerun(app)
        box = app.selectbox(key="pp_category")
        self.assertEqual(box.value, "Rushing Yards")
        self.assertTrue(box.proto.set_value, "the browser would keep showing Passing Yards for a running back")

    def test_later_runs_do_not_override_what_the_user_chose(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(RB_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receiving Yards")
        self.rerun(app)
        app = self.set_line(app, "33.5")
        self.assertEqual(app.selectbox(key="pp_category").value, "Receiving Yards")
        for key in ("pp_team", "pp_player", "pp_category"):
            self.assertFalse(app.selectbox(key=key).proto.set_value, key)

    def test_a_stale_player_from_another_team_falls_back_to_the_teams_first_player(self):
        app = self.run_page()
        app.selectbox(key="pp_team").set_value("BUF")
        app.selectbox(key="pp_player").set_value(RB_ID)  # a KC player, still in the browser's state
        self.rerun(app)
        self.assertEqual(app.selectbox(key="pp_player").value, "00-BUF-QB1")
        self.assertEqual(app.selectbox(key="pp_category").value, "Passing Yards")
        self.assertIn("Rob Runner", hero_html(app))

    def test_a_stale_category_the_player_cannot_have_falls_back_to_his_default(self):
        real = npa.eligible_categories
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receiving TDs")
        self.rerun(app)
        self.assertFalse(app.selectbox(key="pp_category").proto.set_value)
        # The market stops being offered to him (new data, a changed pool): the stored choice is now stale.
        with mock.patch.object(npa, "eligible_categories", side_effect=lambda p: [c for c in real(p) if c != "Receiving TDs"]):
            self.rerun(app)
        box = app.selectbox(key="pp_category")
        self.assertEqual(box.value, "Receiving Yards")
        self.assertTrue(box.proto.set_value)

    def test_the_line_comes_back_after_the_page_state_was_lost(self):
        # Streamlit drops a widget's state when a run never reaches it (leaving the page, an outage). The
        # remembered "line is for this player and stat" marker survives, so the box used to come back empty.
        app = self.run_page()
        suggestion = app.text_input(key="pp_line").value
        self.assertTrue(suggestion)
        self.pool = []
        next(b for b in app.button if b.key == "pp_reload").click()
        self.rerun(app)
        self.assertIn("Player data is unavailable", html_of(app))
        self.pool = build_pool()
        next(b for b in app.button if b.key == "pp_retry").click()
        self.rerun(app)
        self.assertEqual(app.text_input(key="pp_line").value, suggestion)
        self.assertTrue(verdict_heads(app))

    def test_a_line_the_user_cleared_stays_cleared(self):
        app = self.run_page()
        app = self.set_line(app, "")
        app = self.set_line(app, "")
        self.assertEqual(app.text_input(key="pp_line").value, "")
        self.assertEqual(verdict_heads(app), [])


class HonestyAndPolishTests(PlayerPropsPageCase):
    def test_an_engine_error_is_not_followed_by_a_paste_a_line_prompt(self):
        real = npa.evaluate_line

        def flaky(player, category, line, *args, **kwargs):
            if abs(line - 240.5) < 1e-9:
                raise ValueError("line must be a positive number, got 240.5")
            return real(player, category, line, *args, **kwargs)

        with mock.patch.object(npa, "evaluate_line", side_effect=flaky):
            app = self.run_page()
            self.set_line(app, "240.5")
        self.assertTrue([n for n in notices(app) if "positive number" in n])
        self.assertNotIn("Paste a line to see the verdict", html_of(app))

    def test_an_unavailable_player_is_never_marked_as_a_lean(self):
        # Count market: the chart is still drawn for a player whose projection is zero.
        for status in ("OUT", "SUSPENDED"):
            self.set_status(WR_ID, status)
            app = self.run_page()
            app.selectbox(key="pp_player").select(WR_ID)
            self.rerun(app)
            app.selectbox(key="pp_category").select("Receptions")
            self.rerun(app)
            self.set_line(app, "4.5")
            text = content_html(app)
            self.assertTrue(any("what-if" in head for head in verdict_heads(app)), status)
            self.assertNotIn("(lean)", text, status)
            self.assertNotIn("bt-t-gold", text, status)
            self.assertNotIn("bt-c-gold", text, status)
            self.assertFalse(gold_in_chart(app), status)
        # Control: the same market for a healthy player does get the gold lean treatment on a lopsided line.
        self.set_status("nobody", "OUT")
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        self.set_line(app, "1.5")
        self.assertIn("(lean)", html_of(app))
        self.assertIn("bt-t-gold", content_html(app))
        self.assertTrue(gold_in_chart(app))

    def test_no_edge_is_calm_unless_the_price_is_steep(self):
        app = self.run_page()
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Passing Yards")
        line = f"{npa.suggest_line(projection):g}"
        self.set_line(app, line)
        self.assertEqual(verdict_heads(app), ["No edge at these prices"])
        self.assertEqual(verdict_tone(app), "steel", "a plain coin flip at -110 / -110 is not an alarm")
        self.assertNotIn("Overpriced.", html_of(app).split("bt-v-detail")[1].split("</div>")[0])
        self.set_line(app, line, over_price="-300", under_price="-300")
        self.assertEqual(verdict_heads(app), ["No edge at these prices"])
        self.assertEqual(verdict_tone(app), "bad", "a price 25 points over break-even is a real warning")

    def test_count_stat_cards_do_not_borrow_the_normal_curve_rules_of_thumb(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Anytime TD (Rush + Rec)")
        self.rerun(app)
        text = html_of(app)
        self.assertNotIn("two games in three", text)
        self.assertNotIn("one game in ten lands below", text)
        self.assertIn("At least 80% of games land in this range", text)
        # A yardage card keeps the normal-curve wording, because there it is true under the model.
        app.selectbox(key="pp_category").select("Receiving Yards")
        self.rerun(app)
        text = html_of(app)
        self.assertIn("two games in three", text)
        self.assertIn("The middle 80% of games under the model", text)

    def test_the_range_tile_shows_whole_yards_so_it_fits_one_line_on_a_phone(self):
        app = self.run_page()
        self.assertIn("Typical range, p10 to p90 (yds)", html_of(app))
        tiles = tile_values(app)
        self.assertTrue([t for t in tiles if re.fullmatch(r"[\d,]+ to [\d,]+", t)], tiles)
        self.assertFalse([t for t in tiles if re.search(r"\d\.\d to \d", t)], tiles)

    def test_the_qb_width_caveat_sits_next_to_the_spread_and_is_not_repeated(self):
        projection = npa.project_stat(npa.find_player(self.pool, QB_ID), "Passing Yards")
        self.assertGreater(projection["cv"], npa.OBSERVED_QB_PASSING_CV[2])  # the fixture QB is a wide one
        app = self.run_page()
        for line in ("", "240.5"):  # with and without a verdict
            self.set_line(app, line)
            caveat = [n for n in notices(app) if n.startswith("Passing-yards spread")]
            self.assertEqual(len(caveat), 1, f"line {line!r}")
            self.assertIn("starting quarterbacks", caveat[0])
        text = html_of(app)
        self.assertIn("see the note below", text)  # the spread tile points at it
        self.assertLess(text.index("Passing-yards spread"), text.index("Price the line"))
        # A receiver's yardage never gets the quarterback caveat.
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        self.assertEqual([n for n in notices(app) if n.startswith("Passing-yards spread")], [])
        self.assertNotIn("see the note below", html_of(app))

    def test_the_qb_caveat_disappears_when_the_curve_is_within_history(self):
        with mock.patch.object(npa, "OBSERVED_QB_PASSING_CV", (0.2, 0.4, 0.9)):
            app = self.run_page()
        self.assertEqual([n for n in notices(app) if n.startswith("Passing-yards spread")], [])
        self.assertNotIn("see the note below", html_of(app))

    def test_no_qb_caveat_for_an_unavailable_quarterback_with_no_curve(self):
        self.set_status(QB_ID, "OUT")
        app = self.run_page()
        self.assertEqual([n for n in notices(app) if n.startswith("Passing-yards spread")], [])

    def test_the_card_text_quotes_the_engines_numbers_not_stale_constants(self):
        app = self.run_page()
        self.assertIn(f"about {npa.hold(npa.DEFAULT_PRICE, npa.DEFAULT_PRICE) * 100:.1f}%", html_of(app))
        body = "\n".join(str(m.value) for e in app.expander for m in e.markdown)
        for status in ("QUESTIONABLE", "DOUBTFUL"):
            share = npa.status_utils.adjust_projection_for_status(1.0, status)
            self.assertIn(f"{share:.0%}", body, status)

    def test_a_flagged_player_card_does_not_call_the_scaled_rate_raw(self):
        self.set_status(QB_ID, "QUESTIONABLE")
        app = self.run_page()
        hero = hero_html(app)
        self.assertIn("scaled for his availability flag", hero)
        self.assertNotIn("the raw per-game rate from last season", hero)
        self.assertIn("the raw per-game rate from last season", hero_html(self.run_page_for_healthy()))

    def run_page_for_healthy(self):
        self.set_status("nobody", "OUT")
        return self.run_page()

    def test_hostile_text_in_a_player_name_is_escaped_everywhere(self):
        pool = build_pool()
        pool[0]["name"] = 'Evil <b>Name</b> & "Co" <img src=x onerror=alert(1)>'
        self.pool = pool
        app = self.run_page()
        app = self.set_line(app, "240.5")
        text = html_of(app)
        self.assertNotIn("<b>Name</b>", text)
        self.assertNotIn("<img src=x", text)
        self.assertIn("&lt;b&gt;Name&lt;/b&gt;", hero_html(app))
        self.assertTrue(verdict_heads(app))

    def test_stake_extremes_price_cleanly(self):
        app = self.run_page()
        self.set_line(app, "240.5")
        for stake in (1, 1_000_000):
            app.number_input(key="pp_stake").set_value(stake)
            self.rerun(app)
            self.assertTrue(verdict_heads(app), stake)
            self.assertIn(f"EV on {stake:,}", html_of(app))
            self.assertNoLeaks(app, f"(stake {stake})")

    def test_a_far_off_line_does_not_turn_the_peak_into_a_polygon(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(ROOKIE_ID)
        self.rerun(app)
        self.set_line(app, "225.5")  # the top of the range, 5 stdevs above a 50-yard projection
        projection = npa.project_stat(npa.find_player(self.pool, ROOKIE_ID), "Receiving Yards")
        mean, stdev = projection["mean"], projection["stdev"]
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        curve = max(spec["data"], key=lambda trace: len(trace.get("x", [])))
        near_the_peak = [x for x in curve["x"] if abs(x - mean) <= 2 * stdev]
        self.assertGreaterEqual(len(near_the_peak), 200)


class PastedTextTests(PlayerPropsPageCase):
    """What a sportsbook cell looks like when pasted, and the prices / lines nobody posts (QA round 2)."""

    def test_a_decimal_comma_price_is_read_as_decimal_odds_not_plus_191(self):
        # "1,91" used to lose its comma and be priced as +191 on both sides (a -31% hold) with a gold "Lean OVER".
        app = self.run_page()
        self.set_line(app, "240.5", over_price="1,91", under_price="1,91")
        table = side_table(app)
        self.assertIn(table["over"][1], ("-110", "-111"))
        self.assertIn(table["under"][1], ("-110", "-111"))
        self.assertNotIn("+191", html_of(app))
        self.assertEqual(app.text_input(key="pp_over_price").value, "1,91")  # the box keeps what was typed

    def test_a_decimal_comma_line_is_read_as_a_decimal_line(self):
        app = self.run_page()
        self.set_line(app, "240,5")  # used to become 2405 and report "must be at most 450.5"
        self.assertIn("Over 240.5", side_table(app)["over"][0])
        self.assertTrue(verdict_heads(app))
        self.assertEqual(app.text_input(key="pp_line").value, "240,5")
        # ... and a decimal-odds number pasted into the line box is no longer read as a 191-yard line
        self.set_line(app, "1,91")
        self.assertTrue([n for n in notices(app) if "must be at least 50.5" in n], notices(app))
        self.assertEqual(verdict_heads(app), [])

    def test_sportsbook_decorations_around_a_line_are_forgiven(self):
        app = self.run_page()
        for text in ("O 240.5", "Over 240.5", "u240.5", "o/u 240.5", "240.5 yds", "240.5 yards", "Over 240.5 yds", " 240.5 YDS "):
            self.set_line(app, text)
            self.assertTrue(side_table(app)["over"][0].startswith("Over 240.5"), f"{text!r}: {notices(app)}")
            self.assertTrue(verdict_heads(app), text)
            self.assertEqual(app.text_input(key="pp_line").value, text)

    def test_counts_take_their_own_unit_words(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        for text in ("5.5 rec", "5.5 receptions", "Over 5.5 rec"):
            self.set_line(app, text)
            self.assertTrue(side_table(app)["over"][0].startswith("Over 5.5"), text)
        # "O.5" is "over .5" (half a catch), not "over, 5": the dot belongs to the number
        self.set_line(app, "O.5")
        self.assertTrue(side_table(app)["over"][0].startswith("Over 0.5"), notices(app))

    def test_text_that_is_still_not_a_number_gets_the_friendly_error(self):
        app = self.run_page()
        for text, message in (
            ("Overs 240.5", "as a number"),
            ("240.5 furlongs", "as a number"),
            ("yds", "as a number"),
            ("O", "as a number"),
            ("1,000", "must be at most"),  # a thousands separator: 1000 yards is out of range, not 1.0
        ):
            self.set_line(app, text)
            self.assertTrue([n for n in notices(app) if message in n], f"{text!r}: {notices(app)}")
            self.assertEqual(verdict_heads(app), [], text)
            self.assertNoLeaks(app, f"({text!r})")

    def test_a_price_in_brackets_is_forgiven(self):
        app = self.run_page()
        self.set_line(app, "240.5", over_price="(+120)", under_price="(-140)")
        table = side_table(app)
        self.assertEqual((table["over"][1], table["under"][1]), ("+120", "-140"))

    def test_a_price_no_sportsbook_posts_gets_a_friendly_error_and_no_verdict(self):
        app = self.run_page()
        for text in ("+1000000", "-999999", "999999999999999999", "100001"):
            self.set_line(app, "240.5", over_price=text, under_price="")
            self.assertTrue(
                [n for n in notices(app) if n.startswith("Over price") and "typo" in n], f"{text!r}: {notices(app)}"
            )
            self.assertEqual(verdict_heads(app), [], text)
            self.assertNotIn("000000", content_html(app), text)  # no 19-digit price reaches the page
            self.assertNoLeaks(app, f"({text!r})")
        self.set_line(app, "240.5", over_price="", under_price="-9999999")
        self.assertTrue([n for n in notices(app) if n.startswith("Under price") and "typo" in n], notices(app))
        # The largest price the page accepts still prices normally.
        self.set_line(app, "240.5", over_price="+100000", under_price="")
        self.assertTrue(verdict_heads(app))
        self.assertEqual(side_table(app)["over"][1], "+100000")

    def test_the_fair_price_tile_never_prints_an_absurd_number(self):
        # The engine clamps a near-certain side at 1e-6, which prices out near -99,999,900 (and -499,900 for a
        # receiver's 15.5 catches). Those are bookkeeping limits, not prices anyone would see on a board.
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        self.set_line(app, "15.5")
        self.assertIn("beyond -10,000", tile_values(app))
        self.assertFalse([t for t in tile_values(app) if re.fullmatch(r"[+-]?\d{5,}", t)], tile_values(app))
        self.set_line(app, "4.5")  # an ordinary line still shows a real price
        self.assertTrue([t for t in tile_values(app) if re.fullmatch(r"[+-]\d{3,4}", t)], tile_values(app))
        self.assertNotIn("beyond", " ".join(tile_values(app)))
        self.set_status(QB_ID, "OUT")
        app = self.run_page()
        self.assertIn("beyond -10,000", tile_values(app))
        self.assertFalse([t for t in tile_values(app) if re.fullmatch(r"[+-]?\d{5,}", t)], tile_values(app))


class StaleInputTests(PlayerPropsPageCase):
    def test_prices_do_not_follow_the_user_to_another_player_or_stat(self):
        # The line was already reset on a new player / stat; a price quoted for one market silently priced the next.
        app = self.run_page()
        self.set_line(app, "240.5", over_price="-130", under_price="+110")
        self.assertEqual(side_table(app)["over"][1], "-130")
        app.number_input(key="pp_stake").set_value(250)
        self.rerun(app)
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        self.assertEqual((app.text_input(key="pp_over_price").value, app.text_input(key="pp_under_price").value), ("", ""))
        self.assertEqual((side_table(app)["over"][1], side_table(app)["under"][1]), ("-110", "-110"))
        self.assertEqual(app.number_input(key="pp_stake").value, 250)  # the stake is the user's own preference
        self.set_line(app, "70.5", over_price="+105", under_price="-125")
        self.assertEqual(side_table(app)["over"][1], "+105")
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        self.assertEqual((app.text_input(key="pp_over_price").value, app.text_input(key="pp_under_price").value), ("", ""))
        self.set_line(app, "5.5", over_price="+105", under_price="-125")
        app.selectbox(key="pp_team").select("BUF")
        self.rerun(app)
        self.assertEqual((app.text_input(key="pp_over_price").value, app.text_input(key="pp_under_price").value), ("", ""))

    def test_prices_survive_reruns_that_do_not_change_the_market(self):
        app = self.run_page()
        self.set_line(app, "240.5", over_price="-130", under_price="+110")
        app.number_input(key="pp_stake").set_value(300)
        self.rerun(app)
        self.set_line(app, "231.5")
        next(b for b in app.button if b.key == "pp_use_suggested").click()
        self.rerun(app)
        self.assertEqual((app.text_input(key="pp_over_price").value, app.text_input(key="pp_under_price").value), ("-130", "+110"))
        self.assertEqual(side_table(app)["over"][1], "-130")


class InputChecksAndPolishTests(PlayerPropsPageCase):
    def _receptions(self) -> AppTest:
        app = self.run_page()
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        app.selectbox(key="pp_category").select("Receptions")
        self.rerun(app)
        return app

    def assertSitsUnderTheVerdict(self, app: AppTest, phrase: str) -> None:
        found = [n for n in notices(app) if phrase in n]
        self.assertEqual(len(found), 1, f"{phrase!r} should appear exactly once: {notices(app)}")
        text = content_html(app)
        at = text.index(phrase)
        self.assertGreater(at, text.index('bt-v-head"'))
        self.assertLess(at, text.index('bt-prob-pct"'), "the check must come before the probability bar")
        self.assertLess(at, text.index("Why the model prices it this way"))

    def test_a_line_far_from_the_projection_is_flagged_right_under_the_verdict(self):
        app = self._receptions()
        self.set_line(app, "15.5")  # 4.2 standard deviations above a 5.6-catch projection
        self.assertSitsUnderTheVerdict(app, "standard deviations from the projection")

    def test_a_price_typo_is_flagged_right_under_the_verdict(self):
        app = self._receptions()
        self.set_line(app, "4.5", over_price="+110", under_price="+110")  # both sides plus money: a negative hold
        self.assertSitsUnderTheVerdict(app, "imply a negative hold")
        self.set_line(app, "4.5", over_price="-5000", under_price="-110")
        self.assertSitsUnderTheVerdict(app, "carry a 50.4% hold")

    def test_other_engine_warnings_stay_in_the_reasoning_section(self):
        app = self.run_page()
        app.selectbox(key="pp_player").select(ROOKIE_ID)
        self.rerun(app)
        self.set_line(app, "54")  # an integer line on a thin sample: both notices are about the player and the push
        text = content_html(app)
        for phrase in ("Thin sample: the per-game rate", "Integer line: it can push"):
            found = [n for n in notices(app) if phrase in n]
            self.assertEqual(len(found), 1, (phrase, notices(app)))
            self.assertGreater(text.index(phrase), text.index("Why the model prices it this way"), phrase)

    def test_a_phone_hint_sits_under_the_side_table_only_when_there_is_a_table(self):
        app = self.run_page()
        text = content_html(app)
        self.assertEqual(text.count('<div class="pp-swipe">'), 1)
        self.assertGreater(text.index('<div class="pp-swipe">'), text.index("bt-tablewrap"))
        css = "\n".join(str(m.value) for m in app.markdown if str(m.value).lstrip().startswith("<style") and "pp-hero" in str(m.value))
        self.assertIn(".pp-swipe{display:none", css)  # hidden on wide screens ...
        self.assertIn("@media (max-width:900px){.pp-swipe{display:block}}", css)  # ... shown on phones and tablets
        self.assertIn(".bt-table .bt-th,.bt-table .bt-td{padding:.5rem .55rem !important}", css)
        self.set_line(app, "")
        self.assertEqual(content_html(app).count('<div class="pp-swipe">'), 0)
        self.set_line(app, "nonsense")
        self.assertEqual(content_html(app).count('<div class="pp-swipe">'), 0)

    def test_the_qb_caveat_does_not_oversell_a_line_far_from_the_projection(self):
        app = self.run_page()
        caveat = [unescape(n) for n in notices(app) if n.startswith("Passing-yards spread")]
        self.assertEqual(len(caveat), 1)
        self.assertIn("only holds if the projected mean is right", caveat[0])
        self.assertIn("last season's raw average", caveat[0])
        self.assertNotIn("if anything, understated", caveat[0])
        body = "\n".join(str(m.value) for e in app.expander for m in e.markdown)
        self.assertIn("if the projected mean is right", body)

    def test_a_one_game_sample_reads_one_game_not_one_games(self):
        self.pool = build_pool() + [
            make_player("00-KC-WR8", "One Game Wonder", "WR", "KC", games=1, projection=40.0, receiving_yards=60.0,
                        receiving_tds=1.0, receptions=5.0, targets=8.0)
        ]
        app = self.run_page()
        app.selectbox(key="pp_player").select("00-KC-WR8")
        self.rerun(app)
        text = content_html(app)
        self.assertIn("Thin sample · 1 game · 2025", hero_html(app))
        self.assertIn("1 game behind the rate", text)
        self.assertIn("1 game", tile_values(app))
        # (the engine's own reasoning list, further down, still says "1 games": see the cross-unit notes)
        self.assertNotRegex(text.split("What drives this read")[0], r"\b1 games\b")
        # a full season keeps the plural
        app.selectbox(key="pp_player").select(WR_ID)
        self.rerun(app)
        self.assertIn("17 games · 2025", hero_html(app))
        self.assertIn("17 games", tile_values(app))

    def test_the_value_gap_thresholds_read_with_the_right_grammar(self):
        app = self.run_page()
        self.assertIn("Strong value starts at 3 points, slight at 1.", html_of(app))
        body = "\n".join(str(m.value) for e in app.expander for m in e.markdown)
        self.assertIn("slight value starts at 1 point, strong at 3 points", body)
        self.assertNotIn("1 points", body)
        self.assertNotIn("1 points", html_of(app))


class SourceHygieneTests(unittest.TestCase):
    def test_no_network_or_scraping_in_the_page_source(self):
        source = Path(PAGE).read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source)

    def test_every_name_the_page_uses_is_defined(self):
        # A half-finished edit once left a helper reading an undefined constant; it would only have failed (with a
        # NameError, on the page) on the one rare path that called it. A static check catches that class of slip.
        import ast
        import builtins

        tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
        defined = set(dir(builtins)) | {"__file__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        self.assertEqual(sorted(used - defined), [])

    def test_the_page_does_no_network_work_at_import_time(self):
        # The loader sits inside a cached function behind a spinner, not at module level.
        source = Path(PAGE).read_text(encoding="utf-8")
        self.assertIn("@st.cache_data(ttl=3600, show_spinner=False)", source)
        self.assertIn("st.spinner(", source)


# ---------------------------------------------------------------------------
# Live data (one smoke test, skipped when the pool cannot be loaded)
# ---------------------------------------------------------------------------
class LiveDataSmokeTest(unittest.TestCase):
    def test_renders_with_the_real_player_pool(self):
        st.cache_data.clear()
        npa.clear_player_pool_cache()
        try:
            pool = npa.load_player_pool()
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"player pool unavailable: {error}")
        if not pool or not npa.teams_in_pool(pool):
            self.skipTest("player pool unavailable (offline)")
        try:
            app = AppTest.from_file(PAGE, default_timeout=180)
            app.run()
            self.assertEqual([str(e.value) for e in app.exception], [])
            team = app.selectbox(key="pp_team").value
            self.assertIn(team, npa.teams_in_pool(pool))
            self.assertTrue(verdict_heads(app))  # the line is pre-filled with the model's suggestion
            probabilities(app)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertIsNone(LEAK.search(visible_text(app)))
            app.text_input(key="pp_line").set_value("1")
            app.run()
            self.assertEqual([str(e.value) for e in app.exception], [])
            self.assertEqual(verdict_heads(app), [])
        finally:
            st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
