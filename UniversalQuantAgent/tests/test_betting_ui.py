"""Contract for app.betting_ui -- the Midnight Circuit component kit behind the
MoneyLines, Player Props and Parlays pages.

Everything here is offline and fast: the builders are pure HTML-string
functions, so these tests pin their *output* (escaping, finite numbers,
Streamlit-markdown safety, palette discipline, responsive/accessibility CSS)
without a browser. A single AppTest smoke run confirms a real Streamlit
session can render one of each component without raising.
"""

import math
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app import betting_ui as bt
from app.style import CIRCUIT_CYAN, MIDNIGHT_NAVY, PALETTE, SIGNAL_GOLD, SLATE_BLUE, SOFT_WHITE
from modules.nfl import NFL_TEAMS
from modules.odds_input import format_american, format_pct

try:  # markdown-it is a transitive dependency of the app's UI stack, not a pinned one
    from markdown_it import MarkdownIt
except ImportError:  # pragma: no cover
    MarkdownIt = None

HOSTILE = "<script>alert(1)</script>"
AMP = "A&B"
NAN = float("nan")
INF = float("inf")
NEG_INF = float("-inf")

# --- representative inputs, reused across tests ----------------------------------
AWAY = {
    "team": "TB", "name": "Tampa Bay Buccaneers", "subtitle": "QB B. Mayfield", "record": "3-1",
    "win_probability": 0.24, "rating": 58.2, "expected_points": 20.1,
}
HOME = {
    "team": "DAL", "name": "Dallas Cowboys", "subtitle": "QB D. Prescott", "record": "2-1-1",
    "win_probability": 0.76, "rating": 71.3, "expected_points": 29.6,
}
LEGS = [
    {"label": "Dallas Cowboys ML", "sublabel": "vs Tampa Bay", "type": "Moneyline", "probability": 0.76, "price": -535, "tone": "good"},
    {"label": "CeeDee Lamb over 78.5 rec yds", "sublabel": "17 games", "type": "Prop", "probability": 0.552, "price": -115},
    {"label": "KC -3.5", "type": "Spread", "probability": 0.51, "price": 120},
]
REASONS = [
    {"title": "Dallas outscores Tampa Bay", "detail": "29.6 vs 21.0 projected.", "tone": "pro", "team": "DAL"},
    {"title": "Tampa pass defense is top-10", "detail": "-0.04 EPA/play.", "tone": "con", "team": "TB"},
    {"title": "Rest edge", "detail": "7 days vs 6.", "tone": "neutral"},
]
CORR = [[1, 0.55, 0.3], [0.55, 1, -0.2], [0.3, -0.2, 1]]


def all_components() -> dict[str, str]:
    """One realistic render of every public builder."""
    return {
        "logo": bt.team_logo_html("DAL"),
        "matchup": bt.matchup_card_html(
            away=AWAY, home=HOME, kickoff="Sun 1:00 PM", chips=["NFC", "Dome"], favored="DAL",
            spread_text="DAL -9.5", total_text="O/U 49.5", selected=True,
        ),
        "tile": bt.game_tile_html(
            away_team="TB", home_team="DAL", kickoff="Sun 1:00 PM", status="Live", selected=True,
            subtitle="Q3 10:21", model_lean="Model: DAL 76%",
        ),
        "stat_tile": bt.stat_tile_html("Model win prob", "76.0%", "vs 71.2% implied", "gold"),
        "stat_grid": bt.stat_grid_html(
            [{"label": "A", "value": 1.5, "tone": "good"}, ("B", "2", "sub", "bad"), ("C", None)], columns=3
        ),
        "meter": bt.meter_html("Confidence", 0.72, caption="17 games"),
        "prob_bar": bt.probability_bar_html("Over 249.5", 0.56, "Under 249.5", 0.44, caption="Model"),
        "reasoning": bt.reasoning_panel_html(REASONS, intro="Three factors."),
        "verdict": bt.verdict_banner_html("Strong value", "+4.8 pts", "success"),
        "table": bt.comparison_table_html(
            [{"m": "ML", "model": {"value": 0.76, "format": "pct"}, "ev": {"value": 4.2, "format": "money", "tone": "good"}}],
            [("m", "Market"), ("model", "Model"), ("ev", "EV")], highlight_col="model",
        ),
        "risk": bt.risk_meter_html("Medium", 0.52, factors=["Correlated legs"]),
        "corr": bt.correlation_grid_html(["A", "B", "C"], CORR, caption="Same game"),
        "slip": bt.parlay_slip_html(LEGS, summary=[("Combined price", "+212")]),
        "empty_slip": bt.parlay_slip_html([]),
        "chip": bt.chip_html("Divisional", "gold"),
        "notice": bt.notice_html("Odds file not loaded.", "warn"),
        "empty": bt.empty_card_html("Pick a game", "Select one above."),
        "disclaimer": bt.disclaimer_html(),
    }


def assert_markdown_safe(test: unittest.TestCase, markup: str, name: str = "") -> None:
    """The Streamlit-markdown gotchas: no blank line, no 4-space-indented line."""
    test.assertIsInstance(markup, str, name)
    for number, line in enumerate(markup.split("\n")):
        test.assertTrue(line.strip(), f"{name}: blank line at {number}")
        test.assertIsNone(re.match(r" {4,}|\t", line), f"{name}: indented line {number}: {line[:40]!r}")


def count(markup: str, needle: str) -> int:
    return markup.count(needle)


def wcag_luminance(hex_color: str) -> float:
    """Independent WCAG relative luminance (does not reuse the module's helper)."""
    value = hex_color.lstrip("#")
    channels = []
    for index in (0, 2, 4):
        scaled = int(value[index:index + 2], 16) / 255
        channels.append(scaled / 12.92 if scaled <= 0.03928 else ((scaled + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_vs_navy(hex_color: str) -> float:
    return (wcag_luminance(hex_color) + 0.05) / (wcag_luminance(MIDNIGHT_NAVY) + 0.05)


LEAK = re.compile(r"(?i)(?<![\w-])(nan|none|inf|infinity)(?![\w-])")


class StructureTests(unittest.TestCase):
    def test_every_builder_returns_one_safe_line(self):
        for name, markup in all_components().items():
            with self.subTest(component=name):
                assert_markdown_safe(self, markup, name)
                self.assertNotIn("\n", markup, "components are single-line so Markdown cannot split them")

    def test_block_builders_start_with_a_div_so_streamlit_parses_one_html_block(self):
        # Inline-first output becomes a Markdown paragraph, where $...$ is LaTeX.
        inline = {"logo", "chip"}
        for name, markup in all_components().items():
            if name in inline:
                self.assertTrue(markup.startswith("<span"), name)
            else:
                self.assertTrue(markup.startswith("<div"), f"{name}: {markup[:30]!r}")

    @unittest.skipIf(MarkdownIt is None, "markdown-it not installed")
    def test_a_commonmark_parser_leaves_every_component_as_raw_html(self):
        parser = MarkdownIt("commonmark", {"html": True})
        block_components = {k: v for k, v in all_components().items() if k not in {"logo", "chip"}}
        block_components["css"] = bt.BETTING_CSS
        for name, markup in block_components.items():
            with self.subTest(component=name):
                rendered = parser.render(markup)
                self.assertNotIn("<pre>", rendered)
                self.assertNotIn("<code>", rendered)
                self.assertFalse(rendered.lstrip().startswith("<p>"), name)

    def test_multiline_user_text_is_collapsed_not_passed_through(self):
        out = bt.reasoning_panel_html([{"title": "Line one\n\n    line two", "detail": "a\r\n\r\nb\t\tc"}], intro="x\n\ny")
        assert_markdown_safe(self, out)
        self.assertIn("Line one line two", out)
        self.assertIn("a b c", out)

    def test_component_urls_are_only_the_espn_logo_host(self):
        joined = " ".join(all_components().values())
        for url in re.findall(r"""(?:src|href)=["']([^"']+)["']""", joined):
            self.assertRegex(url, r"^https://a\.espncdn\.com/i/teamlogos/nfl/500/[a-z]{2,3}\.png$")
        self.assertNotIn("<script", joined)
        self.assertNotIn("@import", joined)

    def test_inline_styles_only_set_namespaced_custom_properties(self):
        # A dynamic inline `color:` would fight the stylesheet's !important text colors.
        joined = " ".join(all_components().values())
        for style in re.findall(r'style="([^"]*)"', joined):
            for declaration in filter(None, (part.strip() for part in style.split(";"))):
                self.assertTrue(declaration.startswith("--bt-"), declaration)

    def test_every_class_a_builder_emits_is_defined_in_the_css(self):
        joined = " ".join(all_components().values())
        used = {token for chunk in re.findall(r'class="([^"]+)"', joined) for token in chunk.split()}
        self.assertTrue(used)
        for token in sorted(used):
            self.assertTrue(token.startswith("bt"), token)
            self.assertIn(f".{token}", bt.BETTING_CSS, f"{token} has no CSS rule")

    def test_every_component_is_idempotent(self):
        first, second = all_components(), all_components()
        self.assertEqual(first, second)

    def test_import_has_no_streamlit_side_effects_and_render_helpers_use_unsafe_html(self):
        with mock.patch.object(bt.st, "markdown") as markdown:
            bt.render("<div>x</div>")
            bt.inject_betting_css()
            bt.render(html="<div>kw</div>")  # the documented parameter name is `html`
        self.assertEqual(markdown.call_count, 3)
        self.assertEqual(markdown.call_args_list[2], mock.call("<div>kw</div>", unsafe_allow_html=True))
        self.assertEqual(markdown.call_args_list[0], mock.call("<div>x</div>", unsafe_allow_html=True))
        self.assertEqual(markdown.call_args_list[1], mock.call(bt.BETTING_CSS, unsafe_allow_html=True))


class EscapingTests(unittest.TestCase):
    """Raw tags must never survive; entities must."""

    def hostile_outputs(self) -> dict[str, str]:
        both = [HOSTILE, AMP]
        return {
            "matchup": bt.matchup_card_html(
                away={"team": "TB", "name": HOSTILE, "subtitle": AMP, "record": HOSTILE},
                home={"team": "DAL", "name": AMP, "subtitle": HOSTILE, "record": AMP},
                kickoff=HOSTILE, chips=both, favored="DAL", spread_text=AMP, total_text=HOSTILE,
            ),
            "tile": bt.game_tile_html(
                away_team="TB", home_team="DAL", kickoff=HOSTILE, status=AMP, subtitle=HOSTILE, model_lean=AMP,
            ),
            "stat_tile": bt.stat_tile_html(HOSTILE, AMP, HOSTILE),
            "stat_grid": bt.stat_grid_html([{"label": HOSTILE, "value": AMP, "sub": HOSTILE}]),
            "meter": bt.meter_html(HOSTILE, 0.5, display=AMP, caption=HOSTILE),
            "prob_bar": bt.probability_bar_html(HOSTILE, 0.4, AMP, 0.6, caption=HOSTILE),
            "reasoning": bt.reasoning_panel_html(
                [{"title": HOSTILE, "detail": AMP}, {"title": AMP, "detail": HOSTILE, "tone": "con"}],
                title=HOSTILE, intro=AMP,
            ),
            "verdict": bt.verdict_banner_html(HOSTILE, AMP, "success"),
            "table": bt.comparison_table_html(
                [{"a": HOSTILE, "b": AMP}], [("a", HOSTILE), ("b", AMP)],
            ),
            "risk": bt.risk_meter_html("High", 0.9, factors=both),
            "corr": bt.correlation_grid_html(both, [[1, 0.2], [0.2, 1]], caption=HOSTILE),
            "slip": bt.parlay_slip_html(
                [{"label": HOSTILE, "sublabel": AMP, "type": HOSTILE, "probability": 0.5, "price": -110}],
                title=AMP, summary=[(HOSTILE, AMP)],
            ),
            "chip": bt.chip_html(f"{HOSTILE} {AMP}", "gold"),
            "notice": bt.notice_html(f"{HOSTILE} {AMP}"),
            "empty": bt.empty_card_html(HOSTILE, AMP),
        }

    def test_script_tags_are_escaped_and_ampersands_become_entities(self):
        for name, markup in self.hostile_outputs().items():
            with self.subTest(component=name):
                self.assertNotIn("<script", markup)
                self.assertNotIn("</script", markup)
                self.assertIn("&lt;script", markup)
                self.assertIn("A&amp;B", markup)
                self.assertNotIn("A&B", markup)
                assert_markdown_safe(self, markup, name)

    def test_attribute_breakout_attempts_stay_inside_their_attribute(self):
        breakout = '"><img src=x onerror=alert(1)>'
        outputs = [
            bt.matchup_card_html(away={"team": "TB", "name": breakout}, home={"team": "DAL"}, kickoff=breakout),
            bt.meter_html(breakout, 0.4),
            bt.correlation_grid_html([breakout, "B"], [[1, 0.1], [0.1, 1]]),
            bt.parlay_slip_html([{"label": breakout, "type": breakout}]),
        ]
        for markup in outputs:
            self.assertNotIn('"><img', markup)
            self.assertNotIn("<img src=x", markup)
            self.assertIn("&quot;&gt;&lt;img", markup)

    def test_hostile_team_codes_colors_and_tones_never_reach_the_markup(self):
        outputs = [
            bt.team_logo_html(HOSTILE),
            bt.matchup_card_html(away={"team": HOSTILE}, home={"team": '"><b>'}, favored=HOSTILE),
            bt.game_tile_html(away_team=HOSTILE, home_team=AMP),
            bt.probability_bar_html("a", 0.5, "b", 0.5, left_color=HOSTILE, right_color="red;background:url(x)"),
            bt.stat_tile_html("a", "b", tone=HOSTILE),
            bt.verdict_banner_html("a", tone=HOSTILE),
            bt.risk_meter_html(HOSTILE, HOSTILE),
            bt.reasoning_panel_html([{"title": "a", "tone": HOSTILE, "team": HOSTILE}]),
        ]
        for markup in outputs:
            self.assertNotIn("script", markup.lower())
            self.assertNotIn("url(x)", markup)
            self.assertNotIn("alert", markup)
            self.assertNotIn("<b>", markup)


class MissingValueTests(unittest.TestCase):
    """None / NaN / inf / absent keys render an em dash (or nothing), never 'nan'."""

    def assert_clean(self, markup: str, name: str) -> None:
        # The logo's onerror handler legitimately says display='none' (a CSS keyword, not data).
        visible = re.sub(r'onerror="[^"]*"', "", markup)
        found = LEAK.search(visible)
        self.assertIsNone(found, f"{name}: leaked a non-finite token near {visible[max(0, found.start() - 40):found.end() + 20]!r}" if found else "")
        assert_markdown_safe(self, markup, name)

    def test_matchup_card_survives_every_kind_of_missing(self):
        bad = [None, NAN, INF, NEG_INF, "nan", "None", "inf", np.float64("nan"), np.float32("nan"), pd.NA]
        for value in bad:
            with self.subTest(value=repr(value)):
                markup = bt.matchup_card_html(
                    away={"team": "TB", "win_probability": value, "rating": value, "expected_points": value, "record": value, "name": value},
                    home={"team": "DAL", "win_probability": value, "rating": value, "expected_points": value, "subtitle": value},
                    kickoff=value, chips=[value, None], favored=value, spread_text=value, total_text=value,
                )
                self.assert_clean(markup, "matchup")
                self.assertEqual(count(markup, "MODEL FAVORS"), 0)
                self.assertIn("—", markup)
        for away, home in ((None, None), ({}, {}), ("DAL", "NYG"), (5, [1, 2])):
            with self.subTest(away=away, home=home):
                self.assert_clean(bt.matchup_card_html(away=away, home=home), "matchup-bare")

    def test_game_tile_stat_tiles_and_meters(self):
        self.assert_clean(bt.game_tile_html(away_team=None, home_team=NAN, kickoff=None, status=None, subtitle=NAN, model_lean=INF), "tile")
        for value in (None, NAN, INF, "nan", np.float64("nan")):
            with self.subTest(value=repr(value)):
                tile = bt.stat_tile_html(None, value, NAN)
                self.assert_clean(tile, "stat_tile")
                self.assertIn(">—<", tile)
                self.assert_clean(bt.meter_html(None, value, max_value=value, display=value, caption=value), "meter")
        grid = bt.stat_grid_html([None, 5, "x", ("only-label",), {"label": None, "value": NAN}, ("a", NAN, NAN, None)], columns=NAN)
        self.assert_clean(grid, "stat_grid")

    def test_probability_bar_and_reasoning_and_verdict(self):
        values = (None, NAN, INF, 1.5, -0.1, "nan", "abc")
        for left in values:
            for right in values:
                with self.subTest(left=repr(left), right=repr(right)):
                    self.assert_clean(bt.probability_bar_html(None, left, NAN, right, caption=None), "prob_bar")
        self.assert_clean(bt.reasoning_panel_html([{}, {"title": None}, None, 7, {"detail": NAN}], title=None, intro=NAN), "reasoning")
        self.assert_clean(bt.verdict_banner_html(None, NAN, None), "verdict")

    def test_table_risk_correlation_slip(self):
        rows = [{"a": None, "b": NAN, "c": INF}, {"a": "x"}, "not-a-row", None, {"b": np.float64("nan"), "c": {"value": NAN, "format": "pct"}}]
        self.assert_clean(bt.comparison_table_html(rows, ["a", "b", "c", "missing"], highlight_col="b"), "table")
        self.assert_clean(bt.comparison_table_html(rows, [], highlight_col=None), "table-derived")
        for tier, score in ((None, None), (NAN, INF), ("???", NAN), ("High", "abc"), (None, 5000)):
            with self.subTest(tier=repr(tier), score=repr(score)):
                self.assert_clean(bt.risk_meter_html(tier, score, factors=[None, NAN, "", "ok"]), "risk")
        matrix = [[1, None, NAN], [INF, 1], []]
        self.assert_clean(bt.correlation_grid_html([None, NAN, "C"], matrix), "corr")
        bad_legs = [{}, {"label": None, "probability": NAN, "price": INF, "type": None}, None, "x", {"price": "abc", "probability": "nan"}]
        self.assert_clean(bt.parlay_slip_html(bad_legs, title=None, summary=[(None, NAN), "bad", ("a",)]), "slip")

    def test_missing_numbers_become_an_em_dash_in_the_right_places(self):
        table = bt.comparison_table_html([{"a": "row", "b": None, "c": NAN}], ["a", "b", "c"])
        self.assertEqual(count(table, '<td class="bt-td bt-al-r bt-na">—</td>'), 2)
        slip = bt.parlay_slip_html([{"label": "L", "type": "Prop", "probability": None, "price": NAN}])
        self.assertIn('<div class="bt-leg-prob">—', slip)
        self.assertIn('<span class="bt-leg-price">—</span>', slip)
        self.assertEqual(bt.format_value(None), "—")
        self.assertEqual(bt.format_value(NAN, "pct"), "—")
        self.assertEqual(bt.format_value(INF, "money"), "—")


class FormatValueTests(unittest.TestCase):
    def test_auto_formatting(self):
        self.assertEqual(bt.format_value(3), "3")
        self.assertEqual(bt.format_value(249.0), "249")
        self.assertEqual(bt.format_value(0.5), "0.5")
        self.assertEqual(bt.format_value(0.543), "0.54")
        self.assertEqual(bt.format_value(1234.5), "1,234.5")
        self.assertEqual(bt.format_value(-0.0004), "0")
        self.assertEqual(bt.format_value("+212"), "+212")
        self.assertEqual(bt.format_value("  76.0%  "), "76.0%")
        self.assertEqual(bt.format_value(True), "True")

    def test_typed_formatting_delegates_to_the_shared_odds_helpers(self):
        self.assertEqual(bt.format_value(0.543, "pct"), format_pct(0.543))
        self.assertEqual(bt.format_value(-110, "american"), format_american(-110))
        self.assertEqual(bt.format_value(120, "american"), "+120")
        self.assertEqual(bt.format_value(4.2, "money"), "+$4.20")
        self.assertEqual(bt.format_value(-11, "money"), "-$11.00")
        self.assertEqual(bt.format_value(0.032, "signed_pct"), "+3.2 pts")
        self.assertEqual(bt.format_value(7.6, "int"), "8")
        self.assertEqual(bt.format_value(7.66, "dec1"), "7.7")


class TeamLogoTests(unittest.TestCase):
    def test_nflverse_codes_map_to_espn_slugs(self):
        self.assertIn("/nfl/500/lar.png", bt.team_logo_html("LA"))
        self.assertIn("/nfl/500/wsh.png", bt.team_logo_html("WAS"))
        self.assertIn("/nfl/500/kc.png", bt.team_logo_html("kc"))
        self.assertIn("/nfl/500/lv.png", bt.team_logo_html(" lv "))
        for alias, slug in (("LAR", "lar"), ("WSH", "wsh"), ("JAC", "jax"), ("OAK", "lv"), ("STL", "lar")):
            self.assertEqual(bt.team_logo_url(alias), f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png", alias)

    def test_every_nfl_team_gets_a_valid_logo_url(self):
        special = {"LA": "lar", "WAS": "wsh"}
        self.assertEqual(len(NFL_TEAMS), 32)
        for code in NFL_TEAMS:
            with self.subTest(team=code):
                self.assertEqual(bt.team_logo_url(code), f"https://a.espncdn.com/i/teamlogos/nfl/500/{special.get(code, code.lower())}.png")

    def test_unknown_and_blank_codes_render_a_monogram_without_an_image_request(self):
        for team, mono in (("ZZZ", "ZZZ"), ("xyz", "XYZ"), (None, "?"), ("", "?"), (NAN, "?"), (HOSTILE, "?"), ("TOOLONG", "?"), (42, "?")):
            with self.subTest(team=repr(team)):
                markup = bt.team_logo_html(team)
                self.assertNotIn("<img", markup)
                self.assertIn(f'<span class="bt-logo-mono" aria-hidden="true">{mono}</span>', markup)
                self.assertEqual(bt.team_logo_url(team), "")

    def test_known_team_layers_underlay_image_and_css_fallback_hooks(self):
        markup = bt.team_logo_html("DAL", 60)
        self.assertEqual(count(markup, "<img"), 1)
        self.assertIn('alt=""', markup)
        self.assertIn('data-code="DAL"', markup)  # the CSS ::after monogram reads this
        self.assertIn("onerror=", markup)
        self.assertIn('aria-hidden="true">DAL</span>', markup)  # always-present underlay
        self.assertLess(markup.index("bt-logo-mono"), markup.index("<img"), "underlay sits below the image")
        self.assertIn("--bt-s:60px", markup)
        self.assertIn(".bt-logo-img::after", bt.BETTING_CSS)
        self.assertIn("attr(data-code)", bt.BETTING_CSS)

    def test_size_is_clamped(self):
        self.assertIn("--bt-s:16px", bt.team_logo_html("DAL", 2))
        self.assertIn("--bt-s:160px", bt.team_logo_html("DAL", 9999))
        self.assertIn("--bt-s:44px", bt.team_logo_html("DAL", "big"))
        self.assertIn("--bt-s:44px", bt.team_logo_html("DAL", NAN))

    def test_all_team_colors_are_visible_on_navy(self):
        for code in NFL_TEAMS:
            with self.subTest(team=code):
                color = bt.team_display_color(code)
                self.assertRegex(color, r"^#[0-9A-F]{6}$")
                self.assertGreaterEqual(contrast_vs_navy(color), 3.0, f"{code} {color}")
        self.assertEqual(bt.team_display_color("ZZZ"), CIRCUIT_CYAN)
        self.assertEqual(bt.team_display_color(None), CIRCUIT_CYAN)

    def test_matchup_colors_are_always_distinguishable(self):
        def distance(a: str, b: str) -> float:
            return math.dist([int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)])

        codes = list(NFL_TEAMS) + ["ZZZ"]
        worst = min(distance(*bt.matchup_colors(away, home)) for away in codes for home in codes)
        self.assertGreaterEqual(worst, 90, "two sides of a split bar must never look like one color")
        away, home = bt.matchup_colors("SEA", "NE")  # both navy-primary teams
        self.assertGreaterEqual(distance(away, home), 90)
        away, home = bt.matchup_colors("DAL", "DAL")
        self.assertGreaterEqual(distance(away, home), 90)

    def test_team_names(self):
        self.assertEqual(bt.team_name("la"), "Los Angeles Rams")
        self.assertEqual(bt.team_name("WSH"), "Washington Commanders")
        self.assertEqual(bt.team_name("ZZZ"), "ZZZ")
        self.assertEqual(bt.team_name(None), "")


class MatchupCardTests(unittest.TestCase):
    @staticmethod
    def team_blocks(markup: str) -> list[tuple[bool, str]]:
        return [
            (bool(flag.strip()), name)
            for flag, name in re.findall(r'class="bt-mu-team( bt-fav)?">.*?<div class="bt-mu-name">([^<]*)</div>', markup)
        ]

    def card(self, **overrides) -> str:
        args = {"away": AWAY, "home": HOME, "kickoff": "Sun 1:00 PM ET", "spread_text": "DAL -9.5", "total_text": "O/U 49.5"}
        args.update(overrides)
        return bt.matchup_card_html(**args)

    def test_favored_side_is_marked_exactly_once_on_the_right_team(self):
        for favored, expected in (("DAL", 1), ("dal", 1), ("home", 1), ("TB", 0), ("tb", 0), ("away", 0)):
            with self.subTest(favored=favored):
                markup = self.card(favored=favored)
                self.assertEqual(count(markup, "MODEL FAVORS"), 1)
                self.assertEqual(len(re.findall(r"\bbt-fav\b", markup)), 1)
                blocks = self.team_blocks(markup)
                self.assertEqual([name for _, name in blocks], ["Tampa Bay Buccaneers", "Dallas Cowboys"])
                self.assertEqual([fav for fav, _ in blocks].index(True), expected)

    def test_no_badge_when_favored_is_none_blank_or_unknown(self):
        for favored in (None, "", "   ", "ZZZ", HOSTILE, NAN, 5):
            with self.subTest(favored=repr(favored)):
                markup = self.card(favored=favored)
                self.assertEqual(count(markup, "MODEL FAVORS"), 0)
                self.assertEqual(len(re.findall(r"\bbt-fav\b", markup)), 0)
                self.assertNotIn("bt-badge-fav", markup)

    def test_identical_codes_on_both_sides_still_mark_one_side(self):
        markup = bt.matchup_card_html(away={"team": "DAL"}, home={"team": "DAL"}, favored="DAL")
        self.assertEqual(count(markup, "MODEL FAVORS"), 1)

    def test_selected_adds_the_glow_class_only_when_asked(self):
        self.assertIn("bt-selected", self.card(selected=True))
        self.assertNotIn("bt-selected", self.card())

    def test_center_column_and_chips(self):
        markup = self.card(chips=["NFC", "Dome", "", None])
        for text in ("Sun 1:00 PM ET", "DAL -9.5", "O/U 49.5", "76.0%", "24.0%", "Rating <span", "20.1", "29.6"):
            self.assertIn(text, markup)
        self.assertEqual(count(markup, 'class="bt-chip bt-t-neutral"'), 2)
        self.assertEqual(count(markup, "bt-chips"), 1)
        self.assertEqual(count(self.card(chips=()), "bt-chips"), 0)
        self.assertEqual(count(self.card(chips="Dome"), 'class="bt-chip bt-t-neutral"'), 1, "a bare string is one chip, not letters")

    def test_missing_names_fall_back_to_the_nfl_team_name(self):
        markup = bt.matchup_card_html(away={"team": "la"}, home={"team": "WAS"})
        self.assertEqual([name for _, name in self.team_blocks(markup)], ["Los Angeles Rams", "Washington Commanders"])
        self.assertEqual([name for _, name in self.team_blocks(bt.matchup_card_html(away={}, home={}))], ["TBD", "TBD"])

    def test_split_bar_is_proportional_normalised_and_complements_a_missing_side(self):
        def weights(markup: str) -> list[float]:
            return [float(w) for w in re.findall(r"--bt-w:([0-9.]+)\"", markup)]

        self.assertEqual(weights(self.card()), [24.0, 76.0])
        vigged = bt.matchup_card_html(away={"team": "TB", "win_probability": 0.52}, home={"team": "DAL", "win_probability": 0.50})
        self.assertEqual(weights(vigged), [50.98, 49.02])
        one_sided = bt.matchup_card_html(away={"team": "TB", "win_probability": 0.3}, home={"team": "DAL"})
        self.assertEqual(weights(one_sided), [30.0, 70.0])
        self.assertIn("—", one_sided)
        neither = bt.matchup_card_html(away={"team": "TB"}, home={"team": "DAL"})
        self.assertIn("bt-bar-empty", neither)
        self.assertIn("Win probability unavailable", neither)

    def test_out_of_range_probabilities_are_missing_not_rescaled(self):
        markup = bt.matchup_card_html(away={"team": "TB", "win_probability": 54.0}, home={"team": "DAL", "win_probability": 0.4})
        self.assertNotIn("5400", markup)
        self.assertNotIn("54.0%", markup)
        self.assertIn("40.0%", markup)

    def test_bar_segments_use_distinct_visible_team_colors(self):
        markup = bt.matchup_card_html(away={"team": "SEA", "win_probability": 0.5}, home={"team": "NE", "win_probability": 0.5})
        colors = re.findall(r"--bt-c:(#[0-9A-Fa-f]{6});--bt-w", markup)
        self.assertEqual(len(colors), 2)
        self.assertNotEqual(colors[0].upper(), colors[1].upper())

    def test_accessible_name_and_semantics(self):
        markup = self.card()
        self.assertIn('role="group"', markup)
        self.assertIn('aria-label="Tampa Bay Buccaneers at Dallas Cowboys"', markup)
        self.assertIn('role="img" aria-label="Win probability: TB 24.0%, DAL 76.0%"', markup)

    def test_bare_string_teams_are_read_as_codes(self):
        markup = bt.matchup_card_html(away="la", home="was")
        self.assertEqual([name for _, name in self.team_blocks(markup)], ["Los Angeles Rams", "Washington Commanders"])


class GameTileTests(unittest.TestCase):
    def tile(self, **kw) -> str:
        args = {"away_team": "TB", "home_team": "dal", "kickoff": "Sun 1:00 PM", "status": "Upcoming"}
        args.update(kw)
        return bt.game_tile_html(**args)

    def test_basic_content_and_selected_state(self):
        markup = self.tile(subtitle="DAL -9.5", model_lean="Model: DAL 76%", selected=True)
        for text in ("TB", "DAL", "Sun 1:00 PM", "Upcoming", "DAL -9.5", "Model: DAL 76%", "bt-selected", "teamlogos/nfl/500/tb.png", "teamlogos/nfl/500/dal.png"):
            self.assertIn(text, markup)
        plain = self.tile()
        self.assertNotIn("bt-selected", plain)
        self.assertNotIn("bt-tile-lean", plain)
        self.assertNotIn("bt-tile-sub", plain)

    def test_status_tones(self):
        self.assertIn("bt-live-dot", self.tile(status="Live"))
        self.assertIn("bt-live-dot", self.tile(status="Q3 10:21"))
        self.assertNotIn("bt-live-dot", self.tile(status="Final 27-24"))
        self.assertIn('class="bt-chip bt-t-neutral">Final 27-24', self.tile(status="Final 27-24"))
        self.assertIn('class="bt-chip bt-t-warn">Postponed', self.tile(status="Postponed"))
        self.assertIn('class="bt-chip bt-t-steel">Upcoming', self.tile())
        self.assertNotIn("bt-chip", self.tile(status=""))

    def test_unknown_teams_and_missing_time_degrade_gracefully(self):
        markup = self.tile(away_team="???", home_team=None, kickoff="")
        self.assertEqual(count(markup, "TBD"), 3)  # two codes + the missing kickoff
        self.assertNotIn("<img", markup)


class StatTileAndGridTests(unittest.TestCase):
    def test_tile_content_tones_and_number_formatting(self):
        markup = bt.stat_tile_html("Edge", 4.8, "model minus price", "good")
        self.assertIn("bt-t-good", markup)
        self.assertIn(">4.8<", markup)
        self.assertIn("model minus price", markup)
        for tone in ("cyan", "gold", "good", "bad", "neutral"):
            self.assertIn(f"bt-t-{tone}", bt.stat_tile_html("a", "b", tone=tone))
        self.assertIn("bt-t-cyan", bt.stat_tile_html("a", "b", tone="mystery"))
        self.assertIn(">1,234.5<", bt.stat_tile_html("a", 1234.5))
        self.assertNotIn("bt-stat-sub", bt.stat_tile_html("a", "b"))

    def test_grid_accepts_dicts_and_tuples_and_counts_tiles(self):
        tiles = [{"label": "A", "value": 1, "sub": "s", "tone": "gold"}, ("B", 2), ("C", 3, "sub"), ("D", 4, "sub", "bad"), None, 7, ("only",)]
        markup = bt.stat_grid_html(tiles, columns=4)
        self.assertEqual(count(markup, 'class="bt bt-stat '), 4)
        self.assertIn("bt-t-gold", markup)
        self.assertIn("bt-t-bad", markup)
        self.assertIn("--bt-cols:4", markup)

    def test_column_count_is_clamped(self):
        for columns, expected in ((0, 1), (-3, 1), (99, 6), (3, 3), ("x", 4), (NAN, 4), (2.6, 3)):
            with self.subTest(columns=columns):
                self.assertIn(f"--bt-cols:{expected}", bt.stat_grid_html([("a", 1)], columns=columns))

    def test_empty_grid_is_valid_markup(self):
        markup = bt.stat_grid_html([])
        assert_markdown_safe(self, markup)
        self.assertEqual(count(markup, "bt-stat "), 0)


class MeterAndBarTests(unittest.TestCase):
    def test_meter_fill_is_clamped_and_display_defaults(self):
        self.assertIn("--bt-w:72.0%", bt.meter_html("Confidence", 0.72))
        self.assertIn(">72%<", bt.meter_html("Confidence", 0.72))
        self.assertIn("--bt-w:100.0%", bt.meter_html("c", 1.7))
        self.assertIn("--bt-w:0.0%", bt.meter_html("c", -1))
        scaled = bt.meter_html("Efficiency", 38, max_value=100)
        self.assertIn("--bt-w:38.0%", scaled)
        self.assertIn("38 / 100", scaled)
        self.assertIn(">custom<", bt.meter_html("c", 0.5, display="custom"))
        self.assertIn("aria-valuenow=\"38\"", scaled)

    def test_meter_missing_value_is_an_empty_track(self):
        markup = bt.meter_html("Confidence", None)
        self.assertIn("--bt-w:0%", markup)
        self.assertIn(">—<", markup)
        self.assertNotIn("aria-valuenow", markup)
        for bad_max in (0, -5, NAN, None, "x"):
            self.assertIn("--bt-w:50.0%", bt.meter_html("c", 0.5, max_value=bad_max), repr(bad_max))

    def test_meter_tone_and_caption(self):
        markup = bt.meter_html("Vol", 0.4, tone="gold", caption="CV 0.41")
        self.assertIn("bt-t-gold", markup)
        self.assertIn("CV 0.41", markup)
        self.assertNotIn("bt-meter-cap", bt.meter_html("Vol", 0.4))

    def test_probability_bar_labels_percentages_and_split(self):
        markup = bt.probability_bar_html("Over 249.5", 0.563, "Under 249.5", 0.437, caption="Model")
        for text in ("Over 249.5", "Under 249.5", "56.3%", "43.7%", "Model"):
            self.assertIn(text, markup)
        self.assertEqual(re.findall(r"--bt-w:([0-9.]+)\"", markup), ["56.30", "43.70"])

    def test_probability_bar_default_and_custom_colors(self):
        default = bt.probability_bar_html("a", 0.5, "b", 0.5)
        self.assertIn(f"--bt-c:{CIRCUIT_CYAN}", default)
        self.assertIn(f"--bt-c:{bt.STEEL_BLUE}", default)
        self.assertNotIn(SIGNAL_GOLD, default, "gold stays reserved for the favored signal")
        custom = bt.probability_bar_html("a", 0.5, "b", 0.5, left_color="#112233", right_color="#abc")
        self.assertIn("--bt-c:#112233", custom)
        self.assertIn("--bt-c:#abc", custom)
        invalid = bt.probability_bar_html("a", 0.5, "b", 0.5, left_color="red", right_color="#12")
        self.assertIn(f"--bt-c:{CIRCUIT_CYAN}", invalid)

    def test_probability_bar_one_sided_and_empty(self):
        self.assertEqual(re.findall(r"--bt-w:([0-9.]+)\"", bt.probability_bar_html("a", 0.25, "b", None)), ["25.00", "75.00"])
        empty = bt.probability_bar_html("a", None, "b", None)
        self.assertIn("bt-bar-empty", empty)
        self.assertEqual(count(empty, "—"), 2)


class ReasoningPanelTests(unittest.TestCase):
    def test_one_block_per_item_numbered_in_order(self):
        markup = bt.reasoning_panel_html(REASONS)
        self.assertEqual(count(markup, 'class="bt-reason bt-t-'), 3)
        self.assertEqual(re.findall(r'<div class="bt-reason-n">(\d+)</div>', markup), ["1", "2", "3"])
        self.assertIn("3 factors", markup)
        for text in ("Dallas outscores Tampa Bay", "29.6 vs 21.0 projected.", "Why the model leans this way"):
            self.assertIn(text, markup)

    def test_tone_classes_and_text_tags(self):
        markup = bt.reasoning_panel_html(REASONS)
        blocks = re.findall(r'class="bt-reason (bt-t-\w+)"', markup)
        self.assertEqual(blocks, ["bt-t-cyan", "bt-t-warn", "bt-t-steel"])
        for tag in ("Supports", "Caution", "Context"):
            self.assertIn(tag, markup)  # meaning is never color alone

    def test_unknown_tone_reads_neutral_and_strings_are_titles(self):
        markup = bt.reasoning_panel_html([{"title": "x", "tone": "weird"}, "plain sentence"])
        self.assertEqual(count(markup, 'class="bt-reason bt-t-steel"'), 2)
        self.assertIn("plain sentence", markup)
        self.assertIn(">2 factors<", markup)
        self.assertIn(">1 factor<", bt.reasoning_panel_html([{"title": "solo"}]))

    def test_empty_and_blank_lists_render_a_placeholder_not_an_empty_card(self):
        for items in ([], None, [{}], [None, {"title": ""}, {"detail": None}]):
            with self.subTest(items=repr(items)):
                markup = bt.reasoning_panel_html(items)
                self.assertEqual(count(markup, 'class="bt-reason '), 0)
                self.assertIn("No reasoning is available", markup)
                assert_markdown_safe(self, markup)

    def test_team_logo_title_and_intro(self):
        markup = bt.reasoning_panel_html(REASONS, title="Custom title", intro="Because.")
        self.assertIn("Custom title", markup)
        self.assertIn("Because.", markup)
        self.assertEqual(count(markup, 'class="bt-logo"'), 2)  # only the items that name a team
        self.assertNotIn("bt-panel-intro", bt.reasoning_panel_html(REASONS))

    def test_detail_only_item_uses_the_detail_as_its_headline(self):
        markup = bt.reasoning_panel_html([{"detail": "Only a detail"}])
        self.assertIn('<span class="bt-reason-title">Only a detail</span>', markup)
        self.assertEqual(count(markup, "bt-reason-detail"), 0)


class VerdictBannerTests(unittest.TestCase):
    def test_tone_mapping_matches_the_app_pill_tones(self):
        expected = {"success": ("bt-t-good", "✓"), "info": ("bt-t-cyan", ">i<"), "neutral": ("bt-t-steel", "≈"),
                    "danger": ("bt-t-bad", "✕"), "warning": ("bt-t-warn", ">!<")}
        self.assertEqual(set(expected), set(bt.VERDICT_TONES))
        for tone, (css, glyph) in expected.items():
            with self.subTest(tone=tone):
                markup = bt.verdict_banner_html("Headline", "Detail text", tone)
                self.assertIn(css, markup)
                self.assertIn(glyph, markup)
                self.assertIn('role="status"', markup)
                self.assertIn('aria-hidden="true"', markup)
        self.assertIn("bt-t-steel", bt.verdict_banner_html("h", tone="unknown"))
        self.assertIn("bt-t-steel", bt.verdict_banner_html("h"))

    def test_headline_detail_and_optional_detail(self):
        markup = bt.verdict_banner_html("Strong value", "+4.8 pts vs break-even", "success")
        self.assertIn(">Strong value<", markup)
        self.assertIn("+4.8 pts vs break-even", markup)
        self.assertNotIn("bt-v-detail", bt.verdict_banner_html("Fair price"))


class ComparisonTableTests(unittest.TestCase):
    ROWS = [
        {"market": "Moneyline DAL", "model": {"value": 0.76, "format": "pct"}, "price": {"value": -535, "format": "american"}, "ev": {"value": -3.2, "format": "money", "tone": "bad"}},
        {"market": "Moneyline TB", "model": {"value": 0.24, "format": "pct"}, "price": {"value": 400, "format": "american"}, "ev": {"value": 12.5, "format": "money", "tone": "good"}},
        {"market": "Spread", "model": 9.4, "price": None, "ev": NAN},
    ]
    COLUMNS = [("market", "Market"), ("model", "Model"), ("price", "Price"), ("ev", "EV / $100")]

    def table(self, **kw) -> str:
        return bt.comparison_table_html(self.ROWS, self.COLUMNS, **kw)

    def test_structure_counts(self):
        markup = self.table()
        self.assertEqual(count(markup, "<th "), 4)
        self.assertEqual(count(markup, "<tr>"), 1 + 3)
        self.assertEqual(count(markup, "<td "), 12)
        self.assertEqual(count(markup, 'scope="col"'), 4)
        self.assertIn("EV / $100", markup)

    def test_typed_cells_use_the_shared_formatters_and_tones(self):
        markup = self.table()
        for text in (">76.0%<", ">24.0%<", ">-535<", ">+400<", ">-$3.20<", ">+$12.50<", ">9.4<"):
            self.assertIn(text, markup)
        self.assertIn("bt-c-bad", markup)
        self.assertIn("bt-c-good", markup)

    def test_missing_cells_are_dim_dashes(self):
        markup = self.table()
        self.assertEqual(count(markup, "bt-na"), 2)  # None price, NaN ev
        self.assertEqual(count(markup, ">—<"), 2)

    def test_highlight_column_by_key_or_label_and_unknown_is_ignored(self):
        for highlight in ("model", "Model", "MODEL"):
            with self.subTest(highlight=highlight):
                markup = self.table(highlight_col=highlight)
                self.assertEqual(count(markup, "bt-hl"), 1 + 3)  # header + each row
        self.assertEqual(count(self.table(), "bt-hl"), 0)
        self.assertEqual(count(self.table(highlight_col="nope"), "bt-hl"), 0)

    def test_alignment_defaults_and_overrides(self):
        markup = self.table()
        self.assertIn('<th class="bt-th bt-al-l" scope="col">Market', markup)
        self.assertIn('<th class="bt-th bt-al-r" scope="col">Model', markup)
        custom = bt.comparison_table_html([{"a": 1, "b": 2}], [{"key": "a", "label": "A", "align": "center"}, "b"])
        self.assertIn("bt-al-c", custom)

    def test_columns_default_to_the_first_rows_keys_and_empty_inputs_are_calm(self):
        derived = bt.comparison_table_html([{"x": 1, "y": 2}], [])
        self.assertEqual(count(derived, "<th "), 2)
        for rows, columns in (([], self.COLUMNS), (None, self.COLUMNS), ([], []), ("junk", self.COLUMNS), ([None, 3], None)):
            markup = bt.comparison_table_html(rows, columns)
            self.assertIn("Nothing to compare yet", markup)
            self.assertNotIn("<table", markup)

    def test_wrapper_scrolls_on_narrow_screens(self):
        self.assertIn('class="bt bt-tablewrap"', self.table())
        self.assertRegex(bt.BETTING_CSS, r"\.bt-tablewrap\{[^}]*overflow-x:auto")


class RiskMeterTests(unittest.TestCase):
    def test_exactly_one_segment_is_lit_for_a_recognised_tier(self):
        for tier, label in (("Low", "Low"), ("medium", "Medium"), ("HIGH", "High"), ("low risk", "Low"), ("Moderate", "Medium"), ("very high", "High")):
            with self.subTest(tier=tier):
                markup = bt.risk_meter_html(tier)
                self.assertEqual(count(markup, 'class="bt-risk-seg bt-t-'), 3)
                self.assertEqual(count(markup, " bt-on\"></span>"), 1)
                self.assertIn(f'class="bt-risk-tier">{label}<', markup)
                self.assertIn(f'aria-label="Risk: {label}"', markup)

    def test_tier_colors(self):
        self.assertIn("bt-risk bt-t-good", bt.risk_meter_html("Low"))
        self.assertIn("bt-risk bt-t-warn", bt.risk_meter_html("Medium"))
        self.assertIn("bt-risk bt-t-bad", bt.risk_meter_html("High"))

    def test_tier_is_derived_from_score_when_missing_and_unrated_otherwise(self):
        self.assertIn('class="bt-risk-tier">Low<', bt.risk_meter_html(None, 0.2))
        self.assertIn('class="bt-risk-tier">Medium<', bt.risk_meter_html("n/a", 0.5))
        self.assertIn('class="bt-risk-tier">High<', bt.risk_meter_html(None, 0.9))
        unrated = bt.risk_meter_html(None, None)
        self.assertIn("Unrated", unrated)
        self.assertEqual(count(unrated, "bt-on"), 0)

    def test_score_scale_marker_and_label(self):
        fraction = bt.risk_meter_html("Medium", 0.42)
        self.assertIn("--bt-pos:42.0%", fraction)
        self.assertIn("Risk score 42 / 100", fraction)
        self.assertIn("Risk score 55 / 100", bt.risk_meter_html("Medium", 55))
        self.assertIn("--bt-pos:100.0%", bt.risk_meter_html("High", 5000))
        self.assertIn("--bt-pos:0.0%", bt.risk_meter_html("Low", -3))
        self.assertNotIn("bt-risk-mark", bt.risk_meter_html("Low"))

    def test_factor_bullets(self):
        markup = bt.risk_meter_html("High", 0.8, factors=["one", "two", "", None, NAN])
        self.assertEqual(count(markup, 'class="bt-factor"'), 2)
        self.assertEqual(count(bt.risk_meter_html("High"), "bt-factor"), 0)
        self.assertEqual(count(bt.risk_meter_html("High", factors="single"), 'class="bt-factor"'), 1)


class CorrelationGridTests(unittest.TestCase):
    @staticmethod
    def cells(markup: str) -> int:
        return len(re.findall(r'class="bt-cc[ "]', markup))

    def test_renders_n_by_n_cells_for_every_size(self):
        for size in range(0, 7):
            with self.subTest(n=size):
                labels = [f"L{i}" for i in range(size)]
                matrix = [[1.0 if i == j else 0.1 * (i + 1) for j in range(size)] for i in range(size)]
                markup = bt.correlation_grid_html(labels, matrix)
                self.assertEqual(self.cells(markup), size * size)
                assert_markdown_safe(self, markup)
                if size:
                    self.assertEqual(count(markup, "bt-diag"), size)
                    self.assertIn(f"--bt-n:{size}", markup)

    def test_n_zero_is_a_calm_placeholder(self):
        for labels, matrix in (([], []), (None, None), ([], None)):
            markup = bt.correlation_grid_html(labels, matrix)
            self.assertIn("No correlation data", markup)
            self.assertEqual(self.cells(markup), 0)

    def test_ragged_and_short_inputs_never_raise_and_fill_with_dashes(self):
        markup = bt.correlation_grid_html(["A", "B", "C"], [[1, 0.5, 0.2], [0.5], []])
        self.assertEqual(self.cells(markup), 9)
        # row 1 supplies one value (its diagonal and (1,2) are holes), row 2 none at all: 2 + 3 dashes.
        self.assertEqual(count(markup, ">—<"), 5)
        longer = bt.correlation_grid_html(["A"], [[1, 0.2], [0.2, 1]])
        self.assertEqual(self.cells(longer), 4, "N is the larger of labels and matrix rows")
        self.assertIn("#2", longer)  # missing label
        shorter = bt.correlation_grid_html(["A", "B", "C"], [[1, 0.2], [0.2, 1]])
        self.assertEqual(self.cells(shorter), 9)
        self.assertEqual(self.cells(bt.correlation_grid_html(["A", "B"], [1, 2])), 4)
        self.assertEqual(self.cells(bt.correlation_grid_html(["A", "B"], "oops")), 4)

    def test_numpy_and_dataframe_matrices(self):
        array = np.array([[1.0, 0.4], [0.4, 1.0]])
        frame = pd.DataFrame(array, columns=["A", "B"], index=["A", "B"])
        self.assertEqual(self.cells(bt.correlation_grid_html(["A", "B"], array)), 4)
        self.assertEqual(self.cells(bt.correlation_grid_html(["A", "B"], frame)), 4)
        self.assertEqual(bt.correlation_grid_html(["A", "B"], array), bt.correlation_grid_html(["A", "B"], frame))

    def test_cell_colors_encode_sign_and_strength(self):
        def background(markup: str, title_fragment: str) -> str:
            return re.search(rf'title="[^"]*{re.escape(title_fragment)}[^"]*" style="--bt-bg:([^;]+);', markup).group(1)

        markup = bt.correlation_grid_html(["A", "B", "C", "D"], [
            [1, 0.9, -0.9, 0.01],
            [0.9, 1, 0.2, 0],
            [-0.9, 0.2, 1, 0],
            [0.01, 0, 0, 1],
        ])
        strong_pos = background(markup, "A vs B")
        strong_neg = background(markup, "A vs C")
        near_zero = background(markup, "A vs D")
        self.assertTrue(strong_pos.startswith("rgba(0,200,255,"), strong_pos)  # cyan
        self.assertTrue(strong_neg.startswith("rgba(255,107,107,"), strong_neg)  # red
        self.assertTrue(near_zero.startswith("rgba(31,46,69,"), near_zero)  # slate
        weak_pos = background(markup, "B vs C")
        self.assertLess(float(weak_pos.rstrip(")").split(",")[-1]), float(strong_pos.rstrip(")").split(",")[-1]))
        self.assertTrue(background(markup, "A vs A").startswith("rgba(31,46,69,0.55"))  # diagonal is muted slate

    def test_labels_are_truncated_safely_with_the_full_text_in_the_tooltip(self):
        long = "Patrick Mahomes passing yards over 289.5"
        markup = bt.correlation_grid_html([long, "B"], [[1, 0.2], [0.2, 1]])
        self.assertIn(">Patrick Mah…<", markup)
        self.assertNotIn(f">{long}<", markup)
        self.assertIn(f'title="{long}"', markup)
        visible_text = re.sub(r"<[^>]+>", "", markup)  # tags (and their attributes) removed
        self.assertNotIn("Patrick Mahomes", visible_text)

    def test_scrolls_horizontally_and_has_a_legend(self):
        markup = bt.correlation_grid_html(["A", "B"], [[1, 0.2], [0.2, 1]], caption="Same game")
        self.assertIn("bt-corr-scroll", markup)
        self.assertRegex(bt.BETTING_CSS, r"\.bt-corr-scroll\{[^}]*overflow-x:auto")
        for text in ("Move together", "Move opposite", "Unrelated", "Same game"):
            self.assertIn(text, markup)

    def test_values_are_clamped_for_color_but_displayed(self):
        markup = bt.correlation_grid_html(["A", "B"], [[1, 7.5], [7.5, 1]])
        self.assertIn("+7.50", markup)
        alphas = [float(alpha) for alpha in re.findall(r"rgba\(\d+,\d+,\d+,([0-9.]+)\)", markup)]
        self.assertTrue(alphas)
        self.assertLessEqual(max(alphas), 1.0, "an out-of-range correlation must not overflow the color")


class ParlaySlipTests(unittest.TestCase):
    def test_one_row_per_leg_with_chips_probability_and_price(self):
        markup = bt.parlay_slip_html(LEGS)
        self.assertEqual(count(markup, 'class="bt-leg bt-t-'), 3)
        self.assertIn("3 legs", markup)
        for chip in ("Moneyline", "Prop", "Spread"):
            self.assertIn(f">{chip}</span>", markup)
        self.assertIn("76.0%", markup)
        self.assertIn("55.2%", markup)
        for price in (-535, -115, 120):
            self.assertIn(f'<span class="bt-leg-price">{format_american(price)}</span>', markup)
        self.assertIn("Your slip", markup)

    def test_type_chips_get_their_own_tones(self):
        markup = bt.parlay_slip_html(LEGS)
        self.assertIn('class="bt-chip bt-t-cyan">Moneyline', markup)
        self.assertIn('class="bt-chip bt-t-gold">Prop', markup)
        self.assertIn('class="bt-chip bt-t-steel">Spread', markup)
        loose = bt.parlay_slip_html([{"type": "moneyline"}, {"type": "ML"}, {"type": "ATS"}, {"type": "Parlay boost"}, {}])
        self.assertEqual(count(loose, ">Moneyline</span>"), 2)
        self.assertIn(">Spread</span>", loose)
        self.assertIn(">Parlay boost</span>", loose)
        self.assertIn(">Leg</span>", loose)

    def test_leg_tone_colors_the_rail_and_defaults_to_cyan(self):
        markup = bt.parlay_slip_html(LEGS)
        self.assertEqual(re.findall(r'class="bt-leg (bt-t-\w+)"', markup), ["bt-t-good", "bt-t-cyan", "bt-t-cyan"])
        self.assertIn("bt-t-cyan", bt.parlay_slip_html([{"label": "x", "tone": "weird"}]))

    def test_empty_slip_is_calm_and_has_no_rows(self):
        for legs in ([], None):
            markup = bt.parlay_slip_html(legs)
            self.assertIn("Your slip is empty", markup)
            self.assertIn("0 legs", markup)
            self.assertEqual(count(markup, 'class="bt-leg '), 0)
            assert_markdown_safe(self, markup)

    def test_singular_plural_title_and_summary_footer(self):
        self.assertIn("1 leg<", bt.parlay_slip_html(LEGS[:1]).replace("</span>", "<"))
        markup = bt.parlay_slip_html(LEGS, title="Same-game parlay", summary=[("Combined price", "+212"), ("Joint probability", 0.214)])
        self.assertIn("Same-game parlay", markup)
        self.assertIn("Combined price", markup)
        self.assertIn(">+212<", markup)
        self.assertIn(">0.21<", markup)
        self.assertEqual(count(markup, 'class="bt-slip-sum"'), 2)
        self.assertEqual(count(bt.parlay_slip_html(LEGS), "bt-slip-foot"), 0)

    def test_unlabelled_legs_get_a_position_label(self):
        markup = bt.parlay_slip_html([{"type": "Prop"}, {"type": "Prop"}])
        self.assertIn(">Leg 1<", markup)
        self.assertIn(">Leg 2<", markup)


class SmallComponentTests(unittest.TestCase):
    def test_disclaimer_text_is_exact(self):
        text = "Research and educational analysis only - not financial or wagering advice. Nothing here places a bet."
        self.assertIn(text, bt.disclaimer_html())
        self.assertEqual(count(bt.disclaimer_html(), "bt-disclaimer"), 1)

    def test_chip_notice_and_empty_card(self):
        self.assertEqual(bt.chip_html("Hi", "gold"), '<span class="bt-chip bt-t-gold">Hi</span>')
        self.assertEqual(bt.chip_html("Hi", "weird"), '<span class="bt-chip bt-t-neutral">Hi</span>')
        self.assertEqual(bt.chip_html(None), "")
        self.assertEqual(bt.chip_html("   "), "")
        self.assertIn("bt-t-cyan", bt.notice_html("x", "info"))
        self.assertIn("bt-t-warn", bt.notice_html("x", "warn"))
        self.assertIn("bt-t-warn", bt.notice_html("x", "nonsense"))
        self.assertEqual(bt.notice_html(None), "")
        self.assertEqual(bt.notice_html("  "), "")
        self.assertIn('role="note"', bt.notice_html("x"))
        empty = bt.empty_card_html("Pick a game", "Select one.")
        self.assertIn("Pick a game", empty)
        self.assertIn("Select one.", empty)
        self.assertNotIn("bt-empty-body", bt.empty_card_html("Only title"))
        self.assertIn("Nothing to show yet", bt.empty_card_html(None))


class CssTests(unittest.TestCase):
    CSS = bt.BETTING_CSS

    def test_is_one_style_block_without_markdown_hazards(self):
        self.assertTrue(self.CSS.startswith("<style>"))
        self.assertTrue(self.CSS.endswith("</style>"))
        self.assertEqual(count(self.CSS, "<style>"), 1)
        assert_markdown_safe(self, self.CSS, "css")
        self.assertNotIn("$", self.CSS, "an unresolved $token would reach the browser")
        self.assertEqual(count(self.CSS, "{"), count(self.CSS, "}"))
        self.assertEqual(count(self.CSS, "("), count(self.CSS, ")"))

    def test_respects_reduced_motion(self):
        match = re.search(r"@media \(prefers-reduced-motion:reduce\)\{(.*?)\n\}", self.CSS, re.S)
        self.assertIsNotNone(match, "missing prefers-reduced-motion block")
        block = match.group(1)
        self.assertIn("animation:none !important", block)
        self.assertIn("transition:none !important", block)
        self.assertIn("transform:none", block)

    def test_has_phone_breakpoints(self):
        self.assertIsNotNone(re.search(r"@media \(max-width:\d+px\)", self.CSS))
        self.assertIn("@media (max-width:820px)", self.CSS)  # matches the app's own mobile breakpoint
        self.assertRegex(self.CSS, r"@media \(max-width:360px\)\{[^@]*\.bt-grid\{grid-template-columns:minmax\(0,1fr\)\}")

    def test_only_palette_and_declared_constants_appear_as_hex(self):
        allowed = {value.upper() for value in PALETTE.values()} | {v.upper() for v in bt.BETTING_COLOR_CONSTANTS.values()}
        self.assertEqual(allowed & bt.PALETTE_HEX, bt.PALETTE_HEX)
        found = {match.upper() for match in re.findall(r"#[0-9A-Fa-f]{3,8}\b", self.CSS)}
        self.assertTrue(found)
        self.assertEqual(found - allowed, set(), "undeclared hex colors in BETTING_CSS")
        # Every declared constant is genuinely used (as a hex or via rgba()), so the allow-list cannot rot.
        via_rgba = {
            "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
            for r, g, b in re.findall(r"rgba\((\d+),(\d+),(\d+),", self.CSS)
        }
        self.assertEqual(allowed - (found | via_rgba), set(), "declared color never used")

    def test_rgba_colors_derive_from_allowed_hexes(self):
        allowed_rgb = set()
        for value in [*PALETTE.values(), *bt.BETTING_COLOR_CONSTANTS.values()]:
            allowed_rgb.add(tuple(int(value[i:i + 2], 16) for i in (1, 3, 5)))
        for red, green, blue, alpha in re.findall(r"rgba\((\d+),(\d+),(\d+),([0-9.]+)\)", self.CSS):
            self.assertIn((int(red), int(green), int(blue)), allowed_rgb)
            self.assertLessEqual(float(alpha), 1.0)

    def test_semantic_constants_are_distinct_valid_and_readable_on_navy(self):
        constants = bt.BETTING_COLOR_CONSTANTS
        self.assertEqual(len(set(constants.values())), len(constants))
        for name, value in constants.items():
            self.assertRegex(value, r"^#[0-9A-Fa-f]{6}$", name)
        for name in ("GOOD_GREEN", "BAD_RED", "WARN_AMBER", "STEEL_BLUE", "MUTED_TEXT", "SUBTLE_TEXT", "DIM_TEXT"):
            self.assertGreaterEqual(contrast_vs_navy(constants[name]), 4.5, f"{name} text must read on navy")
        self.assertEqual({SIGNAL_GOLD, CIRCUIT_CYAN, SLATE_BLUE, SOFT_WHITE, MIDNIGHT_NAVY} & set(constants.values()), set())

    def test_brand_colors_come_from_app_style(self):
        self.assertIn(f"{SOFT_WHITE} !important", self.CSS)
        self.assertEqual(bt.PALETTE_HEX, frozenset(value.upper() for value in PALETTE.values()))

    def test_all_class_selectors_are_namespaced(self):
        selectors = set(re.findall(r"(?<![\w.])\.([A-Za-z_][\w-]*)", re.sub(r"\([^)]*\)", "", self.CSS)))
        self.assertTrue(selectors)
        self.assertEqual({s for s in selectors if s != "bt" and not s.startswith("bt-")}, set())

    def test_no_selector_targets_the_tags_the_light_theme_recolors(self):
        # inject_app_css paints p/li/label/strong/em/h1-h6 ink; the kit must not depend on them.
        forbidden = {"p", "li", "label", "strong", "em", "h1", "h2", "h3", "h4", "h5", "h6", "a", "ul", "ol"}
        body = self.CSS[len("<style>"):-len("</style>")]
        elements = set()
        for head in re.findall(r"([^{}]+)\{", body):
            head = head.strip()
            if head.startswith("@"):
                continue
            for selector in head.split(","):
                for token in re.split(r"[\s>+~]+", selector.strip()):
                    element = re.match(r"[a-zA-Z][a-zA-Z0-9]*", token)
                    if element:
                        elements.add(element.group(0).lower())
        self.assertIn("span", elements)  # sanity: the parser really saw `.bt-corr-legend span`
        self.assertEqual(elements & forbidden, set())

    def test_streamlit_widgets_are_restyled_for_the_dark_canvas(self):
        # Streamlit 1.60 draws these with react-aria markup the app theme does not reach (white fields, red radio dot).
        for selector in (
            '[data-testid="stSelectbox"] [role="group"]',
            '[data-testid="stTextInputRootElement"]',
            '[data-testid="stNumberInputContainer"]',
            '[data-testid="stRadioOption"]',
            '[data-testid="stExpander"] summary',
            '[data-testid="stSelectboxVirtualDropdown"]',
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.CSS)
        self.assertRegex(self.CSS, r'\[data-testid="stRadioOption"\]\[data-selected="true"\][^{]*\{background:#F5C542 !important')

    def test_text_colors_are_explicit_and_important(self):
        self.assertRegex(self.CSS, r"\n\.bt\{[^}]*color:#F2F4F7 !important")
        for rule in (".bt-muted", ".bt-stat-label", ".bt-meter-label", ".bt-disclaimer", ".bt-reason-detail", ".bt-v-detail"):
            self.assertRegex(self.CSS, rf"{re.escape(rule)}\{{[^}}]*color:#[0-9A-Fa-f]{{6}} !important", rule)
        self.assertRegex(self.CSS, r"\.bt-table \.bt-td\{[^}]*color:#F2F4F7 !important")

    def test_table_cells_beat_streamlit_markdown_table_styles(self):
        self.assertRegex(self.CSS, r"\.bt-table \.bt-th,\.bt-table \.bt-td\{[^}]*border:0 !important[^}]*background:transparent !important")

    def test_no_external_resources(self):
        for needle in ("http://", "https://", "@import", "url(", "<script", "<link"):
            self.assertNotIn(needle, self.CSS)

    def test_hover_lift_is_confined_to_interactive_components(self):
        self.assertRegex(self.CSS, r"\.bt-lift:hover\{[^}]*translateY\(-2px\)")
        for builder in ("matchup", "tile"):
            self.assertIn("bt-lift", all_components()[builder])
        for builder in ("reasoning", "table", "slip", "corr", "verdict"):
            self.assertNotIn("bt-lift", all_components()[builder])


# =====================================================================================
# Independent-audit regressions. Each class below pins a defect that the original
# suite let through (and says in its docstring what the defect was).
# =====================================================================================
def hex_contrast(foreground: str, background: str) -> float:
    """WCAG contrast between two solid hex colors (independent of the module's helper)."""
    first, second = wcag_luminance(foreground), wcag_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def blend(rgba: str, backdrop: str) -> str:
    """Composite a CSS ``rgba(r,g,b,a)`` string over a solid hex backdrop."""
    red, green, blue, alpha = re.match(r"rgba\((\d+),(\d+),(\d+),([0-9.]+)\)", rgba).groups()
    base = [int(backdrop[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(int(c) * float(alpha) + base[i] * (1 - float(alpha))) for i, c in enumerate((red, green, blue))]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def cell_style(markup: str, fragment: str) -> tuple[str, str]:
    """``(--bt-bg, --bt-fg)`` of the correlation cell whose tooltip contains ``fragment``."""
    found = re.search(rf'title="[^"]*{re.escape(fragment)}[^"]*" style="--bt-bg:([^;]+);--bt-fg:([^"]+)"', markup)
    assert found, fragment
    return found.group(1), found.group(2)


class ProbabilityToleranceTests(unittest.TestCase):
    """``1 - (1 - p)`` style arithmetic can land one ulp above 1.0; that is a probability, not missing data."""

    def test_float_noise_just_outside_the_unit_interval_is_clamped_not_dropped(self):
        markup = bt.matchup_card_html(
            away={"team": "TB", "win_probability": 1.0000000000000002}, home={"team": "DAL", "win_probability": 0.0}
        )
        self.assertIn(">100.0%<", markup)
        self.assertIn(">0.0%<", markup)
        self.assertNotIn(">—<", markup)
        tiny_negative = bt.probability_bar_html("a", -1e-12, "b", 1.0)
        self.assertEqual(re.findall(r"--bt-w:([0-9.]+)\"", tiny_negative), ["0.00", "100.00"])
        self.assertIn(">0.0%<", tiny_negative)
        slip = bt.parlay_slip_html([{"label": "x", "type": "Prop", "probability": 1.0000000002, "price": -110}])
        self.assertIn("100.0%", slip)

    def test_genuinely_out_of_range_values_are_still_missing_never_rescaled(self):
        for bad in (1.001, 1.5, 54.0, -0.01, -1.0):
            with self.subTest(value=bad):
                markup = bt.matchup_card_html(away={"team": "TB", "win_probability": bad}, home={"team": "DAL", "win_probability": 0.4})
                self.assertNotIn(format_pct(bad), markup)
                self.assertIn(">—<", markup)


class CorrelationCellDisplayTests(unittest.TestCase):
    """The grid printed ``-0.00`` for a tiny negative, and white text sat on strong cyan at only 2.7:1."""

    def test_a_tiny_negative_never_prints_negative_zero(self):
        markup = bt.correlation_grid_html(["A", "B"], [[-0.0, -0.001], [-0.004, 1]])
        self.assertNotIn("-0.00", markup)
        self.assertIn(">+0.00<", markup)  # an off-diagonal tiny value keeps its explicit sign
        self.assertIn(">0.00<", markup)  # ... and a diagonal -0.0 is a plain 0.00
        self.assertIn(">-0.10<", bt.correlation_grid_html(["A", "B"], [[1, -0.1], [-0.1, 1]]))

    def test_every_cell_text_is_legible_on_its_blended_fill(self):
        for backdrop_name, backdrop in (("slate", SLATE_BLUE), ("navy", MIDNIGHT_NAVY)):
            worst_all, worst_weak = 99.0, 99.0
            for step in range(-100, 101):
                value = step / 100
                text, background, foreground = bt._corr_cell(value, False)
                ratio = hex_contrast(foreground, blend(background, backdrop))
                worst_all = min(worst_all, ratio)
                if abs(value) <= 0.5:
                    worst_weak = min(worst_weak, ratio)
            # Neither soft white nor navy can beat ~4:1 on a mid-luminance fill, so the strong cells
            # bottom out just under 4; before the fix r=+1 was 2.7:1 (white on bright cyan).
            self.assertGreaterEqual(worst_all, 3.7, f"{backdrop_name}: worst cell contrast")
            self.assertGreaterEqual(worst_weak, 4.5, f"{backdrop_name}: ordinary cells must meet AA")

    def test_strong_positive_cells_switch_to_dark_text(self):
        markup = bt.correlation_grid_html(["A", "B", "C"], [[1, 0.95, 0.3], [0.95, 1, -0.95], [0.3, -0.95, 1]])
        strong_fill, strong_text = cell_style(markup, "A vs B")
        weak_fill, weak_text = cell_style(markup, "A vs C")
        self.assertEqual(strong_text.upper(), MIDNIGHT_NAVY.upper())
        self.assertEqual(weak_text.upper(), SOFT_WHITE.upper())
        self.assertGreaterEqual(hex_contrast(strong_text, blend(strong_fill, SLATE_BLUE)), 4.5)


class LogoMonogramContrastTests(unittest.TestCase):
    """The monogram badge (the logo's failure fallback) used a luminance cut-off that left white text on orange at 3.1:1."""

    def test_monogram_text_is_the_higher_contrast_of_soft_white_and_navy_for_every_team(self):
        for code in NFL_TEAMS:
            with self.subTest(team=code):
                markup = bt.team_logo_html(code)
                primary = re.search(r"--bt-p:(#[0-9A-Fa-f]{6})", markup).group(1)
                chosen = re.search(r"--bt-fg:(#[0-9A-Fa-f]{6})", markup).group(1)
                other = MIDNIGHT_NAVY if chosen.upper() == SOFT_WHITE.upper() else SOFT_WHITE
                self.assertGreaterEqual(hex_contrast(chosen, primary), hex_contrast(other, primary))
                self.assertGreaterEqual(hex_contrast(chosen, primary), 4.0, f"{code} {primary} on {chosen}")

    def test_bright_orange_and_mid_blue_teams_get_navy_text(self):
        for code in ("CIN", "DEN"):
            self.assertIn(f"--bt-fg:{MIDNIGHT_NAVY}", bt.team_logo_html(code), code)
        self.assertIn(f"--bt-fg:{SOFT_WHITE}", bt.team_logo_html("DAL"))


class GameStatusToneTests(unittest.TestCase):
    """``startswith('ot')`` lit the live dot for ``Other``/``Otters``; ``2nd Qtr`` and ``Overtime`` were not live."""

    @staticmethod
    def live(status: str) -> bool:
        return "bt-live-dot" in bt.game_tile_html(away_team="TB", home_team="DAL", status=status)

    def test_live_labels(self):
        for status in ("Live", "live now", "In progress", "In-Progress", "Q1", "Q4 0:41", "1st Qtr", "2nd Quarter", "3rd",
                       "Quarter 2", "Halftime", "Half", "OT", "OT 2:11", "OT2", "Overtime", "End of Q2"):
            with self.subTest(status=status):
                self.assertTrue(self.live(status))

    def test_everything_else_is_not_live(self):
        for status in ("Other", "Otters", "Odds posted", "Lively", "Liverpool", "Q5", "Q10", "Upcoming", "Scheduled", "Today",
                       "Final", "Final 27-24", "Final/OT", "Postponed", "Not started", "Quarterback out", "Halfway"):
            with self.subTest(status=status):
                self.assertFalse(self.live(status))


class FormatValueTextTests(unittest.TestCase):
    """A typed kind only formats numbers; ``"76%"`` under ``pct`` used to be swallowed into an em dash."""

    def test_non_numeric_text_is_shown_as_given_under_every_kind(self):
        for kind in ("auto", "pct", "signed_pct", "american", "money", "int", "dec1", "dec2"):
            with self.subTest(kind=kind):
                self.assertEqual(bt.format_value("76%", kind), "76%")
                self.assertEqual(bt.format_value("  N/A  ", kind), "N/A")
                self.assertEqual(bt.format_value(None, kind), "—")
                self.assertEqual(bt.format_value(NAN, kind), "—")
                self.assertEqual(bt.format_value("   ", kind), "—")
        self.assertIn(">76%<", bt.comparison_table_html([{"a": {"value": "76%", "format": "pct"}}], ["a"]))

    def test_numeric_strings_are_still_formatted_by_kind(self):
        self.assertEqual(bt.format_value("0.543", "pct"), "54.3%")
        self.assertEqual(bt.format_value("-110", "american"), "-110")
        self.assertEqual(bt.format_value("4.2", "money"), "+$4.20")

    def test_kind_is_case_insensitive_and_has_friendly_aliases(self):
        self.assertEqual(bt.format_value(0.543, "PCT"), "54.3%")
        self.assertEqual(bt.format_value(0.543, " Percent "), "54.3%")
        self.assertEqual(bt.format_value(120, "odds"), "+120")
        self.assertEqual(bt.format_value(-11, "usd"), "-$11.00")
        self.assertEqual(bt.format_value(0.543, "no-such-kind"), "0.54")


class PrintedNonFiniteTests(unittest.TestCase):
    """The schedule's market columns are NaN for unlined games, so a page that writes
    ``f"O/U {total}"`` hands the kit the *string* ``"O/U nan"``. Only whole numbers were
    guarded before; the user would have read ``O/U nan`` on a card."""

    BAD = "O/U nan"

    def test_every_text_field_replaces_a_printed_token_with_an_em_dash(self):
        bad = self.BAD
        outputs = {
            "matchup": bt.matchup_card_html(
                away={"team": "TB", "name": bad, "subtitle": bad, "record": "3-nan"}, home={"team": "DAL"},
                kickoff=bad, chips=[bad], spread_text="DAL +nan", total_text=bad,
            ),
            "tile": bt.game_tile_html(away_team="TB", home_team="DAL", kickoff=bad, status="Final nan-nan", subtitle=bad, model_lean=bad),
            "stat_tile": bt.stat_tile_html(bad, bad, bad),
            "meter": bt.meter_html(bad, 0.5, display=bad, caption=bad),
            "prob_bar": bt.probability_bar_html(bad, 0.4, "-inf", 0.6, caption=bad),
            "reasoning": bt.reasoning_panel_html([{"title": bad, "detail": bad}], title=bad, intro=bad),
            "verdict": bt.verdict_banner_html(bad, bad, "info"),
            "table": bt.comparison_table_html([{"a": bad, "b": {"value": "nan%", "format": "pct"}}], [("a", bad), ("b", "<NA>")]),
            "risk": bt.risk_meter_html("High", 0.9, factors=[bad]),
            "corr": bt.correlation_grid_html([bad, "B"], [[1, 0.2], [0.2, 1]], caption=bad),
            "slip": bt.parlay_slip_html([{"label": bad, "sublabel": bad, "type": bad, "probability": 0.5, "price": -110}], title=bad, summary=[(bad, bad)]),
            "chip": bt.chip_html(bad),
            "notice": bt.notice_html(bad),
            "empty": bt.empty_card_html(bad, bad),
        }
        for name, markup in outputs.items():
            with self.subTest(component=name):
                visible = re.sub(r'onerror="[^"]*"', "", markup)  # the logo handler legitimately says display='none'
                found = LEAK.search(visible)
                self.assertIsNone(found, f"{name}: {found and visible[max(0, found.start() - 30):found.end() + 10]!r}")
                self.assertNotIn("NaT", markup)
                self.assertNotIn("&lt;NA&gt;", markup)
                self.assertIn("—", markup)
                assert_markdown_safe(self, markup, name)
        self.assertIn("O/U —", outputs["chip"])
        self.assertIn("Final —-—", outputs["tile"])

    def test_real_words_and_a_capitalised_none_are_left_alone(self):
        for text in ("finance", "information", "banana", "Nanette", "infinite", "None reported", "Infinity war", "a.nan", "1.5nan"):
            with self.subTest(text=text):
                self.assertEqual(bt.chip_html(text), f'<span class="bt-chip bt-t-neutral">{text}</span>')
        self.assertEqual(bt.chip_html("NaT now"), '<span class="bt-chip bt-t-neutral">— now</span>')
        self.assertEqual(bt.chip_html("NaN"), "")  # the whole value is the token: that is plain missing data
        self.assertEqual(bt.format_value("1.5 nan"), "1.5 —")


class TableInputShapeTests(unittest.TestCase):
    """Int-keyed rows showed every cell as missing, and a DataFrame was read as its column names."""

    def test_integer_keyed_rows_resolve_against_string_column_specs(self):
        explicit = bt.comparison_table_html([{0: "ML DAL", 1: 0.76}], [(0, "Market"), (1, "Model")])
        self.assertIn(">ML DAL<", explicit)
        self.assertIn(">0.76<", explicit)
        self.assertEqual(count(explicit, "bt-na"), 0)
        derived = bt.comparison_table_html([{0: "x", 1: 2}], [])
        self.assertEqual(count(derived, "bt-na"), 0)
        self.assertIn(">x<", derived)

    def test_dataframe_rows_and_series_rows_are_accepted(self):
        frame = pd.DataFrame([{"market": "ML DAL", "model": 0.76}, {"market": "ML TB", "model": None}])
        for rows in (frame, [frame.iloc[0], frame.iloc[1]]):
            with self.subTest(kind=type(rows).__name__):
                markup = bt.comparison_table_html(rows, [("market", "Market"), ("model", "Model")])
                self.assertEqual(count(markup, "<tr>"), 3)
                self.assertIn(">ML TB<", markup)
                self.assertEqual(count(markup, "bt-na"), 1)
        self.assertIn("Nothing to compare yet", bt.comparison_table_html(pd.DataFrame(), ["a"]))

    def test_dataframe_and_series_inputs_elsewhere(self):
        tiles = pd.DataFrame([{"label": "A", "value": 1}, {"label": "B", "value": 2, "tone": "gold"}])
        self.assertEqual(count(bt.stat_grid_html(tiles), 'class="bt bt-stat '), 2)
        reasons = pd.DataFrame([{"title": "One", "detail": "d", "tone": "pro"}, {"title": "Two", "tone": "con"}])
        self.assertEqual(count(bt.reasoning_panel_html(reasons), 'class="bt-reason '), 2)
        legs = pd.DataFrame([{"label": "L1", "type": "Prop", "probability": 0.5, "price": -110}])
        self.assertEqual(count(bt.parlay_slip_html(legs), 'class="bt-leg '), 1)
        series = bt.matchup_card_html(away=pd.Series({"team": "TB", "name": "Bucs side"}), home=pd.Series({"team": "DAL"}))
        self.assertEqual(MatchupCardTests.team_blocks(series), [(False, "Bucs side"), (False, "Dallas Cowboys")])


class TeamResolutionTests(unittest.TestCase):
    """A page naturally passes ``"Dallas Cowboys"`` or ``"Jets"``; both used to render as ``TBD`` with no logo."""

    def test_names_nicknames_and_aliases_resolve_to_the_nflverse_code(self):
        cases = {"Dallas Cowboys": "DAL", "dallas cowboys": "DAL", "Cowboys": "DAL", "Jets": "NYJ", "Rams": "LA",
                 "Los Angeles Rams": "LA", "KAN": "KC", "GNB": "GB", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",
                 "Bucs": "TB", "Washington": "WAS", "WSH": "WAS", "ARZ": "ARI", "JAC": "JAX", "OAK": "LV", "SD": "LAC",
                 "STL": "LA", "  kc  ": "KC", "49ers": "SF"}
        for given, code in cases.items():
            with self.subTest(given=given):
                self.assertEqual(bt._team_code(given), code)
                self.assertIn(f'data-code="{code}"', bt.team_logo_html(given))
                self.assertTrue(bt.team_logo_url(given).endswith(".png"))

    def test_every_nfl_team_resolves_from_every_spelling_the_app_knows(self):
        for code, (full_name, nickname, aliases) in NFL_TEAMS.items():
            for spelling in (code, code.lower(), full_name, full_name.upper(), nickname, *aliases):
                with self.subTest(code=code, spelling=spelling):
                    self.assertEqual(bt._team_code(spelling), code)

    def test_game_tile_and_matchup_card_accept_names(self):
        tile = bt.game_tile_html(away_team="Tampa Bay Buccaneers", home_team="Cowboys")
        self.assertEqual(re.findall(r'bt-tile-code">([^<]*)<', tile), ["TB", "DAL"])
        self.assertIn("teamlogos/nfl/500/dal.png", tile)
        card = bt.matchup_card_html(away={"team": "Tampa Bay Buccaneers"}, home={"team": "Cowboys"}, favored="Dallas Cowboys")
        self.assertEqual(MatchupCardTests.team_blocks(card), [(False, "Tampa Bay Buccaneers"), (True, "Dallas Cowboys")])
        self.assertEqual(count(card, "MODEL FAVORS"), 1)
        reasoning = bt.reasoning_panel_html([{"title": "x", "team": "Cowboys"}])
        self.assertIn('aria-label="Dallas Cowboys"', reasoning)

    def test_unknown_names_and_codes_keep_their_old_behaviour(self):
        self.assertEqual(bt._team_code("ZZZ"), "ZZZ")
        self.assertEqual(bt._team_code("Springfield Isotopes"), "")
        self.assertEqual(bt._team_code("TOOLONG"), "")
        self.assertEqual(bt.team_logo_url("ZZZ"), "")
        self.assertEqual(bt.team_name("Springfield Isotopes"), "")


class ResponsiveLayoutTests(unittest.TestCase):
    """Layout defects measured in a real Streamlit 1.60 session (Chromium), pinned here as markup/CSS contracts."""

    def test_the_subtitle_wraps_and_only_the_record_is_nowrap(self):
        long_subtitle = "QB Baker Mayfield (questionable - right ankle sprain)"
        markup = bt.matchup_card_html(
            away={"team": "TB", "subtitle": long_subtitle, "record": "3-1"}, home={"team": "DAL", "record": "2-1-1"}
        )
        self.assertIn(f'<div class="bt-mu-sub">{long_subtitle} · <span class="bt-nw">3-1</span></div>', markup)
        self.assertIn('<div class="bt-mu-sub"><span class="bt-nw">2-1-1</span></div>', markup)
        self.assertNotRegex(markup, r'bt-nw">[^<]*Mayfield')  # a nowrap subtitle spilled out of its column by 42px each side
        self.assertRegex(bt.BETTING_CSS, r"\.bt-mu-sub\{[^}]*overflow-wrap:anywhere")
        self.assertRegex(bt.BETTING_CSS, r"\.bt-nw\{white-space:nowrap\}")

    def test_compact_forces_the_stacked_layout_only_when_asked(self):
        default = bt.matchup_card_html(away=AWAY, home=HOME)
        compact = bt.matchup_card_html(away=AWAY, home=HOME, compact=True)
        self.assertNotIn("bt-compact", default)
        self.assertIn('class="bt bt-card bt-lift bt-matchup bt-compact"', compact)
        self.assertEqual(count(compact, "MODEL FAVORS"), 0)
        css = bt.BETTING_CSS
        self.assertRegex(css, r"\.bt-compact \.bt-mu-body\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\)\}")
        self.assertRegex(css, r"\.bt-compact \.bt-mu-mid\{[^}]*grid-column:1 / -1[^}]*grid-row:2")
        # The compact rules are the same declarations the 820px media query applies on phones.
        phone = re.search(r"@media \(max-width:820px\)\{(.*?)\n\}", css, re.S).group(1)
        self.assertIn(".bt-mu-body{grid-template-columns:repeat(2,minmax(0,1fr))}", phone)
        self.assertIn(".bt-mu-mid{grid-column:1 / -1;grid-row:2;", phone)

    def test_the_card_is_never_a_size_container(self):
        # `container-type: inline-size` on the card collapsed it to its 44px of padding inside
        # st.container(horizontal=True) (a shrink-to-fit parent), so width must come from the parent.
        self.assertNotRegex(bt.BETTING_CSS, r"container(-type)?\s*:")
        self.assertNotIn("@container", bt.BETTING_CSS)

    def test_the_badge_wraps_instead_of_spilling_out_of_a_narrow_team_block(self):
        self.assertRegex(bt.BETTING_CSS, r"\.bt-badge-fav\{[^}]*max-width:100%")
        badge_rule = re.search(r"\.bt-badge-fav\{([^}]*)\}", bt.BETTING_CSS).group(1)
        self.assertNotIn("white-space:nowrap", badge_rule)

    def test_stat_grid_adapts_to_its_container_not_just_the_window(self):
        rule = re.search(r"\.bt-grid\{([^}]*)\}", bt.BETTING_CSS).group(1)
        self.assertIn("repeat(auto-fill,minmax(max(8rem,", rule)
        self.assertIn("var(--bt-cols,4)", rule)
        self.assertIn("--bt-cols:6", bt.stat_grid_html([("a", 1)], columns=6))

    def test_chips_do_not_shrink_and_long_titles_wrap(self):
        self.assertRegex(bt.BETTING_CSS, r"\.bt-chip\{[^}]*flex:none")  # a slip's "3 legs" chip was squeezed to 24px
        wrap_rule = re.search(r"(\.bt-tile-kick,[^{]*)\{([^}]*)\}", bt.BETTING_CSS)
        self.assertIsNotNone(wrap_rule)
        for selector in (".bt-slip-title", ".bt-panel-title", ".bt-meter-label", ".bt-factor", ".bt-notice-text", ".bt-empty-body"):
            self.assertIn(selector, wrap_rule.group(1))
        self.assertIn("overflow-wrap:anywhere", wrap_rule.group(2))
        self.assertIn('<div class="bt-notice-text">', bt.notice_html("x"))


class InputRobustnessTests(unittest.TestCase):
    def test_selected_and_compact_flags_never_raise_and_nan_is_not_true(self):
        for flag in (pd.NA, NAN, None, "", 0, np.array([1, 2])):
            with self.subTest(flag=repr(flag)):
                self.assertNotIn("bt-selected", bt.matchup_card_html(away=AWAY, home=HOME, selected=flag))
                self.assertNotIn("bt-selected", bt.game_tile_html(away_team="TB", home_team="DAL", selected=flag))
                self.assertNotIn("bt-compact", bt.matchup_card_html(away=AWAY, home=HOME, compact=flag))
        for flag in (True, 1, np.bool_(True), "yes"):
            with self.subTest(flag=repr(flag)):
                self.assertIn("bt-selected", bt.game_tile_html(away_team="TB", home_team="DAL", selected=flag))

    def test_render_ignores_missing_and_blank_markup(self):
        with mock.patch.object(bt.st, "markdown") as markdown:
            for nothing in (None, NAN, pd.NA, "", "   ", "\n"):
                bt.render(nothing)
            bt.render(bt.chip_html(None))
            bt.render(bt.notice_html(""))
            self.assertEqual(markdown.call_count, 0, "st.markdown(None) would print the word None")
            bt.render("<div>ok</div>")
            bt.render(5)
        self.assertEqual(markdown.call_args_list, [mock.call("<div>ok</div>", unsafe_allow_html=True), mock.call("5", unsafe_allow_html=True)])

    def test_slip_skips_things_that_are_not_legs_and_numbers_the_rest(self):
        markup = bt.parlay_slip_html([None, "x", 7, {"type": "Prop"}, [], {}, {"type": "Spread"}])
        self.assertEqual(count(markup, 'class="bt-leg '), 3)  # three dicts are legs (even an empty one); None, str, int, [] are not
        self.assertIn("3 legs", markup)
        self.assertEqual(re.findall(r'bt-leg-label">(Leg \d+)<', markup), ["Leg 1", "Leg 2", "Leg 3"])
        self.assertIn("Your slip is empty", bt.parlay_slip_html([None, "x"]))

    def test_reasoning_accepts_common_tone_synonyms(self):
        items = [{"title": word, "tone": word} for word in ("positive", "Supports", "negative", "caution", "context", "wat")]
        self.assertEqual(re.findall(r'class="bt-reason (bt-t-\w+)"', bt.reasoning_panel_html(items)),
                         ["bt-t-cyan", "bt-t-cyan", "bt-t-warn", "bt-t-warn", "bt-t-steel", "bt-t-steel"])

    def test_meter_aria_values_keep_full_precision(self):
        markup = bt.meter_html("Big", 1234567, max_value=2000000)
        self.assertIn('aria-valuenow="1234567"', markup)
        self.assertIn('aria-valuemax="2000000"', markup)
        self.assertNotIn("e+", markup)
        self.assertIn('aria-valuenow="0.72"', bt.meter_html("c", 0.72))

    def test_risk_tier_derivation_boundaries(self):
        for score, tier in ((0, "Low"), (0.33, "Low"), (0.34, "Medium"), (0.66, "Medium"), (0.67, "High"), (33, "Low"), (34, "Medium"), (67, "High")):
            with self.subTest(score=score):
                self.assertIn(f'class="bt-risk-tier">{tier}<', bt.risk_meter_html(None, score))

    def test_hostile_text_still_cannot_break_out_after_the_audit_changes(self):
        hostile = bt.matchup_card_html(
            away={"team": "TB", "subtitle": HOSTILE, "record": AMP}, home={"team": "DAL", "subtitle": AMP}, favored=HOSTILE, compact=HOSTILE
        )
        self.assertNotIn("<script", hostile)
        self.assertIn("&lt;script&gt;", hostile)
        self.assertIn("A&amp;B", hostile)
        assert_markdown_safe(self, hostile)


class StreamlitSmokeTests(unittest.TestCase):
    SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from app import betting_ui as bt

bt.inject_betting_css()
bt.render(bt.team_logo_html("DAL"))
bt.render(bt.matchup_card_html(
    away={{"team": "TB", "name": "Tampa Bay Buccaneers", "win_probability": 0.24, "record": "3-1"}},
    home={{"team": "DAL", "name": "Dallas Cowboys", "win_probability": 0.76}},
    kickoff="Sun 1:00 PM", chips=["Dome"], favored="DAL", spread_text="DAL -9.5", total_text="O/U 49.5", selected=True))
bt.render(bt.game_tile_html(away_team="TB", home_team="DAL", kickoff="Sun 1:00 PM", status="Live", model_lean="DAL 76%"))
bt.render(bt.stat_tile_html("Edge", "+4.8 pts", "vs price", "good"))
bt.render(bt.stat_grid_html([("A", 1), ("B", "2.0", "sub", "gold")], columns=2))
bt.render(bt.meter_html("Confidence", 0.72, caption="17 games"))
bt.render(bt.probability_bar_html("Over", 0.56, "Under", 0.44))
bt.render(bt.reasoning_panel_html([{{"title": "Reason", "detail": "Because $4.20 and $5", "tone": "pro", "team": "DAL"}}]))
bt.render(bt.verdict_banner_html("Strong value", "EV +$6.10", "success"))
bt.render(bt.comparison_table_html([{{"m": "ML", "v": 0.5}}], [("m", "Market"), ("v", "Model")], highlight_col="v"))
bt.render(bt.risk_meter_html("Medium", 0.5, factors=["a"]))
bt.render(bt.correlation_grid_html(["A", "B"], [[1, 0.3], [0.3, 1]]))
bt.render(bt.parlay_slip_html([{{"label": "DAL ML", "type": "Moneyline", "probability": 0.76, "price": -535}}]))
bt.render(bt.parlay_slip_html([]))
bt.render(bt.chip_html("Chip"))
bt.render(bt.notice_html("Note"))
bt.render(bt.empty_card_html("Empty"))
bt.render(bt.disclaimer_html())
"""

    def test_a_real_session_renders_every_component_without_error(self):
        app = AppTest.from_string(self.SCRIPT.format(root=_PROJECT_ROOT), default_timeout=60).run()
        self.assertEqual([str(item.value) for item in app.exception], [])
        bodies = [element.value for element in app.markdown]
        self.assertEqual(len(bodies), 19)  # 1 stylesheet + 18 renders
        self.assertTrue(bodies[0].lstrip().startswith("<style>"))
        joined = "\n".join(bodies)
        for marker in ("bt-matchup", "MODEL FAVORS", "bt-tile", "bt-stat", "bt-meter", "bt-prob", "bt-reason", "bt-verdict",
                       "bt-table", "bt-risk", "bt-corr", "bt-slip", "bt-notice", "bt-empty", "bt-disclaimer"):
            self.assertIn(marker, joined)


if __name__ == "__main__":
    unittest.main()
