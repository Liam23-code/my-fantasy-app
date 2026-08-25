"""Offline rendering contract for the five betting pages (30-34).

Same pattern as tests/test_fantasy_pages.py: runs each real Streamlit
script through AppTest so a broken import, a widget key collision, or a
crash on real data is caught here instead of in the browser. The single
30_Betting_Engine.py page (with a sport toggle) was split into one page
per sport plus a shared cross-sport tools page -- see ui_betting_tabs.md.
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamlit.testing.v1 import AppTest

PAGES_DIR = Path(_PROJECT_ROOT) / "app" / "pages"


def _run(page_name: str):
    app = AppTest.from_file(str(PAGES_DIR / page_name), default_timeout=120)
    app.run()
    raised = [str(element.value) for element in app.exception]
    assert raised == [], f"{page_name} raised: {raised}"
    return app


class NflBettingPageTests(unittest.TestCase):
    def test_renders_three_tabs_with_no_exceptions(self):
        app = _run("30_NFL_Betting.py")
        self.assertEqual(len(app.tabs), 3)

    def test_props_table_has_real_rows(self):
        app = _run("30_NFL_Betting.py")
        props_table = next(df.value for df in app.dataframe if "Market" in df.value.columns)
        self.assertFalse(props_table.empty)
        for column in ("Player", "Market", "Line", "Edge", "Confidence", "Risk"):
            self.assertIn(column, props_table.columns)

    def test_money_lines_table_has_real_rows(self):
        app = _run("30_NFL_Betting.py")
        moneyline_table = next(df.value for df in app.dataframe if "Home" in df.value.columns)
        self.assertFalse(moneyline_table.empty)
        for column in ("Home", "Away", "Model spread (home)", "Model total"):
            self.assertIn(column, moneyline_table.columns)

    def test_parlay_tab_computes_metrics_for_default_legs(self):
        app = _run("30_NFL_Betting.py")
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertIn("Adjusted hit probability", metrics)
        self.assertIn("Adjusted EV / $100", metrics)
        self.assertIn("Confidence", metrics)

    def test_quick_add_button_populates_the_leg_picker(self):
        app = _run("30_NFL_Betting.py")
        app.number_input(key="NFL_quick_add_n").set_value(4).run()
        quick_add = next(b for b in app.button if b.label == "Quick-add")
        quick_add.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.multiselect(key="NFL_parlay_leg_picker").value), 4)

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "30_NFL_Betting.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


class NbaBettingPageTests(unittest.TestCase):
    def test_renders_three_tabs_with_no_exceptions(self):
        app = _run("31_NBA_Betting.py")
        self.assertEqual(len(app.tabs), 3)

    def test_props_table_has_real_rows(self):
        # NBA props are only priced against today's real live schedule (see
        # matchup_engine.md) -- off-season/no-game-day correctly falls back
        # to the raw default-line table (Player/Category/Line/Basis) instead
        # of a priced one. Assert whichever real state is currently true
        # rather than assuming a game is on today.
        app = _run("31_NBA_Betting.py")
        props_table = next(df.value for df in app.dataframe if "Category" in df.value.columns)
        self.assertFalse(props_table.empty)
        priced_columns = {"Player", "Category", "Line", "EV / $100", "Risk"}
        raw_columns = {"Player", "Category", "Line", "Basis"}
        self.assertTrue(
            priced_columns.issubset(props_table.columns) or raw_columns.issubset(props_table.columns),
            msg=f"unexpected props table shape: {list(props_table.columns)}",
        )

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "31_NBA_Betting.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


class CfbBettingPageTests(unittest.TestCase):
    def test_renders_three_tabs_with_no_exceptions(self):
        app = _run("32_CFB_Betting.py")
        self.assertEqual(len(app.tabs), 3)

    def test_no_key_configured_shows_the_disclosed_empty_state(self):
        app = _run("32_CFB_Betting.py")
        text = "\n".join(str(el.value) for el in list(app.caption) + list(app.warning) + list(app.markdown))
        self.assertIn("CFBD_API_KEY", text)

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "32_CFB_Betting.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


class CbbBettingPageTests(unittest.TestCase):
    def test_renders_three_tabs_with_no_exceptions(self):
        app = _run("33_CBB_Betting.py")
        self.assertEqual(len(app.tabs), 3)

    def test_props_table_has_real_rows_and_every_row_discloses_real_basis(self):
        app = _run("33_CBB_Betting.py")
        props_table = next(df.value for df in app.dataframe if "Basis" in df.value.columns)
        self.assertFalse(props_table.empty)
        for column in ("Player", "Category", "Line", "Edge", "Risk", "Basis"):
            self.assertIn(column, props_table.columns)
        self.assertTrue((props_table["Basis"].str.contains("real", case=False)).all())

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "33_CBB_Betting.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


class CrossSportToolsPageTests(unittest.TestCase):
    def test_renders_two_tabs_with_no_exceptions(self):
        app = _run("34_Cross_Sport_Tools.py")
        self.assertEqual(len(app.tabs), 2)

    def test_sport_toggle_offers_all_four_sports_for_comparison(self):
        app = _run("34_Cross_Sport_Tools.py")
        radio = app.radio(key="cross_sport_tools_compare_sport")
        self.assertEqual(list(radio.options), ["NFL", "NBA", "CFB", "CBB"])

    def test_player_comparison_tab_renders_a_valid_side_by_side_table(self):
        app = _run("34_Cross_Sport_Tools.py")
        comparison_tables = [df.value for df in app.dataframe if "Metric" in df.value.columns]
        self.assertTrue(comparison_tables, "expected a comparison table with a 'Metric' column")
        table = comparison_tables[0]
        self.assertIn("Line", table["Metric"].values)
        self.assertEqual(len(table.columns), 3)  # Metric + two selected props

    def test_cfb_and_cbb_switch_the_comparison_sport_without_exceptions(self):
        for college_sport in ("CFB", "CBB"):
            app = _run("34_Cross_Sport_Tools.py")
            app.radio(key="cross_sport_tools_compare_sport").set_value(college_sport).run()
            self.assertEqual([str(e.value) for e in app.exception], [], msg=f"{college_sport} comparison raised")

    def test_cross_sport_tab_load_button_does_not_raise(self):
        # NBA has no real games most of the year (this test runs whenever CI
        # runs it) -- when NBA has nothing to price, the tab correctly shows
        # an empty state rather than a working leg picker. This test asserts
        # the load-and-render path itself never raises, in either state;
        # test_unified_parlay_engine.py covers the mixed-parlay math directly
        # with real leg data, independent of live NBA schedule availability.
        app = _run("34_Cross_Sport_Tools.py")
        load_button = next(b for b in app.button if "cross-sport parlay" in b.label.lower())
        load_button.click().run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        try:
            picker = app.multiselect(key="cross_sport_leg_picker")
        except KeyError:
            return  # NBA had nothing to price today -- empty state rendered instead, which is correct
        self.assertGreaterEqual(len(picker.options), 2)

    def test_cfb_and_cbb_included_in_cross_sport_options(self):
        app = _run("34_Cross_Sport_Tools.py")
        load_button = next(b for b in app.button if "cross-sport parlay" in b.label.lower())
        load_button.click().run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        picker = app.multiselect(key="cross_sport_leg_picker")
        self.assertTrue(any(option.startswith("CBB:") for option in picker.options))

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "34_Cross_Sport_Tools.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


if __name__ == "__main__":
    unittest.main()
