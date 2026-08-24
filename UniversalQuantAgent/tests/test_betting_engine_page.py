"""Offline rendering contract for the Betting Engine page.

Same pattern as tests/test_fantasy_pages.py: runs the real Streamlit script
through AppTest so a broken import, a widget key collision, or a crash on
real data is caught here instead of in the browser.
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamlit.testing.v1 import AppTest

PAGE = Path(_PROJECT_ROOT) / "app" / "pages" / "30_Betting_Engine.py"


class BettingEnginePageContract(unittest.TestCase):
    def _run(self):
        app = AppTest.from_file(str(PAGE), default_timeout=120)
        app.run()
        raised = [str(element.value) for element in app.exception]
        self.assertEqual(raised, [], msg=f"page raised: {raised}")
        return app

    def test_page_renders_with_five_tabs_and_no_exceptions(self):
        app = self._run()
        self.assertEqual(len(app.tabs), 5)

    def test_sport_toggle_offers_all_four_sports(self):
        app = self._run()
        radio = app.radio(key="betting_engine_sport")
        self.assertEqual(list(radio.options), ["NFL", "NBA", "CFB", "CBB"])

    def test_cfb_branch_renders_with_no_exceptions(self):
        app = self._run()
        app.radio(key="betting_engine_sport").set_value("CFB").run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertEqual(len(app.tabs), 5)

    def test_cbb_branch_renders_with_real_props_and_no_exceptions(self):
        app = self._run()
        app.radio(key="betting_engine_sport").set_value("CBB").run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        props_table = app.dataframe[0].value
        self.assertFalse(props_table.empty)
        for column in ("Player", "Category", "Line", "Edge", "Risk", "Basis"):
            self.assertIn(column, props_table.columns)
        self.assertTrue((props_table["Basis"].str.contains("real", case=False)).all())

    def test_cfb_and_cbb_parlay_tabs_render_without_exceptions(self):
        # CFB has no real props without a CFBD_API_KEY, so the parlay tab
        # correctly shows an empty state -- this only asserts the render
        # path itself never raises for either college sport.
        for college_sport in ("CFB", "CBB"):
            app = self._run()
            app.radio(key="betting_engine_sport").set_value(college_sport).run()
            self.assertEqual([str(e.value) for e in app.exception], [], msg=f"{college_sport} parlay tab raised")

    def test_cfb_and_cbb_included_in_cross_sport_options(self):
        app = self._run()
        app.radio(key="betting_engine_sport").set_value("CBB").run()
        load_button = next(b for b in app.button if "cross-sport parlay" in b.label.lower())
        load_button.click().run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        picker = app.multiselect(key="cross_sport_leg_picker")
        self.assertTrue(any(option.startswith("CBB:") for option in picker.options))

    def test_player_props_table_has_real_rows(self):
        app = self._run()
        props_table = app.dataframe[0].value
        self.assertFalse(props_table.empty)
        for column in ("Player", "Market", "Line", "Edge", "Confidence", "Risk"):
            self.assertIn(column, props_table.columns)

    def test_money_lines_table_has_real_rows(self):
        app = self._run()
        moneyline_table = app.dataframe[1].value
        self.assertFalse(moneyline_table.empty)
        for column in ("Home", "Away", "Model spread (home)", "Model total"):
            self.assertIn(column, moneyline_table.columns)

    def test_parlay_tab_computes_metrics_for_default_legs(self):
        app = self._run()
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertIn("Adjusted hit probability", metrics)
        self.assertIn("Adjusted EV / $100", metrics)
        self.assertIn("Confidence", metrics)

    def test_player_comparison_tab_renders_a_valid_side_by_side_table(self):
        app = self._run()
        # Comparison table is the 3rd dataframe: props (0), money lines (1), comparison (2)
        # -- the parlay tab's leg summary dataframe renders after it in script order.
        comparison_tables = [df.value for df in app.dataframe if "Metric" in df.value.columns]
        self.assertTrue(comparison_tables, "expected a comparison table with a 'Metric' column")
        table = comparison_tables[0]
        self.assertIn("Line", table["Metric"].values)
        self.assertEqual(len(table.columns), 3)  # Metric + two selected props

    def test_quick_add_button_populates_the_leg_picker(self):
        app = self._run()
        app.number_input(key="NFL_quick_add_n").set_value(4).run()
        quick_add = next(b for b in app.button if b.label == "Quick-add")
        quick_add.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.multiselect(key="NFL_parlay_leg_picker").value), 4)

    def test_cross_sport_tab_load_button_does_not_raise(self):
        # NBA has no real games most of the year (this test runs whenever CI
        # runs it) -- when NBA has nothing to price, the tab correctly shows
        # an empty state rather than a working leg picker. This test asserts
        # the load-and-render path itself never raises, in either state;
        # test_unified_parlay_engine.py covers the mixed-parlay math directly
        # with real leg data, independent of live NBA schedule availability.
        app = self._run()
        load_button = next(b for b in app.button if "cross-sport parlay" in b.label.lower())
        load_button.click().run()
        self.assertEqual([str(e.value) for e in app.exception], [])
        try:
            picker = app.multiselect(key="cross_sport_leg_picker")
        except KeyError:
            return  # NBA had nothing to price today -- empty state rendered instead, which is correct
        self.assertGreaterEqual(len(picker.options), 2)

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = PAGE.read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


if __name__ == "__main__":
    unittest.main()
