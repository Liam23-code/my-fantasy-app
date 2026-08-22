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

    def test_page_renders_with_three_tabs_and_no_exceptions(self):
        app = self._run()
        self.assertEqual(len(app.tabs), 3)

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

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = PAGE.read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")


if __name__ == "__main__":
    unittest.main()
