"""Offline rendering contract for the MLB (35) and NHL (36) betting pages.

Same AppTest pattern as tests/test_betting_engine_page.py. Unlike NFL/NBA/
CBB (which ship real default data) or CFB (key-gated), MLB and NHL ship
no default data at all this cycle (see mlb_pipeline.md / nhl_pipeline.md)
-- every tab should render its disclosed empty state without raising,
with exactly the three tabs ui_betting_tabs.md specifies for every sport.
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


class MlbBettingPageTests(unittest.TestCase):
    def test_renders_three_tabs_with_no_exceptions(self):
        app = _run("35_MLB_Betting.py")
        self.assertEqual(len(app.tabs), 3)

    def test_no_data_configured_shows_the_disclosed_empty_state(self):
        app = _run("35_MLB_Betting.py")
        text = "\n".join(str(el.value) for el in list(app.caption) + list(app.markdown))
        self.assertIn("mlb_pipeline.md", text)

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "35_MLB_Betting.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")

    def test_no_scraping_or_network_imports_anywhere_in_the_mlb_engine(self):
        modules_dir = Path(_PROJECT_ROOT) / "modules"
        banned = ("import requests", "import httpx", "import aiohttp", "urllib3", "BeautifulSoup", "selenium", "playwright")
        for path in modules_dir.glob("mlb_*.py"):
            source = path.read_text(encoding="utf-8")
            for term in banned:
                self.assertNotIn(term, source, msg=f"{path.name} references banned network/scraping term: {term}")


class NhlBettingPageTests(unittest.TestCase):
    def test_renders_three_tabs_with_no_exceptions(self):
        app = _run("36_NHL_Betting.py")
        self.assertEqual(len(app.tabs), 3)

    def test_no_data_configured_shows_the_disclosed_empty_state(self):
        app = _run("36_NHL_Betting.py")
        text = "\n".join(str(el.value) for el in list(app.caption) + list(app.markdown))
        self.assertIn("nhl_pipeline.md", text)

    def test_no_scraping_or_network_imports_in_the_page_source(self):
        source = (PAGES_DIR / "36_NHL_Betting.py").read_text(encoding="utf-8")
        for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup", "sportsbook.draftkings", "sportsbook.fanduel"):
            self.assertNotIn(banned, source, msg=f"found network/scraping reference: {banned}")

    def test_no_scraping_or_network_imports_anywhere_in_the_nhl_engine(self):
        modules_dir = Path(_PROJECT_ROOT) / "modules"
        banned = ("import requests", "import httpx", "import aiohttp", "urllib3", "BeautifulSoup", "selenium", "playwright")
        for path in modules_dir.glob("nhl_*.py"):
            source = path.read_text(encoding="utf-8")
            for term in banned:
                self.assertNotIn(term, source, msg=f"{path.name} references banned network/scraping term: {term}")


if __name__ == "__main__":
    unittest.main()
