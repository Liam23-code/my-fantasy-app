"""Contracts for the focused Fantasy navigation and selected-team flow."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FantasyNavigationContracts(unittest.TestCase):
    def test_fantasy_navigation_contains_the_six_focused_destinations_in_order(self):
        source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
        start = source.index("fantasy_pages = [")
        end = source.index("groups = {", start)
        fantasy_navigation = source[start:end]
        destinations = (
            ("pages/25_Fantasy_Hub.py", 'title="Fantasy Hub"'),
            ("pages/25_Fantasy_Draft_Room.py", 'title="Mock Draft"'),
            ("pages/26_Fantasy_Saved_Teams.py", 'title="Saved Teams"'),
            ("pages/28_Fantasy_My_Team.py", 'title="My Team"'),
            ("pages/27_Fantasy_Season_Tools.py", 'title="Weekly Tools"'),
            ("pages/29_Graph_Lab.py", 'title="Graph Lab"'),
        )
        offsets = []
        for page, title in destinations:
            with self.subTest(page=page):
                self.assertIn(page, fantasy_navigation)
                self.assertIn(title, fantasy_navigation)
                offsets.append(fantasy_navigation.index(page))
        self.assertEqual(offsets, sorted(offsets))

    def test_fantasy_hub_only_links_to_the_four_primary_workflows(self):
        source = (PROJECT_ROOT / "app" / "pages" / "25_Fantasy_Hub.py").read_text(encoding="utf-8")
        ast.parse(source)
        for label in ("Mock Draft", "Saved Teams", "Weekly Tools", "Graph Lab"):
            self.assertIn(f'"{label}"', source)
        self.assertNotIn("load_user_team", source)
        self.assertNotIn("Saved roster", source)

    def test_saved_teams_and_my_team_are_distinct_pages(self):
        saved_page = PROJECT_ROOT / "app" / "pages" / "26_Fantasy_Saved_Teams.py"
        manager_page = PROJECT_ROOT / "app" / "pages" / "28_Fantasy_My_Team.py"
        self.assertTrue(saved_page.exists())
        saved_source = saved_page.read_text(encoding="utf-8")
        manager_source = manager_page.read_text(encoding="utf-8")
        ast.parse(saved_source)
        ast.parse(manager_source)
        self.assertIn("list_saved_teams", saved_source)
        self.assertIn("create_new_team_save", saved_source)
        self.assertIn("delete_team_save", saved_source)
        self.assertIn("team_id", manager_source)
        self.assertIn("Back to Saved Teams", manager_source)

    def test_application_shell_accepts_the_new_page_registry(self):
        application = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=60)
        application.run()
        self.assertEqual([str(element.value) for element in application.exception], [])
        self.assertIn("Sports Hub", [element.value for element in application.title])

    def test_saved_teams_and_unselected_manager_render_cleanly(self):
        application = AppTest.from_file(
            str(PROJECT_ROOT / "app" / "app.py"),
            default_timeout=60,
        ).run()
        for filename in ("26_Fantasy_Saved_Teams.py", "28_Fantasy_My_Team.py"):
            with self.subTest(page=filename):
                application.switch_page(f"pages/{filename}").run()
                self.assertEqual(
                    [str(element.value) for element in application.exception],
                    [],
                )
                if filename == "26_Fantasy_Saved_Teams.py":
                    self.assertTrue(
                        any(button.label == "Create New Team Save" for button in application.button)
                    )


if __name__ == "__main__":
    unittest.main()
