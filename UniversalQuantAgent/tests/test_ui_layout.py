"""Offline UI layout and interaction contracts for the visual overhaul."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import fantasy_shared
from app.components.player_card import player_card_markup
from app.style import GLOBAL_CSS, PALETTE, drag_drop_lineup_html, stacked_card_html
from fantasy import my_team_manager


class UILayoutContracts(unittest.TestCase):
    def test_application_shell_renders_the_sports_hub_cleanly(self):
        application = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=60)
        application.run()
        self.assertEqual([str(element.value) for element in application.exception], [])
        self.assertIn("Sports Hub", [element.value for element in application.title])

    def test_fantasy_hub_and_graph_lab_render_from_projection_data(self):
        players = [
            {
                "player_id": f"player-{index}",
                "name": f"Player {index}",
                "position": position,
                "team": "DEN",
                "projection": projection,
                "expected_fantasy_points": projection,
                "projection_confidence": 0.82 - index * 0.04,
                "games_played": 17,
                "adp": float(index + 1),
                "scoring_mode": "ppr",
                "projection_basis": "offline UI contract pool",
            }
            for index, (position, projection) in enumerate(
                (("QB", 330.0), ("RB", 275.0), ("WR", 260.0), ("TE", 190.0))
            )
        ]
        original_loader = fantasy_shared.load_pool
        original_team_loader = my_team_manager.load_user_team
        fantasy_shared.load_pool = lambda *_args, **_kwargs: (list(players), "offline UI contract pool")
        my_team_manager.load_user_team = list
        try:
            application = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=60)
            application.run()
            for filename in ("25_Fantasy_Hub.py", "29_Graph_Lab.py"):
                with self.subTest(page=filename):
                    application.switch_page(f"pages/{filename}").run()
                    self.assertEqual([str(element.value) for element in application.exception], [])
                    self.assertGreater(len(application.get("plotly_chart")), 0)
        finally:
            fantasy_shared.load_pool = original_loader
            my_team_manager.load_user_team = original_team_loader

    def test_stacked_card_is_vertical_safe_and_uses_shared_classes(self):
        markup = stacked_card_html(
            "Fantasy <Hub>",
            "Weekly & reliable",
            kicker="Sports",
            stats={"Projected Score": "123.4", "Confidence": "84%"},
            rarity_rank=1,
        )
        self.assertIn('class="quant-card stacked-card', markup)
        self.assertIn("quant-card-stats", markup)
        self.assertIn("rarity-mythic", markup)
        self.assertIn("Fantasy &lt;Hub&gt;", markup)
        self.assertIn("Weekly &amp; reliable", markup)

    def test_player_card_contains_rarity_rank_confidence_and_projection(self):
        markup = player_card_markup(
            {
                "name": "Sample Runner",
                "position": "RB",
                "team": "DEN",
                "overall_rank": 4,
                "projection": 245.7,
                "projection_confidence": 0.82,
            }
        )
        self.assertIn("rarity-legendary", markup)
        self.assertIn("#4", markup)
        self.assertIn("82%", markup)
        self.assertIn("245.7", markup)

    def test_drag_drop_board_has_pointer_keyboard_and_persistence_contracts(self):
        markup = drag_drop_lineup_html(
            [
                {"player_id": "p1", "name": "Starter", "position": "RB", "slot": "RB", "rank": 1},
                {"player_id": "p2", "name": "Reserve", "position": "WR", "slot": "BENCH", "rank": 22},
            ],
            key="contract-board",
        )
        self.assertEqual(markup.count('draggable="true"'), 2)
        self.assertIn('data-lane="starters"', markup)
        self.assertIn('data-lane="bench"', markup)
        self.assertIn("dragstart", markup)
        self.assertIn("dragover", markup)
        self.assertIn("drop", markup)
        self.assertIn("event.key !== 'Enter'", markup)
        self.assertIn("window.localStorage", markup)
        self.assertIn("uqa-lineup-contract-board", markup)

    def test_palette_and_global_layout_are_exact(self):
        self.assertEqual(PALETTE["midnight_navy"], "#0A1A2F")
        self.assertEqual(PALETTE["circuit_cyan"], "#00C8FF")
        self.assertEqual(PALETTE["signal_gold"], "#F5C542")
        self.assertEqual(PALETTE["slate_blue"], "#1F2E45")
        self.assertEqual(PALETTE["soft_white"], "#F2F4F7")
        self.assertIn("grid-template-columns:1fr", GLOBAL_CSS)
        self.assertIn("prefers-reduced-motion", GLOBAL_CSS)

    def test_pages_and_primary_navigation_expose_required_surfaces(self):
        app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
        required_pages = [
            "pages/Home.py",
            "pages/25_Fantasy_Hub.py",
            "pages/27_Fantasy_Season_Tools.py",
            "pages/28_Fantasy_My_Team.py",
            "pages/29_Graph_Lab.py",
        ]
        for page in required_pages:
            with self.subTest(page=page):
                self.assertIn(page, app_source)
        for label in ("NFL", "NBA", "Fantasy", "Betting", "Fantasy Hub", "Graph Lab"):
            with self.subTest(label=label):
                self.assertIn(f'"{label}"', app_source)

    def test_my_team_and_weekly_tools_use_branded_charts_and_rarity_cards(self):
        my_team = (PROJECT_ROOT / "app" / "pages" / "28_Fantasy_My_Team.py").read_text(encoding="utf-8")
        weekly_tools = (PROJECT_ROOT / "app" / "pages" / "27_Fantasy_Season_Tools.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("render_drag_drop_lineup", my_team)
        self.assertIn("rarity_rank=rank", my_team)
        self.assertIn("gold_glow_line_chart", my_team)
        self.assertIn("gold_glow_line_chart", weekly_tools)
        self.assertIn("rarity_rank=selected_rank", weekly_tools)
        for surface in ("Weekly Projections", "Start / Sit", "Waivers", "Trades"):
            self.assertIn(surface, weekly_tools)


if __name__ == "__main__":
    unittest.main()
