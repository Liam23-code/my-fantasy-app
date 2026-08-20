"""Fantasy command center with team outlook, quick actions, and trends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

import streamlit as st
from app.components.player_card import render_player_card
from app.fantasy_shared import league_setup
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    section_header,
)
from app.style import CIRCUIT_CYAN, SIGNAL_GOLD, stacked_card_html
from fantasy.my_team_manager import load_user_team, weekly_team_projection
from fantasy.utils import safe_float


def _team_players(team: Any) -> list[dict[str, Any]]:
    if isinstance(team, list):
        return [dict(player) for player in team if isinstance(player, dict)]
    if isinstance(team, dict):
        for key in ("players", "roster", "team"):
            value = team.get(key)
            if isinstance(value, list):
                return [dict(player) for player in value if isinstance(player, dict)]
    return []


apply_global_theme()
setup = league_setup()

page_header(
    "Fantasy Hub",
    "Your compact command center for drafting, weekly decisions, and season-long roster management.",
    eyebrow="Fantasy · command center",
)

try:
    saved_team = load_user_team()
except ValueError:
    saved_team = []
saved_players = _team_players(saved_team)

if saved_players:
    outlook = weekly_team_projection(saved_team, 1)
    projected_score = safe_float(outlook.get("total_points"))
    confidence = safe_float(outlook.get("confidence"))
else:
    projected_score = 0.0
    confidence = 0.0

st.markdown(
    stacked_card_html(
        "Team Outlook",
        "Week 1 optimized lineup forecast from the saved roster.",
        kicker="Live projection",
        stats={
            "Projected Score": f"{projected_score:.2f}",
            "Confidence Score": f"{confidence:.0%}",
            "Rostered": len(saved_players),
        },
        extra_class="fantasy-outlook-card",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f"<style>.fantasy-outlook-card .quant-card-stats div:first-child strong{{color:{SIGNAL_GOLD}}}"
    f".fantasy-outlook-card .quant-card-stats div:nth-child(2) strong{{color:{CIRCUIT_CYAN}}}"
    ".fantasy-outlook-card .quant-card-stats strong{font-size:1.75rem}</style>",
    unsafe_allow_html=True,
)

section_header("Quick Actions", "Move directly into the three decisions that matter most.")
actions = (
    (
        "Draft Room",
        "Run a live snake or linear draft and save the completed roster.",
        "pages/25_Fantasy_Draft_Room.py",
        ":material/emoji_events:",
    ),
    (
        "My Team",
        "Manage starters, bench moves, health, waivers, and trades.",
        "pages/28_Fantasy_My_Team.py",
        ":material/groups:",
    ),
    (
        "Weekly Tools",
        "Explore every 18-week curve, matchup, and confidence signal.",
        "pages/27_Fantasy_Season_Tools.py",
        ":material/calendar_month:",
    ),
)
for title, body, page, icon in actions:
    st.markdown(
        stacked_card_html(title, body, kicker="Quick action", extra_class="quick-action-card"),
        unsafe_allow_html=True,
    )
    st.page_link(page, label=f"Open {title}", icon=icon, use_container_width=True)

section_header("Trending Players", "Top forward projections, classified by overall rarity tier.")
projections = [dict(player) for player in setup["projections"] if isinstance(player, dict)]
projections.sort(
    key=lambda player: safe_float(player.get("projection", player.get("expected_fantasy_points"))),
    reverse=True,
)
trending = projections[:6]
for rank, player in enumerate(trending, start=1):
    player.setdefault("overall_rank", rank)

if not trending:
    empty_state(
        "No projection pool",
        "Load current projection data from League & draft setup to see trending players.",
        icon="📈",
    )
else:
    for index in range(0, len(trending), 2):
        columns = st.columns(2)
        for column, player in zip(columns, trending[index : index + 2]):
            with column:
                render_player_card(
                    player,
                    scoring_mode=setup["scoring_mode"],
                    key=f"fantasy-hub-{player.get('player_id') or player.get('name')}",
                )
