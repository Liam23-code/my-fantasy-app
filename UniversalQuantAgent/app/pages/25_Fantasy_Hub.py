"""Minimal Fantasy command center and navigation hub."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

import streamlit as st
from app.fantasy_shared import league_setup
from app.page_runtime import apply_global_theme, page_header, section_header
from app.style import (
    CIRCUIT_CYAN,
    SIGNAL_GOLD,
    gold_glow_chart,
    safe_number,
    stacked_card_html,
)
from fantasy.weekly_projections import build_weekly_projection

apply_global_theme()
setup = league_setup(show_uploads=False)

page_header(
    "Fantasy Hub",
    "Draft, organize team saves, and open the weekly tools from one clean command center.",
    eyebrow="Fantasy · command center",
)

projections = [dict(player) for player in setup["projections"] if isinstance(player, dict)]
projections.sort(
    key=lambda player: safe_number(player.get("projection", player.get("expected_fantasy_points"))),
    reverse=True,
)
leader = projections[0] if projections else None
leader_curve = build_weekly_projection(leader, setup["scoring_mode"]) if leader else {}
projected_score = leader_curve.get(1, {}).get("points") if leader_curve else None
confidence = leader_curve.get(1, {}).get("confidence") if leader_curve else None

st.markdown(
    stacked_card_html(
        "Projection Pulse",
        (
            f"Current Week 1 signal leader: {leader.get('name', 'Unknown player')}."
            if leader
            else "Load a forward projection pool from Mock Draft or Weekly Tools to activate this signal."
        ),
        kicker="League-wide outlook",
        stats={
            "Projected Score": f"{projected_score:.2f}" if projected_score is not None else "—",
            "Confidence Score": f"{confidence:.0%}" if confidence is not None else "—",
        },
        rarity_rank=1 if leader else None,
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

if leader_curve:
    st.plotly_chart(
        gold_glow_chart(
            leader_curve,
            title="Projection leader · weekly curve",
            name="Projected points",
            height=280,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
        key="fantasy-hub-projection-pulse",
    )

section_header("Fantasy Tools", "Each workflow has one focused destination.")
actions = (
    (
        "Mock Draft",
        "Run a clean snake or linear mock draft without writing to a team save.",
        "pages/25_Fantasy_Draft_Room.py",
        ":material/emoji_events:",
    ),
    (
        "Saved Teams",
        "Create, select, and manage separate save files for every league.",
        "pages/26_Fantasy_Saved_Teams.py",
        ":material/folder_open:",
    ),
    (
        "Weekly Tools",
        "Review matchup-adjusted projections, waivers, lineups, trades, and scoring.",
        "pages/27_Fantasy_Season_Tools.py",
        ":material/calendar_month:",
    ),
    (
        "Graph Lab",
        "Inspect projection radar, momentum, confidence, and efficiency visuals.",
        "pages/29_Graph_Lab.py",
        ":material/monitoring:",
    ),
)

for index, (title, body, page, icon) in enumerate(actions):
    st.markdown(
        stacked_card_html(
            title,
            body,
            kicker="Fantasy destination",
            card_id=f"fantasy-action-{index}",
            extra_class="quick-action-card",
        ),
        unsafe_allow_html=True,
    )
    st.page_link(page, label=f"Open {title}", icon=icon, use_container_width=True)
