"""Weekly manager for one explicitly selected team save."""

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

import pandas as pd
import streamlit as st
from app.fantasy_shared import league_setup
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    section_header,
)
from app.style import gold_glow_chart, render_drag_drop_lineup, stacked_card_html
from fantasy.my_team_manager import (
    find_weak_positions,
    load_saved_team,
    recommend_add_drop,
    recommend_lineup_swaps,
    recommend_trades,
    save_saved_team,
    team_confidence_curve,
    team_health_status,
    weekly_team_projection,
)
from fantasy.utils import safe_float


def _players(team: Any) -> list[dict[str, Any]]:
    if isinstance(team, list):
        return [dict(player) for player in team if isinstance(player, dict)]
    if isinstance(team, dict):
        for key in ("players", "roster", "team"):
            value = team.get(key)
            if isinstance(value, list):
                return [dict(player) for player in value if isinstance(player, dict)]
    return []


def _requested_team_id() -> str:
    query_value = st.query_params.get("team_id")
    if isinstance(query_value, list):
        query_value = query_value[0] if query_value else ""
    requested = str(query_value or "").strip()
    if requested:
        st.session_state["fantasy_selected_team_id"] = requested
        return requested
    return str(st.session_state.get("fantasy_selected_team_id") or "").strip()


apply_global_theme()
team_id = _requested_team_id()

if not team_id:
    page_header(
        "My Team",
        "Select a save before opening the weekly manager.",
        eyebrow="Fantasy · weekly manager",
    )
    st.page_link(
        "pages/26_Fantasy_Saved_Teams.py",
        label="Back to Saved Teams",
        icon=":material/arrow_back:",
    )
    empty_state(
        "No team selected",
        "Open Saved Teams and choose the league you want to manage.",
        icon="📁",
    )
    st.stop()

try:
    saved_team = load_saved_team(team_id)
except (FileNotFoundError, OSError, TypeError, ValueError) as error:
    page_header(
        "My Team",
        "The selected team save could not be opened.",
        eyebrow="Fantasy · weekly manager",
    )
    st.error(str(error))
    st.page_link(
        "pages/26_Fantasy_Saved_Teams.py",
        label="Back to Saved Teams",
        icon=":material/arrow_back:",
    )
    st.stop()

st.session_state["fantasy_selected_team_id"] = saved_team["team_id"]
if st.query_params.get("team_id") != saved_team["team_id"]:
    st.query_params["team_id"] = saved_team["team_id"]

setup = league_setup()
projections = setup["projections"]
session_roster = st.session_state.fantasy_roster
available_players = st.session_state.fantasy_available

page_header(
    saved_team.get("name") or "My Team",
    f"{saved_team.get('league') or 'Fantasy League'} · matchup-adjusted weekly management.",
    eyebrow=f"Fantasy · save {saved_team['team_id']}",
)
st.page_link(
    "pages/26_Fantasy_Saved_Teams.py",
    label="Back to Saved Teams",
    icon=":material/arrow_back:",
)

control_col, week_col = st.columns([2, 1])
with control_col:
    st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    if session_roster and st.button(
        "Replace saved roster with current roster",
        type="primary",
        key=f"my-team-replace-roster-{team_id}",
    ):
        replacement = dict(saved_team)
        replacement.pop("roster", None)
        replacement.pop("team", None)
        replacement["players"] = list(session_roster)
        try:
            saved_team = save_saved_team(team_id, replacement)
        except (OSError, TypeError, ValueError) as error:
            st.error(f"The selected save could not be updated: {error}")
        else:
            st.success(f"Saved {len(session_roster)} players to {saved_team['name']}.")
week = int(
    week_col.number_input(
        "Week",
        min_value=1,
        max_value=18,
        value=1,
        key=f"my-team-week-{team_id}",
    )
)

saved_players = _players(saved_team)
if not saved_players:
    empty_state(
        "This team save has no roster",
        "Load or draft a roster, then use the explicit replace button above. Mock Draft never auto-saves.",
        icon="🏈",
    )
    st.stop()

health = team_health_status(saved_team)
current = weekly_team_projection(saved_team, week)
curve = team_confidence_curve(saved_team)
curve_frame = pd.DataFrame(
    [
        {
            "Week": week_number,
            "Projected points": values["points"],
            "Confidence": values["confidence"],
        }
        for week_number, values in curve.items()
    ]
).set_index("Week")

health_col, points_col, confidence_col, active_col = st.columns(4)
health_col.metric("Team health", f"{health['health_score']:.0f}/100", health["status"].title())
points_col.metric(f"Week {week} points", f"{current['total_points']:.2f}")
confidence_col.metric("Confidence", f"{current['confidence']:.0%}")
active_col.metric("Available", f"{health['available_players']}/{health['total_players']}")

section_header("Saved Roster", "Rarity-tiered player cards from this save file only.")
for index, player in enumerate(saved_players):
    projection_value = player.get("projection", player.get("expected_fantasy_points"))
    rank = player.get("overall_rank", player.get("rank", player.get("adp", 41)))
    status = str(player.get("injury_status") or "Available").upper()
    st.markdown(
        stacked_card_html(
            player.get("name", "Unknown player"),
            f"{player.get('position', '—')} · {player.get('team') or player.get('nfl_team') or 'FA'}",
            kicker=f"Roster slot · {player.get('slot') or 'Unassigned'}",
            stats={
                "Season projection": f"{safe_float(projection_value):.1f}" if projection_value is not None else "—",
                "Health": status,
            },
            rarity_rank=rank,
            card_id=f"roster-player-{team_id}-{index}",
            extra_class="roster-player-card",
        ),
        unsafe_allow_html=True,
    )

if health["issues"]:
    with st.expander(f"Health report ({len(health['issues'])} issue(s))"):
        st.dataframe(pd.DataFrame(health["issues"]), width="stretch", hide_index=True)

section_header("Weekly Projection", "Optimized starting-lineup points across all 18 regular-season weeks.")
st.plotly_chart(
    gold_glow_chart(
        curve_frame,
        y="Projected points",
        title="Weekly team projection",
        name="Projected points",
        height=370,
    ),
    use_container_width=True,
    config={"displayModeBar": False},
    key=f"my-team-weekly-projection-{team_id}",
)

section_header(
    "Drag & Drop Lineup Management",
    "Move player cards between starters and bench. Changes persist separately for this save in this browser.",
)
render_drag_drop_lineup(saved_players, key=f"my-team-lineup-{team_id}", height=520)

section_header("Start/Sit Recommendations", f"The point-maximizing legal lineup for Week {week}.")
start_sit_rows = [
    {**player, "recommendation": "START"} for player in current["starters"]
] + [{**player, "recommendation": "BENCH"} for player in current["bench"]]
if start_sit_rows:
    st.dataframe(
        pd.DataFrame(start_sit_rows)[
            ["recommendation", "name", "position", "opponent", "points", "confidence", "reason"]
        ],
        width="stretch",
        hide_index=True,
        height=380,
    )

weak_positions = find_weak_positions(saved_team, week)
if weak_positions:
    with st.expander("Weak positions"):
        st.dataframe(pd.DataFrame(weak_positions), width="stretch", hide_index=True)

section_header("Bench Swap Suggestions", "Changes between saved slots and this week's optimized lineup.")
swaps = recommend_lineup_swaps(saved_team, week)
if swaps:
    st.dataframe(pd.DataFrame(swaps), width="stretch", hide_index=True)
else:
    st.success("No bench swap improves the current saved lineup.")

section_header("Waiver Recommendations", "Add/drop upgrades from the Weekly Tools waiver pool.")
waiver_key = f"my-team-waiver-recommendations-{team_id}"
if st.button("Run waiver analysis", type="primary", key=f"my-team-run-waivers-{team_id}"):
    st.session_state[waiver_key] = recommend_add_drop(saved_team, week, available_players)
waiver_recommendations = st.session_state.get(waiver_key, [])
if waiver_recommendations:
    st.dataframe(pd.DataFrame(waiver_recommendations), width="stretch", hide_index=True, height=380)
elif not available_players:
    st.caption("Load an available-player pool in League & draft setup.")

section_header("Trade Recommendations", "Direct upgrades from the Weekly Tools season projection pool.")
trade_key = f"my-team-trade-recommendations-{team_id}"
if st.button("Run trade analysis", type="primary", key=f"my-team-run-trades-{team_id}"):
    st.session_state[trade_key] = recommend_trades(saved_team, week, projections)
trade_recommendations = st.session_state.get(trade_key, [])
if trade_recommendations:
    st.dataframe(pd.DataFrame(trade_recommendations), width="stretch", hide_index=True, height=380)
elif not projections:
    st.caption("Load the season projection pool in League & draft setup.")

section_header("Team Confidence Curve", "Weighted confidence of the optimized lineup in every week.")
confidence_series = curve_frame["Confidence"].mul(100.0).rename("Confidence")
st.plotly_chart(
    gold_glow_chart(
        confidence_series,
        title="Team confidence by week",
        name="Confidence",
        height=330,
        y_suffix="%",
    ),
    use_container_width=True,
    config={"displayModeBar": False},
    key=f"my-team-confidence-curve-{team_id}",
)
