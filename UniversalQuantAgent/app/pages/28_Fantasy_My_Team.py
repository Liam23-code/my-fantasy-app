import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

# Fantasy My Team: the persistent, week-to-week home for a drafted roster.
#
# The path preamble must run before app/fantasy imports, matching the other
# Fantasy Streamlit pages. This page delegates every recommendation to
# fantasy.my_team_manager and only owns presentation state.

import pandas as pd
import streamlit as st
from app.fantasy_shared import league_setup
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    section_header,
)
from app.style import gold_glow_line_chart, render_drag_drop_lineup, stacked_card_html
from fantasy.my_team_manager import (
    find_weak_positions,
    load_user_team,
    recommend_add_drop,
    recommend_lineup_swaps,
    recommend_trades,
    save_user_team,
    team_confidence_curve,
    team_health_status,
    weekly_team_projection,
)
from fantasy.utils import safe_float

apply_global_theme()
setup = league_setup()

page_header(
    "My Team",
    "Your saved roster, managed week by week with matchup-adjusted projections, lineup moves, waivers, and trades.",
    eyebrow="Fantasy · weekly manager",
)

projections = setup["projections"]
session_roster = st.session_state.fantasy_roster
available_players = st.session_state.fantasy_available

try:
    saved_team = load_user_team()
except ValueError as error:
    saved_team = []
    st.error(str(error))

save_col, week_col = st.columns([2, 1])
with save_col:
    st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    if session_roster and st.button("Save current roster", type="primary", key="my_team_save_roster"):
        saved_team = save_user_team(session_roster)
        st.success(f"Saved {len(session_roster)} players.")
week = int(week_col.number_input("Week", min_value=1, max_value=18, value=1, key="my_team_week"))

if isinstance(saved_team, dict):
    saved_players = next(
        (
            value
            for key in ("players", "roster", "team")
            if isinstance((value := saved_team.get(key)), list)
        ),
        [],
    )
else:
    saved_players = saved_team if isinstance(saved_team, list) else []

if not saved_players:
    empty_state(
        "No saved team",
        "Complete a draft or upload a roster in League & draft setup, then save it to start weekly management.",
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

section_header("Team Roster Card", "The persisted roster used by every recommendation below.")
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
            card_id=f"roster-player-{index}",
            extra_class="roster-player-card",
        ),
        unsafe_allow_html=True,
    )

if health["issues"]:
    with st.expander(f"Health report ({len(health['issues'])} issue(s))"):
        st.dataframe(pd.DataFrame(health["issues"]), width="stretch", hide_index=True)

section_header("Weekly Projection Chart", "Optimized starting-lineup points across all 18 regular-season weeks.")
st.plotly_chart(
    gold_glow_line_chart(
        curve_frame["Projected points"].tolist(),
        curve_frame.index.tolist(),
        title="Weekly team projection",
        name="Projected points",
        height=370,
    ),
    use_container_width=True,
    config={"displayModeBar": False},
    key="my-team-weekly-projection",
)

section_header(
    "Drag & Drop Lineup Management",
    "Move player cards between starters and bench. Changes persist in this browser.",
)
render_drag_drop_lineup(saved_players, key="my-team-lineup", height=520)

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

waiver_col, trade_col = st.columns(2)
with waiver_col:
    section_header("Waiver Recommendations", "Add/drop upgrades from the Season Tools waiver pool.")
    if st.button("Run waiver analysis", type="primary", key="my_team_run_waivers"):
        st.session_state["my_team_waiver_recommendations"] = recommend_add_drop(
            saved_team, week, available_players
        )
    waiver_recommendations = st.session_state.get("my_team_waiver_recommendations", [])
    if waiver_recommendations:
        st.dataframe(pd.DataFrame(waiver_recommendations), width="stretch", hide_index=True, height=380)
    elif not available_players:
        st.caption("Load an available-player pool in League & draft setup.")

with trade_col:
    section_header("Trade Recommendations", "Direct upgrades from the Season Tools season projection pool.")
    if st.button("Run trade analysis", type="primary", key="my_team_run_trades"):
        st.session_state["my_team_trade_recommendations"] = recommend_trades(saved_team, week, projections)
    trade_recommendations = st.session_state.get("my_team_trade_recommendations", [])
    if trade_recommendations:
        st.dataframe(pd.DataFrame(trade_recommendations), width="stretch", hide_index=True, height=380)
    elif not projections:
        st.caption("Load the season projection pool in League & draft setup.")

section_header("Team Confidence Curve", "Weighted confidence of the optimized lineup in every week.")
st.plotly_chart(
    gold_glow_line_chart(
        (curve_frame["Confidence"] * 100.0).tolist(),
        curve_frame.index.tolist(),
        title="Team confidence by week",
        name="Confidence",
        height=330,
        y_suffix="%",
    ),
    use_container_width=True,
    config={"displayModeBar": False},
    key="my-team-confidence-curve",
)
