from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

import streamlit as st

from app.page_runtime import PLOTLY_CONFIG, apply_global_theme, nfl_profile_header, run_analysis, section_header
from modules.nfl import latest_completed_nfl_season
from modules.nfl_graph_lab import (
    render_defensive_pressure_map, render_pace_play_volume,
    render_qb_passing_map, render_rb_usage_funnel, render_route_tree,
)

apply_global_theme()
st.title("NFL Graph Lab")
st.write("Team-colored, context-aware football visuals with lightweight Plotly interactions.")
VIEWS = {
    "QB Passing Map": render_qb_passing_map,
    "WR/TE Route Tree": render_route_tree,
    "RB Usage Funnel": render_rb_usage_funnel,
    "Defensive Pressure Map": render_defensive_pressure_map,
    "Pace & Play Volume": render_pace_play_volume,
}
with st.form("nfl_graph_lab_form"):
    first, second, third = st.columns([2, 2, 1])
    player = first.text_input("NFL player", "Josh Allen")
    opponent = second.text_input("Opponent", "KC")
    season = third.number_input("Season", 1999, latest_completed_nfl_season(), latest_completed_nfl_season())
    view = st.selectbox("Visual", list(VIEWS))
    pocket = st.segmented_control("Pocket split", ["Clean pocket", "Pressure"], default="Clean pocket", disabled=view != "QB Passing Map")
    one, two = st.columns(2)
    mode = one.radio("Mode", ["Raw", "Adjusted"], index=1, horizontal=True)
    comparison = two.radio("Comparison", ["League", "Position"], horizontal=True)
    submitted = st.form_submit_button("Build NFL visual", type="primary")
if submitted:
    renderer = VIEWS[view]
    if view == "QB Passing Map":
        operation = lambda: renderer(player, opponent or None, int(season), pocket or "Clean pocket", mode, comparison)
    else:
        operation = lambda: renderer(player, opponent or None, int(season), mode, comparison)
    figure = run_analysis(view, operation)
    if figure is not None:
        st.session_state["nfl_graph_lab"] = {"figure": figure, "view": view}
payload = st.session_state.get("nfl_graph_lab")
if not payload:
    st.info("Choose a player and visual to open the NFL Graph Lab.")
else:
    figure = payload["figure"]
    meta = figure.layout.meta or {}
    nfl_profile_header(meta.get("player", player), meta.get("team", ""), f"{meta.get('position', '')} · {payload['view']}")
    section_header(payload["view"], "Hover for values; switch Raw/Adjusted and League/Position to audit context.")
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="nfl_graph_lab_figure")
    for warning in meta.get("warnings", []):
        st.caption(f"Data note: {warning}")
