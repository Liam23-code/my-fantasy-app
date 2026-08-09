from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

import streamlit as st

from app.page_runtime import PLOTLY_CONFIG, apply_global_theme, run_analysis, section_header
from modules.nba_advanced import latest_season
from modules.trend_graphs import (
    POSITIONS, render_momentum_line, render_opponent_difficulty_curve,
    render_pace_impact_curve, render_performance_timeline,
    render_usage_efficiency_scatter,
)

apply_global_theme()
st.title("Trend Graphs")
st.write("Move from player momentum to league-wide usage, opponent difficulty, and pace context.")
section_header("Basketball storytelling", "Five focused views with lightweight controls and consistent hover detail.")
momentum_tab, timeline_tab, scatter_tab, difficulty_tab, pace_tab = st.tabs([
    "Momentum Line", "Performance Timeline", "Usage vs Efficiency",
    "Opponent Difficulty", "Pace Impact",
])
with momentum_tab:
    with st.form("momentum_form"):
        first, second = st.columns([2, 1])
        player = first.text_input("NBA player", "Nikola Jokic", key="momentum_player")
        season = second.text_input("Season", latest_season(), key="momentum_season")
        submitted = st.form_submit_button("Build momentum line", type="primary")
    if submitted:
        figure = run_analysis("momentum", lambda: render_momentum_line(player, season))
        if figure is not None:
            st.session_state["graph_momentum"] = figure
    figure = st.session_state.get("graph_momentum")
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_momentum_figure")
        meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
        for warning in meta.get("warnings", []):
            st.caption(f"Data note: {warning}")

with timeline_tab:
    with st.form("timeline_form"):
        first, second = st.columns([2, 1])
        player = first.text_input("NBA player", "Nikola Jokic", key="timeline_player")
        season = second.text_input("Season", latest_season(), key="timeline_season")
        submitted = st.form_submit_button("Build performance timeline", type="primary")
    if submitted:
        figure = run_analysis("performance timeline", lambda: render_performance_timeline(player, season))
        if figure is not None:
            st.session_state["graph_timeline"] = figure
    figure = st.session_state.get("graph_timeline")
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_timeline_figure")

with scatter_tab:
    with st.form("usage_scatter_form"):
        first, second = st.columns([2, 1])
        highlight = first.text_input("Highlight player (optional)", "Nikola Jokic")
        scatter_season = second.text_input("Season", latest_season(), key="scatter_season")
        submitted = st.form_submit_button("Build league scatter", type="primary")
    if submitted:
        figure = run_analysis("usage and efficiency", lambda: render_usage_efficiency_scatter(scatter_season, highlight or None))
        if figure is not None:
            st.session_state["graph_usage_scatter"] = figure
    figure = st.session_state.get("graph_usage_scatter")
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_usage_scatter_figure")

with difficulty_tab:
    with st.form("difficulty_curve_form"):
        first, second = st.columns([2, 1])
        position = first.segmented_control("Position group", POSITIONS, default="Guards")
        difficulty_season = second.text_input("Season", latest_season(), key="difficulty_season")
        submitted = st.form_submit_button("Build difficulty curve", type="primary")
    if submitted:
        figure = run_analysis("opponent difficulty", lambda: render_opponent_difficulty_curve(position or "Guards", difficulty_season))
        if figure is not None:
            st.session_state["graph_difficulty"] = figure
    figure = st.session_state.get("graph_difficulty")
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_difficulty_figure")

with pace_tab:
    with st.form("pace_curve_form"):
        pace_season = st.text_input("Season", latest_season(), key="pace_curve_season")
        submitted = st.form_submit_button("Build pace impact curve", type="primary")
    if submitted:
        figure = run_analysis("pace impact", lambda: render_pace_impact_curve(pace_season))
        if figure is not None:
            st.session_state["graph_pace"] = figure
    figure = st.session_state.get("graph_pace")
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_pace_figure")