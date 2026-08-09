from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

from app.page_runtime import (
    PLOTLY_CONFIG, apply_global_theme, nfl_profile_header, records_table,
    render_insight_list, run_analysis, section_header, style_figure,
)
from modules.nfl import latest_completed_nfl_season
from modules.nfl_analysis import analyze_nfl_player

apply_global_theme()
st.title("NFL Player Analysis")
st.write("Explore role, efficiency, context, volatility, and blended league/position percentiles.")
with st.form("nfl_player_analysis_form"):
    one, two, three = st.columns([2, 2, 1])
    player = one.text_input("NFL player", "Josh Allen")
    opponent = two.text_input("Opponent (optional)", "KC")
    season = three.number_input("Season", 1999, latest_completed_nfl_season(), latest_completed_nfl_season())
    mode_col, compare_col = st.columns(2)
    mode = mode_col.radio("Mode", ["Raw", "Adjusted"], index=1, horizontal=True)
    comparison = compare_col.radio("Comparison", ["League", "Position"], horizontal=True)
    submitted = st.form_submit_button("Analyze NFL player", type="primary")
if submitted:
    result = run_analysis("NFL player", lambda: analyze_nfl_player(player, opponent or None, int(season), mode, comparison))
    if result:
        st.session_state["nfl_player_analysis"] = result
result = st.session_state.get("nfl_player_analysis")
if not result:
    st.info("Choose a player to build the NFL identity profile.")
else:
    nfl_profile_header(result["player"], result["team"], f"{result['position']} · {result['season']}")
    render_insight_list([result["identity_summary"]], "Player identity")
    overview, traits, context = st.tabs(["Overview", "Strengths & weaknesses", "Matchup context"])
    with overview:
        a, b, c, d = st.columns(4)
        a.metric("Snap share", f"{result['usage_profile']['snap_share']:.1f}%")
        b.metric("Volume/game", result["usage_profile"]["volume_per_game"])
        c.metric("Role stability", f"{result['usage_profile']['role_stability']:.0f}/100")
        d.metric("Volatility", f"{result['volatility_score']:.0f}/100")
        chart = pd.DataFrame(result["attributes"])
        figure = px.bar(chart, x="blended_percentile", y="metric", orientation="h", color="blended_percentile", color_continuous_scale=["#d73a32", "#f0c419", "#149253"], range_color=[0, 100], title="Blended 60/40 skill percentiles")
        st.plotly_chart(style_figure(figure, 480), width="stretch", config=PLOTLY_CONFIG)
    with traits:
        left, right = st.columns(2)
        left.subheader("Strengths")
        left.dataframe(records_table(result["strengths"]), hide_index=True, width="stretch")
        right.subheader("Weaknesses")
        right.dataframe(records_table(result["weaknesses"]), hide_index=True, width="stretch")
    with context:
        section_header("Matchup and environment", "Opponent, pace, weather, and offensive-line context remain auditable.")
        st.dataframe(records_table([result["matchup"], result["pace"], result["weather"], result["offensive_line"]]), hide_index=True, width="stretch")
    for warning in result.get("warnings", []):
        st.caption(f"Data note: {warning}")
