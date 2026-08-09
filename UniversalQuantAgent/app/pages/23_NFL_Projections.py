from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st

from app.page_runtime import apply_global_theme, confidence_ring, nfl_profile_header, reliability_bar, render_insight_list, run_analysis
from modules.nfl import latest_completed_nfl_season
from modules.nfl_projections import project_nfl_player

apply_global_theme()
st.title("NFL Player Projections")
st.write("Project volume, yardage, touchdowns, and fantasy points with matchup and weather context.")
with st.form("nfl_projection_form"):
    one, two, three = st.columns([2, 2, 1])
    player = one.text_input("NFL player", "Saquon Barkley")
    opponent = two.text_input("Opponent", "DAL")
    season = three.number_input("Season", 1999, latest_completed_nfl_season(), latest_completed_nfl_season())
    left, right = st.columns(2)
    mode = left.radio("Mode", ["Raw", "Adjusted"], index=1, horizontal=True)
    comparison = right.radio("Comparison", ["League", "Position"], horizontal=True)
    submitted = st.form_submit_button("Project NFL player", type="primary")
if submitted:
    result = run_analysis("NFL projection", lambda: project_nfl_player(player, opponent, int(season), mode, comparison))
    if result:
        st.session_state["nfl_projection"] = result
result = st.session_state.get("nfl_projection")
if not result:
    st.info("Choose a player and opponent to build a projection.")
else:
    nfl_profile_header(result["player"], result["team"], f"{result['position']} vs {result['opponent']}")
    projection, drivers, reliability, context = st.tabs(["Projection", "Model Drivers", "Reliability", "Player Context"])
    with projection:
        p = result["projection"]
        position_metrics = {
            "QB": [("Fantasy points", "expected_fantasy_points"), ("Passing yards", "passing_yards"), ("Rushing yards", "rushing_yards"), ("Interceptions", "interceptions")],
            "RB": [("Fantasy points", "expected_fantasy_points"), ("Carries", "carries"), ("Rushing yards", "rushing_yards"), ("Receiving yards", "receiving_yards")],
            "WR": [("Fantasy points", "expected_fantasy_points"), ("Targets", "targets"), ("Receptions", "receptions"), ("Receiving yards", "receiving_yards")],
            "TE": [("Fantasy points", "expected_fantasy_points"), ("Targets", "targets"), ("Receptions", "receptions"), ("Receiving yards", "receiving_yards")],
            "DEF": [("Fantasy points", "expected_fantasy_points"), ("Pressure rate", "pressure_rate"), ("Sack probability", "sack_probability"), ("EPA allowed", "epa_allowed")],
        }
        metric_specs = position_metrics[result["position"]]
        cols = st.columns(4)
        for column, (label, key) in zip(cols, metric_specs):
            column.metric(label, p.get(key, 0))
        touchdown_metrics = {
            "QB": [("Passing TDs", "passing_tds"), ("Rushing TDs", "rushing_tds")],
            "RB": [("Rushing TDs", "rushing_tds"), ("Receiving TDs", "receiving_tds")],
            "WR": [("Receiving TDs", "receiving_tds")],
            "TE": [("Receiving TDs", "receiving_tds")],
        }.get(result["position"], [])
        if touchdown_metrics:
            td_columns = st.columns(len(touchdown_metrics))
            for column, (label, key) in zip(td_columns, touchdown_metrics):
                column.metric(label, f"{p.get(key, 0):.2f}")
        st.success(result["confidence"]["label"])
        chart_keys = [key for _, key in metric_specs[1:]]
        st.bar_chart(pd.DataFrame({"Projection": {key: p.get(key, 0) for key in chart_keys}}))
    with drivers:
        render_insight_list(result["drivers"], "Projection drivers")
    with reliability:
        left, right = st.columns([1, 2])
        with left:
            confidence_ring(result["confidence"]["score"], "Projection confidence", "nfl_projection_confidence")
        with right:
            reliability_bar(result["confidence"]["score"], "Model reliability")
            st.metric("Confidence interval", f"{result['confidence']['low']:.1f}–{result['confidence']['high']:.1f} FP")
            st.metric("Volatility", f"{result['volatility_score']:.0f}/100")
    with context:
        st.json({"matchup": result["analysis"]["matchup"], "weather": result["analysis"]["weather"], "pace": result["analysis"]["pace"]}, expanded=True)
    for warning in result.get("warnings", []):
        st.caption(f"Data note: {warning}")
