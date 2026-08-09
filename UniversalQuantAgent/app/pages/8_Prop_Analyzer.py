from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]
"""Analyze one player's live lines across all configured sportsbooks."""
import pandas as pd
import re
import streamlit as st
from app.page_runtime import apply_global_theme, records_table, run_analysis
from modules.projections import project_player_statline
from modules.props import compare_props

apply_global_theme()

st.title("Prop Analyzer")
st.caption("Live free sources can be delayed or geofenced. Empty results are never replaced with made-up lines.")
left, right = st.columns(2)
player = left.text_input("Player name", "Nikola Jokic")
opponent = right.text_input("Opponent team", "BOS")
categories = st.multiselect("Categories", ["points","rebounds","assists","PRA","3PM"], ["points","rebounds","assists"])
season = st.text_input("Season (optional)", placeholder="2025-26")
if st.button("Compare props", type="primary"):
    result = run_analysis("prop", lambda: compare_props(player, opponent, categories, season or None))
    if result is not None:
        st.session_state["prop_insights"] = result
result = st.session_state.get("prop_insights", [])
if result:
    st.dataframe(records_table(result), use_container_width=True, hide_index=True)
    first = result[0]
    reliability_text = next((driver for driver in first["key_drivers"] if driver.startswith("Quant reliability:")), "")
    reliability_match = re.search(r"([\d.]+)/100", reliability_text)
    columns = st.columns(5)
    columns[0].metric("Fusion projection", first["projection"])
    columns[1].metric("Adjusted projection", first["minutes_adjusted_projection"])
    columns[2].metric("Line", first["sportsbook_line"])
    columns[3].metric("Edge", first["edge"])
    columns[4].metric("Reliability", f"{reliability_match.group(1)}/100" if reliability_match else "—")
    st.write(f"Confidence range: {first['confidence_low']}–{first['confidence_high']} · Best book: {first['best_sportsbook']}")
    st.markdown("**Key drivers**")
    for driver in first["key_drivers"]: st.write("- " + driver)
    recent = run_analysis("recent form", lambda: project_player_statline(player, opponent, season or None))
    recent_value = recent.get("recent_10_game_averages", {}).get(first["category"].lower()) if recent else None
    chart_values = {"Projection":first["projection"], "Adjusted":first["minutes_adjusted_projection"], "Line":first["sportsbook_line"]}
    if recent_value is not None: chart_values["Recent 10"] = recent_value
    st.bar_chart(pd.DataFrame({"Value":chart_values}))
elif "prop_insights" in st.session_state:
    st.info("No verified sportsbook lines were available for that player and category.")
