from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]
"""Evaluate a group of previously analyzed or recommended prop legs."""
import streamlit as st
from app.page_runtime import apply_global_theme, records_table
from modules.parlay import build_parlay

apply_global_theme()

st.title("Parlay Builder")
available = st.session_state.get("prop_recommendations", []) or st.session_state.get("prop_insights", [])
if not available:
    st.info("Run Prop Analyzer or Prop Recommendations first, then return here.")
else:
    labels = [f"{i+1}. {p['player']} {p['category']} {p['lean']} {p['sportsbook']} ({p['sportsbook_line']})" for i,p in enumerate(available)]
    chosen = st.multiselect("Select legs", labels)
    if st.button("Evaluate parlay", type="primary") and chosen:
        indexes = [labels.index(label) for label in chosen]
        result = build_parlay([available[index] for index in indexes])
        st.session_state["parlay_insights"] = result
    result = st.session_state.get("parlay_insights")
    if result:
        cols = st.columns(4)
        cols[0].metric("Combined edge", result["combined_edge"])
        cols[1].metric("Risk score", result["risk_score"])
        cols[2].metric("Blowout risk", result["blowout_risk"])
        cols[3].metric("Expected value", f"{result['expected_value']:.1f}%")
        st.dataframe(records_table(result["props"]), use_container_width=True, hide_index=True)
        st.write(result["summary"])
        for warning in result["correlation_warnings"] + result["injury_warnings"]: st.warning(warning)
