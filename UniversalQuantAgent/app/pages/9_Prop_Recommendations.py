from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]
"""Rank model-versus-market edges for today's slate."""
import pandas as pd
import streamlit as st
from app.page_runtime import apply_global_theme, records_table, run_analysis
from modules.recommendations import recommend_props

apply_global_theme()

st.title("Prop Recommendations")
categories = st.multiselect("Categories", ["points","rebounds","assists","PRA","3PM"], ["points","rebounds","assists"])
left, middle, right = st.columns(3)
min_edge = left.number_input("Minimum absolute edge", 0.0, 20.0, 1.5, .5)
min_confidence = middle.slider("Minimum confidence", 0, 100, 55)
min_reliability = right.slider("Minimum reliability", 0, 100, 55)
if st.button("Build recommendations", type="primary"):
    result = run_analysis("daily recommendation", lambda: recommend_props(categories, min_edge, min_confidence, min_reliability=min_reliability))
    if result is not None:
        st.session_state["prop_recommendations"] = result
result = st.session_state.get("prop_recommendations", [])
if result:
    st.dataframe(records_table(result), use_container_width=True, hide_index=True)
    chart = pd.DataFrame(result).set_index("player")[["edge","reliability_score","recommendation_score"]]
    st.bar_chart(chart)
    for row in result[:10]:
        with st.expander(f"{row['player']} {row['category']} · {row['lean'].upper()} · {row['sportsbook']}"):
            st.write(row["explanation"])
            st.bar_chart(pd.DataFrame({"Value":{"Projection":row["minutes_adjusted_projection"], "Line":row["sportsbook_line"]}}))
elif "prop_recommendations" in st.session_state:
    st.info("No verified lines met the selected edge and confidence filters.")
