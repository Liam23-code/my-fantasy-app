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
from app.page_runtime import (
    apply_global_theme, available_slate_edges, render_edge_heatmap_view,
    section_header,
)

apply_global_theme()
st.title("Edge Heatmap")
st.write("Scan saved prop projections against sportsbook lines by edge, reliability, and confidence.")
section_header("Slate edge map", "Green favors positive model edge, yellow is marginal, and red is negative.")
props = available_slate_edges()
if props:
    render_edge_heatmap_view(props, "betting_edge_lab")
else:
    st.info("Run Prop Analyzer, Prop Recommendations, or Daily Slate to populate the edge map.")