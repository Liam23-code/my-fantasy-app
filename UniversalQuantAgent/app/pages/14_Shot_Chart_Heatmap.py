from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]
from typing import Any
import streamlit as st

from app.page_runtime import PLOTLY_CONFIG, apply_global_theme, profile_header, run_analysis, section_header
from modules.nba_advanced import latest_season
from modules.shot_chart import SHOT_MODES, render_shot_chart

apply_global_theme()
st.title("Shot-Chart Heatmap")
st.write("Explore where a player shoots, converts, and concentrates volume on an NBA half court.")
section_header("Shot profile", "Team-colored location density with attempt, make, efficiency, and volume views.")
with st.form("graph_lab_shot_form"):
    first, second = st.columns([2, 1])
    player = first.text_input("NBA player", "Nikola Jokic")
    season = second.text_input("Season", latest_season())
    mode = st.radio("Heatmap measure", SHOT_MODES, index=1, horizontal=True)
    submitted = st.form_submit_button("Build shot chart", type="primary")
if submitted:
    result = run_analysis("shot chart", lambda: render_shot_chart(player, season, mode))
    if result is not None:
        st.session_state["graph_lab_shot_chart"] = result
figure = st.session_state.get("graph_lab_shot_chart")
if figure is not None:
    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    profile_header(meta.get("player", player), meta.get("team", ""), f"{meta.get('season', season)} · {meta.get('attempts', 0)} shots")
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_lab_shot_figure")
    for warning in meta.get("warnings", []):
        st.caption(f"Data note: {warning}")
else:
    st.info("Choose a player and build the chart to open the half-court view.")