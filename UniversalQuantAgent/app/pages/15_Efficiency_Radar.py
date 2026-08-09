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

from app.page_runtime import PLOTLY_CONFIG, apply_global_theme, profile_header, run_analysis, section_header
from modules.nba_advanced import latest_season
from modules.radar_chart import RADAR_WINDOWS, render_efficiency_radar

apply_global_theme()
st.title("Efficiency Radar")
st.write("Compare a player's role, efficiency, ball security, defense, and pace fit on one shared scale.")
section_header("Efficiency radar", "Every spoke is a league-relative 0–100 rating; hover to inspect the underlying value.")
with st.form("graph_lab_radar_form"):
    first, second = st.columns([2, 1])
    player = first.text_input("NBA player", "Nikola Jokic", key="radar_player")
    season = second.text_input("Season", latest_season(), key="radar_season")
    window = st.radio("Sample window", list(RADAR_WINDOWS), horizontal=True)
    submitted = st.form_submit_button("Build efficiency radar", type="primary")
if submitted:
    result = run_analysis("efficiency radar", lambda: render_efficiency_radar(player, season, window))
    if result is not None:
        st.session_state["graph_lab_radar"] = result
figure = st.session_state.get("graph_lab_radar")
if figure is not None:
    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    profile_header(meta.get("player", player), meta.get("team", ""), f"{meta.get('season', season)} · {meta.get('window', window)}")
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="graph_lab_radar_figure")
    for warning in meta.get("warnings", []):
        st.caption(f"Data note: {warning}")
else:
    st.info("Build a radar to compare seven efficiency and role dimensions.")