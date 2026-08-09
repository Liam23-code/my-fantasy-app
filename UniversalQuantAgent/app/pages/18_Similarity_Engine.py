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
from app.page_runtime import apply_global_theme, render_similarity_view, run_analysis, section_header
from modules.data_quality import as_dict
from modules.nba_advanced import latest_season
from modules.similarity_engine import compute_player_similarity

apply_global_theme()
st.title("Similarity Engine")
st.write("Find players with comparable roles, production shapes, and matchup context.")
section_header("Player similarity", "Normalized dimensions and cosine similarity with an auditable breakdown.")
with st.form("advanced_similarity_form"):
    first, second = st.columns([2, 1])
    player = first.text_input("NBA player", "Nikola Jokic")
    season = second.text_input("Season", latest_season())
    submitted = st.form_submit_button("Find similar players", type="primary")
if submitted:
    result = run_analysis("player similarity", lambda: compute_player_similarity(player, season))
    if result:
        st.session_state["advanced_similarity"] = result
result = as_dict(st.session_state.get("advanced_similarity"))
if result:
    render_similarity_view(result, "advanced_similarity")
else:
    st.info("Run the engine to rank the ten closest player profiles.")