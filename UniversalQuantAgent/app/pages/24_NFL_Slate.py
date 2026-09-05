from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

# ``fantasy`` is editable-installed for the app; adding the engine root keeps a
# local/offline launch self-contained now that this page reads the live
# player-status overlay for its availability watchlist.
_ENGINE_ROOT = str(Path(__file__).resolve().parents[3] / "fantasy_engine")
if Path(_ENGINE_ROOT).is_dir() and _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

from app.page_runtime import PLOTLY_CONFIG, apply_global_theme, records_table, render_insight_list, run_analysis, section_header, style_figure
from app.status_ui import render_flagged_watchlist, render_status_strip
from modules.nfl import latest_completed_nfl_season
from modules.nfl_slate import analyze_nfl_slate

apply_global_theme()
st.title("NFL Slate Analysis")
st.write("Compare game environments across pace, trench mismatches, explosive plays, weather, and fantasy scoring.")

# Display-only. Slate rows are game environments, not players, so the overlay
# cannot change a number here -- it contributes the watchlist below and nothing
# else. No game is filtered and no environment score is adjusted.
render_status_strip(
    "nfl_slate_refresh_player_status",
    hint="slate rows are game environments, which availability never changes. ",
)

with st.form("nfl_slate_form"):
    season = st.number_input("Season", 1999, latest_completed_nfl_season(), latest_completed_nfl_season())
    submitted = st.form_submit_button("Analyze NFL slate", type="primary")
if submitted:
    result = run_analysis("NFL slate", lambda: analyze_nfl_slate(season=int(season)))
    if result:
        st.session_state["nfl_slate"] = result
result = st.session_state.get("nfl_slate")
if not result:
    st.info("Build the slate to compare the current game environments.")
else:
    games = result["games"]
    section_header("Slate environments", f"{len(games)} games · {result['season']} season context")
    table = pd.DataFrame(games)
    figure = px.scatter(table, x="pace_projection", y="fantasy_scoring_environment", size="explosive_play_probability", color="matchup_difficulty", hover_name="away_team", hover_data=["home_team", "identity_summary"], color_continuous_scale=["#149253", "#f0c419", "#d73a32"], title="Pace vs fantasy scoring environment")
    st.plotly_chart(style_figure(figure, 470), width="stretch", config=PLOTLY_CONFIG)
    st.dataframe(records_table(games), hide_index=True, width="stretch")
    render_insight_list([f"{game['away_team']} at {game['home_team']}: {game['identity_summary']}" for game in games], "Game identities")
    for warning in result.get("warnings", []):
        st.caption(f"Data note: {warning}")

# The slate has no player rows of its own, so the badges come from the overlay
# directly: everyone currently flagged, league-wide, as context for the game
# environments above. Read-only -- nothing here filters or rescales a slate row.
with st.expander("Availability watchlist"):
    st.caption(
        "Every player the live status overlay currently flags. Slate environments above are computed "
        "from team-level history and are not adjusted by this list."
    )
    render_flagged_watchlist()
