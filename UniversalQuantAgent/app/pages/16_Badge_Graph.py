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
    PLOTLY_CONFIG, apply_global_theme, profile_header, records_table,
    render_insight_list, run_analysis, section_header,
)
from modules.badge_graph import (
    COMPARISON_MODES, DISPLAY_MODES, render_badge_graph,
    render_spider_badge_graph,
)
from modules.insights_engine import generate_badge_insights
from modules.nba_advanced import latest_season

apply_global_theme()
st.title("Badge Graph")
st.write("Explore one dilution-aware player identity model through a transparent spider or premium wheel.")
section_header(
    "Player identity profile",
    "The eight-axis spider shows the 60/40 league-position rating, sample confidence, and close-shot dilution; dunking contributes only to Finishing.",
)
with st.form("graph_lab_badge_form"):
    first, second, third = st.columns([2, 1, 1])
    player = first.text_input("NBA player", "Nikola Jokic", key="badge_player")
    season = second.text_input("Season", latest_season(), key="badge_season")
    opponent = third.text_input("Opponent (optional)", "", placeholder="BOS", key="badge_opponent")
    view_column, mode_column, comparison_column, filter_column = st.columns(4)
    view = view_column.segmented_control(
        "View", ["Spider", "Wheel"], default="Spider", key="badge_view"
    )
    display_mode = mode_column.segmented_control(
        "Rating mode", list(DISPLAY_MODES), default="Adjusted", key="badge_display_mode"
    )
    comparison_mode = comparison_column.segmented_control(
        "Comparison marker", list(COMPARISON_MODES), default="Entire league",
        key="badge_comparison_mode",
    )
    filter_samples = filter_column.toggle(
        "Filter by minimum sample size", value=True, key="badge_filter_samples"
    )
    gradient_fill = st.toggle(
        "Spider gradient depth", value=True, key="badge_gradient_fill",
        help="Adds progressively darker transparent layers toward the center.",
    )
    submitted = st.form_submit_button("Build badge profile", type="primary")
if submitted:
    if (view or "Spider") == "Spider":
        renderer = lambda: render_spider_badge_graph(
            player, display_mode or "Adjusted",
            comparison_mode or "Entire league", season=season,
            filter_minimum_samples=filter_samples,
            opponent_team=opponent or None,
            gradient_fill=gradient_fill,
        )
    else:
        renderer = lambda: render_badge_graph(
            player, season, display_mode or "Adjusted",
            comparison_mode or "Entire league", filter_samples,
            opponent or None,
        )
    result = run_analysis("badge profile", renderer)
    if result is not None:
        st.session_state["graph_lab_badge"] = result
figure = st.session_state.get("graph_lab_badge")
if figure is not None:
    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    profile_header(
        meta.get("player", player), meta.get("team", ""),
        f"{meta.get('position_group', 'NBA')} · {meta.get('view', 'Wheel')} · Season {meta.get('season', season)}",
    )
    render_insight_list(generate_badge_insights(meta), "Player identity")
    labels = meta.get("skill_labels", [])
    if labels:
        st.markdown("**Skill labels**  ")
        st.write(" · ".join(str(label) for label in labels))
    if "Low sample size: ratings may be unstable." in meta.get("warnings", []):
        st.warning("Low sample size: ratings may be unstable.")
    st.plotly_chart(
        figure, width="stretch", config=PLOTLY_CONFIG,
        key="graph_lab_badge_figure",
    )
    with st.expander("Percentile, sample, and dilution details"):
        columns = [
            "attribute", "badge_value", "raw_value", "adjusted_value",
            "rating", "league_percentile", "position_percentile", "attempts",
            "dynamic_minimum", "sample_confidence", "context_factor", "tier",
            "dilution_contributions",
        ]
        rows = [
            {key: item.get(key) for key in columns}
            for item in meta.get("attributes", [])
        ]
        st.dataframe(records_table(rows), width="stretch", hide_index=True)
    for warning in meta.get("warnings", []):
        if warning != "Low sample size: ratings may be unstable.":
            st.caption(f"Data note: {warning}")
else:
    st.info("Build a profile to reveal the default transparent spider view.")