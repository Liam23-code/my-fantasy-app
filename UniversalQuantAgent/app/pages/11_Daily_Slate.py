from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]
"""Browse today's schedule, projections, correlations, and slate-wide edges."""
import pandas as pd
import streamlit as st

from app.page_runtime import (
    apply_global_theme,
    available_slate_edges,
    confidence_card,
    confidence_ring,
    profile_header,
    records_table,
    reliability_bar,
    render_correlation_heatmap,
    render_edge_heatmap_view,
    render_insight_list,
    run_analysis,
    section_header,
)
from modules.correlation_engine import compute_slate_correlations
from modules.daily_slate import get_daily_slate
from modules.data_quality import safe_number
from modules.insights_engine import generate_slate_insights


apply_global_theme()

st.title("Daily Slate")
st.write("Review games, available props, generated projections, and slate relationships.")
if st.button("Load daily slate", type="primary"):
    result = run_analysis("daily slate", get_daily_slate)
    if result is not None:
        st.session_state["daily_slate"] = result

slate = st.session_state.get("daily_slate")
if slate:
    overview_tab, correlation_tab, edge_tab = st.tabs(
        ["Slate Overview", "Correlation Heatmaps", "Slate Edge Heatmap"]
    )
    render_insight_list(generate_slate_insights(slate), "Daily slate insights")
    with overview_tab:
        section_header(
            "Slate overview",
            "Games, player environments, and public prop coverage in one workspace.",
        )
        projections = [
            row for row in slate.get("projections", []) if isinstance(row, dict)
        ]
        def projection_completeness(row):
            statline = row.get("projection", {})
            expected = [
                statline.get(stat) if isinstance(statline, dict) else None
                for stat in ("points", "rebounds", "assists", "pra")
            ] + [
                row.get("minutes"), row.get("pace"), row.get("matchup_difficulty")
            ]
            return 100.0 * sum(value is not None for value in expected) / len(expected)

        confidence_scores = [
            safe_number(row.get("confidence"), projection_completeness(row))
            for row in projections
        ]
        reliability_scores = [
            safe_number(row.get("reliability"), projection_completeness(row))
            for row in projections
        ]
        if confidence_scores or reliability_scores:
            average_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores else 0.0
            )
            average_reliability = (
                sum(reliability_scores) / len(reliability_scores)
                if reliability_scores else average_confidence
            )
            ring_column, confidence_column, reliability_column = st.columns(
                [1, 1, 1], vertical_alignment="center"
            )
            with ring_column:
                confidence_ring(
                    average_confidence, "Slate confidence", "daily_slate_confidence_ring"
                )
            with confidence_column:
                confidence_card(average_confidence, "Data confidence")
                reliability_bar(average_confidence, "Confidence")
            with reliability_column:
                confidence_card(average_reliability, "Slate reliability")
                reliability_bar(average_reliability, "Reliability")
        featured = projections[:3]
        if featured:
            st.markdown("### Featured player environments")
            featured_columns = st.columns(len(featured))
            for column, row in zip(featured_columns, featured):
                with column:
                    profile_header(
                        row.get("player", "Player"),
                        row.get("team", ""),
                        f"{safe_number(row.get('pace'), 100):.1f} pace",
                    )
        st.subheader(f"Games · {slate['date']}")
        st.dataframe(
            records_table(slate["games"]),
            use_container_width=True,
            hide_index=True,
        )
        props = pd.DataFrame(slate["player_props"])
        if not props.empty:
            first, second, third = st.columns(3)
            teams = ["All"] + sorted(props["team"].dropna().unique().tolist())
            team = first.selectbox("Team", teams)
            player = second.text_input("Filter player")
            categories = third.multiselect(
                "Categories", sorted(props["category"].dropna().unique())
            )
            filtered = props.copy()
            if team != "All":
                filtered = filtered[filtered["team"] == team]
            if player:
                filtered = filtered[
                    filtered["player"].str.contains(player, case=False, na=False)
                ]
            if categories:
                filtered = filtered[filtered["category"].isin(categories)]
            st.subheader("Sportsbook props")
            st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.subheader("Projections")
        st.dataframe(
            records_table(slate["projections"]),
            use_container_width=True,
            hide_index=True,
        )

    with correlation_tab:
        payload = compute_slate_correlations(slate.get("projections", []))
        options = payload.get("available_stats", [])
        selected = st.multiselect(
            "Slate statistics",
            options,
            default=options,
            key="daily_slate_correlation_stats",
        )
        render_correlation_heatmap(
            payload, selected, "daily_slate_correlations"
        )

    with edge_tab:
        edges = available_slate_edges()
        if edges:
            render_edge_heatmap_view(edges, "daily_slate_edges")
        else:
            st.info(
                "No projection-versus-line matches are available yet. "
                "Load props and projections or run Prop Analyzer first."
            )
else:
    st.info("Load the slate to unlock games, correlations, and edge heatmaps.")