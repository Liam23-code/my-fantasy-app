"""Player-detail modal with weekly projection and confidence context."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from fantasy.weekly_projections import build_weekly_projection, weekly_matchups

from app.style import (
    CIRCUIT_CYAN,
    SIGNAL_GOLD,
    gold_glow_line_chart,
    rarity_icon,
    safe_number,
)


def _rank(player: Mapping[str, Any]) -> int:
    value = player.get("overall_rank", player.get("rank", player.get("adp", 41)))
    return max(1, int(safe_number(value, 41)))


def _confidence(player: Mapping[str, Any]) -> float:
    value = safe_number(player.get("projection_confidence", player.get("confidence", 0.6)), 0.6)
    return max(0.0, min(1.0, value / 100.0 if value > 1.0 else value))


def player_detail_figure(
    player: Mapping[str, Any],
    scoring_mode: str | None = None,
) -> go.Figure:
    """Return the branded 18-week detail chart for one player."""
    curve = build_weekly_projection(player, scoring_mode)
    return gold_glow_line_chart(
        [curve[week]["points"] for week in curve],
        list(curve),
        title="Weekly projection curve",
        name="Projected points",
        height=360,
    )


@st.dialog("Player Detail", width="large")
def open_player_detail(player: Mapping[str, Any], scoring_mode: str | None = None) -> None:
    """Open the player-detail modal from any card surface."""
    rank = _rank(player)
    confidence = _confidence(player)
    projection = safe_number(player.get("projection", player.get("expected_fantasy_points")))
    curve = build_weekly_projection(player, scoring_mode)
    matchups = weekly_matchups(player)

    st.markdown(
        f'<div class="player-detail-title">{rarity_icon(rank)}'
        f'<div><h2>{escape(str(player.get("name", "Unknown player")))}</h2>'
        f'<p>{escape(str(player.get("position", "—")))} · '
        f'{escape(str(player.get("team") or player.get("nfl_team") or "FA"))}</p>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    rank_col, projection_col, confidence_col = st.columns(3)
    rank_col.metric("Overall rank", f"#{rank}")
    projection_col.metric("Season projection", f"{projection:.1f}")
    confidence_col.metric("Confidence", f"{confidence:.0%}")

    st.plotly_chart(
        player_detail_figure(player, scoring_mode),
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"detail-curve-{player.get('player_id') or player.get('id') or player.get('name')}",
    )

    weekly_points = [values["points"] for values in curve.values() if values["points"] > 0]
    average = sum(weekly_points) / len(weekly_points) if weekly_points else 0.0
    matchup_values = [
        values["defensive_adjustment"]
        for values in matchups.values()
        if values["defensive_adjustment"] > 0
    ]
    matchup_average = sum(matchup_values) / len(matchup_values) if matchup_values else 1.0
    floor = min(weekly_points, default=0.0)
    ceiling = max(weekly_points, default=0.0)
    st.markdown(
        '<div class="quant-card-stats player-detail-stats">'
        f'<div><span>Weekly average</span><strong style="color:{SIGNAL_GOLD}">{average:.2f}</strong></div>'
        f'<div><span>Floor</span><strong>{floor:.2f}</strong></div>'
        f'<div><span>Ceiling</span><strong style="color:{CIRCUIT_CYAN}">{ceiling:.2f}</strong></div>'
        f'<div><span>Matchup index</span><strong>{matchup_average:.3f}</strong></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.page_link("pages/29_Graph_Lab.py", label="Open in Graph Lab", icon=":material/monitoring:")


__all__ = ["open_player_detail", "player_detail_figure"]
