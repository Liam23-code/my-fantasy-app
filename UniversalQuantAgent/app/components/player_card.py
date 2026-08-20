"""Compact, clickable fantasy player cards."""

from __future__ import annotations

import re
from collections.abc import Mapping
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from fantasy.weekly_projections import build_weekly_projection

from app.components.player_detail import open_player_detail
from app.style import gold_glow_line_chart, rarity_icon, safe_number


def _rank(player: Mapping[str, Any]) -> int:
    value = player.get("overall_rank", player.get("rank", player.get("adp", 41)))
    return max(1, int(safe_number(value, 41)))


def _confidence(player: Mapping[str, Any]) -> float:
    value = safe_number(player.get("projection_confidence", player.get("confidence", 0.6)), 0.6)
    return max(0.0, min(1.0, value / 100.0 if value > 1.0 else value))


def _identity(player: Mapping[str, Any]) -> str:
    raw = player.get("player_id") or player.get("id") or player.get("name") or "player"
    return re.sub(r"[^a-z0-9]+", "-", str(raw).lower()).strip("-") or "player"


def player_card_markup(player: Mapping[str, Any]) -> str:
    """Return safe, testable markup for a player card header and stats."""
    rank = _rank(player)
    confidence = _confidence(player)
    projection = safe_number(player.get("projection", player.get("expected_fantasy_points")))
    name = escape(str(player.get("name") or player.get("player_name") or "Unknown player"))
    position = escape(str(player.get("position") or "—").upper())
    team = escape(str(player.get("team") or player.get("nfl_team") or "FA").upper())
    return (
        '<article class="quant-card stacked-card player-card-premium">'
        '<div class="quant-card-head"><div>'
        f'<div class="quant-card-kicker">{position} · {team}</div><h3>{name}</h3>'
        f'</div>{rarity_icon(rank)}</div>'
        '<div class="quant-card-stats">'
        f'<div><span>Rank</span><strong>#{rank}</strong></div>'
        f'<div><span>Confidence</span><strong>{confidence:.0%}</strong></div>'
        f'<div><span>Season projection</span><strong>{projection:.1f}</strong></div>'
        "</div></article>"
    )


def player_mini_chart(player: Mapping[str, Any], scoring_mode: str | None = None) -> go.Figure:
    """Return the small gold weekly curve shown inside a player card."""
    curve = build_weekly_projection(player, scoring_mode)
    figure = gold_glow_line_chart(
        [curve[week]["points"] for week in curve],
        list(curve),
        name="Weekly points",
        height=180,
    )
    figure.update_layout(margin={"l": 8, "r": 8, "t": 8, "b": 12})
    figure.update_xaxes(showticklabels=False, title=None)
    figure.update_yaxes(showticklabels=False, title=None)
    return figure


def render_player_card(
    player: Mapping[str, Any],
    *,
    scoring_mode: str | None = None,
    key: str | None = None,
    show_chart: bool = True,
) -> None:
    """Render one card and open its detail modal when selected."""
    identity = key or _identity(player)
    with st.container(border=True):
        st.markdown(player_card_markup(player), unsafe_allow_html=True)
        if show_chart:
            st.plotly_chart(
                player_mini_chart(player, scoring_mode),
                use_container_width=True,
                config={"displayModeBar": False, "staticPlot": True},
                key=f"player-mini-{identity}",
            )
        if st.button("View player detail", key=f"player-detail-{identity}", use_container_width=True):
            open_player_detail(player, scoring_mode)


__all__ = ["player_card_markup", "player_mini_chart", "render_player_card"]
