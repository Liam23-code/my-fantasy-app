"""Premium visual laboratory for fantasy player projection signals."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

import plotly.graph_objects as go
import streamlit as st
from app.fantasy_shared import league_setup
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    section_header,
)
from app.style import (
    CIRCUIT_CYAN,
    SIGNAL_GOLD,
    SLATE_BLUE,
    SOFT_WHITE,
    apply_gold_glow_theme,
    gold_glow_line_chart,
    rarity_icon,
    safe_number,
)
from fantasy.weekly_projections import build_weekly_projection, weekly_matchups


def _projection(player: dict[str, Any]) -> float:
    return safe_number(player.get("projection", player.get("expected_fantasy_points")))


def _confidence(player: dict[str, Any]) -> float:
    value = safe_number(player.get("projection_confidence", player.get("confidence", 0.6)), 0.6)
    return max(0.0, min(1.0, value / 100.0 if value > 1.0 else value))


def _radar_figure(
    player: dict[str, Any],
    points: list[float],
    matchup_index: float,
    projection_percentile: float,
) -> go.Figure:
    active = [point for point in points if point > 0]
    average = sum(active) / len(active) if active else 0.0
    floor = min(active, default=0.0)
    ceiling = max(active, default=0.0)
    categories = ["Projection", "Confidence", "Floor", "Ceiling", "Matchup", "Stability"]
    values = [
        projection_percentile,
        _confidence(player) * 100.0,
        min(100.0, 100.0 * floor / average) if average else 0.0,
        min(100.0, 55.0 * ceiling / average) if average else 0.0,
        max(0.0, min(100.0, 50.0 + (matchup_index - 1.0) * 200.0)),
        max(0.0, 100.0 - ((ceiling - floor) / average * 100.0)) if average else 0.0,
    ]
    figure = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            line={"color": SIGNAL_GOLD, "width": 2},
            fillcolor="rgba(245,197,66,.18)",
            marker={"color": CIRCUIT_CYAN, "size": 7},
            name="Player profile",
        )
    )
    figure.update_layout(
        title="Projection radar",
        polar={
            "bgcolor": "rgba(10,26,47,.7)",
            "radialaxis": {"range": [0, 100], "gridcolor": "rgba(0,200,255,.15)", "tickfont": {"color": SOFT_WHITE}},
            "angularaxis": {"gridcolor": "rgba(0,200,255,.15)", "tickfont": {"color": SOFT_WHITE}},
        },
        showlegend=False,
    )
    return apply_gold_glow_theme(figure, height=430)


def _momentum_figure(points: list[float]) -> go.Figure:
    rolling = [sum(points[max(0, index - 2) : index + 1]) / min(index + 1, 3) for index in range(len(points))]
    figure = gold_glow_line_chart(
        points,
        list(range(1, 19)),
        title="Weekly momentum",
        name="Projection",
        height=360,
    )
    figure.add_trace(
        go.Scatter(
            x=list(range(1, 19)),
            y=rolling,
            mode="lines",
            name="3-week momentum",
            line={"color": CIRCUIT_CYAN, "width": 1.6, "dash": "dot"},
        )
    )
    figure.update_layout(showlegend=True)
    return figure


def _efficiency_wheel(player: dict[str, Any], points: list[float], matchup_index: float) -> go.Figure:
    active = [point for point in points if point > 0]
    average = sum(active) / len(active) if active else 0.0
    ceiling = max(active, default=0.0)
    floor = min(active, default=0.0)
    figure = go.Figure(
        go.Pie(
            labels=["Weekly output", "Confidence", "Upside", "Matchup"],
            values=[average, _confidence(player) * 20.0, max(0.0, ceiling - floor), matchup_index * 15.0],
            hole=0.64,
            marker={"colors": [SIGNAL_GOLD, CIRCUIT_CYAN, "#7891B4", SLATE_BLUE]},
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Efficiency wheel",
        annotations=[
            {
                "text": f"{average:.1f}<br><span style='font-size:11px'>PTS/WK</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": SOFT_WHITE, "size": 20},
            }
        ],
        showlegend=False,
    )
    return apply_gold_glow_theme(figure, height=430)


apply_global_theme()
setup = league_setup()

page_header(
    "Graph Lab",
    "Inspect a player's projection shape, stability, momentum, matchup context, and efficiency profile.",
    eyebrow="Visual intelligence",
)

players = [dict(player) for player in setup["projections"] if isinstance(player, dict)]
players.sort(key=_projection, reverse=True)
for rank, player in enumerate(players, start=1):
    player.setdefault("overall_rank", rank)

if not players:
    empty_state(
        "No player data",
        "Load the forward projection pool in League & draft setup to open Graph Lab.",
        icon="📊",
    )
    st.stop()

player_by_id = {
    str(player.get("player_id") or player.get("id") or player.get("name")): player for player in players
}
selected_id = st.selectbox(
    "Player",
    list(player_by_id),
    format_func=lambda player_id: (
        f"{player_by_id[player_id].get('name', player_id)} · "
        f"{player_by_id[player_id].get('position', '—')} · #{player_by_id[player_id].get('overall_rank', 41)}"
    ),
    key="graph_lab_player",
)
selected = player_by_id[selected_id]
rank = int(safe_number(selected.get("overall_rank"), 41))

st.markdown(
    f'<div class="player-detail-title">{rarity_icon(rank)}<div>'
    f'<h2>{escape(str(selected.get("name", "Unknown player")))}</h2>'
    f'<p>{escape(str(selected.get("position", "—")))} · '
    f'{escape(str(selected.get("team") or selected.get("nfl_team") or "FA"))}</p>'
    "</div></div>",
    unsafe_allow_html=True,
)

curve = build_weekly_projection(selected, setup["scoring_mode"])
matchups = weekly_matchups(selected)
weeks = list(curve)
points = [curve[week]["points"] for week in weeks]
confidences = [curve[week]["confidence"] * 100.0 for week in weeks]
matchup_values = [
    values["defensive_adjustment"]
    for values in matchups.values()
    if values["defensive_adjustment"] > 0
]
matchup_index = sum(matchup_values) / len(matchup_values) if matchup_values else 1.0
season_totals = [_projection(player) for player in players]
projection_percentile = 100.0 * sum(total <= _projection(selected) for total in season_totals) / len(season_totals)

section_header("Projection Curve", "Thin gold weekly signal with cyan peak markers.")
with st.container(border=True):
    st.plotly_chart(
        gold_glow_line_chart(points, weeks, title="18-week projected points", name="Projected points", height=390),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-projection",
    )

section_header("Projection Radar", "Six-axis view of output, confidence, stability, upside, and matchup context.")
with st.container(border=True):
    st.plotly_chart(
        _radar_figure(selected, points, matchup_index, projection_percentile),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-radar",
    )

section_header("Efficiency Wheel", "How weekly output, confidence, upside, and matchup contribute to the profile.")
with st.container(border=True):
    st.plotly_chart(
        _efficiency_wheel(selected, points, matchup_index),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-wheel",
    )

section_header("Momentum", "Weekly output plus the three-week projection trend.")
with st.container(border=True):
    st.plotly_chart(
        _momentum_figure(points),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-momentum",
    )

section_header("Confidence Curve", "Forecast certainty fades modestly with time and reacts to matchup data.")
with st.container(border=True):
    st.plotly_chart(
        gold_glow_line_chart(
            confidences,
            weeks,
            title="Weekly confidence",
            name="Confidence",
            height=320,
            y_suffix="%",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-confidence",
    )
