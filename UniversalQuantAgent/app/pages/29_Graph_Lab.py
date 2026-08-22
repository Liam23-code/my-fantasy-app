"""Quant-powered visual laboratory for fantasy player signals."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

# ``fantasy`` is installed editable for the app, but ``quant`` is a newly
# added sibling package and may not yet be present in an older editable-install
# mapping. Adding the engine root keeps local/offline launches self-contained.
_ENGINE_ROOT = str(Path(__file__).resolve().parents[3] / "fantasy_engine")
if Path(_ENGINE_ROOT).is_dir() and _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

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
    gold_glow_chart,
    rarity_icon,
    safe_number,
)
from fantasy.weekly_projections import build_weekly_projection, weekly_matchups
from quant.quant_engine import (
    compute_base_projections,
    compute_confidence_scores,
    compute_efficiency_scores,
    compute_final_projection,
    compute_momentum,
    compute_player_similarity,
    compute_rarity_tier,
    compute_trend_lines,
    compute_usage_rates,
    compute_volatility,
    compute_weekly_matchup_score,
)


def _player_id(player: Mapping[str, Any]) -> str:
    return str(player.get("player_id") or player.get("id") or player.get("name") or "").strip()


def _projection(player: Mapping[str, Any]) -> float:
    return safe_number(player.get("projection", player.get("expected_fantasy_points")))


def _confidence(player: Mapping[str, Any]) -> float:
    value = safe_number(player.get("projection_confidence", player.get("confidence", 0.6)), 0.6)
    return max(0.0, min(1.0, value / 100.0 if value > 1.0 else value))


def _quant_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a quant calculation at the UI boundary with an offline-safe fallback."""

    try:
        payload = function(*args, **kwargs)
    except (ArithmeticError, AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        # Uploaded records are user-controlled and may be sparse. Engine
        # validation should not make the surrounding visual lab unavailable.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _quant_row(payload: Mapping[str, Any], player: Mapping[str, Any]) -> dict[str, Any]:
    player_id = _player_id(player)
    by_player = payload.get("by_player")
    if isinstance(by_player, Mapping):
        row = by_player.get(player_id)
        if isinstance(row, Mapping):
            return dict(row)
    result = payload.get("result")
    if isinstance(result, Mapping) and (not player_id or _player_id(result) in {"", player_id}):
        return dict(result)
    results = payload.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        for candidate in results:
            if isinstance(candidate, Mapping) and _player_id(candidate) == player_id:
                return dict(candidate)
    return {}


def _weekly_analytics_player(
    player: Mapping[str, Any],
    scoring_mode: str,
    base_row: Mapping[str, Any],
    final_row: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    analytical = dict(player)
    advanced = final_row or {}
    base_projection = safe_number(
        advanced.get("final_projection", advanced.get("projection")),
        safe_number(base_row.get("base_projection"), _projection(player)),
    )
    if base_projection > 0.0:
        analytical["projection"] = base_projection
        analytical["expected_fantasy_points"] = base_projection
    analytical["scoring_mode"] = scoring_mode
    curve = build_weekly_projection(analytical, scoring_mode)
    for week in curve:
        matchup_payload = _quant_call(
            compute_weekly_matchup_score,
            analytical,
            week,
            scoring_mode=scoring_mode,
        )
        matchup_row = _quant_row(matchup_payload, analytical)
        if matchup_row:
            curve[week]["points"] = safe_number(
                matchup_row.get("weekly_projected_points", matchup_row.get("adjusted_projection")),
                curve[week]["points"],
            )
            curve[week]["confidence"] = max(
                0.0,
                min(1.0, safe_number(matchup_row.get("confidence"), curve[week]["confidence"])),
            )
    if not any(key in analytical for key in ("history", "game_log", "weekly_points", "recent_points")):
        analytical["weekly_projection"] = curve
    return analytical, curve


def _radar_figure(
    player: Mapping[str, Any],
    points: list[float],
    matchup_index: float,
    projection_percentile: float,
    quant_metrics: Mapping[str, float] | None = None,
) -> go.Figure:
    active = [point for point in points if point > 0]
    average = sum(active) / len(active) if active else 0.0
    floor = min(active, default=0.0)
    ceiling = max(active, default=0.0)
    metrics = quant_metrics or {}
    categories = ["Projection", "Confidence", "Efficiency", "Momentum", "Matchup", "Stability"]
    default_stability = max(0.0, 100.0 - ((ceiling - floor) / average * 100.0)) if average else 0.0
    values = [
        projection_percentile,
        safe_number(metrics.get("confidence"), _confidence(player) * 100.0),
        safe_number(metrics.get("efficiency"), min(100.0, 100.0 * floor / average) if average else 0.0),
        max(0.0, min(100.0, 50.0 + safe_number(metrics.get("momentum"), 0.0) * 50.0)),
        max(0.0, min(100.0, 50.0 + (matchup_index - 1.0) * 200.0)),
        safe_number(metrics.get("stability"), default_stability),
    ]
    values = [max(0.0, min(100.0, value)) for value in values]
    figure = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            line={"color": SIGNAL_GOLD, "width": 2},
            fillcolor="rgba(245,197,66,.18)",
            marker={"color": CIRCUIT_CYAN, "size": 7},
            name="Quant profile",
        )
    )
    figure.update_layout(
        title="Quant projection radar",
        polar={
            "bgcolor": "rgba(10,26,47,.7)",
            "radialaxis": {"range": [0, 100], "gridcolor": "rgba(0,200,255,.15)", "tickfont": {"color": SOFT_WHITE}},
            "angularaxis": {"gridcolor": "rgba(0,200,255,.15)", "tickfont": {"color": SOFT_WHITE}},
        },
        showlegend=False,
    )
    return apply_gold_glow_theme(figure, height=430)


def _momentum_figure(
    points: list[float],
    rolling: Sequence[float | None] | None = None,
    *,
    labels: Sequence[Any] | None = None,
    direction: str = "flat",
    momentum: float = 0.0,
) -> go.Figure:
    x_values = list(labels) if labels is not None and len(labels) == len(points) else list(range(1, len(points) + 1))
    rolling_values = list(rolling) if rolling is not None and len(rolling) == len(points) else [
        sum(points[max(0, index - 2) : index + 1]) / min(index + 1, 3) for index in range(len(points))
    ]
    figure = gold_glow_chart(
        rolling_values,
        x=x_values,
        title=f"Quant trend · {direction.upper()} · {momentum:+.1%}",
        name="3-week trend",
        height=370,
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=points,
            mode="lines+markers",
            name="Weekly signal",
            line={"color": CIRCUIT_CYAN, "width": 1.4, "dash": "dot"},
            marker={"color": CIRCUIT_CYAN, "size": 5},
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(showlegend=True)
    return figure


def _efficiency_wheel(
    player: Mapping[str, Any],
    points: list[float],
    matchup_index: float,
    *,
    efficiency_score: float | None = None,
    usage_percent: float | None = None,
    stability: float | None = None,
) -> go.Figure:
    active = [point for point in points if point > 0]
    average = sum(active) / len(active) if active else 0.0
    ceiling = max(active, default=0.0)
    floor = min(active, default=0.0)
    fallback_stability = max(0.0, 100.0 - (ceiling - floor) / average * 100.0) if average else 0.0
    efficiency = max(0.0, min(100.0, safe_number(efficiency_score, average * 5.0)))
    usage = max(0.0, min(100.0, safe_number(usage_percent, _confidence(player) * 100.0)))
    stable = max(0.0, min(100.0, safe_number(stability, fallback_stability)))
    matchup = max(0.0, min(100.0, 50.0 + (matchup_index - 1.0) * 200.0))
    figure = go.Figure(
        go.Pie(
            labels=["Efficiency", "Usage", "Stability", "Matchup"],
            values=[efficiency, usage, stable, matchup],
            hole=0.64,
            marker={"colors": [SIGNAL_GOLD, CIRCUIT_CYAN, "#7891B4", SLATE_BLUE]},
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Quant efficiency wheel",
        annotations=[
            {
                "text": f"{efficiency:.0f}<br><span style='font-size:11px'>EFF</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": SOFT_WHITE, "size": 20},
            }
        ],
        showlegend=False,
    )
    return apply_gold_glow_theme(figure, height=430)


def _volatility_figure(points: Sequence[float], standard_deviation: float, labels: Sequence[Any]) -> go.Figure:
    figure = gold_glow_chart(
        list(points),
        x=list(labels),
        title="Projection with quant volatility band",
        name="Projected points",
        height=350,
    )
    deviation = max(0.0, standard_deviation)
    lower = [max(0.0, point - deviation) for point in points]
    upper = [point + deviation for point in points]
    figure.add_trace(
        go.Scatter(
            x=list(labels),
            y=lower,
            mode="lines",
            line={"color": "rgba(0,200,255,.22)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=list(labels),
            y=upper,
            mode="lines",
            line={"color": "rgba(0,200,255,.22)", "width": 1},
            fill="tonexty",
            fillcolor="rgba(0,200,255,.08)",
            name="Volatility band",
            hovertemplate="Upper band: %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(showlegend=True)
    return figure


def _similarity_figure(comparisons: Sequence[Mapping[str, Any]]) -> go.Figure:
    ordered = list(reversed(comparisons[:5]))
    figure = go.Figure(
        go.Bar(
            x=[safe_number(row.get("similarity_percent"), safe_number(row.get("similarity")) * 100.0) for row in ordered],
            y=[str(row.get("name") or row.get("player_id") or "Comparable") for row in ordered],
            orientation="h",
            marker={"color": [SIGNAL_GOLD if index == len(ordered) - 1 else CIRCUIT_CYAN for index in range(len(ordered))]},
            hovertemplate="%{y}: %{x:.1f}% similar<extra></extra>",
        )
    )
    figure.update_layout(title="Nearest player comparisons", xaxis={"range": [0, 100], "ticksuffix": "%"}, showlegend=False)
    return apply_gold_glow_theme(figure, height=max(280, 72 * len(ordered)))


def _rarity_breakdown_figure(rows: Sequence[Mapping[str, Any]]) -> tuple[go.Figure, list[dict[str, Any]]]:
    tier_order = ("Mythic", "Legendary", "Elite", "Pro", "Starter", "Depth")
    counts = {tier: 0 for tier in tier_order}
    for row in rows:
        tier = str(row.get("rarity_tier") or row.get("tier") or "Depth").title()
        counts[tier if tier in counts else "Depth"] += 1
    total = max(1, sum(counts.values()))
    summary = [
        {"Tier": tier, "Players": counts[tier], "Pool share": f"{counts[tier] / total:.1%}"}
        for tier in tier_order
        if counts[tier] > 0
    ]
    figure = go.Figure(
        go.Bar(
            x=[row["Tier"] for row in summary],
            y=[row["Players"] for row in summary],
            marker={
                "color": [SIGNAL_GOLD, "#E0B0FF", CIRCUIT_CYAN, "#7891B4", SLATE_BLUE, "#43536A"][: len(summary)]
            },
            hovertemplate="%{x}: %{y} players<extra></extra>",
        )
    )
    figure.update_layout(title="Pool-wide Quant rarity distribution", showlegend=False, yaxis={"dtick": 1})
    return apply_gold_glow_theme(figure, height=330), summary


apply_global_theme()
setup = league_setup()

page_header(
    "Graph Lab",
    "Inspect unified Quant Engine projections, comps, trends, momentum, efficiency, volatility, confidence, and rarity.",
    eyebrow="Visual intelligence",
)

players = [dict(player) for player in setup["projections"] if isinstance(player, Mapping)]
players.sort(key=_projection, reverse=True)
for rank_number, player_row in enumerate(players, start=1):
    player_row.setdefault("overall_rank", rank_number)

if not players:
    empty_state(
        "No player data",
        "Load the forward projection pool in League & draft setup to open Graph Lab.",
        icon="📊",
    )
    st.stop()

player_by_id = {_player_id(player): player for player in players}
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

base_payload = _quant_call(compute_base_projections, players, scoring_mode=setup["scoring_mode"])
base_row = _quant_row(base_payload, selected)
final_projection_row = _quant_call(
    compute_final_projection,
    selected,
    players=players,
    scoring_mode=setup["scoring_mode"],
)
analytical_player, curve = _weekly_analytics_player(
    selected,
    setup["scoring_mode"],
    base_row,
    final_projection_row,
)
matchups = weekly_matchups(analytical_player)
weeks = list(curve)
points = [safe_number(curve[week].get("points")) for week in weeks]

confidence_row = _quant_row(_quant_call(compute_confidence_scores, analytical_player), analytical_player)
volatility_row = _quant_row(_quant_call(compute_volatility, analytical_player), analytical_player)
trend_row = _quant_row(_quant_call(compute_trend_lines, analytical_player, window=3), analytical_player)
momentum_row = _quant_row(_quant_call(compute_momentum, analytical_player, window=3), analytical_player)
efficiency_row = _quant_row(_quant_call(compute_efficiency_scores, players), selected)
usage_row = _quant_row(_quant_call(compute_usage_rates, players), selected)
rarity_payload = _quant_call(compute_rarity_tier, players)
rarity_row = _quant_row(rarity_payload, selected)
similarity_payload = _quant_call(compute_player_similarity, analytical_player, players, limit=5)

rank = max(1, int(safe_number(rarity_row.get("rank"), selected.get("overall_rank", 41))))
rarity_tier = str(rarity_row.get("rarity_tier") or rarity_row.get("tier") or "Depth")
archetype = str(similarity_payload.get("archetype") or "Balanced Archetype")

st.markdown(
    f'<div class="player-detail-title">{rarity_icon(rank)}<div>'
    f'<h2>{escape(str(selected.get("name", "Unknown player")))}</h2>'
    f'<p>{escape(str(selected.get("position", "—")))} · '
    f'{escape(str(selected.get("team") or selected.get("nfl_team") or "FA"))} · '
    f'{escape(rarity_tier)} · {escape(archetype)}</p>'
    "</div></div>",
    unsafe_allow_html=True,
)

quant_projection = safe_number(
    final_projection_row.get("final_projection", final_projection_row.get("projection")),
    safe_number(base_row.get("base_projection"), _projection(selected)),
)
quant_confidence = max(0.0, min(1.0, safe_number(confidence_row.get("confidence_score"), _confidence(selected))))
volatility = max(0.0, min(1.0, safe_number(volatility_row.get("volatility"), 1.0 - quant_confidence)))
standard_deviation = max(0.0, safe_number(volatility_row.get("standard_deviation")))
momentum = max(-1.0, min(1.0, safe_number(momentum_row.get("momentum"))))
direction = str(momentum_row.get("direction") or trend_row.get("direction") or "flat")
efficiency_score = max(0.0, min(100.0, safe_number(efficiency_row.get("efficiency_score"))))
usage_percent = max(
    0.0,
    min(
        100.0,
        safe_number(usage_row.get("usage_percent"), safe_number(usage_row.get("usage_rate")) * 100.0),
    ),
)

summary_columns = st.columns(3)
summary_columns[0].metric("Quant projection", f"{quant_projection:.1f}")
summary_columns[1].metric("Confidence", f"{quant_confidence:.0%}")
summary_columns[2].metric("Volatility", f"{volatility:.0%}", str(volatility_row.get("risk_level") or "modeled").title())
signal_columns = st.columns(3)
signal_columns[0].metric("Momentum", f"{momentum:+.1%}", direction.title())
signal_columns[1].metric("Efficiency", f"{efficiency_score:.0f}/100")
signal_columns[2].metric("Rarity", rarity_tier, f"Rank #{rank}")

matchup_values = [
    safe_number(values.get("defensive_adjustment"))
    for values in matchups.values()
    if safe_number(values.get("defensive_adjustment")) > 0
]
matchup_index = sum(matchup_values) / len(matchup_values) if matchup_values else 1.0
base_rows = base_payload.get("results") if isinstance(base_payload.get("results"), Sequence) else []
season_totals = [safe_number(row.get("base_projection")) for row in base_rows if isinstance(row, Mapping)]
if not season_totals:
    season_totals = [_projection(player) for player in players]
projection_percentile = 100.0 * sum(total <= quant_projection for total in season_totals) / max(1, len(season_totals))

section_header("Projection Curve", "The weekly engine is rebased to the unified Quant Engine season projection.")
with st.container(border=True):
    st.plotly_chart(
        gold_glow_chart(points, x=weeks, title="18-week quant projection", name="Projected points", height=390),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-projection",
    )

section_header("Projection Radar", "Quant confidence, efficiency, momentum, matchup, stability, and value in one view.")
with st.container(border=True):
    st.plotly_chart(
        _radar_figure(
            selected,
            points,
            matchup_index,
            projection_percentile,
            {
                "confidence": quant_confidence * 100.0,
                "efficiency": efficiency_score,
                "momentum": momentum,
                "stability": (1.0 - volatility) * 100.0,
            },
        ),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-radar",
    )

section_header("Trend + Momentum", "Quant rolling averages distinguish direction from one-week noise.")
with st.container(border=True):
    trend_points_source = trend_row.get("points")
    trend_points = (
        [safe_number(value) for value in trend_points_source]
        if isinstance(trend_points_source, Sequence) and not isinstance(trend_points_source, (str, bytes, bytearray))
        else points
    )
    trend_line = trend_row.get("rolling_points")
    if not isinstance(trend_line, Sequence) or isinstance(trend_line, (str, bytes, bytearray)):
        trend_line = None
    trend_labels = trend_row.get("games")
    if not isinstance(trend_labels, Sequence) or isinstance(trend_labels, (str, bytes, bytearray)):
        trend_labels = None
    st.plotly_chart(
        _momentum_figure(trend_points or points, trend_line, labels=trend_labels, direction=direction, momentum=momentum),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-momentum",
    )

section_header("Efficiency Wheel", "Position-relative efficiency, role signal, stability, and schedule context.")
with st.container(border=True):
    st.plotly_chart(
        _efficiency_wheel(
            selected,
            points,
            matchup_index,
            efficiency_score=efficiency_score,
            usage_percent=usage_percent,
            stability=(1.0 - volatility) * 100.0,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-wheel",
    )

section_header("Volatility Profile", "Quant dispersion creates an uncertainty band around every active week.")
with st.container(border=True):
    st.plotly_chart(
        _volatility_figure(points, standard_deviation, weeks),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-volatility",
    )

weekly_confidences = [safe_number(curve[week].get("confidence"), quant_confidence) for week in weeks]
source_confidence = _confidence(selected)
confidence_scale = quant_confidence / source_confidence if source_confidence > 0.0 else 1.0
confidences = [max(0.0, min(1.0, value * confidence_scale)) * 100.0 for value in weekly_confidences]
section_header("Confidence Curve", "Quant evidence strength is carried through the full weekly horizon.")
with st.container(border=True):
    st.plotly_chart(
        gold_glow_chart(confidences, x=weeks, title="Weekly quant confidence", name="Confidence", height=320, y_suffix="%"),
        use_container_width=True,
        config={"displayModeBar": False},
        key="graph-lab-confidence",
    )

comparisons = similarity_payload.get("comparisons")
comparison_rows = [dict(row) for row in comparisons if isinstance(row, Mapping)] if isinstance(comparisons, Sequence) else []
section_header("Similarity Comps", f"Cosine stat-vector neighbors for the {archetype.lower()} profile.")
with st.container(border=True):
    if comparison_rows:
        st.plotly_chart(
            _similarity_figure(comparison_rows),
            use_container_width=True,
            config={"displayModeBar": False},
            key="graph-lab-similarity",
        )
        st.dataframe(
            [
                {
                    "Player": row.get("name") or row.get("player_id"),
                    "Position": row.get("position") or "—",
                    "Similarity": f"{safe_number(row.get('similarity_percent')):.1f}%",
                    "Archetype": row.get("archetype") or "Balanced",
                }
                for row in comparison_rows
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No same-position comparison with enough normalized data is available in this pool.")
