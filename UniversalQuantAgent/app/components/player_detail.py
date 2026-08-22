"""Quant-powered player-detail modal with weekly context and comparisons."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Direct component imports can run outside a Fantasy page's path preamble. The
# sibling source root keeps the new ``quant`` package available to older local
# editable installs without changing normal installed-package resolution.
_ENGINE_ROOT = str(Path(__file__).resolve().parents[3] / "fantasy_engine")
if Path(_ENGINE_ROOT).is_dir() and _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from fantasy.weekly_projections import build_weekly_projection, weekly_matchups
from quant.quant_engine import (
    compute_base_projections,
    compute_confidence_scores,
    compute_efficiency_scores,
    compute_momentum,
    compute_player_similarity,
    compute_rarity_tier,
    compute_trend_lines,
    compute_volatility,
)

from app.style import (
    CIRCUIT_CYAN,
    SIGNAL_GOLD,
    gold_glow_chart,
    rarity_icon,
    safe_number,
)


def _player_id(player: Mapping[str, Any]) -> str:
    return str(player.get("player_id") or player.get("id") or player.get("name") or "").strip()


def _rank(player: Mapping[str, Any]) -> int:
    value = player.get("overall_rank", player.get("rank", player.get("adp", 41)))
    return max(1, int(safe_number(value, 41)))


def _confidence(player: Mapping[str, Any]) -> float:
    value = safe_number(player.get("projection_confidence", player.get("confidence", 0.6)), 0.6)
    return max(0.0, min(1.0, value / 100.0 if value > 1.0 else value))


def _quant_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        payload = function(*args, **kwargs)
    except (ArithmeticError, AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _quant_row(payload: Mapping[str, Any], player: Mapping[str, Any]) -> dict[str, Any]:
    player_id = _player_id(player)
    by_player = payload.get("by_player")
    if isinstance(by_player, Mapping) and isinstance(by_player.get(player_id), Mapping):
        return dict(by_player[player_id])
    result = payload.get("result")
    if isinstance(result, Mapping) and (not player_id or _player_id(result) in {"", player_id}):
        return dict(result)
    results = payload.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        for row in results:
            if isinstance(row, Mapping) and _player_id(row) == player_id:
                return dict(row)
    return {}


def _pool_with_player(
    player: Mapping[str, Any],
    player_pool: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source: Any = player_pool
    if source is None:
        source = st.session_state.get("fantasy_projections", [])
    pool = [dict(candidate) for candidate in source if isinstance(candidate, Mapping)] if isinstance(source, Sequence) else []
    target_id = _player_id(player)
    if not any(_player_id(candidate) == target_id for candidate in pool):
        pool.append(dict(player))
    return pool


def _analytical_player(
    player: Mapping[str, Any],
    scoring_mode: str | None,
) -> tuple[dict[str, Any], dict[int, dict[str, float]], dict[str, Any]]:
    mode = str(scoring_mode or player.get("scoring_mode") or "ppr").strip().lower()
    base_payload = _quant_call(compute_base_projections, dict(player), scoring_mode=mode)
    base_row = _quant_row(base_payload, player)
    analytical = dict(player)
    projection = safe_number(
        base_row.get("base_projection"),
        player.get("projection", player.get("expected_fantasy_points")),
    )
    if projection > 0.0:
        analytical["projection"] = projection
        analytical["expected_fantasy_points"] = projection
    analytical["scoring_mode"] = mode
    curve = build_weekly_projection(analytical, mode)
    if not any(key in analytical for key in ("history", "game_log", "weekly_points", "recent_points")):
        analytical["weekly_projection"] = curve
    return analytical, curve, base_row


def player_detail_figure(
    player: Mapping[str, Any],
    scoring_mode: str | None = None,
) -> go.Figure:
    """Return the branded weekly chart rebased to the Quant Engine projection."""

    _player, curve, _base = _analytical_player(player, scoring_mode)
    return gold_glow_chart(curve, title="Weekly quant projection curve", name="Projected points", height=360)


@st.dialog("Player Detail", width="large")
def open_player_detail(
    player: Mapping[str, Any],
    scoring_mode: str | None = None,
    player_pool: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Open a unified Quant Engine player profile from any fantasy card."""

    analytical, curve, base_row = _analytical_player(player, scoring_mode)
    pool = _pool_with_player(player, player_pool)
    analytical_pool = [analytical if _player_id(candidate) == _player_id(player) else candidate for candidate in pool]

    confidence_row = _quant_row(_quant_call(compute_confidence_scores, analytical), analytical)
    volatility_row = _quant_row(_quant_call(compute_volatility, analytical), analytical)
    trend_row = _quant_row(_quant_call(compute_trend_lines, analytical, window=3), analytical)
    momentum_row = _quant_row(_quant_call(compute_momentum, analytical, window=3), analytical)
    efficiency_row = _quant_row(_quant_call(compute_efficiency_scores, analytical_pool), analytical)
    rarity_row = _quant_row(_quant_call(compute_rarity_tier, analytical_pool), analytical)
    similarity_payload = _quant_call(compute_player_similarity, analytical, analytical_pool, limit=3)

    rank = max(1, int(safe_number(rarity_row.get("rank"), _rank(player))))
    rarity_tier = str(rarity_row.get("rarity_tier") or rarity_row.get("tier") or "Depth")
    archetype = str(similarity_payload.get("archetype") or "Balanced Archetype")
    confidence = max(0.0, min(1.0, safe_number(confidence_row.get("confidence_score"), _confidence(player))))
    projection = safe_number(
        base_row.get("base_projection"),
        analytical.get("projection", analytical.get("expected_fantasy_points")),
    )
    volatility = max(0.0, min(1.0, safe_number(volatility_row.get("volatility"), 1.0 - confidence)))
    momentum = max(-1.0, min(1.0, safe_number(momentum_row.get("momentum"))))
    direction = str(momentum_row.get("direction") or trend_row.get("direction") or "flat")
    efficiency = max(0.0, min(100.0, safe_number(efficiency_row.get("efficiency_score"))))

    st.markdown(
        f'<div class="player-detail-title">{rarity_icon(rank)}'
        f'<div><h2>{escape(str(player.get("name", "Unknown player")))}</h2>'
        f'<p>{escape(str(player.get("position", "—")))} · '
        f'{escape(str(player.get("team") or player.get("nfl_team") or "FA"))} · '
        f'{escape(rarity_tier)} · {escape(archetype)}</p>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    primary_columns = st.columns(3)
    primary_columns[0].metric("Quant projection", f"{projection:.1f}")
    primary_columns[1].metric("Confidence", f"{confidence:.0%}")
    primary_columns[2].metric("Rarity", rarity_tier, f"Rank #{rank}")
    signal_columns = st.columns(3)
    signal_columns[0].metric("Momentum", f"{momentum:+.1%}", direction.title())
    signal_columns[1].metric("Efficiency", f"{efficiency:.0f}/100")
    signal_columns[2].metric(
        "Volatility",
        f"{volatility:.0%}",
        str(volatility_row.get("risk_level") or "modeled").title(),
    )

    st.plotly_chart(
        gold_glow_chart(curve, title="Weekly quant projection curve", name="Projected points", height=350),
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"detail-curve-{_player_id(player)}",
    )

    weekly_points = [safe_number(values.get("points")) for values in curve.values() if safe_number(values.get("points")) > 0]
    average = sum(weekly_points) / len(weekly_points) if weekly_points else 0.0
    matchups = weekly_matchups(analytical)
    matchup_values = [
        safe_number(values.get("defensive_adjustment"))
        for values in matchups.values()
        if safe_number(values.get("defensive_adjustment")) > 0
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

    trend_points_source = trend_row.get("points")
    trend_points = (
        [safe_number(value) for value in trend_points_source]
        if isinstance(trend_points_source, Sequence) and not isinstance(trend_points_source, (str, bytes, bytearray))
        else []
    )
    trend_line = trend_row.get("rolling_points")
    trend_labels = trend_row.get("games")
    if trend_points and isinstance(trend_line, Sequence) and not isinstance(trend_line, (str, bytes, bytearray)):
        labels = (
            list(trend_labels)
            if isinstance(trend_labels, Sequence) and not isinstance(trend_labels, (str, bytes, bytearray))
            else list(range(1, len(trend_points) + 1))
        )
        trend_figure = gold_glow_chart(
            list(trend_line),
            x=labels,
            title=f"Quant trend · {direction.upper()} · {momentum:+.1%}",
            name="3-week trend",
            height=300,
        )
        trend_figure.add_trace(
            go.Scatter(
                x=labels,
                y=trend_points,
                mode="lines+markers",
                name="Weekly signal",
                line={"color": CIRCUIT_CYAN, "width": 1.3, "dash": "dot"},
                marker={"color": CIRCUIT_CYAN, "size": 5},
            )
        )
        st.plotly_chart(
            trend_figure,
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"detail-trend-{_player_id(player)}",
        )

    weeks = list(curve)
    raw_confidences = [safe_number(curve[week].get("confidence"), confidence) for week in weeks]
    source_confidence = _confidence(player)
    confidence_scale = confidence / source_confidence if source_confidence > 0.0 else 1.0
    confidence_curve = [max(0.0, min(1.0, value * confidence_scale)) * 100.0 for value in raw_confidences]
    st.plotly_chart(
        gold_glow_chart(
            confidence_curve,
            x=weeks,
            title="Quant confidence curve",
            name="Confidence",
            height=280,
            y_suffix="%",
        ),
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"detail-confidence-{_player_id(player)}",
    )

    comparisons = similarity_payload.get("comparisons")
    comparison_rows = [dict(row) for row in comparisons if isinstance(row, Mapping)] if isinstance(comparisons, Sequence) else []
    st.markdown("#### Nearest player comps")
    if comparison_rows:
        st.dataframe(
            [
                {
                    "Player": row.get("name") or row.get("player_id"),
                    "Similarity": f"{safe_number(row.get('similarity_percent')):.1f}%",
                    "Archetype": row.get("archetype") or "Balanced",
                }
                for row in comparison_rows
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No same-position comparison is available in the current player pool.")

    st.page_link("pages/29_Graph_Lab.py", label="Open in Graph Lab", icon=":material/monitoring:")


__all__ = ["open_player_detail", "player_detail_figure"]
