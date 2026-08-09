"""Multi-source ADP fusion.

Usage::

    from fantasy.adp_fusion import apply_fused_adp, fuse_adp

    players = apply_fused_adp(players)  # overlays fused_adp onto player["adp"]
    table = fuse_adp(players)           # {player_id: {"fused_adp": ..., ...}}

What's real and what isn't -- read this before trusting a number here:

Real
    FantasyPros consensus ADP (``fantasy.data_loader.load_adp``) -- the one
    genuine multi-expert-aggregated ADP source available to this project.

Synthetic, explicitly labeled (per direct instruction, not a default choice)
    ``espn`` and ``sleeper`` are deterministic *persona* overlays on the real
    FantasyPros baseline: documented positional multipliers approximating each
    platform's well-known drafting tendencies (ESPN's default rooms skew
    RB-early and chalky; Sleeper's userbase skews more pass-catcher-friendly
    and analytics-forward). They are **not** fetched from espn.com or
    sleeper.app -- no such feed exists through any tool available to this
    project (``nflreadpy`` exposes 25 loaders; none is a per-site ADP feed --
    see ``fantasy.mock_data_ingestion`` for the same finding). Every value
    under these keys is derived arithmetically from the real baseline, never
    an independent measurement.

Not present at all
    Underdog, Yahoo, CBS, and NFL.com have no real or synthetic signal here.
    They are simply absent from the fusion for every player; their configured
    weight is excluded and the rest renormalized, rather than inventing a
    placeholder for them.

Planned, not built
    :func:`real_sleeper_adp` is a stub returning ``{}``. Sleeper has a public
    API, but whether it exposes genuine consensus ADP/ranking data (as
    opposed to its primary purpose, league/roster management) has not been
    verified. Wire a real implementation there once confirmed; :func:`fuse_adp`
    picks it up automatically in place of the synthetic Sleeper persona.

``adp_trend_direction`` is **not** a time-series trend -- there is exactly one
scrape date in the real feed (checked directly), so no player-rising/falling-
over-time signal can be computed honestly. What's returned instead is whether
the fused view sits earlier ("rising"), later ("falling"), or about the same
("stable") as the real FantasyPros baseline alone -- a same-snapshot
cross-source comparison, not a week-over-week movement claim.
"""

from __future__ import annotations

import statistics
from typing import Any

from fantasy.utils import safe_float

#: Section 1's specified per-source weights. A player's fusion only ever uses
#: whichever of these actually have a value (today: fantasypros, espn,
#: sleeper) -- the rest are excluded and the remaining weights renormalized.
SOURCE_WEIGHTS: dict[str, float] = {
    "espn": 0.30,
    "sleeper": 0.20,
    "underdog": 0.15,
    "fantasypros": 0.15,
    "yahoo": 0.10,
    "cbs": 0.05,
    "nfl": 0.05,
}

#: Sources with a real data path today.
REAL_SOURCES = ("fantasypros",)

#: Synthetic-persona sources: deterministic positional multipliers on the real
#: FantasyPros baseline, approximating each platform's documented drafting
#: tendencies. Not measured from real per-site data -- see module docstring.
ESPN_SYNTHETIC_POSITION_BIAS: dict[str, float] = {
    "RB": 0.95,
    "WR": 0.98,
    "QB": 1.08,
    "TE": 1.05,
    "K": 1.0,
    "DST": 1.0,
}
SLEEPER_SYNTHETIC_POSITION_BIAS: dict[str, float] = {
    "RB": 1.04,
    "WR": 0.96,
    "QB": 1.02,
    "TE": 0.97,
    "K": 1.0,
    "DST": 1.0,
}

#: Sources with no real or synthetic signal in this project at all.
UNAVAILABLE_SOURCES = ("underdog", "yahoo", "cbs", "nfl")


def real_sleeper_adp(players: list[dict[str, Any]]) -> dict[str, float]:  # noqa: ARG001
    """Placeholder for a genuine Sleeper API integration.

    Returns ``{}`` today. See the module docstring's "Planned, not built"
    section -- Sleeper's public API has not been confirmed to expose real
    consensus ADP data. :func:`fuse_adp` uses this in place of the synthetic
    Sleeper persona for any player_id present in its return value.
    """
    return {}


def _synthetic_source_adp(real_adp: float, position: str, bias: dict[str, float]) -> float:
    return round(real_adp * bias.get(position, 1.0), 2)


def _fuse_one_player(sources: dict[str, float]) -> dict[str, Any]:
    weights = {source: SOURCE_WEIGHTS[source] for source in sources if source in SOURCE_WEIGHTS}
    total_weight = sum(weights.values())
    if not weights or total_weight <= 0:
        return {
            "fused_adp": None,
            "adp_volatility": None,
            "adp_confidence": None,
            "adp_trend_direction": None,
            "sources": dict(sources),
        }

    fused = sum(sources[source] * weight for source, weight in weights.items()) / total_weight
    values = list(sources.values())
    volatility = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0

    # Confidence blends coverage (how much of the configured weight is
    # actually backed by a value today -- Underdog/Yahoo/CBS/NFL.com always
    # contribute 0) with agreement (how tightly the available sources cluster).
    coverage = total_weight / sum(SOURCE_WEIGHTS.values())
    agreement = 1.0 / (1.0 + volatility / 10.0)
    confidence = round(coverage * agreement, 3)

    baseline = sources.get("fantasypros")
    trend: str | None
    if baseline is None:
        trend = None
    else:
        delta = baseline - fused  # positive: fused view sits earlier than the real baseline alone
        if delta > 1.0:
            trend = "rising"
        elif delta < -1.0:
            trend = "falling"
        else:
            trend = "stable"

    return {
        "fused_adp": round(fused, 2),
        "adp_volatility": volatility,
        "adp_confidence": confidence,
        "adp_trend_direction": trend,
        "sources": dict(sources),
    }


def fuse_adp(players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fuse each player's available ADP sources into one weighted view.

    Returns a table keyed by ``player_id``::

        {"00-0033280": {"fused_adp": 8.9, "adp_volatility": 1.4,
                         "adp_confidence": 0.577, "adp_trend_direction": "stable",
                         "sources": {"fantasypros": 8.5, "espn": 8.08, "sleeper": 8.84}},
         ...}

    Players missing a real (FantasyPros) ``adp`` or a ``player_id`` are
    skipped entirely -- there is nothing real to fuse for them.
    """
    table: dict[str, dict[str, Any]] = {}
    sleeper_real = real_sleeper_adp(players)
    for player in players:
        player_id = player.get("player_id") or player.get("id")
        real_adp = player.get("adp")
        if not player_id or real_adp is None:
            continue
        real_adp_value = safe_float(real_adp)
        position = str(player.get("position", "")).strip().upper()

        sources: dict[str, float] = {"fantasypros": real_adp_value}
        sources["espn"] = _synthetic_source_adp(real_adp_value, position, ESPN_SYNTHETIC_POSITION_BIAS)
        player_id_str = str(player_id)
        sources["sleeper"] = (
            sleeper_real[player_id_str]
            if player_id_str in sleeper_real
            else _synthetic_source_adp(real_adp_value, position, SLEEPER_SYNTHETIC_POSITION_BIAS)
        )

        table[player_id_str] = _fuse_one_player(sources)
    return table


def apply_fused_adp(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay the fused ADP view onto each player, in place of their raw ADP.

    This is the single integration point: every downstream module
    (``room_brain``, ``user_brain`` via ``assistant``, ``tiering``,
    ``simulate_draft``) already reads ``player["adp"]`` for timing decisions,
    so promoting the fused value to that same field reaches all of them
    without touching their internals. The original single-source value is
    preserved under ``adp_fantasypros_only`` -- nothing is lost, and a caller
    that wants the real-only baseline still has it.

    Players with no real ADP pass through unchanged (nothing to fuse).
    """
    fused_table = fuse_adp(players)
    updated: list[dict[str, Any]] = []
    for player in players:
        player = dict(player)
        player_id = str(player.get("player_id") or player.get("id") or "")
        fused = fused_table.get(player_id)
        if fused and fused.get("fused_adp") is not None:
            player["adp_fantasypros_only"] = player.get("adp")
            player["adp"] = fused["fused_adp"]
            player["adp_volatility"] = fused["adp_volatility"]
            player["adp_confidence"] = fused["adp_confidence"]
            player["adp_trend_direction"] = fused["adp_trend_direction"]
            player["adp_sources"] = fused["sources"]
        updated.append(player)
    return updated
