"""player_status_utils: the betting engine's read-only view of live availability.

A thin adapter over :mod:`fantasy.player_status` (the single source of truth,
populated by :mod:`fantasy.online.player_status_fetcher`). The betting package
never fetches anything; this module only ever reads the local
``fantasy_engine/data/player_status.json`` and is a total no-op until that
file is refreshed.

Every hard rule keys on :func:`live_status` -- the live overlay *only*, never a
projection row's own (possibly stale, offseason) ``injury_status`` -- so the
overlay is strictly additive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fantasy.player_status import (
    DOUBTFUL,
    HEALTHY,
    HOLDOUT,
    OUT,
    QUESTIONABLE,
    SUSPENDED,
    UNAVAILABLE_STATUSES,
    adjust_projection_for_status,
    adjust_score_for_status,
    effective_status,
    has_status_data,
    load_player_status,
    status_last_updated,
)

__all__ = [
    "DOUBTFUL",
    "HEALTHY",
    "HOLDOUT",
    "OUT",
    "QUESTIONABLE",
    "SUSPENDED",
    "UNAVAILABLE_STATUSES",
    "adjust_projection_for_status",
    "adjust_score_for_status",
    "effective_status",
    "exclude_out",
    "has_status_data",
    "is_holdout",
    "is_out",
    "is_suspended",
    "is_unavailable",
    "live_status",
    "load_player_status",
    "status_adjusted_row",
    "status_last_updated",
    "team_scoring_penalty",
]

#: Flat expected team-points a healthy starter at each offensive position is
#: worth -- the magnitude a moneyline/total shades by when that starter is
#: ruled out. A documented modelling assumption (roughly one score for a QB,
#: a field goal's worth for a lead back or WR1), *not* a per-player empirical
#: estimate; it exists so "star player OUT" can move a team total at all,
#: which the team-scoring-history model otherwise has no hook for.
POSITION_OUT_POINTS: dict[str, float] = {"QB": 7.0, "RB": 2.5, "WR": 2.5, "TE": 1.5}

#: How much of that flat penalty each status applies.
_PENALTY_SHARE: dict[str, float] = {
    OUT: 1.0, HOLDOUT: 1.0, SUSPENDED: 1.0, DOUBTFUL: 0.5, QUESTIONABLE: 0.15, HEALTHY: 0.0,
}


def _as_player(player_or_id: Mapping[str, Any] | str) -> Mapping[str, Any]:
    return {"player_id": player_or_id} if isinstance(player_or_id, str) else player_or_id


def live_status(player_or_id: Mapping[str, Any] | str) -> str:
    """The status from the live overlay only -- HEALTHY when the feed is silent.

    Accepts a player mapping (matched by ``player_id`` then normalized name) or
    a bare player-id string.
    """
    from fantasy.player_status import live_status as _live

    return _live(_as_player(player_or_id))


def is_out(player_or_id: Mapping[str, Any] | str) -> bool:
    return live_status(player_or_id) == OUT


def is_holdout(player_or_id: Mapping[str, Any] | str) -> bool:
    return live_status(player_or_id) == HOLDOUT


def is_suspended(player_or_id: Mapping[str, Any] | str) -> bool:
    return live_status(player_or_id) == SUSPENDED


def is_unavailable(player_or_id: Mapping[str, Any] | str) -> bool:
    """OUT, HOLDOUT or SUSPENDED per the live overlay -- cannot play."""
    return live_status(player_or_id) in UNAVAILABLE_STATUSES


def exclude_out(players: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    """Drop every live-OUT player. A no-op copy when the overlay is empty."""
    source = list(players or [])
    if not has_status_data():
        return source
    return [player for player in source if live_status(player) != OUT]


def status_adjusted_row(
    player: Mapping[str, Any],
    fields: Iterable[str],
    *,
    status: str | None = None,
) -> dict[str, Any]:
    """Shallow-copy ``player`` with each of ``fields`` scaled by availability:
    0 for OUT / HOLDOUT / SUSPENDED, a mild factor for DOUBTFUL / QUESTIONABLE,
    unchanged for HEALTHY (and whenever the overlay is empty).
    """
    row = dict(player)
    if not has_status_data():
        return row
    flag = status or live_status(player)
    if flag == HEALTHY:
        return row
    for field in fields:
        row[field] = adjust_projection_for_status(row.get(field) or 0.0, flag)
    row["status"] = flag
    return row


def team_scoring_penalty(
    roster_by_team: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    *,
    factor: float = 1.0,
) -> dict[str, float]:
    """Per-team expected-points penalty from unavailable offensive starters,
    keyed by upper-cased team code. ``{}`` when the overlay is empty.

    Feed the result to :func:`betting.moneyline_model.evaluate_game`'s
    ``team_status_penalty``. Magnitudes come from :data:`POSITION_OUT_POINTS`
    -- a documented flat assumption, not a per-player estimate.
    """
    if not has_status_data() or not roster_by_team:
        return {}
    penalties: dict[str, float] = {}
    for team, players in roster_by_team.items():
        total = 0.0
        for player in players or []:
            flag = live_status(player)
            share = _PENALTY_SHARE.get(flag, 0.0)
            if not share:
                continue
            base = POSITION_OUT_POINTS.get(str(player.get("position", "")).strip().upper(), 0.0)
            total += base * share
        total *= float(factor)
        if total:
            penalties[str(team).strip().upper()] = round(total, 2)
    return penalties
