"""multi_sport_status: the NFL player-status overlay, widened to MLB/NHL/NBA/CFB/CBB.

A small, read-only layer over ``fantasy_engine/data/multi_sport_status.json`` --
the file the five :mod:`fantasy.online.player_status_fetcher_<sport>` modules
write from a public feed. Every engine stays offline: it reads this local JSON
and nothing else.

This is the exact sibling of :mod:`fantasy.player_status` (which owns NFL and is
left completely untouched). The two never share a file or a cache; the canonical
status vocabulary and the projection / score multipliers are *imported* from
:mod:`fantasy.player_status` rather than re-defined, so the availability maths is
provably identical across every sport.

File shape -- one sub-map per sport, each keyed by the same join keys the sport's
prop rows already carry (a normalized-name key always, an ``id:`` key when the
feed gives an athlete id)::

    {
      "mlb": {
        "nm:zacgallen": {"status": "OUT", "last_updated": "2026-09-05T18:00:00Z",
                          "source": "espn", "name": "Zac Gallen", "team": "ARI"},
        "id:39910":     {"status": "OUT", "last_updated": "...", "source": "espn", ...}
      },
      "nhl": { ... }
    }

Every function takes a ``sport`` string ("mlb" / "nhl" / "nba" / "cfb" / "cbb")
and is a **total no-op until that sport's sub-map holds at least one record** --
:func:`has_status_data` gates every engine integration, so the shipped (absent)
file changes nothing and leaves every existing test green.

``STATUS_PATH`` is resolved at call time, so a test can point the whole module at
a temp file with
``monkeypatch.setattr(multi_sport_status, "STATUS_PATH", tmp); multi_sport_status.clear_cache()``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fantasy.player_status import (
    CANONICAL_STATUSES,
    DOUBTFUL,
    HEALTHY,
    HOLDOUT,
    OUT,
    QUESTIONABLE,
    SUSPENDED,
    UNAVAILABLE_STATUSES,
    adjust_projection_for_status,
    adjust_score_for_status,
    normalize_status,
)
from fantasy.utils import normalize_player_name

__all__ = [
    "CANONICAL_STATUSES",
    "DOUBTFUL",
    "HEALTHY",
    "HOLDOUT",
    "OUT",
    "QUESTIONABLE",
    "SPORTS",
    "STATUS_PATH",
    "SUSPENDED",
    "UNAVAILABLE_STATUSES",
    "adjust_projection_for_status",
    "adjust_score_for_status",
    "availability_flag",
    "clear_cache",
    "effective_status",
    "flagged_count",
    "has_status_data",
    "is_holdout",
    "is_out",
    "is_suspended",
    "is_unavailable",
    "live_status",
    "load_multi_sport_status",
    "normalize_status",
    "status_last_updated",
    "team_status_penalty",
]

#: Where the fetchers write and every non-NFL engine reads. Deliberately *not*
#: ``player_status.json`` -- NFL keeps its own file untouched.
STATUS_PATH = Path(__file__).resolve().parent.parent / "data" / "multi_sport_status.json"

#: The five sports this overlay covers. NFL is handled by
#: :mod:`fantasy.player_status`; it is intentionally absent here.
SPORTS: frozenset[str] = frozenset({"mlb", "nhl", "nba", "cfb", "cbb"})

# mtime-keyed cache, keyed by resolved path -- the Refresh button rewrites the
# file and the next read picks it up. Separate dict from player_status' cache.
_cache: dict[str, tuple[float, dict[str, dict[str, dict[str, Any]]]]] = {}


def _resolve(path: Path | str | None) -> Path:
    return Path(path) if path is not None else STATUS_PATH


def _normalize_sport(sport: Any) -> str:
    return str(sport or "").strip().lower()


def _load(path: Path | str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """The whole ``{sport: {key: record}}`` map. ``{}`` when missing / unreadable."""
    resolved = _resolve(path)
    key = str(resolved)
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        _cache.pop(key, None)
        return {}
    cached = _cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _cache[key] = (mtime, {})
        return {}
    data: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(raw, dict):
        for sport, sub in raw.items():
            if not isinstance(sub, Mapping):
                continue
            data[_normalize_sport(sport)] = {str(k): dict(v) for k, v in sub.items() if isinstance(v, Mapping)}
    _cache[key] = (mtime, data)
    return data


def _sport_map(sport: str, path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    return _load(path).get(_normalize_sport(sport), {})


def clear_cache() -> None:
    """Drop the in-process cache (tests; also safe after a manual file edit)."""
    _cache.clear()


def load_multi_sport_status(path: Path | str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """The raw ``{sport: {key: record}}`` map. ``{}`` when the file is absent."""
    return {sport: dict(sub) for sport, sub in _load(path).items()}


def has_status_data(sport: Any = None, path: Path | str | None = None) -> bool:
    """True once the live file holds a record -- for ``sport`` when given, for any
    sport otherwise. Every engine's status handling is a no-op until this is
    true, so the shipped (absent) file leaves behaviour and tests unchanged."""
    data = _load(path)
    if sport is None:
        return any(data.values())
    return bool(data.get(_normalize_sport(sport)))


def status_last_updated(sport: Any = None, path: Path | str | None = None) -> str | None:
    """The newest ``last_updated`` stamp -- for ``sport`` when given, else across
    every sport. ``None`` when there is nothing recorded."""
    data = _load(path)
    sub_maps = [data.get(_normalize_sport(sport), {})] if sport is not None else list(data.values())
    stamps = [str(rec.get("last_updated")) for sub in sub_maps for rec in sub.values() if isinstance(rec, Mapping) and rec.get("last_updated")]
    return max(stamps) if stamps else None


def _distinct_players(sub: Mapping[str, Any]) -> int:
    seen: set[str] = set()
    keyless = 0
    for rec in sub.values():
        name = rec.get("name") if isinstance(rec, Mapping) else None
        if name:
            seen.add(str(name).strip().casefold())
        else:
            keyless += 1
    return len(seen) + keyless if seen else len(sub)


def flagged_count(sport: Any = None, path: Path | str | None = None) -> int:
    """How many distinct players carry a non-HEALTHY flag -- for ``sport`` when
    given, else summed across every sport. Records written by the fetchers carry
    a ``name``, so an id key and its ``nm:`` fallback collapse to one."""
    data = _load(path)
    sub_maps = [data.get(_normalize_sport(sport), {})] if sport is not None else list(data.values())
    return sum(_distinct_players(sub) for sub in sub_maps)


def _keys_for(player: Mapping[str, Any] | str) -> tuple[str, ...]:
    """The lookup keys for a player row or a bare id / name string, in priority
    order: the explicit id first, then the normalized-name fallback."""
    if isinstance(player, str):
        text = player.strip()
        keys: list[str] = []
        if text:
            keys.append(f"id:{text}")
        name_key = normalize_player_name(text)
        if name_key:
            keys.append(f"nm:{name_key}")
        return tuple(keys)
    identifier = str(player.get("player_id") or player.get("athlete_id") or player.get("id") or "").strip()
    name_key = normalize_player_name(player.get("name") or player.get("player_name") or player.get("player") or player.get("full_name"))
    keys = []
    if identifier:
        keys.append(f"id:{identifier}")
    if name_key:
        keys.append(f"nm:{name_key}")
    return tuple(keys)


def _live_status(sport: str, player: Mapping[str, Any] | str, path: Path | str | None = None) -> str | None:
    sub = _sport_map(sport, path)
    if not sub:
        return None
    for key in _keys_for(player):
        rec = sub.get(key)
        if isinstance(rec, Mapping) and rec.get("status"):
            return normalize_status(rec["status"])
    return None


def live_status(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> str:
    """The player's status **from the live overlay only** -- HEALTHY when the
    feed says nothing about them (or the sport has no data yet).

    This is what every hard rule keys on (filter / zero / badge), so the overlay
    is purely additive: a player the feed does not mention is treated exactly as
    before this feature.
    """
    return _live_status(_normalize_sport(sport), player, path) or HEALTHY


def effective_status(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> str:
    """Best available knowledge: the live feed wins, else the row's own
    ``injury_status`` / ``status`` field, else HEALTHY. Not used for filtering --
    see :func:`live_status` -- but handy for a "what do we know" badge."""
    live = _live_status(_normalize_sport(sport), player, path)
    if live is not None:
        return live
    if isinstance(player, Mapping):
        return normalize_status(player.get("injury_status") or player.get("status"))
    return HEALTHY


def availability_flag(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> str:
    """The value a UI badge shows -- the live-overlay status only."""
    return live_status(sport, player, path)


def is_out(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> bool:
    return live_status(sport, player, path) == OUT


def is_holdout(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> bool:
    return live_status(sport, player, path) == HOLDOUT


def is_suspended(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> bool:
    return live_status(sport, player, path) == SUSPENDED


def is_unavailable(sport: Any, player: Mapping[str, Any] | str, path: Path | str | None = None) -> bool:
    """OUT, HOLDOUT or SUSPENDED **per the live overlay** -- cannot play."""
    return live_status(sport, player, path) in UNAVAILABLE_STATUSES


#: How much of a flat, caller-supplied per-team points penalty each status
#: applies -- the same shares :mod:`betting.player_status_utils` uses for NFL, so
#: an unavailable key player moves a team total by the same fraction in every
#: sport. Purely opt-in: it only ever runs when a caller passes a roster.
_PENALTY_SHARE: dict[str, float] = {
    OUT: 1.0,
    HOLDOUT: 1.0,
    SUSPENDED: 1.0,
    DOUBTFUL: 0.5,
    QUESTIONABLE: 0.15,
    HEALTHY: 0.0,
}


def team_status_penalty(
    sport: Any,
    roster_by_team: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    *,
    points_by_position: Mapping[str, float] | None = None,
    default_points: float = 0.0,
    factor: float = 1.0,
    path: Path | str | None = None,
) -> dict[str, float]:
    """Per-team expected-points penalty from unavailable players, keyed by
    upper-cased team code. ``{}`` whenever the overlay is empty for ``sport`` or
    no roster is supplied -- so the moneyline models stay byte-identical until a
    caller opts in by handing this map to ``team_status_penalty=``.

    ``points_by_position`` maps an upper-cased position code to the flat points a
    healthy starter there is worth; a position not in the map (or a sport with no
    map) uses ``default_points``. This is a documented modelling hook, not a
    per-player estimate.
    """
    if not has_status_data(sport, path) or not roster_by_team:
        return {}
    position_points = {str(k).strip().upper(): float(v) for k, v in (points_by_position or {}).items()}
    penalties: dict[str, float] = {}
    for team, players in roster_by_team.items():
        total = 0.0
        for player in players or []:
            share = _PENALTY_SHARE.get(live_status(sport, player, path), 0.0)
            if not share:
                continue
            base = position_points.get(str(player.get("position", "")).strip().upper(), default_points)
            total += base * share
        total *= float(factor)
        if total:
            penalties[str(team).strip().upper()] = round(total, 2)
    return penalties
