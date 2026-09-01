"""player_status: overlay live player availability onto the offline pool.

A small, read-only layer over ``fantasy_engine/data/player_status.json`` -- the
file :mod:`fantasy.online.player_status_fetcher` writes from a public feed.
Every engine stays offline: it reads this local JSON and nothing else.

The file is a flat map keyed by the same player id the projection pool uses
(nflverse GSIS, e.g. ``"00-0039075"``), with a normalized-name fallback key
(``"nm:<normalized>"``) for feed rows that carry no GSIS id::

    {
      "00-0039075": {"status": "OUT", "last_updated": "2026-08-31T18:00:00Z", "source": "sleeper"},
      "nm:someguy": {"status": "HOLDOUT", "last_updated": "...", "source": "sleeper"}
    }

``status`` is one of OUT / DOUBTFUL / QUESTIONABLE / HOLDOUT / SUSPENDED /
HEALTHY. When a player is absent from the file, :func:`effective_status` falls
back to whatever ``injury_status`` the pool row already carries -- so an empty
status file (the shipped default) changes nothing, and every engine's existing
injury handling keeps working unchanged.

Every ``path`` argument defaults to :data:`STATUS_PATH`, resolved at call time,
so a test can point the whole module at a temp file with
``monkeypatch.setattr(player_status, "STATUS_PATH", tmp); player_status.clear_cache()``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fantasy.utils import normalize_player_name, safe_float

#: Where the fetcher writes and every engine reads.
STATUS_PATH = Path(__file__).resolve().parents[1] / "data" / "player_status.json"

#: The six canonical values. HEALTHY is the "nothing wrong" sentinel.
OUT = "OUT"
DOUBTFUL = "DOUBTFUL"
QUESTIONABLE = "QUESTIONABLE"
HOLDOUT = "HOLDOUT"
SUSPENDED = "SUSPENDED"
HEALTHY = "HEALTHY"

CANONICAL_STATUSES: frozenset[str] = frozenset(
    {OUT, DOUBTFUL, QUESTIONABLE, HOLDOUT, SUSPENDED, HEALTHY}
)

#: Cannot be started this week -- removed from rooms, boards and slates.
UNAVAILABLE_STATUSES: frozenset[str] = frozenset({OUT, HOLDOUT, SUSPENDED})

#: Raw feed / pool strings -> canonical. Anything unmapped is HEALTHY.
_ALIASES: dict[str, str] = {
    "out": OUT, "o": OUT, "ir": OUT, "ir-r": OUT, "inactive": OUT, "pup": OUT,
    "nfi": OUT, "nfi-r": OUT, "did not play": OUT, "dnp": OUT, "covid": OUT, "cel": OUT,
    "doubtful": DOUBTFUL, "d": DOUBTFUL,
    "questionable": QUESTIONABLE, "q": QUESTIONABLE, "gtd": QUESTIONABLE,
    "day-to-day": QUESTIONABLE, "day to day": QUESTIONABLE, "probable": QUESTIONABLE,
    "holdout": HOLDOUT, "hold out": HOLDOUT, "hold-in": HOLDOUT, "holdin": HOLDOUT,
    "contract dispute": HOLDOUT,
    "sus": SUSPENDED, "susp": SUSPENDED, "suspended": SUSPENDED, "suspension": SUSPENDED,
    "healthy": HEALTHY, "active": HEALTHY, "": HEALTHY, "none": HEALTHY, "na": HEALTHY,
    "null": HEALTHY,
}

#: Projection multiplier by status. Cannot-play statuses zero it.
_PROJECTION_FACTOR: dict[str, float] = {
    OUT: 0.0, HOLDOUT: 0.0, SUSPENDED: 0.0,
    DOUBTFUL: 0.45, QUESTIONABLE: 0.88, HEALTHY: 1.0,
}

#: Recommendation-score multiplier by status. A touch harsher than the
#: projection factor for HOLDOUT/SUSPENDED so they sink even on a thin board.
_SCORE_FACTOR: dict[str, float] = {
    OUT: 0.0, HOLDOUT: 0.05, SUSPENDED: 0.05,
    DOUBTFUL: 0.6, QUESTIONABLE: 0.85, HEALTHY: 1.0,
}

# mtime-keyed cache: the Refresh button rewrites the file, and the next read
# picks it up automatically. Keyed by resolved path so a test's temp file and
# the real file never collide.
_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def _resolve(path: Path | str | None) -> Path:
    return Path(path) if path is not None else STATUS_PATH


def normalize_status(value: Any) -> str:
    """Coerce any raw status string onto the six canonical values."""
    text = str(value or "").strip().lower()
    if text in _ALIASES:
        return _ALIASES[text]
    upper = text.upper()
    if upper in CANONICAL_STATUSES:
        return upper
    return HEALTHY


def _load(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
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
    data = (
        {str(k): v for k, v in raw.items() if isinstance(v, Mapping)}
        if isinstance(raw, dict)
        else {}
    )
    _cache[key] = (mtime, data)
    return data


def clear_cache() -> None:
    """Drop the in-process cache (tests; also safe after a manual file edit)."""
    _cache.clear()


def has_status_data(path: Path | str | None = None) -> bool:
    """True once the live file holds at least one record. Every engine's status
    handling is a no-op until this is true, so the shipped empty file leaves
    behaviour and tests unchanged."""
    return bool(_load(path))


def load_player_status(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """The raw status map. ``{}`` when the file is missing or unreadable."""
    return dict(_load(path))


def status_last_updated(path: Path | str | None = None) -> str | None:
    """The newest ``last_updated`` stamp in the file, or ``None`` when empty."""
    stamps = [
        str(rec.get("last_updated"))
        for rec in _load(path).values()
        if isinstance(rec, Mapping) and rec.get("last_updated")
    ]
    return max(stamps) if stamps else None


def flagged_count(path: Path | str | None = None) -> int:
    """How many distinct players carry a non-HEALTHY flag. Records written by
    the fetcher carry a ``name``, so the id key and its ``nm:`` fallback key
    collapse to one; hand-written files without names fall back to key count."""
    data = _load(path)
    return _distinct_players(data)


def _distinct_players(data: Mapping[str, Any]) -> int:
    seen: set[str] = set()
    keyless = 0
    for rec in data.values():
        name = rec.get("name") if isinstance(rec, Mapping) else None
        if name:
            seen.add(str(name).strip().casefold())
        else:
            keyless += 1
    return len(seen) + keyless if seen else len(data)


def _keys_for(player: Mapping[str, Any]) -> tuple[str, ...]:
    pid = str(player.get("player_id") or player.get("id") or "").strip()
    name_key = normalize_player_name(player.get("name") or player.get("player_name"))
    keys: list[str] = []
    if pid:
        keys.append(pid)
    if name_key:
        keys.append(f"nm:{name_key}")
    return tuple(keys)


def _live_status(player: Mapping[str, Any], path: Path | str | None = None) -> str | None:
    data = _load(path)
    if not data:
        return None
    for key in _keys_for(player):
        rec = data.get(key)
        if isinstance(rec, Mapping) and rec.get("status"):
            return normalize_status(rec["status"])
    return None


def live_status(player: Mapping[str, Any], path: Path | str | None = None) -> str:
    """The player's status **from the live overlay only** -- HEALTHY when the
    feed says nothing about them.

    This is what every hard rule keys on (filter / zero / badge), so the live
    overlay is purely additive: a player the feed does not mention is treated
    exactly as before this feature, and the projection pool's own -- possibly
    stale, offseason -- ``injury_status`` is never made more aggressive than
    the multiplier it already drove.
    """
    return _live_status(player, path) or HEALTHY


def effective_status(player: Mapping[str, Any], path: Path | str | None = None) -> str:
    """Best available knowledge: the live feed wins, else the pool row's own
    ``injury_status``, else HEALTHY. Not used for filtering -- see
    :func:`live_status` -- but handy for a "what do we know" read."""
    live = _live_status(player, path)
    if live is not None:
        return live
    return normalize_status(player.get("injury_status"))


def availability_flag(player: Mapping[str, Any], path: Path | str | None = None) -> str:
    """The value a UI badge shows -- the live-overlay status only."""
    return live_status(player, path)


def is_out(player: Mapping[str, Any], path: Path | str | None = None) -> bool:
    return live_status(player, path) == OUT


def is_holdout(player: Mapping[str, Any], path: Path | str | None = None) -> bool:
    return live_status(player, path) == HOLDOUT


def is_suspended(player: Mapping[str, Any], path: Path | str | None = None) -> bool:
    return live_status(player, path) == SUSPENDED


def is_unavailable(player: Mapping[str, Any], path: Path | str | None = None) -> bool:
    """OUT, HOLDOUT or SUSPENDED **per the live overlay** -- cannot be started."""
    return live_status(player, path) in UNAVAILABLE_STATUSES


def adjust_projection_for_status(projection: Any, status: str) -> float:
    """Scale a projection by availability. OUT / HOLDOUT / SUSPENDED -> 0."""
    return round(safe_float(projection) * _PROJECTION_FACTOR.get(normalize_status(status), 1.0), 4)


def adjust_score_for_status(score: Any, status: str) -> float:
    """Scale a recommendation score by availability, **sign-safely**: a
    downweight always makes the score rank worse, on either side of zero
    (recommendation scores are routinely negative in the mid/late rounds)."""
    factor = _SCORE_FACTOR.get(normalize_status(status), 1.0)
    value = safe_float(score)
    if factor <= 0.0:
        return 0.0
    if factor >= 1.0 or value == 0.0:
        return round(value * factor, 4)
    return round(value * factor if value > 0 else value / factor, 4)


def overlay_pool_status(
    players: Iterable[Mapping[str, Any]] | None,
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Shallow-copy ``players`` with ``injury_status`` set to the effective
    status and the projection zeroed for a cannot-play status, so the existing
    injury handling in every engine picks the live status up unchanged.

    A plain ``list(players)`` (no copies) when the status file is empty.
    """
    source = list(players or [])
    if not _load(path):
        return source
    result: list[dict[str, Any]] = []
    for player in source:
        if not isinstance(player, Mapping):
            result.append(player)  # type: ignore[arg-type]
            continue
        status = live_status(player, path)
        if status == HEALTHY:
            result.append(player if isinstance(player, dict) else dict(player))
            continue
        row = dict(player)
        row["injury_status"] = status
        if status in UNAVAILABLE_STATUSES:
            row["projection"] = 0.0
            row["expected_fantasy_points"] = 0.0
        result.append(row)
    return result
