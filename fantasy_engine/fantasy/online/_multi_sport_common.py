"""Shared plumbing for the five ``player_status_fetcher_<sport>`` modules.

Exactly the pattern :mod:`fantasy.online.player_status_fetcher` uses for NFL --
stdlib :mod:`urllib` only (no ``requests`` / ``httpx``), ``None`` on any failure,
an atomic write, and a ``fetch=`` injection seam its tests use so the network is
never touched. Factored out here so MLB / NHL / NBA / CFB / CBB do not carry five
copies of the same forty lines; each sport module is then just its feed URL, its
sport code, and a one-line ``refresh_player_status`` wrapper.

Source: ESPN's public, keyless per-league injuries JSON
(``https://site.api.espn.com/apis/site/v2/sports/<sport>/<league>/injuries``).
It can be sparse or empty in the off-season, and ESPN may rate-limit or reject a
request outright -- every one of those cases degrades to "keep the file exactly
as it was", so a bad pull can never make an engine worse than no overlay at all.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fantasy.multi_sport_status import STATUS_PATH, clear_cache
from fantasy.player_status import HEALTHY, normalize_status
from fantasy.utils import normalize_player_name

SOURCE = "espn"
_TIMEOUT_SECONDS = 20.0
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

#: ``<sport code> -> (espn sport, espn league)`` for the injuries endpoint.
ESPN_LEAGUE_PATHS: dict[str, tuple[str, str]] = {
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "nba": ("basketball", "nba"),
    "cfb": ("football", "college-football"),
    "cbb": ("basketball", "mens-college-basketball"),
}

#: ESPN spells a handful of statuses in ways :func:`normalize_status` does not
#: know verbatim; fold them onto a token it does before the generic pass. An
#: injured-list stint of any length means the player cannot play now.
_ESPN_STATUS_ALIASES: dict[str, str] = {
    "day-to-day": "day-to-day",
    "day to day": "day-to-day",
    "injured reserve": "ir",
    "injured-reserve": "ir",
    "il": "ir",
    "60-day-il": "ir",
    "15-day-il": "ir",
    "10-day-il": "ir",
    "7-day-il": "ir",
    "60 day il": "ir",
    "15 day il": "ir",
    "10 day il": "ir",
    "7 day il": "ir",
    "long-term-injured-reserve": "ir",
}

_ID_IN_HREF = re.compile(r"/id/(\d+)")


def _as_mapping(value: Any) -> dict[str, Any]:
    """``value`` as a plain ``dict`` when it is a mapping, else ``{}`` -- one
    call site for the "narrow this ``Any`` to something with ``.get``" pattern."""
    return dict(value) if isinstance(value, Mapping) else {}


def espn_injuries_url(sport: str) -> str:
    espn_sport, espn_league = ESPN_LEAGUE_PATHS[sport]
    return f"{_ESPN_BASE}/{espn_sport}/{espn_league}/injuries"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def http_get_json(url: str, timeout: float = _TIMEOUT_SECONDS) -> Any | None:
    """GET ``url`` and parse JSON. ``None`` on any network / decode failure."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "UniversalQuantAgent/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _canonical_status(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    return normalize_status(_ESPN_STATUS_ALIASES.get(text, text))


def _athlete_id(athlete: Mapping[str, Any]) -> str:
    identifier = str(athlete.get("id") or athlete.get("uid") or "").strip()
    if identifier.isdigit():
        return identifier
    for link in athlete.get("links") or []:
        if isinstance(link, Mapping):
            match = _ID_IN_HREF.search(str(link.get("href") or ""))
            if match:
                return match.group(1)
    return ""


def _athlete_name(athlete: Mapping[str, Any]) -> str:
    name = athlete.get("displayName") or athlete.get("fullName")
    if not name:
        name = " ".join(str(part) for part in (athlete.get("firstName"), athlete.get("lastName")) if part)
    return str(name or "").strip()


def _record(name: str, team: str, status: str, stamp: str) -> dict[str, Any]:
    return {"status": status, "last_updated": stamp, "source": SOURCE, "name": name, "team": team}


def _add_row(
    result: dict[str, dict[str, Any]],
    *,
    identifier: str,
    name: str,
    team: str,
    status: str,
    stamp: str,
) -> None:
    if status == HEALTHY:
        return
    entry = _record(name, team, status, stamp)
    if identifier:
        result[f"id:{identifier}"] = dict(entry)
    name_key = normalize_player_name(name)
    if name_key:
        result.setdefault(f"nm:{name_key}", dict(entry))


def normalize_espn_injuries(payload: Any, *, now: str | None = None) -> dict[str, dict[str, Any]]:
    """ESPN injuries JSON -> the unified per-sport status map. ``{}`` on junk.

    Accepts the documented team-grouped shape
    (``{"injuries": [{"displayName": <team>, "injuries": [{"status", "athlete"}]}]}``)
    and, for robustness / uploads, a flat list of ``{athlete|player|name, status,
    team}`` rows. Only the not-HEALTHY rows are kept, so the written file stays
    small. Keyed by ``id:<athlete id>`` when the feed carries one, always also by
    ``nm:<normalized name>``.
    """
    stamp = now or utc_now()
    result: dict[str, dict[str, Any]] = {}

    groups: Any
    if isinstance(payload, Mapping):
        groups = payload.get("injuries")
    elif isinstance(payload, list):
        groups = payload
    else:
        return {}
    if not isinstance(groups, list):
        return {}

    for group in groups:
        if not isinstance(group, Mapping):
            continue
        # Team-grouped node: {"displayName": <team>, "injuries": [ ... ]}.
        inner = group.get("injuries")
        if isinstance(inner, list):
            team = str(group.get("abbreviation") or group.get("displayName") or "").strip()
            for item in inner:
                if not isinstance(item, Mapping):
                    continue
                athlete = _as_mapping(item.get("athlete"))
                athlete_team = _as_mapping(athlete.get("team"))
                item_type = _as_mapping(item.get("type"))
                _add_row(
                    result,
                    identifier=_athlete_id(athlete),
                    name=_athlete_name(athlete) or str(item.get("displayName") or "").strip(),
                    team=str(athlete_team.get("abbreviation") or "").strip() or team,
                    # ESPN always sends the string ``status``; ``type`` is a
                    # nested object, so read its ``description`` if ``status``
                    # is ever missing rather than stringifying the whole dict.
                    status=_canonical_status(
                        item.get("status") or item_type.get("description") or item_type.get("abbreviation")
                    ),
                    stamp=stamp,
                )
            continue
        # Flat row: {"athlete"/"player"/"name", "status"/"injury_status", "team"}.
        athlete = _as_mapping(group.get("athlete"))
        name = _athlete_name(athlete) if athlete else ""
        name = name or str(group.get("player_name") or group.get("player") or group.get("name") or "").strip()
        if not name:
            continue
        flat_id = str(group.get("athlete_id") or group.get("player_id") or group.get("id") or "").strip()
        _add_row(
            result,
            identifier=_athlete_id(athlete) or flat_id,
            name=name,
            team=str(group.get("team") or group.get("team_abbreviation") or "").strip(),
            status=_canonical_status(group.get("status") or group.get("injury_status")),
            stamp=stamp,
        )
    return result


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(serialized)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def load_existing(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def ensure_status_file(path: Path = STATUS_PATH) -> Path:
    """Create an empty ``{}`` status file if it is missing. Returns the path.

    Uses an ``O_EXCL`` create so two sport refreshes starting at the same moment
    can't both try to seed the file (on Windows the loser's ``os.replace`` would
    raise). Whoever loses the create just sees the file already there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_BINARY (no-op on POSIX) keeps the seed bytes exactly b"{}\n" on
        # Windows rather than letting the CRT translate the LF to CRLF -- the
        # same LF discipline the rest of this module writes with.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
    except FileExistsError:
        return path
    try:
        os.write(fd, b"{}\n")
    finally:
        os.close(fd)
    clear_cache()
    return path


#: Best-effort lock timings for the read-merge-write of one sport's slice.
_LOCK_POLL_SECONDS = 0.05
_LOCK_MAX_WAIT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 15.0


@contextlib.contextmanager
def _slice_write_lock(target: Path) -> Iterator[None]:
    """Serialize the read-merge-write of one sport's slice across processes, so
    two concurrent per-sport refreshes (two Streamlit sessions clicking Refresh
    on different sport pages) cannot lose each other's write.

    A sibling ``<file>.lock`` created with ``O_EXCL`` is the mutex. The wait is
    bounded: a lock older than :data:`_LOCK_STALE_SECONDS` (a writer that
    crashed mid-refresh) is stolen, and if the lock still cannot be taken within
    :data:`_LOCK_MAX_WAIT_SECONDS` the caller proceeds anyway -- a rare lost
    update is less bad than a Refresh button that never returns. The NFL fetcher
    needs none of this because it owns its whole file.

    Deliberately best-effort, not a strict mutex:
    * The stale-lock steal (unlink-then-recreate) is itself racy -- two waiters
      can both remove one abandoned lock -- but ``_LOCK_MAX_WAIT_SECONDS`` <
      ``_LOCK_STALE_SECONDS`` means a lock held by a *live* writer is never
      stealable, so this only bites against a crashed-writer orphan plus a
      concurrent refresh, and the loser just does a last-writer-wins slice
      update that the next Refresh repairs.
    * Cleanup in ``finally`` is wrapped so a transient FS error on the tiny lock
      file (an antivirus / sync-client handle) can never propagate out of a
      ``with`` body whose write already succeeded.
    """
    lock_path = target.with_name(target.name + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + _LOCK_MAX_WAIT_SECONDS
    while fd is None and time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > _LOCK_STALE_SECONDS:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            time.sleep(_LOCK_POLL_SECONDS)
        except OSError:
            break  # cannot create a lock here at all -- proceed unlocked
    try:
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                lock_path.unlink(missing_ok=True)


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


def _newest_stamp(sub: Mapping[str, Any]) -> str | None:
    stamps = [str(rec.get("last_updated")) for rec in sub.values() if isinstance(rec, Mapping) and rec.get("last_updated")]
    return max(stamps) if stamps else None


def refresh_sport_status(
    sport: str,
    path: Path = STATUS_PATH,
    url: str | None = None,
    *,
    fetch: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Fetch ``sport``'s feed and rewrite **only that sport's slice** of
    ``path``. On any failure the file is left exactly as it was.

    Returns a small result dict for the UI::

        {"ok": bool, "written": bool, "sport": str, "count": int,
         "last_updated": str | None, "source": str, "error": str | None}
    """
    sport = sport.strip().lower()
    target_url = url or espn_injuries_url(sport)
    ensure_status_file(path)
    getter = fetch or http_get_json
    payload = getter(target_url)
    if payload is None:
        existing_sub = _as_mapping(load_existing(path).get(sport))
        return {
            "ok": False,
            "written": False,
            "sport": sport,
            "count": _distinct_players(existing_sub),
            "last_updated": _newest_stamp(existing_sub),
            "source": SOURCE,
            "error": "fetch failed -- kept the existing status file",
        }
    sub_map = normalize_espn_injuries(payload)
    # Lock only the read-merge-write, not the (slow) network fetch above: every
    # other sport's slice is carried forward from a read taken inside the lock,
    # so a concurrent refresh of a different sport can't be silently reverted.
    with _slice_write_lock(path):
        merged = {k: v for k, v in load_existing(path).items() if k != sport}
        merged[sport] = sub_map
        _atomic_write_json(path, merged)
    clear_cache()
    return {
        "ok": True,
        "written": True,
        "sport": sport,
        "count": _distinct_players(sub_map),
        "last_updated": _newest_stamp(sub_map),
        "source": SOURCE,
        "error": None,
    }
