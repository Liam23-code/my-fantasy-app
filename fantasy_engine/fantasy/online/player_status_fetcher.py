"""player_status_fetcher: refresh ``fantasy_engine/data/player_status.json``.

The one online touch-point for player availability, modelled exactly on
:mod:`fantasy.draft_ingestion.ffcalculator` -- stdlib :mod:`urllib` only (no
``requests``/``httpx`` dependency), ``None`` on any failure, and its tests mock
the network. Nothing imports this at load time and no engine calls it; it runs
only when a user clicks **Refresh Player Status**.

Source: Sleeper's public NFL players dump
(``https://api.sleeper.app/v1/players/nfl``), no API key. Each record carries
``gsis_id`` -- the same id the projection pool uses -- and ``injury_status``;
only the not-HEALTHY rows are kept, so the written file stays small.

Multi-sport note: this is an NFL-only overlay. NBA / MLB / NHL keep their own
local injury loaders; there is no free keyless live-status feed for them.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fantasy.player_status import (
    HEALTHY,
    STATUS_PATH,
    _distinct_players,
    clear_cache,
    normalize_status,
)
from fantasy.utils import normalize_player_name

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SOURCE = "sleeper"
_TIMEOUT_SECONDS = 20.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _http_get_json(url: str, timeout: float = _TIMEOUT_SECONDS) -> Any | None:
    """GET ``url`` and parse JSON. ``None`` on any network / decode failure."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "UniversalQuantAgent/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _is_current(record: dict[str, Any]) -> bool:
    """A player worth carrying: on an NFL team right now, or explicitly active.

    Sleeper's dump is every player it has ever seen (~11k, mostly retired). A
    retired player has ``team=None`` and ``active=False``; a genuinely injured
    one keeps his team. This gate is what keeps the written file to the few
    dozen names that actually matter instead of thousands of stale rows.
    """
    return bool(record.get("team")) or record.get("active") is True


def normalize_feed(payload: Any, *, now: str | None = None) -> dict[str, dict[str, Any]]:
    """Sleeper players dump -> the unified status map. ``{}`` on junk input.

    Keeps only currently-rostered players carrying a real ``injury_status``
    designation (the roster-membership ``status`` field -- "Inactive",
    "Practice Squad" -- is deliberately *not* treated as an injury). Keyed by
    ``gsis_id`` when present, always also by ``nm:<normalized name>`` as a
    fallback join key.
    """
    if not isinstance(payload, dict):
        return {}
    stamp = now or _utc_now()
    result: dict[str, dict[str, Any]] = {}
    for record in payload.values():
        if not isinstance(record, dict) or not _is_current(record):
            continue
        status = normalize_status(record.get("injury_status"))
        if status == HEALTHY:
            continue
        name = (record.get("full_name") or " ".join(
            str(part) for part in (record.get("first_name"), record.get("last_name")) if part
        )).strip()
        entry = {"status": status, "last_updated": stamp, "source": SOURCE, "name": name}
        gsis = str(record.get("gsis_id") or "").strip()
        if gsis:
            result[gsis] = dict(entry)
        name_key = normalize_player_name(name)
        if name_key:
            result.setdefault(f"nm:{name_key}", dict(entry))
    return result


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n",
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent, delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(serialized)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def ensure_status_file(path: Path = STATUS_PATH) -> Path:
    """Create an empty ``{}`` status file if it is missing. Returns the path."""
    if not path.exists():
        _atomic_write_json(path, {})
        clear_cache()
    return path


def load_existing(path: Path = STATUS_PATH) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _player_count(mapping: dict[str, Any]) -> int:
    return _distinct_players(mapping)


def _newest_stamp(mapping: dict[str, Any]) -> str | None:
    stamps = [
        str(rec.get("last_updated"))
        for rec in mapping.values()
        if isinstance(rec, dict) and rec.get("last_updated")
    ]
    return max(stamps) if stamps else None


def refresh_player_status(
    path: Path = STATUS_PATH,
    url: str = SLEEPER_PLAYERS_URL,
    *,
    fetch: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Fetch the feed and rewrite ``path``. On **any** failure the existing
    file is left untouched.

    Returns a small result dict for the UI::

        {"ok": bool, "written": bool, "count": int,
         "last_updated": str | None, "source": str, "error": str | None}

    ``fetch`` is an injection seam for tests; production uses
    :func:`_http_get_json` (which routes through ``urllib.request.urlopen``,
    the same call the draft-ingestion tests monkeypatch).
    """
    ensure_status_file(path)
    getter = fetch or _http_get_json
    payload = getter(url)
    if payload is None:
        existing = load_existing(path)
        return {
            "ok": False,
            "written": False,
            "count": _player_count(existing),
            "last_updated": _newest_stamp(existing),
            "source": SOURCE,
            "error": "fetch failed -- kept the existing status file",
        }
    mapping = normalize_feed(payload)
    _atomic_write_json(path, mapping)
    clear_cache()
    return {
        "ok": True,
        "written": True,
        "count": _player_count(mapping),
        "last_updated": _newest_stamp(mapping),
        "source": SOURCE,
        "error": None,
    }
