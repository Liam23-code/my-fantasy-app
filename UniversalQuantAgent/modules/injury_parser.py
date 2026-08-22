"""Safe, offline injury-data parsing, normalization, and impact modeling.

Every function here operates on data the caller already has -- a payload
already loaded from ``injuries.json``, a user upload, or (historically) a
live fetch. Nothing in this module makes a network request. The live-fetch
half of the former ``modules/injury_model.py`` is now permanently disabled
in :mod:`modules.injury_scraper_disabled`.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.data_quality import (
    fuzzy_name_match,
    safe_dict,
    safe_get,
    safe_list,
    safe_scalar_to_dict,
)
from modules.sportsbook_parser import normalize_player_name, normalize_team_name

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INJURIES_PATH = DATA_ROOT / "injuries.json"

#: status -> impact score, 100 meaning "cannot play". Snap-share/usage/
#: volatility impact all key off this one severity read, so a caller only
#: has to derive one number instead of three independent guesses.
STATUS_IMPACT_SCORE: dict[str, float] = {
    "OUT": 100.0,
    "QUESTIONABLE": 55.0,
    "PROBABLE": 20.0,
    "ACTIVE": 0.0,
}

#: status -> expected fraction of normal snap share/usage a player who
#: *does* suit up will see. A "questionable" player who plays is usually on
#: a real but reduced workload; "probable" is close to a normal role.
STATUS_SNAP_SHARE_MULTIPLIER: dict[str, float] = {
    "OUT": 0.0,
    "QUESTIONABLE": 0.75,
    "PROBABLE": 0.92,
    "ACTIVE": 1.0,
}

#: status -> added volatility (0-1) on top of a player's normal week-to-week
#: variance -- an uncertain workload is itself a source of variance,
#: independent of how big the workload turns out to be.
STATUS_VOLATILITY_BUMP: dict[str, float] = {
    "OUT": 0.0,  # not playing isn't "volatile", it's a known zero
    "QUESTIONABLE": 0.30,
    "PROBABLE": 0.10,
    "ACTIVE": 0.0,
}


def normalize_status(value: Any) -> str:
    """Collapse provider-specific injury labels into four stable values."""
    if isinstance(value, dict):
        value = safe_get(value, "description", safe_get(value, "value", ""))
    text = str(value or "").upper()
    if any(word in text for word in ("OUT", "SUSPENDED", "DOUBTFUL")):
        return "OUT"
    if any(word in text for word in ("QUESTIONABLE", "GAME-TIME", "DAY-TO-DAY")):
        return "QUESTIONABLE"
    if "PROBABLE" in text:
        return "PROBABLE"
    return "ACTIVE"


def severity_impact_score(status: str) -> float:
    """0-100 impact score for a normalized status, 100 meaning cannot play."""
    return STATUS_IMPACT_SCORE.get(normalize_status(status), 45.0)


def participation_probability(status: str) -> float:
    """0-1 probability the player actually takes the field/court, from severity alone."""
    return 1.0 - severity_impact_score(status) / 100.0


def snap_share_impact(status: str, baseline_snap_share: float) -> float:
    """Expected snap share given a real baseline and this injury status."""
    multiplier = STATUS_SNAP_SHARE_MULTIPLIER.get(normalize_status(status), 0.85)
    return max(0.0, min(1.0, float(baseline_snap_share) * multiplier))


def usage_impact(status: str, baseline_usage_rate: float) -> float:
    """Expected usage rate given a real baseline and this injury status (same multiplier as snap share)."""
    multiplier = STATUS_SNAP_SHARE_MULTIPLIER.get(normalize_status(status), 0.85)
    return max(0.0, min(1.0, float(baseline_usage_rate) * multiplier))


def volatility_impact(status: str, baseline_volatility: float) -> float:
    """Baseline week-to-week volatility (0-1), bumped for status-driven uncertainty."""
    bump = STATUS_VOLATILITY_BUMP.get(normalize_status(status), 0.15)
    return max(0.0, min(1.0, float(baseline_volatility) + bump))


def _text(value: Any, default: str = "") -> str:
    """Convert a provider value to text without exposing container reprs."""
    if isinstance(value, dict):
        value = safe_get(value, "displayName") or safe_get(value, "description") or safe_get(value, "name") or safe_get(value, "value")
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return default
    try:
        return str(value).strip()
    except Exception:
        return default


def _details(node: Any, *, source: str) -> dict[str, Any]:
    """Build the stable details object used by every injury record."""
    payload = safe_dict(safe_get(node, "details"))
    description = (
        _text(safe_get(payload, "description"))
        or _text(safe_get(node, "shortComment"))
        or _text(safe_get(node, "longComment"))
        or _text(safe_get(node, "details"))
    )
    return {"description": description, "source": source, "timestamp": datetime.now(timezone.utc).isoformat()}


def walk_injury_records(node: Any, team: str, output: list[dict[str, Any]], *, source: str = "offline") -> None:
    """Recursively parse an already-obtained, arbitrarily-nested injury payload.

    ``node`` is data the caller already has in memory (loaded from
    ``injuries.json`` or a user upload); this performs no I/O of its own.
    """
    if isinstance(node, list):
        for item in safe_list(node):
            walk_injury_records(item, team, output, source=source)
        return
    if not isinstance(node, dict):
        return

    current_team = team
    if isinstance(safe_get(node, "injuries"), list):
        current_team = _text(safe_get(node, "abbreviation")) or _text(safe_get(node, "displayName"), team)

    athlete = safe_dict(safe_get(node, "athlete"))
    name = _text(safe_get(athlete, "displayName")) or _text(safe_get(node, "playerName")) or _text(safe_get(node, "player"))

    raw_status = safe_get(node, "status") or safe_get(safe_get(node, "type"), "description")
    if raw_status is None:
        raw_status = safe_get(safe_scalar_to_dict(safe_get(node, "type")), "value")

    if name:
        output.append(
            {
                "team": normalize_team_name(current_team or ""),
                "player": normalize_player_name(name),
                "status": normalize_status(raw_status),
                "details": _details(node, source=source),
            }
        )

    for child in node.values():
        if isinstance(child, (dict, list)):
            walk_injury_records(child, current_team, output, source=source)


def clean_injury_record(record: Any) -> dict[str, Any] | None:
    """Validate one result before it crosses the ingestion boundary."""
    row = safe_dict(record)
    player = normalize_player_name(_text(safe_get(row, "player")))
    if not player:
        return None
    details = safe_dict(safe_get(row, "details"))
    return {
        "team": normalize_team_name(_text(safe_get(row, "team"))),
        "player": player,
        "status": normalize_status(safe_get(row, "status")),
        "details": {
            "description": _text(safe_get(details, "description")),
            "source": _text(safe_get(details, "source"), "offline"),
            "timestamp": _text(safe_get(details, "timestamp"), datetime.now(timezone.utc).isoformat()),
        },
    }


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: dict[str, dict[str, Any]] = {}
    for record in records:
        cleaned = clean_injury_record(record)
        if cleaned:
            clean[cleaned["player"].lower()] = cleaned
    return list(clean.values())


def load_injury_data_from_file(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load a real injury report from a local JSON file (default: ``data/injuries.json``).

    A missing file is a normal "no injuries configured" state, not an
    error -- returns ``[]`` rather than raising, matching the offline
    betting engine's odds-loading convention.
    """
    target = Path(path) if path is not None else DEFAULT_INJURIES_PATH
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    records: list[dict[str, Any]] = []
    walk_injury_records(payload, "", records, source="file")
    return _dedupe(records)


def load_injury_data_from_user_upload(source: Any, *, file_format: str | None = None) -> list[dict[str, Any]]:
    """Parse a user-uploaded injury file: JSON, CSV, or delimited text.

    ``source`` may be a path, a file-like object (e.g. a Streamlit
    ``UploadedFile``), or raw ``str``/``bytes`` content. Expected columns
    for CSV/text: ``player, team, status`` (plus an optional ``description``).
    """
    name = ""
    if isinstance(source, (str, Path)) and "\n" not in str(source) and not str(source).strip().startswith(("{", "[")):
        path = Path(source)
        if not path.is_file():
            return []
        name = path.suffix.lower()
        text = path.read_text(encoding="utf-8-sig")
    elif hasattr(source, "read"):
        name = Path(getattr(source, "name", "")).suffix.lower()
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else str(raw)
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    elif isinstance(source, str):
        text = source
    else:
        return []

    fmt = file_format or {".json": "json", ".csv": "csv", ".tsv": "csv", ".txt": "text"}.get(name)
    if fmt is None:
        fmt = "json" if text.strip().startswith(("{", "[")) else "text"

    records: list[dict[str, Any]] = []
    if fmt == "json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        walk_injury_records(payload, "", records, source="upload")
    else:
        delimiter = "\t" if "\t" in text.splitlines()[0] else "|" if "|" in text.splitlines()[0] else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            row = {str(key).strip().lower(): value for key, value in row.items()}
            player = row.get("player") or row.get("name")
            if not player:
                continue
            records.append(
                {
                    "team": row.get("team", ""),
                    "player": player,
                    "status": row.get("status", ""),
                    "details": {"description": row.get("description", ""), "source": "upload"},
                }
            )
    return _dedupe(records)


def get_player_availability(player_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Availability plus an impact score (100 = cannot play) for one player, from already-loaded records.

    Pure lookup -- ``records`` must already be loaded (via
    :func:`load_injury_data_from_file` or
    :func:`load_injury_data_from_user_upload`); this makes no fetch of its
    own.
    """
    target = normalize_player_name(player_name)
    rows = [safe_dict(row) for row in safe_list(records) if isinstance(row, dict)]
    names = {_text(safe_get(row, "player")).lower(): row for row in rows if _text(safe_get(row, "player"))}
    matched_name = fuzzy_name_match(target, list(names), cutoff=0.82)
    if matched_name:
        row = names[matched_name]
        status = normalize_status(safe_get(row, "status"))
        details = safe_dict(safe_get(row, "details"))
        return {
            "player": target,
            "availability": status,
            "impact_score": severity_impact_score(status),
            "detail": _text(safe_get(details, "description")),
            "source": _text(safe_get(details, "source"), "offline"),
            "warning": "",
        }
    return {
        "player": target,
        "availability": "ACTIVE",
        "impact_score": 0.0,
        "detail": "",
        "source": "fallback",
        "warning": "Player was not listed in the loaded injury data; ACTIVE is a fallback, not confirmation.",
    }
