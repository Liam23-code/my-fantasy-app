"""Safe, offline MLB injury-data loading -- our own file plus user uploads, never a live fetch.

Mirrors modules.cbb_injuries_loader's loading functions exactly,
substituting MLB's fixed team alias table (modules.mlb_common) for
college normalization. The status/impact math has no sport in it at all
-- imported directly from :mod:`modules.injury_parser` rather than
duplicated. ``data/mlb_injuries.json`` ships empty by design.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.data_quality import safe_dict, safe_get, safe_list, safe_scalar_to_dict
from modules.injury_parser import normalize_status
from modules.mlb_common import normalize_player_name, normalize_team_name

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INJURIES_PATH = DATA_ROOT / "mlb_injuries.json"


def _text(value: Any, default: str = "") -> str:
    if isinstance(value, dict):
        value = safe_get(value, "displayName") or safe_get(value, "description") or safe_get(value, "name") or safe_get(value, "value")
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return default
    try:
        return str(value).strip()
    except Exception:
        return default


def _details(node: Any, *, source: str) -> dict[str, Any]:
    payload = safe_dict(safe_get(node, "details"))
    description = (
        _text(safe_get(payload, "description"))
        or _text(safe_get(node, "shortComment"))
        or _text(safe_get(node, "longComment"))
        or _text(safe_get(node, "details"))
    )
    return {"description": description, "source": source, "timestamp": datetime.now(timezone.utc).isoformat()}


def walk_injury_records(node: Any, team: str, output: list[dict[str, Any]], *, source: str = "offline") -> None:
    """Recursively parse an already-obtained, arbitrarily-nested MLB injury payload."""
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
        own_team = _text(safe_get(node, "team"))
        output.append(
            {
                "team": normalize_team_name(own_team or current_team or ""),
                "player": normalize_player_name(name),
                "status": normalize_status(raw_status),
                "details": _details(node, source=source),
            }
        )

    for child in node.values():
        if isinstance(child, (dict, list)):
            walk_injury_records(child, current_team, output, source=source)


def clean_injury_record(record: Any) -> dict[str, Any] | None:
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
    """Load a real MLB injury report from a local JSON file (default: ``data/mlb_injuries.json``).

    A missing file is a normal "no injuries configured" state, not an
    error -- returns ``[]`` rather than raising.
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
    """Parse a user-uploaded MLB injury file: JSON, CSV, or delimited text."""
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
        lines = text.splitlines()
        delimiter = "\t" if lines and "\t" in lines[0] else "|" if lines and "|" in lines[0] else ","
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
    """Availability plus an impact score (100 = cannot play) for one player, from already-loaded records."""
    from modules.data_quality import fuzzy_name_match
    from modules.injury_parser import severity_impact_score

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
