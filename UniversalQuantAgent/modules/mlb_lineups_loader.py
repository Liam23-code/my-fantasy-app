"""MLB starting lineups: our own default file plus user uploads -- never a live fetch.

No other sport in this codebase has an equivalent loader (NBA/CFB/CBB's
"today's games" come from a live schedule fetch this engine deliberately
doesn't add for MLB -- see mlb_pipeline.md); this mirrors the general
default-file-plus-upload shape of modules.cbb_props_loader, adapted for a
per-team lineup object instead of a flat list of prop rows. Feeds the
batting-order/handedness context modules.mlb_lineup_model and
modules.mlb_batter_vs_pitcher consume. ``data/mlb_lineups.json`` ships
empty by design.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from betting.cache_utils import ttl_cache

from modules.mlb_common import normalize_player_name, normalize_team_name

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_LINEUPS_PATH = DATA_ROOT / "mlb_lineups.json"
_DEFAULT_FILE_CACHE_SECONDS = 300

_VALID_HANDS = frozenset({"L", "R", "S"})


def _normalize_hand(value: Any) -> str:
    hand = str(value or "").strip().upper()[:1]
    return hand if hand in _VALID_HANDS else ""


def _normalize_pitcher(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = normalize_player_name(str(value.get("player_name") or value.get("name") or ""))
    if not name:
        return None
    return {"player_name": name, "hand": _normalize_hand(value.get("hand"))}


def _normalize_batting_order(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    order: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        name = normalize_player_name(str(entry.get("player_name") or entry.get("name") or ""))
        if not name:
            continue
        try:
            position = int(entry.get("position"))
        except (TypeError, ValueError):
            continue
        if not 1 <= position <= 9:
            continue
        order.append({"position": position, "player_name": name, "hand": _normalize_hand(entry.get("hand"))})
    order.sort(key=lambda row: row["position"])
    return order


def _normalize_lineup(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    team = normalize_team_name(str(row.get("team") or ""))
    if not team:
        return None
    lineup: dict[str, Any] = {"team": team, "source": source}
    opponent = row.get("opponent")
    if opponent:
        lineup["opponent"] = normalize_team_name(str(opponent))
    pitcher = _normalize_pitcher(row.get("starting_pitcher"))
    if pitcher is not None:
        lineup["starting_pitcher"] = pitcher
    batting_order = _normalize_batting_order(row.get("batting_order"))
    if batting_order:
        lineup["batting_order"] = batting_order
    if "starting_pitcher" not in lineup and "batting_order" not in lineup:
        return None
    return lineup


def _rows_from_dicts(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    result = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_lineup(raw, source=source)
        if normalized is not None:
            result.append(normalized)
    return result


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: dict[str, dict[str, Any]] = {}
    for row in rows:
        clean[row["team"]] = row
    return list(clean.values())


def _parse_json_payload(payload: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "lineups" in payload:
        payload = payload["lineups"]
    if isinstance(payload, list):
        return _rows_from_dicts([row for row in payload if isinstance(row, dict)], source=source)
    return []


@ttl_cache(_DEFAULT_FILE_CACHE_SECONDS)
def load_lineups_from_file(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load our own default MLB lineups (default: ``data/mlb_lineups.json``).

    A missing file is a normal "no lineups configured" state, not an
    error -- returns ``[]`` rather than raising.
    """
    target = Path(path) if path is not None else DEFAULT_LINEUPS_PATH
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    return _dedupe(_parse_json_payload(payload, source="default"))


def load_lineups_from_user_upload(source: Any, *, file_format: str | None = None) -> list[dict[str, Any]]:
    """Parse a user-uploaded MLB lineups file (JSON only -- a lineup is too structured for flat CSV rows)."""
    if isinstance(source, (dict, list)):
        return _dedupe(_parse_json_payload(source, source="upload"))

    if isinstance(source, (str, Path)) and "\n" not in str(source) and not str(source).strip().startswith(("{", "[")):
        path = Path(source)
        if not path.is_file():
            return []
        text = path.read_text(encoding="utf-8-sig")
    elif hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else str(raw)
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    elif isinstance(source, str):
        text = source
    else:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _dedupe(_parse_json_payload(payload, source="upload"))


def unified_lineups(
    *, default_path: str | Path | None = None, uploaded: Any = None, uploaded_format: str | None = None
) -> list[dict[str, Any]]:
    """Our default MLB lineups, optionally merged with an uploaded override (keyed by team)."""
    default_rows = load_lineups_from_file(default_path)
    if uploaded is None:
        return default_rows
    uploaded_rows = load_lineups_from_user_upload(uploaded, file_format=uploaded_format)
    merged: dict[str, dict[str, Any]] = {row["team"]: row for row in default_rows}
    for row in uploaded_rows:
        merged[row["team"]] = row
    return list(merged.values())


def get_team_lineup(team: str, lineups: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One real team's already-loaded lineup, or ``None`` if not present."""
    target = normalize_team_name(team)
    return next((row for row in lineups if row["team"] == target), None)
