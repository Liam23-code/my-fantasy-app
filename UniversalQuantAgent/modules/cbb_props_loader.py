"""CBB player prop lines: our own default file plus user uploads -- never a sportsbook fetch.

Mirrors modules.nba_props_loader's default-file-plus-upload convention
exactly. CBB shares NBA's basketball stat vocabulary (points/rebounds/
assists/PRA/3PM), so :func:`modules.sportsbook_parser.normalize_category`
is reused directly rather than redefined; only team-name normalization
differs (no fixed alias table -- see modules.college_sports_common).
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from betting.cache_utils import ttl_cache

from modules.college_sports_common import normalize_college_team_name, normalize_player_name
from modules.sportsbook_parser import normalize_category, now_iso, parse_market_json

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROPS_PATH = DATA_ROOT / "cbb_props.json"
_DEFAULT_FILE_CACHE_SECONDS = 300

_PLAYER_ALIASES = ("player_name", "player", "name", "athlete")
_TEAM_ALIASES = ("team", "cbb_team")
_CATEGORY_ALIASES = ("category", "market", "stat", "prop", "prop_type")
_LINE_ALIASES = ("line", "prop_line", "value", "handicap")
_SPORTSBOOK_ALIASES = ("sportsbook", "source", "book")
_OVER_PRICE_ALIASES = ("over_price", "over", "over_odds")
_UNDER_PRICE_ALIASES = ("under_price", "under", "under_odds")
_DEFAULT_PRICE = -110.0


def _first(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _safe_price(value: Any) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_PRICE
    return price if price == price else _DEFAULT_PRICE


def _normalize_row(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    player = normalize_player_name(str(_first(row, _PLAYER_ALIASES) or ""))
    category = normalize_category(str(_first(row, _CATEGORY_ALIASES) or ""))
    line_raw = _first(row, _LINE_ALIASES)
    if not player or category is None or line_raw in (None, ""):
        return None
    try:
        line = round(float(line_raw), 2)
    except (TypeError, ValueError):
        return None
    team_raw = _first(row, _TEAM_ALIASES)
    over_price_raw, under_price_raw = _first(row, _OVER_PRICE_ALIASES), _first(row, _UNDER_PRICE_ALIASES)
    normalized = {
        "player_name": player,
        "team": normalize_college_team_name(str(team_raw)) if team_raw else "",
        "category": category,
        "line": line,
        "over_price": _safe_price(over_price_raw) if over_price_raw is not None else _DEFAULT_PRICE,
        "under_price": _safe_price(under_price_raw) if under_price_raw is not None else _DEFAULT_PRICE,
        "sportsbook": str(_first(row, _SPORTSBOOK_ALIASES) or source),
        "timestamp": str(row.get("timestamp") or now_iso()),
    }
    basis = row.get("basis")
    if basis:
        normalized["basis"] = str(basis)
    return normalized


def _rows_from_dicts(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    result = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = {str(key).strip().lower(): value for key, value in raw.items()}
        normalized = _normalize_row(row, source=source)
        if normalized is not None:
            result.append(normalized)
    return result


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        clean[(row["player_name"], row["category"])] = row
    return list(clean.values())


def _parse_json_payload(payload: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "props" in payload:
        payload = payload["props"]
    if isinstance(payload, list):
        flat = _rows_from_dicts([row for row in payload if isinstance(row, dict)], source=source)
        if flat:
            return flat
    return parse_market_json(payload, source=source)


def _sniff_delimiter(text: str) -> str:
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "\t" in first_line:
        return "\t"
    if "|" in first_line:
        return "|"
    return ","


def _parse_delimited_text(text: str, *, source: str) -> list[dict[str, Any]]:
    delimiter = _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        return []
    return _rows_from_dicts([dict(row) for row in reader], source=source)


@ttl_cache(_DEFAULT_FILE_CACHE_SECONDS)
def load_props_from_file(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load our own default CBB prop lines (default: ``data/cbb_props.json``).

    A missing file is a normal "no props configured" state, not an error --
    returns ``[]`` rather than raising.
    """
    target = Path(path) if path is not None else DEFAULT_PROPS_PATH
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    return _dedupe(_parse_json_payload(payload, source="default"))


def load_props_from_user_upload(source: Any, *, file_format: str | None = None) -> list[dict[str, Any]]:
    """Parse a user-uploaded CBB prop-lines file: JSON, CSV, or delimited text."""
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

    if fmt == "json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _dedupe(_parse_json_payload(payload, source="upload"))
    return _dedupe(_parse_delimited_text(text, source="upload"))


def unified_props(
    *, default_path: str | Path | None = None, uploaded: Any = None, uploaded_format: str | None = None
) -> list[dict[str, Any]]:
    """Our default CBB prop lines, optionally merged with an uploaded override."""
    default_rows = load_props_from_file(default_path)
    if uploaded is None:
        return default_rows
    uploaded_rows = load_props_from_user_upload(uploaded, file_format=uploaded_format)
    merged: dict[tuple[str, str], dict[str, Any]] = {(row["player_name"], row["category"]): row for row in default_rows}
    for row in uploaded_rows:
        merged[(row["player_name"], row["category"])] = row
    return list(merged.values())
