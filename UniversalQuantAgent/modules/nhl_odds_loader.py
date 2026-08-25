"""NHL game odds (moneyline): our own default file plus user uploads -- never a sportsbook fetch.

Mirrors modules.mlb_odds_loader's shape and alias-based normalization,
substituting NHL's fixed 32-team alias table (modules.nhl_common). NHL's
puck line (the hockey equivalent of a spread) is out of scope this cycle
-- only moneyline and total are modeled. ``data/nhl_game_odds.json`` ships
empty by design.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from betting.cache_utils import ttl_cache

from modules.nhl_common import normalize_team_name

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_GAME_ODDS_PATH = DATA_ROOT / "nhl_game_odds.json"
_DEFAULT_FILE_CACHE_SECONDS = 300

_HOME_TEAM_ALIASES = ("home_team", "home")
_AWAY_TEAM_ALIASES = ("away_team", "away")
_ML_HOME_ALIASES = ("moneyline_home", "home_ml", "home_moneyline", "ml_home")
_ML_AWAY_ALIASES = ("moneyline_away", "away_ml", "away_moneyline", "ml_away")
_TOTAL_LINE_ALIASES = ("total_line", "total", "over_under")
_TOTAL_OVER_PRICE_ALIASES = ("total_over_price", "over_price_total", "total_over")
_TOTAL_UNDER_PRICE_ALIASES = ("total_under_price", "under_price_total", "total_under")


class GameOddsLoadError(ValueError):
    """A game-odds source (default or uploaded) could not be parsed."""


def _first(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _matchup_key(home: str, away: str) -> str:
    return f"{away}@{home}"


def _normalize_game_row(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    home = normalize_team_name(str(_first(row, _HOME_TEAM_ALIASES) or ""))
    away = normalize_team_name(str(_first(row, _AWAY_TEAM_ALIASES) or ""))
    if not (home and away):
        return None

    game: dict[str, Any] = {"home_team": home, "away_team": away, "source": source}
    ml_home = _safe_float(_first(row, _ML_HOME_ALIASES))
    ml_away = _safe_float(_first(row, _ML_AWAY_ALIASES))
    if ml_home is not None or ml_away is not None:
        game["moneyline"] = {"home": ml_home, "away": ml_away}

    total_line = _safe_float(_first(row, _TOTAL_LINE_ALIASES))
    if total_line is not None:
        game["total"] = {
            "line": total_line,
            "over_price": _safe_float(_first(row, _TOTAL_OVER_PRICE_ALIASES), -110.0),
            "under_price": _safe_float(_first(row, _TOTAL_UNDER_PRICE_ALIASES), -110.0),
        }

    if not any(key in game for key in ("moneyline", "total")):
        return None
    basis = row.get("basis")
    if basis:
        game["basis"] = str(basis)
    return game


def _rows_from_dicts(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    games: dict[str, Any] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = {str(key).strip().lower(): value for key, value in raw.items()}
        normalized = _normalize_game_row(row, source=source)
        if normalized is not None:
            games[_matchup_key(normalized["home_team"], normalized["away_team"])] = normalized
    return {"games": games}


def _parse_json_payload(payload: Any, *, source: str) -> dict[str, Any]:
    if isinstance(payload, dict) and "games" in payload:
        raw_games = payload.get("games")
        if isinstance(raw_games, dict):
            rows = list(raw_games.values())
        elif isinstance(raw_games, list):
            rows = [row for row in raw_games if isinstance(row, dict)]
        else:
            rows = []
        return _rows_from_dicts(rows, source=source)
    if isinstance(payload, list):
        return _rows_from_dicts([row for row in payload if isinstance(row, dict)], source=source)
    raise GameOddsLoadError("JSON game-odds payload must be a {'games': ...} object or a list of rows")


def _sniff_delimiter(text: str) -> str:
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "\t" in first_line:
        return "\t"
    if "|" in first_line:
        return "|"
    return ","


def _parse_delimited_text(text: str, *, source: str) -> dict[str, Any]:
    delimiter = _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        return {"games": {}}
    return _rows_from_dicts([dict(row) for row in reader], source=source)


@ttl_cache(_DEFAULT_FILE_CACHE_SECONDS)
def load_default_game_odds(path: str | Path | None = None) -> dict[str, Any]:
    """Load our default NHL game-odds file (default: ``data/nhl_game_odds.json``).

    A missing file is a normal "no game odds configured" state, not an
    error -- returns ``{"games": {}}`` rather than raising.
    """
    target = Path(path) if path is not None else DEFAULT_GAME_ODDS_PATH
    if not target.is_file():
        return {"games": {}}
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GameOddsLoadError(f"could not read default game-odds file {target}: {error}") from error
    return _parse_json_payload(payload, source="default")


def load_uploaded_game_odds(source: Any, *, file_format: str | None = None) -> dict[str, Any]:
    """Parse a user-uploaded NHL game-odds file: JSON, CSV, or delimited text."""
    if isinstance(source, (dict, list)):
        return _parse_json_payload(source, source="uploaded")

    name = ""
    if isinstance(source, (str, Path)) and "\n" not in str(source) and not str(source).strip().startswith(("{", "[")):
        path = Path(source)
        if not path.is_file():
            raise GameOddsLoadError(f"uploaded game-odds source does not exist: {path}")
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
        raise GameOddsLoadError(f"unsupported uploaded game-odds source type: {type(source).__name__}")

    fmt = file_format or {".json": "json", ".csv": "csv", ".tsv": "csv", ".txt": "text"}.get(name)
    if fmt is None:
        fmt = "json" if text.strip().startswith(("{", "[")) else "text"

    if fmt == "json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise GameOddsLoadError(f"invalid JSON game-odds upload: {error}") from error
        return _parse_json_payload(payload, source="uploaded")
    return _parse_delimited_text(text, source="uploaded")


def merge_game_odds(default: dict[str, Any], uploaded: dict[str, Any] | None) -> dict[str, Any]:
    """Merge default game odds with an uploaded override, keyed by matchup."""
    merged = {"games": dict(default.get("games") or {})}
    if not uploaded:
        return merged
    for key, game in (uploaded.get("games") or {}).items():
        merged["games"][key] = game
    return merged


def unified_game_odds(
    *, default_path: str | Path | None = None, uploaded: Any = None, uploaded_format: str | None = None
) -> dict[str, Any]:
    """Our default NHL game odds, optionally merged with an uploaded override."""
    default = load_default_game_odds(default_path)
    if uploaded is None:
        return default
    uploaded_odds = load_uploaded_game_odds(uploaded, file_format=uploaded_format)
    return merge_game_odds(default, uploaded_odds)


def index_by_matchup(game_odds: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Re-key a unified game-odds object's ``games`` by ``(home_team, away_team)`` for lookup."""
    return {(row["home_team"], row["away_team"]): row for row in (game_odds.get("games") or {}).values()}
