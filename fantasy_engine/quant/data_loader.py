"""Offline-safe ingestion and normalization for the fantasy Quant Engine.

The public loaders in this module deliberately do not fetch the network.  A
caller may pass an in-memory mapping/list, a JSON/JSONL/CSV/TSV path, or leave
``source`` unset to use a matching file in :data:`DATA_ROOT`.  Missing default
files are an ordinary empty-data condition; malformed explicit inputs can be
reported by opting into ``strict=True``.

Every loader emits the same canonical player shape.  Source-specific payloads
remain available in nested fields (``injury``, ``schedule``, ``news`` and so
on), which lets downstream analytics share identity and projection fields
without throwing useful provider data away.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

SOURCE_NAMES: tuple[str, ...] = (
    "player_stats",
    "historical_data",
    "injury_data",
    "depth_charts",
    "adp_feeds",
    "schedule_data",
    "team_strength",
    "weather_data",
    "news_signals",
)

CANONICAL_PLAYER_SCHEMA: dict[str, str] = {
    "player_id": "str",
    "name": "str",
    "position": "str",
    "team": "str",
    "opponent": "str",
    "season": "int",
    "week": "int",
    "projection": "float",
    "expected_fantasy_points": "float",
    "projection_confidence": "float",
    "points_per_game": "float",
    "games_played": "int",
    "floor": "float",
    "median": "float",
    "ceiling": "float",
    "adp": "float | None",
    "bye_week": "int | None",
    "injury_status": "str",
    "injury_risk": "float",
    "health_score": "float",
    "depth_order": "int | None",
    "depth_role": "str",
    "team_strength": "float",
    "opponent_strength": "float",
    "weather_risk": "float",
    "news_sentiment": "float",
    "usage_rate": "float",
    "efficiency": "float",
    "volatility": "float",
    "age": "int | None",
    "years_experience": "int | None",
    "expected_games": "float | None",
    "market_projection": "float | None",
    "stats": "dict",
    "history": "list",
    "injury": "dict",
    "depth_chart": "dict",
    "adp_data": "dict",
    "schedule": "list",
    "team_strength_data": "dict",
    "weather": "dict",
    "news": "list",
    "metadata": "dict",
    "sources": "list[str]",
}

_DEFAULTS: dict[str, Any] = {
    "player_id": "",
    "name": "",
    "position": "",
    "team": "",
    "opponent": "",
    "season": 0,
    "week": 0,
    "projection": 0.0,
    "expected_fantasy_points": 0.0,
    "projection_confidence": 0.5,
    "points_per_game": 0.0,
    "games_played": 0,
    "floor": 0.0,
    "median": 0.0,
    "ceiling": 0.0,
    "adp": None,
    "bye_week": None,
    "injury_status": "ACTIVE",
    "injury_risk": 0.0,
    "health_score": 1.0,
    "depth_order": None,
    "depth_role": "",
    "team_strength": 0.5,
    "opponent_strength": 0.5,
    "weather_risk": 0.0,
    "news_sentiment": 0.0,
    "usage_rate": 0.0,
    "efficiency": 0.0,
    "volatility": 0.0,
    "age": None,
    "years_experience": None,
    "expected_games": None,
    "market_projection": None,
    "stats": {},
    "history": [],
    "injury": {},
    "depth_chart": {},
    "adp_data": {},
    "schedule": [],
    "team_strength_data": {},
    "weather": {},
    "news": [],
    "metadata": {},
    "sources": [],
}

_IDENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "player_id": ("player_id", "playerId", "id", "gsis_id", "nfl_id"),
    "name": ("name", "player_name", "player_display_name", "full_name", "player"),
    "position": ("position", "pos", "player_position"),
    "team": ("team", "nfl_team", "club", "recent_team"),
    "opponent": ("opponent", "opp", "opponent_team"),
}

_FLOAT_ALIASES: dict[str, tuple[str, ...]] = {
    "projection": ("projection", "projected_points", "fantasy_projection", "season_projection"),
    "expected_fantasy_points": ("expected_fantasy_points", "fantasy_points", "points", "actual_points"),
    "projection_confidence": ("projection_confidence", "confidence", "confidence_score"),
    "points_per_game": ("points_per_game", "ppg", "fantasy_points_per_game"),
    "floor": ("floor", "projection_floor"),
    "median": ("median", "projection_median"),
    "ceiling": ("ceiling", "projection_ceiling"),
    "adp": ("adp", "average_draft_position", "ecr", "rank"),
    "injury_risk": ("injury_risk", "risk", "injury_probability"),
    "health_score": ("health_score", "availability", "availability_score"),
    "team_strength": ("team_strength", "strength", "power_rating", "offensive_strength"),
    "opponent_strength": ("opponent_strength", "defensive_strength", "defense_rating"),
    "weather_risk": ("weather_risk", "weather_impact", "risk_score"),
    "news_sentiment": ("news_sentiment", "sentiment", "signal"),
    "usage_rate": ("usage_rate", "usage", "opportunity_share", "snap_share"),
    "efficiency": ("efficiency", "efficiency_score", "fantasy_points_per_opportunity"),
    "volatility": ("volatility", "coefficient_of_variation"),
    "expected_games": ("expected_games", "projected_games"),
    "market_projection": ("market_projection", "market_proj"),
}

_INT_ALIASES: dict[str, tuple[str, ...]] = {
    "season": ("season", "year"),
    "week": ("week", "game_week"),
    "games_played": ("games_played", "games", "gp"),
    "bye_week": ("bye_week", "bye"),
    "depth_order": ("depth_order", "depth", "order", "rank"),
    "age": ("age",),
    "years_experience": ("years_experience", "experience", "years_pro"),
}

#: Fields that stay ``None`` when absent from the source record instead of
#: falling back to a zero-like default. Consumers (e.g. the projection
#: ensemble) treat "not provided" and "provided as zero" differently.
_NULLABLE_FLOAT_FIELDS = frozenset({"adp", "expected_games", "market_projection"})
_NULLABLE_INT_FIELDS = frozenset({"bye_week", "depth_order", "age", "years_experience"})

_STAT_ALIASES: dict[str, tuple[str, ...]] = {
    "passing_yards": ("passing_yards", "pass_yards"),
    "passing_tds": ("passing_tds", "pass_tds", "passing_touchdowns"),
    "interceptions": ("interceptions", "passing_interceptions", "ints"),
    "rushing_yards": ("rushing_yards", "rush_yards"),
    "rushing_tds": ("rushing_tds", "rush_tds", "rushing_touchdowns"),
    "receptions": ("receptions", "rec"),
    "receiving_yards": ("receiving_yards", "rec_yards"),
    "receiving_tds": ("receiving_tds", "rec_tds", "receiving_touchdowns"),
    "fumbles_lost": ("fumbles_lost", "fumbles"),
    "targets": ("targets",),
    "carries": ("carries", "rushing_attempts", "rush_attempts"),
    "snaps": ("snaps", "offensive_snaps"),
    "routes": ("routes", "routes_run"),
    "touches": ("touches",),
    "target_share": ("target_share",),
    "snap_share": ("snap_share",),
    "route_participation": ("route_participation", "route_share"),
    "rush_share": ("rush_share", "carry_share"),
}

_CONTAINER_KEYS = ("players", "data", "results", "records", "items", "rows")
_RECORD_HINTS = frozenset(alias for aliases in (_IDENTITY_ALIASES | _FLOAT_ALIASES | _INT_ALIASES).values() for alias in aliases)

_DEFAULT_FILE_STEMS: dict[str, tuple[str, ...]] = {
    "player_stats": ("player_stats", "players", "projections"),
    "historical_data": ("historical_data", "history", "historical"),
    "injury_data": ("injury_data", "injuries"),
    "depth_charts": ("depth_charts", "depth_chart"),
    "adp_feeds": ("adp_feeds", "adp", "rankings"),
    "schedule_data": ("schedule_data", "schedule"),
    "team_strength": ("team_strength", "team_ratings"),
    "weather_data": ("weather_data", "weather"),
    "news_signals": ("news_signals", "news"),
}


class DataLoadError(ValueError):
    """An explicit Quant data source could not be parsed or normalized."""


def canonical_player_schema() -> dict[str, Any]:
    """Return a new empty canonical record (nested values are never shared)."""
    return copy.deepcopy(_DEFAULTS)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, _safe_float(value, lower)))


def _first(row: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return dict(value.model_dump())
    if hasattr(value, "_asdict") and callable(value._asdict):
        return dict(value._asdict())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected a mapping-like record, got {type(value).__name__}")


def _looks_like_record(payload: Mapping[str, Any]) -> bool:
    return bool(set(payload) & _RECORD_HINTS)


def _read_path(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise DataLoadError(f"unsupported data format {suffix!r}; use JSON, JSONL, CSV, or TSV")


def _records_from_payload(payload: Any, *, strict: bool, explicit: bool = True) -> list[dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, (str, Path)):
        path = Path(payload).expanduser()
        if path.exists():
            if not path.is_file():
                if strict:
                    raise DataLoadError(f"data source is not a file: {path}")
                return []
            try:
                return _records_from_payload(_read_path(path), strict=strict)
            except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, DataLoadError) as error:
                if strict:
                    raise DataLoadError(f"could not read {path}: {error}") from error
                return []
        text = str(payload).strip()
        if text.startswith(("{", "[")):
            try:
                return _records_from_payload(json.loads(text), strict=strict)
            except json.JSONDecodeError as error:
                if strict:
                    raise DataLoadError(f"invalid JSON source: {error}") from error
                return []
        if strict and explicit:
            raise DataLoadError(f"data source does not exist: {path}")
        return []

    if hasattr(payload, "to_dict") and callable(payload.to_dict) and not isinstance(payload, Mapping):
        try:
            records = payload.to_dict("records")
        except (TypeError, ValueError):
            records = None
        if records is not None:
            return _records_from_payload(records, strict=strict)

    if isinstance(payload, Mapping):
        row = dict(payload)
        for key in _CONTAINER_KEYS:
            nested = row.get(key)
            if isinstance(nested, (Mapping, list, tuple)):
                return _records_from_payload(nested, strict=strict)
        if _looks_like_record(row) or not row:
            return [row] if row else []
        if row and all(isinstance(value, Mapping) for value in row.values()):
            records: list[dict[str, Any]] = []
            for key, value in row.items():
                record = dict(value)
                record.setdefault("player_id", key)
                records.append(record)
            return records
        if strict:
            raise DataLoadError("mapping is neither a player record nor a recognized record container")
        return []

    if isinstance(payload, Iterable) and not isinstance(payload, (bytes, bytearray)):
        records = []
        for item in payload:
            try:
                records.append(_as_mapping(item))
            except TypeError as error:
                if strict:
                    raise DataLoadError(str(error)) from error
        return records

    try:
        return [_as_mapping(payload)]
    except TypeError as error:
        if strict:
            raise DataLoadError(str(error)) from error
        return []


def _default_source(source_name: str) -> Path | None:
    for stem in _DEFAULT_FILE_STEMS[source_name]:
        for suffix in (".json", ".jsonl", ".ndjson", ".csv", ".tsv"):
            candidate = DATA_ROOT / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _stable_player_id(name: str, team: str, position: str, source_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", f"{name}|{team}|{position}".lower())
    if normalized:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"quant:{digest}"
    if team:
        return f"team:{team.lower()}"
    return f"{source_name}:unknown"


def _status_risk(status: str) -> float:
    normalized = status.strip().upper().replace("-", "_").replace(" ", "_")
    return {
        "IR": 1.0,
        "PUP": 0.95,
        "OUT": 1.0,
        "DOUBTFUL": 0.8,
        "QUESTIONABLE": 0.35,
        "LIMITED": 0.2,
        "PROBABLE": 0.08,
        "ACTIVE": 0.0,
        "HEALTHY": 0.0,
    }.get(normalized, 0.1 if normalized else 0.0)


def _source_payload(row: Mapping[str, Any], source_name: str) -> tuple[str, Any]:
    payload = dict(row)
    if source_name == "player_stats":
        return "stats", {key: value for key, value in payload.items() if key in _STAT_ALIASES or key == "stats"}
    if source_name == "historical_data":
        history = payload.get("history")
        return "history", list(history) if isinstance(history, (list, tuple)) else [payload]
    if source_name == "injury_data":
        return "injury", dict(payload.get("injury", {})) | payload if isinstance(payload.get("injury"), Mapping) else payload
    if source_name == "depth_charts":
        return "depth_chart", dict(payload.get("depth_chart", {})) | payload if isinstance(payload.get("depth_chart"), Mapping) else payload
    if source_name == "adp_feeds":
        return "adp_data", dict(payload.get("adp_data", {})) | payload if isinstance(payload.get("adp_data"), Mapping) else payload
    if source_name == "schedule_data":
        schedule = payload.get("schedule")
        return "schedule", list(schedule) if isinstance(schedule, (list, tuple)) else [payload]
    if source_name == "team_strength":
        if isinstance(payload.get("team_strength_data"), Mapping):
            return "team_strength_data", dict(payload["team_strength_data"]) | payload
        return "team_strength_data", payload
    if source_name == "weather_data":
        return "weather", dict(payload.get("weather", {})) | payload if isinstance(payload.get("weather"), Mapping) else payload
    news = payload.get("news")
    return "news", list(news) if isinstance(news, (list, tuple)) else [payload]


def normalize_player_record(record: Any, *, source_name: str = "player_stats", strict: bool = False) -> dict[str, Any] | None:
    """Normalize one mapping/object to the canonical Quant player schema.

    Invalid records return ``None`` by default and raise :class:`DataLoadError`
    in strict mode.  Identity-less team data is valid: it receives a stable
    ``team:<abbr>`` entity id so it can enrich every player on that team later.
    """
    if source_name not in SOURCE_NAMES:
        raise ValueError(f"unknown source_name {source_name!r}")
    try:
        raw = _as_mapping(record)
    except TypeError as error:
        if strict:
            raise DataLoadError(str(error)) from error
        return None

    nested_player = raw.get("player")
    if isinstance(nested_player, Mapping):
        row = dict(nested_player) | raw
        row.pop("player", None)
    else:
        row = raw

    canonical = canonical_player_schema()
    for field, aliases in _IDENTITY_ALIASES.items():
        value = _first(row, aliases)
        canonical[field] = str(value or "").strip()
    canonical["position"] = canonical["position"].upper()
    canonical["team"] = canonical["team"].upper()
    canonical["opponent"] = canonical["opponent"].upper()

    for field, aliases in _FLOAT_ALIASES.items():
        raw_value = _first(row, aliases)
        if field in _NULLABLE_FLOAT_FIELDS and raw_value is None:
            canonical[field] = None
        elif raw_value is not None:
            canonical[field] = _safe_float(raw_value, canonical[field])
    for field, aliases in _INT_ALIASES.items():
        raw_value = _first(row, aliases)
        if field in _NULLABLE_INT_FIELDS and raw_value is None:
            canonical[field] = None
        elif raw_value is not None:
            canonical[field] = _safe_int(raw_value, canonical[field] or 0)

    if canonical["expected_fantasy_points"] <= 0 < canonical["projection"]:
        canonical["expected_fantasy_points"] = canonical["projection"]
    if canonical["projection"] <= 0 < canonical["expected_fantasy_points"]:
        canonical["projection"] = canonical["expected_fantasy_points"]
    if canonical["median"] <= 0 < canonical["projection"]:
        canonical["median"] = canonical["projection"]

    status = _first(row, ("injury_status", "status", "game_status", "practice_status"))
    if status is not None:
        canonical["injury_status"] = str(status).strip().upper() or "ACTIVE"
    provided_risk = _first(row, _FLOAT_ALIASES["injury_risk"])
    canonical["injury_risk"] = _clamp(provided_risk if provided_risk is not None else _status_risk(canonical["injury_status"]))
    provided_health = _first(row, _FLOAT_ALIASES["health_score"])
    canonical["health_score"] = _clamp(provided_health if provided_health is not None else 1.0 - canonical["injury_risk"])
    role = _first(row, ("depth_role", "role", "designation"))
    canonical["depth_role"] = str(role or "").strip().upper()
    canonical["projection_confidence"] = _clamp(canonical["projection_confidence"])
    canonical["usage_rate"] = _clamp(canonical["usage_rate"])
    canonical["weather_risk"] = _clamp(canonical["weather_risk"])
    canonical["news_sentiment"] = max(-1.0, min(1.0, canonical["news_sentiment"]))

    nested_stats = row.get("stats") if isinstance(row.get("stats"), Mapping) else {}
    stats = {str(key): value for key, value in nested_stats.items()}
    for field, aliases in _STAT_ALIASES.items():
        value = _first(row, aliases)
        if value is not None:
            stats[field] = _safe_float(value)
            canonical[field] = stats[field]
    canonical["stats"] = stats

    payload_field, payload = _source_payload(row, source_name)
    if isinstance(payload, Mapping) and isinstance(canonical[payload_field], Mapping):
        canonical[payload_field] = dict(canonical[payload_field]) | copy.deepcopy(dict(payload))
    elif isinstance(payload, (list, tuple)) and isinstance(canonical[payload_field], list):
        canonical[payload_field] = copy.deepcopy(list(payload))
    else:
        canonical[payload_field] = copy.deepcopy(payload)
    for nested_field in (
        "stats",
        "history",
        "injury",
        "depth_chart",
        "adp_data",
        "schedule",
        "team_strength_data",
        "weather",
        "news",
    ):
        incoming = row.get(nested_field)
        if isinstance(incoming, Mapping) and isinstance(canonical[nested_field], Mapping):
            canonical[nested_field] = dict(canonical[nested_field]) | copy.deepcopy(dict(incoming))
        elif isinstance(incoming, (list, tuple)) and isinstance(canonical[nested_field], list):
            canonical[nested_field] = copy.deepcopy(list(incoming))
    canonical["metadata"] = {
        "source_type": source_name,
        "source_fields": sorted(str(key) for key in raw),
        "raw": copy.deepcopy(raw),
    }
    canonical["sources"] = [source_name]
    canonical["player_id"] = canonical["player_id"] or _stable_player_id(
        canonical["name"], canonical["team"], canonical["position"], source_name
    )

    if not any((canonical["name"], canonical["team"], canonical["player_id"])):
        if strict:
            raise DataLoadError("record has no player or team identity")
        return None
    return canonical


def _load(source_name: str, source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    explicit = source is not None
    payload = source if explicit else _default_source(source_name)
    records = _records_from_payload(payload, strict=strict, explicit=explicit)
    normalized: list[dict[str, Any]] = []
    for record in records:
        player = normalize_player_record(record, source_name=source_name, strict=strict)
        if player is not None:
            normalized.append(player)
    return normalized


def load_player_stats(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical current/player-stat records."""
    return _load("player_stats", source, strict=strict)


def load_historical_data(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical historical player records or weekly histories."""
    return _load("historical_data", source, strict=strict)


def load_injury_data(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical injury and availability records."""
    return _load("injury_data", source, strict=strict)


def load_depth_charts(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical depth-chart records."""
    return _load("depth_charts", source, strict=strict)


def load_adp_feeds(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical average-draft-position records."""
    return _load("adp_feeds", source, strict=strict)


def load_schedule_data(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical player- or team-level schedule records."""
    return _load("schedule_data", source, strict=strict)


def load_team_strength(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical team strength and opponent defense records."""
    return _load("team_strength", source, strict=strict)


def load_weather_data(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical weather-impact records."""
    return _load("weather_data", source, strict=strict)


def load_news_signals(source: Any = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load canonical player news and sentiment signals."""
    return _load("news_signals", source, strict=strict)


_LOADERS = {
    "player_stats": load_player_stats,
    "historical_data": load_historical_data,
    "injury_data": load_injury_data,
    "depth_charts": load_depth_charts,
    "adp_feeds": load_adp_feeds,
    "schedule_data": load_schedule_data,
    "team_strength": load_team_strength,
    "weather_data": load_weather_data,
    "news_signals": load_news_signals,
}

_SOURCE_ALIASES = {
    "players": "player_stats",
    "stats": "player_stats",
    "history": "historical_data",
    "historical": "historical_data",
    "injuries": "injury_data",
    "depth_chart": "depth_charts",
    "adp": "adp_feeds",
    "schedule": "schedule_data",
    "strength": "team_strength",
    "weather": "weather_data",
    "news": "news_signals",
}


def _identity_key(player: Mapping[str, Any]) -> tuple[str, str, str]:
    name = re.sub(r"[^a-z0-9]+", "", str(player.get("name") or "").lower())
    return name, str(player.get("team") or "").upper(), str(player.get("position") or "").upper()


def merge_player_records(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two canonical records without mutating either input."""
    merged = canonical_player_schema() | copy.deepcopy(dict(base))
    incoming = copy.deepcopy(dict(update))
    nested_dicts = {"stats", "injury", "depth_chart", "adp_data", "team_strength_data", "weather", "metadata"}
    nested_lists = {"history", "schedule", "news", "sources"}
    metadata = incoming.get("metadata") if isinstance(incoming.get("metadata"), Mapping) else {}
    raw = metadata.get("raw") if isinstance(metadata.get("raw"), Mapping) else {}
    aliases_by_field = _IDENTITY_ALIASES | _FLOAT_ALIASES | _INT_ALIASES

    for field in nested_dicts:
        current = merged.get(field) if isinstance(merged.get(field), Mapping) else {}
        addition = incoming.get(field) if isinstance(incoming.get(field), Mapping) else {}
        merged[field] = dict(current) | dict(addition)
    for field in nested_lists:
        current = list(merged.get(field) or [])
        for item in list(incoming.get(field) or []):
            if item not in current:
                current.append(item)
        merged[field] = current

    for field, value in incoming.items():
        if field in nested_dicts | nested_lists:
            continue
        if field == "player_id" and merged.get("player_id"):
            continue
        if value is None or value == "":
            continue
        aliases = aliases_by_field.get(field, (field,))
        if field == "injury_status":
            aliases = ("injury_status", "status", "game_status", "practice_status")
        elif field == "depth_role":
            aliases = ("depth_role", "role", "designation")
        explicit = any(alias in raw for alias in aliases)
        if not explicit and field in _DEFAULTS and value == _DEFAULTS[field] and merged.get(field) != _DEFAULTS[field]:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0 and merged.get(field) not in (None, 0, 0.0, ""):
            continue
        merged[field] = value
    return merged


def load_all_player_data(
    sources: Mapping[str, Any] | Any | None = None,
    *,
    players: Any = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Load and merge every configured source into canonical player records.

    ``sources`` is normally ``{"player_stats": ..., "injury_data": ...}``.
    Passing a list/path directly is shorthand for ``player_stats``.  ``players``
    is an explicit base pool and is merged before all enrichments.
    Team-level schedule, strength, and weather rows are applied to each player
    on that team instead of becoming phantom roster entries.
    """
    source_map: dict[str, Any]
    if sources is None:
        source_map = {}
    elif isinstance(sources, Mapping) and any(str(key) in SOURCE_NAMES or str(key) in _SOURCE_ALIASES for key in sources):
        source_map = dict(sources)
    else:
        source_map = {"player_stats": sources}
    if players is not None:
        source_map["player_stats"] = players

    datasets: list[tuple[str, list[dict[str, Any]]]] = []
    for source_name in SOURCE_NAMES:
        supplied = next(
            (value for key, value in source_map.items() if _SOURCE_ALIASES.get(str(key), str(key)) == source_name),
            None,
        )
        should_use_default = sources is None and players is None
        rows = _LOADERS[source_name](supplied if supplied is not None else (None if should_use_default else []), strict=strict)
        datasets.append((source_name, rows))

    unified: list[dict[str, Any]] = []
    by_id: dict[str, int] = {}
    by_identity: dict[tuple[str, str, str], int] = {}
    by_name_team: dict[tuple[str, str], int] = {}
    by_name: dict[str, int] = {}

    for source_name, rows in datasets:
        for row in rows:
            team_only = not row.get("name") and not row.get("position") and bool(row.get("team"))
            if team_only and source_name in {"schedule_data", "team_strength", "weather_data"} and unified:
                matched = False
                for index, player in enumerate(unified):
                    if player.get("team") == row.get("team"):
                        unified[index] = merge_player_records(player, row)
                        matched = True
                if matched:
                    continue

            player_id = str(row.get("player_id") or "")
            identity = _identity_key(row)
            index = by_id.get(player_id)
            if index is None and any(identity):
                index = by_identity.get(identity)
            if index is None and identity[0]:
                index = by_name_team.get((identity[0], identity[1]))
            if index is None and identity[0]:
                index = by_name.get(identity[0])
            if index is None:
                index = len(unified)
                unified.append(copy.deepcopy(row))
            else:
                unified[index] = merge_player_records(unified[index], row)
            by_id[player_id] = index
            if any(identity):
                by_identity[identity] = index
            if identity[0]:
                by_name_team[(identity[0], identity[1])] = index
                by_name.setdefault(identity[0], index)

    return unified


__all__ = [
    "CANONICAL_PLAYER_SCHEMA",
    "DATA_ROOT",
    "DataLoadError",
    "SOURCE_NAMES",
    "canonical_player_schema",
    "load_adp_feeds",
    "load_all_player_data",
    "load_depth_charts",
    "load_historical_data",
    "load_injury_data",
    "load_news_signals",
    "load_player_stats",
    "load_schedule_data",
    "load_team_strength",
    "load_weather_data",
    "merge_player_records",
    "normalize_player_record",
]
