"""Shared data-cleaning and fallback helpers for every analytics domain.

Provider adapters should normalize at their boundary.  Keeping these rules in
one small module prevents finance, NBA, NFL, and sportsbook pipelines from
quietly interpreting the same missing value in different ways.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")

COLUMN_VARIANTS = {
    "playerid":"player_id", "gameid":"game_id", "teamid":"team_id",
    "game_date_est":"game_date", "date":"game_date", "minutes":"min",
    "points":"pts", "rebounds":"reb", "assists":"ast",
    "team":"team_abbreviation", "team_abbr":"team_abbreviation",
    "usage_pct":"usg_pct", "usage_rate":"usg_pct",
    "true_shooting_pct":"ts_pct", "true_shooting_percentage":"ts_pct",
    "offensive_rating":"off_rating", "offrtg":"off_rating", "ortg":"off_rating",
    "defensive_rating":"def_rating", "defrtg":"def_rating", "drtg":"def_rating",
    "opponent":"opponent_abbreviation", "opp":"opponent_abbreviation",
    "opponent_team":"opponent_abbreviation", "opp_team":"opponent_abbreviation",
    "opponent_team_abbreviation":"opponent_abbreviation",
    "is_available":"available", "active":"available",
}

COMMON_NUMERIC_COLUMNS = {
    "player_id", "game_id", "team_id", "min", "pts", "reb", "ast",
    "stl", "blk", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "tov",
    "plus_minus", "usg_pct", "ts_pct", "pace", "off_rating", "def_rating",
    "net_rating", "gp", "w", "l",
}

def safe_get(node: Any, key: Any, default: Any = None) -> Any:
    """Read one mapping key without trusting the provider payload type.

    Lists, scalars, None, and unusual mapping implementations all return the
    default instead of leaking an attribute or indexing exception.
    """
    if not isinstance(node, dict):
        return default
    try:
        return node.get(key, default)
    except Exception:
        return default


def safe_dict(node: Any) -> dict[str, Any]:
    """Return a real dictionary or an empty dictionary for malformed input."""
    return node if isinstance(node, dict) else {}


def safe_list(node: Any) -> list[Any]:
    """Return a real list; provider scalars are never treated as iterables."""
    return node if isinstance(node, list) else []


def safe_scalar_to_dict(node: Any) -> dict[str, Any]:
    """Wrap a provider scalar in a predictable mapping without ever raising."""
    if isinstance(node, dict):
        return node
    if node is None or isinstance(node, (list, tuple, set)):
        return {}
    try:
        return {"value": node}
    except Exception:
        return {}


def as_dict(value: Any) -> dict[str, Any]:
    """Backward-compatible alias for safe_dict."""
    return safe_dict(value)


def as_list(value: Any) -> list[Any]:
    """Normalize optional provider lists without iterating strings or scalars."""
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]

def safe_number(value: Any, default: float = 0.0) -> float:
    """Convert one value to a finite float without leaking NaN or infinity."""
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)

def normalize_text(value: Any) -> str:
    """Lowercase text and collapse punctuation/whitespace for matching."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())

def normalize_column_name(value: Any) -> str:
    key = "_".join(normalize_text(value).split())
    return COLUMN_VARIANTS.get(key, key)

def normalize_columns(frame: pd.DataFrame,
                      variants: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalize provider columns and merge duplicate variants left-to-right."""
    data = frame.copy()
    mapping = dict(COLUMN_VARIANTS)
    if variants:
        mapping.update(variants)
    renamed = {}
    for column in data.columns:
        key = "_".join(normalize_text(column).split())
        renamed[column] = mapping.get(key, key)
    data = data.rename(columns=renamed)
    if data.columns.duplicated().any():
        merged = {}
        for column in dict.fromkeys(data.columns):
            matches = data.loc[:, data.columns == column]
            merged[column] = matches.bfill(axis=1).iloc[:, 0]
        data = pd.DataFrame(merged, index=data.index)
    return data

def parse_minutes(value: Any, default: float = np.nan) -> float:
    """Parse numeric minutes and NBA ``MM:SS`` strings."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return default
    text = str(value).strip()
    try:
        if ":" in text:
            minute, second = text.split(":", 1)
            return float(minute) + float(second) / 60.0
        return float(text)
    except (TypeError, ValueError):
        return default

def coerce_numeric(frame: pd.DataFrame,
                   columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Normalize and coerce known/requested numeric columns."""
    data = normalize_columns(frame)
    requested = set(columns or COMMON_NUMERIC_COLUMNS)
    if "min" in data and "min" in requested:
        data["min"] = data["min"].map(parse_minutes)
    for column in requested:
        if column in data and column != "min":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.replace([np.inf, -np.inf], np.nan)

def fill_missing(frame: pd.DataFrame,
                 defaults: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fill numeric gaps with medians, then apply explicit provider defaults."""
    data = frame.replace([np.inf, -np.inf], np.nan).copy()
    for column in data.select_dtypes(include="number"):
        median = data[column].median()
        if pd.notna(median):
            data[column] = data[column].fillna(median)
    for column, value in (defaults or {}).items():
        if column not in data:
            data[column] = value
        else:
            data[column] = data[column].fillna(value)
    return data

def rolling_average(series: pd.Series, window: int, fallback: float = 0.0,
                    prior_only: bool = False) -> pd.Series:
    """Return a stable rolling average even when fewer than ``window`` rows exist."""
    numeric = pd.to_numeric(series, errors="coerce")
    source = numeric.shift(1) if prior_only else numeric
    result = source.rolling(window, min_periods=1).mean()
    result = result.fillna(source.expanding(min_periods=1).mean())
    return result.fillna(safe_number(numeric.median(), fallback))

def window_average(series: pd.Series, window: int, fallback: float = 0.0) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return safe_number(numeric.tail(min(window, len(numeric))).mean(), fallback) if len(numeric) else float(fallback)

def pace_fallback(value: Any, team_average: Any = None,
                  league_average: float = 100.0) -> float:
    for candidate in (value, team_average, league_average):
        number = safe_number(candidate)
        if 70.0 <= number <= 130.0:
            return number
    return league_average

def defense_fallback(value: Any, team_average: Any = None,
                     league_average: float = 110.0) -> float:
    for candidate in (value, team_average, league_average):
        number = safe_number(candidate)
        if 80.0 <= number <= 150.0:
            return number
    return league_average

def availability_flag(status: Any) -> float:
    """Convert normalized injury state into a projected availability fraction."""
    return {"OUT":0.0, "QUESTIONABLE":0.45, "PROBABLE":0.8,
            "ACTIVE":1.0}.get(str(status).upper(), 0.75)

def fuzzy_name_match(query: str, candidates: Sequence[T],
                     key: Callable[[T], str] | None = None,
                     cutoff: float = 0.72) -> T | None:
    """Return the best exact, partial, or fuzzy match from a provider list."""
    getter = key or (lambda item: str(item))
    target = normalize_text(query)
    if not target:
        return None
    exact = [item for item in candidates if normalize_text(getter(item)) == target]
    if len(exact) == 1:
        return exact[0]
    partial = [item for item in candidates if target in normalize_text(getter(item))]
    if len(partial) == 1:
        return partial[0]
    ranked = sorted(candidates, key=lambda item: SequenceMatcher(
        None, target, normalize_text(getter(item))).ratio(), reverse=True)
    if not ranked:
        return None
    score = SequenceMatcher(None, target, normalize_text(getter(ranked[0]))).ratio()
    return ranked[0] if score >= cutoff else None

def data_completeness(data: pd.DataFrame | dict[str, Any],
                      required: Iterable[str]) -> float:
    """Return 0-100 completeness for required fields."""
    fields = list(required)
    if not fields:
        return 100.0
    if isinstance(data, pd.DataFrame):
        present = sum(column in data and data[column].notna().mean() >= .8 for column in fields)
    else:
        present = sum(field in data and data[field] not in (None, "") for field in fields)
    return round(present / len(fields) * 100.0, 1)
