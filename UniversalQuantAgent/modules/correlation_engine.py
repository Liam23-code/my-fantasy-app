"""Correlation matrices for NBA players, teams, and daily prop slates.

The engine returns plain dictionaries instead of chart objects. Streamlit,
future APIs, and notebooks can therefore choose their own visualization layer.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from nba_api.stats.static import players

from modules.data_quality import (
    coerce_numeric,
    fuzzy_name_match,
    normalize_columns,
    safe_dict,
    safe_get,
    safe_list,
    safe_number,
)
from modules.nba_advanced import latest_season
from modules.projections import (
    _find_player,
    _game_log_with_fuzzy_fallback,
    build_projection_features,
    fetch_team_context,
)
from modules.sports import _fetch_team_tables, _team_lookup


PLAYER_COLUMNS = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "usage": "usage_rate",
    "minutes": "minutes",
    "ts_pct": "true_shooting_pct",
    "pace": "pace",
    "opponent_difficulty": "opponent_difficulty",
}
TEAM_COLUMNS = {
    "pace": "pace",
    "offensive_rating": "off_rating",
    "defensive_rating": "def_rating",
    "net_rating": "net_rating",
    "scoring": "pts",
    "assists": "ast",
    "rebounds": "reb",
}
SLATE_COLUMNS = ("pra", "points", "rebounds", "assists", "fantasy", "minutes")


def _matrix_payload(
    subject: str,
    frame: pd.DataFrame,
    requested: Iterable[str],
    warnings: list[str] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a finite correlation matrix from every usable requested column."""
    data = frame.copy()
    available: list[str] = []
    dropped: list[str] = []
    for column in requested:
        if column not in data:
            dropped.append(column)
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].notna().sum() >= 2 and data[column].nunique(dropna=True) > 1:
            available.append(column)
        else:
            dropped.append(column)

    notes = list(warnings or [])
    if dropped:
        notes.append(
            "Unavailable or constant statistics were omitted: "
            + ", ".join(dropped)
            + "."
        )
    if not available:
        return {
            "subject": subject,
            "observations": int(len(data)),
            "available_stats": [],
            "correlation_matrix": {},
            "records": records or [],
            "warnings": list(dict.fromkeys(notes + ["No variable statistics were available."])),
        }

    clean = data[available].replace([np.inf, -np.inf], np.nan)
    matrix = clean.corr(min_periods=2).fillna(0.0).clip(-1.0, 1.0)
    for column in available:
        matrix.loc[column, column] = 1.0
    return {
        "subject": subject,
        "observations": int(len(data)),
        "available_stats": available,
        "correlation_matrix": {
            row: {column: round(safe_number(value), 3) for column, value in values.items()}
            for row, values in matrix.to_dict(orient="index").items()
        },
        "records": records or [],
        "warnings": list(dict.fromkeys(notes)),
    }


def _resolve_player(player_id_or_name: int | str) -> dict[str, Any]:
    if isinstance(player_id_or_name, int) or str(player_id_or_name).isdigit():
        player_id = int(player_id_or_name)
        match = next(
            (player for player in players.get_players() if int(player["id"]) == player_id),
            None,
        )
        if match:
            return match
        raise ValueError(f"Unknown NBA player id: {player_id}.")
    return _find_player(str(player_id_or_name))


def compute_player_correlations(
    player_id_or_name: int | str,
    season: str | None = None,
) -> dict[str, Any]:
    """Correlate one player's game-level production and matchup context."""
    season = season or latest_season()
    primary = _resolve_player(player_id_or_name)
    player, raw_games, warnings = _game_log_with_fuzzy_fallback(
        primary["full_name"], primary, season
    )
    team_context = fetch_team_context(season)
    features = build_projection_features(raw_games, team_context, season)
    data = features.copy()
    defense = pd.to_numeric(
        data.get("opponent_defensive_rating", pd.Series(index=data.index, dtype=float)),
        errors="coerce",
    )
    if defense.notna().any():
        data["opponent_difficulty"] = (1.0 - defense.rank(pct=True)) * 100.0
    else:
        data["opponent_difficulty"] = 50.0

    renamed = pd.DataFrame(index=data.index)
    for label, source in PLAYER_COLUMNS.items():
        renamed[label] = pd.to_numeric(data.get(source), errors="coerce")
    if "game_date" in data:
        dates = pd.to_datetime(data["game_date"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=data.index)
    records = []
    for position, (_, row) in enumerate(renamed.tail(40).iterrows(), start=1):
        record = {"game": position}
        if position - 1 < len(dates.tail(40)):
            value = dates.tail(40).iloc[position - 1]
            record["date"] = value.date().isoformat() if pd.notna(value) else ""
        record.update(
            {key: round(safe_number(value), 2) for key, value in row.items()}
        )
        records.append(record)
    warnings.extend(safe_list(features.attrs.get("warnings")))
    return _matrix_payload(
        player["full_name"],
        renamed,
        PLAYER_COLUMNS,
        warnings,
        records,
    )


def compute_team_correlations(
    team: str,
    season: str | None = None,
) -> dict[str, Any]:
    """Correlate league-wide team profiles and identify the requested team."""
    season = season or latest_season()
    selected = _team_lookup(team)
    base, advanced = _fetch_team_tables(season)
    base = normalize_columns(base)
    advanced = normalize_columns(advanced)
    if base.empty and advanced.empty:
        return _matrix_payload(
            selected["full_name"], pd.DataFrame(), TEAM_COLUMNS,
            ["No NBA team statistics were returned."],
        )
    if base.empty:
        merged = advanced.copy()
    elif advanced.empty:
        merged = base.copy()
    else:
        advanced_extra = [
            column for column in advanced.columns
            if column == "team_id" or column not in base.columns
        ]
        merged = base.merge(advanced[advanced_extra], on="team_id", how="outer")
    data = coerce_numeric(merged)
    renamed = pd.DataFrame(index=data.index)
    for label, source in TEAM_COLUMNS.items():
        renamed[label] = pd.to_numeric(data.get(source), errors="coerce")

    profile = {}
    if "team_id" in data:
        rows = data[data["team_id"] == selected["id"]]
        if not rows.empty:
            row = rows.iloc[0]
            profile = {
                label: round(safe_number(safe_get(row.to_dict(), source)), 2)
                for label, source in TEAM_COLUMNS.items()
            }
    payload = _matrix_payload(
        selected["full_name"], renamed, TEAM_COLUMNS, records=[]
    )
    payload["team"] = selected["abbreviation"]
    payload["team_profile"] = profile
    payload["season"] = season
    return payload


def _category_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    aliases = {
        "pts": "points",
        "point": "points",
        "reb": "rebounds",
        "rebs": "rebounds",
        "ast": "assists",
        "asts": "assists",
        "points rebounds assists": "pra",
        "fantasy points": "fantasy",
        "fantasy score": "fantasy",
        "mins": "minutes",
    }
    return aliases.get(text, text)


def _slate_rows(slate_props: Any) -> pd.DataFrame:
    """Pivot mixed line, projection, or slate records into one row per player."""
    buckets: dict[str, dict[str, Any]] = {}
    for raw in safe_list(slate_props):
        row = safe_dict(raw)
        player = str(
            safe_get(row, "player", safe_get(row, "player_name", ""))
        ).strip()
        if not player:
            continue
        bucket = buckets.setdefault(player, {"player": player})
        projection = safe_get(row, "projection")
        if isinstance(projection, dict):
            for category, value in projection.items():
                normalized = _category_name(category)
                if normalized in SLATE_COLUMNS:
                    bucket[normalized] = safe_number(value)
        category = _category_name(safe_get(row, "category"))
        if category in SLATE_COLUMNS:
            candidate = safe_get(
                row,
                "minutes_adjusted_projection",
                safe_get(row, "projection", safe_get(row, "line")),
            )
            if not isinstance(candidate, dict):
                bucket[category] = safe_number(candidate)
        minutes = safe_get(row, "minutes")
        if minutes is not None:
            bucket["minutes"] = safe_number(minutes)

    records = [
        row for row in buckets.values()
        if any(category in row for category in SLATE_COLUMNS)
    ]
    for row in records:
        if not row.get("pra") and all(key in row for key in ("points", "rebounds", "assists")):
            row["pra"] = row["points"] + row["rebounds"] + row["assists"]
        if not row.get("fantasy") and all(key in row for key in ("points", "rebounds", "assists")):
            row["fantasy"] = (
                row["points"] + 1.2 * row["rebounds"] + 1.5 * row["assists"]
            )
    return pd.DataFrame(records)


def compute_slate_correlations(slate_props: Any) -> dict[str, Any]:
    """Correlate projected stat categories across every player on a slate."""
    data = _slate_rows(slate_props)
    subject = "Current slate"
    records = data.replace({np.nan: None}).to_dict(orient="records") if not data.empty else []
    return _matrix_payload(
        subject,
        data,
        SLATE_COLUMNS,
        records=records,
        warnings=[] if len(data) >= 3 else [
            "Slate correlations are more stable with at least three projected players."
        ],
    )