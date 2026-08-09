"""Explainable NBA player similarity using normalized season profiles."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from nba_api.stats.static import players

from modules.data_quality import fuzzy_name_match, safe_number
from modules.nba_advanced import fetch_player_tables, latest_season


DIMENSIONS = {
    "usage": ["USG_PCT"],
    "minutes": ["MIN"],
    "efficiency": ["TS_PCT", "PER_ESTIMATE", "PIE"],
    "pace_context": ["PACE"],
    "role": ["USG_PCT", "AST_PCT", "REB_PCT"],
    "scoring_profile": ["PTS", "FGA", "FG3A", "FTA"],
    "rebounding_profile": ["REB", "OREB", "DREB", "REB_PCT"],
    "assisting_profile": ["AST", "AST_PCT", "AST_TO"],
    "defensive_profile": ["STL", "BLK", "DREB", "DEF_RATING"],
    "recent_form": ["RECENT_PTS", "RECENT_REB", "RECENT_AST", "RECENT_USG_PCT"],
    "opponent_matchup_profile": ["PACE", "DEF_RATING", "NET_RATING"],
}


def _per_estimate(table: pd.DataFrame) -> pd.Series:
    def column(name: str) -> pd.Series:
        return pd.to_numeric(
            table.get(name, pd.Series(0.0, index=table.index)), errors="coerce"
        ).fillna(0.0)

    minutes = column("MIN").replace(0, np.nan)
    positives = (
        column("PTS") + column("REB") + column("AST") + column("STL") + column("BLK")
    )
    negatives = (
        column("FGA") - column("FGM") + column("FTA") - column("FTM") + column("TOV")
    )
    return ((positives - negatives) / minutes * 15.0).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


def _merge_recent(season_table: pd.DataFrame, recent_table: pd.DataFrame) -> pd.DataFrame:
    data = season_table.copy()
    if recent_table.empty or "PLAYER_ID" not in recent_table:
        for column in ("PTS", "REB", "AST", "USG_PCT"):
            data[f"RECENT_{column}"] = pd.to_numeric(
                data.get(column, 0.0), errors="coerce"
            )
        return data
    columns = [
        column for column in ("PLAYER_ID", "PTS", "REB", "AST", "USG_PCT")
        if column in recent_table
    ]
    recent = recent_table[columns].copy().rename(
        columns={column: f"RECENT_{column}" for column in columns if column != "PLAYER_ID"}
    )
    data = data.merge(recent, on="PLAYER_ID", how="left")
    for column in ("PTS", "REB", "AST", "USG_PCT"):
        recent_name = f"RECENT_{column}"
        if recent_name not in data:
            data[recent_name] = data.get(column, 0.0)
        else:
            data[recent_name] = data[recent_name].fillna(data.get(column, 0.0))
    return data


def _resolve_known_player(target_player: str) -> dict[str, Any] | None:
    return fuzzy_name_match(
        target_player,
        players.get_players(),
        key=lambda player: player["full_name"],
        cutoff=.72,
    )


def _role(row: pd.Series, usage_median: float) -> str:
    assists = safe_number(row.get("AST"))
    rebounds = safe_number(row.get("REB"))
    points = safe_number(row.get("PTS"))
    usage = safe_number(row.get("USG_PCT"))
    if assists >= 6:
        return "primary creator"
    if rebounds >= 8:
        return "interior"
    if points >= 20 or usage >= usage_median:
        return "scorer"
    return "connector"


def _numeric_series(
    table: pd.DataFrame, column: str, default: float = 0.0
) -> pd.Series:
    source = (
        table[column]
        if column in table
        else pd.Series(default, index=table.index, dtype=float)
    )
    return pd.to_numeric(source, errors="coerce").fillna(default)


def _minmax(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = pd.DataFrame(index=table.index)
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        median = safe_number(values.median())
        values = values.fillna(median)
        low, high = safe_number(values.min()), safe_number(values.max())
        if high - low <= 1e-9:
            normalized[column] = .5
        else:
            normalized[column] = (values - low) / (high - low)
    return normalized


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(first, second) / denominator)))


def _dimension_similarity(
    target: pd.Series,
    candidate: pd.Series,
    features: list[str],
) -> float:
    if not features:
        return 0.0
    first = target[features].to_numpy(dtype=float)
    second = candidate[features].to_numpy(dtype=float)
    if len(features) == 1:
        return max(0.0, min(100.0, (1.0 - abs(first[0] - second[0])) * 100.0))
    return _cosine(first, second) * 100.0


def compute_player_similarity(
    target_player: str,
    season: str | None = None,
    limit: int = 10,
    player_table: pd.DataFrame | None = None,
    recent_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the most similar NBA season profiles using cosine similarity."""
    season = season or latest_season()
    known = _resolve_known_player(target_player)
    canonical_name = known["full_name"] if known else str(target_player).strip()
    warnings: list[str] = []

    table = (
        player_table.copy()
        if player_table is not None
        else fetch_player_tables(season)
    )
    if table.empty or "PLAYER_NAME" not in table:
        return {
            "target": {"player": canonical_name, "team": "", "player_id": known.get("id") if known else None},
            "season": season,
            "dimensions": [],
            "similar_players": [],
            "warnings": ["No season player statistics were available."],
        }

    recent = (
        recent_table.copy()
        if recent_table is not None
        else fetch_player_tables(season, last_n_games=10)
    )
    data = _merge_recent(table, recent)
    data["PER_ESTIMATE"] = _per_estimate(data)
    names = data["PLAYER_NAME"].dropna().astype(str).tolist()
    matched_name = fuzzy_name_match(canonical_name, names, cutoff=.72)
    if not matched_name:
        return {
            "target": {
                "player": canonical_name,
                "team": "",
                "player_id": known.get("id") if known else None,
            },
            "season": season,
            "dimensions": [],
            "similar_players": [],
            "warnings": [
                f"{canonical_name} has no qualifying statistics for {season}."
            ],
        }

    # Keep rotation players for useful comparisons, but always retain the target.
    minutes = _numeric_series(data, "MIN")
    games = _numeric_series(data, "GP", 5.0)
    target_mask = data["PLAYER_NAME"].astype(str).str.lower() == matched_name.lower()
    qualified = data[(minutes >= 8) & (games >= 5) | target_mask].copy()
    target_rows = qualified[
        qualified["PLAYER_NAME"].astype(str).str.lower() == matched_name.lower()
    ]
    if target_rows.empty:
        target_rows = data[target_mask].copy()
        qualified = pd.concat([qualified, target_rows]).drop_duplicates("PLAYER_ID")
    target_index = target_rows.index[0]

    feature_groups = {
        name: [column for column in columns if column in qualified]
        for name, columns in DIMENSIONS.items()
    }
    feature_groups = {
        name: columns for name, columns in feature_groups.items() if columns
    }
    features = list(dict.fromkeys(
        column for columns in feature_groups.values() for column in columns
    ))
    usable = []
    for column in features:
        numeric = pd.to_numeric(qualified[column], errors="coerce")
        if numeric.notna().sum() >= 2 and numeric.nunique(dropna=True) > 1:
            usable.append(column)
    if not usable:
        return {
            "target": {
                "player": matched_name,
                "team": str(target_rows.iloc[0].get("TEAM_ABBREVIATION", "")),
                "player_id": int(safe_number(target_rows.iloc[0].get("PLAYER_ID"))),
            },
            "season": season,
            "dimensions": [],
            "similar_players": [],
            "warnings": ["The available player statistics had no usable variation."],
        }

    normalized = _minmax(qualified, usable)
    target_position = qualified.index.get_loc(target_index)
    target_vector = normalized.iloc[target_position].to_numpy(dtype=float)
    scores = []
    usage_median = safe_number(_numeric_series(qualified, "USG_PCT").median())
    for position, (_, row) in enumerate(qualified.iterrows()):
        if position == target_position:
            continue
        similarity = _cosine(
            target_vector, normalized.iloc[position].to_numpy(dtype=float)
        ) * 100.0
        breakdown = {}
        for dimension, columns in feature_groups.items():
            available = [column for column in columns if column in usable]
            if available:
                breakdown[dimension] = round(
                    _dimension_similarity(
                        normalized.iloc[target_position],
                        normalized.iloc[position],
                        available,
                    ),
                    1,
                )
        scores.append(
            {
                "player": str(row.get("PLAYER_NAME", "Unknown")),
                "team": str(row.get("TEAM_ABBREVIATION", "")),
                "player_id": int(safe_number(row.get("PLAYER_ID"))),
                "role": _role(row, usage_median),
                "similarity_score": round(similarity, 1),
                "dimension_scores": breakdown,
            }
        )
    scores.sort(key=lambda row: row["similarity_score"], reverse=True)
    target_row = qualified.iloc[target_position]
    if len(scores) < limit:
        warnings.append(
            f"Only {len(scores)} qualified comparison players were available."
        )
    return {
        "target": {
            "player": str(target_row.get("PLAYER_NAME", matched_name)),
            "team": str(target_row.get("TEAM_ABBREVIATION", "")),
            "player_id": int(safe_number(target_row.get("PLAYER_ID"))),
            "role": _role(target_row, usage_median),
        },
        "season": season,
        "dimensions": list(feature_groups),
        "features_used": usable,
        "similar_players": scores[: max(1, int(limit))],
        "warnings": warnings,
        "method": "Min-max normalized feature vectors with cosine similarity.",
    }