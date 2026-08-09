"""NBA prop comparison and recommendation engine.

The engine composes the independent projection, minutes, pace, matchup, injury,
and sportsbook modules.  Keeping those boundaries makes every component easy to
replace with a stronger model later.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable
import numpy as np
import pandas as pd

from modules.fusion_model import fuse_projection
from modules.reliability import get_reliability_score
from modules.data_quality import as_dict, safe_number
from modules.projections import (_find_player, _game_log_with_fuzzy_fallback, _player_team,
                                 latest_season, normalize_columns, project_player_statline)
from modules.sportsbook import (fetch_all_sportsbook_props, fetch_daily_games,
                                normalize_category, normalize_player_name,
                                normalize_team_name)

VALID_CATEGORIES = ("points", "rebounds", "assists", "PRA", "3PM")

def _categories(values: Iterable[str]) -> list[str]:
    normalized = [normalize_category(value) for value in values]
    return list(dict.fromkeys(value for value in normalized if value in VALID_CATEGORIES))

def _player_team_name(player_name: str, season: str) -> str:
    try:
        player = _find_player(player_name)
        _, games, _ = _game_log_with_fuzzy_fallback(player_name, player, season)
        return normalize_team_name(_player_team(normalize_columns(games)))
    except Exception:
        return ""

def _three_projection(player_name: str, season: str) -> tuple[float, dict, float]:
    """Use a transparent recent-average projection because V1 has no 3PM model."""
    player = _find_player(player_name)
    _, raw, _ = _game_log_with_fuzzy_fallback(player_name, player, season)
    games = normalize_columns(raw)
    values = pd.to_numeric(games.get("fg3m", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty: return 0.0, {"low":0.0, "high":0.0}, 0.0
    recent = float(values.tail(min(10, len(values))).mean())
    spread = max(0.5, float(values.tail(10).std(ddof=0) or .5))
    return round(recent,1), {"low":round(max(0,recent-1.645*spread),1), "high":round(recent+1.645*spread,1)}, recent

def _best_book(lines: list[dict], projection: float) -> str:
    median = float(np.median([line["line"] for line in lines]))
    chosen = min(lines, key=lambda row: row["line"]) if projection >= median else max(lines, key=lambda row: row["line"])
    return chosen["sportsbook"]

def compare_props(player_name: str, opponent_team: str, categories: Iterable[str],
                  season: str | None = None) -> list[dict[str, Any]]:
    """Compare reliability-scored fused projections with every verified line."""
    season = season or latest_season()
    player_key = normalize_player_name(player_name).lower()
    wanted = _categories(categories)
    if not wanted:
        raise ValueError("Choose at least one supported prop category.")
    lines = [row for row in fetch_all_sportsbook_props()
             if normalize_player_name(row.get("player_name","")).lower() == player_key
             and normalize_category(row.get("category","")) in wanted]
    if not lines:
        return []

    fusion = fuse_projection(player_name, opponent_team, season)
    reliability = get_reliability_score(player_name, opponent_team, season,
                                        fusion_result=fusion)
    base = fusion["base_projection"]
    context = fusion["context"]
    team = next((normalize_team_name(row.get("team","")) for row in lines if row.get("team")),"")
    team = team or fusion.get("team") or _player_team_name(player_name, season)
    by_category: dict[str,list[dict]] = defaultdict(list)
    for line in lines:
        by_category[normalize_category(line["category"])].append(line)
    output = []
    for category, category_lines in by_category.items():
        key = category.lower()
        if category == "3PM":
            raw_three, raw_range, _ = _three_projection(player_name, season)
            minute_data = as_dict(as_dict(context.get("components")).get("minutes"))
            rolling = as_dict(minute_data.get("rolling_minutes"))
            baseline = safe_number(rolling.get("last_10"))
            if baseline <= 0:
                baseline = safe_number(as_dict(context.get("minutes_trend")).get("last_10"),24)
            baseline = max(baseline,1)
            factor = safe_number(minute_data.get("minutes_projection"),baseline)/baseline
            projection = max(0.0, raw_three*max(.7,min(1.25,factor)))
            confidence = raw_range
        else:
            projection = safe_number(fusion["final_projection"].get(key))
            confidence = fusion["confidence_range"].get(key,{"low":projection*.8,"high":projection*1.2})
        best = _best_book(category_lines, projection)
        drivers = list(fusion.get("key_drivers",[]))
        drivers.append(context["opponent_matchup_trend"]["explanation"])
        drivers.append(f"Quant reliability: {reliability['score']:.1f}/100 ({reliability['rating']}).")
        if context["injury_impact"].get("warning"):
            drivers.append(context["injury_impact"]["warning"])
        for line in category_lines:
            edge = projection-float(line["line"])
            output.append({"player":base["player"], "team":team, "category":category,
                           "projection":round(projection,1),
                           "minutes_adjusted_projection":round(projection,1),
                           "sportsbook_line":round(float(line["line"]),1),
                           "edge":round(edge,1),
                           "confidence_low":round(safe_number(confidence.get("low")),1),
                           "confidence_high":round(safe_number(confidence.get("high")),1),
                           "best_sportsbook":best,
                           "lean":"over" if projection >= float(line["line"]) else "under",
                           "key_drivers":list(dict.fromkeys(drivers)),
                           "sportsbook":line["sportsbook"], "timestamp":line.get("timestamp","")})
    return sorted(output,key=lambda row:abs(row["edge"]),reverse=True)

def _opponent_map(games: list[dict]) -> dict[str, str]:
    result = {}
    for game in games:
        result[game["home_team"]] = game["away_team"]
        result[game["away_team"]] = game["home_team"]
    return result

def prop_confidence(prop: dict) -> float:
    width = max(0.0, float(prop["confidence_high"]) - float(prop["confidence_low"]))
    center = max(abs(float(prop["minutes_adjusted_projection"])), 1.0)
    return round(max(0.0, min(100.0, 100.0 - width / center * 55.0)), 1)

def recommend_props(categories: Iterable[str] = VALID_CATEGORIES, min_edge: float = 1.5,
                    min_confidence: float = 55.0, season: str | None = None,
                    max_players: int = 25, min_reliability: float = 0.0
                    ) -> list[dict[str, Any]]:
    """Backward-compatible delegate to the reliability-aware recommendation module."""
    from modules.recommendations import recommend_props as build_recommendations
    return build_recommendations(categories, min_edge, min_confidence, season,
                                 max_players, min_reliability)