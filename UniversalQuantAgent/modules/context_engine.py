"""Build a reusable, explainable NBA player context snapshot."""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from modules.data_quality import (availability_flag, coerce_numeric,
                                  normalize_columns, safe_dict, safe_get, safe_list,
                                  safe_number, safe_scalar_to_dict, window_average)
from modules.injury_parser import get_player_availability, load_injury_data_from_file
from modules.nba_cache import select_fallback_minutes
from modules.matchup_model import project_matchup_difficulty
from modules.minutes_model import project_minutes
from modules.pace_model import project_pace
from modules.projections import (_find_player, _game_log_with_fuzzy_fallback,
                                 _player_team, latest_season)

def _trend(series: pd.Series) -> dict[str, Any]:
    last5 = window_average(series, 5)
    last10 = window_average(series, 10, last5)
    change = last5 - last10
    return {"last_5":round(last5,2), "last_10":round(last10,2),
            "change":round(change,2),
            "direction":"up" if change > .5 else "down" if change < -.5 else "stable"}

def _split_average(games: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    rows = games[mask]
    return {name:round(safe_number(rows[column].mean()),1) for name, column in
            (("ppg","pts"),("rpg","reb"),("apg","ast"),("mpg","min")) if column in rows}


def _warning_list(node: Any) -> list[str]:
    """Keep only displayable warning scalars from a provider/model result."""
    candidates = safe_list(node)
    if not candidates and isinstance(node, (str, int, float)):
        candidates = [node]
    return [str(item) for item in candidates if isinstance(item, (str, int, float))]


def _model_component(node: Any, value_key: str, fallback: float) -> dict[str, Any]:
    """Normalize either a legacy model mapping or a malformed scalar result."""
    payload = safe_dict(node).copy()
    if not payload:
        payload = safe_scalar_to_dict(node)
    value = safe_number(
        safe_get(payload, value_key, safe_get(payload, "value", fallback)),
        fallback,
    )
    payload[value_key] = value
    payload["value"] = value
    payload["confidence"] = max(
        0.0, min(100.0, safe_number(safe_get(payload, "confidence"), 50.0))
    )
    payload["details"] = safe_dict(safe_get(payload, "details"))
    return payload

def get_player_context(player_name: str, opponent_team: str,
                       season: str | None = None) -> dict[str, Any]:
    """Return role, trend, schedule, injury, pace, and matchup context."""
    season = season or latest_season()
    player = _find_player(player_name)
    resolved, raw, warnings = _game_log_with_fuzzy_fallback(player_name, player, season)
    games = coerce_numeric(normalize_columns(raw)).copy()
    if "game_date" in games:
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
        games = games.sort_values("game_date")
    for column in ("pts","reb","ast","min","fga","fta","tov"):
        if column not in games: games[column] = 0.0
    minutes = games["min"].replace(0, np.nan)
    usage = (games["fga"] + .44 * games["fta"] + games["tov"]) / minutes * 40
    attempts = games["fga"] + .44 * games["fta"]
    efficiency = games["pts"] / (2 * attempts.replace(0, np.nan)) * 100
    role_stability = max(0.0, min(100.0, 100.0 - safe_number(minutes.tail(10).std(ddof=0)) / max(safe_number(minutes.tail(10).mean()), 1) * 100))
    within_two = (minutes.tail(10) - window_average(minutes, 10)).abs().le(2).mean() * 100
    coach_rotation = round((role_stability + within_two) / 2, 1)

    matchup_text = games.get("matchup", pd.Series("", index=games.index)).astype(str)
    home_away = {"home":_split_average(games, matchup_text.str.contains("vs", case=False, na=False)),
                 "away":_split_average(games, matchup_text.str.contains("@", na=False))}
    b2b_mask = pd.Series(False, index=games.index)
    if "game_date" in games:
        b2b_mask = games["game_date"].diff().dt.days.le(1).fillna(False)
    b2b_points = safe_number(games.loc[b2b_mask, "pts"].mean(), safe_number(games["pts"].mean()))
    normal_points = safe_number(games.loc[~b2b_mask, "pts"].mean(), safe_number(games["pts"].mean()))
    fatigue = round(b2b_points - normal_points, 2)

    team = _player_team(games)
    availability = get_player_availability(resolved["full_name"], load_injury_data_from_file())
    try:
        minute_model = project_minutes(
            resolved["full_name"], opponent_team, season
        )
    except Exception as error:
        season_minutes = safe_number(minutes.mean(), np.nan)
        last_10_minutes = window_average(minutes, 10, np.nan)
        last_5_minutes = window_average(minutes, 5, np.nan)
        selected_minutes, selected_source = select_fallback_minutes(
            season_minutes, last_10_minutes, last_5_minutes
        )
        minute_model = {
            "minutes_projection": selected_minutes,
            "season_average_minutes": season_minutes,
            "rolling_minutes": {
                "last_10": last_10_minutes,
                "last_5": last_5_minutes,
            },
            "fallback_minutes": selected_minutes,
            "fallback_minutes_source": selected_source,
            "fallback_minutes_quality": {
                "season_average": 92.0,
                "last_10": 76.0,
                "last_5": 62.0,
            }[selected_source],
            "blowout_risk": 35.0,
            "confidence_low": selected_minutes,
            "confidence_high": selected_minutes,
            "warnings": [str(error)],
        }
    try: pace_model = project_pace(team, opponent_team, season)
    except Exception as error:
        pace_model = {"pace_projection":100.0, "recent_pace":100.0, "confidence":50.0,
                      "explanation":"League pace fallback.", "warnings":[str(error)]}
    try: matchup_model = project_matchup_difficulty(resolved["full_name"], opponent_team, season)
    except Exception as error:
        matchup_model = {"difficulty_score":50.0, "matchup_history_points":None,
                         "explanation":"Neutral matchup fallback.", "warnings":[str(error)]}

    minute_model = _model_component(
        minute_model, "minutes_projection", window_average(minutes, 10)
    )
    pace_model = _model_component(pace_model, "pace_projection", 100.0)
    matchup_model = _model_component(matchup_model, "difficulty_score", 50.0)
    availability_payload = safe_dict(availability).copy()
    if not availability_payload:
        availability_payload = safe_scalar_to_dict(availability)
    scalar_status = safe_get(availability_payload, "value")
    availability_payload.setdefault(
        "availability", scalar_status if isinstance(scalar_status, str) else "ACTIVE"
    )
    status = safe_get(availability_payload, "availability", "ACTIVE")
    availability_payload["availability"] = (
        str(status).upper() if isinstance(status, str) else "ACTIVE"
    )
    availability_payload["impact_score"] = safe_number(
        safe_get(availability_payload, "impact_score")
    )
    availability = availability_payload
    minute_model.setdefault("blowout_risk",35.0)
    pace_model.setdefault("recent_pace",100.0)
    context_confidence = round(
        (
            safe_number(safe_get(minute_model, "confidence"), 50.0)
            + safe_number(safe_get(pace_model, "confidence"), 50.0)
            + safe_number(safe_get(matchup_model, "confidence"), 50.0)
        )
        / 3.0,
        1,
    )
    return {"value":round(role_stability,1), "confidence":context_confidence,
            "details":{"games":len(games), "component_confidence":{
                "minutes":safe_get(minute_model, "confidence", 50.0),
                "pace":safe_get(pace_model, "confidence", 50.0),
                "matchup":safe_get(matchup_model, "confidence", 50.0)}},
            "player":resolved["full_name"], "team":team, "opponent":opponent_team,
            "season":season, "games":len(games), "role_stability":round(role_stability,1),
            "usage_trend":_trend(usage), "minutes_trend":_trend(minutes),
            "efficiency_trend":_trend(efficiency),
            "pace_trend":{"projected":pace_model.get("pace_projection",100),
                          "recent":pace_model.get("recent_pace",100),
                          "change":round(safe_number(pace_model.get("pace_projection"),100)-safe_number(pace_model.get("recent_pace"),100),2)},
            "opponent_matchup_trend":{"difficulty_score":matchup_model.get("difficulty_score",50),
                                      "historical_points":matchup_model.get("matchup_history_points"),
                                      "explanation":matchup_model.get("explanation","")},
            "injury_impact":{"status":availability["availability"],
                             "impact_score":availability["impact_score"],
                             "availability_flag":availability_flag(availability["availability"]),
                             "warning":availability.get("warning","")},
            "blowout_risk":minute_model.get("blowout_risk",35),
            "home_away_splits":home_away,
            "back_to_back_fatigue":{"b2b_games":int(b2b_mask.sum()),
                                    "points_difference":fatigue,
                                    "interpretation":"negative means lower scoring on back-to-backs"},
            "coach_rotation_tendencies":{"stability_score":coach_rotation,
                                         "last_10_within_two_minutes_pct":round(within_two,1)},
            "components":{"minutes":minute_model, "pace":pace_model,
                          "matchup":matchup_model, "availability":availability},
            "warnings":list(dict.fromkeys(
                _warning_list(warnings)
                + _warning_list(safe_get(minute_model, "warnings"))
                + _warning_list(safe_get(pace_model, "warnings"))
                + _warning_list(safe_get(matchup_model, "warnings"))
            ))}
