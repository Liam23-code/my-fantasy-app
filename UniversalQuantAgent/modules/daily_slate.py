"""Compose schedule, offline prop lines, and lightweight player projections."""
from __future__ import annotations
from datetime import date
from typing import Any
from modules.fusion_model import fuse_projection
from modules.data_quality import safe_number
from modules.sportsbook_parser import normalize_team_name
from modules.nba_props_loader import unified_props
from modules.nba_schedule import fetch_todays_games

def get_daily_slate(max_projections: int | None = None) -> dict[str, Any]:
    games = fetch_todays_games()
    props = unified_props()
    opponents = {}
    for game in games:
        opponents[game["home_team"]] = game["away_team"]
        opponents[game["away_team"]] = game["home_team"]
    projections, seen = [], set()
    for prop in props:
        key = (prop["player_name"], normalize_team_name(prop.get("team", "")))
        if key in seen or (max_projections is not None and len(projections) >= max_projections) or key[1] not in opponents: continue
        seen.add(key)
        opponent = opponents[key[1]]
        try:
            fusion = fuse_projection(key[0], opponent)
            context = fusion["context"]
            minutes = context["components"]["minutes"]
            pace = context["components"]["pace"]
            matchup = context["components"]["matchup"]
        except Exception:
            continue
        projections.append({"player":fusion["player"], "team":key[1],
                            "projection":fusion["final_projection"],
                            "minutes":round(safe_number(minutes.get("minutes_projection")),1),
                            "pace":round(safe_number(pace.get("pace_projection"),100),1),
                            "matchup_difficulty":round(safe_number(matchup.get("difficulty_score"),50),1)})
    clean_props = [{"player":p["player_name"], "team":normalize_team_name(p.get("team","")),
                    "category":p["category"], "line":p["line"], "sportsbook":p["sportsbook"]} for p in props]
    return {"date":date.today().isoformat(), "games":games, "player_props":clean_props,
            "projections":projections}
