"""NFL game-environment analysis for a compact weekly slate."""
from __future__ import annotations

from datetime import date
from typing import Any

from modules.data_quality import safe_number
from modules.nfl import compare_nfl_teams, latest_completed_nfl_season, lookup_team


DEFAULT_GAMES=[("BUF","KC"),("PHI","DAL"),("BAL","CIN"),("SF","LA")]


def _identity(pace: float, explosive: float, fantasy: float) -> str:
    if pace>=70 and explosive>=65: return "High-paced shootout with strong explosive play potential."
    if fantasy>=65: return "Efficient scoring environment with multiple red-zone paths."
    if pace<45: return "Slower, matchup-driven environment where volume concentration matters."
    return "Balanced game environment with moderate pace and scoring potential."


def analyze_nfl_slate(games: list[dict[str,str]]|None=None,season: int|None=None) -> dict[str,Any]:
    """Analyze supplied games, degrading gracefully when live data is offline."""
    season=season or latest_completed_nfl_season(); pairs=[]
    for game in games or [{"away_team":a,"home_team":h} for a,h in DEFAULT_GAMES]:
        pairs.append((lookup_team(game["away_team"])["abbreviation"],lookup_team(game["home_team"])["abbreviation"]))
    results=[]; warnings=[]
    for away,home in pairs:
        try:
            matchup=compare_nfl_teams(away,home,season); teams=matchup["teams"]
            pace_values=[safe_number(t["metrics"].get("pace_seconds_per_play"),28.5) for t in teams]
            pace=max(0,min(100,100-(sum(pace_values)/2-24)*8)); net=[safe_number(t["metrics"].get("net_efficiency")) for t in teams]
            explosive=max(0,min(100,50+sum(net)*.35)); line_mismatch=abs(safe_number(teams[0]["metrics"].get("offensive_efficiency_score"))-safe_number(teams[1]["metrics"].get("defensive_efficiency_score")))
            fantasy=max(0,min(100,pace*.45+explosive*.4+(100-matchup["matchup"]["difficulty_score"])*.15))
            weather=92.0; red_zone=max(0,min(100,sum(safe_number(t["metrics"].get("red_zone_efficiency"),50) for t in teams)/2))
        except Exception:
            warnings.append(f"{away}-{home} uses neutral offline game context."); pace=55.0; explosive=52.0; line_mismatch=8.0; fantasy=54.0; weather=95.0; red_zone=52.0
        results.append({"away_team":away,"home_team":home,"pace_projection":round(pace,1),"matchup_difficulty":round(100-abs(fantasy-50),1),"weather_impact":weather,"ol_dl_mismatch":round(line_mismatch,1),"explosive_play_probability":round(explosive,1),"red_zone_funnel_probability":round(red_zone,1),"fantasy_scoring_environment":round(fantasy,1),"identity_summary":_identity(pace,explosive,fantasy)})
    return {"domain":"nfl_slate","date":date.today().isoformat(),"season":season,"games":results,"warnings":list(dict.fromkeys(warnings))}
