"""Explainable NFL player context, identity, and matchup analysis."""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import safe_number
from modules.nfl import get_team_stats
from modules.nfl_stats import canonical_player_stats, normalized_profile, resolve_player


def _tier(value: float) -> str:
    return "Elite" if value >= 90 else "Strong" if value >= 80 else "Average" if value >= 65 else "Weak"


def _weather_context(raw: dict[str, Any]) -> dict[str, Any]:
    temperature=safe_number(raw.get("temperature", raw.get("temp", 60)),60)
    wind=safe_number(raw.get("wind", 8),8); precipitation=safe_number(raw.get("precipitation",0))
    impact=max(0.72,1.0-max(wind-12,0)*.008-max(precipitation,0)*.004-max(32-temperature,0)*.003)
    return {"temperature":temperature,"wind_mph":wind,"precipitation_pct":precipitation,"factor":round(impact,3),"source":"season context or neutral fallback"}


def _position_efficiency(position: str, raw: dict[str, Any]) -> dict[str, float]:
    """Return only useful per-snap, per-route, or per-play adjustments."""
    if position == "QB":
        return {"epa_per_play": safe_number(raw.get("epa_per_play")), "cpoe": safe_number(raw.get("cpoe"))}
    if position == "RB":
        yards = safe_number(raw.get("rush_yards")) + safe_number(raw.get("rec_yards"))
        touches = safe_number(raw.get("rush_attempts")) + safe_number(raw.get("receptions"))
        return {
            "yards_per_snap": yards / max(safe_number(raw.get("plays")), 1.0),
            "yards_per_touch": yards / max(touches, 1.0),
        }
    if position in {"WR", "TE"}:
        return {"yards_per_route_run": safe_number(raw.get("yards_per_route_run"))}
    return {"epa_allowed_per_play": safe_number(raw.get("epa_allowed"))}


def _identity(position: str, attrs: list[dict[str, Any]], raw: dict[str, Any]) -> str:
    scores={item["metric"]:safe_number(item.get("badge_value",item["blended_percentile"])) for item in attrs}
    if position=="QB":
        style="Efficient" if scores.get("epa_per_play",50)>=75 else "Aggressive" if scores.get("explosive_pass_rate",50)>=75 else "Steady"
        risk="low turnover risk" if scores.get("turnover_worthy_play_rate",50)>=70 else "volatile decision profile"
        return f"{style} QB with {risk} and a {('strong' if scores.get('cpoe',50)>=70 else 'developing')} accuracy profile."
    if position=="DEF":
        pressure="disruptive" if scores.get("pressure_rate",50)>=75 else "coverage-led"
        return f"{pressure.title()} defense with {('strong' if scores.get('run_stop_win_rate',50)>=70 else 'average')} run control and a {('low' if scores.get('explosive_plays_allowed',50)>=70 else 'volatile')} explosive-play allowance."
    if position=="RB":
        volume="High-volume RB1" if safe_number(raw.get("snap_share"))>=70 else "Change-of-pace back"
        trait="elite contact balance" if scores.get("yards_after_contact",50)>=85 else "balanced rushing efficiency"
        return f"{volume} with {trait} and {('strong' if scores.get('red_zone_share',50)>=70 else 'limited')} red-zone usage."
    style="Explosive" if scores.get("explosive_play_rate",50)>=80 else "Volume-driven"
    depth="deep target profile" if safe_number(raw.get("average_depth_of_target"))>=12 else "intermediate target profile"
    return f"{style} {position} with a {depth} and {('high' if scores.get('target_share',50)>=80 else 'moderate')} weekly role."


def analyze_nfl_player(
    player_name: str,
    opponent_team: str | None = None,
    season: int | None = None,
    mode: str = "Adjusted",
    comparison_mode: str = "League",
) -> dict[str, Any]:
    """Return a premium, plain-dictionary NFL player analysis."""
    profile=normalized_profile(player_name,season,comparison_mode,mode)
    raw=canonical_player_stats(profile["raw"]); attrs=list(profile["attributes"]); position=profile["position"]
    ranked=sorted(attrs,key=lambda item:safe_number(item.get("badge_value",item["blended_percentile"])),reverse=True)
    labels=[{"metric":item["metric"],"tier":_tier(safe_number(item.get("badge_value",item["blended_percentile"]))),"percentile":item.get("badge_value",item["blended_percentile"])} for item in ranked]
    games=max(safe_number(raw.get("games")),1); snap=safe_number(raw.get("snap_share"),65)
    if position=="QB": volume_total=safe_number(raw.get("pass_attempts"))+safe_number(raw.get("rush_attempts"))
    elif position=="RB": volume_total=safe_number(raw.get("rush_attempts"))+safe_number(raw.get("targets"))
    elif position in {"WR","TE"}: volume_total=safe_number(raw.get("targets"))
    else: volume_total=safe_number(raw.get("plays"))
    volume=volume_total/games
    role_stability=max(20.0,min(96.0,55+snap*.4-min(volume,35)*.15))
    explosive_metric="explosive_pass_rate" if position=="QB" else "explosive_run_rate" if position=="RB" else "explosive_play_rate"
    explosive=safe_number(raw.get(explosive_metric))
    adot=safe_number(raw.get("average_depth_of_target"),8)
    volatility=max(5.0,min(95.0,35+explosive*1.8+max(adot-8,0)*2-(role_stability-50)*.35))
    opponent={"team":opponent_team or "","difficulty":50.0,"epa_allowed":0.0,"pressure_rate":25.0,"coverage_grade":50.0,"source":"neutral"}
    if opponent_team:
        try:
            defense=get_team_stats(opponent_team,profile["season"]); metrics=defense["metrics"]
            opponent.update({"team":defense["abbreviation"],"difficulty":round(safe_number(metrics.get("defensive_efficiency_score"),50),1),"epa_allowed":safe_number(metrics.get("defensive_epa_per_play")),"pressure_rate":safe_number(metrics.get("pressure_rate"),25),"coverage_grade":safe_number(metrics.get("defensive_efficiency_score"),50),"source":"nflverse team table"})
        except Exception:
            profile["warnings"].append("Opponent context unavailable; neutral matchup values were used.")
    pace={"team_seconds_per_play":safe_number(raw.get("team_seconds_per_play"),28.5),"opponent_seconds_per_play":safe_number(raw.get("opponent_seconds_per_play"),28.5)}
    pace["projected_seconds_per_play"]=round((pace["team_seconds_per_play"]+pace["opponent_seconds_per_play"])/2,2)
    weather=_weather_context(raw)
    line={"pass_block_win_rate":safe_number(raw.get("ol_pass_block_win_rate"),63),"run_block_win_rate":safe_number(raw.get("ol_run_block_win_rate"),70),"source":"player/team context or league fallback"}
    adjustment=1.0
    if mode=="Adjusted": adjustment*=weather["factor"]*(1+(50-opponent["difficulty"])/500)
    for item in attrs:
        item["context_adjustment"]=round(adjustment,3)
        if mode=="Adjusted":
            item["badge_value"]=round(max(0,min(99,safe_number(item["badge_value"])*adjustment)),1)
            item["display_value"]=item["badge_value"]
    ranked=sorted(attrs,key=lambda item:safe_number(item.get("badge_value",item["blended_percentile"])),reverse=True)
    labels=[{"metric":item["metric"],"tier":_tier(safe_number(item.get("badge_value",item["blended_percentile"]))),"percentile":item.get("badge_value",item["blended_percentile"])} for item in ranked]
    return {"domain":"nfl_player","player":profile["player"],"player_id":profile["player_id"],"team":profile["team"],"position":position,"season":profile["season"],"mode":mode,"comparison_mode":comparison_mode,"source":profile["source"],"raw_stats":raw,"identity_summary":_identity(position,attrs,raw),"strengths":[item for item in labels if item["tier"] in {"Elite","Strong"}][:4],"weaknesses":[item for item in reversed(labels) if item["tier"]=="Weak"][:3],"attributes":attrs,"usage_profile":{"snap_share":snap,"volume_per_game":round(volume,1),"red_zone_share":safe_number(raw.get("red_zone_share",raw.get("red_zone_target_rate",0))),"target_share":safe_number(raw.get("target_share")),"role_stability":round(role_stability,1)},"efficiency_profile":{**_position_efficiency(position,raw),"explosive_play_probability":explosive},"trends":{"snap_share":"stable" if role_stability>=70 else "variable","targets_per_game":round(safe_number(raw.get("targets"))/games,1),"carries_per_game":round(safe_number(raw.get("rush_attempts"))/games,1)},"matchup":opponent,"pace":pace,"weather":weather,"offensive_line":line,"context_factor":round(adjustment,3),"volatility_score":round(volatility,1),"warnings":list(dict.fromkeys(profile["warnings"]))}
