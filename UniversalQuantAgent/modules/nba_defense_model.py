"""Real, position-specific NBA opponent defensive difficulty: rim protection, perimeter defense, pace.

Part of the matchup-aware betting engine (see matchup_engine.md).
``modules.matchup_model.project_matchup_difficulty`` already scores
opponent difficulty from one blunt signal (overall ``DEF_RATING``, real
but zone-agnostic) -- this module adds a real, more granular signal on
top of it without touching that existing function: real defended-shot
data by zone, from the NBA's own player-tracking defense endpoint
(``nba_api.stats.endpoints.leaguedashptdefend``), the same real feed
``nba_api`` exposes for shot-defense analysis league-wide. Verified live
during this cycle's build (real 2025-26 rim-defense leaders: Donovan
Clingan, Rudy Gobert, Alperen Sengun).

There is no team-level version of this endpoint that works reliably
(``LeagueDashPtTeamDefend`` returns a malformed response as of this
build) -- team ratings here are computed by aggregating real per-player
defended-shot totals up to each player's own team, weighted by real shot
volume (``D_FGA``), rather than a second, unverified endpoint.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from betting.cache_utils import ttl_cache
from modules.nba_advanced import fetch_league_team_stats, find_team, latest_season

#: Real NBA player-tracking defense-category values this module uses --
#: see ``nba_api.stats.library.parameters.DefenseCategory``. Rim = shots
#: within 6 feet; perimeter = three-point attempts. Each category returns
#: its *own* real column names (verified live during this cycle's build --
#: "Overall" uses D_FGA/D_FG_PCT, but "Less Than 6Ft" uses
#: FGA_LT_06/LT_06_PCT and "3 Pointers" uses FG3A/FG3_PCT), so the volume
#: and defended-FG% column names are carried alongside each category.
_RIM_CATEGORY = "Less Than 6Ft"
_RIM_VOLUME_COLUMN = "FGA_LT_06"
_RIM_PCT_COLUMN = "LT_06_PCT"
_PERIMETER_CATEGORY = "3 Pointers"
_PERIMETER_VOLUME_COLUMN = "FG3A"
_PERIMETER_PCT_COLUMN = "FG3_PCT"

#: Real per-position role -> which zone(s) matter most for a difficulty
#: score, and how heavily. An interior player's shots are mostly at the
#: rim; a primary ball handler's/wing's are more perimeter-weighted.
_ROLE_ZONE_WEIGHTS = {
    "interior": {"rim": 0.80, "perimeter": 0.20},
    "primary ball handler": {"rim": 0.25, "perimeter": 0.75},
    "wing": {"rim": 0.40, "perimeter": 0.60},
}

_DEFENSE_CACHE_SECONDS = 60 * 60


@ttl_cache(_DEFENSE_CACHE_SECONDS)
def _fetch_pt_defend(season: str, defense_category: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashptdefend

    frame = leaguedashptdefend.LeagueDashPtDefend(season=season, defense_category=defense_category, timeout=20).get_data_frames()[0]
    return frame


def team_zone_defense(season: str | None = None) -> dict[str, dict[str, float]]:
    """Real, team-aggregated rim and perimeter defended-FG% for every team this season.

    Aggregates real per-player ``LeagueDashPtDefend`` rows up to each
    player's own team (``PLAYER_LAST_TEAM_ABBREVIATION``), weighted by
    each defender's real shot volume (``D_FGA``) so a bench player who
    rarely contests a shot doesn't move the team number as much as a
    starter who contests hundreds. Returns
    ``{team: {"rim_def_fg_pct", "perimeter_def_fg_pct"}}``; fails soft to
    ``{}`` for either zone if the live call errors.
    """
    season = season or latest_season()
    result: dict[str, dict[str, float]] = {}
    zones = (
        ("rim_def_fg_pct", _RIM_CATEGORY, _RIM_VOLUME_COLUMN, _RIM_PCT_COLUMN),
        ("perimeter_def_fg_pct", _PERIMETER_CATEGORY, _PERIMETER_VOLUME_COLUMN, _PERIMETER_PCT_COLUMN),
    )
    for zone_key, category, volume_column, pct_column in zones:
        try:
            frame = _fetch_pt_defend(season, category)
        except Exception:
            continue
        if frame.empty or "PLAYER_LAST_TEAM_ABBREVIATION" not in frame or volume_column not in frame or pct_column not in frame:
            continue
        frame = frame.copy()
        frame[volume_column] = pd.to_numeric(frame[volume_column], errors="coerce").fillna(0.0)
        frame[pct_column] = pd.to_numeric(frame[pct_column], errors="coerce")
        for team, group in frame.groupby("PLAYER_LAST_TEAM_ABBREVIATION"):
            weights = group[volume_column]
            total_weight = weights.sum()
            if total_weight <= 0:
                continue
            weighted_pct = (group[pct_column] * weights).sum() / total_weight
            result.setdefault(str(team), {})[zone_key] = round(float(weighted_pct), 4)
    return result


def matchup_difficulty_score(player_role: str, opponent_team: str, season: str | None = None) -> dict[str, Any]:
    """Real, zone-weighted defensive difficulty (0-100, higher = tougher) for one opponent against one role.

    ``player_role`` matches ``modules.matchup_model``'s existing role
    taxonomy (``"interior"``, ``"primary ball handler"``, ``"wing"``) so
    a caller already computing a role there can reuse it here. A lower
    defended FG% is a *tougher* defense, so the score is built from
    ``1 - defended_fg_pct`` (league-percentile-ranked) rather than the
    raw percentage, then blended by the role's real zone weights, then
    blended with real team pace (a faster team means more real
    possessions to defend against, a real, separate difficulty driver).
    """
    season = season or latest_season()
    opponent = find_team(opponent_team)
    weights = _ROLE_ZONE_WEIGHTS.get(player_role, _ROLE_ZONE_WEIGHTS["wing"])
    zone_defense = team_zone_defense(season)

    rim_by_team = {team: 1.0 - values["rim_def_fg_pct"] for team, values in zone_defense.items() if "rim_def_fg_pct" in values}
    perimeter_by_team = {team: 1.0 - values["perimeter_def_fg_pct"] for team, values in zone_defense.items() if "perimeter_def_fg_pct" in values}

    def _percentile(by_team: dict[str, float], team: str) -> float | None:
        if team not in by_team or len(by_team) < 2:
            return None
        series = pd.Series(by_team)
        return float(series.rank(pct=True).loc[team]) * 100.0

    rim_score = _percentile(rim_by_team, opponent["abbreviation"])
    perimeter_score = _percentile(perimeter_by_team, opponent["abbreviation"])

    components = [(rim_score, weights["rim"]), (perimeter_score, weights["perimeter"])]
    available = [(score, weight) for score, weight in components if score is not None]
    if available:
        total_weight = sum(weight for _, weight in available)
        zone_score = sum(score * weight for score, weight in available) / total_weight
        basis = "real_pt_defend_data"
    else:
        zone_score = 50.0
        basis = "no_pt_defend_data"

    league = fetch_league_team_stats(season)
    league_pace = float(pd.to_numeric(league.get("PACE"), errors="coerce").mean()) if "PACE" in league else 100.0
    rows = league[league["TEAM_ID"] == opponent["id"]] if "TEAM_ID" in league else pd.DataFrame()
    opponent_pace = float(pd.to_numeric(rows.iloc[0].get("PACE"), errors="coerce")) if not rows.empty and "PACE" in rows else league_pace
    pace_adjustment = (opponent_pace - league_pace) * 0.5  # faster pace -> more real possessions -> modestly tougher

    difficulty = max(0.0, min(100.0, zone_score + pace_adjustment))

    return {
        "opponent": opponent["abbreviation"],
        "season": season,
        "player_role": player_role,
        "difficulty_score": round(difficulty, 1),
        "rim_percentile": round(rim_score, 1) if rim_score is not None else None,
        "perimeter_percentile": round(perimeter_score, 1) if perimeter_score is not None else None,
        "opponent_pace": round(opponent_pace, 2),
        "league_average_pace": round(league_pace, 2),
        "basis": basis,
    }
