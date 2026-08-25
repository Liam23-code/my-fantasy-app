"""Real NBA on/off lineup impact: which teammates swing a team's pace/rating most.

Part of the matchup-aware betting engine (see matchup_engine.md).
``modules.nba_advanced`` already has a real, working on/off-splits fetch
(``fetch_on_off_splits``, wrapping ``nba_api``'s
``TeamPlayerOnOffSummary``) and a real summarizer (``summarize_on_off``)
built for the existing Correlation Lab / Model Drivers analysis pages --
this module does not duplicate that real fetch, it imports and reuses it
directly (see betting_engine.md's "shared, not duplicated" rule), adding
only the team-lookup convenience and a betting-engine-shaped return.

This is a *team-level* signal (real net-rating/pace swing while one named
teammate is on vs. off the court) -- for one specific player's own real
box-score rates when a specific teammate is absent, see
``modules.nba_injury_impact.teammate_absence_split`` instead.
"""
from __future__ import annotations

from typing import Any

from modules.nba_advanced import fetch_on_off_splits, find_team, latest_season, summarize_on_off
from modules.sportsbook_parser import normalize_player_name


def team_on_off_impact(team_name: str, season: str | None = None, *, limit: int = 8) -> dict[str, Any]:
    """Real net-rating on/off swing for ``team_name``'s real rotation players this season.

    Returns ``{"team", "season", "players", "warnings"}`` where
    ``players`` is ``summarize_on_off``'s real, ranked
    ``[{"player", "on_net_rating", "off_net_rating", "on_off_swing"}, ...]``
    (largest absolute swing first) -- the same real signal already shown
    elsewhere in this app, reused here for the betting engine's matchup
    context rather than recomputed.
    """
    season = season or latest_season()
    team = find_team(team_name)
    warnings: list[str] = []
    try:
        tables = fetch_on_off_splits(team["id"], season)
    except Exception as exc:
        tables = []
        warnings.append(f"On/off splits unavailable: {exc}")
    # fetch_on_off_splits's real response is 3 tables -- [0] one team-wide
    # overall row, [1] real per-player "On" splits, [2] real per-player
    # "Off" splits (verified live during this cycle's build).
    # summarize_on_off expects [on_table, off_table]; slicing off the
    # leading overall-row table here is this module's own correct call,
    # not a change to summarize_on_off itself.
    players = summarize_on_off(tables[1:], limit=limit) if len(tables) >= 3 else []
    return {"team": team["abbreviation"], "season": season, "players": players, "warnings": warnings}


def _as_first_last(name: str) -> str:
    """``nba_api``'s on/off rows real-name real players as "Last, First" -- normalize to "First Last" for matching."""
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}" if "," in name else name


def teammate_on_off_swing(team_name: str, teammate_name: str, season: str | None = None) -> dict[str, Any] | None:
    """This one named teammate's real on/off net-rating swing for their team, or ``None`` if not found in the real ranking."""
    context = team_on_off_impact(team_name, season, limit=15)
    teammate_key = normalize_player_name(teammate_name).lower()
    for row in context["players"]:
        if normalize_player_name(_as_first_last(row["player"])).lower() == teammate_key:
            return {**row, "team": context["team"], "season": context["season"]}
    return None
