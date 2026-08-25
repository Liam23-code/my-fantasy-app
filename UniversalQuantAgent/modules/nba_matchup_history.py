"""Real opponent-specific NBA game history: last-N vs. one opponent, real per-game splits.

Part of the matchup-aware betting engine (see matchup_engine.md). Reuses
the exact same real game-log fetch already powering
``modules.matchup_model``/``modules.projections``
(``modules.projections._find_player`` / ``_game_log_with_fuzzy_fallback``)
rather than a second, duplicate real-data fetch -- this module's job is to
turn that already-fetched real log into a reusable, disclosed,
opponent-filtered view, not to fetch anything new.

Every result discloses its real sample size (``games_sampled``) so a
one-game "history" is never presented with the same weight as ten.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import coerce_numeric, safe_number
from modules.nba_advanced import find_team, latest_season
from modules.projections import _find_player, _game_log_with_fuzzy_fallback

#: Below this many real games vs. the opponent, the average is disclosed
#: but not treated as a reliable signal by callers (see nba_matchup_engine.py).
MIN_RELIABLE_GAMES = 3

_STATS = ("pts", "reb", "ast", "min")


def games_vs_opponent(player_name: str, opponent_team: str, season: str | None = None, *, last_n: int = 10) -> dict[str, Any]:
    """Real per-game stats from this player's own log, filtered to games against ``opponent_team``.

    Returns ``{"games_sampled", "games", "averages", "season_averages",
    "delta_vs_season", "warnings"}`` -- ``games`` is every real matched
    game (most recent first, capped at ``last_n``), ``delta_vs_season`` is
    each stat's real matchup average minus the real full-season average
    (positive = plays better against this opponent).
    """
    season = season or latest_season()
    opponent = find_team(opponent_team)
    player = _find_player(player_name)
    resolved, raw, warnings = _game_log_with_fuzzy_fallback(player_name, player, season)
    games = coerce_numeric(raw)

    season_averages = {stat: round(safe_number(pd.to_numeric(games.get(stat), errors="coerce").mean()), 2) for stat in _STATS if stat in games}

    if "matchup" not in games or games.empty:
        return {
            "player": resolved.get("full_name", player_name),
            "opponent": opponent["abbreviation"],
            "season": season,
            "games_sampled": 0,
            "games": [],
            "averages": {},
            "season_averages": season_averages,
            "delta_vs_season": {},
            "warnings": warnings + ["No game log available to filter by opponent."],
        }

    versus = games[games["matchup"].astype(str).str.contains(opponent["abbreviation"], case=False, na=False)]
    if "game_date" in versus:
        versus = versus.sort_values("game_date", ascending=False)
    versus = versus.head(last_n)

    averages = {stat: round(safe_number(pd.to_numeric(versus.get(stat), errors="coerce").mean()), 2) for stat in _STATS if stat in versus}
    delta = {stat: round(averages[stat] - season_averages[stat], 2) for stat in averages if stat in season_averages}

    game_rows = [
        {
            "date": str(row.get("game_date", "")),
            "matchup": str(row.get("matchup", "")),
            "pts": safe_number(row.get("pts")),
            "reb": safe_number(row.get("reb")),
            "ast": safe_number(row.get("ast")),
            "min": safe_number(row.get("min")),
        }
        for _, row in versus.iterrows()
    ]

    return {
        "player": resolved.get("full_name", player_name),
        "opponent": opponent["abbreviation"],
        "season": season,
        "games_sampled": len(game_rows),
        "games": game_rows,
        "averages": averages,
        "season_averages": season_averages,
        "delta_vs_season": delta,
        "warnings": warnings,
    }
