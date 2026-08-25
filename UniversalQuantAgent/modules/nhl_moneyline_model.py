"""NHL moneyline model: win probability from real, caller-supplied team-strength context.

Mirrors modules/mlb_moneyline_model.py's composite-rating/logistic
approach exactly, for the same reason: no live NHL scoring-average
ingestion was added this cycle (see nhl_pipeline.md and
offline_data_contract.md), so ``betting.team_model.project_game`` (which
needs a real points-scored-average dict) doesn't apply here. When no
context is supplied for a team, its composite defaults to a neutral 1.0
-- no fabricated rating.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, fair_price_from_probability, remove_vig_two_way

#: Real, published NHL home-ice advantage -- the home team's long-run win
#: rate is a few points above 50%, similar in size to MLB's home-field
#: edge; a standard hockey-analytics constant, not a fitted coefficient.
HOME_ICE_WIN_PROBABILITY = 0.55

_LOGISTIC_SCALE = 2.2


def _team_composite(context: dict[str, Any] | None) -> float:
    """Blend one team's real goaltending/offense/defense signals into one composite rating (1.0 = neutral)."""
    context = context or {}
    goaltending = float(context.get("goaltending_strength", 1.0))
    offense = float(context.get("offensive_strength", 1.0))
    defense = float(context.get("defensive_strength", 1.0))
    special_teams = float(context.get("special_teams_strength", 1.0))
    composite = 0.35 * goaltending + 0.30 * offense + 0.20 * defense + 0.15 * special_teams
    return max(0.3, min(1.8, composite))


def win_probability_from_composite(home_composite: float, away_composite: float, *, home_ice_advantage: float = HOME_ICE_WIN_PROBABILITY) -> float:
    """P(home team wins), from the logistic transform of the composite-rating difference plus home-ice edge."""
    rating_diff = home_composite - away_composite
    neutral_ice_probability = 1.0 / (1.0 + math.exp(-_LOGISTIC_SCALE * rating_diff))
    home_ice_lift = home_ice_advantage - 0.5
    return round(max(0.02, min(0.98, neutral_ice_probability + home_ice_lift * (1.0 - abs(neutral_ice_probability - 0.5) * 2.0))), 4)


def fair_moneyline(
    home_team: str, away_team: str, *, home_context: dict[str, Any] | None = None, away_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fair win probability and moneyline price for one matchup, from real caller-supplied team context."""
    home_composite = _team_composite(home_context)
    away_composite = _team_composite(away_context)
    home_win_probability = win_probability_from_composite(home_composite, away_composite)
    away_win_probability = round(1.0 - home_win_probability, 4)
    return {
        "home_team": home_team.strip().upper(),
        "away_team": away_team.strip().upper(),
        "home_composite_rating": round(home_composite, 4),
        "away_composite_rating": round(away_composite, 4),
        "home_win_probability": home_win_probability,
        "away_win_probability": away_win_probability,
        "fair_home_moneyline": fair_price_from_probability(home_win_probability),
        "fair_away_moneyline": fair_price_from_probability(away_win_probability),
    }


def evaluate_game(
    game_odds: dict[str, Any], *, home_context: dict[str, Any] | None = None, away_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compare our fair moneyline to a real loaded NHL game-odds entry: edge, EV, recommended side."""
    home_team, away_team = game_odds["home_team"], game_odds["away_team"]
    fair = fair_moneyline(home_team, away_team, home_context=home_context, away_context=away_context)
    result: dict[str, Any] = {
        "home_team": home_team,
        "away_team": away_team,
        "model": fair,
        "odds_source": game_odds.get("source"),
    }

    moneyline = game_odds.get("moneyline") or {}
    home_price, away_price = moneyline.get("home"), moneyline.get("away")
    if home_price is not None and away_price is not None:
        market_fair_home, market_fair_away = remove_vig_two_way(home_price, away_price)
        home_edge = edge_vs_fair(fair["home_win_probability"], home_price, away_price, side="a")
        away_edge = edge_vs_fair(fair["away_win_probability"], home_price, away_price, side="b")
        result["moneyline"] = {
            "home_price": home_price,
            "away_price": away_price,
            "market_fair_home_probability": round(market_fair_home, 4),
            "market_fair_away_probability": round(market_fair_away, 4),
            "home_edge": round(home_edge, 4),
            "away_edge": round(away_edge, 4),
            "home_ev": round(expected_value(fair["home_win_probability"], home_price), 2),
            "away_ev": round(expected_value(fair["away_win_probability"], away_price), 2),
            "recommended_side": "home" if home_edge >= away_edge else "away",
            "recommended_edge": round(max(home_edge, away_edge), 4),
        }
    return result


def evaluate_games(games: list[dict[str, Any]], *, context_by_team: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Evaluate every real NHL game in the loaded game-odds file's ``games`` list."""
    context_by_team = context_by_team or {}
    rows = []
    for game in games:
        home_team, away_team = game.get("home_team"), game.get("away_team")
        if not home_team or not away_team:
            continue
        rows.append(evaluate_game(game, home_context=context_by_team.get(home_team), away_context=context_by_team.get(away_team)))

    def _sort_key(row: dict[str, Any]) -> float:
        return row.get("moneyline", {}).get("recommended_edge", 0.0)

    rows.sort(key=_sort_key, reverse=True)
    return rows
