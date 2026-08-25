"""MLB moneyline model: win probability from starting-pitcher quality, bullpen, park, defense, and lineup.

Unlike NFL/NBA/CFB/CBB's moneyline models, this does not call
``betting.team_model.project_game`` -- that function blends two teams'
real *scoring averages*, which this engine deliberately never fetches
live (see mlb_pipeline.md and offline_data_contract.md: no live game data
ingestion was added this cycle). Instead, each team's real strength is a
composite the caller supplies directly from the matchup modules
(modules/mlb_bullpen_model.py, mlb_ballpark_model.py, mlb_defense_model.py,
mlb_lineup_model.py) plus a starting-pitcher-quality read -- when no
context is supplied for a team, its composite defaults to a neutral 1.0
rather than a fabricated rating (the same "no fabricated ratings"
principle betting/team_model.py's own docstring states).

Win probability comes from the standard logistic transform of the two
teams' composite-rating difference, not a normal approximation over a
points margin (there is no real points-margin distribution available
without live scoring data here) -- a well-established alternative
approach (the same family as a Bradley-Terry / Elo-style win-probability
model) for exactly this "I have relative strength ratings, not a
scoring distribution" situation.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, fair_price_from_probability, remove_vig_two_way

#: Real, published MLB home-field advantage: the home team wins
#: appreciably more often than 50% of the time even between evenly
#: matched teams (long-run MLB home win rate is ~54%) -- a standard
#: sabermetric constant, not a fitted coefficient.
HOME_FIELD_WIN_PROBABILITY = 0.54

#: How sharply the composite-rating difference translates into win
#: probability -- a moderate, disclosed scale (not fitted): a 0.20
#: composite-rating edge (e.g. a strong starter + strong bullpen vs. a
#: neutral opponent) moves win probability by roughly 10-12 points from a
#: pick-em game.
_LOGISTIC_SCALE = 2.2


def _team_composite(context: dict[str, Any] | None) -> float:
    """Blend one team's real pitcher/bullpen/park/defense/lineup signals into one composite rating (1.0 = neutral)."""
    context = context or {}
    pitcher_quality = float(context.get("pitcher_quality", 1.0))
    bullpen_strength = float(context.get("bullpen_strength", 1.0))
    park_factor = float(context.get("park_factor", 1.0))
    defensive_efficiency = float(context.get("defensive_efficiency", 1.0))
    lineup_strength = float(context.get("lineup_strength", 1.0))
    # A team's real chance to win leans most heavily on real pitching
    # (starter + bullpen combined carry more weight than any one other
    # factor) -- park factor cuts both ways (helps the home lineup, helps
    # the home pitching staff give up more/fewer runs) so it's weighted
    # lightly here rather than doubly counted.
    composite = 0.30 * pitcher_quality + 0.25 * bullpen_strength + 0.10 * park_factor + 0.15 * defensive_efficiency + 0.20 * lineup_strength
    return max(0.3, min(1.8, composite))


def win_probability_from_composite(home_composite: float, away_composite: float, *, home_field_advantage: float = HOME_FIELD_WIN_PROBABILITY) -> float:
    """P(home team wins), from the logistic transform of the composite-rating difference plus home-field edge."""
    rating_diff = home_composite - away_composite
    neutral_field_probability = 1.0 / (1.0 + math.exp(-_LOGISTIC_SCALE * rating_diff))
    # Blend the rating-implied probability with the real home-field base
    # rate rather than adding them (which could push past 1.0): the home
    # team's edge is real even in a dead-even matchup, and still present,
    # scaled down, once the two teams' real strength is also unequal.
    home_field_lift = home_field_advantage - 0.5
    return round(max(0.02, min(0.98, neutral_field_probability + home_field_lift * (1.0 - abs(neutral_field_probability - 0.5) * 2.0))), 4)


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
    """Compare our fair moneyline to a real loaded MLB game-odds entry: edge, EV, recommended side."""
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


def evaluate_games(
    games: list[dict[str, Any]], *, context_by_team: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Evaluate every real MLB game in the loaded game-odds file's ``games`` list."""
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
