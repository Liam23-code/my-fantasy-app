"""Moneyline/spread/total model: win probability and fair prices from real team projections.

Converts a point-spread projection (:mod:`betting.team_model`) into a win
probability using the standard, published NFL scoring-margin volatility
(~13.5 points) -- a widely-used sports-analytics constant, not a claim of
proprietary precision. Everything downstream (fair moneyline, fair total
price, edges, EV) is closed-form math on that probability and real odds the
caller supplies; no network access.
"""

from __future__ import annotations

import math
from typing import Any

from .odds_math import edge_vs_fair, expected_value, fair_price_from_probability, remove_vig_two_way
from .team_model import project_game

#: Standard deviation of an NFL game's final scoring margin, a widely
#: published sports-analytics constant (games settle within roughly one
#: score of the pre-game spread most of the time). Used to convert a point
#: spread into a win probability via the normal approximation -- the same
#: approach real market-making models use as a baseline.
NFL_MARGIN_STDEV = 13.5

#: Standard deviation of an NFL game's total points, for pricing the
#: over/under. Real game totals vary more than either team's margin alone.
NFL_TOTAL_STDEV = 10.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def win_probability_from_spread(spread: float, *, stdev: float = NFL_MARGIN_STDEV) -> float:
    """P(the favored-by-``spread`` team wins), from the normal approximation to NFL margins."""
    return round(_normal_cdf(spread / stdev), 4)


def fair_moneyline(home_team: str, away_team: str, *, averages: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Fair spread, total, and moneyline price for one matchup, from real team scoring."""
    projection = project_game(home_team, away_team, averages=averages)
    home_win_probability = win_probability_from_spread(projection["spread"])
    away_win_probability = round(1.0 - home_win_probability, 4)
    return {
        **projection,
        "home_win_probability": home_win_probability,
        "away_win_probability": away_win_probability,
        "fair_home_moneyline": fair_price_from_probability(home_win_probability),
        "fair_away_moneyline": fair_price_from_probability(away_win_probability),
    }


def evaluate_game(game_odds: dict[str, Any], *, averages: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Compare our fair moneyline/total to a real unified-odds game entry: edges, EV, confidence.

    ``game_odds`` is one entry from a unified odds object
    (:func:`betting.odds_loader.unified_odds`) -- must carry at minimum
    ``home_team``/``away_team``; ``moneyline`` and/or ``total`` sub-objects
    are compared when present.
    """
    home_team, away_team = game_odds["home_team"], game_odds["away_team"]
    fair = fair_moneyline(home_team, away_team, averages=averages)
    result: dict[str, Any] = {
        "game_id": game_odds.get("game_id"),
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

    total = game_odds.get("total") or {}
    total_line, over_price, under_price = total.get("line"), total.get("over_price"), total.get("under_price")
    if total_line is not None and over_price is not None and under_price is not None:
        z = (float(total_line) - fair["total"]) / NFL_TOTAL_STDEV
        probability_under = round(_normal_cdf(z), 4)
        probability_over = round(1.0 - probability_under, 4)
        over_edge = edge_vs_fair(probability_over, over_price, under_price, side="a")
        under_edge = edge_vs_fair(probability_under, over_price, under_price, side="b")
        result["total"] = {
            "line": total_line,
            "model_total": fair["total"],
            "probability_over": probability_over,
            "probability_under": probability_under,
            "over_edge": round(over_edge, 4),
            "under_edge": round(under_edge, 4),
            "recommended_side": "over" if over_edge >= under_edge else "under",
            "recommended_edge": round(max(over_edge, under_edge), 4),
        }

    return result


def evaluate_games(odds: dict[str, Any], *, averages: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Evaluate every game in a unified odds object, ranked by the larger of its moneyline/total edge."""
    rows = []
    for game in (odds.get("games") or {}).values():
        if not game.get("home_team") or not game.get("away_team"):
            continue
        rows.append(evaluate_game(game, averages=averages))

    def _sort_key(row: dict[str, Any]) -> float:
        edges = [row.get("moneyline", {}).get("recommended_edge", 0.0), row.get("total", {}).get("recommended_edge", 0.0)]
        return max(edges)

    rows.sort(key=_sort_key, reverse=True)
    return rows
