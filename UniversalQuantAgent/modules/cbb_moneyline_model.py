"""CBB moneyline/spread/total model: win probability and fair prices from real team scoring.

Mirrors modules/nba_moneyline_model.py and modules/cfb_moneyline_model.py
exactly: reuses ``betting.team_model.project_game`` and every function in
``betting.odds_math`` directly. The only sport-specific piece is the
margin/total standard deviation, from :mod:`modules.cbb_team_model`'s real,
computed CBB volatility.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, fair_price_from_probability, remove_vig_two_way
from betting.team_model import project_game


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def win_probability_from_spread(spread: float, *, stdev: float) -> float:
    """P(the favored-by-``spread`` team wins), from the normal approximation to CBB margins."""
    return round(_normal_cdf(spread / stdev), 4)


def fair_moneyline(home_team: str, away_team: str, *, averages: dict[str, dict[str, float]], margin_stdev: float) -> dict[str, Any]:
    """Fair spread, total, and moneyline price for one matchup, from real CBB team scoring."""
    projection = project_game(home_team, away_team, averages=averages)
    home_win_probability = win_probability_from_spread(projection["spread"], stdev=margin_stdev)
    away_win_probability = round(1.0 - home_win_probability, 4)
    return {
        **projection,
        "home_win_probability": home_win_probability,
        "away_win_probability": away_win_probability,
        "fair_home_moneyline": fair_price_from_probability(home_win_probability),
        "fair_away_moneyline": fair_price_from_probability(away_win_probability),
    }


def evaluate_game(game_odds: dict[str, Any], *, averages: dict[str, dict[str, float]], volatility: dict[str, float]) -> dict[str, Any]:
    """Compare our fair moneyline/total to a real game-odds entry: edges, EV, confidence."""
    home_team, away_team = game_odds["home_team"], game_odds["away_team"]
    fair = fair_moneyline(home_team, away_team, averages=averages, margin_stdev=volatility["margin_stdev"])
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
        z = (float(total_line) - fair["total"]) / volatility["total_stdev"]
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


def evaluate_games(
    games: list[dict[str, Any]],
    *,
    averages: dict[str, dict[str, float]],
    odds_by_matchup: dict[tuple[str, str], dict[str, Any]] | None = None,
    volatility: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate real CBB matchups, optionally compared against loaded CBB game odds."""
    odds_by_matchup = odds_by_matchup or {}
    rows = []
    for game in games:
        home_team, away_team = game.get("home_team"), game.get("away_team")
        if not home_team or not away_team:
            continue
        game_odds = odds_by_matchup.get((home_team, away_team), {"home_team": home_team, "away_team": away_team})
        rows.append(evaluate_game(game_odds, averages=averages, volatility=volatility))

    def _sort_key(row: dict[str, Any]) -> float:
        edges = [row.get("moneyline", {}).get("recommended_edge", 0.0), row.get("total", {}).get("recommended_edge", 0.0)]
        return max(edges)

    rows.sort(key=_sort_key, reverse=True)
    return rows
