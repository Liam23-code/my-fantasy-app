"""Player prop model: real projections -> over/under probability, edge, EV, risk.

A prop stat (passing yards, receptions, ...) is modeled as approximately
Gaussian around the player's real per-game rate (season-total stat divided
by real games played -- the same real data :mod:`betting.odds_generator`
uses), with its spread scaled by the Quant Engine's own volatility read for
that player (:func:`projections.projection_engine.compute_final_projection`).
Every input traces to a real, already-computed signal; nothing here is
fabricated or fetched over the network.
"""

from __future__ import annotations

import math
from typing import Any

from projections.projection_engine import compute_final_projection

from .odds_math import edge_vs_fair, expected_value, remove_vig_two_way

#: stat market -> season-total field on a real player record (matches odds_generator).
STAT_FIELDS: dict[str, str] = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "receptions": "receptions",
}

#: Count-type markets (touchdowns, receptions) are discrete, low-mean, and
#: naturally right-skewed -- a Gaussian with stdev = mean * cv badly
#: understates their true spread (a Poisson(0.29) stat has stdev ~0.54, not
#: the ~0.19 the yardage-model formula below would give it), which inflates
#: apparent edge against a real market. Modeled as Poisson instead: variance
#: equals the mean, no extra free parameter. Real NFL scoring is somewhat
#: over-dispersed relative to pure Poisson (role/opportunity swings add
#: variance a memoryless process wouldn't have); this is a documented
#: simplification, not a claim of exact fit.
_COUNT_MARKETS = frozenset({"passing_tds", "rushing_tds", "receiving_tds", "receptions"})

#: Coefficient-of-variation range a stat's per-game standard deviation is
#: scaled to, from the Quant Engine's own 0-1 volatility read. Even a "safe"
#: (volatility=0) player's weekly yardage/TD count varies meaningfully, so
#: the floor is not tiny. A modeling choice, not observed per-stat variance
#: (we don't yet carry weekly stat-level history) -- documented rather than
#: silently assumed.
CV_LOW = 0.28
CV_HIGH = 0.65

#: cv upper bound -> risk tier label.
_RISK_TIERS: tuple[tuple[float, str], ...] = ((0.35, "low"), (0.50, "medium"), (float("inf"), "high"))

#: How many real games are needed before the per-game rate is fully trusted.
_FULL_SAMPLE_GAMES = 12.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam), by direct PMF summation.

    Stable and simple at the magnitudes real NFL counting stats take (k and
    lam both well under 20); no need for scipy or log-space tricks.
    """
    if lam <= 0.0:
        return 1.0
    if k < 0:
        return 0.0
    total = math.exp(-lam)
    term = total
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def stat_distribution(player: dict[str, Any], market: str, *, ensemble: dict[str, Any] | None = None) -> dict[str, Any]:
    """Real per-game mean and standard deviation for one stat market.

    Continuous yardage markets get a volatility-scaled Gaussian spread;
    count markets (see :data:`_COUNT_MARKETS`) get the natural Poisson
    spread (``sqrt(mean)``) instead -- see the module-level note on why a
    Gaussian badly understates a low-mean count stat's real variance.
    """
    field = STAT_FIELDS.get(market)
    if field is None:
        raise ValueError(f"unsupported prop market: {market!r} (expected one of {sorted(STAT_FIELDS)})")
    games = float(player.get("games_played") or player.get("prior_games_played") or 0.0)
    if games <= 0:
        raise ValueError("player has no real games_played to derive a per-game rate from")
    mean = float(player.get(field) or 0.0) / games

    ensemble = ensemble if ensemble is not None else compute_final_projection(dict(player))
    volatility = max(0.0, min(1.0, float(ensemble.get("volatility") or 0.0)))

    if market in _COUNT_MARKETS:
        stdev = math.sqrt(max(mean, 1e-6))
        cv = stdev / mean if mean > 0 else None
    else:
        cv = CV_LOW + (CV_HIGH - CV_LOW) * volatility
        stdev = max(mean * cv, 0.01)

    return {
        "mean": round(mean, 3),
        "stdev": round(stdev, 3),
        "cv": round(cv, 4) if cv is not None else None,
        "volatility": round(volatility, 4),
        "games": games,
        "family": "poisson" if market in _COUNT_MARKETS else "gaussian",
    }


def over_under_probability(
    player: dict[str, Any], market: str, line: float, *, ensemble: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Probability the real stat lands over/under ``line``, from the player's own distribution."""
    distribution = stat_distribution(player, market, ensemble=ensemble)
    line = float(line)
    if distribution["family"] == "poisson":
        # A half-integer line (the sportsbook-standard "no push" convention)
        # is the only realistic case: floor(line) is the highest count that
        # still counts as "under".
        probability_under = _poisson_cdf(math.floor(line), distribution["mean"])
    else:
        z = (line - distribution["mean"]) / distribution["stdev"]
        probability_under = _normal_cdf(z)
    probability_over = 1.0 - probability_under
    return {
        "market": market,
        "line": line,
        "probability_over": round(probability_over, 4),
        "probability_under": round(probability_under, 4),
        "distribution": distribution,
    }


def _confidence(player: dict[str, Any], ensemble: dict[str, Any], games: float) -> float:
    """Blend of real sample size and the ensemble's own confidence read."""
    sample_confidence = min(1.0, games / _FULL_SAMPLE_GAMES)
    model_confidence = float(ensemble.get("confidence") or 0.5)
    return round(0.5 * sample_confidence + 0.5 * model_confidence, 4)


def _risk_tier(cv: float | None) -> str:
    if cv is None:  # a ~0 mean count stat -- inherently the least predictable case
        return _RISK_TIERS[-1][1]
    for upper_bound, label in _RISK_TIERS:
        if cv <= upper_bound:
            return label
    return _RISK_TIERS[-1][1]  # pragma: no cover - inf tier always matches


def evaluate_prop(player: dict[str, Any], prop_odds: dict[str, Any]) -> dict[str, Any]:
    """Full evaluation of one real prop line: model probability, market fair probability, edge, EV, confidence, risk.

    ``prop_odds`` is one entry from a unified odds object
    (:func:`betting.odds_loader.unified_odds`) -- ``{"market", "line",
    "over_price", "under_price", ...}``.
    """
    market = prop_odds["market"]
    line = prop_odds["line"]
    over_price = prop_odds["over_price"]
    under_price = prop_odds["under_price"]

    ensemble = compute_final_projection(dict(player))
    result = over_under_probability(player, market, line, ensemble=ensemble)

    fair_over, fair_under = remove_vig_two_way(over_price, under_price)
    edge_over = edge_vs_fair(result["probability_over"], over_price, under_price, side="a")
    edge_under = edge_vs_fair(result["probability_under"], over_price, under_price, side="b")
    ev_over = expected_value(result["probability_over"], over_price)
    ev_under = expected_value(result["probability_under"], under_price)

    recommended_side = "over" if edge_over >= edge_under else "under"
    recommended_edge = edge_over if recommended_side == "over" else edge_under
    recommended_ev = ev_over if recommended_side == "over" else ev_under

    return {
        "player_id": player.get("player_id"),
        "name": player.get("name"),
        "team": player.get("team"),
        "market": market,
        "line": line,
        "over_price": over_price,
        "under_price": under_price,
        "model_probability_over": result["probability_over"],
        "model_probability_under": result["probability_under"],
        "market_fair_probability_over": round(fair_over, 4),
        "market_fair_probability_under": round(fair_under, 4),
        "edge_over": round(edge_over, 4),
        "edge_under": round(edge_under, 4),
        "ev_over": round(ev_over, 2),
        "ev_under": round(ev_under, 2),
        "recommended_side": recommended_side,
        "recommended_edge": round(recommended_edge, 4),
        "recommended_ev": round(recommended_ev, 2),
        "confidence": _confidence(player, ensemble, result["distribution"]["games"]),
        "risk_tier": _risk_tier(result["distribution"]["cv"]),
        "distribution": result["distribution"],
        "odds_source": prop_odds.get("source"),
    }


def evaluate_props(players_by_id: dict[str, dict[str, Any]], odds: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate every prop in a unified odds object against the matching real player.

    Silently skips a prop when its player isn't in ``players_by_id`` or its
    market isn't one we model (see :data:`STAT_FIELDS`) -- a partial pool or
    an exotic market is not an error, just nothing to evaluate yet.
    """
    rows: list[dict[str, Any]] = []
    for prop in (odds.get("player_props") or {}).values():
        player = players_by_id.get(str(prop.get("player_id")))
        if player is None or prop.get("market") not in STAT_FIELDS:
            continue
        try:
            rows.append(evaluate_prop(player, prop))
        except ValueError:
            continue
    rows.sort(key=lambda row: -row["recommended_edge"])
    return rows
