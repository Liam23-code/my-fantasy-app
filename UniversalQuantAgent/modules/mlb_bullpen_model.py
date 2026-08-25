"""MLB bullpen model: fatigue, real ERA rating, and high-leverage usage -> a composite strength score.

One of five DFS matchup modules feeding modules/mlb_fusion_model.py (see
mlb_matchup_engine.md) and an input to mlb_moneyline_model.py. Pure math
over real, caller-supplied bullpen data -- no live fetch, no fabricated
appearance log.
"""
from __future__ import annotations

from typing import Any

_LEAGUE_AVERAGE_BULLPEN_ERA = 4.10


def bullpen_fatigue_index(recent_appearances: list[dict[str, Any]]) -> float:
    """0-1 fatigue read from a bullpen's real appearances over the last few days (0 = fresh, 1 = fully taxed).

    ``recent_appearances`` is a list of real per-reliever entries from the
    last 3 real days: ``{"pitches_last_3_days": int, "appearances_last_3_days": int}``.
    A reliever with 3 real appearances or 45+ real pitches in 3 days is
    treated as fully fatigued -- a standard bullpen-management heuristic,
    not a fitted model.
    """
    if not recent_appearances:
        return 0.0
    scores = []
    for reliever in recent_appearances:
        pitches = float(reliever.get("pitches_last_3_days", 0) or 0)
        appearances = float(reliever.get("appearances_last_3_days", 0) or 0)
        scores.append(max(min(pitches / 45.0, 1.0), min(appearances / 3.0, 1.0)))
    return round(sum(scores) / len(scores), 4)


def bullpen_era_rating(bullpen_era: float, league_average_era: float = _LEAGUE_AVERAGE_BULLPEN_ERA) -> float:
    """A bounded 0-1+ rating from a bullpen's real ERA relative to the real league average (1.0 = average)."""
    if league_average_era <= 0:
        return 1.0
    return round(max(0.4, min(1.8, league_average_era / max(float(bullpen_era), 0.5))), 4)


def leverage_usage_score(high_leverage_innings_pct: float) -> float:
    """0-1+ score from the real share of a bullpen's innings thrown by its real high-leverage relievers.

    A bullpen that saves its best real arms for high-leverage innings
    (rather than mop-up duty) is real evidence of a well-managed,
    effectively-stronger unit than raw ERA alone reflects.
    """
    league_average_share = 0.35
    if league_average_share <= 0:
        return 1.0
    return round(max(0.5, min(1.5, float(high_leverage_innings_pct) / league_average_share)), 4)


def bullpen_strength(components: dict[str, Any]) -> dict[str, Any]:
    """Combine real fatigue, ERA, and leverage-usage signals into one composite 0-1 bullpen-strength score.

    ``components``: ``{"recent_appearances", "bullpen_era", "high_leverage_innings_pct"}``.
    Missing pieces fall back to a neutral read rather than raising.
    """
    fatigue = bullpen_fatigue_index(components.get("recent_appearances") or [])
    era_rating = bullpen_era_rating(components.get("bullpen_era", _LEAGUE_AVERAGE_BULLPEN_ERA))
    leverage = leverage_usage_score(components.get("high_leverage_innings_pct", 0.35))

    # ERA quality and leverage usage combine multiplicatively (both matter
    # together); real fatigue then discounts the result -- a great, fresh
    # bullpen scores highest, a great but exhausted one is capped down.
    raw = era_rating * (0.7 + 0.3 * leverage)
    fatigue_discount = 1.0 - 0.4 * fatigue
    composite = max(0.1, min(1.6, raw * fatigue_discount))
    return {
        "composite_strength": round(composite, 4),
        "fatigue_index": fatigue,
        "era_rating": era_rating,
        "leverage_usage_score": leverage,
    }
