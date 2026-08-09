"""Cross-domain opportunity scoring and ranking."""

from __future__ import annotations

from typing import Any

from modules.utils import clamp, safe_number


def _finance_opportunities(insights: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert finance patterns into comparable 0-100 opportunities."""
    opportunities = []
    for pattern in insights.get("patterns", []):
        strength = safe_number(pattern.get("strength"))
        # Pattern confidence supplies 80 points; having a full year of data adds
        # up to 20. Future versions can add model confidence and risk penalties.
        score = clamp(strength * 80 + min(insights.get("data_points", 0) / 252, 1) * 20)
        opportunities.append({
            "domain": "finance",
            "subject": insights.get("subject", "Unknown"),
            "opportunity": pattern.get("name", "market_pattern").replace("_", " ").title(),
            "score": round(score, 1),
            "reason": pattern.get("description", "A market pattern was detected."),
        })
    return opportunities


def _sports_opportunities(insights: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert basic and optional advanced matchup edges into opportunities."""
    matchup = insights.get("matchup", {})
    edge = safe_number(matchup.get("net_rating_edge"))
    score = clamp(40 + min(edge / 15, 1) * 60)
    opportunities = [{
        "domain": "sports",
        "subject": insights.get("subject", "NBA matchup"),
        "opportunity": f"Matchup edge: {matchup.get('favored_team', 'Unknown')}",
        "score": round(score, 1),
        "reason": f"Season net-rating advantage: {edge:.2f} points per 100 possessions.",
    }]
    advanced = insights.get("advanced")
    if advanced:
        matchup = advanced.get("matchup", {})
        adjusted_edge = safe_number(matchup.get("opponent_adjusted_net_rating_edge"))
        difficulty = safe_number(matchup.get("difficulty_score"), 50)
        score = clamp(35 + min(adjusted_edge / 15, 1) * 50 +
                      (100 - difficulty) * 0.15)
        opportunities.append({
            "domain": "sports_advanced",
            "subject": advanced.get("subject", insights.get("subject", "NBA matchup")),
            "opportunity": f"Adjusted matchup edge: {matchup.get('favored_team', 'Unknown')}",
            "score": round(score, 1),
            "reason": (f"Opponent-adjusted net-rating edge {adjusted_edge:.2f}; "
                       f"matchup difficulty {difficulty:.1f}/100."),
        })
    return opportunities


def _nfl_opportunities(insights: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an NFL efficiency edge into a comparable opportunity."""
    matchup = insights.get("matchup", {})
    edge = safe_number(matchup.get("net_efficiency_edge"))
    difficulty = safe_number(matchup.get("difficulty_score"), 50)
    # This is an opportunity ranking strength, not a win probability.
    score = clamp(35 + min(edge / 30, 1) * 50 + (100 - difficulty) * 0.15)
    return [{
        "domain": "nfl",
        "subject": insights.get("subject", "NFL matchup"),
        "opportunity": f"NFL matchup edge: {matchup.get('favored_team', 'Unknown')}",
        "score": round(score, 1),
        "reason": (f"Net-efficiency edge {edge:.2f}; "
                   f"matchup difficulty {difficulty:.1f}/100."),
    }]

def rank_opportunities(
    finance_insights: dict[str, Any] | None,
    sports_insights: dict[str, Any] | None,
    nfl_insights: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank finance, NBA, and optional NFL opportunities by strength.

    The optional third argument preserves all Version 1 and Version 2 calls.
    """
    opportunities: list[dict[str, Any]] = []
    if finance_insights:
        opportunities.extend(_finance_opportunities(finance_insights))
    if sports_insights:
        opportunities.extend(_sports_opportunities(sports_insights))
    if nfl_insights:
        opportunities.extend(_nfl_opportunities(nfl_insights))
    return sorted(opportunities, key=lambda item: item["score"], reverse=True)