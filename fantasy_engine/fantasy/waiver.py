"""Waiver-wire / free-agent recommendations.

Usage::

    from fantasy.waiver import waiver_recommendations

    ranked = waiver_recommendations(league_state, available_players, "ppr", budget=82)

Every candidate's replacement-value (computed against the free-agent pool
itself, reusing the same VOR machinery as the draft assistant) is combined
with a roster-need multiplier and mild adjustments for ownership percentage,
injury risk, this-week bye, and schedule difficulty. When a FAAB budget is
supplied, a suggested bid is included; when the league is an auction league
(``league_settings.is_auction``), a suggested auction-style dollar value is
included instead.
"""

from __future__ import annotations

from typing import Any

from fantasy.adapter import normalize_projection
from fantasy.draft import _position_need_multiplier, _replacement_levels
from fantasy.models import LeagueSettings
from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import clamp

INACTIVE_STATUSES = {"OUT", "IR", "DOUBTFUL"}


def _coerce_league_settings(league_settings: dict[str, Any] | LeagueSettings) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _ownership_multiplier(ownership_pct: float | None) -> float:
    """A widely-owned-elsewhere player who is still available is a market signal worth a small boost."""
    if ownership_pct is None:
        return 1.0
    return round(1.0 + clamp(ownership_pct, 0.0, 100.0) / 500.0, 4)


def _injury_multiplier(injury_status: str | None) -> float:
    if injury_status in {"OUT", "IR"}:
        return 0.3
    if injury_status == "DOUBTFUL":
        return 0.5
    if injury_status == "QUESTIONABLE":
        return 0.9
    return 1.0


def _bye_week_multiplier(bye_week: int | None, current_week: int | None) -> float:
    if bye_week is not None and current_week is not None and bye_week == current_week:
        return 0.3
    return 1.0


def _schedule_multiplier(schedule_difficulty: float | None) -> float:
    """schedule_difficulty is 0-100 (higher = tougher upcoming schedule)."""
    if schedule_difficulty is None:
        return 1.0
    return round(1.0 - (clamp(schedule_difficulty, 0.0, 100.0) - 50.0) / 500.0, 4)


def _rationale(candidate: dict[str, Any]) -> str:
    parts = [f"{candidate['replacement_value']:+.1f} pts over waiver-wire replacement at {candidate['position']}"]
    if candidate["need_multiplier"] > 1.0:
        parts.append("fills a starting need on your roster")
    elif candidate["need_multiplier"] < 1.0:
        parts.append("position already well-stocked on your roster")
    if candidate.get("ownership_pct") is not None:
        parts.append(f"{candidate['ownership_pct']:.0f}% owned league-wide")
    if candidate.get("injury_status") in INACTIVE_STATUSES:
        parts.append(f"injury risk: {candidate['injury_status']}")
    elif candidate.get("injury_status") == "QUESTIONABLE":
        parts.append("questionable for this week")
    if candidate.get("bye_week_conflict"):
        parts.append("on bye this week")
    if candidate.get("schedule_difficulty") is not None:
        difficulty = candidate["schedule_difficulty"]
        parts.append("tough upcoming schedule" if difficulty >= 65 else "favorable upcoming schedule" if difficulty <= 35 else "neutral schedule")
    return "; ".join(parts)


def waiver_recommendations(
    league_state: dict[str, Any],
    available_players: list[Any],
    scoring_mode: str | None = None,
    budget: float | None = None,
) -> list[dict[str, Any]]:
    """Rank free agents for waiver claims or free-agent pickups.

    ``league_state`` expects ``{"league_settings": {...}, "my_roster": [...],
    "current_week": int}`` (all optional; sane defaults apply). ``scoring_mode``
    overrides ``league_settings.scoring_mode`` when provided. ``budget`` is the
    caller's remaining FAAB dollars (FAAB leagues) or total auction budget
    (auction leagues); omit it to skip bid suggestions entirely.
    """
    settings = _coerce_league_settings(league_state.get("league_settings", {}))
    if scoring_mode:
        settings = settings.model_copy(update={"scoring_mode": scoring_mode})
    current_week = league_state.get("current_week")

    my_roster = league_state.get("my_roster", [])
    roster_position_counts: dict[str, int] = {}
    for player in my_roster:
        position = str(player.get("position", "")).strip().upper()
        roster_position_counts[position] = roster_position_counts.get(position, 0) + 1

    scored: list[dict[str, Any]] = []
    for source in available_players:
        canonical = normalize_projection(source)
        points = calculate_fantasy_points(canonical, mode=settings.scoring_mode, custom_rules=settings.custom_rules)["total_points"]
        scored.append({**canonical, "points": round(points, 2)})

    replacement_levels = _replacement_levels(scored, settings, settings.n_teams)

    candidates: list[dict[str, Any]] = []
    for player in scored:
        replacement_value = round(player["points"] - replacement_levels.get(player["position"], 0.0), 2)
        need_multiplier = _position_need_multiplier(roster_position_counts, settings, player["position"])
        ownership_multiplier = _ownership_multiplier(player.get("ownership_pct"))
        injury_multiplier = _injury_multiplier(player.get("injury_status"))
        bye_conflict = player.get("bye_week") is not None and current_week is not None and player["bye_week"] == current_week
        bye_multiplier = _bye_week_multiplier(player.get("bye_week"), current_week)
        schedule_multiplier = _schedule_multiplier(player.get("schedule_difficulty"))
        composite_score = round(
            replacement_value * need_multiplier * ownership_multiplier * injury_multiplier * bye_multiplier * schedule_multiplier,
            2,
        )
        candidates.append(
            {
                **player,
                "replacement_value": replacement_value,
                "need_multiplier": need_multiplier,
                "bye_week_conflict": bye_conflict,
                "composite_score": composite_score,
            }
        )

    for candidate in candidates:
        candidate["rationale"] = _rationale(candidate)

    candidates.sort(key=lambda c: c["composite_score"], reverse=True)
    for index, candidate in enumerate(candidates, start=1):
        candidate["waiver_rank"] = index

    if budget is not None and candidates:
        top_score = max(c["composite_score"] for c in candidates)
        for candidate in candidates:
            if candidate["composite_score"] <= 0 or top_score <= 0:
                bid_fraction = 0.0
            else:
                bid_fraction = 0.35 * (candidate["composite_score"] / top_score)
            if settings.is_auction:
                candidate["suggested_auction_bid"] = round(clamp(bid_fraction * budget, 0.0, budget), 1)
                candidate["suggested_faab_bid"] = None
            else:
                candidate["suggested_faab_bid"] = round(clamp(bid_fraction * budget, 0.0, budget), 1)
                candidate["suggested_auction_bid"] = None
    else:
        for candidate in candidates:
            candidate["suggested_faab_bid"] = None
            candidate["suggested_auction_bid"] = None

    return candidates
