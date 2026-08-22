"""Context-aware, deterministic fantasy trade valuation.

This is the analytical layer for comparing assets and constructing sensible
one-for-one proposals.  It complements (rather than replaces) any stochastic
league simulator: values here are transparent, reproducible, and safe to show
directly in product surfaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .scarcity_engine import (
    DEFAULT_ROSTER_SLOTS,
    _clamp,
    _coerce_players,
    _coerce_roster_slots,
    _finite_number,
    _position,
    _projection,
    scarcity_report,
)

INACTIVE_STATUSES = {"OUT", "IR", "PUP", "SUSPENDED"}
QUESTIONABLE_STATUSES = {"DOUBTFUL", "QUESTIONABLE"}


def _normalize_fraction(value: Any, default: float) -> float:
    number = _finite_number(value)
    if number is None:
        return default
    if number > 1.0:
        number /= 100.0
    return _clamp(number, 0.0, 1.0)


def _player_context(context: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if context is None:
        return [], {}
    if isinstance(context, Mapping):
        row = dict(context)
        pool = row.get("player_pool", row.get("players", row.get("projections", [])))
        return (_coerce_players(pool) if pool else []), row
    return _coerce_players(context), {}


def _injury_factor(row: Mapping[str, Any]) -> float:
    status = str(row.get("injury_status", row.get("status", "")) or "").strip().upper()
    if status in INACTIVE_STATUSES:
        return 0.45
    if status == "DOUBTFUL":
        return 0.7
    if status == "QUESTIONABLE":
        return 0.9
    return 1.0


def _volatility(row: Mapping[str, Any], projection: float) -> tuple[float, float]:
    raw = _finite_number(row.get("volatility", row.get("projection_volatility")))
    if raw is not None:
        ratio = raw / max(projection, 1.0) if raw > 1.0 else raw
        return _clamp(ratio, 0.0, 1.5), max(0.0, raw)
    floor = _finite_number(row.get("floor"))
    ceiling = _finite_number(row.get("ceiling"))
    if floor is not None and ceiling is not None and ceiling >= floor:
        spread = ceiling - floor
        return _clamp(spread / max(2.0 * projection, 1.0), 0.0, 1.5), spread / 3.29
    confidence = _normalize_fraction(row.get("projection_confidence", row.get("confidence")), 0.65)
    return 1.0 - confidence, projection * (1.0 - confidence)


def player_value_score(player: Any, context: Any = None) -> dict[str, Any]:
    """Return an explainable rest-of-season value record for one player.

    ``context`` may be a player pool or a mapping containing ``player_pool``,
    ``replacement_levels``, ``scarcity_multipliers``, and ``weeks_remaining``.
    No context is required when the player already carries VOR/scarcity fields.
    """

    candidates = _coerce_players(player)
    if len(candidates) != 1:
        raise ValueError("player must resolve to exactly one player")
    row = candidates[0]
    pool, settings = _player_context(context)
    if not pool:
        pool = [row]
    elif all(candidate["player_id"] != row["player_id"] for candidate in pool):
        pool.append(row)

    projection = _projection(row)
    position = row["position"]
    replacement_levels = settings.get("replacement_levels") if settings else None
    multipliers = settings.get("scarcity_multipliers") if settings else None
    context_replacement = _finite_number(replacement_levels.get(position)) if isinstance(replacement_levels, Mapping) else None
    if context_replacement is not None:
        replacement = context_replacement
        raw_multiplier = multipliers.get(position, 1.0) if isinstance(multipliers, Mapping) else 1.0
        parsed_multiplier = _finite_number(raw_multiplier)
        if parsed_multiplier is None or parsed_multiplier < 0:
            raise ValueError(f"scarcity multiplier for {position} must be a non-negative finite number")
        scarcity_multiplier = parsed_multiplier
        scarcity_score = _clamp((scarcity_multiplier - 0.9) / 0.6 * 100.0, 0.0, 100.0)
    elif len(pool) > 1:
        teams = int(settings.get("teams", settings.get("n_teams", 12))) if settings else 12
        roster_slots = settings.get("roster_slots", settings.get("roster_requirements")) if settings else None
        report = scarcity_report(pool, teams=teams, roster_slots=roster_slots)
        details = report[position]
        replacement = details["replacement_level"]
        scarcity_multiplier = details["scarcity_multiplier"]
        scarcity_score = details["scarcity_score"]
    else:
        replacement = _finite_number(row.get("replacement_level"), 0.0) or 0.0
        scarcity_multiplier = _finite_number(row.get("scarcity_multiplier"), 1.0) or 1.0
        scarcity_score = _clamp((scarcity_multiplier - 0.9) / 0.6 * 100.0, 0.0, 100.0)

    explicit_vor = _finite_number(row.get("value_over_replacement", row.get("vor")))
    value_over_replacement = explicit_vor if explicit_vor is not None else projection - replacement
    confidence = _normalize_fraction(row.get("projection_confidence", row.get("confidence")), 0.65)
    volatility_ratio, volatility_points = _volatility(row, projection)
    health_factor = _injury_factor(row)
    expected_games = _finite_number(row.get("expected_games", row.get("games_remaining")))
    availability = _clamp(expected_games / 17.0, 0.0, 1.0) if expected_games is not None else 1.0
    role_share = _normalize_fraction(
        row.get("snap_share", row.get("opportunity_share", row.get("usage_rate"))),
        0.6,
    )

    reliability_factor = 0.65 + 0.35 * confidence
    role_factor = 0.85 + 0.15 * role_share
    base_component = projection * reliability_factor * health_factor * availability * role_factor
    scarcity_component = max(0.0, value_over_replacement) * max(0.0, scarcity_multiplier - 0.75)
    upside = max(0.0, (_finite_number(row.get("ceiling"), projection) or projection) - projection) * 0.08
    risk_penalty = volatility_points * 0.1 + projection * max(0.0, volatility_ratio - 0.5) * 0.04
    value_score = max(0.0, base_component + scarcity_component + upside - risk_penalty)

    return {
        "player_id": row["player_id"],
        "name": row["name"],
        "position": position,
        "projection": round(projection, 3),
        "replacement_level": round(replacement, 3),
        "value_over_replacement": round(value_over_replacement, 3),
        "scarcity_multiplier": round(scarcity_multiplier, 4),
        "scarcity_score": round(scarcity_score, 2),
        "confidence": round(confidence, 4),
        "volatility": round(volatility_ratio, 4),
        "health_factor": round(health_factor, 4),
        "availability": round(availability, 4),
        "value_score": round(value_score, 3),
        "components": {
            "base_projection_value": round(base_component, 3),
            "scarcity_value": round(scarcity_component, 3),
            "upside_value": round(upside, 3),
            "risk_penalty": round(risk_penalty, 3),
        },
    }


def compute_trade_value(players: Any, *, team_context: Any = None) -> dict[str, Any]:
    """Batch player trade values in a facade-friendly envelope."""

    normalized = _coerce_players(players)
    if isinstance(team_context, Mapping):
        context = {**team_context, "player_pool": normalized}
    else:
        context = {"player_pool": normalized}
    results = [player_value_score(player, context) for player in normalized]
    results.sort(key=lambda row: (-row["value_score"], row["player_id"].casefold()))
    raw = [row["value_score"] for row in results]
    low, high = (min(raw), max(raw)) if raw else (0.0, 0.0)
    for row in results:
        row["value_percentile"] = round(50.0 if high == low else 100.0 * (row["value_score"] - low) / (high - low), 2)
    return {
        "metric": "trade_value",
        "results": results,
        "by_player": {row["player_id"]: row for row in results},
        "metadata": {
            "player_count": len(results),
            "method": "risk-adjusted projection, replacement value, and scarcity",
        },
    }


def _team_players(team: Any) -> list[dict[str, Any]]:
    if team is None:
        return []
    if isinstance(team, Mapping):
        row = dict(team)
        for key in ("players", "roster", "my_roster", "user_roster"):
            if key in row:
                return _coerce_players(row[key])
        sections: list[Any] = []
        for key in ("starters", "bench", "ir", "taxi"):
            value = row.get(key)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                sections.extend(value)
        if sections:
            return _coerce_players(sections)
    return _coerce_players(team)


def _roster_requirements(roster_requirements: Any) -> dict[str, int]:
    slots, _ = _coerce_roster_slots(roster_requirements)
    return {position: count for position, count in slots.items() if position not in {"FLEX", "SUPERFLEX"}}


def team_need_score(
    team: Any,
    position: str | None = None,
    *,
    roster_requirements: Any = None,
) -> dict[str, Any]:
    """Measure roster need by position on a 0--100 scale (higher is weaker)."""

    players = _team_players(team)
    requirements = _roster_requirements(roster_requirements or DEFAULT_ROSTER_SLOTS)
    counts: dict[str, int] = {key: 0 for key in requirements}
    active_counts: dict[str, int] = {key: 0 for key in requirements}
    for player in players:
        player_position = player["position"]
        counts[player_position] = counts.get(player_position, 0) + 1
        status = str(player.get("injury_status", player.get("status", "")) or "").upper()
        if status not in INACTIVE_STATUSES:
            active_counts[player_position] = active_counts.get(player_position, 0) + 1

    by_position: dict[str, float] = {}
    for player_position, required in sorted(requirements.items()):
        if required <= 0:
            by_position[player_position] = 0.0
            continue
        active = active_counts.get(player_position, 0)
        owned = counts.get(player_position, 0)
        if active < required:
            need = 65.0 + 35.0 * (required - active) / required
        elif owned == required:
            need = 42.0
        elif owned == required + 1:
            need = 20.0
        else:
            need = max(0.0, 12.0 - 4.0 * (owned - required - 1))
        by_position[player_position] = round(_clamp(need, 0.0, 100.0), 2)

    requested = _position({"position": position}) if position is not None else None
    if position is not None and not requested:
        raise ValueError("position must not be empty")
    if requested is not None and requested not in by_position:
        by_position[requested] = 50.0 if counts.get(requested, 0) == 0 else 10.0
    score = by_position.get(requested, 0.0) if requested else (sum(by_position.values()) / len(by_position) if by_position else 0.0)
    return {
        "position": requested,
        "score": round(score, 2),
        "by_position": by_position,
        "position_counts": dict(sorted(counts.items())),
        "active_position_counts": dict(sorted(active_counts.items())),
        "roster_requirements": requirements,
        "weakest_position": max(by_position, key=lambda key: (by_position[key], key), default=None),
    }


def positional_balance_score(team: Any, *, roster_requirements: Any = None) -> dict[str, Any]:
    """Return overall roster balance, coverage, and surplus by position."""

    needs = team_need_score(team, roster_requirements=roster_requirements)
    requirements = needs["roster_requirements"]
    counts = needs["active_position_counts"]
    coverage = {
        position: round(min(1.5, counts.get(position, 0) / required), 3) if required > 0 else 1.0
        for position, required in requirements.items()
    }
    surplus = {position: max(0, needs["position_counts"].get(position, 0) - required) for position, required in requirements.items()}
    need_values = list(needs["by_position"].values())
    average_need = sum(need_values) / len(need_values) if need_values else 0.0
    imbalance_penalty = max(need_values, default=0.0) * 0.25
    score = _clamp(100.0 - average_need - imbalance_penalty, 0.0, 100.0)
    return {
        "score": round(score, 2),
        "coverage": coverage,
        "surplus": surplus,
        "needs": needs["by_position"],
        "weakest_position": needs["weakest_position"],
        "balanced": score >= 70.0,
    }


def _side_value(players: list[dict[str, Any]], context: Mapping[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    valued = [player_value_score(player, context) for player in players]
    if not valued:
        return 0.0, []
    raw_total = sum(row["value_score"] for row in valued)
    # A roster spot has value: increasingly deep packages are modestly
    # discounted, while the best asset earns a small consolidation premium.
    package_discount = max(0.82, 1.0 - 0.045 * (len(valued) - 1))
    adjusted_total = raw_total * package_discount + max(row["value_score"] for row in valued) * 0.04
    return adjusted_total, valued


def trade_fairness_score(
    team_a_gives: Any,
    team_b_gives: Any,
    *,
    team_a: Any = None,
    team_b: Any = None,
    player_pool: Any = None,
    roster_requirements: Any = None,
) -> dict[str, Any]:
    """Evaluate fairness and roster fit for both sides of a proposed trade."""

    gives_a = _coerce_players(team_a_gives)
    gives_b = _coerce_players(team_b_gives)
    if not gives_a or not gives_b:
        raise ValueError("both trade sides must include at least one player")
    roster_a = _team_players(team_a)
    roster_b = _team_players(team_b)
    pool = _coerce_players(player_pool) if player_pool is not None else []
    combined = _coerce_players([*pool, *roster_a, *roster_b, *gives_a, *gives_b])
    context = {"player_pool": combined, "roster_requirements": roster_requirements or DEFAULT_ROSTER_SLOTS}
    a_gives_value, valued_a = _side_value(gives_a, context)
    b_gives_value, valued_b = _side_value(gives_b, context)
    midpoint = max((a_gives_value + b_gives_value) / 2.0, 1.0)
    difference = b_gives_value - a_gives_value  # positive means A receives more
    percent_difference = abs(difference) / midpoint
    fairness = _clamp(100.0 * (1.0 - percent_difference), 0.0, 100.0)

    a_incoming_need = [team_need_score(roster_a, player["position"], roster_requirements=roster_requirements)["score"] for player in gives_b]
    b_incoming_need = [team_need_score(roster_b, player["position"], roster_requirements=roster_requirements)["score"] for player in gives_a]
    a_fit = sum(a_incoming_need) / len(a_incoming_need) if roster_a and a_incoming_need else 50.0
    b_fit = sum(b_incoming_need) / len(b_incoming_need) if roster_b and b_incoming_need else 50.0

    if fairness >= 85.0:
        label = "fair"
        favored_side = "even"
    elif difference > 0:
        label = "favors_team_a"
        favored_side = "team_a"
    else:
        label = "favors_team_b"
        favored_side = "team_b"
    return {
        "fairness_score": round(fairness, 2),
        "fairness_label": label,
        "favored_side": favored_side,
        "team_a_gives_value": round(a_gives_value, 3),
        "team_b_gives_value": round(b_gives_value, 3),
        "team_a_net_value": round(difference, 3),
        "team_b_net_value": round(-difference, 3),
        "absolute_value_difference": round(abs(difference), 3),
        "percent_difference": round(percent_difference * 100.0, 2),
        "team_a_fit_score": round(a_fit, 2),
        "team_b_fit_score": round(b_fit, 2),
        "team_a_gives": valued_a,
        "team_b_gives": valued_b,
        "recommendation": "balanced exchange" if label == "fair" else f"value advantage for {favored_side.replace('_', ' ')}",
    }


def recommend_trades(
    team: Any,
    player_pool: Any,
    *,
    trade_partner_team: Any = None,
    roster_requirements: Any = None,
    max_results: int = 10,
    minimum_fairness: float = 55.0,
) -> list[dict[str, Any]]:
    """Rank deterministic one-for-one trades that improve value or roster fit."""

    if isinstance(max_results, bool) or int(max_results) != max_results or max_results < 0:
        raise ValueError("max_results must be a non-negative integer")
    if not 0.0 <= minimum_fairness <= 100.0:
        raise ValueError("minimum_fairness must be between 0 and 100")
    roster = _team_players(team)
    candidates = _team_players(trade_partner_team) if trade_partner_team is not None else _coerce_players(player_pool)
    if not roster or not candidates or max_results == 0:
        return []
    roster_ids = {player["player_id"] for player in roster}
    candidates = [player for player in candidates if player["player_id"] not in roster_ids]
    combined = _coerce_players([*roster, *candidates])
    context = {"player_pool": combined, "roster_requirements": roster_requirements or DEFAULT_ROSTER_SLOTS}
    value_by_id = {player["player_id"]: player_value_score(player, context) for player in combined}
    needs = team_need_score(roster, roster_requirements=roster_requirements)["by_position"]

    recommendations: list[dict[str, Any]] = []
    for outgoing in roster:
        outgoing_value = value_by_id[outgoing["player_id"]]["value_score"]
        outgoing_need = needs.get(outgoing["position"], 30.0)
        for incoming in candidates:
            incoming_value = value_by_id[incoming["player_id"]]["value_score"]
            # This constructor only emits one-for-one ideas.  Both sides have
            # the same single-asset consolidation premium, so fairness can be
            # computed from the cached values.  Avoid rebuilding scarcity
            # curves for every Cartesian-product pair in a large trade pool.
            fairness_midpoint = max((outgoing_value + incoming_value) / 2.0, 1.0)
            fairness_score = _clamp(
                100.0 * (1.0 - abs(incoming_value - outgoing_value) / fairness_midpoint),
                0.0,
                100.0,
            )
            if fairness_score < minimum_fairness:
                continue
            value_delta = incoming_value - outgoing_value
            incoming_need = needs.get(incoming["position"], 50.0)
            fit_gain = incoming_need - outgoing_need
            value_grade = _clamp(50.0 + 100.0 * value_delta / max(outgoing_value + incoming_value, 1.0), 0.0, 100.0)
            fit_grade = _clamp(50.0 + fit_gain / 2.0, 0.0, 100.0)
            recommendation_score = 0.45 * fairness_score + 0.35 * value_grade + 0.2 * fit_grade
            recommendations.append(
                {
                    "give": {
                        "player_id": outgoing["player_id"],
                        "name": outgoing["name"],
                        "position": outgoing["position"],
                        "value_score": round(outgoing_value, 3),
                    },
                    "receive": {
                        "player_id": incoming["player_id"],
                        "name": incoming["name"],
                        "position": incoming["position"],
                        "value_score": round(incoming_value, 3),
                    },
                    "value_delta": round(value_delta, 3),
                    "fit_gain": round(fit_gain, 2),
                    "fairness_score": round(fairness_score, 2),
                    "recommendation_score": round(recommendation_score, 2),
                    "recommendation": "pursue" if value_delta >= 0 or fit_gain >= 20 else "consider",
                    "rationale": (
                        f"Exchange {outgoing['name']} for {incoming['name']}; "
                        f"fairness {fairness_score:.0f}/100, "
                        f"value change {value_delta:+.1f}, roster-fit change {fit_gain:+.0f}."
                    ),
                }
            )
    recommendations.sort(
        key=lambda row: (
            -row["recommendation_score"],
            -row["fairness_score"],
            -row["value_delta"],
            row["receive"]["player_id"].casefold(),
            row["give"]["player_id"].casefold(),
        )
    )
    for rank, recommendation in enumerate(recommendations[: int(max_results)], start=1):
        recommendation["rank"] = rank
    return recommendations[: int(max_results)]


# Public verb variants used by callers with different naming conventions.
compute_player_value_score = player_value_score
compute_team_need_score = team_need_score
compute_positional_balance_score = positional_balance_score
compute_trade_fairness_score = trade_fairness_score
trade_recommendations = recommend_trades
trade_recommendation_engine = recommend_trades
calculate_player_value = player_value_score
calculate_team_need = team_need_score
calculate_positional_balance = positional_balance_score
calculate_trade_fairness = trade_fairness_score


__all__ = [
    "compute_player_value_score",
    "compute_positional_balance_score",
    "compute_team_need_score",
    "compute_trade_fairness_score",
    "compute_trade_value",
    "calculate_player_value",
    "calculate_positional_balance",
    "calculate_team_need",
    "calculate_trade_fairness",
    "player_value_score",
    "positional_balance_score",
    "recommend_trades",
    "team_need_score",
    "trade_fairness_score",
    "trade_recommendation_engine",
    "trade_recommendations",
]
