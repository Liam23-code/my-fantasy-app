"""Fantasy scoring engine.

Usage::

    from fantasy.scoring import calculate_fantasy_points

    result = calculate_fantasy_points(projection, mode="ppr")
    print(result["total_points"], result["breakdown"])

``custom_rules`` is a small, pure-data overlay -- never code -- so it is safe
to accept from an API request body: it can only rescale existing stat
categories and swap the bonus list, never execute anything.
"""

from __future__ import annotations

from typing import Any

from fantasy.utils import safe_float

VALID_MODES = {"standard", "half-ppr", "ppr", "custom"}

# Per-unit point value for each raw stat. Yardage stats are expressed as a
# fraction (1 point per N yards) so the total is a single multiply-and-sum.
BASE_MULTIPLIERS: dict[str, float] = {
    "passing_yards": 1.0 / 25.0,
    "passing_tds": 4.0,
    "interceptions": -2.0,
    "rushing_yards": 1.0 / 10.0,
    "rushing_tds": 6.0,
    "receiving_yards": 1.0 / 10.0,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
    # Standard kicker scoring (mode-independent, like the yardage stats above
    # -- PPR/half-PPR only ever change the reception multiplier). Points
    # scale with attempt distance; nflverse already buckets makes this way.
    "field_goals_0_19": 3.0,
    "field_goals_20_29": 3.0,
    "field_goals_30_39": 3.0,
    "field_goals_40_49": 4.0,
    "field_goals_50_59": 5.0,
    "field_goals_60_plus": 5.0,
    "extra_points_made": 1.0,
    # Standard team defense/special-teams (DST) scoring. Turnovers and
    # defensive/return touchdowns are per-unit like everything else above;
    # points allowed is a banded score, not a per-unit rate, so
    # fantasy.data_loader precomputes the correct band via
    # points_allowed_score() and hands it in here as a single number scored
    # 1-for-1 -- the multiply-and-sum engine below never needs to know about
    # banding.
    "def_sacks": 1.0,
    "def_interceptions": 2.0,
    "def_fumble_recoveries": 2.0,
    "def_safeties": 2.0,
    "def_touchdowns": 6.0,
    "def_blocked_kicks": 2.0,
    "points_allowed_score": 1.0,
}

#: Standard DST points-allowed bands: (inclusive upper bound, fantasy points).
#: Checked in order, so list from fewest points allowed to most.
DST_POINTS_ALLOWED_TIERS: tuple[tuple[float, float], ...] = (
    (0.0, 10.0),
    (6.0, 7.0),
    (13.0, 4.0),
    (20.0, 1.0),
    (27.0, 0.0),
    (34.0, -1.0),
    (float("inf"), -4.0),
)


def points_allowed_score(points_allowed: Any) -> float:
    """Band real points allowed into the standard DST points-allowed score."""
    allowed = safe_float(points_allowed, 0.0)
    for upper_bound, points in DST_POINTS_ALLOWED_TIERS:
        if allowed <= upper_bound:
            return points
    return DST_POINTS_ALLOWED_TIERS[-1][1]  # pragma: no cover - inf tier always matches


RECEPTION_MULTIPLIER_BY_MODE: dict[str, float] = {
    "standard": 0.0,
    "half-ppr": 0.5,
    "ppr": 1.0,
    "custom": 0.0,
}

DEFAULT_BONUS_RULES: list[dict[str, Any]] = [
    {"stat": "passing_yards", "threshold": 300, "points": 3},
    {"stat": "rushing_yards", "threshold": 100, "points": 3},
    {"stat": "receiving_yards", "threshold": 100, "points": 3},
]

# Every stat the engine knows how to score. Anything else on the projection
# dict is ignored (e.g. identity fields like "name" or "team").
SCORABLE_STATS: tuple[str, ...] = (*BASE_MULTIPLIERS.keys(), "receptions")


def _resolve_multipliers(mode: str, custom_rules: dict[str, Any] | None) -> dict[str, float]:
    multipliers = dict(BASE_MULTIPLIERS)
    multipliers["receptions"] = RECEPTION_MULTIPLIER_BY_MODE.get(mode, 0.0)
    overrides = (custom_rules or {}).get("multipliers") or {}
    for stat, value in overrides.items():
        multipliers[stat] = safe_float(value)
    return multipliers


def _resolve_bonus_rules(custom_rules: dict[str, Any] | None) -> list[dict[str, Any]]:
    if custom_rules and "bonuses" in custom_rules:
        rules = custom_rules["bonuses"] or []
        if not isinstance(rules, list):
            raise ValueError("custom_rules['bonuses'] must be a list of {stat, threshold, points} dicts")
        return rules
    return DEFAULT_BONUS_RULES


def calculate_fantasy_points(
    projection: dict[str, Any],
    mode: str = "ppr",
    custom_rules: dict[str, Any] | None = None,
    bonuses: bool = True,
) -> dict[str, Any]:
    """Score one player-week projection.

    Parameters
    ----------
    projection:
        A mapping containing any subset of ``SCORABLE_STATS``. Missing keys
        default to ``0``. Values may be numbers or numeric strings.
    mode:
        One of ``"standard"``, ``"half-ppr"``, ``"ppr"``, or ``"custom"``.
        ``"custom"`` requires ``custom_rules`` (there is no inherent default
        for a mode named "custom"). For the other modes, ``custom_rules`` is
        still applied as an overlay on top of the mode's defaults, so callers
        can e.g. run PPR scoring with a league-specific TD bonus.
    custom_rules:
        Optional ``{"multipliers": {stat: points_per_unit}, "bonuses": [...]}``
        overlay. Pure data -- never evaluated as code.
    bonuses:
        Toggle the yardage bonus rules on or off entirely.

    Returns
    -------
    dict with ``total_points``, ``breakdown`` (points contributed per stat),
    ``mode``, ``bonuses_applied`` (list of triggered bonus rules), and
    ``raw_projection`` (the original input, unmodified).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown scoring mode {mode!r}; expected one of {sorted(VALID_MODES)}")
    if mode == "custom" and not custom_rules:
        raise ValueError("mode='custom' requires custom_rules (e.g. {'multipliers': {...}})")
    if not isinstance(projection, dict):
        raise TypeError(f"projection must be a dict, got {type(projection).__name__}")

    multipliers = _resolve_multipliers(mode, custom_rules)
    bonus_rules = _resolve_bonus_rules(custom_rules)

    breakdown: dict[str, float] = {}
    total_points = 0.0
    for stat, multiplier in multipliers.items():
        value = safe_float(projection.get(stat, 0))
        points = round(value * multiplier, 4)
        breakdown[stat] = points
        total_points += points

    bonuses_applied: list[dict[str, Any]] = []
    if bonuses:
        for rule in bonus_rules:
            stat = rule["stat"]
            threshold = safe_float(rule["threshold"])
            points = safe_float(rule["points"])
            value = safe_float(projection.get(stat, 0))
            if value >= threshold:
                total_points += points
                bonuses_applied.append({"stat": stat, "threshold": threshold, "points": points, "value": value})

    return {
        "total_points": round(total_points, 2),
        "breakdown": breakdown,
        "mode": mode,
        "bonuses_applied": bonuses_applied,
        "raw_projection": projection,
    }


def batch_calculate_fantasy_points(
    projections: list[dict[str, Any]],
    mode: str = "ppr",
    custom_rules: dict[str, Any] | None = None,
    bonuses: bool = True,
) -> list[dict[str, Any]]:
    """Score many projections at once.

    Multipliers and bonus rules are resolved once and reused across the whole
    batch (the per-call cost in :func:`calculate_fantasy_points` is dict
    construction, not real work), which is what keeps 10k-player batches fast.
    """
    return [calculate_fantasy_points(projection, mode, custom_rules, bonuses) for projection in projections]
