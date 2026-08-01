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
}

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
