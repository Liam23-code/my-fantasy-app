"""Deterministic week-by-week fantasy football projections.

The forward projection in :mod:`fantasy.projections` is the authoritative
season baseline.  This module distributes that total across the NFL's 18-week
regular season, zeros the player's bye, and applies two deliberately bounded
week-level effects:

* an opponent adjustment derived from defense-vs-position data supplied on
  the player/schedule record; and
* a deterministic volatility curve whose amplitude is controlled by the
  season projection's ``projection_confidence``.

No schedule or defense is fabricated.  Callers may provide a compact
``schedule``/``opponents`` mapping, a list of weekly matchup dictionaries, or
the flat ``opponent`` and ``opponent_defense`` fields used by the canonical
projection adapter.  Missing matchup data is neutral (a 1.0 multiplier).

The primary output is stable and JSON-friendly::

    {
        1: {"points": 18.42, "confidence": 0.81},
        2: {"points": 16.97, "confidence": 0.80},
        ...
        18: {"points": 0.0, "confidence": 1.0},
    }

Week numbers are integers in Python.  JSON serialization naturally converts
them to string object keys.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from fantasy.models import LeagueSettings
from fantasy.projections import projected_points
from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import clamp, safe_float

REGULAR_SEASON_WEEKS = 18
NFL_GAMES_PER_TEAM = 17
WEEKS = tuple(range(1, REGULAR_SEASON_WEEKS + 1))

# Defense adjustments are intentionally capped.  A matchup is useful signal,
# but it should never overwhelm the season-long role and talent projection.
MIN_DEFENSE_MULTIPLIER = 0.80
MAX_DEFENSE_MULTIPLIER = 1.20

_SCORING_ALIASES = {
    "half": "half-ppr",
    "half_ppr": "half-ppr",
    "halfppr": "half-ppr",
    "non-ppr": "standard",
    "non_ppr": "standard",
}
_RECEPTION_POINTS = {"standard": 0.0, "half-ppr": 0.5, "ppr": 1.0}
_BYE_MARKERS = {"BYE", "OFF", "OPEN"}
_POSITION_FALLBACKS = {
    "QB": ("PASS", "OFFENSE", "OVERALL", "ALL", "DEFAULT"),
    "RB": ("RUSH", "RUN", "FLEX", "SKILL", "OVERALL", "ALL", "DEFAULT"),
    "WR": ("PASS", "REC", "RECEIVING", "FLEX", "SKILL", "OVERALL", "ALL", "DEFAULT"),
    "TE": ("PASS", "REC", "RECEIVING", "FLEX", "SKILL", "OVERALL", "ALL", "DEFAULT"),
    "K": ("SPECIAL_TEAMS", "OVERALL", "ALL", "DEFAULT"),
    "DST": ("OFFENSE", "OVERALL", "ALL", "DEFAULT"),
}


def _as_mapping(player: Any) -> dict[str, Any]:
    if isinstance(player, Mapping):
        return dict(player)
    if hasattr(player, "model_dump") and callable(player.model_dump):
        return dict(player.model_dump())
    if hasattr(player, "__dict__"):
        return dict(vars(player))
    raise TypeError(f"player must be a mapping or object with fields, got {type(player).__name__}")


def _validated_week(week: Any) -> int:
    if isinstance(week, bool):
        raise ValueError("week must be an integer from 1 through 18")
    try:
        number = int(week)
    except (TypeError, ValueError) as error:
        raise ValueError("week must be an integer from 1 through 18") from error
    if number not in WEEKS:
        raise ValueError("week must be an integer from 1 through 18")
    return number


def _scoring_mode(value: Any) -> str:
    mode = str(value or "ppr").strip().lower()
    return _SCORING_ALIASES.get(mode, mode)


def _position_value(value: Any, position: str) -> Any:
    """Select a position-specific value from a nested defense field."""
    if not isinstance(value, Mapping):
        return value
    normalized = {str(key).strip().upper(): item for key, item in value.items()}
    for key in (position, *_POSITION_FALLBACKS.get(position, ("OVERALL", "ALL", "DEFAULT"))):
        if key in normalized and normalized[key] is not None:
            return normalized[key]
    return None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank_multiplier(rank: float) -> float:
    # Rank 1 is the toughest defense; rank 32 is the most favorable matchup.
    normalized = (clamp(rank, 1.0, 32.0) - 1.0) / 31.0
    return MIN_DEFENSE_MULTIPLIER + normalized * (MAX_DEFENSE_MULTIPLIER - MIN_DEFENSE_MULTIPLIER)


def defensive_strength_adjustment(defense: Any, position: str) -> float:
    """Return a bounded opponent multiplier for ``position``.

    Supported defense shapes include:

    * an explicit multiplier/adjustment (overall or keyed by position);
    * a defense-vs-position rank, where 1 is toughest and 32 is easiest;
    * fantasy points allowed plus a league-average value;
    * a 0-100 strength/rating, where higher means a stronger defense; and
    * labels such as ``"elite"``, ``"strong"``, ``"weak"``.

    A bare numeric value from 1 through 32 is treated as a rank.  Unknown or
    incomplete data is neutral.  The returned multiplier is always between
    0.80 and 1.20.
    """
    normalized_position = str(position or "").strip().upper()
    if defense is None:
        return 1.0

    if isinstance(defense, str):
        label = defense.strip().lower().replace("_", " ").replace("-", " ")
        labels = {
            "elite": 0.82,
            "very strong": 0.85,
            "strong": 0.90,
            "tough": 0.90,
            "above average": 0.95,
            "average": 1.0,
            "neutral": 1.0,
            "below average": 1.05,
            "weak": 1.10,
            "very weak": 1.15,
            "poor": 1.12,
        }
        if label in labels:
            return labels[label]
        numeric = _numeric(defense)
        return round(_rank_multiplier(numeric), 4) if numeric is not None and 1 <= numeric <= 32 else 1.0

    if not isinstance(defense, Mapping):
        numeric = _numeric(defense)
        if numeric is None:
            return 1.0
        if 1.0 <= numeric <= 32.0:
            return round(_rank_multiplier(numeric), 4)
        if 32.0 < numeric <= 100.0:
            return round(clamp(1.20 - 0.004 * numeric, MIN_DEFENSE_MULTIPLIER, MAX_DEFENSE_MULTIPLIER), 4)
        # Outside rank range, accept a plausible explicitly supplied factor.
        if MIN_DEFENSE_MULTIPLIER <= numeric <= MAX_DEFENSE_MULTIPLIER:
            return round(numeric, 4)
        return 1.0

    row = {str(key).strip().lower(): value for key, value in defense.items()}

    for key in (
        "position_adjustment",
        "position_adjustments",
        "matchup_adjustment",
        "matchup_multiplier",
        "adjustment",
        "adjustments",
        "multiplier",
        "multipliers",
        "factor",
    ):
        if key in row:
            numeric = _numeric(_position_value(row[key], normalized_position))
            if numeric is not None:
                return round(clamp(numeric, MIN_DEFENSE_MULTIPLIER, MAX_DEFENSE_MULTIPLIER), 4)

    for key in (
        "rank_vs_position",
        "position_rank",
        "defense_rank",
        "fantasy_points_allowed_rank",
        "rank",
    ):
        if key in row:
            numeric = _numeric(_position_value(row[key], normalized_position))
            if numeric is not None:
                return round(_rank_multiplier(numeric), 4)

    allowed = None
    for key in ("fantasy_points_allowed", "points_allowed", "fppg_allowed"):
        if key in row:
            allowed = _numeric(_position_value(row[key], normalized_position))
            if allowed is not None:
                break
    average = None
    for key in ("league_average", "league_avg", "average_points_allowed", "avg_points_allowed"):
        if key in row:
            average = _numeric(_position_value(row[key], normalized_position))
            if average is not None:
                break
    if allowed is not None and average is not None and average > 0:
        return round(clamp(allowed / average, MIN_DEFENSE_MULTIPLIER, MAX_DEFENSE_MULTIPLIER), 4)

    for key in ("strength_by_position", "position_strength", "defensive_strength", "strength", "rating"):
        if key not in row:
            continue
        raw = _position_value(row[key], normalized_position)
        if isinstance(raw, str) and _numeric(raw) is None:
            return defensive_strength_adjustment(raw, normalized_position)
        numeric = _numeric(raw)
        if numeric is None:
            continue
        if -1.0 <= numeric <= 1.0:
            # Signed strength uses 0 as neutral; positive means tougher.
            factor = 1.0 - 0.20 * numeric
        elif 0.0 <= numeric <= 100.0:
            factor = 1.20 - 0.004 * numeric
        else:
            continue
        return round(clamp(factor, MIN_DEFENSE_MULTIPLIER, MAX_DEFENSE_MULTIPLIER), 4)

    # Some feeds use the position itself as the top-level key.  Recurse into a
    # nested record (e.g. {"RB": {"rank": 3}}); accept a plausible explicit
    # factor, but do not guess whether an arbitrary larger scalar is rank or
    # strength when the field name supplies no semantics.
    direct = _position_value(defense, normalized_position)
    if isinstance(direct, Mapping) and direct is not defense:
        return defensive_strength_adjustment(direct, normalized_position)
    direct_numeric = _numeric(direct)
    if direct_numeric is not None and MIN_DEFENSE_MULTIPLIER <= direct_numeric <= MAX_DEFENSE_MULTIPLIER:
        return round(direct_numeric, 4)
    return 1.0


def _week_from_container(container: Any, week: int) -> Any:
    if isinstance(container, Mapping):
        if week in container:
            return container[week]
        if str(week) in container:
            return container[str(week)]
        return None
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
        for entry in container:
            if isinstance(entry, Mapping) and int(safe_float(entry.get("week"), 0.0)) == week:
                return entry
        return container[week - 1] if len(container) >= week else None
    return None


def _schedule_entry(player: Mapping[str, Any], week: int) -> Any:
    for key in ("schedule", "weekly_schedule", "matchups", "opponents", "opponent_by_week"):
        entry = _week_from_container(player.get(key), week)
        if entry is not None:
            return entry
    return None


def weekly_opponent(player: Any, week: int) -> str:
    """Return the supplied opponent abbreviation for a week, or ``"TBD"``."""
    row = _as_mapping(player)
    week_number = _validated_week(week)
    entry = _schedule_entry(row, week_number)
    if isinstance(entry, Mapping):
        for key in ("opponent", "opp", "opponent_team"):
            if entry.get(key) is not None:
                value = str(entry[key]).strip().upper()
                return "BYE" if value in _BYE_MARKERS else (value or "TBD")
    elif entry is not None:
        value = str(entry).strip().upper()
        return "BYE" if value in _BYE_MARKERS else (value or "TBD")

    if _is_bye_week(row, week_number):
        return "BYE"
    value = str(row.get("opponent") or row.get("opp") or row.get("opponent_team") or "").strip().upper()
    return value or "TBD"


def _bye_weeks(player: Mapping[str, Any]) -> set[int]:
    raw = player.get("bye_week", player.get("bye_weeks"))
    values: Sequence[Any]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = raw
    else:
        values = (raw,)
    result: set[int] = set()
    for value in values:
        number = int(safe_float(value, 0.0))
        if number in WEEKS:
            result.add(number)
    return result


def _is_bye_week(player: Mapping[str, Any], week: int) -> bool:
    if week in _bye_weeks(player):
        return True
    entry = _schedule_entry(player, week)
    if isinstance(entry, Mapping):
        if entry.get("is_bye") is True or entry.get("bye") is True:
            return True
        opponent = entry.get("opponent", entry.get("opp", entry.get("opponent_team")))
        return str(opponent or "").strip().upper() in _BYE_MARKERS
    return str(entry or "").strip().upper() in _BYE_MARKERS


def _defense_for_week(player: Mapping[str, Any], week: int) -> Any:
    entry = _schedule_entry(player, week)
    if isinstance(entry, Mapping):
        for key in ("defense", "opponent_defense", "defensive_strength", "matchup"):
            if entry.get(key) is not None:
                return {key: entry[key]} if key == "defensive_strength" else entry[key]
        if any(
            key in entry
            for key in (
                "rank",
                "rank_vs_position",
                "position_rank",
                "defense_rank",
                "adjustment",
                "multiplier",
                "strength",
                "rating",
            )
        ):
            return entry

    opponent = weekly_opponent(player, week)
    for key in ("defense_by_week", "weekly_defenses", "opponent_defenses", "defenses"):
        container = player.get(key)
        by_week = _week_from_container(container, week)
        if by_week is not None:
            return by_week
        if isinstance(container, Mapping):
            for candidate in (opponent, opponent.upper(), opponent.lower()):
                if candidate in container:
                    return container[candidate]

    for key in ("opponent_defense", "defensive_strength", "defense"):
        if player.get(key) is not None:
            return {key: player[key]} if key == "defensive_strength" else player[key]
    return None


def _season_projection(player: Mapping[str, Any], scoring_mode: str) -> float | None:
    mode = _scoring_mode(scoring_mode)
    settings = LeagueSettings(scoring_mode=mode)
    projection = projected_points(player, settings)
    if projection is not None:
        return max(0.0, projection)

    # A projection baked in another reception format can be translated when
    # the season reception total is available.  This keeps the season forecast
    # as the baseline instead of silently substituting last year's raw score.
    direct = None
    for key in ("projection", "expected_fantasy_points"):
        value = player.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            direct = float(value)
            break
    if direct is None:
        return None
    source_mode = _scoring_mode(player.get("scoring_mode") or mode)
    if source_mode != mode and source_mode in _RECEPTION_POINTS and mode in _RECEPTION_POINTS:
        receptions = max(0.0, safe_float(player.get("receptions"), 0.0))
        direct += (_RECEPTION_POINTS[mode] - _RECEPTION_POINTS[source_mode]) * receptions
    return max(0.0, direct)


def _active_week_count(player: Mapping[str, Any]) -> int:
    bye_count = len(_bye_weeks(player))
    if bye_count:
        return max(1, REGULAR_SEASON_WEEKS - bye_count)
    # If a full schedule explicitly marks a bye, honor it even when bye_week
    # was omitted.  Otherwise distribute across all 18 displayed weeks so the
    # curve does not overstate the authoritative season total.
    scheduled_byes = sum(1 for week in WEEKS if _is_bye_week(player, week))
    return max(1, REGULAR_SEASON_WEEKS - scheduled_byes)


def weekly_confidence(player: Any) -> float:
    """Return the player's season projection confidence on a 0-1 scale."""
    row = _as_mapping(player)
    raw = row.get("projection_confidence", row.get("confidence"))
    if isinstance(raw, Mapping):
        raw = raw.get("score", raw.get("value", raw.get("confidence")))
    numeric = _numeric(raw)
    if numeric is not None:
        if numeric > 1.0:
            numeric /= 100.0
        return round(clamp(numeric, 0.0, 1.0), 3)

    games = safe_float(row.get("prior_games_played", row.get("games_played")), 0.0)
    if games > 0:
        sample = clamp(games / NFL_GAMES_PER_TEAM, 0.0, 1.0)
        has_market = row.get("adp") is not None
        return round(clamp(0.35 + 0.45 * sample + (0.20 if has_market else 0.0), 0.0, 1.0), 3)
    return 0.60


def weekly_volatility(player: Any) -> float:
    """Return the volatility-curve amplitude implied by confidence.

    A fully supported projection still moves a modest 3% around its baseline;
    a zero-confidence projection can move 18%.  The function returns the
    amplitude, not a random draw, so projections remain reproducible.
    """
    confidence = weekly_confidence(player)
    return round(0.03 + 0.15 * (1.0 - confidence), 4)


def _phase(player: Mapping[str, Any]) -> float:
    identity = str(player.get("player_id") or player.get("id") or player.get("name") or "player")
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return fraction * 2.0 * math.pi


def _volatility_multiplier(player: Mapping[str, Any], week: int) -> float:
    phase = _phase(player)
    wave = (
        math.sin((2.0 * math.pi * (week - 1) / 6.0) + phase)
        + 0.35 * math.sin((2.0 * math.pi * (week - 1) / 3.0) + phase / 2.0)
    ) / 1.35
    return max(0.65, 1.0 + weekly_volatility(player) * wave)


def bye_week_projection(player: Any) -> float:
    """Return the guaranteed fantasy-point projection for a bye week."""
    # Validate the public input consistently even though a bye is always zero.
    _as_mapping(player)
    return 0.0


def _unadjusted_weekly_points(player: Mapping[str, Any], week: int, scoring_mode: str) -> float:
    if _is_bye_week(player, week):
        return bye_week_projection(player)

    season_total = _season_projection(player, scoring_mode)
    if season_total is not None:
        baseline = season_total / _active_week_count(player)
    else:
        # A raw stat line with no season projection is conventionally a single
        # matchup projection in this engine (the adapter fixtures use exactly
        # this shape), so score it as one week rather than divide it by 18.
        settings = LeagueSettings(scoring_mode=_scoring_mode(scoring_mode))
        baseline = float(
            calculate_fantasy_points(
                player,
                mode=settings.scoring_mode,
                custom_rules=settings.custom_rules,
            )["total_points"]
        )
    return max(0.0, baseline * _volatility_multiplier(player, week))


def matchup_adjusted_projection(player: Any, week: int, scoring_mode: str) -> float:
    """Return one week's projection after bye, volatility, and matchup effects."""
    row = _as_mapping(player)
    week_number = _validated_week(week)
    if _is_bye_week(row, week_number):
        return bye_week_projection(row)
    base = _unadjusted_weekly_points(row, week_number, _scoring_mode(scoring_mode))
    adjustment = defensive_strength_adjustment(_defense_for_week(row, week_number), row.get("position", ""))
    return round(max(0.0, base * adjustment), 2)


def weekly_points(player: Any, week: int) -> float:
    """Return final projected points for ``week`` using the player's scoring mode."""
    row = _as_mapping(player)
    return matchup_adjusted_projection(row, week, _scoring_mode(row.get("scoring_mode") or "ppr"))


def _confidence_for_week(player: Mapping[str, Any], week: int) -> float:
    if _is_bye_week(player, week):
        # Zero points on a known bye is more certain than an on-field outcome.
        return 1.0

    for key in ("weekly_confidence", "confidence_by_week"):
        explicit = _week_from_container(player.get(key), week)
        numeric = _numeric(explicit)
        if numeric is not None:
            if numeric > 1.0:
                numeric /= 100.0
            return round(clamp(numeric, 0.0, 1.0), 3)

    base = weekly_confidence(player)
    phase = _phase(player)
    wave = abs(math.sin((2.0 * math.pi * (week - 1) / 6.0) + phase))
    volatility_penalty = weekly_volatility(player) * wave * 0.35
    horizon_penalty = 0.04 * ((week - 1) / (REGULAR_SEASON_WEEKS - 1))
    # Known opponent data removes a small uncertainty penalty.  This makes the
    # confidence curve explainable without claiming matchup data is predictive
    # enough to raise confidence above the season model itself.
    matchup_penalty = 0.0 if _defense_for_week(player, week) is not None else 0.02
    return round(clamp(base - volatility_penalty - horizon_penalty - matchup_penalty, 0.0, 1.0), 3)


def build_weekly_projection(player: Any, scoring_mode: str | None = None) -> dict[int, dict[str, float]]:
    """Build the complete 18-week ``{week: {points, confidence}}`` curve."""
    row = _as_mapping(player)
    mode = _scoring_mode(scoring_mode or row.get("scoring_mode") or "ppr")
    return {
        week: {
            "points": matchup_adjusted_projection(row, week, mode),
            "confidence": _confidence_for_week(row, week),
        }
        for week in WEEKS
    }


def weekly_matchups(player: Any) -> dict[int, dict[str, Any]]:
    """Return display metadata for each matchup without changing output shape."""
    row = _as_mapping(player)
    position = str(row.get("position") or "").strip().upper()
    result: dict[int, dict[str, Any]] = {}
    for week in WEEKS:
        defense = _defense_for_week(row, week)
        result[week] = {
            "opponent": weekly_opponent(row, week),
            "defensive_adjustment": 0.0 if _is_bye_week(row, week) else defensive_strength_adjustment(defense, position),
            "has_defense_data": defense is not None,
        }
    return result


__all__ = [
    "REGULAR_SEASON_WEEKS",
    "WEEKS",
    "build_weekly_projection",
    "bye_week_projection",
    "defensive_strength_adjustment",
    "matchup_adjusted_projection",
    "weekly_confidence",
    "weekly_matchups",
    "weekly_opponent",
    "weekly_points",
    "weekly_volatility",
]
