"""Positional hot-zones and dead-zones, measured from real ADP data.

Usage::

    from fantasy.positional_pressure import build_positional_pressure

    pressure = build_positional_pressure(players, n_teams=12, rounds=14)
    pressure.hot_zone_multiplier("WR", 3)    # > 1.0 where WRs really go
    pressure.dead_zone_penalty("RB", 6)      # < 1.0 where RBs really don't

Every number here is derived from the ADP already attached to the pool -- the
share of each position's players whose ADP falls in each round -- rather than
from hard-coded round ranges. Feeding it a real *historical* snapshot (see
:mod:`fantasy.historical_adp`) yields that season's real zones; feeding it
the present-day fused board yields today's. Nothing is assumed about which
rounds "should" be WR waves or RB dead zones; if the data says RBs cluster in
rounds 1-3 and thin out by round 6, that falls out of the measurement.

A separate, deliberate use: ``starter_saturation_penalty`` prices the roster
-construction mistake the holdout exposed. A second QB in a 1-QB league is
worth close to nothing (only one starts), yet a pure best-market-value
ranking will happily take one at ADP 52 -- the simulated user team did
exactly that, drafting Lamar Jackson in round 2 and Patrick Mahomes in round
5. The room's fixed "QB patience" bias avoids this by accident; this makes it
explicit and data-driven instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from fantasy.models import LeagueSettings
from fantasy.utils import safe_float

#: How far a round's measured share may move a weight, in each direction.
HOT_ZONE_MAX_BOOST = 0.40
DEAD_ZONE_MAX_PENALTY = 0.35

#: Multiplier applied once a position's *starting* requirement is already
#: filled and the position cannot absorb the extra player into a FLEX slot.
#: A backup QB/K/DST is close to dead weight in a standard league; a 3rd WR
#: is not, because FLEX exists.
SATURATED_NON_FLEX_PENALTY = 0.45


@dataclass
class PositionalPressure:
    """Measured per-position, per-round draft pressure."""

    hot: dict[str, dict[int, float]] = field(default_factory=dict)
    dead: dict[str, dict[int, float]] = field(default_factory=dict)
    rounds: int = 0

    def hot_zone_multiplier(self, position: str, round_number: int) -> float:
        """>= 1.0 in rounds where this position genuinely clusters."""
        return self.hot.get(str(position).strip().upper(), {}).get(int(round_number), 1.0)

    def dead_zone_penalty(self, position: str, round_number: int) -> float:
        """<= 1.0 in rounds where this position is genuinely thin."""
        return self.dead.get(str(position).strip().upper(), {}).get(int(round_number), 1.0)

    def combined(self, position: str, round_number: int) -> float:
        """Both effects at once -- what callers normally want."""
        return self.hot_zone_multiplier(position, round_number) * self.dead_zone_penalty(position, round_number)

    def zones(self, position: str) -> dict[str, list[int]]:
        """The rounds this position is hot in and dead in, for reporting."""
        position = str(position).strip().upper()
        return {
            "hot_rounds": sorted(r for r, v in self.hot.get(position, {}).items() if v > 1.0),
            "dead_rounds": sorted(r for r, v in self.dead.get(position, {}).items() if v < 1.0),
        }


def build_positional_pressure(
    players: list[dict[str, Any]],
    n_teams: int = 12,
    rounds: int = 14,
) -> PositionalPressure:
    """Measure hot/dead zones from the ADP actually attached to ``players``.

    For each position, the share of its players whose ADP lands in each round
    is compared against that position's own average share across rounds. A
    round carrying more than its share is "hot"; less is "dead". Positions
    with no ADP data at all get no entries, so :class:`PositionalPressure`
    returns a neutral 1.0 for them rather than an invented adjustment.
    """
    if n_teams < 1 or rounds < 1:
        raise ValueError("n_teams and rounds must both be >= 1")

    by_position: dict[str, Counter[int]] = {}
    for player in players:
        adp = player.get("adp")
        if adp is None:
            continue
        position = str(player.get("position", "")).strip().upper()
        round_number = int((safe_float(adp) - 1) // n_teams) + 1
        if not position or not 1 <= round_number <= rounds:
            continue
        by_position.setdefault(position, Counter())[round_number] += 1

    hot: dict[str, dict[int, float]] = {}
    dead: dict[str, dict[int, float]] = {}
    for position, counts in by_position.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        # Expected share if the position were spread evenly across the rounds
        # it actually appears in.
        occupied_rounds = max(1, len([r for r in range(1, rounds + 1) if counts.get(r, 0) > 0]))
        expected_share = 1.0 / occupied_rounds
        hot[position] = {}
        dead[position] = {}
        for round_number in range(1, rounds + 1):
            share = counts.get(round_number, 0) / total
            ratio = share / expected_share if expected_share > 0 else 1.0
            if ratio >= 1.0:
                hot[position][round_number] = round(1.0 + min(HOT_ZONE_MAX_BOOST, (ratio - 1.0) * HOT_ZONE_MAX_BOOST), 3)
                dead[position][round_number] = 1.0
            else:
                hot[position][round_number] = 1.0
                dead[position][round_number] = round(1.0 - min(DEAD_ZONE_MAX_PENALTY, (1.0 - ratio) * DEAD_ZONE_MAX_PENALTY), 3)
    return PositionalPressure(hot=hot, dead=dead, rounds=rounds)


def _flex_eligible(settings: LeagueSettings) -> set[str]:
    return {str(position).strip().upper() for position in settings.flex_eligible}


def starter_saturation_penalty(
    position: str,
    roster_counts: dict[str, int],
    league_settings: dict[str, Any] | LeagueSettings | None = None,
) -> float:
    """Discount a position whose starting slots are already filled.

    Returns :data:`SATURATED_NON_FLEX_PENALTY` for a non-FLEX-eligible
    position (QB/K/DST) already at its starting requirement -- a 2nd QB in a
    1-QB league genuinely cannot be started, so paying market price for one is
    a real error, and the holdout caught the simulated user team doing exactly
    that. FLEX-eligible positions (RB/WR/TE) are never penalized here: extra
    RBs and WRs have real starting value through the FLEX slot and as injury
    cover, and the existing need multiplier already prices them.
    """
    settings = league_settings if isinstance(league_settings, LeagueSettings) else LeagueSettings(**(league_settings or {}))
    position = str(position).strip().upper()
    if position in _flex_eligible(settings):
        return 1.0
    required = settings.roster_requirements.starting_slots().get(position, 0)
    if required <= 0:
        return 1.0
    have = roster_counts.get(position, 0)
    return SATURATED_NON_FLEX_PENALTY if have >= required else 1.0


def describe_zones(pressure: PositionalPressure) -> dict[str, dict[str, list[int]]]:
    """All measured zones, per position -- handy for tests and reporting."""
    return {position: pressure.zones(position) for position in sorted(pressure.hot)}
