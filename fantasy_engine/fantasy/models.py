"""Pydantic models for the fantasy engine's public/API boundary.

Internal hot paths (scoring a batch of 10k projections, optimizing a lineup)
deliberately work on plain dicts/dataclasses rather than these models -- see
the architecture note in ``README.md``. Pydantic models live here so the API
layer, config files, and anything a user hand-writes gets real validation and
helpful error messages.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

KNOWN_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
FLEX_DEFAULT_ELIGIBLE = ["RB", "WR", "TE"]

#: (min, max) real players a roster should carry at each position, independent
#: of ``RosterRequirements`` -- ``RosterRequirements`` is *starting-lineup*
#: slots (how many you start each week); this is *roster construction* (how
#: many you're allowed to own at all, bench included). FLEX has no entry here
#: because it is not a player position: a team's FLEX starter is just one more
#: RB/WR/TE, already covered by those positions' upper bound. Anything not
#: listed (there is nothing draftable outside ``KNOWN_POSITIONS``) is
#: uncapped.
ROSTER_POSITION_LIMITS: dict[str, tuple[int, int]] = {
    "QB": (1, 2),
    "RB": (2, 4),
    "WR": (3, 5),
    "TE": (1, 2),
    "K": (1, 1),
    "DST": (1, 1),
}


def roster_position_limits(position: str) -> tuple[int, int] | None:
    """(min, max) for ``position``, or ``None`` when it isn't capped."""
    return ROSTER_POSITION_LIMITS.get(str(position).strip().upper())


def roster_cap_reached(position_counts: dict[str, int], position: str) -> bool:
    """True once ``position_counts`` already holds the max allowed at ``position``."""
    limits = roster_position_limits(position)
    if limits is None:
        return False
    return position_counts.get(str(position).strip().upper(), 0) >= limits[1]


def roster_min_required(position: str) -> int:
    """Minimum a roster should carry at ``position`` (0 when uncapped)."""
    limits = roster_position_limits(position)
    return limits[0] if limits else 0


class CanonicalProjection(BaseModel):
    """The one projection shape every fantasy engine module consumes.

    Build this via :func:`fantasy.adapter.normalize_projection` rather than
    constructing it directly from an arbitrary source -- the adapter handles
    the dict/object/``project_nfl_player``-output/minimal-stat-dict cases the
    model itself does not know about.
    """

    model_config = ConfigDict(extra="ignore")

    player_id: str
    name: str
    position: str
    team: str = ""
    opponent: str = ""
    season: int = 0

    passing_yards: float = 0.0
    passing_tds: float = 0.0
    interceptions: float = 0.0
    rushing_yards: float = 0.0
    rushing_tds: float = 0.0
    receptions: float = 0.0
    receiving_yards: float = 0.0
    receiving_tds: float = 0.0
    fumbles_lost: float = 0.0

    expected_fantasy_points: float | None = None
    floor: float | None = None
    median: float | None = None
    ceiling: float | None = None
    drivers: list[str] = Field(default_factory=list)

    @field_validator("position", "team", "opponent", mode="before")
    @classmethod
    def _upper_strip(cls, value: Any) -> str:
        return str(value or "").strip().upper()

    @field_validator("player_id", "name", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> str:
        return str(value if value is not None else "").strip()


class RosterRequirements(BaseModel):
    """Starting-lineup slot counts. ``BENCH``/``IR``/``TAXI`` are not started."""

    QB: int = 1
    RB: int = 2
    WR: int = 2
    TE: int = 1
    FLEX: int = 1
    DST: int = 1
    K: int = 1
    BENCH: int = 6
    IR: int = 0
    TAXI: int = 0

    def starting_slots(self) -> dict[str, int]:
        return {slot: count for slot, count in self.model_dump().items() if slot not in {"BENCH", "IR", "TAXI"} and count > 0}


class LeagueSettings(BaseModel):
    """Everything downstream modules need to know about a league's rules."""

    n_teams: int = 12
    scoring_mode: str = "ppr"
    custom_rules: dict[str, Any] | None = None
    roster_requirements: RosterRequirements = Field(default_factory=RosterRequirements)
    flex_eligible: list[str] = Field(default_factory=lambda: list(FLEX_DEFAULT_ELIGIBLE))
    max_players_per_nfl_team: int | None = None
    is_auction: bool = False
    faab_budget: float | None = 100.0

    @field_validator("flex_eligible", mode="before")
    @classmethod
    def _upper_positions(cls, value: Any) -> list[str]:
        return [str(item).strip().upper() for item in (value or [])]


class RosterPlayer(BaseModel):
    """One player on a fantasy roster, with the context need-based logic uses."""

    player_id: str
    name: str
    position: str
    nfl_team: str = ""
    slot: str = "BENCH"  # QB / RB / WR / TE / FLEX / DST / K / BENCH / IR / TAXI
    bye_week: int | None = None
    injury_status: str | None = None  # e.g. "OUT", "DOUBTFUL", "QUESTIONABLE", "IR"
    ownership_pct: float | None = None
    acquired_via: str = "draft"  # draft / waiver / trade / free_agent

    @field_validator("position", "nfl_team", "slot", mode="before")
    @classmethod
    def _upper_strip(cls, value: Any) -> str:
        return str(value or "").strip().upper()


class Roster(BaseModel):
    """A team's full roster across weeks."""

    team_name: str
    players: list[RosterPlayer] = Field(default_factory=list)

    def starters(self) -> list[RosterPlayer]:
        return [p for p in self.players if p.slot not in {"BENCH", "IR", "TAXI"}]

    def bench(self) -> list[RosterPlayer]:
        return [p for p in self.players if p.slot == "BENCH"]

    def by_position(self, position: str) -> list[RosterPlayer]:
        position = position.strip().upper()
        return [p for p in self.players if p.position == position]

    def is_active(self, player: RosterPlayer) -> bool:
        return player.slot not in {"IR", "TAXI"} and player.injury_status not in {"OUT", "IR"}
