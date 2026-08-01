"""Pluggable projection provider backing GET /player/{player_id}/projection.

This package is deliberately decoupled from any specific NFL data source
(see the migration guide in the root README). Out of the box, the API serves
a small bundled sample so it runs standalone; a real deployment overrides the
provider with a callable backed by ``project_nfl_player`` or any other
source::

    from api.data import set_projection_provider
    from modules.nfl_projections import project_nfl_player  # sibling repo

    set_projection_provider(lambda player_id: project_nfl_player(player_id))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fantasy.adapter import normalize_projection

_SAMPLE_PROJECTIONS: dict[str, dict[str, Any]] = {
    "nfl:player:00-0034796": {
        "player_id": "nfl:player:00-0034796",
        "name": "Lamar Jackson",
        "position": "QB",
        "team": "BAL",
        "passing_yards": 245.0,
        "passing_tds": 1.8,
        "interceptions": 0.4,
        "rushing_yards": 65.0,
        "rushing_tds": 0.5,
        "floor": 8.2,
        "median": 15.6,
        "ceiling": 28.4,
        "drivers": ["pace", "red_zone_usage", "injury_risk"],
    },
    "nfl:player:00-0034857": {
        "player_id": "nfl:player:00-0034857",
        "name": "Josh Allen",
        "position": "QB",
        "team": "BUF",
        "passing_yards": 219.5,
        "passing_tds": 1.65,
        "interceptions": 0.35,
        "rushing_yards": 31.2,
        "rushing_tds": 0.71,
        "floor": 14.0,
        "median": 22.5,
        "ceiling": 34.0,
        "drivers": ["pace", "goal_line_role"],
    },
    "nfl:player:00-0034844": {
        "player_id": "nfl:player:00-0034844",
        "name": "Saquon Barkley",
        "position": "RB",
        "team": "PHI",
        "rushing_yards": 125.3,
        "rushing_tds": 0.81,
        "receptions": 2.1,
        "receiving_yards": 17.4,
        "receiving_tds": 0.12,
        "floor": 12.0,
        "median": 21.9,
        "ceiling": 34.5,
        "drivers": ["volume", "red_zone_share", "matchup"],
    },
    "nfl:player:00-0033280": {
        "player_id": "nfl:player:00-0033280",
        "name": "Christian McCaffrey",
        "position": "RB",
        "team": "SF",
        "rushing_yards": 91.2,
        "rushing_tds": 0.85,
        "receptions": 3.9,
        "receiving_yards": 32.5,
        "receiving_tds": 0.42,
        "floor": 15.0,
        "median": 24.0,
        "ceiling": 38.0,
        "drivers": ["target_share", "goal_line_role"],
    },
    "nfl:player:00-0039075": {
        "player_id": "nfl:player:00-0039075",
        "name": "Puka Nacua",
        "position": "WR",
        "team": "LA",
        "receptions": 6.5,
        "receiving_yards": 92.0,
        "receiving_tds": 0.43,
        "floor": 9.0,
        "median": 17.5,
        "ceiling": 27.0,
        "drivers": ["target_share", "air_yards"],
    },
    "nfl:player:00-0030506": {
        "player_id": "nfl:player:00-0030506",
        "name": "Travis Kelce",
        "position": "TE",
        "team": "KC",
        "receptions": 5.6,
        "receiving_yards": 47.4,
        "receiving_tds": 0.17,
        "floor": 4.0,
        "median": 10.5,
        "ceiling": 18.0,
        "drivers": ["red_zone_targets"],
    },
}

ProjectionProvider = Callable[[str], dict[str, Any] | None]


def _default_provider(player_id: str) -> dict[str, Any] | None:
    raw = _SAMPLE_PROJECTIONS.get(player_id)
    return normalize_projection(raw) if raw else None


_provider: ProjectionProvider = _default_provider


def set_projection_provider(provider: ProjectionProvider) -> None:
    """Override the projection source used by GET /player/{player_id}/projection."""
    global _provider
    _provider = provider


def reset_projection_provider() -> None:
    """Restore the bundled sample provider (mainly for tests)."""
    global _provider
    _provider = _default_provider


def get_player_projection(player_id: str) -> dict[str, Any] | None:
    return _provider(player_id)
