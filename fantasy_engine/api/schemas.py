"""Pydantic request/response schemas for the FastAPI surface.

Every list field carries a ``max_length`` and every simulation parameter a
sane upper bound: these are the API's only real attack surface (arbitrary
compute cost via an oversized or degenerate payload), since ``custom_rules``
is pure data and never evaluated as code.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

MAX_PLAYERS_PER_REQUEST = 500
MAX_TRADE_SIDE_SIZE = 20
MAX_CUSTOM_RULE_ENTRIES = 50


class CustomRules(BaseModel):
    multipliers: dict[str, float] = Field(default_factory=dict, max_length=MAX_CUSTOM_RULE_ENTRIES)
    bonuses: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_CUSTOM_RULE_ENTRIES)


class ScoreRequest(BaseModel):
    projection: dict[str, Any]
    mode: str = "ppr"
    custom_rules: CustomRules | None = None
    bonuses: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "projection": {"passing_yards": 245.0, "passing_tds": 1.8, "rushing_yards": 65.0, "rushing_tds": 0.5},
                "mode": "ppr",
                "bonuses": True,
            }
        }
    }


class ScoreResponse(BaseModel):
    total_points: float
    breakdown: dict[str, float]
    mode: str
    bonuses_applied: list[dict[str, Any]]
    raw_projection: dict[str, Any]


class OptimizeRequest(BaseModel):
    roster: list[dict[str, Any]] | dict[str, Any] = Field(..., max_length=MAX_PLAYERS_PER_REQUEST)
    week_projections: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_PLAYERS_PER_REQUEST)
    league_settings: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "roster": [{"player_id": "rb1", "name": "RB One", "position": "RB", "slot": "RB"}],
                "week_projections": [{"player_id": "rb1", "name": "RB One", "position": "RB", "rushing_yards": 90}],
                "league_settings": {"n_teams": 12, "scoring_mode": "ppr"},
            }
        }
    }


class WaiverRequest(BaseModel):
    league_state: dict[str, Any] = Field(default_factory=dict)
    available_players: list[dict[str, Any]] = Field(..., max_length=MAX_PLAYERS_PER_REQUEST)
    scoring_mode: str | None = None
    budget: float | None = Field(default=None, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "league_state": {"league_settings": {"n_teams": 12, "scoring_mode": "ppr"}, "my_roster": []},
                "available_players": [{"player_id": "fa1", "name": "Free Agent", "position": "WR", "receiving_yards": 40}],
                "budget": 100,
            }
        }
    }


class TradeEvalRequest(BaseModel):
    team_a_players: list[Any] = Field(..., max_length=MAX_TRADE_SIDE_SIZE)
    team_b_players: list[Any] = Field(..., max_length=MAX_TRADE_SIDE_SIZE)
    league_settings: dict[str, Any] = Field(default_factory=dict)
    projections: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_PLAYERS_PER_REQUEST)
    monte_carlo_iterations: int = Field(default=5000, ge=1, le=50000)
    weeks_remaining: int = Field(default=10, ge=1, le=25)
    team_a_roster: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_PLAYERS_PER_REQUEST)
    team_b_roster: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_PLAYERS_PER_REQUEST)
    seed: int | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "team_a_players": [{"player_id": "a", "name": "Give Guy", "position": "RB", "rushing_yards": 60}],
                "team_b_players": [{"player_id": "b", "name": "Receive Guy", "position": "WR", "receiving_yards": 70}],
                "league_settings": {"n_teams": 12, "scoring_mode": "ppr"},
                "monte_carlo_iterations": 2000,
            }
        }
    }
