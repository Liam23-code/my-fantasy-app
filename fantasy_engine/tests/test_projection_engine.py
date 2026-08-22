"""Tests for the advanced projection ensemble."""

from __future__ import annotations

import pytest

from projections.projection_engine import (
    ProjectionEngine,
    breakout_bust_probability_model,
    compute_final_projection,
    injury_adjusted_projection_model,
    regression_to_mean_model,
    weighted_historical_model,
)


def _player(**overrides) -> dict:
    player = {
        "player_id": "rb-1",
        "name": "Quant Back",
        "position": "RB",
        "team": "DEN",
        "projection": 240.0,
        "expected_fantasy_points": 240.0,
        "historical_points": [180.0, 210.0, 250.0],
        "games_played": 17,
        "expected_games": 17,
        "usage_rate": 0.72,
        "projection_confidence": 0.82,
        "floor": 165.0,
        "ceiling": 310.0,
        "adp": 28.0,
    }
    player.update(overrides)
    return player


def test_weighted_history_favors_the_most_recent_observation():
    result = weighted_historical_model(_player(), decay=0.7)

    simple_average = sum([180.0, 210.0, 250.0]) / 3
    assert result["estimate"] > simple_average
    assert result["components"]["samples"] == 3


def test_regression_to_mean_shrinks_a_small_sample_more_than_a_full_sample():
    small = regression_to_mean_model(
        _player(games_played=2),
        base_projection=300.0,
        position_mean=150.0,
    )
    full = regression_to_mean_model(
        _player(games_played=17),
        base_projection=300.0,
        position_mean=150.0,
    )

    assert small["estimate"] < full["estimate"] < 300.0


def test_injury_model_reduces_unavailable_player_projection():
    healthy = injury_adjusted_projection_model(_player(), base_projection=240.0)
    injured = injury_adjusted_projection_model(
        _player(injury_status="OUT", expected_games=9),
        base_projection=240.0,
    )

    assert injured["estimate"] < healthy["estimate"]
    assert injured["components"]["multiplier"] < 1.0


def test_breakout_and_bust_probabilities_are_bounded():
    probabilities = breakout_bust_probability_model(_player(age=22, opportunity_growth=0.35))

    assert 0.0 <= probabilities["breakout_probability"] <= 1.0
    assert 0.0 <= probabilities["bust_probability"] <= 1.0


def test_final_projection_is_structured_auditable_and_input_immutable():
    player = _player()
    original = dict(player)

    result = compute_final_projection(player)

    assert player == original
    assert result["player_id"] == "rb-1"
    assert result["floor"] <= result["final_projection"] <= result["ceiling"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["components"]) >= {
        "historical",
        "usage",
        "matchup",
        "regression",
        "injury",
        "volatility",
        "probabilities",
    }


def test_projection_engine_resolves_ids_from_its_pool():
    engine = ProjectionEngine([_player()])

    result = engine.compute_final_projection("rb-1")

    assert result["name"] == "Quant Back"
    assert result["final_projection"] > 0


def test_unknown_player_id_raises_clear_error():
    with pytest.raises(KeyError, match="No normalized data"):
        compute_final_projection("definitely-not-registered", [])
