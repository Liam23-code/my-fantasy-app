"""Tests for deterministic positional scarcity and draft adjustments."""

from __future__ import annotations

import copy

import pytest

from quant.scarcity_engine import (
    compute_draft_value,
    compute_positional_scarcity,
    draft_value_adjustment,
    positional_depth_curve,
    positional_depth_curves,
    replacement_level_model,
    scarcity_multiplier,
    scarcity_multipliers,
)


def _player(player_id: str, position: str, projection: float, adp: float | None = None) -> dict:
    row = {
        "player_id": player_id,
        "name": player_id.upper(),
        "position": position,
        "projection": projection,
    }
    if adp is not None:
        row["adp"] = adp
    return row


def test_depth_curves_rank_each_position_by_projection_without_mutation():
    players = [_player("rb2", "RB", 180), _player("wr1", "WR", 210), _player("rb1", "RB", 230)]
    original = copy.deepcopy(players)

    curves = positional_depth_curves(players)

    assert [row["player_id"] for row in curves["RB"]] == ["rb1", "rb2"]
    assert curves["RB"][0]["points_drop_to_next"] == pytest.approx(50.0)
    assert curves["RB"][0]["rank"] == 1
    assert players == original


def test_singular_depth_curve_requires_position_for_mixed_pool():
    players = [_player("rb", "RB", 100), _player("wr", "WR", 100)]
    with pytest.raises(ValueError, match="position is required"):
        positional_depth_curve(players)
    assert positional_depth_curve(players, "wr")[0]["player_id"] == "wr"


def test_replacement_model_allocates_flex_to_best_remaining_player():
    players = [
        _player("rb1", "RB", 30),
        _player("rb2", "RB", 20),
        _player("rb3", "RB", 10),
        _player("wr1", "WR", 25),
        _player("wr2", "WR", 15),
        _player("wr3", "WR", 5),
    ]
    levels = replacement_level_model(players, {"RB": 1, "WR": 1, "FLEX": 1}, teams=1)

    # RB2 wins the one flex job. Replacement is consequently RB3, while WR2
    # remains the first player outside the WR starter allocation.
    assert levels == {"RB": 10.0, "WR": 15.0}


def test_steep_replacement_cliff_is_scarcer_than_deep_position():
    players = [
        _player("te1", "TE", 100),
        _player("te2", "TE", 80),
        _player("te3", "TE", 20),
        _player("wr1", "WR", 100),
        _player("wr2", "WR", 80),
        _player("wr3", "WR", 75),
    ]
    multipliers = scarcity_multipliers(players, replacement_rank=3, teams=1)
    assert multipliers["TE"] > multipliers["WR"]
    assert 0.9 <= multipliers["TE"] <= 1.5


def test_single_scarcity_multiplier_is_neutral_for_absent_position():
    assert scarcity_multiplier("QB", [_player("rb", "RB", 100)], teams=1) == 1.0


def test_scarcity_envelope_has_position_and_player_indexes():
    players = [_player("a", "RB", 200), _player("b", "RB", 150)]
    result = compute_positional_scarcity(players, replacement_rank=2, teams=1)

    assert result["metric"] == "positional_scarcity"
    assert result["by_player"]["a"]["value_over_replacement"] == pytest.approx(50.0)
    assert result["by_position"]["RB"]["replacement_level"] == pytest.approx(150.0)
    assert result["metadata"]["player_count"] == 2


def test_draft_value_rewards_player_falling_past_adp():
    players = [_player("fall", "RB", 200, adp=10), _player("on_time", "RB", 200, adp=30)]
    result = compute_draft_value(players, current_pick=30, teams=1, replacement_rank=2)

    assert result["by_player"]["fall"]["market_adjustment"] > result["by_player"]["on_time"]["market_adjustment"]
    assert result["by_player"]["fall"]["draft_value"] > result["by_player"]["on_time"]["draft_value"]


def test_one_player_draft_adjustment_adds_target_to_pool_when_missing():
    target = _player("target", "WR", 220, adp=20)
    pool = [_player("other", "WR", 180, adp=40)]

    result = draft_value_adjustment(target, pool, current_pick=25, teams=1)

    assert result["player_id"] == "target"
    assert result["projection"] == pytest.approx(220.0)
    assert "draft_value_score" in result


def test_keyed_mapping_input_gets_stable_ids_and_deduplicates():
    players = {
        "b": {"name": "B", "pos": "rb", "projected_points": 100},
        "a": {"name": "A", "pos": "rb", "projected_points": 120},
    }
    curves = positional_depth_curves(players)
    assert [row["player_id"] for row in curves["RB"]] == ["a", "b"]

    duplicated = [
        _player("same", "RB", 100),
        _player("same", "RB", 999),
    ]
    assert len(positional_depth_curves(duplicated)["RB"]) == 1


@pytest.mark.parametrize("teams", [0, -1, 1.5, True, "many"])
def test_invalid_team_counts_are_rejected(teams):
    with pytest.raises(ValueError, match="teams"):
        replacement_level_model([_player("x", "RB", 100)], teams=teams)


def test_invalid_players_and_replacement_rank_are_rejected():
    with pytest.raises(ValueError, match="missing a position"):
        positional_depth_curves([{"player_id": "x", "projection": 10}])
    with pytest.raises(ValueError, match="replacement_rank"):
        compute_positional_scarcity([_player("x", "RB", 100)], replacement_rank=0)

