"""Tests for player values, roster needs, fairness, and trade proposals."""

from __future__ import annotations

import pytest

from quant.trade_engine import (
    compute_trade_value,
    player_value_score,
    positional_balance_score,
    recommend_trades,
    team_need_score,
    trade_fairness_score,
)

REQUIREMENTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}


def _player(player_id: str, position: str, projection: float, **extra) -> dict:
    return {
        "player_id": player_id,
        "name": player_id.replace("_", " ").title(),
        "position": position,
        "projection": projection,
        "projection_confidence": 0.8,
        **extra,
    }


def test_player_value_orders_better_projection_above_weaker_peer():
    high = _player("high", "RB", 260)
    low = _player("low", "RB", 150)
    pool = [high, low, _player("replacement", "RB", 100)]

    assert player_value_score(high, pool)["value_score"] > player_value_score(low, pool)["value_score"]


def test_player_value_applies_health_and_volatility_risk():
    healthy = _player("healthy", "WR", 220, volatility=0.15)
    injured = _player("injured", "WR", 220, volatility=0.8, injury_status="OUT")
    pool = [healthy, injured, _player("replacement", "WR", 100)]

    healthy_value = player_value_score(healthy, pool)
    injured_value = player_value_score(injured, pool)
    assert healthy_value["value_score"] > injured_value["value_score"]
    assert injured_value["health_factor"] < 1.0


def test_batch_trade_value_returns_deterministic_indexes_and_percentiles():
    players = [_player("b", "RB", 150), _player("a", "RB", 250)]
    first = compute_trade_value(players)
    second = compute_trade_value(players)

    assert first == second
    assert first["metric"] == "trade_value"
    assert first["results"][0]["player_id"] == "a"
    assert first["by_player"]["a"]["value_percentile"] == 100.0


def test_team_need_detects_missing_starter_and_accounts_for_injury():
    roster = [
        _player("qb", "QB", 200),
        _player("rb1", "RB", 220),
        _player("rb2", "RB", 180, injury_status="IR"),
        _player("wr1", "WR", 190),
        _player("te", "TE", 140),
    ]

    needs = team_need_score(roster, roster_requirements=REQUIREMENTS)
    assert needs["by_position"]["RB"] > needs["by_position"]["QB"]
    assert needs["by_position"]["WR"] > needs["by_position"]["QB"]
    assert team_need_score(roster, "rb", roster_requirements=REQUIREMENTS)["score"] == needs["by_position"]["RB"]


def test_positional_balance_rewards_complete_roster():
    complete = [
        _player("qb", "QB", 200),
        _player("rb1", "RB", 220),
        _player("rb2", "RB", 180),
        _player("wr1", "WR", 190),
        _player("wr2", "WR", 180),
        _player("te", "TE", 140),
    ]
    incomplete = complete[:2]
    assert positional_balance_score(complete, roster_requirements=REQUIREMENTS)["score"] > positional_balance_score(
        incomplete,
        roster_requirements=REQUIREMENTS,
    )["score"]


def test_equal_trade_is_labeled_fair():
    a = _player("a", "RB", 200)
    b = _player("b", "RB", 200)
    result = trade_fairness_score([a], [b], player_pool=[a, b])

    assert result["fairness_score"] == pytest.approx(100.0)
    assert result["fairness_label"] == "fair"
    assert result["favored_side"] == "even"


def test_unbalanced_trade_identifies_side_receiving_more_value():
    weak = _player("weak", "WR", 100)
    star = _player("star", "WR", 300)

    # Team A sends the weak player and receives Team B's star.
    result = trade_fairness_score([weak], [star], player_pool=[weak, star])
    assert result["fairness_label"] == "favors_team_a"
    assert result["team_a_net_value"] > 0
    assert result["fairness_score"] < 85


def test_trade_fairness_supports_keyed_mapping_sides():
    result = trade_fairness_score(
        {"a": {"name": "A", "position": "TE", "projection": 150}},
        {"b": {"name": "B", "position": "TE", "projection": 150}},
    )
    assert result["fairness_label"] == "fair"


def test_recommendation_engine_targets_a_roster_need():
    roster = [
        _player("qb", "QB", 200),
        _player("rb1", "RB", 220),
        _player("rb2", "RB", 205),
        _player("wr1", "WR", 130),
        _player("te", "TE", 150),
    ]
    pool = [
        _player("wr_upgrade", "WR", 215),
        _player("rb_lateral", "RB", 205),
        _player("te_low", "TE", 80),
    ]

    recommendations = recommend_trades(
        roster,
        pool,
        roster_requirements=REQUIREMENTS,
        max_results=20,
        minimum_fairness=35,
    )

    assert recommendations
    wr_recommendations = [row for row in recommendations if row["receive"]["player_id"] == "wr_upgrade"]
    assert wr_recommendations
    assert max(row["fit_gain"] for row in wr_recommendations) > 0
    assert [row["rank"] for row in recommendations] == list(range(1, len(recommendations) + 1))


def test_recommendations_are_deterministic_and_do_not_offer_owned_players():
    roster = [_player("mine", "RB", 150)]
    pool = [_player("mine", "RB", 999), _player("target", "RB", 155)]
    first = recommend_trades(roster, pool, roster_requirements={"RB": 1}, minimum_fairness=0)
    second = recommend_trades(roster, pool, roster_requirements={"RB": 1}, minimum_fairness=0)
    assert first == second
    assert all(row["receive"]["player_id"] != "mine" for row in first)


def test_invalid_trade_and_limit_inputs_are_rejected():
    player = _player("x", "RB", 100)
    with pytest.raises(ValueError, match="both trade sides"):
        trade_fairness_score([], [player])
    with pytest.raises(ValueError, match="max_results"):
        recommend_trades([player], [player], max_results=-1)
    with pytest.raises(ValueError, match="minimum_fairness"):
        recommend_trades([player], [player], minimum_fairness=101)

