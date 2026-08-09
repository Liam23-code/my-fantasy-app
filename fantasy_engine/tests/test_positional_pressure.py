"""Tests for fantasy.positional_pressure: measured hot/dead zones and saturation."""

from __future__ import annotations

import pytest

from fantasy.positional_pressure import (
    SATURATED_NON_FLEX_PENALTY,
    build_positional_pressure,
    describe_zones,
    starter_saturation_penalty,
)

SETTINGS = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
}


def _p(name, position, adp):
    return {"player_id": name, "name": name, "position": position, "adp": adp}


# --- measured zones ------------------------------------------------------------


def test_a_position_clustered_in_one_round_is_hot_there_and_dead_elsewhere():
    # 20 RBs all inside round 1 (picks 1-12) plus a couple far later.
    players = [_p(f"RB{i}", "RB", 1.0 + (i % 12)) for i in range(20)]
    players += [_p("LateRB1", "RB", 100.0), _p("LateRB2", "RB", 110.0)]
    pressure = build_positional_pressure(players, n_teams=12, rounds=14)
    assert pressure.hot_zone_multiplier("RB", 1) > 1.0
    assert pressure.dead_zone_penalty("RB", 9) < 1.0


def test_hot_and_dead_are_mutually_exclusive_per_round():
    players = [_p(f"WR{i}", "WR", 1.0 + i * 3) for i in range(30)]
    pressure = build_positional_pressure(players, n_teams=12, rounds=14)
    for round_number in range(1, 15):
        hot = pressure.hot_zone_multiplier("WR", round_number)
        dead = pressure.dead_zone_penalty("WR", round_number)
        assert hot == 1.0 or dead == 1.0  # never both adjusted at once


def test_multipliers_stay_within_documented_bounds():
    from fantasy.positional_pressure import DEAD_ZONE_MAX_PENALTY, HOT_ZONE_MAX_BOOST

    players = [_p(f"TE{i}", "TE", 1.0) for i in range(50)] + [_p("Lone", "TE", 150.0)]
    pressure = build_positional_pressure(players, n_teams=12, rounds=14)
    for round_number in range(1, 15):
        assert 1.0 <= pressure.hot_zone_multiplier("TE", round_number) <= 1.0 + HOT_ZONE_MAX_BOOST
        assert 1.0 - DEAD_ZONE_MAX_PENALTY <= pressure.dead_zone_penalty("TE", round_number) <= 1.0


def test_positions_without_adp_data_are_neutral_not_penalized():
    """No measurement must never become an invented adjustment."""
    pressure = build_positional_pressure([_p("RB1", "RB", 5.0)], n_teams=12, rounds=14)
    assert pressure.hot_zone_multiplier("DST", 3) == 1.0
    assert pressure.dead_zone_penalty("DST", 3) == 1.0
    assert pressure.combined("DST", 3) == 1.0


def test_players_without_adp_are_skipped_entirely():
    players = [_p("HasADP", "RB", 5.0), {"player_id": "x", "name": "NoADP", "position": "RB"}]
    pressure = build_positional_pressure(players, n_teams=12, rounds=14)
    assert pressure.hot_zone_multiplier("RB", 1) >= 1.0  # built from the one real data point


def test_empty_pool_yields_neutral_pressure():
    pressure = build_positional_pressure([], n_teams=12, rounds=14)
    assert pressure.combined("RB", 1) == 1.0
    assert describe_zones(pressure) == {}


@pytest.mark.parametrize(("teams", "rounds"), [(0, 14), (12, 0), (-1, 5)])
def test_invalid_shape_is_rejected(teams, rounds):
    with pytest.raises(ValueError):
        build_positional_pressure([], n_teams=teams, rounds=rounds)


def test_describe_zones_reports_hot_and_dead_rounds():
    players = [_p(f"RB{i}", "RB", 1.0 + (i % 12)) for i in range(20)] + [_p("Late", "RB", 120.0)]
    zones = describe_zones(build_positional_pressure(players, n_teams=12, rounds=14))
    assert 1 in zones["RB"]["hot_rounds"]


# --- starter saturation (the roster-construction fix) -------------------------


def test_second_qb_in_a_one_qb_league_is_penalized():
    """A backup QB literally cannot be started -- market price for one is an error."""
    assert starter_saturation_penalty("QB", {"QB": 1}, SETTINGS) == SATURATED_NON_FLEX_PENALTY
    assert starter_saturation_penalty("QB", {"QB": 0}, SETTINGS) == 1.0


def test_flex_eligible_positions_are_never_saturation_penalized():
    """Extra RB/WR/TE have real value through FLEX and as injury cover."""
    for position in ("RB", "WR", "TE"):
        assert starter_saturation_penalty(position, {position: 5}, SETTINGS) == 1.0


def test_kicker_and_dst_saturate_like_qb():
    assert starter_saturation_penalty("K", {"K": 1}, SETTINGS) == SATURATED_NON_FLEX_PENALTY
    assert starter_saturation_penalty("DST", {"DST": 1}, SETTINGS) == SATURATED_NON_FLEX_PENALTY


def test_position_the_league_does_not_start_is_neutral():
    settings = {**SETTINGS, "roster_requirements": {"QB": 0, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 6}}
    assert starter_saturation_penalty("QB", {"QB": 3}, settings) == 1.0


def test_saturation_penalty_accepts_missing_settings():
    assert 0.0 < starter_saturation_penalty("QB", {"QB": 1}, None) <= 1.0
