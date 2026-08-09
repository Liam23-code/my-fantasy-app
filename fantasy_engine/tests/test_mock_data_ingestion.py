"""Tests for fantasy.mock_data_ingestion: real-ADP-derived market signals."""

from __future__ import annotations

from fantasy.mock_data_ingestion import (
    positional_run_patterns,
    reach_steal_dispersion,
    reach_tolerance_scale,
    round_by_round_trends,
)


def _player(name, position, adp=None, adp_sd=None):
    player = {"player_id": name, "name": name, "position": position}
    if adp is not None:
        player["adp"] = adp
    if adp_sd is not None:
        player["adp_sd"] = adp_sd
    return player


def _tight_cluster():
    return [_player(f"RB{i}", "RB", adp=1.0 + i) for i in range(5)]


# --- positional_run_patterns (re-export) ------------------------------------


def test_positional_run_patterns_matches_identify_adp_clusters():
    from fantasy.draft import identify_adp_clusters

    players = _tight_cluster()
    assert positional_run_patterns(players) == identify_adp_clusters(players)


def test_positional_run_patterns_of_empty_input_is_empty():
    assert positional_run_patterns([]) == []


# --- reach_steal_dispersion --------------------------------------------------


def test_dispersion_averages_real_adp_sd_per_position():
    players = [
        _player("RB1", "RB", adp=1.0, adp_sd=2.0),
        _player("RB2", "RB", adp=2.0, adp_sd=4.0),
        _player("WR1", "WR", adp=3.0, adp_sd=1.0),
    ]
    dispersion = reach_steal_dispersion(players)
    assert dispersion["RB"] == 3.0  # average of 2.0 and 4.0
    assert dispersion["WR"] == 1.0


def test_dispersion_ignores_players_missing_adp_or_adp_sd():
    players = [_player("RB1", "RB", adp=1.0), _player("RB2", "RB", adp_sd=5.0)]
    assert reach_steal_dispersion(players) == {}


def test_dispersion_ignores_kicker_and_dst():
    players = [_player("K1", "K", adp=100.0, adp_sd=10.0)]
    assert reach_steal_dispersion(players) == {}


def test_dispersion_only_considers_top_n_by_adp():
    players = [_player(f"RB{i}", "RB", adp=float(i), adp_sd=float(i)) for i in range(10)]
    dispersion = reach_steal_dispersion(players, top_n=3)
    # Top 3 by ADP (0, 1, 2) have adp_sd 0, 1, 2 -> average 1.0.
    assert dispersion["RB"] == 1.0


# --- reach_tolerance_scale ----------------------------------------------------


def test_reach_tolerance_scale_centers_around_one():
    players = [
        _player("RB1", "RB", adp=1.0, adp_sd=2.0),
        _player("WR1", "WR", adp=2.0, adp_sd=6.0),
    ]
    scale = reach_tolerance_scale(players)
    # Average dispersion is (2+6)/2 = 4; RB scores below 1, WR scores above 1.
    assert scale["RB"] < 1.0
    assert scale["WR"] > 1.0


def test_reach_tolerance_scale_empty_when_no_real_dispersion_data():
    assert reach_tolerance_scale([_player("RB1", "RB", adp=1.0)]) == {}


# --- round_by_round_trends ----------------------------------------------------


def test_round_by_round_trends_buckets_by_adp_and_n_teams():
    players = [_player(f"P{i}", "WR", adp=float(i)) for i in range(24)]
    trends = round_by_round_trends(players, n_teams=12, rounds=5)
    assert len(trends) == 2  # only 24 players -> exactly 2 full rounds of 12
    assert trends[0]["round"] == 1
    assert trends[0]["sample_size"] == 12
    assert trends[0]["position_counts"] == {"WR": 12}


def test_round_by_round_trends_stops_at_real_data_coverage():
    players = [_player(f"P{i}", "RB", adp=float(i)) for i in range(5)]
    trends = round_by_round_trends(players, n_teams=12, rounds=10)
    assert len(trends) == 1  # not even one full round of real data -- no padding
    assert trends[0]["sample_size"] == 5


def test_round_by_round_trends_ignores_players_without_adp():
    players = [_player("NoADP", "RB")]
    assert round_by_round_trends(players, n_teams=12) == []


def test_round_by_round_trends_rejects_invalid_n_teams():
    import pytest

    with pytest.raises(ValueError):
        round_by_round_trends([], n_teams=0)
