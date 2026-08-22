"""Tests for waiver opportunity, breakout, trends, and priority ranking."""

from __future__ import annotations

import pytest

from quant.waiver_engine import (
    breakout_probability,
    compute_waiver_priority,
    matchup_advantage,
    opportunity_score,
    rank_waiver_priority,
    usage_trend,
    volatility_profile,
)


def _player(player_id: str, position: str = "WR", projection: float = 150, **extra) -> dict:
    return {
        "player_id": player_id,
        "name": player_id.replace("_", " ").title(),
        "position": position,
        "projection": projection,
        **extra,
    }


def test_opportunity_score_rewards_role_and_volume():
    featured = _player(
        "featured",
        snap_share=0.88,
        route_participation=0.9,
        target_share=0.28,
        targets=12,
        red_zone_share=0.4,
        depth_chart_rank=1,
    )
    reserve = _player("reserve", snap_share=0.2, route_participation=0.15, target_share=0.03, targets=1, depth_chart_rank=4)

    assert opportunity_score(featured)["score"] > opportunity_score(reserve)["score"]
    assert opportunity_score(featured)["data_coverage"] > 0


def test_usage_trend_detects_up_down_and_flat_curves():
    rising = usage_trend(_player("up", usage_history=[0.2, 0.3, 0.45, 0.6, 0.75]))
    falling = usage_trend(_player("down", usage_history=[0.8, 0.7, 0.55, 0.35, 0.2]))
    flat = usage_trend(_player("flat", usage_history=[0.5, 0.5, 0.5, 0.5]))

    assert rising["direction"] == "up"
    assert falling["direction"] == "down"
    assert flat["direction"] == "flat"
    assert rising["score"] > falling["score"]


def test_usage_trend_accepts_weekly_stat_mappings():
    player = _player(
        "mapped",
        weekly_stats={
            "2": {"snap_share": 0.7, "targets": 9},
            "1": {"snap_share": 0.3, "targets": 3},
        },
    )
    result = usage_trend(player)
    assert result["sample_size"] == 2
    assert result["direction"] == "up"


def test_breakout_probability_combines_youth_efficiency_and_growing_role():
    breakout = _player(
        "breakout",
        age=22,
        experience=1,
        snap_share=0.82,
        target_share=0.27,
        yards_per_route_run=2.8,
        usage_history=[0.25, 0.4, 0.55, 0.72],
        projection_delta=30,
        projection_confidence=0.85,
    )
    veteran_reserve = _player(
        "reserve",
        age=32,
        experience=9,
        snap_share=0.2,
        target_share=0.04,
        yards_per_route_run=0.7,
        usage_history=[0.3, 0.25, 0.2, 0.15],
        projection_delta=-20,
    )
    assert breakout_probability(breakout)["probability"] > breakout_probability(veteran_reserve)["probability"]


def test_matchup_advantage_reads_week_schedule_and_handles_bye():
    player = _player(
        "scheduled",
        schedule={
            3: {"opponent": "LV", "defense_rank_vs_position": 30},
            4: {"opponent": "BYE"},
        },
    )
    favorable = matchup_advantage(player, 3)
    bye = matchup_advantage(player, 4)

    assert favorable["score"] > 75
    assert favorable["opponent"] == "LV"
    assert bye["score"] == 0
    assert bye["label"] == "bye"


def test_volatility_profile_uses_observed_weekly_variance():
    stable = _player("stable", weekly_points=[10, 11, 10, 11, 10])
    volatile = _player("volatile", weekly_points=[1, 25, 2, 30, 0])
    assert volatility_profile(volatile)["score"] > volatility_profile(stable)["score"]
    assert volatility_profile(volatile)["sample_size"] == 5


def test_waiver_priority_ranks_complete_upside_profile_first():
    featured = _player(
        "featured",
        projection=220,
        snap_share=0.9,
        target_share=0.3,
        targets=12,
        age=23,
        yards_per_route_run=2.7,
        usage_history=[0.3, 0.5, 0.7, 0.85],
        defense_rank_vs_position=29,
        volatility=0.55,
    )
    reserve = _player(
        "reserve",
        projection=90,
        snap_share=0.15,
        target_share=0.03,
        targets=1,
        age=30,
        yards_per_route_run=0.5,
        usage_history=[0.25, 0.2, 0.15, 0.1],
        defense_rank_vs_position=3,
        volatility=0.9,
    )

    ranked = rank_waiver_priority([reserve, featured], week=2)

    assert [row["player_id"] for row in ranked] == ["featured", "reserve"]
    assert [row["waiver_rank"] for row in ranked] == [1, 2]
    assert ranked[0]["waiver_priority_score"] > ranked[1]["waiver_priority_score"]
    assert ranked[0]["suggested_faab_pct"] > ranked[1]["suggested_faab_pct"]


def test_team_need_boosts_weak_position():
    team = [
        _player("qb", "QB", 180),
        _player("rb1", "RB", 200),
        _player("rb2", "RB", 190),
        _player("wr1", "WR", 180),
        _player("wr2", "WR", 170),
    ]
    same_profile = {
        "projection": 150,
        "snap_share": 0.6,
        "usage_history": [0.5, 0.55, 0.6],
        "defense_rank_vs_position": 16,
        "volatility": 0.5,
    }
    tight_end = _player("te", "TE", **same_profile)
    receiver = _player("wr", "WR", **same_profile)

    ranked = rank_waiver_priority(
        [receiver, tight_end],
        team=team,
        roster_requirements={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        week=1,
    )
    assert ranked[0]["player_id"] == "te"
    assert ranked[0]["team_need_score"] > ranked[1]["team_need_score"]


def test_injury_and_bye_apply_priority_penalties():
    base = {
        "projection": 160,
        "snap_share": 0.7,
        "target_share": 0.2,
        "usage_history": [0.5, 0.6, 0.7],
        "volatility": 0.5,
    }
    healthy = _player("healthy", **base)
    injured = _player("injured", injury_status="OUT", **base)
    bye = _player("bye", bye_week=5, **base)
    by_id = {row["player_id"]: row for row in rank_waiver_priority([injured, bye, healthy], week=5)}
    assert by_id["healthy"]["priority_score"] > by_id["bye"]["priority_score"] > by_id["injured"]["priority_score"]


def test_waiver_ranking_is_deterministic_and_supports_keyed_mapping():
    players = {
        "b": {"name": "B", "position": "RB", "projection": 100, "snap_share": 0.5},
        "a": {"name": "A", "position": "RB", "projection": 100, "snap_share": 0.5},
    }
    first = rank_waiver_priority(players)
    second = rank_waiver_priority(players)
    assert first == second
    assert [row["player_id"] for row in first] == ["a", "b"]


def test_waiver_envelope_has_consistent_indexes():
    result = compute_waiver_priority([_player("a"), _player("b")], max_results=1)
    assert result["metric"] == "waiver_priority"
    assert len(result["results"]) == 1
    assert set(result["by_player"]) == {result["results"][0]["player_id"]}


@pytest.mark.parametrize("week", [0, 19, 1.5, True])
def test_invalid_weeks_are_rejected(week):
    with pytest.raises(ValueError, match="week"):
        rank_waiver_priority([_player("x")], week=week)


def test_invalid_result_limit_is_rejected():
    with pytest.raises(ValueError, match="max_results"):
        rank_waiver_priority([_player("x")], max_results=-1)

