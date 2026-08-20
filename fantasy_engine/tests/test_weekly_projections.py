"""Tests for the 18-week projection engine and its public integrations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fantasy.assistant import weekly_start_sit_advice
from fantasy.draft import rank_players_for_draft
from fantasy.weekly_projections import (
    build_weekly_projection,
    bye_week_projection,
    defensive_strength_adjustment,
    matchup_adjusted_projection,
    weekly_confidence,
    weekly_points,
    weekly_volatility,
)


def _player(**overrides) -> dict:
    player = {
        "player_id": "weekly-rb-1",
        "name": "Weekly Running Back",
        "position": "RB",
        "team": "DEN",
        "projection": 272.0,
        "expected_fantasy_points": 272.0,
        "projection_confidence": 0.80,
        "scoring_mode": "ppr",
        "bye_week": 9,
        "schedule": {
            1: {"opponent": "KC", "defense": {"rank_vs_position": {"RB": 1}}},
            2: {"opponent": "LV", "defense": {"rank_vs_position": {"RB": 32}}},
            9: {"opponent": "BYE"},
        },
    }
    player.update(overrides)
    return player


def test_bye_week_is_zero_and_present_in_full_18_week_curve():
    player = _player()
    projection = build_weekly_projection(player, "ppr")

    assert set(projection) == set(range(1, 19))
    assert bye_week_projection(player) == 0.0
    assert weekly_points(player, 9) == 0.0
    assert projection[9] == {"points": 0.0, "confidence": 1.0}
    assert all(set(value) == {"points", "confidence"} for value in projection.values())


def test_defensive_adjustment_rewards_weak_defenses_and_penalizes_strong_ones():
    strongest = defensive_strength_adjustment({"rank_vs_position": {"RB": 1}}, "RB")
    neutral = defensive_strength_adjustment(None, "RB")
    weakest = defensive_strength_adjustment({"rank_vs_position": {"RB": 32}}, "RB")

    assert strongest == pytest.approx(0.80)
    assert neutral == pytest.approx(1.0)
    assert weakest == pytest.approx(1.20)
    assert strongest < neutral < weakest

    common = _player(bye_week=None, schedule=None)
    tough = {**common, "opponent_defense": {"rank_vs_position": {"RB": 1}}}
    easy = {**common, "opponent_defense": {"rank_vs_position": {"RB": 32}}}
    assert matchup_adjusted_projection(tough, 4, "ppr") < matchup_adjusted_projection(easy, 4, "ppr")


def test_volatility_curve_expands_as_projection_confidence_falls():
    high_confidence = _player(player_id="high", projection_confidence=0.95, schedule=None)
    low_confidence = _player(player_id="low", projection_confidence=0.25, schedule=None)

    assert weekly_volatility(low_confidence) > weekly_volatility(high_confidence)

    # Give both players the same deterministic phase so the observed range is
    # measuring amplitude alone, not a different part of the sine curve.
    low_confidence["player_id"] = high_confidence["player_id"]
    high_curve = build_weekly_projection(high_confidence)
    low_curve = build_weekly_projection(low_confidence)
    active_weeks = [week for week in range(1, 19) if week != 9]
    high_range = max(high_curve[w]["points"] for w in active_weeks) - min(
        high_curve[w]["points"] for w in active_weeks
    )
    low_range = max(low_curve[w]["points"] for w in active_weeks) - min(
        low_curve[w]["points"] for w in active_weeks
    )
    assert low_range > high_range


def test_weekly_confidence_is_bounded_and_drives_the_confidence_curve():
    high = _player(player_id="same", projection_confidence=0.90)
    low = _player(player_id="same", projection_confidence=0.35)

    assert weekly_confidence(high) == pytest.approx(0.90)
    assert weekly_confidence(low) == pytest.approx(0.35)

    high_curve = build_weekly_projection(high)
    low_curve = build_weekly_projection(low)
    active_weeks = [week for week in range(1, 19) if week != 9]
    assert all(0.0 <= high_curve[week]["confidence"] <= 1.0 for week in range(1, 19))
    assert sum(high_curve[w]["confidence"] for w in active_weeks) > sum(
        low_curve[w]["confidence"] for w in active_weeks
    )


def test_draft_player_detail_contains_weekly_projection_curve():
    settings = {
        "n_teams": 1,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0},
        "flex_eligible": [],
    }
    ranked = rank_players_for_draft([_player()], settings)

    assert set(ranked[0]["weekly_projection"]) == set(range(1, 19))
    assert ranked[0]["weekly_projection"][9]["points"] == 0.0


def test_weekly_start_sit_uses_the_selected_week_curve():
    settings = {
        "n_teams": 1,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0},
        "flex_eligible": [],
    }
    roster = [
        {"player_id": "alpha", "name": "Alpha Back", "position": "RB", "nfl_team": "DEN"},
        {"player_id": "beta", "name": "Beta Back", "position": "RB", "nfl_team": "LV"},
    ]
    projections = [
        _player(player_id="alpha", name="Alpha Back", projection=306.0, expected_fantasy_points=306.0, bye_week=9),
        _player(player_id="beta", name="Beta Back", projection=170.0, expected_fantasy_points=170.0, bye_week=10),
    ]

    advice = weekly_start_sit_advice(roster, projections, 4, settings)
    by_player = {row["player"]: row for row in advice}

    assert by_player["Alpha Back"]["start_or_bench"] == "start"
    assert by_player["Alpha Back"]["projected_points"] > by_player["Beta Back"]["projected_points"]
    assert by_player["Alpha Back"]["week"] == 4
    assert "confidence" in by_player["Alpha Back"]


def test_season_tools_page_integrates_weekly_chart_matchups_and_confidence():
    page_path = (
        Path(__file__).resolve().parents[2]
        / "UniversalQuantAgent"
        / "app"
        / "pages"
        / "27_Fantasy_Season_Tools.py"
    )
    source = page_path.read_text(encoding="utf-8")

    # Parse the whole module so this catches malformed integration code, then
    # verify the visible tab and all three required weekly presentation areas.
    ast.parse(source)
    assert '"Weekly Projections"' in source
    assert "Weekly scoring curve" in source
    assert "Opponent matchups" in source
    assert "Confidence curve" in source
    assert "build_weekly_projection" in source
    assert "weekly_start_sit_advice" in source
