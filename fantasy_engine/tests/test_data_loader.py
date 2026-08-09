"""Tests for the real-NFL-data loader.

The network-backed paths (:func:`load_real_projections`, :func:`load_adp`) are
deliberately not exercised here -- these tests cover the filtering, validation,
and derivation logic that decides *what counts as a real player*, which is the
part that must never regress.
"""

from __future__ import annotations

import datetime as dt

import pytest

from fantasy.data_loader import (
    DRAFTABLE_POSITIONS,
    _build_drivers,
    _normalize_name,
    _percentile,
    drop_synthetic,
    is_synthetic,
    latest_completed_season,
    validate_players,
)


def _player(**overrides):
    base = {
        "player_id": "00-0033280",
        "name": "Christian McCaffrey",
        "position": "RB",
        "team": "SF",
        "projection": 428.6,
    }
    base.update(overrides)
    return base


# --- synthetic detection ----------------------------------------------------


@pytest.mark.parametrize(
    "player",
    [
        {"player_id": "synthetic:qb:0"},
        {"player_id": "SYNTHETIC:RB:12"},
        {"id": "synthetic:wr:3"},
        {"player_id": "", "name": "Synthetic QB 0"},
        {"player_id": "x", "name": "synthetic te 4"},
    ],
)
def test_is_synthetic_catches_fabricated_records(player):
    assert is_synthetic(player) is True


@pytest.mark.parametrize(
    "player",
    [
        {"player_id": "nfl:player:00-0034796", "name": "Lamar Jackson"},
        {"player_id": "00-0033280", "name": "Christian McCaffrey"},
        {"player_id": "syn", "name": "Sy Nthetic"},  # prefix must be the whole word
    ],
)
def test_is_synthetic_passes_real_players(player):
    assert is_synthetic(player) is False


def test_is_synthetic_reads_objects_not_just_dicts():
    class Row:
        player_id = "synthetic:rb:9"
        name = "Synthetic RB 9"

    assert is_synthetic(Row()) is True


def test_drop_synthetic_keeps_only_real_players():
    pool = [
        _player(),
        {"player_id": "synthetic:qb:0", "name": "Synthetic QB 0"},
        {"player_id": "synthetic:wr:1", "name": "Synthetic WR 1"},
    ]
    remaining = drop_synthetic(pool)
    assert [p["name"] for p in remaining] == ["Christian McCaffrey"]


def test_drop_synthetic_handles_empty_and_none():
    assert drop_synthetic([]) == []
    assert drop_synthetic(None) == []


# --- validation -------------------------------------------------------------


def test_validate_players_accepts_a_complete_record():
    valid, rejected = validate_players([_player()])
    assert len(valid) == 1
    assert rejected == []


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"name": ""}, "missing name"),
        ({"player_id": ""}, "missing id"),
        ({"position": ""}, "missing position"),
        ({"team": ""}, "missing team"),
        ({"projection": None}, "missing projection"),
        ({"projection": "lots"}, "non-numeric projection"),
        ({"position": "LB"}, "undraftable position 'LB'"),
    ],
)
def test_validate_players_rejects_incomplete_records(overrides, expected_reason):
    valid, rejected = validate_players([_player(**overrides)])
    assert valid == []
    assert expected_reason in rejected[0]["reasons"]


def test_validate_players_rejects_synthetic_even_when_complete():
    valid, rejected = validate_players([_player(player_id="synthetic:rb:0", name="Synthetic RB 0")])
    assert valid == []
    assert "synthetic player" in rejected[0]["reasons"]


def test_validate_players_can_skip_the_projection_requirement():
    valid, rejected = validate_players([_player(projection=None)], require_projection=False)
    assert len(valid) == 1
    assert rejected == []


def test_validate_players_reports_every_problem_at_once():
    _valid, rejected = validate_players([{"player_id": "", "name": "", "position": "", "team": ""}])
    assert set(rejected[0]["reasons"]) >= {"missing id", "missing name", "missing position", "missing team"}


def test_validate_players_rejects_non_mappings():
    valid, rejected = validate_players(["not a player", 42])
    assert valid == []
    assert all("not a mapping" in entry["reasons"] for entry in rejected)


def test_every_draftable_position_is_accepted():
    for position in DRAFTABLE_POSITIONS:
        valid, _ = validate_players([_player(position=position)])
        assert len(valid) == 1, f"{position} should be draftable"


# --- derivations ------------------------------------------------------------


def test_normalize_name_folds_case_accents_punctuation_and_suffixes():
    assert _normalize_name("Marvin Harrison Jr.") == _normalize_name("marvin harrison")
    assert _normalize_name("Ja'Marr Chase") == "jamarrchase"
    assert _normalize_name("Amon-Ra St. Brown") == "amonrastbrown"


def test_normalize_name_handles_empty_input():
    assert _normalize_name(None) == ""
    assert _normalize_name("") == ""


def test_percentile_picks_by_nearest_rank():
    ordered = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _percentile(ordered, 0.0) == 0.0
    assert _percentile(ordered, 0.5) == 20.0
    assert _percentile(ordered, 1.0) == 40.0


def test_percentile_of_empty_series_is_zero():
    assert _percentile([], 0.5) == 0.0


def test_build_drivers_orders_by_contribution():
    drivers = _build_drivers(
        {"rushing_yards": 1202.0, "receiving_yards": 924.0, "receptions": 102.0, "rushing_tds": 10.0}
    )
    assert drivers[0].endswith("rushing yards")
    assert any("total TDs" in driver for driver in drivers)


def test_build_drivers_of_an_empty_stat_line_is_empty():
    assert _build_drivers({}) == []


def test_latest_completed_season_waits_for_the_season_to_finish():
    # A season labelled Y runs Sep Y -> Feb Y+1, so in Jan 2026 the newest
    # complete season is still 2024.
    assert latest_completed_season(dt.date(2026, 1, 15)) == 2024
    assert latest_completed_season(dt.date(2026, 8, 1)) == 2025
    assert latest_completed_season(dt.date(2025, 12, 1)) == 2024
