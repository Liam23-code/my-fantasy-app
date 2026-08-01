"""Unit tests for fantasy.tiering."""
from __future__ import annotations

import csv
import io

import pytest

from fantasy.tiering import export_cheatsheet_csv, generate_printable_cheatsheet, tier_players


def _wr(name, median, volatility=0.0, **extra):
    return {"player_id": name, "name": name, "position": "WR", "median": median, "volatility": volatility, **extra}


def _rb(name, median, volatility=0.0, **extra):
    return {"player_id": name, "name": name, "position": "RB", "median": median, "volatility": volatility, **extra}


def test_well_separated_players_cluster_into_expected_tiers():
    players = [
        _wr("WR1", 30), _wr("WR2", 29), _wr("WR3", 28),
        _wr("WR4", 15), _wr("WR5", 14),
        _wr("WR6", 5),
    ]
    tiered = tier_players(players, max_tiers=3)
    by_name = {p["name"]: p for p in tiered}
    assert by_name["WR1"]["tier"] == by_name["WR2"]["tier"] == by_name["WR3"]["tier"]
    assert by_name["WR4"]["tier"] == by_name["WR5"]["tier"]
    assert by_name["WR6"]["tier"] != by_name["WR4"]["tier"]
    assert by_name["WR1"]["tier"] < by_name["WR4"]["tier"] < by_name["WR6"]["tier"]  # tier 1 = best


def test_tiers_are_computed_independently_per_position():
    players = [
        _wr("WR Great", 100), _wr("WR Bad", 1),
        _rb("RB Great", 100), _rb("RB Bad", 1),
    ]
    tiered = tier_players(players, max_tiers=2)
    by_name = {p["name"]: p for p in tiered}
    # Each position's own top/bottom get tier 1 / tier 2, independent of the other position's values.
    assert by_name["WR Great"]["tier"] == 1
    assert by_name["RB Great"]["tier"] == 1
    assert by_name["WR Bad"]["tier"] == 2
    assert by_name["RB Bad"]["tier"] == 2


def test_single_player_at_a_position_gets_tier_one():
    players = [_wr("Lonely WR", 42)]
    tiered = tier_players(players, max_tiers=5)
    assert tiered[0]["tier"] == 1
    assert tiered[0]["tier_label"] == "Tier 1"


def test_tier_count_never_exceeds_available_players():
    players = [_wr("WR1", 30), _wr("WR2", 20)]
    tiered = tier_players(players, max_tiers=5)
    assert max(p["tier"] for p in tiered) <= 2


def test_max_tiers_of_one_puts_every_player_in_a_single_tier():
    players = [_wr("WR1", 30), _wr("WR2", 5)]
    tiered = tier_players(players, max_tiers=1)
    assert all(p["tier"] == 1 for p in tiered)


def test_max_tiers_below_one_raises_value_error():
    with pytest.raises(ValueError):
        tier_players([_wr("WR1", 30)], max_tiers=0)


def test_volatility_can_separate_identical_median_players_into_different_tiers():
    # Identical medians isolate the test to the volatility dimension alone --
    # any median gap would compete with volatility as a clustering signal.
    players = [
        _wr("Safe Floor", 20, volatility=0.1),
        _wr("Boom Bust", 20, volatility=2.5),
        _wr("Another Safe", 20, volatility=0.15),
        _wr("Another Boom Bust", 20, volatility=2.4),
    ]
    tiered = tier_players(players, max_tiers=2)
    by_name = {p["name"]: p for p in tiered}
    assert by_name["Safe Floor"]["tier"] == by_name["Another Safe"]["tier"]
    assert by_name["Boom Bust"]["tier"] == by_name["Another Boom Bust"]["tier"]
    assert by_name["Safe Floor"]["tier"] != by_name["Boom Bust"]["tier"]


def test_falls_back_to_points_field_when_median_missing():
    # Two players so this exercises the real clustering path (the len==1
    # shortcut in _cluster_position_group never calls _value_and_volatility).
    players = [
        {"player_id": "a", "name": "A", "position": "RB", "points": 50.0},
        {"player_id": "b", "name": "B", "position": "RB", "points": 5.0},
    ]
    tiered = tier_players(players, max_tiers=2)
    by_name = {p["name"]: p for p in tiered}
    assert by_name["A"]["tier"] < by_name["B"]["tier"]


def test_input_dicts_are_not_mutated():
    original = [_wr("WR1", 30)]
    tier_players(original)
    assert "tier" not in original[0]


def test_input_order_is_preserved():
    players = [_wr("WR3", 5), _wr("WR1", 30), _wr("WR2", 15)]
    tiered = tier_players(players, max_tiers=3)
    assert [p["name"] for p in tiered] == ["WR3", "WR1", "WR2"]


def test_printable_cheatsheet_groups_by_position_then_tier():
    players = [_wr("WR1", 30, team="LA"), _wr("WR2", 5, team="SF"), _rb("RB1", 40, team="PHI")]
    tiered = tier_players(players, max_tiers=2)
    text = generate_printable_cheatsheet(tiered)
    assert "=== RB ===" in text
    assert "=== WR ===" in text
    assert text.index("=== RB ===") < text.index("=== WR ===")  # alphabetical position order
    assert "WR1 (LA)" in text
    assert "-- Tier 1 --" in text


def test_csv_export_has_expected_columns_and_row_count():
    players = [_wr("WR1", 30, team="LA"), _rb("RB1", 40, team="PHI")]
    tiered = tier_players(players)
    csv_text = export_cheatsheet_csv(tiered)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 2
    assert set(reader.fieldnames) == {
        "tier", "position", "position_rank", "overall_rank", "name", "team",
        "points", "vor", "median", "floor", "ceiling", "volatility", "rationale",
    }


def test_csv_export_writes_to_file(tmp_path):
    players = [_wr("WR1", 30, team="LA")]
    tiered = tier_players(players)
    destination = tmp_path / "cheatsheet.csv"
    csv_text = export_cheatsheet_csv(tiered, file_path=str(destination))
    # Path.read_text() applies universal-newline translation (\r\n -> \n) on
    # read, but csv.writer's raw output (and the file, written with
    # newline="") keeps literal \r\n -- normalize before comparing.
    written = destination.read_text(encoding="utf-8")
    assert written.replace("\r\n", "\n") == csv_text.replace("\r\n", "\n")
    assert "WR1" in written
