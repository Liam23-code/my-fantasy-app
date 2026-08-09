"""Unit tests for fantasy.tiering."""

from __future__ import annotations

import csv
import io

import pytest

from fantasy.tiering import (
    ESPN_STYLE_TIER_LABELS,
    combined_tier,
    export_cheatsheet_csv,
    generate_printable_cheatsheet,
    tier_players,
    tier_players_by_adp,
    tier_players_by_scarcity,
)


def _wr(name, median, volatility=0.0, **extra):
    return {"player_id": name, "name": name, "position": "WR", "median": median, "volatility": volatility, **extra}


def _rb(name, median, volatility=0.0, **extra):
    return {"player_id": name, "name": name, "position": "RB", "median": median, "volatility": volatility, **extra}


def test_well_separated_players_cluster_into_expected_tiers():
    players = [
        _wr("WR1", 30),
        _wr("WR2", 29),
        _wr("WR3", 28),
        _wr("WR4", 15),
        _wr("WR5", 14),
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
        _wr("WR Great", 100),
        _wr("WR Bad", 1),
        _rb("RB Great", 100),
        _rb("RB Bad", 1),
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
        "tier",
        "position",
        "position_rank",
        "overall_rank",
        "name",
        "team",
        "points",
        "vor",
        "median",
        "floor",
        "ceiling",
        "volatility",
        "rationale",
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


# --- tier_players_by_adp ------------------------------------------------------


def test_tier_players_by_adp_orders_tier_1_by_lowest_adp():
    players = [
        {"name": f"RB{i}", "position": "RB", "adp": float(i + 1)} for i in range(30)
    ]
    settings = {"n_teams": 12, "roster_requirements": {"RB": 1, "FLEX": 0, "BENCH": 0}, "flex_eligible": []}
    tiered = tier_players_by_adp(players, settings, n_teams=12)
    tier1 = [p for p in tiered if p["adp_tier"] == 1]
    # Starters-at-position (RB=1) * n_teams=12 -> tier size 12.
    assert len(tier1) == 12
    assert {p["name"] for p in tier1} == {f"RB{i}" for i in range(12)}
    assert all(p["adp_tier_label"] == "Elite" for p in tier1)


def test_tier_players_by_adp_sorts_missing_adp_last():
    players = [
        {"name": "HasADP", "position": "WR", "adp": 5.0},
        {"name": "NoADP", "position": "WR"},
    ]
    tiered = tier_players_by_adp(players, n_teams=1)
    by_name = {p["name"]: p for p in tiered}
    assert by_name["HasADP"]["adp_tier"] <= by_name["NoADP"]["adp_tier"]


def test_tier_players_by_adp_caps_at_the_label_count():
    players = [{"name": f"P{i}", "position": "K", "adp": float(i)} for i in range(200)]
    tiered = tier_players_by_adp(players, n_teams=1)
    assert max(p["adp_tier"] for p in tiered) == len(ESPN_STYLE_TIER_LABELS)


def test_tier_players_by_adp_without_league_settings_still_works():
    players = [{"name": "X", "position": "RB", "adp": 1.0}]
    tiered = tier_players_by_adp(players)
    assert tiered[0]["adp_tier"] == 1


# --- tier_players_by_scarcity --------------------------------------------------


def test_tier_players_by_scarcity_top_slice_is_tier_one():
    players = [{"name": f"P{i}", "position": "WR", "vor": float(100 - i)} for i in range(20)]
    tiered = tier_players_by_scarcity(players)
    tier1 = sorted((p for p in tiered if p["scarcity_tier"] == 1), key=lambda p: p["name"])
    # Boundary is the first 10% -> 2 players out of 20.
    assert len(tier1) == 2
    assert {p["name"] for p in tier1} == {"P0", "P1"}


def test_tier_players_by_scarcity_falls_back_to_points():
    players = [{"name": "A", "position": "RB", "points": 50.0}, {"name": "B", "position": "RB", "points": 10.0}]
    tiered = tier_players_by_scarcity(players)
    by_name = {p["name"]: p for p in tiered}
    assert by_name["A"]["scarcity_tier"] <= by_name["B"]["scarcity_tier"]


def test_tier_players_by_scarcity_is_independent_per_position():
    players = [{"name": "EliteRB", "position": "RB", "vor": 100.0}, {"name": "WorstWR", "position": "WR", "vor": -50.0}]
    tiered = tier_players_by_scarcity(players)
    # Each position's own single player is necessarily its own top slice.
    assert all(p["scarcity_tier"] == 1 for p in tiered)


# --- combined_tier -------------------------------------------------------------


def test_combined_tier_averages_the_three_component_tiers():
    players = [
        {"name": "A", "player_id": "a", "position": "RB", "adp": 1.0, "vor": 100.0, "median": 50.0},
        {"name": "B", "player_id": "b", "position": "RB", "adp": 50.0, "vor": -10.0, "median": 5.0},
    ]
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "FLEX": 0, "BENCH": 0}, "flex_eligible": []}
    combined = combined_tier(players, settings, n_teams=1)
    for player in combined:
        assert player["combined_tier"] == round(
            (player["adp_tier"] + player["scarcity_tier"] + player["value_tier"]) / 3
        )


def test_combined_tier_keeps_the_original_tier_players_untouched():
    """tier_players itself must not be affected by adding the new combined view."""
    players = [{"name": "A", "position": "RB", "median": 10.0}]
    before = tier_players(players)
    combined_tier(players)
    after = tier_players(players)
    assert before == after


def test_combined_tier_matches_players_by_id_not_position_in_list():
    players = [
        {"name": "Same", "player_id": "x1", "position": "RB", "adp": 1.0, "vor": 10.0, "median": 10.0},
        {"name": "Same", "player_id": "x2", "position": "RB", "adp": 90.0, "vor": -10.0, "median": 1.0},
    ]
    combined = combined_tier(players, n_teams=1)
    by_id = {p["player_id"]: p for p in combined}
    assert by_id["x1"]["combined_tier"] != by_id["x2"]["combined_tier"]


def test_combined_tier_propagates_the_adp_tier_label_string():
    """Regression: combined_tier once dropped adp_tier_label, leaving it blank for every player."""
    players = [{"name": "A", "player_id": "a", "position": "RB", "adp": 1.0, "vor": 10.0, "median": 10.0}]
    combined = combined_tier(players, n_teams=1)
    assert combined[0]["adp_tier_label"] == "Elite"
