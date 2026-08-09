"""Tests for fantasy.draft_fusion: the realism-weighted Draft Result Fusion Engine."""

from __future__ import annotations

from fantasy.draft_fusion import (
    CONFIRMED_UNAVAILABLE_SOURCES,
    REALISM_WEIGHTS,
    apply_fused_draft_results,
    fuse_draft_results,
)


def _player(player_id, name, position, adp):
    return {"player_id": player_id, "name": name, "position": position, "adp": adp}


def _fake_ffc(monkeypatch, records):
    monkeypatch.setattr("fantasy.draft_fusion.fetch_ffcalculator_adp", lambda **kwargs: records)


# --- basic correctness --------------------------------------------------------


def test_fuse_draft_results_keys_by_player_id(monkeypatch):
    _fake_ffc(monkeypatch, [])
    table = fuse_draft_results([_player("p1", "X", "WR", 10.0)])
    assert set(table) == {"p1"}
    assert table["p1"]["player_id"] == "p1"


def test_fuse_draft_results_skips_players_without_real_adp_or_id(monkeypatch):
    _fake_ffc(monkeypatch, [])
    assert fuse_draft_results([{"name": "NoADP", "position": "WR"}]) == {}
    assert fuse_draft_results([{"adp": 5.0, "position": "WR"}]) == {}


def test_sleeper_never_contributes(monkeypatch):
    _fake_ffc(monkeypatch, [])
    table = fuse_draft_results([_player("p1", "X", "WR", 10.0)])
    assert "sleeper" not in table["p1"]["sources"]
    assert "sleeper" in CONFIRMED_UNAVAILABLE_SOURCES


# --- FFC integration (the one real, live source) -----------------------------


def test_ffcalculator_joins_by_normalized_name_not_its_own_player_id(monkeypatch):
    """FFC's player_id (an int, its own numbering) is unrelated to ours -- must join by name."""
    _fake_ffc(monkeypatch, [{"player_id": 999, "name": "Christian McCaffrey", "adp": 5.0, "high": 1, "low": 10, "stdev": 1.5}])
    table = fuse_draft_results([_player("00-0033280", "Christian McCaffrey", "RB", 8.5)])
    entry = table["00-0033280"]
    assert entry["sources"]["ffcalculator"] == 5.0
    assert entry["sources"]["fantasypros"] == 8.5


def test_fused_adp_is_weighted_between_fantasypros_and_ffcalculator(monkeypatch):
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 8, "stdev": 1.0}])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)])
    fp_weight = REALISM_WEIGHTS["fantasypros"]
    ffc_weight = REALISM_WEIGHTS["ffcalculator"]
    expected = round((8.0 * fp_weight + 4.0 * ffc_weight) / (fp_weight + ffc_weight), 2)
    assert table["p1"]["fused_adp"] == expected


def test_volatility_uses_ffcalculators_real_stdev_when_available(monkeypatch):
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 8, "stdev": 2.5}])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)])
    assert table["p1"]["volatility"] == 2.5


def test_reach_and_fall_rates_derived_from_real_high_low_range(monkeypatch):
    # adp=4, high=1 (earliest), low=8 (latest) -> reach 3/7, fall 4/7.
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 8, "stdev": 1.0}])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)])
    entry = table["p1"]
    assert 0.0 <= entry["reach_rate"] <= 1.0
    assert 0.0 <= entry["fall_rate"] <= 1.0
    assert entry["reach_rate"] + entry["fall_rate"] == 1.0  # no gap, no overlap for a 2-point range


def test_round_curve_derived_from_real_high_low_range(monkeypatch):
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 25, "stdev": 5.0}])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)], teams=12)
    curve = table["p1"]["round_curve"]
    assert set(curve) == {1, 2, 3}  # picks 1-25 span rounds 1, 2, 3 at 12 teams
    assert abs(sum(curve.values()) - 1.0) < 0.01  # each share individually rounded to 3dp


def test_player_with_no_ffc_match_falls_back_to_fantasypros_only(monkeypatch):
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "SomeoneElse", "adp": 4.0, "high": 1, "low": 8, "stdev": 1.0}])
    table = fuse_draft_results([_player("p1", "X", "WR", 20.0)])
    entry = table["p1"]
    assert "ffcalculator" not in entry["sources"]
    assert entry["fused_adp"] == 20.0


# --- run_pressure --------------------------------------------------------------


def test_run_pressure_is_high_inside_a_real_cluster_and_zero_outside(monkeypatch):
    _fake_ffc(monkeypatch, [])
    tight_cluster = [_player(f"rb{i}", f"RB{i}", "RB", float(1 + i)) for i in range(5)]
    isolated = _player("wr1", "LonelyWR", "WR", 50.0)
    table = fuse_draft_results(tight_cluster + [isolated])
    assert all(table[f"rb{i}"]["run_pressure"] > 0 for i in range(5))
    assert table["wr1"]["run_pressure"] == 0.0


def test_run_pressure_is_bounded_between_zero_and_one(monkeypatch):
    _fake_ffc(monkeypatch, [])
    cluster = [_player(f"rb{i}", f"RB{i}", "RB", float(1 + i)) for i in range(20)]
    table = fuse_draft_results(cluster)
    assert all(0.0 <= entry["run_pressure"] <= 1.0 for entry in table.values())


# --- extra_boards (user-supplied real sources) --------------------------------


def test_extra_boards_contribute_a_genuine_computed_average(monkeypatch):
    _fake_ffc(monkeypatch, [])
    table = fuse_draft_results(
        [_player("p1", "X", "WR", 10.0)],
        extra_boards={"ffpc": [("X", 2), ("X", 4), ("X", 6)]},
    )
    entry = table["p1"]
    assert entry["sources"]["ffpc"] == 4.0  # mean of 2, 4, 6


def test_extra_boards_for_unmatched_players_do_not_contribute(monkeypatch):
    _fake_ffc(monkeypatch, [])
    table = fuse_draft_results(
        [_player("p1", "X", "WR", 10.0)],
        extra_boards={"ffpc": [("SomeoneElse", 2)]},
    )
    assert "ffpc" not in table["p1"]["sources"]


# --- apply_fused_draft_results integration -----------------------------------


def test_apply_fused_draft_results_promotes_fused_value_into_adp(monkeypatch):
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 8, "stdev": 1.0}])
    players = [_player("p1", "X", "WR", 8.0)]
    updated = apply_fused_draft_results(players)
    assert updated[0]["adp"] != 8.0
    assert updated[0]["adp_fantasypros_only"] == 8.0


def test_apply_fused_draft_results_exposes_run_pressure_and_rates(monkeypatch):
    _fake_ffc(monkeypatch, [])
    updated = apply_fused_draft_results([_player("p1", "X", "WR", 8.0)])
    assert "run_pressure" in updated[0]
    assert "reach_rate" in updated[0]
    assert "fall_rate" in updated[0]
    assert "round_curve" in updated[0]


def test_apply_fused_draft_results_does_not_mutate_input(monkeypatch):
    _fake_ffc(monkeypatch, [])
    players = [_player("p1", "X", "WR", 8.0)]
    apply_fused_draft_results(players)
    assert players[0]["adp"] == 8.0
    assert "run_pressure" not in players[0]


def test_apply_fused_draft_results_leaves_players_without_adp_untouched(monkeypatch):
    _fake_ffc(monkeypatch, [])
    players = [{"player_id": "p1", "name": "NoADP", "position": "WR"}]
    updated = apply_fused_draft_results(players)
    assert "run_pressure" not in updated[0]


# --- sources_used / source_coverage -------------------------------------------


def test_sources_used_lists_only_sources_that_actually_contributed(monkeypatch):
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 8, "stdev": 1.0}])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)])
    assert table["p1"]["sources_used"] == ["fantasypros", "ffcalculator"]


def test_sources_used_shrinks_when_a_source_has_no_match(monkeypatch):
    _fake_ffc(monkeypatch, [])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)])
    assert table["p1"]["sources_used"] == ["fantasypros"]


def test_source_coverage_reflects_the_share_of_configured_weight_backed_by_data(monkeypatch):
    _fake_ffc(monkeypatch, [])
    table = fuse_draft_results([_player("p1", "X", "WR", 8.0)])
    expected = round(REALISM_WEIGHTS["fantasypros"] / sum(REALISM_WEIGHTS.values()), 3)
    assert table["p1"]["source_coverage"] == expected


def test_source_coverage_rises_when_more_sources_contribute(monkeypatch):
    _fake_ffc(monkeypatch, [])
    single = fuse_draft_results([_player("p1", "X", "WR", 8.0)])["p1"]["source_coverage"]
    _fake_ffc(monkeypatch, [{"player_id": 1, "name": "X", "adp": 4.0, "high": 1, "low": 8, "stdev": 1.0}])
    double = fuse_draft_results([_player("p1", "X", "WR", 8.0)])["p1"]["source_coverage"]
    assert double > single


def test_apply_fused_draft_results_exposes_sources_used(monkeypatch):
    _fake_ffc(monkeypatch, [])
    updated = apply_fused_draft_results([_player("p1", "X", "WR", 8.0)])[0]
    assert updated["sources_used"] == ["fantasypros"]
    assert "source_coverage" in updated
