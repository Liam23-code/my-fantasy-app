"""Tests for fantasy.adp_fusion: weighted multi-source ADP fusion."""

from __future__ import annotations

from fantasy.adp_fusion import (
    SOURCE_WEIGHTS,
    UNAVAILABLE_SOURCES,
    apply_fused_adp,
    fuse_adp,
    real_sleeper_adp,
)


def _player(player_id, name, position, adp):
    return {"player_id": player_id, "name": name, "position": position, "adp": adp}


# --- fuse_adp correctness ----------------------------------------------------


def test_fuse_adp_keys_the_table_by_player_id():
    table = fuse_adp([_player("p1", "X", "WR", 10.0)])
    assert set(table) == {"p1"}


def test_fuse_adp_skips_players_without_real_adp_or_id():
    assert fuse_adp([{"name": "NoADP", "position": "WR"}]) == {}
    assert fuse_adp([{"adp": 5.0, "position": "WR"}]) == {}  # no player_id


def test_fuse_adp_only_uses_configured_weights_and_renormalizes():
    """Underdog/Yahoo/CBS/NFL.com are never populated -- weights renormalize
    across whatever *is* available (fantasypros + the two synthetic personas).
    """
    table = fuse_adp([_player("p1", "X", "WR", 10.0)])
    sources = table["p1"]["sources"]
    assert set(sources) == {"fantasypros", "espn", "sleeper"}
    for unavailable in UNAVAILABLE_SOURCES:
        assert unavailable not in sources


def test_fused_adp_is_a_weighted_blend_between_the_available_sources():
    table = fuse_adp([_player("p1", "X", "RB", 10.0)])
    entry = table["p1"]
    values = list(entry["sources"].values())
    assert min(values) <= entry["fused_adp"] <= max(values)


def test_fused_adp_matches_source_weights_hand_computed():
    # WR bias is 0.98 (espn) / 0.96 (sleeper) on a real ADP of 100.
    table = fuse_adp([_player("p1", "X", "WR", 100.0)])
    sources = table["p1"]["sources"]
    assert sources == {"fantasypros": 100.0, "espn": 98.0, "sleeper": 96.0}
    weights = {"fantasypros": SOURCE_WEIGHTS["fantasypros"], "espn": SOURCE_WEIGHTS["espn"], "sleeper": SOURCE_WEIGHTS["sleeper"]}
    total = sum(weights.values())
    expected = sum(sources[s] * w for s, w in weights.items()) / total
    assert table["p1"]["fused_adp"] == round(expected, 2)


def test_adp_volatility_reflects_spread_between_sources():
    # DST's synthetic personas both use a bias of 1.0 -- zero spread, by
    # construction. RB's personas (0.95 / 1.04) diverge from the baseline, so
    # its volatility must be strictly greater.
    zero_spread = fuse_adp([_player("p1", "ZeroSpread", "DST", 100.0)])["p1"]
    has_spread = fuse_adp([_player("p2", "HasSpread", "RB", 100.0)])["p2"]
    assert zero_spread["adp_volatility"] == 0.0
    assert has_spread["adp_volatility"] > zero_spread["adp_volatility"]


def test_adp_confidence_is_between_zero_and_one():
    entry = fuse_adp([_player("p1", "X", "WR", 10.0)])["p1"]
    assert 0.0 < entry["adp_confidence"] <= 1.0


def test_adp_confidence_reflects_partial_source_coverage():
    """Only 3 of 7 configured sources ever have data -- confidence should
    never claim full coverage."""
    entry = fuse_adp([_player("p1", "X", "WR", 10.0)])["p1"]
    full_coverage_weight = sum(SOURCE_WEIGHTS.values())
    covered_weight = SOURCE_WEIGHTS["fantasypros"] + SOURCE_WEIGHTS["espn"] + SOURCE_WEIGHTS["sleeper"]
    assert entry["adp_confidence"] <= covered_weight / full_coverage_weight + 1e-9


def test_adp_trend_direction_is_stable_falling_or_rising_relative_to_baseline():
    entry = fuse_adp([_player("p1", "X", "RB", 100.0)])["p1"]
    assert entry["adp_trend_direction"] in {"rising", "falling", "stable"}


def test_adp_trend_direction_stable_when_personas_roughly_agree_with_baseline():
    # RB bias is 0.95/1.04 -- close enough to baseline that trend should read stable.
    entry = fuse_adp([_player("p1", "X", "RB", 20.0)])["p1"]
    assert entry["adp_trend_direction"] == "stable"


# --- real_sleeper_adp stub ----------------------------------------------------


def test_real_sleeper_adp_stub_returns_empty():
    assert real_sleeper_adp([_player("p1", "X", "WR", 10.0)]) == {}


def test_fuse_adp_would_prefer_real_sleeper_data_when_present(monkeypatch):
    import fantasy.adp_fusion as adp_fusion_module

    monkeypatch.setattr(adp_fusion_module, "real_sleeper_adp", lambda players: {"p1": 5.0})
    table = adp_fusion_module.fuse_adp([_player("p1", "X", "WR", 10.0)])
    assert table["p1"]["sources"]["sleeper"] == 5.0


# --- apply_fused_adp integration ---------------------------------------------


def test_apply_fused_adp_promotes_fused_value_into_the_adp_field():
    players = [_player("p1", "X", "WR", 10.0)]
    updated = apply_fused_adp(players)
    table = fuse_adp(players)
    assert updated[0]["adp"] == table["p1"]["fused_adp"]


def test_apply_fused_adp_preserves_the_original_value():
    players = [_player("p1", "X", "WR", 10.0)]
    updated = apply_fused_adp(players)
    assert updated[0]["adp_fantasypros_only"] == 10.0


def test_apply_fused_adp_leaves_players_without_real_adp_untouched():
    players = [{"player_id": "p1", "name": "NoADP", "position": "WR"}]
    updated = apply_fused_adp(players)
    assert "adp_fantasypros_only" not in updated[0]
    assert "fused_adp" not in updated[0]  # no new fields invented for players with nothing to fuse


def test_apply_fused_adp_does_not_mutate_the_input():
    players = [_player("p1", "X", "WR", 10.0)]
    apply_fused_adp(players)
    assert players[0]["adp"] == 10.0
    assert "adp_fantasypros_only" not in players[0]


def test_apply_fused_adp_exposes_volatility_confidence_and_trend():
    players = [_player("p1", "X", "WR", 10.0)]
    updated = apply_fused_adp(players)[0]
    assert "adp_volatility" in updated
    assert "adp_confidence" in updated
    assert "adp_trend_direction" in updated
    assert "adp_sources" in updated
