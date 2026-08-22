"""Integration contracts between the draft board and unified Quant Engine."""

from __future__ import annotations

import pytest

import fantasy.draft as draft

SETTINGS = {
    "n_teams": 1,
    "scoring_mode": "ppr",
    "roster_requirements": {
        "QB": 0,
        "RB": 1,
        "WR": 1,
        "TE": 0,
        "FLEX": 0,
        "DST": 0,
        "K": 0,
        "BENCH": 0,
    },
    "flex_eligible": [],
}


def test_raw_stat_line_keeps_exact_league_scoring_and_skips_ensemble(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("raw stat lines must not be season-ensemble projected")

    monkeypatch.setattr(draft.quant, "compute_final_projection", unexpected_call)
    ranked = draft.rank_players_for_draft(
        [{"player_id": "raw", "name": "Raw Back", "position": "RB", "rushing_yards": 100}],
        SETTINGS,
    )

    # 10 rushing points plus the engine's configured 100-yard bonus.
    assert ranked[0]["points"] == pytest.approx(13.0)
    assert ranked[0]["projection"] == pytest.approx(13.0)
    assert "quant_projection" not in ranked[0]


def test_explicit_projection_uses_quant_final_projection(monkeypatch):
    def final_projection(player, **kwargs):
        assert player["player_id"] == "forecast"
        assert kwargs["scoring_mode"] == "ppr"
        return {
            "player_id": "forecast",
            "final_projection": 275.5,
            "floor": 220.0,
            "median": 275.5,
            "ceiling": 335.0,
            "projection_confidence": 0.91,
            "volatility": 0.22,
            "breakout_probability": 0.35,
            "bust_probability": 0.12,
        }

    monkeypatch.setattr(draft.quant, "compute_final_projection", final_projection)
    ranked = draft.rank_players_for_draft(
        [{"player_id": "forecast", "name": "Forecast Back", "position": "RB", "projection": 250.0}],
        SETTINGS,
    )
    player = ranked[0]

    assert player["points"] == pytest.approx(275.5)
    assert player["projection"] == pytest.approx(275.5)
    assert player["expected_fantasy_points"] == pytest.approx(275.5)
    assert player["projection_method"] == "quant_ensemble"
    assert player["projection_confidence"] == pytest.approx(0.91)
    assert player["volatility"] == pytest.approx(0.22)
    assert player["quant_projection"]["final_projection"] == pytest.approx(275.5)


def test_ranked_rows_merge_quant_scarcity_draft_value_and_rarity():
    players = [
        {"player_id": "rb1", "name": "RB One", "position": "RB", "projection": 250.0, "adp": 8.0},
        {"player_id": "rb2", "name": "RB Two", "position": "RB", "projection": 180.0, "adp": 35.0},
        {"player_id": "wr1", "name": "WR One", "position": "WR", "projection": 230.0, "adp": 12.0},
        {"player_id": "wr2", "name": "WR Two", "position": "WR", "projection": 170.0, "adp": 42.0},
    ]
    ranked = draft.rank_players_for_draft(players, SETTINGS)

    for player in ranked:
        assert 0 <= player["scarcity_score"] <= 100
        assert 0.9 <= player["scarcity_multiplier"] <= 1.5
        assert isinstance(player["draft_value"], float)
        assert 0 <= player["draft_value_score"] <= 100
        assert player["rarity_tier"]
        assert player["rarity_symbol"]
        assert 0 <= player["rarity_percentile"] <= 1
        assert "Quant tier" in player["rationale"]


def test_projection_baked_for_another_scoring_mode_uses_raw_stats(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("incompatible baked projections must not reach Quant")

    monkeypatch.setattr(draft.quant, "compute_final_projection", unexpected_call)
    settings = {**SETTINGS, "scoring_mode": "standard"}
    source = {
        "player_id": "mode",
        "name": "Mode Back",
        "position": "RB",
        "projection": 250.0,
        "scoring_mode": "ppr",
        "rushing_yards": 100.0,
    }
    ranked = draft.rank_players_for_draft([source], settings)
    assert ranked[0]["points"] == pytest.approx(13.0)


def test_invalid_quant_projection_is_rejected(monkeypatch):
    monkeypatch.setattr(
        draft.quant,
        "compute_final_projection",
        lambda *args, **kwargs: {"final_projection": float("nan")},
    )
    with pytest.raises(ValueError, match="invalid final projection"):
        draft.rank_players_for_draft(
            [{"player_id": "bad", "name": "Bad", "position": "RB", "projection": 100.0}],
            SETTINGS,
        )
