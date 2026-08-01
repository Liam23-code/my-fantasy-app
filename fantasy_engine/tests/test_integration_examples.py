"""Integration tests: the full pipeline against the bundled >=200 player sample dataset.

Unlike the per-module unit tests (which use small, hand-crafted fixtures to
pin down exact numeric behavior), these exercise the real example data end
to end and check for internal consistency and the absence of exceptions --
closer to what actually happens when someone runs ``examples/quickstart.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantasy.draft import generate_cheatsheet, rank_players_for_draft, simulate_draft
from fantasy.optimizer import optimize_lineup, start_sit_advice
from fantasy.tiering import export_cheatsheet_csv, generate_printable_cheatsheet, tier_players
from fantasy.trade import evaluate_trade
from fantasy.waiver import waiver_recommendations

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def sample_projections():
    return json.loads((EXAMPLES_DIR / "sample_projections.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def league_settings():
    return json.loads((EXAMPLES_DIR / "sample_league_settings.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def roster():
    return json.loads((EXAMPLES_DIR / "sample_roster.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def available_players():
    return json.loads((EXAMPLES_DIR / "sample_available_players.json").read_text(encoding="utf-8"))


def test_sample_dataset_has_at_least_two_hundred_players(sample_projections):
    assert len(sample_projections) >= 200


def test_rank_players_for_draft_produces_consistent_ranks(sample_projections, league_settings):
    ranked = rank_players_for_draft(sample_projections, league_settings)
    assert len(ranked) == len(sample_projections)
    overall_ranks = [p["overall_rank"] for p in ranked]
    assert overall_ranks == list(range(1, len(ranked) + 1))
    vor_values = [p["vor"] for p in ranked]
    assert vor_values == sorted(vor_values, reverse=True)
    for player in ranked:
        assert player["position_rank"] >= 1
        assert player["rationale"]


def test_generate_cheatsheet_top_n_is_a_prefix_of_full_ranking(sample_projections, league_settings):
    full = rank_players_for_draft(sample_projections, league_settings)
    cheatsheet = generate_cheatsheet(sample_projections, league_settings, top_n=100)
    assert cheatsheet == full[:100]


def test_tiering_and_cheatsheet_export_do_not_raise(sample_projections, league_settings):
    ranked = rank_players_for_draft(sample_projections, league_settings)
    tiered = tier_players(ranked, max_tiers=6)
    assert all("tier" in player for player in tiered)
    text = generate_printable_cheatsheet(tiered)
    assert "=== QB ===" in text
    csv_text = export_cheatsheet_csv(tiered)
    assert csv_text.count("\n") >= len(tiered)


def test_optimize_lineup_and_start_sit_advice_on_real_roster(sample_projections, league_settings, roster):
    lineup = optimize_lineup(roster, sample_projections, league_settings)
    assert lineup["total_points"] > 0
    assert lineup["starters"]
    advice = start_sit_advice(roster, sample_projections, league_settings)
    assert len(advice) == len(roster)


def test_waiver_recommendations_on_real_free_agent_pool(league_settings, roster, available_players):
    league_state = {"league_settings": league_settings, "my_roster": roster, "current_week": 6}
    ranked = waiver_recommendations(league_state, available_players, "ppr", budget=100)
    assert len(ranked) == len(available_players)
    scores = [c["composite_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_simulate_draft_full_run_on_real_pool(sample_projections, league_settings):
    result = simulate_draft(sample_projections, league_settings, rounds=5, seed=99)
    assert len(result["picks"]) == 5 * league_settings["n_teams"]
    drafted_ids = [pick["player_id"] for pick in result["picks"]]
    assert len(drafted_ids) == len(set(drafted_ids))


def test_evaluate_trade_between_two_named_players(sample_projections, league_settings):
    result = evaluate_trade(
        ["Saquon Barkley"], ["Puka Nacua"], league_settings, sample_projections,
        monte_carlo_iterations=1000, seed=1,
    )
    assert {"fair_value", "recommendation", "win_prob_delta", "rationale"}.issubset(result)
    assert -1.0 <= result["win_prob_delta"] <= 1.0
