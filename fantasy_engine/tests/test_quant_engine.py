"""Contract and behavior tests for the unified Quant Engine facade."""

from __future__ import annotations

import json

import pytest

from quant.data_loader import (
    CANONICAL_PLAYER_SCHEMA,
    DataLoadError,
    load_adp_feeds,
    load_all_player_data,
    load_depth_charts,
    load_historical_data,
    load_injury_data,
    load_news_signals,
    load_player_stats,
    load_schedule_data,
    load_team_strength,
    load_weather_data,
)
from quant.quant_engine import (
    QuantEngine,
    compute_base_projections,
    compute_breakout_probability,
    compute_bust_probability,
    compute_confidence_scores,
    compute_draft_value,
    compute_efficiency_scores,
    compute_final_projection,
    compute_health_adjustments,
    compute_momentum,
    compute_player_similarity,
    compute_positional_scarcity,
    compute_rarity_tier,
    compute_trade_value,
    compute_trend_lines,
    compute_usage_rates,
    compute_value_over_replacement,
    compute_volatility,
    compute_weekly_matchup_score,
)


@pytest.fixture
def player_pool() -> list[dict]:
    return [
        {
            "player_id": "rb-alpha",
            "name": "Alpha Runner",
            "position": "RB",
            "team": "DEN",
            "projection": 300.0,
            "projection_confidence": 0.88,
            "games_played": 17,
            "history": [11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
            "carries": 255,
            "targets": 70,
            "receptions": 54,
            "rushing_yards": 1_260,
            "receiving_yards": 420,
            "rushing_tds": 10,
            "receiving_tds": 3,
            "target_share": 0.13,
            "rush_share": 0.62,
            "snap_share": 0.76,
            "adp": 8.0,
            "age": 24,
            "depth_order": 1,
        },
        {
            "player_id": "rb-beta",
            "name": "Beta Runner",
            "position": "RB",
            "team": "KC",
            "projection": 225.0,
            "projection_confidence": 0.67,
            "games_played": 15,
            "history": [10.0, 11.0, 9.0, 12.0, 13.0, 14.0],
            "carries": 190,
            "targets": 48,
            "receptions": 36,
            "rushing_yards": 890,
            "receiving_yards": 280,
            "rushing_tds": 7,
            "receiving_tds": 2,
            "target_share": 0.10,
            "rush_share": 0.48,
            "snap_share": 0.61,
            "adp": 34.0,
            "age": 26,
            "depth_order": 1,
        },
        {
            "player_id": "wr-gamma",
            "name": "Gamma Receiver",
            "position": "WR",
            "team": "BUF",
            "projection": 270.0,
            "projection_confidence": 0.78,
            "games_played": 17,
            "history": [18.0, 16.0, 15.0, 13.0, 12.0, 10.0],
            "targets": 138,
            "receptions": 91,
            "receiving_yards": 1_210,
            "receiving_tds": 8,
            "target_share": 0.27,
            "snap_share": 0.82,
            "route_participation": 0.90,
            "adp": 16.0,
            "age": 27,
            "depth_order": 1,
        },
    ]


def test_player_loader_normalizes_aliases_and_emits_the_full_schema():
    players = load_player_stats(
        [{"id": 17, "player_name": "Case Test", "pos": "wr", "nfl_team": "buf", "projected_points": "211.5", "rec_yards": "900"}]
    )

    assert len(players) == 1
    player = players[0]
    assert set(CANONICAL_PLAYER_SCHEMA).issubset(player)
    assert player["player_id"] == "17"
    assert player["position"] == "WR"
    assert player["team"] == "BUF"
    assert player["projection"] == 211.5
    assert player["stats"]["receiving_yards"] == 900.0


def test_loader_accepts_json_csv_and_keyed_mapping(tmp_path):
    json_path = tmp_path / "players.json"
    json_path.write_text(json.dumps({"players": [{"id": "json", "name": "Json Player", "position": "QB", "team": "NYJ"}]}), encoding="utf-8")
    csv_path = tmp_path / "players.csv"
    csv_path.write_text("player_id,name,position,team,projection\ncsv,Csv Player,TE,KC,180.5\n", encoding="utf-8")

    assert load_player_stats(json_path)[0]["player_id"] == "json"
    assert load_player_stats(csv_path)[0]["projection"] == 180.5
    keyed = load_player_stats({"keyed": {"name": "Keyed Player", "position": "RB", "team": "DEN"}})
    assert keyed[0]["player_id"] == "keyed"


def test_loader_is_offline_safe_and_strict_mode_is_explicit(tmp_path):
    missing = tmp_path / "not-there.json"
    assert load_player_stats(missing) == []
    with pytest.raises(DataLoadError, match="does not exist"):
        load_player_stats(missing, strict=True)


@pytest.mark.parametrize(
    ("loader", "source_key", "payload"),
    [
        (load_historical_data, "historical_data", {"history": [{"week": 1, "points": 12}]}),
        (load_injury_data, "injury_data", {"status": "questionable"}),
        (load_depth_charts, "depth_charts", {"depth_order": 1}),
        (load_adp_feeds, "adp_feeds", {"adp": 12.5}),
        (load_schedule_data, "schedule_data", {"week": 1, "opponent": "KC"}),
        (load_team_strength, "team_strength", {"strength": 0.8}),
        (load_weather_data, "weather_data", {"weather_risk": 0.3}),
        (load_news_signals, "news_signals", {"sentiment": 0.4}),
    ],
)
def test_every_source_loader_uses_the_canonical_contract(loader, source_key, payload):
    row = {"player_id": "p1", "name": "Player One", "position": "RB", "team": "DEN", **payload}
    result = loader(row)
    assert len(result) == 1
    assert set(CANONICAL_PLAYER_SCHEMA).issubset(result[0])
    assert result[0]["sources"] == [source_key]


def test_unified_loader_merges_player_team_and_name_level_sources():
    result = load_all_player_data(
        {
            "player_stats": [{"player_id": "p1", "name": "Player One", "position": "RB", "team": "DEN", "projection": 220}],
            "injuries": [{"name": "Player One", "team": "DEN", "status": "QUESTIONABLE"}],
            "adp": [{"name": "Player One", "position": "RB", "adp": 24}],
            "schedule": [{"team": "DEN", "week": 3, "opponent": "LV"}],
        }
    )

    assert len(result) == 1
    player = result[0]
    assert player["player_id"] == "p1"
    assert player["injury_status"] == "QUESTIONABLE"
    assert player["adp"] == 24.0
    assert player["schedule"][0]["opponent"] == "LV"
    assert set(player["sources"]) == {"player_stats", "injury_data", "adp_feeds", "schedule_data"}


def test_single_player_metrics_have_envelope_and_top_level_convenience(player_pool):
    result = compute_base_projections(player_pool[0])
    assert result["metric"] == "base_projection"
    assert result["result"] == result["results"][0]
    assert result["base_projection"] == result["score"]
    assert result["by_player"]["rb-alpha"] == result["result"]
    assert result["metadata"]["deterministic"] is True


def test_base_projection_is_deterministic_and_uses_multiple_components(player_pool):
    first = compute_base_projections(player_pool)
    second = compute_base_projections(player_pool)
    assert first == second
    row = first["by_player"]["rb-alpha"]
    assert row["base_projection"] > 0
    assert row["components"]["provided_projection"] == 300.0
    assert row["components"]["history_games"] == 6


def test_confidence_scores_are_bounded_and_reward_evidence(player_pool):
    result = compute_confidence_scores(player_pool)
    assert all(0.0 <= row["confidence_score"] <= 1.0 for row in result["results"])
    assert result["by_player"]["rb-alpha"]["confidence_score"] > result["by_player"]["rb-beta"]["confidence_score"]


def test_volatility_curve_expands_when_projection_confidence_falls():
    base = {"name": "Curve", "position": "WR", "team": "DEN", "projection": 250, "history": [10, 22, 8, 25]}
    high = compute_volatility({**base, "player_id": "high", "projection_confidence": 0.95})
    low = compute_volatility({**base, "player_id": "low", "projection_confidence": 0.20})
    assert low["coefficient_of_variation"] > high["coefficient_of_variation"]
    assert low["standard_deviation"] > high["standard_deviation"]


def test_scarcity_and_value_over_replacement_respect_position_depth(player_pool):
    scarcity = compute_positional_scarcity(player_pool, replacement_rank={"RB": 2, "WR": 1})
    vor = compute_value_over_replacement(player_pool, replacement_rank={"RB": 2, "WR": 1})
    assert scarcity["by_player"]["rb-alpha"]["value_over_replacement"] > 0
    assert scarcity["by_player"]["rb-beta"]["value_over_replacement"] == 0
    assert vor["by_player"]["rb-alpha"]["value_over_replacement"] > vor["by_player"]["rb-beta"]["value_over_replacement"]


def test_similarity_returns_nearest_same_position_comparison(player_pool):
    result = compute_player_similarity(player_pool[0], player_pool, limit=2)
    assert result["player_id"] == "rb-alpha"
    assert result["archetype"]
    assert [row["player_id"] for row in result["comparisons"]] == ["rb-beta"]
    assert 0.0 <= result["comparisons"][0]["similarity"] <= 1.0


def test_draft_and_trade_values_are_ranked_structured_outputs(player_pool):
    draft = compute_draft_value(player_pool, current_pick=12, replacement_rank={"RB": 2, "WR": 1})
    trade = compute_trade_value(player_pool)
    assert draft["results"] == sorted(draft["results"], key=lambda row: -row["draft_value"])
    assert all("draft_value_score" in row for row in draft["results"])
    assert all("trade_value" in row and "components" in row for row in trade["results"])


def test_weekly_matchup_handles_bye_and_defensive_strength(player_pool):
    player = {**player_pool[0], "bye_week": 6}
    bye = compute_weekly_matchup_score(player, 6)
    easy = compute_weekly_matchup_score(player, 5, matchup={"defensive_strength": 0.2})
    hard = compute_weekly_matchup_score(player, 5, matchup={"defensive_strength": 0.9})
    assert bye["is_bye"] is True
    assert bye["adjusted_projection"] == 0.0
    assert easy["adjusted_projection"] > hard["adjusted_projection"]
    assert easy["weekly_matchup_score"] > hard["weekly_matchup_score"]


def test_trends_and_momentum_accept_a_plain_numeric_history():
    trend = compute_trend_lines([4, 6, 8, 10, 12, 14], window=3)
    momentum = compute_momentum([4, 6, 8, 10, 12, 14], window=3)
    assert trend["trend_direction"] == "up"
    assert trend["trend_line"][-1] == 12.0
    assert momentum["direction"] == "up"
    assert 50.0 < momentum["momentum_score"] <= 100.0


def test_rarity_health_usage_and_efficiency_are_ui_ready(player_pool):
    rarity = compute_rarity_tier(player_pool)
    health = compute_health_adjustments({**player_pool[0], "injury_status": "OUT", "injury_risk": 1.0})
    usage = compute_usage_rates(player_pool)
    efficiency = compute_efficiency_scores(player_pool)
    assert rarity["results"][0]["rarity_tier"] in {"Mythic", "Legendary", "Elite", "Pro", "Starter", "Depth"}
    assert health["health_adjusted_projection"] == 0.0
    assert all(0.0 <= row["usage_rate"] <= 1.0 for row in usage["results"])
    assert all(0.0 <= row["efficiency_score"] <= 100.0 for row in efficiency["results"])


def test_breakout_and_bust_probabilities_are_bounded(player_pool):
    breakout = compute_breakout_probability(player_pool)
    bust = compute_bust_probability(player_pool)
    assert all(0.0 <= row["breakout_probability"] <= 1.0 for row in breakout["results"])
    assert all(0.0 <= row["bust_probability"] <= 1.0 for row in bust["results"])
    assert breakout["by_player"]["rb-alpha"]["breakout_probability"] > 0


def test_final_projection_delegates_to_advanced_projection_engine(player_pool):
    result = compute_final_projection(player_pool[0])
    assert result["player_id"] == "rb-alpha"
    assert result["final_projection"] > 0
    assert "model_weights" in result
    assert "historical" in result["components"]


def test_quant_engine_facade_uses_its_configured_pool(player_pool):
    engine = QuantEngine(player_pool, scoring_mode="half_ppr")
    assert len(engine.players) == 3
    assert len(engine.compute_base_projections()["results"]) == 3
    assert engine.compute_player_similarity("rb-alpha")["comparisons"][0]["player_id"] == "rb-beta"
    assert engine.compute_weekly_matchup_score(player_pool[0], 1)["week"] == 1
    assert engine.compute_final_projection("rb-alpha")["player_id"] == "rb-alpha"
