"""Tests for the offline betting odds engine: math primitives and the odds loader."""

from __future__ import annotations

import json

import pytest

from betting.odds_loader import (
    OddsLoadError,
    load_default_odds,
    load_uploaded_odds,
    merge_odds,
    unified_odds,
)
from betting.odds_math import (
    american_to_decimal,
    decimal_to_american,
    edge,
    edge_vs_fair,
    expected_value,
    fair_price_from_probability,
    hold,
    implied_probability,
    remove_vig_two_way,
)

# --- odds_math -----------------------------------------------------------


def test_american_to_decimal_negative_and_positive():
    assert american_to_decimal(-110) == pytest.approx(1.9090909, rel=1e-6)
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_american_to_decimal_rejects_zero():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_decimal_to_american_round_trips_common_prices():
    for price in (-110, -200, 100, 150, 300):
        decimal = american_to_decimal(price)
        assert decimal_to_american(decimal) == price


def test_implied_probability_matches_known_values():
    assert implied_probability(-110) == pytest.approx(0.5238095, rel=1e-6)
    assert implied_probability(100) == pytest.approx(0.5)


def test_remove_vig_two_way_symmetric_market_is_a_coin_flip():
    fair_a, fair_b = remove_vig_two_way(-110, -110)
    assert fair_a == pytest.approx(0.5)
    assert fair_b == pytest.approx(0.5)
    assert fair_a + fair_b == pytest.approx(1.0)


def test_remove_vig_two_way_asymmetric_market_sums_to_one():
    fair_a, fair_b = remove_vig_two_way(-150, 130)
    assert fair_a + fair_b == pytest.approx(1.0)
    assert fair_a > fair_b  # the -150 favorite is more likely than the +130 dog


def test_hold_is_positive_for_a_standard_vig_market():
    assert hold(-110, -110) == pytest.approx(0.047619, rel=1e-4)


def test_fair_price_from_probability_even_money():
    assert fair_price_from_probability(0.5) == pytest.approx(100, abs=1)


def test_expected_value_positive_when_true_probability_beats_market():
    # -110 implies ~52.4%; a true 55% win probability should be +EV.
    assert expected_value(0.55, -110, 100) > 0


def test_expected_value_negative_when_true_probability_below_market():
    assert expected_value(0.45, -110, 100) < 0


def test_edge_vs_fair_exceeds_raw_edge_by_roughly_half_the_hold():
    raw = edge(0.55, -110)
    vs_fair = edge_vs_fair(0.55, -110, -110, side="a")
    assert vs_fair > raw


# --- odds_loader -----------------------------------------------------------


def test_load_default_odds_missing_file_returns_empty_not_error(tmp_path):
    result = load_default_odds(tmp_path / "does_not_exist.json")
    assert result == {"games": {}, "player_props": {}}


def test_load_default_odds_reads_real_generated_file():
    result = load_default_odds()
    assert len(result["player_props"]) > 0
    sample = next(iter(result["player_props"].values()))
    assert sample["source"] == "default"
    assert sample["over_price"] and sample["under_price"]


def test_load_uploaded_odds_from_csv_props():
    csv_text = (
        "player_id,name,team,market,line,over_price,under_price\n"
        "00-1234,Test Player,KC,receiving_yards,65.5,-115,-105\n"
    )
    result = load_uploaded_odds(csv_text, file_format="csv")
    entry = result["player_props"]["00-1234:receiving_yards"]
    assert entry["line"] == 65.5
    assert entry["over_price"] == -115
    assert entry["source"] == "uploaded"


def test_load_uploaded_odds_from_json_list():
    payload = [
        {"player_id": "00-9999", "name": "X", "market": "receptions", "line": 5.5, "over_price": -120, "under_price": 100}
    ]
    result = load_uploaded_odds(payload)
    assert result["player_props"]["00-9999:receptions"]["line"] == 5.5


def test_load_uploaded_odds_from_game_csv():
    csv_text = "home_team,away_team,week,moneyline_home,moneyline_away,total_line\nKC,BUF,1,-150,130,47.5\n"
    result = load_uploaded_odds(csv_text, file_format="csv")
    assert len(result["games"]) == 1
    game = next(iter(result["games"].values()))
    assert game["home_team"] == "KC"
    assert game["moneyline"]["home"] == -150
    assert game["total"]["line"] == 47.5


def test_load_uploaded_odds_invalid_json_raises():
    with pytest.raises(OddsLoadError):
        load_uploaded_odds("{not valid json", file_format="json")


def test_load_uploaded_odds_missing_path_raises():
    with pytest.raises(OddsLoadError):
        load_uploaded_odds("Z:/definitely/not/a/real/path.csv")


def test_merge_odds_upload_overrides_matching_key_only():
    default = {
        "games": {},
        "player_props": {
            "a:pts": {"player_id": "a", "market": "pts", "line": 20.5, "source": "default"},
            "b:pts": {"player_id": "b", "market": "pts", "line": 15.5, "source": "default"},
        },
    }
    uploaded = {"games": {}, "player_props": {"a:pts": {"player_id": "a", "market": "pts", "line": 22.5, "source": "uploaded"}}}
    merged = merge_odds(default, uploaded)
    assert merged["player_props"]["a:pts"]["line"] == 22.5
    assert merged["player_props"]["a:pts"]["source"] == "uploaded"
    assert merged["player_props"]["b:pts"]["line"] == 15.5
    assert merged["player_props"]["b:pts"]["source"] == "default"
    assert len(merged["player_props"]) == 2


def test_merge_odds_upload_adds_new_keys():
    default = {"games": {}, "player_props": {"a:pts": {"player_id": "a", "market": "pts", "line": 20.5}}}
    uploaded = {"games": {}, "player_props": {"c:pts": {"player_id": "c", "market": "pts", "line": 10.5}}}
    merged = merge_odds(default, uploaded)
    assert len(merged["player_props"]) == 2


def test_merge_odds_with_no_upload_returns_default_unchanged():
    default = {"games": {}, "player_props": {"a:pts": {"player_id": "a", "market": "pts", "line": 20.5}}}
    merged = merge_odds(default, None)
    assert merged == default
    assert merged is not default  # defensive copy, not the same object


def test_unified_odds_merges_default_and_upload():
    csv_text = "player_id,name,team,market,line,over_price,under_price\n00-9999,New Player,SF,receptions,4.5,-110,-110\n"
    result = unified_odds(uploaded=csv_text, uploaded_format="csv")
    assert "00-9999:receptions" in result["player_props"]
    assert len(result["player_props"]) > 1  # default props still present


def test_unified_odds_with_no_upload_equals_default():
    assert unified_odds() == load_default_odds()


def test_odds_loader_is_deterministic_across_runs():
    a = unified_odds()
    b = unified_odds()
    assert a == b


def test_generated_default_odds_file_is_valid_json():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "odds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "player_props" in payload
    assert len(payload["player_props"]) > 0
