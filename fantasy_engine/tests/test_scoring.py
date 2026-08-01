"""Unit tests for fantasy.scoring."""

from __future__ import annotations

import pytest

from fantasy.scoring import batch_calculate_fantasy_points, calculate_fantasy_points


def test_standard_mode_scores_no_reception_points():
    result = calculate_fantasy_points({"receptions": 5, "receiving_yards": 50}, mode="standard")
    assert result["breakdown"]["receptions"] == 0.0
    assert result["total_points"] == pytest.approx(5.0)  # 50 yards / 10


def test_half_ppr_mode_scores_half_point_per_reception():
    result = calculate_fantasy_points({"receptions": 4}, mode="half-ppr")
    assert result["total_points"] == pytest.approx(2.0)
    assert result["mode"] == "half-ppr"


def test_ppr_mode_scores_full_point_per_reception():
    result = calculate_fantasy_points({"receptions": 4}, mode="ppr")
    assert result["total_points"] == pytest.approx(4.0)


def test_qb_passing_and_rushing_scored_correctly():
    projection = {"passing_yards": 250, "passing_tds": 2, "interceptions": 1, "rushing_yards": 40, "rushing_tds": 1}
    result = calculate_fantasy_points(projection, mode="ppr", bonuses=False)
    expected = 250 / 25 + 2 * 4 - 1 * 2 + 40 / 10 + 1 * 6
    assert result["total_points"] == pytest.approx(expected)


def test_missing_fields_default_to_zero_not_error():
    result = calculate_fantasy_points({}, mode="ppr")
    assert result["total_points"] == 0.0
    assert result["bonuses_applied"] == []


def test_unknown_stat_keys_are_ignored():
    result = calculate_fantasy_points({"passing_yards": 100, "some_future_stat": 999}, mode="ppr", bonuses=False)
    assert result["total_points"] == pytest.approx(4.0)


def test_zero_projection_scores_zero():
    projection = {stat: 0 for stat in ("passing_yards", "passing_tds", "rushing_yards", "receiving_yards", "receptions")}
    result = calculate_fantasy_points(projection, mode="ppr")
    assert result["total_points"] == 0.0


def test_negative_values_reduce_total_points():
    baseline = calculate_fantasy_points({"passing_yards": 200}, mode="ppr", bonuses=False)["total_points"]
    with_pick = calculate_fantasy_points({"passing_yards": 200, "interceptions": 2}, mode="ppr", bonuses=False)
    assert with_pick["total_points"] < baseline
    assert with_pick["breakdown"]["interceptions"] == pytest.approx(-4.0)


def test_numeric_strings_are_coerced():
    result = calculate_fantasy_points({"passing_yards": "250.0", "passing_tds": "2"}, mode="ppr", bonuses=False)
    assert result["total_points"] == pytest.approx(250 / 25 + 8)


def test_non_numeric_garbage_defaults_to_zero_instead_of_raising():
    result = calculate_fantasy_points({"passing_yards": "not-a-number", "passing_tds": None}, mode="ppr")
    assert result["total_points"] == 0.0


def test_bonus_triggers_at_threshold_and_not_below():
    just_under = calculate_fantasy_points({"passing_yards": 299}, mode="ppr")
    at_threshold = calculate_fantasy_points({"passing_yards": 300}, mode="ppr")
    assert just_under["bonuses_applied"] == []
    assert len(at_threshold["bonuses_applied"]) == 1
    assert at_threshold["bonuses_applied"][0]["stat"] == "passing_yards"
    assert at_threshold["total_points"] - just_under["total_points"] == pytest.approx(3 + 1 / 25)


def test_multiple_bonuses_can_stack():
    projection = {"rushing_yards": 100, "receiving_yards": 100}
    result = calculate_fantasy_points(projection, mode="ppr")
    stats_bonused = {bonus["stat"] for bonus in result["bonuses_applied"]}
    assert stats_bonused == {"rushing_yards", "receiving_yards"}


def test_bonuses_can_be_disabled():
    projection = {"passing_yards": 350}
    with_bonus = calculate_fantasy_points(projection, mode="ppr", bonuses=True)
    without_bonus = calculate_fantasy_points(projection, mode="ppr", bonuses=False)
    assert with_bonus["bonuses_applied"]
    assert without_bonus["bonuses_applied"] == []
    assert with_bonus["total_points"] - without_bonus["total_points"] == pytest.approx(3.0)


def test_invalid_mode_raises_value_error():
    with pytest.raises(ValueError):
        calculate_fantasy_points({"passing_yards": 100}, mode="bogus")


def test_custom_mode_without_rules_raises_value_error():
    with pytest.raises(ValueError):
        calculate_fantasy_points({"passing_yards": 100}, mode="custom")


def test_custom_mode_applies_supplied_multipliers():
    custom_rules = {"multipliers": {"passing_yards": 0.1, "receptions": 2.0}}
    result = calculate_fantasy_points({"passing_yards": 100, "receptions": 3}, mode="custom", custom_rules=custom_rules, bonuses=False)
    assert result["total_points"] == pytest.approx(100 * 0.1 + 3 * 2.0)


def test_custom_rules_overlay_on_top_of_ppr_defaults():
    custom_rules = {"multipliers": {"passing_tds": 6.0}}  # league scores passing TDs like rushing/receiving TDs
    result = calculate_fantasy_points({"passing_tds": 2, "receptions": 1}, mode="ppr", custom_rules=custom_rules, bonuses=False)
    assert result["breakdown"]["passing_tds"] == pytest.approx(12.0)
    assert result["breakdown"]["receptions"] == pytest.approx(1.0)  # ppr default untouched


def test_custom_bonus_rules_must_be_a_list():
    with pytest.raises(ValueError):
        calculate_fantasy_points({"passing_yards": 100}, mode="ppr", custom_rules={"bonuses": {"stat": "passing_yards"}})


def test_custom_bonus_rules_replace_defaults():
    custom_rules = {"bonuses": [{"stat": "receiving_tds", "threshold": 2, "points": 5}]}
    result = calculate_fantasy_points({"passing_yards": 400, "receiving_tds": 2}, mode="ppr", custom_rules=custom_rules)
    # The default 300+ passing yard bonus must NOT fire since bonuses were replaced wholesale.
    assert len(result["bonuses_applied"]) == 1
    assert result["bonuses_applied"][0] == {"stat": "receiving_tds", "threshold": 2, "points": 5, "value": 2}


def test_breakdown_values_sum_to_total_points_before_bonuses():
    projection = {"passing_yards": 210, "passing_tds": 1, "rushing_yards": 12, "receptions": 3, "receiving_yards": 25}
    result = calculate_fantasy_points(projection, mode="ppr", bonuses=False)
    assert sum(result["breakdown"].values()) == pytest.approx(result["total_points"])


def test_raw_projection_is_echoed_unmodified():
    projection = {"passing_yards": 100}
    result = calculate_fantasy_points(projection, mode="ppr")
    assert result["raw_projection"] is projection


def test_projection_must_be_a_mapping():
    with pytest.raises(TypeError):
        calculate_fantasy_points(["not", "a", "dict"], mode="ppr")  # type: ignore[arg-type]


def test_batch_scoring_matches_individual_scoring(named_player_projections):
    individual = [calculate_fantasy_points(p, mode="ppr") for p in named_player_projections]
    batched = batch_calculate_fantasy_points(named_player_projections, mode="ppr")
    assert [r["total_points"] for r in individual] == [r["total_points"] for r in batched]


@pytest.mark.parametrize(
    "fixture_name,min_expected_points",
    [
        ("lamar_jackson_projection", 10.0),
        ("josh_allen_projection", 10.0),
        ("saquon_barkley_projection", 15.0),
        ("cmc_projection", 15.0),
        ("puka_nacua_projection", 10.0),
        ("travis_kelce_projection", 5.0),
    ],
)
def test_named_players_score_realistic_ppr_points(fixture_name, min_expected_points, request):
    projection = request.getfixturevalue(fixture_name)
    result = calculate_fantasy_points(projection, mode="ppr")
    assert result["total_points"] >= min_expected_points


def test_qb_receiving_fields_never_contribute_when_zero(lamar_jackson_projection):
    result = calculate_fantasy_points(lamar_jackson_projection, mode="ppr")
    assert result["breakdown"]["receptions"] == 0.0
    assert result["breakdown"]["receiving_yards"] == 0.0
    assert result["breakdown"]["receiving_tds"] == 0.0


def test_rb_with_receiving_work_gets_nonzero_reception_points(saquon_barkley_projection):
    result = calculate_fantasy_points(saquon_barkley_projection, mode="ppr")
    assert result["breakdown"]["receptions"] > 0
    assert result["breakdown"]["receiving_yards"] > 0
