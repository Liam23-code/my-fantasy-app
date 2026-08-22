from __future__ import annotations

import copy

import pytest

from quant.trend_engine import (
    calculate_rolling_average,
    compute_momentum,
    compute_trend_lines,
    momentum_score,
    rolling_average,
    rolling_efficiency,
    rolling_projection_deltas,
    rolling_usage,
    trend_direction,
)


def test_rolling_average_uses_a_trailing_window():
    assert rolling_average([1, 2, 3, 4], window=3) == [1.0, 1.5, 2.0, 3.0]
    assert calculate_rolling_average([2, 4, 8], window=2) == [2.0, 3.0, 6.0]


def test_rolling_average_ignores_missing_values_and_honors_min_periods():
    assert rolling_average([None, 2, None, 6], window=3, min_periods=2) == [None, None, None, 4.0]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"window": 0}, ValueError),
        ({"window": True}, TypeError),
        ({"window": 2, "min_periods": 3}, ValueError),
    ],
)
def test_rolling_average_validates_configuration(kwargs, error):
    with pytest.raises(error):
        rolling_average([1, 2], **kwargs)


def test_rolling_average_rejects_non_finite_or_non_numeric_values():
    with pytest.raises(ValueError, match="finite"):
        rolling_average([1, float("inf")])
    with pytest.raises(TypeError, match="numeric"):
        rolling_average([1, "not-a-number"])


def test_rolling_efficiency_uses_explicit_values_or_derives_points_per_opportunity():
    history = [
        {"points": 10, "opportunities": 5},
        {"points": 12, "opportunities": 4},
        {"points": 9, "efficiency_score": 1.5},
    ]
    assert rolling_efficiency(history, window=2) == [2.0, 2.5, 2.25]
    assert rolling_efficiency([1.0, 2.0, 3.0], window=2) == [1.0, 1.5, 2.5]


def test_rolling_usage_normalizes_percentage_shares_and_derives_volume():
    history = [
        {"usage_rate": 50},
        {"target_share": 60},
        {"carries": 8, "targets": 2, "team_opportunities": 20},
    ]
    assert rolling_usage(history, window=2) == [0.5, 0.55, 0.55]
    assert rolling_usage([0.2, 0.4, 0.6], window=2) == [0.2, 0.3, 0.5]


def test_rolling_projection_deltas_accepts_separate_actual_and_projection_series():
    assert rolling_projection_deltas([10, 12, 9], [8, 11, 10], window=2) == [2.0, 1.5, 0.0]
    with pytest.raises(ValueError, match="same length"):
        rolling_projection_deltas([10], [9, 8])


def test_rolling_projection_deltas_reads_records_and_projection_only_series():
    records = [
        {"points": 10, "projection": 8},
        {"points": 12, "projection": 11},
        {"points": 9, "projection": 10},
    ]
    assert rolling_projection_deltas(records, window=2) == [2.0, 1.5, 0.0]
    assert rolling_projection_deltas([10, 12, 11], window=2) == [0.0, 1.0, 0.5]


def test_momentum_score_and_direction_have_expected_signs():
    increasing = [10, 12, 14, 16, 18, 20]
    decreasing = list(reversed(increasing))

    assert momentum_score(increasing) > 0
    assert momentum_score(decreasing) < 0
    assert momentum_score([10, 10, 10]) == 0.0
    assert compute_momentum(increasing) == momentum_score(increasing)
    assert trend_direction(increasing) == "up"
    assert trend_direction(decreasing) == "down"
    assert trend_direction([10, 10.1, 10], threshold=0.05) == "flat"


def test_trend_direction_validates_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        trend_direction([1, 2], threshold=1.1)
    with pytest.raises(TypeError, match="numeric"):
        trend_direction([1, 2], threshold=True)


def test_compute_trend_lines_builds_complete_payload_from_weekly_records():
    history = [
        {"week": 1, "points": 10, "projection": 8, "usage_rate": 0.40, "opportunities": 5},
        {"week": 2, "points": 12, "projection": 11, "usage_rate": 0.50, "opportunities": 6},
        {"week": 3, "points": 15, "projection": 13, "usage_rate": 0.60, "opportunities": 5},
        {"week": 4, "points": 18, "projection": 16, "usage_rate": 0.70, "opportunities": 6},
    ]
    before = copy.deepcopy(history)

    result = compute_trend_lines(history, window=2)

    assert result == {
        "window": 2,
        "games": [1, 2, 3, 4],
        "points": [10.0, 12.0, 15.0, 18.0],
        "rolling_points": [10.0, 11.0, 13.5, 16.5],
        "efficiency": [2.0, 2.0, 3.0, 3.0],
        "rolling_efficiency": [2.0, 2.0, 2.5, 3.0],
        "usage": [0.4, 0.5, 0.6, 0.7],
        "rolling_usage": [0.4, 0.45, 0.55, 0.65],
        "projection_deltas": [2.0, 1.0, 2.0, 2.0],
        "rolling_projection_deltas": [2.0, 1.5, 1.5, 2.0],
        "momentum": momentum_score([10, 12, 15, 18], window=2),
        "direction": "up",
    }
    assert history == before


def test_compute_trend_lines_supports_projection_only_records():
    result = compute_trend_lines(
        [
            {"week": 1, "projection": 10},
            {"week": 2, "projection": 12},
            {"week": 3, "projection": 11},
        ],
        window=2,
    )
    assert result["points"] == [10.0, 12.0, 11.0]
    assert result["projection_deltas"] == [0.0, 2.0, -1.0]
    assert result["rolling_projection_deltas"] == [0.0, 1.0, 0.5]


def test_compute_trend_lines_supports_columnar_payloads_and_labels():
    result = compute_trend_lines(
        {
            "weeks": [4, 5, 6],
            "points": [8, 10, 12],
            "projected_points": [9, 9, 11],
            "usage_rate": [0.3, 0.4, 0.5],
        },
        window=2,
    )
    assert result["games"] == [4, 5, 6]
    assert result["points"] == [8.0, 10.0, 12.0]
    assert result["projection_deltas"] == [-1.0, 1.0, 1.0]
    assert result["rolling_usage"] == [0.3, 0.35, 0.45]


def test_compute_trend_lines_supports_week_keyed_mappings():
    result = compute_trend_lines({"weekly_projections": {3: {"points": 12}, 1: {"points": 8}, 2: {"points": 10}}}, window=2)
    assert result["games"] == [1, 2, 3]
    assert result["points"] == [8.0, 10.0, 12.0]


def test_empty_and_single_observation_histories_are_neutral():
    empty = compute_trend_lines([])
    assert empty["points"] == []
    assert empty["momentum"] == 0.0
    assert empty["direction"] == "flat"
    assert momentum_score([12]) == 0.0
    assert trend_direction([12]) == "flat"
