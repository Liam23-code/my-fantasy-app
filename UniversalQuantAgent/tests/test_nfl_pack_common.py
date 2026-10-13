"""Tests for the shared scoring / adjustment helpers of the NFL model upgrade pack."""
from __future__ import annotations

import logging
import math

import pytest

from modules import nfl_pack_common as c

FACTORS = (
    c.Factor("pressure", mean=0.25, sd=0.05, weight=2.0, direction=+1, label="Pressure", kind="rate", unit="%"),
    c.Factor("ttt", mean=2.7, sd=0.2, weight=1.0, direction=+1, label="Time to throw", unit="s", aliases=("time_to_throw",)),
    c.Factor("continuity", mean=0.6, sd=0.2, weight=1.0, direction=-1, label="OL continuity", kind="rate"),
)


# --- coercion ---------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [(3, 3.0), ("3.5", 3.5), ("1,204", 1204.0), ("31.5%", 0.315), (None, None), (True, None), ("n/a", None),
     (float("nan"), None), (float("inf"), None), ("", None), ([], None), (object(), None)],
)
def test_to_float_is_total(raw, expected):
    assert c.to_float(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [(0.31, 0.31), (31.0, 0.31), (1.0, 1.0), ("31%", 0.31), (0, 0.0), (-0.1, None), (101, None), (None, None), ("x", None)],
)
def test_to_rate_reads_percentages_and_rejects_nonsense(raw, expected):
    got = c.to_rate(raw)
    assert (got is None) if expected is None else math.isclose(got, expected)


def test_to_float_handles_pandas_na():
    pd = pytest.importorskip("pandas")
    assert c.to_float(pd.NA) is None
    assert c.to_float(pd.NaT) is None


# --- logistic ---------------------------------------------------------------
def test_sigmoid_logit_roundtrip_and_extremes():
    for p in (0.01, 0.2, 0.5, 0.9):
        assert math.isclose(c.sigmoid(c.logit(p)), p, rel_tol=1e-9)
    assert c.sigmoid(1e6) == pytest.approx(1.0)
    assert c.sigmoid(-1e6) == pytest.approx(0.0)
    assert -20.0 < c.logit(0.0) < -5.0  # clamped, finite
    assert math.isfinite(c.logit(0.0)) and math.isfinite(c.logit(1.0))


def test_neutral_inputs_return_exactly_the_base_rate():
    for inputs in (None, {}, {"pressure": 0.25, "ttt": 2.7, "continuity": 0.6}):
        scored = c.score_factors(inputs, FACTORS)
        assert scored["z"] == pytest.approx(0.0, abs=1e-12)
        assert c.logistic_probability(0.15, scored["z"], gain=1.5) == pytest.approx(0.15)


def test_missing_inputs_shrink_toward_neutral_not_amplify():
    full = c.score_factors({"pressure": 0.35, "ttt": 3.1, "continuity": 0.2}, FACTORS)
    partial = c.score_factors({"pressure": 0.35}, FACTORS)
    assert full["coverage"] == pytest.approx(1.0)
    assert partial["coverage"] == pytest.approx(2.0 / 4.0)
    assert 0 < partial["z"] < full["z"]
    assert partial["defaulted"] == ["ttt", "continuity"]


def test_direction_and_clipping_and_aliases():
    worse_line = c.score_factors({"continuity": 0.0}, FACTORS)  # direction -1: lower continuity -> higher score
    assert worse_line["z"] > 0
    extreme = c.score_factors({"pressure": 5.0 / 100.0 * 1000}, FACTORS)  # 50.0 -> read as 50% -> far above mean
    assert extreme["contributions"]["pressure"]["z"] == pytest.approx(c.Z_CLIP, abs=1e-3)
    aliased = c.score_factors({"time_to_throw": 3.1}, FACTORS)
    assert aliased["used"] == ["ttt"] and aliased["contributions"]["ttt"]["via"] == "time_to_throw"


def test_bad_values_are_treated_as_missing():
    scored = c.score_factors({"pressure": "n/a", "ttt": float("nan"), "continuity": None}, FACTORS)
    assert scored["used"] == [] and scored["coverage"] == 0.0 and scored["z"] == 0.0


def test_factor_validation():
    with pytest.raises(ValueError):
        c.Factor("x", 0.0, 0.0, 1.0, 1, "x")
    with pytest.raises(ValueError):
        c.Factor("x", 0.0, 1.0, 0.0, 1, "x")
    with pytest.raises(ValueError):
        c.Factor("x", 0.0, 1.0, 1.0, 0, "x")
    with pytest.raises(ValueError):
        c.Factor("x", 0.0, 1.0, 1.0, 1, "x", kind="nope")


# --- run_factor_model -------------------------------------------------------
def test_run_factor_model_shape_and_logging(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        result = c.run_factor_model(
            model="demo", primary_key="demo_probability", inputs={"pressure": 0.35, "ttt": 3.0},
            factors=FACTORS, base_rate=0.15, gain=1.5, basis="unit-test prior",
        )
    assert result["demo_probability"] == result["probability"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["probability"] > result["reference_probability"] == 0.15
    assert result["references"] == {"demo_probability": 0.15}
    assert result["drivers"] and "Pressure" in result["drivers"][0]
    assert any("demo_probability=" in rec.getMessage() for rec in caplog.records)


def test_probability_is_monotone_in_each_factor():
    base = {"pressure": 0.25, "ttt": 2.7, "continuity": 0.6}
    p0 = c.run_factor_model(model="m", primary_key="p", inputs=base, factors=FACTORS, base_rate=0.2, gain=1.5, basis="t")["p"]
    for key, delta in (("pressure", +0.05), ("ttt", +0.2), ("continuity", -0.2)):
        bumped = dict(base, **{key: base[key] + delta})
        assert c.run_factor_model(model="m", primary_key="p", inputs=bumped, factors=FACTORS, base_rate=0.2, gain=1.5, basis="t")["p"] > p0


# --- adjustments ------------------------------------------------------------
def test_adjustment_factor_literal_and_recentered():
    assert c.adjustment_factor(0.4, 0.65, direction=-1, recenter=False) == pytest.approx(1 - 0.4 * 0.65)
    assert c.adjustment_factor(0.4, 0.45, direction=+1, recenter=False) == pytest.approx(1 + 0.4 * 0.45)
    assert c.adjustment_factor(0.15, 0.65, direction=-1, reference=0.15) == pytest.approx(1.0)
    assert c.adjustment_factor(0.4, 0.65, direction=-1, reference=0.15) == pytest.approx((1 - 0.26) / (1 - 0.0975))
    assert c.adjustment_factor(1.0, 5.0, direction=-1, recenter=False) == 0.0  # never negative


def test_scale_field_records_audit_and_skips_missing():
    projection = {"qb_yards": 250.0, "qb_completion_pct": 0.66}
    log: list = []
    c.scale_field(projection, "qb_yards", stage="qb_collapse", probability=0.4, coefficient=0.65, direction=-1,
                  reference=0.15, recenter=False, adjustments=log)
    c.scale_field(projection, "qb_completion_pct", stage="qb_collapse", probability=0.99, coefficient=1.0, direction=+1,
                  reference=0.0, recenter=False, adjustments=log, cap=1.0)
    c.scale_field(projection, "qb_td_prob", stage="qb_collapse", probability=0.4, coefficient=0.35, direction=-1,
                  reference=0.15, recenter=True, adjustments=log)
    assert projection["qb_yards"] == pytest.approx(250.0 * (1 - 0.26))
    assert projection["qb_completion_pct"] == 1.0  # capped
    assert log[0]["applied"] and log[0]["before"] == 250.0 and log[0]["direction"] == "haircut"
    assert log[2] == {"stage": "qb_collapse", "field": "qb_td_prob", "applied": False, "reason": "field missing or not numeric"}


def test_add_to_field_recenters_to_the_deviation():
    projection = {"wr_yards": 60.0}
    log: list = []
    c.add_to_field(projection, "wr_yards", stage="explosive_play", probability=0.30, amount=18.0, reference=0.20,
                   recenter=True, adjustments=log)
    assert projection["wr_yards"] == pytest.approx(60.0 + 0.10 * 18.0)
    c.add_to_field(projection, "wr_yards", stage="explosive_play", probability=0.30, amount=18.0, reference=0.20,
                   recenter=False, adjustments=log)
    assert projection["wr_yards"] == pytest.approx(60.0 + 1.8 + 0.30 * 18.0)


def test_guard_cumulative_clamps_and_reports():
    original = {"wr_yards": 100.0, "te_yards": 50.0, "rb_yards": 0.0}
    adjusted = {"wr_yards": 300.0, "te_yards": 45.0, "rb_yards": 5.0}
    notes = c.guard_cumulative(original, adjusted, ["wr_yards", "te_yards", "rb_yards"])
    assert adjusted["wr_yards"] == pytest.approx(200.0)
    assert adjusted["te_yards"] == 45.0
    assert adjusted["rb_yards"] == 5.0  # zero baseline is not guarded
    assert [n["field"] for n in notes] == ["wr_yards"]


def test_blend_man_zone():
    assert c.blend_man_zone(2.0, 1.0, 0.5) == pytest.approx(1.5)
    assert c.blend_man_zone(2.0, None, 0.5) == 2.0
    assert c.blend_man_zone(None, None, 0.5) is None
    assert c.blend_man_zone(2.0, 1.0, None) == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)


def test_merge_inputs_later_wins_but_none_never_erases():
    assert c.merge_inputs({"a": 1, "b": 2}, {"b": None, "c": 3}, None, {"a": 9}) == {"a": 9, "b": 2, "c": 3}
