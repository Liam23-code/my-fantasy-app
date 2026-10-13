"""Tests for ``modules.explosive_play_model`` (offline; no data feed is touched)."""
from __future__ import annotations

import ast
import copy
import json
import logging
import math
from pathlib import Path

import pytest

from modules import explosive_play_model as m
from modules import nfl_pack_common as c

PASS = m.PASS_KEY
RUN = m.RUN_KEY

#: Every input the brief names, with a clearly "explosive" and a clearly "contained" value.
HIGH = {
    "wr_separation": 3.6,
    "wr_deep_route_rate": 0.30,
    "qb_deep_attempt_rate": 0.17,
    "opp_explosive_play_prevention": 10.0,
    "opp_missed_tackle_rate": 0.115,
    "opp_coverage_bust_rate": 0.07,
    "rb_breakaway_rate": 0.08,
    "ol_run_block_win_rate": 0.78,
}
LOW = {
    "wr_separation": 2.2,
    "wr_deep_route_rate": 0.06,
    "qb_deep_attempt_rate": 0.07,
    "opp_explosive_play_prevention": 92.0,
    "opp_missed_tackle_rate": 0.058,
    "opp_coverage_bust_rate": 0.015,
    "rb_breakaway_rate": 0.015,
    "ol_run_block_win_rate": 0.66,
}


def _average() -> dict[str, float]:
    return {f.key: f.mean for f in m.FACTORS}


def _shifted(sds: float) -> dict[str, float]:
    """Every input ``sds`` SD in the direction that raises the output, kept inside its legal domain.

    A grade lives on 0-100 (so it tops out near 1.7 SD), a rate on 0-1 and a separation
    on (0, 20] yards; a value outside those is a data error the model treats as missing.
    """
    out: dict[str, float] = {}
    for f in m.FACTORS:
        value = f.mean + f.direction * sds * f.sd
        if f.key == "opp_explosive_play_prevention":
            value = min(100.0, max(0.0, value))
        elif f.kind == "rate":
            value = min(1.0, max(0.0, value))
        elif f.key == "wr_separation":
            value = min(20.0, max(0.05, value))
        out[f.key] = value
    return out


def _result(p_pass=None, p_run=None, ref_pass=0.2727, ref_run=0.1263) -> dict:
    out: dict = {"references": {PASS: ref_pass, RUN: ref_run}}
    if p_pass is not None:
        out[PASS] = p_pass
    if p_run is not None:
        out[RUN] = p_run
    return out


# ---------------------------------------------------------------------------
# apply: the brief's formula
# ---------------------------------------------------------------------------
def test_brief_coefficients_are_exposed_verbatim():
    assert m.WR_YARDS_COEFFICIENT == 18.0
    assert m.RB_YARDS_COEFFICIENT == 12.0


@pytest.mark.parametrize("p_pass, p_run", [(0.0, 0.0), (0.10, 0.05), (0.2727, 0.1263), (0.45, 0.30), (1.0, 1.0)])
def test_literal_formula_equals_the_brief(p_pass, p_run):
    projection = {"wr_yards": 62.0, "rb_yards": 48.0}
    out, log = m.apply_explosive_play(projection, _result(p_pass, p_run), recenter=False)
    assert out["wr_yards"] == pytest.approx(62.0 + p_pass * 18)  # wr_yards += explosive_play_prob_pass * 18
    assert out["rb_yards"] == pytest.approx(48.0 + p_run * 12)  # rb_yards += explosive_play_prob_run * 12
    assert [e["applied"] for e in log] == [True, True]
    assert all(e["recentered"] is False for e in log)


def test_recenter_is_an_exact_noop_at_the_reference():
    projection = {"wr_yards": 62.0, "rb_yards": 48.0}
    out, log = m.apply_explosive_play(projection, _result(0.2727, 0.1263))
    assert out == projection
    assert all(e["delta"] == 0 for e in log)


def test_recenter_moves_only_the_deviation_from_the_reference():
    out, _ = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, _result(0.37, 0.19))
    assert out["wr_yards"] == pytest.approx(60.0 + (0.37 - 0.2727) * 18)
    assert out["rb_yards"] == pytest.approx(50.0 + (0.19 - 0.1263) * 12)
    out, _ = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, _result(0.15, 0.06))
    assert out["wr_yards"] < 60.0 and out["rb_yards"] < 50.0  # a contained matchup lowers the line


def test_recentered_and_literal_differ_by_exactly_the_reference_term():
    projection = {"wr_yards": 60.0, "rb_yards": 50.0}
    res = _result(0.40, 0.20)
    centred, _ = m.apply_explosive_play(projection, res, recenter=True)
    literal, _ = m.apply_explosive_play(projection, res, recenter=False)
    assert literal["wr_yards"] - centred["wr_yards"] == pytest.approx(0.2727 * 18)
    assert literal["rb_yards"] - centred["rb_yards"] == pytest.approx(0.1263 * 12)


def test_apply_does_not_mutate_the_projection_and_returns_a_new_dict():
    projection = {"wr_yards": 60.0, "rb_yards": 50.0, "meta": {"name": "x"}}
    before = copy.deepcopy(projection)
    out, _ = m.apply_explosive_play(projection, _result(0.5, 0.3))
    assert projection == before
    assert out is not projection
    assert out["wr_yards"] != projection["wr_yards"]


def test_apply_does_not_mutate_the_result():
    res = _result(0.5, 0.3)
    before = copy.deepcopy(res)
    m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, res)
    assert res == before


def test_missing_field_is_recorded_not_raised():
    out, log = m.apply_explosive_play({"rb_yards": 50.0}, _result(0.5, 0.3))
    assert "wr_yards" not in out
    wr = next(e for e in log if e["field"] == "wr_yards")
    assert wr["applied"] is False and "missing" in wr["reason"]
    rb = next(e for e in log if e["field"] == "rb_yards")
    assert rb["applied"] is True and out["rb_yards"] > 50.0


def test_non_numeric_field_is_recorded_as_not_applied():
    out, log = m.apply_explosive_play({"wr_yards": "n/a", "rb_yards": None}, _result(0.5, 0.3))
    assert out == {"wr_yards": "n/a", "rb_yards": None}
    assert [e["applied"] for e in log] == [False, False]


def test_other_fields_are_copied_through_untouched():
    projection = {"wr_yards": 60.0, "te_yards": 40.0, "qb_yards": 250.0, "rb_td_prob": 0.4, "note": "keep me"}
    out, _ = m.apply_explosive_play(projection, _result(0.5, 0.3))
    for key in ("te_yards", "qb_yards", "rb_td_prob", "note"):
        assert out[key] == projection[key]
    assert set(out) == set(projection)


def test_yards_never_go_negative():
    out, _ = m.apply_explosive_play({"wr_yards": 1.0, "rb_yards": 0.5}, _result(0.0, 0.0))
    assert out["wr_yards"] == 0.0 and out["rb_yards"] == 0.0


def test_result_without_probabilities_is_a_recorded_noop():
    projection = {"wr_yards": 60.0, "rb_yards": 50.0}
    for bad in ({}, None, {PASS: "n/a", RUN: float("nan")}, "garbage"):
        out, log = m.apply_explosive_play(projection, bad)
        assert out == projection
        assert [e["applied"] for e in log] == [False, False]
        assert all("reason" in e for e in log)


def test_apply_reads_the_references_carried_in_the_result():
    res = _result(0.50, 0.30, ref_pass=0.45, ref_run=0.25)
    out, _ = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, res)
    assert out["wr_yards"] == pytest.approx(60.0 + 0.05 * 18)
    assert out["rb_yards"] == pytest.approx(50.0 + 0.05 * 12)


def test_apply_falls_back_to_module_base_rates_without_references():
    out, _ = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, {PASS: 0.40, RUN: 0.20})
    assert out["wr_yards"] == pytest.approx(60.0 + (0.40 - m.BASE_RATE_PASS) * 18)
    assert out["rb_yards"] == pytest.approx(50.0 + (0.20 - m.BASE_RATE_RUN) * 12)


def test_literal_formula_on_a_computed_result_adds_the_whole_probability_times_the_coefficient():
    res = m.compute_explosive_play(HIGH)
    out, log = m.apply_explosive_play({"wr_yards": 62.0, "rb_yards": 55.0}, res, recenter=False)
    assert out["wr_yards"] == pytest.approx(62.0 + res[PASS] * 18)
    assert out["rb_yards"] == pytest.approx(55.0 + res[RUN] * 12)
    assert [e["applied"] for e in log] == [True, True]


def test_generic_probability_alias_drives_the_pass_side_only():
    # a result that only carries the contract's generic aliases is read as the pass model
    res = {"probability": 0.40, "reference_probability": 0.30}
    out, log = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, res)
    assert out["wr_yards"] == pytest.approx(60.0 + (0.40 - 0.30) * 18)
    assert out["rb_yards"] == 50.0
    assert [(e["field"], e["applied"]) for e in log] == [("wr_yards", True), ("rb_yards", False)]


def test_out_of_range_probabilities_are_clamped_to_0_1():
    out, _ = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, _result(7.0, -3.0), recenter=False)
    assert out["wr_yards"] == pytest.approx(60.0 + 1.0 * 18)
    assert out["rb_yards"] == pytest.approx(50.0 + 0.0 * 12)


def test_audit_entries_name_stage_field_and_inputs():
    _, log = m.apply_explosive_play({"wr_yards": 60.0, "rb_yards": 50.0}, _result(0.4, 0.2))
    assert [(e["stage"], e["field"]) for e in log] == [("explosive_play", "wr_yards"), ("explosive_play", "rb_yards")]
    assert log[0]["amount"] == 18.0 and log[1]["amount"] == 12.0
    assert log[0]["probability"] == 0.4 and log[0]["reference"] == 0.2727


# ---------------------------------------------------------------------------
# compute: contract shape
# ---------------------------------------------------------------------------
def test_result_carries_every_contract_key_and_both_outputs():
    res = m.compute_explosive_play(HIGH)
    for key in (
        "model", PASS, RUN, "probability", "reference_probability", "references", "z_score", "coverage",
        "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
    ):
        assert key in res, key
    assert res["model"] == "explosive_play"
    assert res["probability"] == res[PASS]
    assert res["reference_probability"] == res["references"][PASS]
    assert set(res["references"]) == {PASS, RUN}
    assert res["references"] == {PASS: m.BASE_RATE_PASS, RUN: m.BASE_RATE_RUN}


def test_result_is_json_serialisable():
    json.dumps(m.compute_explosive_play(HIGH))
    json.dumps(m.compute_explosive_play({}))


def test_neutral_in_reference_out():
    empty = m.compute_explosive_play({})
    none = m.compute_explosive_play(None)
    average = m.compute_explosive_play(_average())
    for res in (empty, none, average):
        assert res[PASS] == res["references"][PASS] == m.BASE_RATE_PASS
        assert res[RUN] == res["references"][RUN] == m.BASE_RATE_RUN
        assert res["z_score"] == 0.0
    assert average["inputs_defaulted"] == [] and empty["inputs_used"] == []


def test_a_clear_matchup_beats_the_reference_and_a_clear_non_matchup_trails_it():
    hi, lo = m.compute_explosive_play(HIGH), m.compute_explosive_play(LOW)
    assert hi[PASS] > m.BASE_RATE_PASS > lo[PASS]
    assert hi[RUN] > m.BASE_RATE_RUN > lo[RUN]


# ---------------------------------------------------------------------------
# compute: monotonicity, independence, bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factor", m.PASS_FACTORS, ids=lambda f: f.key)
def test_each_pass_input_is_monotone_in_its_stated_direction(factor):
    base = _average()
    previous = m.compute_explosive_play(base)[PASS]
    for sds in (0.5, 1.0, 1.5):
        bumped = dict(base, **{factor.key: factor.mean + factor.direction * sds * factor.sd})
        now = m.compute_explosive_play(bumped)[PASS]
        assert now > previous, (factor.key, sds)
        previous = now
    worse = dict(base, **{factor.key: factor.mean - factor.direction * factor.sd})
    assert m.compute_explosive_play(worse)[PASS] < m.BASE_RATE_PASS


@pytest.mark.parametrize("factor", m.RUN_FACTORS, ids=lambda f: f.key)
def test_each_run_input_is_monotone_in_its_stated_direction(factor):
    base = _average()
    previous = m.compute_explosive_play(base)[RUN]
    for sds in (0.5, 1.0, 1.5):
        bumped = dict(base, **{factor.key: factor.mean + factor.direction * sds * factor.sd})
        now = m.compute_explosive_play(bumped)[RUN]
        assert now > previous, (factor.key, sds)
        previous = now
    worse = dict(base, **{factor.key: factor.mean - factor.direction * factor.sd})
    assert m.compute_explosive_play(worse)[RUN] < m.BASE_RATE_RUN


def test_pass_only_and_run_only_inputs_do_not_leak_across_models():
    pass_only = ("wr_separation", "wr_deep_route_rate", "qb_deep_attempt_rate", "opp_coverage_bust_rate")
    run_only = ("rb_breakaway_rate", "ol_run_block_win_rate")
    res = m.compute_explosive_play({k: HIGH[k] for k in pass_only})
    assert res[PASS] > m.BASE_RATE_PASS and res[RUN] == m.BASE_RATE_RUN
    res = m.compute_explosive_play({k: HIGH[k] for k in run_only})
    assert res[RUN] > m.BASE_RATE_RUN and res[PASS] == m.BASE_RATE_PASS


def test_shared_defence_inputs_move_both_models():
    shared = {"opp_explosive_play_prevention": 5.0, "opp_missed_tackle_rate": 0.12}
    res = m.compute_explosive_play(shared)
    assert res[PASS] > m.BASE_RATE_PASS and res[RUN] > m.BASE_RATE_RUN


def test_two_sd_everywhere_stays_well_below_certainty_and_one_sd_is_a_plausible_bad_day():
    one, two, three = (m.compute_explosive_play(_shifted(k)) for k in (1.0, 2.0, 3.0))
    # documented numbers (module docstring): weighted z = 1 -> pass 0.370 / run 0.192; the realisable
    # "+2 SD everywhere" case (a grade tops out at 1.73 SD, so z ~ 1.95) -> pass 0.47 / run 0.28
    assert one[PASS] == pytest.approx(0.370, abs=0.002) and one[RUN] == pytest.approx(0.192, abs=0.002)
    assert two[PASS] == pytest.approx(0.474, abs=0.004) and two[RUN] == pytest.approx(0.277, abs=0.004)
    # the absolute ceiling: the 0-100 grade stops at 1.73 SD, every other input sits on the +3 SD clip
    assert three[PASS] == pytest.approx(0.565, abs=0.003) and three[RUN] == pytest.approx(0.365, abs=0.003)
    assert three["z_scores"][PASS] == pytest.approx(2.76, abs=0.01)
    assert three["z_scores"][RUN] == pytest.approx(2.76, abs=0.01)
    for res in (one, two, three):
        assert res[PASS] < 0.90 and res[RUN] < 0.90
    assert one[PASS] < two[PASS] < three[PASS] and one[RUN] < two[RUN] < three[RUN]
    # and the pure arithmetic of the documented table
    assert c.logistic_probability(m.BASE_RATE_PASS, 3.0, m.GAIN_PASS) == pytest.approx(0.591, abs=0.001)
    assert c.logistic_probability(m.BASE_RATE_RUN, 3.0, m.GAIN_RUN) == pytest.approx(0.393, abs=0.001)


def test_probabilities_are_bounded_even_for_extreme_inputs():
    extreme_high = {k: (100 if f.kind == "rate" else 1e6) for f in m.FACTORS for k in [f.key]}
    for inputs in (extreme_high, {k: 0.0 for k in extreme_high}, _shifted(10.0), _shifted(-10.0)):
        res = m.compute_explosive_play(inputs)
        for key in (PASS, RUN):
            assert 0.0 <= res[key] <= 1.0 and math.isfinite(res[key])
    # the z clip keeps a 10-SD input exactly as strong as a 3-SD one
    assert m.compute_explosive_play(_shifted(10.0))[PASS] == m.compute_explosive_play(_shifted(3.0))[PASS]


# ---------------------------------------------------------------------------
# compute: bad data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "n/a", "", "abc", True, False, [], {}, object()])
def test_unusable_values_are_treated_as_missing(bad):
    for key in HIGH:
        res = m.compute_explosive_play({key: bad})
        assert res[PASS] == m.BASE_RATE_PASS and res[RUN] == m.BASE_RATE_RUN, (key, bad)
        assert res["inputs_used"] == [] and key in res["inputs_defaulted"]


@pytest.mark.parametrize(
    "key, value",
    [
        ("wr_deep_route_rate", -0.1),
        ("qb_deep_attempt_rate", -5),
        ("opp_missed_tackle_rate", 250),
        ("opp_coverage_bust_rate", 101),
        ("rb_breakaway_rate", -0.0001),
        ("ol_run_block_win_rate", 1e9),
        ("opp_explosive_play_prevention", -10.0),
        ("opp_explosive_play_prevention", 140.0),
        ("wr_separation", -1.5),
        ("wr_separation", 0.0),
        ("wr_separation", 1e6),
    ],
)
def test_negative_and_out_of_domain_values_are_data_errors_not_extreme_signals(key, value):
    res = m.compute_explosive_play({key: value})
    assert res[PASS] == m.BASE_RATE_PASS and res[RUN] == m.BASE_RATE_RUN
    assert key in res["inputs_defaulted"]


def test_a_bad_value_does_not_poison_the_good_ones():
    res = m.compute_explosive_play({"wr_deep_route_rate": float("nan"), "rb_breakaway_rate": 0.08, "wr_separation": "x"})
    assert res["inputs_used"] == ["rb_breakaway_rate"]
    assert res[RUN] > m.BASE_RATE_RUN and res[PASS] == m.BASE_RATE_PASS


def test_rates_accept_fractions_percentages_and_percent_strings_identically():
    a = m.compute_explosive_play({"wr_deep_route_rate": 0.25, "rb_breakaway_rate": 0.07})
    b = m.compute_explosive_play({"wr_deep_route_rate": 25.0, "rb_breakaway_rate": 7})
    d = m.compute_explosive_play({"wr_deep_route_rate": "25%", "rb_breakaway_rate": "7%"})
    assert a[PASS] == b[PASS] == d[PASS] and a[RUN] == b[RUN] == d[RUN]
    assert a[PASS] > m.BASE_RATE_PASS


def test_numeric_strings_and_numpy_style_scalars_are_read():
    class FakeScalar:
        def __float__(self):
            return 3.4

    a = m.compute_explosive_play({"wr_separation": 3.4})
    assert m.compute_explosive_play({"wr_separation": "3.4"})[PASS] == a[PASS]
    assert m.compute_explosive_play({"wr_separation": FakeScalar()})[PASS] == a[PASS]


def test_non_mapping_inputs_never_raise():
    for junk in ([1, 2, 3], "abc", 5, 3.2, ("wr_separation",), object()):
        res = m.compute_explosive_play(junk)  # type: ignore[arg-type]
        assert res[PASS] == m.BASE_RATE_PASS and res[RUN] == m.BASE_RATE_RUN


def test_unknown_keys_are_ignored():
    res = m.compute_explosive_play({"not_a_key": 99, "wr_separation": 3.4})
    assert res["inputs_used"] == ["wr_separation"]
    assert res[PASS] == m.compute_explosive_play({"wr_separation": 3.4})[PASS]


def test_wr1_separation_alias_is_accepted_and_canonical_key_wins():
    canonical = m.compute_explosive_play({"wr_separation": 3.5})
    alias = m.compute_explosive_play({"wr1_separation": 3.5})
    assert alias[PASS] == canonical[PASS] > m.BASE_RATE_PASS
    assert alias["contributions"]["wr_separation"]["via"] == "wr1_separation"
    both = m.compute_explosive_play({"wr_separation": 3.5, "wr1_separation": 1.0})
    assert both[PASS] == canonical[PASS]


# ---------------------------------------------------------------------------
# compute: missing data, determinism, logging
# ---------------------------------------------------------------------------
def test_missing_data_shrinks_toward_neutral_not_amplified():
    full = m.compute_explosive_play(HIGH)
    two = m.compute_explosive_play({"wr_deep_route_rate": HIGH["wr_deep_route_rate"], "rb_breakaway_rate": HIGH["rb_breakaway_rate"]})
    assert two[PASS] > m.BASE_RATE_PASS and two[RUN] > m.BASE_RATE_RUN
    assert two[PASS] < full[PASS] and two[RUN] < full[RUN]
    assert two["coverage"] < full["coverage"] == 1.0
    assert set(two["inputs_defaulted"]) == set(HIGH) - {"wr_deep_route_rate", "rb_breakaway_rate"}


def test_coverage_is_pooled_across_both_models_and_reported_per_output():
    pass_only = m.compute_explosive_play({k: HIGH[k] for k in ("wr_deep_route_rate", "qb_deep_attempt_rate")})
    assert pass_only["coverages"][PASS] == pytest.approx((1.5 + 1.2) / sum(f.weight for f in m.PASS_FACTORS), abs=1e-3)
    assert pass_only["coverages"][RUN] == 0.0
    assert 0.0 < pass_only["coverage"] < pass_only["coverages"][PASS]
    assert m.compute_explosive_play({})["coverage"] == 0.0 and m.compute_explosive_play(HIGH)["coverage"] == 1.0


def test_contributions_cover_exactly_the_inputs_used_and_expose_both_models_for_shared_keys():
    res = m.compute_explosive_play({"opp_missed_tackle_rate": 0.12, "wr_separation": 3.4})
    assert set(res["contributions"]) == set(res["inputs_used"]) == {"opp_missed_tackle_rate", "wr_separation"}
    shared = res["contributions"]["opp_missed_tackle_rate"]["outputs"]
    assert set(shared) == {PASS, RUN}
    assert shared[PASS]["weight"] == 0.7 and shared[RUN]["weight"] == 0.9
    assert set(res["contributions"]["wr_separation"]["outputs"]) == {PASS}


def test_drivers_name_the_inputs_that_moved_the_score():
    res = m.compute_explosive_play(HIGH)
    assert res["drivers"]
    text = " ".join(res["drivers"])
    assert "raises the score" in text and ("pass:" in text and "run:" in text)
    assert m.compute_explosive_play({})["drivers"] == []


def test_deterministic_and_input_mapping_untouched():
    inputs = dict(HIGH)
    before = copy.deepcopy(inputs)
    first = m.compute_explosive_play(inputs)
    second = m.compute_explosive_play(inputs)
    assert first == second
    assert inputs == before


def test_debug_log_emitted_under_the_nfl_pack_logger(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        m.compute_explosive_play(HIGH)
    records = [rec for rec in caplog.records if rec.name.startswith("nfl_pack")]
    assert records and records[0].name == "nfl_pack.explosive_play"
    message = records[0].getMessage()
    assert PASS in message and RUN in message and records[0].levelno == logging.DEBUG


def test_no_log_output_when_debug_is_off(caplog):
    with caplog.at_level(logging.WARNING, logger="nfl_pack"):
        m.compute_explosive_play(HIGH)
    assert [rec for rec in caplog.records if rec.name.startswith("nfl_pack")] == []


# ---------------------------------------------------------------------------
# the eight named inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(HIGH))
def test_every_brief_named_input_is_accepted_and_moves_at_least_one_output(key):
    res = m.compute_explosive_play({key: HIGH[key]})
    assert res["inputs_used"] == [key]
    assert res[PASS] > m.BASE_RATE_PASS or res[RUN] > m.BASE_RATE_RUN
    low = m.compute_explosive_play({key: LOW[key]})
    assert low[PASS] < m.BASE_RATE_PASS or low[RUN] < m.BASE_RATE_RUN


def test_the_brief_named_inputs_are_exactly_the_models_inputs():
    assert set(HIGH) == {f.key for f in m.FACTORS} == {f.key for f in m.PASS_FACTORS} | {f.key for f in m.RUN_FACTORS}


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def test_wide_receiver_scenarios_order_as_a_football_person_expects():
    vertical_wr_vs_leaky_d = {
        "wr_deep_route_rate": 0.28, "qb_deep_attempt_rate": 0.16, "wr_separation": 3.5,
        "opp_explosive_play_prevention": 12.0, "opp_missed_tackle_rate": 0.11,
    }
    slot_wr_vs_tight_d = {
        "wr_deep_route_rate": 0.07, "qb_deep_attempt_rate": 0.08, "wr_separation": 2.4,
        "opp_explosive_play_prevention": 88.0, "opp_missed_tackle_rate": 0.065,
    }
    deep_wr_vs_tight_d = dict(vertical_wr_vs_leaky_d, opp_explosive_play_prevention=88.0, opp_missed_tackle_rate=0.065)
    slot_wr_vs_leaky_d = dict(slot_wr_vs_tight_d, opp_explosive_play_prevention=12.0, opp_missed_tackle_rate=0.11)
    best = m.compute_explosive_play(vertical_wr_vs_leaky_d)[PASS]
    worst = m.compute_explosive_play(slot_wr_vs_tight_d)[PASS]
    mixed_a = m.compute_explosive_play(deep_wr_vs_tight_d)[PASS]
    mixed_b = m.compute_explosive_play(slot_wr_vs_leaky_d)[PASS]
    assert best > mixed_a > worst and best > mixed_b > worst
    assert best > 0.37 and worst < 0.20


def test_running_back_scenarios_order_as_expected():
    breakaway_back_good_line_bad_d = {
        "rb_breakaway_rate": 0.075, "ol_run_block_win_rate": 0.77,
        "opp_missed_tackle_rate": 0.115, "opp_explosive_play_prevention": 15.0,
    }
    plodder_bad_line_good_d = {
        "rb_breakaway_rate": 0.02, "ol_run_block_win_rate": 0.67,
        "opp_missed_tackle_rate": 0.06, "opp_explosive_play_prevention": 85.0,
    }
    breakaway_back_only = {"rb_breakaway_rate": 0.075}
    line_only = {"ol_run_block_win_rate": 0.77}
    best = m.compute_explosive_play(breakaway_back_good_line_bad_d)[RUN]
    worst = m.compute_explosive_play(plodder_bad_line_good_d)[RUN]
    assert best > m.BASE_RATE_RUN > worst
    # the runner's own trait carries more prior weight than the line (documented causal ordering)
    assert m.compute_explosive_play(breakaway_back_only)[RUN] > m.compute_explosive_play(line_only)[RUN]


def test_end_to_end_pass_and_run_lines_move_the_right_way():
    projection = {"wr_yards": 62.0, "rb_yards": 55.0, "te_yards": 40.0}
    up, _ = m.apply_explosive_play(projection, m.compute_explosive_play(HIGH))
    down, _ = m.apply_explosive_play(projection, m.compute_explosive_play(LOW))
    flat, _ = m.apply_explosive_play(projection, m.compute_explosive_play({}))
    assert up["wr_yards"] > 62.0 > down["wr_yards"]
    assert up["rb_yards"] > 55.0 > down["rb_yards"]
    assert flat == projection
    assert up["te_yards"] == down["te_yards"] == 40.0
    # a one-SD-on-everything day moves the pass line by under 2 yards (re-centred)
    one_sd, _ = m.apply_explosive_play(projection, m.compute_explosive_play(_shifted(1.0)))
    assert 1.0 < one_sd["wr_yards"] - 62.0 < 2.0
    assert 0.5 < one_sd["rb_yards"] - 55.0 < 1.2


# ---------------------------------------------------------------------------
# constants, honesty, purity
# ---------------------------------------------------------------------------
def test_factor_constants_are_well_formed_and_ordered_by_declared_importance():
    assert isinstance(m.FACTORS, tuple) and all(isinstance(f, c.Factor) for f in m.FACTORS)
    assert len({f.key for f in m.FACTORS}) == len(m.FACTORS) == 8
    for group in (m.PASS_FACTORS, m.RUN_FACTORS):
        weights = [f.weight for f in group]
        assert weights == sorted(weights, reverse=True), "factors are listed from most to least important"
        assert all(f.sd > 0 and f.weight > 0 and f.direction in (-1, 1) for f in group)
        assert all(f.source for f in group)
    for f in m.FACTORS:
        if f.kind == "rate":
            assert 0.0 < f.mean < 1.0 and 0.0 < f.sd < 1.0
    directions = {f.key: f.direction for f in m.FACTORS}
    assert directions["opp_explosive_play_prevention"] == -1  # better prevention lowers explosives
    assert all(directions[k] == +1 for k in directions if k != "opp_explosive_play_prevention")


def test_factors_with_no_free_feed_are_labelled_prior():
    by_key = {f.key: f for f in m.FACTORS}
    assert by_key["opp_coverage_bust_rate"].source.startswith("prior")
    assert by_key["ol_run_block_win_rate"].source.startswith("prior")
    for key in ("wr_separation", "wr_deep_route_rate", "qb_deep_attempt_rate", "rb_breakaway_rate", "opp_missed_tackle_rate"):
        assert not by_key[key].source.startswith("prior"), key
        assert any(ch.isdigit() for ch in by_key[key].source)  # names seasons / n


def test_basis_says_priors_not_fitted_and_names_the_events_and_n():
    res = m.compute_explosive_play(HIGH)
    basis = res["basis"].lower()
    assert "prior" in basis and "not fitted" in basis
    assert "8,383" in res["basis"] and "4,617" in res["basis"]
    assert "30+" in res["basis"] and "no free feed" in basis


def test_base_rates_gains_and_events_are_consistent():
    assert m.BASE_RATE == {PASS: m.BASE_RATE_PASS, RUN: m.BASE_RATE_RUN}
    assert m.GAIN == {PASS: m.GAIN_PASS, RUN: m.GAIN_RUN}
    assert 0.08 <= m.BASE_RATE_RUN <= 0.20 and 0.08 <= m.BASE_RATE_PASS <= 0.35
    assert m.BASE_RATE_N == {PASS: 8383, RUN: 4617}
    assert "30+" in m.EVENT_PASS and "30+" in m.EVENT_RUN
    assert m.GAIN_PASS > 0 and m.GAIN_RUN > 0


def test_module_imports_only_stdlib_and_nfl_pack_common():
    source = Path(m.__file__).read_text(encoding="utf-8")
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "numpy", "nflreadpy", "polars"}
    seen: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            seen.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module if node.module.startswith("modules") else node.module.split(".")[0])
    assert not (seen & banned), seen & banned
    allowed = {"__future__", "logging", "typing", "modules.nfl_pack_common"}
    assert seen <= allowed, seen - allowed
