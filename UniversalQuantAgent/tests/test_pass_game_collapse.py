"""Tests for ``modules.pass_game_collapse`` (offline; no network, no data files)."""
from __future__ import annotations

import ast
import copy
import logging
import math
import re
from pathlib import Path

import pytest

from modules import nfl_pack_common as c
from modules import pass_game_collapse as pgc

BASE = pgc.BASE_RATE  # 0.12

# Physical windows the model rejects outside of (kept here so a test does not lean on a private table).
_BOUNDS = {
    "ol_injuries": (0.0, 5.0),
    "opp_coverage_grade": (0.0, 100.0),
    "wr_separation": (0.0, 15.0),
    "te_route_depth": (-5.0, 40.0),
    "wr_expected_yprr_vs_coverage": (0.0, 10.0),
}
_TE_KEYS = ("te_route_depth", "te_target_share_under_pressure")
_WR_KEYS = ("wr_expected_yprr_vs_coverage", "wr_separation")
_FACTOR = {f.key: f for f in pgc.FACTORS}


def neutral(drop: tuple[str, ...] = ()) -> dict[str, float]:
    """Every input exactly at its league reference (z = 0 everywhere)."""
    return {f.key: f.mean for f in pgc.FACTORS if f.key not in drop}


def shifted(key: str, k: float) -> float:
    """The value of ``key`` that is ``k`` SD in the *risk* direction (kept inside its physical window)."""
    f = _FACTOR[key]
    value = f.mean + k * f.sd * f.direction
    low, high = (0.0, 1.0) if f.kind == "rate" else _BOUNDS[key]
    return min(max(value, low), high)


def at(k: float, drop: tuple[str, ...] = ()) -> dict[str, float]:
    return {f.key: shifted(f.key, k) for f in pgc.FACTORS if f.key not in drop}


def p_of(inputs) -> float:
    return pgc.compute_pass_game_collapse(inputs)["pass_game_collapse"]


PROJECTION = {"wr_yards": 72.0, "wr_td_prob": 0.31, "te_yards": 48.0, "te_td_prob": 0.22}


# --- the brief's formulas, literally ------------------------------------------
@pytest.mark.parametrize("p", [0.0, 0.05, 0.12, 0.30, 0.60, 1.0])
@pytest.mark.parametrize(
    "field, coefficient",
    [("wr_yards", 0.55), ("wr_td_prob", 0.40), ("te_yards", 0.45), ("te_td_prob", 0.35)],
)
def test_literal_formula_equals_the_brief(field, coefficient, p):
    out, log = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": p}, recenter=False)
    assert out[field] == pytest.approx(PROJECTION[field] * (1.0 - p * coefficient))
    entry = next(a for a in log if a["field"] == field)
    assert entry["applied"] and entry["stage"] == "pass_game_collapse" and entry["recentered"] is False


def test_brief_coefficients_are_exposed_verbatim():
    assert pgc.WR_YARDS_COEFFICIENT == 0.55
    assert pgc.WR_TD_PROB_COEFFICIENT == 0.40
    assert pgc.TE_YARDS_COEFFICIENT == 0.45
    assert pgc.TE_TD_PROB_COEFFICIENT == 0.35
    assert pgc.COEFFICIENTS == {"wr_yards": 0.55, "wr_td_prob": 0.40, "te_yards": 0.45, "te_td_prob": 0.35}


# --- recentring ----------------------------------------------------------------
def test_recenter_is_an_exact_noop_at_the_reference():
    result = {"pass_game_collapse": BASE, "references": {"pass_game_collapse": BASE}}
    out, log = pgc.apply_pass_game_collapse(PROJECTION, result)  # recenter defaults to True
    assert out == PROJECTION
    assert all(a["applied"] and a["factor"] == 1.0 and a["recentered"] is True for a in log)


def test_recenter_divides_by_the_same_formula_at_the_reference():
    p = 0.30
    out, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": p, "reference_probability": BASE})
    for field, coefficient in pgc.COEFFICIENTS.items():
        expected = PROJECTION[field] * (1 - p * coefficient) / (1 - BASE * coefficient)
        assert out[field] == pytest.approx(expected)
    # ... and it is a different number from the literal formula
    literal, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": p}, recenter=False)
    assert out["wr_yards"] > literal["wr_yards"]


def test_recenter_uses_the_reference_carried_by_the_result():
    result = {"pass_game_collapse": 0.20, "references": {"pass_game_collapse": 0.20}}
    out, _ = pgc.apply_pass_game_collapse(PROJECTION, result)
    assert out == PROJECTION  # the caller's reference, not BASE_RATE, is the neutral point


def test_below_reference_probability_lifts_the_projection():
    out, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": 0.04})
    assert out["wr_yards"] > PROJECTION["wr_yards"] and out["te_yards"] > PROJECTION["te_yards"]


def test_higher_probability_never_raises_any_field():
    previous = None
    for p in (0.0, 0.1, 0.25, 0.5, 0.9):
        out, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": p})
        if previous is not None:
            assert all(out[k] < previous[k] for k in PROJECTION), p  # strictly lower, field by field
        previous = out
    # p = 0 is below the 0.12 reference, so it *lifts* the projection and p = 0.9 cuts it hard
    best, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": 0.0})
    worst, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": 0.9})
    assert all(best[k] > PROJECTION[k] > worst[k] for k in PROJECTION)


# --- apply contract ------------------------------------------------------------
def test_apply_does_not_mutate_the_input_and_returns_a_new_dict():
    projection = copy.deepcopy(PROJECTION)
    snapshot = copy.deepcopy(projection)
    out, log = pgc.apply_pass_game_collapse(projection, {"pass_game_collapse": 0.4})
    assert projection == snapshot
    assert out is not projection and out != projection
    assert isinstance(log, list) and len(log) == 4


def test_unrelated_keys_are_copied_through_untouched():
    projection = {"wr_yards": 60.0, "qb_yards": 250.0, "rb_yards": 70.0, "player": "X", "team_points": 24.0}
    out, _ = pgc.apply_pass_game_collapse(projection, {"pass_game_collapse": 0.5})
    assert out["qb_yards"] == 250.0 and out["rb_yards"] == 70.0 and out["player"] == "X" and out["team_points"] == 24.0
    assert out["wr_yards"] < 60.0


def test_missing_field_is_recorded_applied_false_not_an_error():
    out, log = pgc.apply_pass_game_collapse({"wr_yards": 60.0}, {"pass_game_collapse": 0.4})
    by_field = {a["field"]: a for a in log}
    assert set(by_field) == {"wr_yards", "wr_td_prob", "te_yards", "te_td_prob"}
    assert by_field["wr_yards"]["applied"] is True
    for field in ("wr_td_prob", "te_yards", "te_td_prob"):
        assert by_field[field]["applied"] is False and "missing" in by_field[field]["reason"]
        assert field not in out  # a missing field is not invented


def test_both_wr_and_te_fields_are_written_when_both_are_present():
    out, log = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": 0.5})
    assert all(a["applied"] for a in log)
    assert all(out[k] < PROJECTION[k] for k in PROJECTION)


def test_wr_only_and_te_only_projections_touch_only_their_own_fields():
    wr, _ = pgc.apply_pass_game_collapse({"wr_yards": 60.0, "wr_td_prob": 0.3}, {"pass_game_collapse": 0.5})
    te, _ = pgc.apply_pass_game_collapse({"te_yards": 40.0, "te_td_prob": 0.2}, {"pass_game_collapse": 0.5})
    assert set(wr) == {"wr_yards", "wr_td_prob"} and set(te) == {"te_yards", "te_td_prob"}
    assert wr["wr_yards"] < 60.0 and te["te_yards"] < 40.0


@pytest.mark.parametrize("bad", ["n/a", None, float("nan"), "", [], True])
def test_non_numeric_fields_are_skipped_and_left_alone(bad):
    out, log = pgc.apply_pass_game_collapse({"wr_yards": bad, "te_yards": 40.0}, {"pass_game_collapse": 0.5})
    assert out["wr_yards"] is bad or (isinstance(bad, float) and math.isnan(out["wr_yards"]))
    assert next(a for a in log if a["field"] == "wr_yards")["applied"] is False
    assert out["te_yards"] < 40.0


@pytest.mark.parametrize("result", [{}, None, {"pass_game_collapse": "n/a"}, {"pass_game_collapse": float("nan")},
                                    {"pass_game_collapse": -0.2}, "oops", {"probability": None}])
def test_unusable_result_is_a_noop_with_an_audit_trail(result):
    out, log = pgc.apply_pass_game_collapse(PROJECTION, result)
    assert out == PROJECTION
    assert len(log) == 4 and not any(a["applied"] for a in log)


def test_probability_is_read_from_the_generic_alias_and_from_percent_scale():
    via_alias, _ = pgc.apply_pass_game_collapse(PROJECTION, {"probability": 0.30}, recenter=False)
    via_percent, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": 30}, recenter=False)
    direct, _ = pgc.apply_pass_game_collapse(PROJECTION, {"pass_game_collapse": 0.30}, recenter=False)
    assert via_alias == direct == via_percent


def test_true_probabilities_are_capped_at_one_or_one_hundred_but_yards_are_not():
    # p = 0 recentres to a boost of 1 / (1 - 0.12 * c): enough to push 0.99 (and 99.0) past the ceiling.
    out, _ = pgc.apply_pass_game_collapse(
        {"wr_td_prob": 0.99, "te_td_prob": 0.99, "wr_yards": 100.0}, {"pass_game_collapse": 0.0}
    )
    assert out["wr_td_prob"] == 1.0 and out["te_td_prob"] == 1.0
    assert out["wr_yards"] == pytest.approx(100.0 / (1 - 0.12 * 0.55)) and out["wr_yards"] > 100.0
    pct, _ = pgc.apply_pass_game_collapse({"wr_td_prob": 99.0, "te_td_prob": 99.0}, {"pass_game_collapse": 0.0})
    assert pct["wr_td_prob"] == 100.0 and pct["te_td_prob"] == 100.0
    # a cap is never allowed to push a value *against* the haircut
    low, _ = pgc.apply_pass_game_collapse({"wr_td_prob": 0.30}, {"pass_game_collapse": 0.9})
    assert 0.0 < low["wr_td_prob"] < 0.30


# --- compute: contract ---------------------------------------------------------
REQUIRED_KEYS = {
    "model", "pass_game_collapse", "probability", "reference_probability", "references", "z_score", "coverage",
    "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
}


def test_result_has_every_contract_key():
    result = pgc.compute_pass_game_collapse(at(1.0))
    assert REQUIRED_KEYS <= set(result)
    assert result["model"] == "pass_game_collapse"
    assert result["probability"] == result["pass_game_collapse"]
    assert result["references"] == {"pass_game_collapse": result["reference_probability"]} == {"pass_game_collapse": 0.12}
    assert isinstance(result["drivers"], list) and result["drivers"]
    assert set(result["contributions"]) == set(result["inputs_used"])
    assert sorted(result["inputs_used"] + result["inputs_defaulted"]) == sorted(f.key for f in pgc.FACTORS)


@pytest.mark.parametrize("inputs", [None, {}, neutral()])
def test_neutral_in_reference_out(inputs):
    result = pgc.compute_pass_game_collapse(inputs)
    assert result["pass_game_collapse"] == pytest.approx(BASE, abs=1e-9)
    assert result["pass_game_collapse"] == result["references"]["pass_game_collapse"] == result["reference_probability"]
    assert result["z_score"] == pytest.approx(0.0, abs=1e-9)


def test_neutral_via_the_man_zone_split_is_also_reference():
    mean = _FACTOR["wr_expected_yprr_vs_coverage"].mean
    inputs = neutral(drop=("wr_expected_yprr_vs_coverage",))
    inputs.update(wr_yprr_vs_man=mean, wr_yprr_vs_zone=mean, opp_man_coverage_rate=0.4)
    assert p_of(inputs) == pytest.approx(BASE, abs=1e-9)


def test_model_constants_match_the_documented_event():
    assert BASE == 0.12 and pgc.GAIN == 0.6
    assert pgc.EVENT_YARDS_FRACTION == 0.70 and pgc.EVENT_EPA_CEILING == 0.0
    assert "0.70" in pgc.EVENT_DEFINITION and "EPA" in pgc.EVENT_DEFINITION
    assert pgc.QB_COLLAPSE_REFERENCE == _FACTOR["qb_collapse_probability"].mean


def test_qb_term_is_on_the_qb_models_measured_scale():
    qb = _FACTOR["qb_collapse_probability"]
    assert qb.mean == 0.16 and qb.sd == 0.06  # QB model neutral output; measured game-level spread 0.063 (2023-2025)
    assert not qb.source.startswith("prior") and "2023-2025" in qb.source
    # one SD of the QB model's own spread moves the answer, a neutral QB does not
    one_sd_up = p_of({"qb_collapse_probability": 0.16 + 0.06})
    assert one_sd_up > BASE and p_of({"qb_collapse_probability": 0.16}) == pytest.approx(BASE, abs=1e-9)
    z = pgc.compute_pass_game_collapse({"qb_collapse_probability": 0.22})["z_score"]
    assert z == pytest.approx(3.0 * 1.0 / sum(f.weight for f in pgc.FACTORS), abs=1e-3)


# --- monotonicity and bounds ---------------------------------------------------
@pytest.mark.parametrize("key", [f.key for f in pgc.FACTORS])
def test_each_input_moves_the_probability_in_its_stated_direction(key):
    low = p_of({key: shifted(key, -1.0)})
    mid = p_of({key: shifted(key, 0.0)})
    high = p_of({key: shifted(key, +1.0)})
    assert mid == pytest.approx(BASE, abs=1e-9)  # an average value is exactly the reference
    assert low < mid < high
    # and the same holds on top of a non-neutral background
    background = at(0.5, drop=(key,))
    assert p_of({**background, key: shifted(key, 1.0)}) > p_of({**background, key: shifted(key, -1.0)})


@pytest.mark.parametrize("key", [f.key for f in pgc.FACTORS])
def test_probability_never_decreases_along_a_grid_of_worsening_values(key):
    grid = [shifted(key, k) for k in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)]
    values = [p_of({key: v}) for v in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    assert values[-1] > values[0]


def test_stated_direction_matches_the_brief():
    expected = {
        "qb_collapse_probability": +1, "ol_pass_block_win_rate": -1, "ol_injuries": +1, "opp_pressure_rate": +1,
        "opp_coverage_grade": +1, "opp_cb1_shadow_rate": +1, "opp_bracket_coverage_rate": +1, "wr_separation": -1,
        "wr_expected_yprr_vs_coverage": -1, "te_route_depth": +1, "te_target_share_under_pressure": -1,
    }
    assert {f.key: f.direction for f in pgc.FACTORS} == expected


def test_outputs_stay_in_unit_interval_for_extreme_inputs():
    extremes = (1e9, -1e9, 0.0, 100.0, 5.0, 1e308, -1e-9)
    for value in extremes:
        result = pgc.compute_pass_game_collapse({f.key: value for f in pgc.FACTORS})
        assert 0.0 <= result["pass_game_collapse"] <= 1.0
        assert math.isfinite(result["z_score"]) and 0.0 <= result["coverage"] <= 1.0


def test_worst_possible_inputs_stay_far_below_certainty():
    worst = {f.key: shifted(f.key, 50.0) for f in pgc.FACTORS}  # clamped into each physical window
    p = p_of(worst)
    assert BASE < p < 0.5
    best = p_of({f.key: shifted(f.key, -50.0) for f in pgc.FACTORS})
    assert 0.0 < best < BASE


def test_gain_anchor_probabilities():
    # all eleven inputs +1 SD bad -> ~0.199; +2 SD (grade pinned at 100 = +1.73 SD) -> ~0.307; -1 SD -> ~0.072
    one_sd = p_of(at(1.0))
    assert one_sd == pytest.approx(c.logistic_probability(BASE, 1.0, pgc.GAIN), abs=1e-4)  # z is exactly 1
    assert one_sd == pytest.approx(0.199, abs=1e-3)
    assert 1.5 * BASE < one_sd < 2.0 * BASE  # "a plausible bad situation": between 1.5x and 2x the base rate
    assert p_of(at(2.0)) == pytest.approx(0.307, abs=2e-3)
    assert p_of(at(-1.0)) == pytest.approx(0.072, abs=2e-3)
    assert p_of(at(2.0)) < 0.95


def test_practical_ceiling_is_far_below_certainty_and_matches_the_docstring():
    # every input at its +3 SD clip or its physical limit (grade 100, injuries 5, rates 0/1, yprr / share floor 0)
    worst = {f.key: shifted(f.key, 50.0) for f in pgc.FACTORS}
    full = p_of(worst)
    wr = p_of({k: v for k, v in worst.items() if k not in _TE_KEYS})
    te = p_of({k: v for k, v in worst.items() if k not in _WR_KEYS})
    assert full == pytest.approx(0.418, abs=2e-3)
    assert wr == pytest.approx(0.383, abs=2e-3) and te == pytest.approx(0.356, abs=2e-3)
    assert te < wr < full < 0.5
    assert pgc.GAIN * c.Z_CLIP + c.logit(BASE) < 0  # even a literal +3 SD z everywhere stays under 50%


def test_role_calls_top_out_below_the_full_coverage_value():
    full = pgc.compute_pass_game_collapse(at(1.0))
    wr = pgc.compute_pass_game_collapse(at(1.0, drop=_TE_KEYS))
    te = pgc.compute_pass_game_collapse(at(1.0, drop=_WR_KEYS))
    assert full["coverage"] == 1.0
    assert 0.9 < wr["coverage"] < 0.93 and 0.82 < te["coverage"] < 0.85
    assert BASE < te["pass_game_collapse"] < wr["pass_game_collapse"] < full["pass_game_collapse"]


# --- bad data ------------------------------------------------------------------
BAD_VALUES = [None, float("nan"), float("inf"), float("-inf"), "n/a", "", "abc", True, False, [], {}, object()]


@pytest.mark.parametrize("bad", BAD_VALUES, ids=repr)
def test_non_numeric_values_are_missing_not_errors(bad):
    result = pgc.compute_pass_game_collapse({f.key: bad for f in pgc.FACTORS})
    assert result["inputs_used"] == [] and result["coverage"] == 0.0
    assert result["pass_game_collapse"] == pytest.approx(BASE)
    assert result["inputs_rejected"] == []  # not-a-number is absent, not "out of range"


@pytest.mark.parametrize("raw", [None, 5, "x", [("qb_collapse_probability", 0.3)], object()])
def test_non_mapping_inputs_never_raise(raw):
    result = pgc.compute_pass_game_collapse(raw)
    assert 0.0 <= result["pass_game_collapse"] <= 1.0


def test_pairs_of_keys_and_values_are_accepted_like_a_mapping():
    assert p_of([("qb_collapse_probability", 0.40)]) == p_of({"qb_collapse_probability": 0.40}) > BASE


def test_percent_and_fraction_rates_agree():
    for key, fraction in (("qb_collapse_probability", 0.31), ("opp_pressure_rate", 0.33),
                          ("opp_cb1_shadow_rate", 0.4), ("te_target_share_under_pressure", 0.08)):
        by_fraction = pgc.compute_pass_game_collapse({key: fraction})
        by_percent = pgc.compute_pass_game_collapse({key: fraction * 100})
        by_text = pgc.compute_pass_game_collapse({key: f"{fraction * 100:g}%"})
        assert by_fraction["pass_game_collapse"] == by_percent["pass_game_collapse"] == by_text["pass_game_collapse"]
        assert by_fraction["contributions"][key]["value"] == pytest.approx(fraction)


@pytest.mark.parametrize(
    "key, value",
    [
        ("qb_collapse_probability", -0.1), ("qb_collapse_probability", 150), ("opp_pressure_rate", -5),
        ("ol_pass_block_win_rate", 1e9), ("ol_injuries", -1), ("ol_injuries", 9), ("opp_coverage_grade", 150),
        ("opp_coverage_grade", -5), ("wr_separation", 99.0), ("wr_separation", -1.0), ("te_route_depth", 400),
        ("wr_expected_yprr_vs_coverage", -2.0), ("wr_expected_yprr_vs_coverage", 55.0),
    ],
)
def test_out_of_domain_numbers_are_rejected_not_clamped(key, value):
    result = pgc.compute_pass_game_collapse({key: value})
    assert key not in result["inputs_used"]
    assert result["pass_game_collapse"] == pytest.approx(BASE)
    assert [r["key"] for r in result["inputs_rejected"]] == [key]
    assert result["inputs_rejected"][0]["reason"]


def test_a_rejected_primary_key_falls_through_to_a_valid_alias():
    result = pgc.compute_pass_game_collapse({"qb_collapse_probability": 150, "collapse_probability": 0.40})
    assert "qb_collapse_probability" in result["inputs_used"]
    assert result["contributions"]["qb_collapse_probability"]["value"] == pytest.approx(0.40)
    assert result["pass_game_collapse"] > BASE
    assert [r["key"] for r in result["inputs_rejected"]] == ["qb_collapse_probability"]


def test_out_of_domain_yprr_split_side_is_dropped_and_the_other_used():
    result = pgc.compute_pass_game_collapse({"wr_yprr_vs_man": -3.0, "wr_yprr_vs_zone": 0.8})
    assert result["contributions"]["wr_expected_yprr_vs_coverage"]["value"] == pytest.approx(0.8)
    assert [r["key"] for r in result["inputs_rejected"]] == ["wr_yprr_vs_man"]


# --- missing data --------------------------------------------------------------
def test_missing_data_shrinks_toward_neutral_rather_than_amplifying():
    keys = ("qb_collapse_probability", "opp_pressure_rate")
    full_inputs = at(1.5)
    partial_inputs = {k: full_inputs[k] for k in keys}
    full = pgc.compute_pass_game_collapse(full_inputs)
    partial = pgc.compute_pass_game_collapse(partial_inputs)
    assert partial["coverage"] < full["coverage"]
    assert 0 < partial["z_score"] < full["z_score"]
    assert BASE < partial["pass_game_collapse"] < full["pass_game_collapse"]
    assert partial["inputs_defaulted"] and set(partial["inputs_defaulted"]) | set(keys) == {f.key for f in pgc.FACTORS}


def test_adding_a_second_input_at_the_same_value_moves_the_result_further():
    one = p_of({"qb_collapse_probability": 0.35})
    two = p_of({"qb_collapse_probability": 0.35, "opp_coverage_grade": 90.0})
    assert two > one > BASE


# --- determinism, purity, logging ---------------------------------------------
def test_compute_is_deterministic_and_does_not_mutate_its_input():
    inputs = {**at(0.7), "wr_yprr_vs_man": 1.1, "wr_yprr_vs_zone": 1.9, "opp_man_coverage_rate": "38%"}
    snapshot = copy.deepcopy(inputs)
    first = pgc.compute_pass_game_collapse(inputs)
    second = pgc.compute_pass_game_collapse(inputs)
    assert first == second
    assert inputs == snapshot


def test_debug_logging_on_the_nfl_pack_logger(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        result = pgc.compute_pass_game_collapse(at(1.0))
        pgc.apply_pass_game_collapse(PROJECTION, result)
    names = {rec.name for rec in caplog.records}
    assert "nfl_pack" in names and "nfl_pack.pass_game_collapse" in names
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert f"{result['pass_game_collapse']:.4f}" in text
    assert "pass_game_collapse apply" in text
    assert pgc.LOG.name == "nfl_pack.pass_game_collapse"


def test_no_logging_cost_or_output_when_debug_is_off(caplog):
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        pgc.compute_pass_game_collapse(at(1.0))
    assert caplog.records == []


def test_module_imports_only_stdlib_and_the_common_helpers():
    source = Path(pgc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0] if node.level == 0 else node.module)
    assert imported <= {"__future__", "dataclasses", "logging", "typing", "modules"}, imported
    from_modules = {
        node.module + "." + a.name if node.module != "modules" else a.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("modules")
        for a in node.names
    }
    assert from_modules <= {"nfl_pack_common", "modules.nfl_pack_common.Factor"}, from_modules
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "numpy", "nflreadpy"}
    assert not (imported & banned)


# --- brief-named inputs ----------------------------------------------------------
BRIEF_INPUTS = {
    "QB collapse probability": ("qb_collapse_probability", 0.34),
    "offensive line pass block win rate": ("ol_pass_block_win_rate", 0.60),
    "offensive line injuries": ("ol_injuries", 2.0),
    "opponent pressure rate": ("opp_pressure_rate", 0.38),
    "opponent coverage grade": ("opp_coverage_grade", 92.0),
    "opponent CB1 shadow rate": ("opp_cb1_shadow_rate", 0.70),
    "opponent bracket coverage rate": ("opp_bracket_coverage_rate", 0.25),
    "WR separation ability": ("wr_separation", 2.1),
    "WR man/zone split (direct)": ("wr_expected_yprr_vs_coverage", 0.9),
    "TE route depth": ("te_route_depth", 11.0),
    "TE target share under pressure": ("te_target_share_under_pressure", 0.03),
}


@pytest.mark.parametrize("name", list(BRIEF_INPUTS))
def test_every_brief_named_input_is_accepted_and_pushes_the_right_way(name):
    key, bad_value = BRIEF_INPUTS[name]
    result = pgc.compute_pass_game_collapse({key: bad_value})
    assert result["inputs_used"] == [key]
    assert result["pass_game_collapse"] > BASE  # each value above is on the risk side of its reference
    assert result["inputs_rejected"] == []
    assert any(_FACTOR[key].label in line for line in result["drivers"])


def test_wr_man_zone_split_is_accepted_through_its_three_named_inputs():
    result = pgc.compute_pass_game_collapse(
        {"wr_yprr_vs_man": 0.7, "wr_yprr_vs_zone": 1.0, "opp_man_coverage_rate": 0.55}
    )
    assert result["inputs_used"] == ["wr_expected_yprr_vs_coverage"]
    assert result["pass_game_collapse"] > BASE
    assert "opp_man_coverage_rate 0.550" in result["inputs_derived"]["wr_expected_yprr_vs_coverage"]


@pytest.mark.parametrize(
    "alias, canonical, value",
    [
        ("collapse_probability", "qb_collapse_probability", 0.3), ("qb_collapse_prob", "qb_collapse_probability", 0.3),
        ("pass_block_win_rate", "ol_pass_block_win_rate", 0.6), ("ol_injury_count", "ol_injuries", 2),
        ("opponent_pressure_rate", "opp_pressure_rate", 0.4), ("opponent_coverage_grade", "opp_coverage_grade", 90),
        ("cb1_shadow_rate", "opp_cb1_shadow_rate", 0.5), ("bracket_coverage_rate", "opp_bracket_coverage_rate", 0.2),
        ("wr1_separation", "wr_separation", 2.2), ("expected_yprr_vs_coverage", "wr_expected_yprr_vs_coverage", 1.0),
        ("te_adot", "te_route_depth", 10.0), ("te_pressure_target_share", "te_target_share_under_pressure", 0.04),
    ],
)
def test_aliases_resolve_to_the_canonical_factor(alias, canonical, value):
    by_alias = pgc.compute_pass_game_collapse({alias: value})
    by_key = pgc.compute_pass_game_collapse({canonical: value})
    assert by_alias["inputs_used"] == [canonical]
    assert by_alias["pass_game_collapse"] == by_key["pass_game_collapse"]
    assert alias in by_alias["inputs_derived"][canonical]


def test_ol_health_is_converted_to_a_starters_out_count():
    # 0.9 health = half a starter out; picked below the +3 SD clip so a wrong conversion cannot hide there
    by_health = pgc.compute_pass_game_collapse({"ol_health": 0.9})
    by_count = pgc.compute_pass_game_collapse({"ol_injuries": 0.5})
    assert by_health["contributions"]["ol_injuries"]["value"] == pytest.approx(0.5)
    assert by_health["pass_game_collapse"] == by_count["pass_game_collapse"] > BASE
    assert by_health["inputs_derived"]["ol_injuries"] == "5 * (1 - ol_health)"
    # a healthier line is lower risk than an average one, a depleted one higher
    assert p_of({"ol_health": 1.0}) < p_of({"ol_health": 0.926}) < p_of({"ol_health": 0.8})
    # health is a rate: a percent is read as one, an out-of-range value is rejected, not converted
    assert p_of({"ol_health": 90}) == by_health["pass_game_collapse"]
    assert pgc.compute_pass_game_collapse({"ol_health": 250})["inputs_rejected"][0]["key"] == "ol_health"
    # an explicit count wins over the derived one
    both = pgc.compute_pass_game_collapse({"ol_health": 0.0, "ol_injuries": 0.0})
    assert both["contributions"]["ol_injuries"]["value"] == 0.0


# --- man / zone ------------------------------------------------------------------
def test_man_zone_split_is_resolved_against_this_defenses_coverage_mix():
    receiver = {"wr_yprr_vs_man": 0.8, "wr_yprr_vs_zone": 2.0}  # a zone beater who struggles against man
    vs_man_heavy = pgc.compute_pass_game_collapse({**receiver, "opp_man_coverage_rate": 0.70})
    vs_zone_heavy = pgc.compute_pass_game_collapse({**receiver, "opp_man_coverage_rate": 0.10})
    assert vs_man_heavy["pass_game_collapse"] > vs_zone_heavy["pass_game_collapse"]
    assert vs_man_heavy["contributions"]["wr_expected_yprr_vs_coverage"]["value"] == pytest.approx(0.7 * 0.8 + 0.3 * 2.0)
    assert vs_zone_heavy["contributions"]["wr_expected_yprr_vs_coverage"]["value"] == pytest.approx(0.1 * 0.8 + 0.9 * 2.0)
    # the mirror-image receiver prefers the man-heavy defense
    man_beater = {"wr_yprr_vs_man": 2.0, "wr_yprr_vs_zone": 0.8}
    assert (
        p_of({**man_beater, "opp_man_coverage_rate": 0.70}) < p_of({**man_beater, "opp_man_coverage_rate": 0.10})
    )


def test_missing_opponent_man_rate_uses_the_helpers_default_and_says_so():
    result = pgc.compute_pass_game_collapse({"wr_yprr_vs_man": 1.0, "wr_yprr_vs_zone": 2.0})
    assert result["contributions"]["wr_expected_yprr_vs_coverage"]["value"] == pytest.approx(
        c.blend_man_zone(1.0, 2.0, None)
    )
    assert "assumed 0.30" in result["inputs_derived"]["wr_expected_yprr_vs_coverage"]
    garbage_rate = pgc.compute_pass_game_collapse(
        {"wr_yprr_vs_man": 1.0, "wr_yprr_vs_zone": 2.0, "opp_man_coverage_rate": "n/a"}
    )
    assert garbage_rate["pass_game_collapse"] == result["pass_game_collapse"]


def test_direct_expected_yprr_wins_over_the_split():
    direct = pgc.compute_pass_game_collapse(
        {"wr_expected_yprr_vs_coverage": 2.2, "wr_yprr_vs_man": 0.2, "wr_yprr_vs_zone": 0.2}
    )
    assert direct["contributions"]["wr_expected_yprr_vs_coverage"]["value"] == pytest.approx(2.2)
    assert direct["pass_game_collapse"] < BASE
    assert "wr_expected_yprr_vs_coverage" not in direct["inputs_derived"]


def test_a_one_sided_split_is_used_as_given_and_flagged():
    zone_only = pgc.compute_pass_game_collapse({"wr_yprr_vs_zone": 0.9})
    assert zone_only["contributions"]["wr_expected_yprr_vs_coverage"]["value"] == pytest.approx(0.9)
    assert "zone only" in zone_only["inputs_derived"]["wr_expected_yprr_vs_coverage"]
    man_only = pgc.compute_pass_game_collapse({"wr_yprr_vs_man": 0.9})
    assert "man only" in man_only["inputs_derived"]["wr_expected_yprr_vs_coverage"]
    assert zone_only["pass_game_collapse"] == man_only["pass_game_collapse"] > BASE


# --- WR call vs TE call --------------------------------------------------------------
def test_wr_call_and_te_call_are_partial_by_construction():
    wr_call = pgc.compute_pass_game_collapse(at(1.0, drop=_TE_KEYS))
    te_call = pgc.compute_pass_game_collapse(at(1.0, drop=_WR_KEYS))
    assert wr_call["role_coverage"] == {"team": 1.0, "wr": 1.0, "te": 0.0}
    assert te_call["role_coverage"] == {"team": 1.0, "wr": 0.0, "te": 1.0}
    assert set(wr_call["inputs_defaulted"]) == set(_TE_KEYS) and set(te_call["inputs_defaulted"]) == set(_WR_KEYS)
    assert 0 < wr_call["coverage"] < 1 and 0 < te_call["coverage"] < 1
    # unsupplied is neutral, not an error: the call still produces a clear above-base risk
    assert wr_call["pass_game_collapse"] > BASE and te_call["pass_game_collapse"] > BASE


def test_te_only_inputs_do_not_move_a_wr_only_neutral_game_off_the_base_rate_when_neutral():
    assert p_of({"te_route_depth": _FACTOR["te_route_depth"].mean,
                 "te_target_share_under_pressure": _FACTOR["te_target_share_under_pressure"].mean}) == pytest.approx(BASE)


def test_shadow_and_bracket_have_no_free_feed_and_are_neutral_until_supplied():
    assert set(pgc.NO_FREE_FEED_INPUTS) == {"opp_cb1_shadow_rate", "opp_bracket_coverage_rate"}
    for key in pgc.NO_FREE_FEED_INPUTS:
        assert _FACTOR[key].source.startswith("prior")
    base = at(0.8, drop=pgc.NO_FREE_FEED_INPUTS)
    absent = pgc.compute_pass_game_collapse(base)
    average = pgc.compute_pass_game_collapse({**base, **{k: _FACTOR[k].mean for k in pgc.NO_FREE_FEED_INPUTS}})
    heavy = pgc.compute_pass_game_collapse({**base, "opp_cb1_shadow_rate": 0.60, "opp_bracket_coverage_rate": 0.25})
    assert absent["pass_game_collapse"] == average["pass_game_collapse"]  # supplying "average" changes nothing
    assert heavy["pass_game_collapse"] > absent["pass_game_collapse"]
    assert absent["no_free_feed_supplied"] == []
    assert sorted(heavy["no_free_feed_supplied"]) == sorted(pgc.NO_FREE_FEED_INPUTS)


# --- weights and references --------------------------------------------------------
def test_qb_collapse_is_the_largest_weight_and_the_largest_single_effect():
    weights = {f.key: f.weight for f in pgc.FACTORS}
    assert max(weights, key=weights.get) == "qb_collapse_probability"
    assert weights["qb_collapse_probability"] > max(w for k, w in weights.items() if k != "qb_collapse_probability")
    qb = p_of({"qb_collapse_probability": shifted("qb_collapse_probability", 3.0)})
    for key in weights:
        if key != "qb_collapse_probability":
            assert qb > p_of({key: shifted(key, 3.0)}), key


def test_factor_table_is_sane_and_weights_follow_the_documented_order():
    keys = [f.key for f in pgc.FACTORS]
    assert len(keys) == len(set(keys)) == 11
    weights = [f.weight for f in pgc.FACTORS]
    assert weights == sorted(weights, reverse=True)  # declared in causal-importance order
    assert all(f.sd > 0 and f.weight > 0 for f in pgc.FACTORS)
    assert [f.key for f in pgc.FACTORS][:2] == ["qb_collapse_probability", "opp_coverage_grade"]
    for f in pgc.FACTORS:
        assert f.source and (f.source.startswith(("prior", "definitional")) or re.search(r"20\d\d", f.source)), f.key
        if f.kind == "rate":
            assert 0.0 < f.mean < 1.0 and f.sd < 0.5
    grade = _FACTOR["opp_coverage_grade"]
    assert grade.mean == 50.0 and grade.sd == pytest.approx(28.9)
    assert sum(f.weight for f in pgc.FACTORS) == pytest.approx(13.3)


def test_basis_says_priors_not_fits_and_names_the_event():
    result = pgc.compute_pass_game_collapse(None)
    assert "PRIORS" in result["basis"] and "not fitted" in result["basis"]
    assert "0.70" in result["basis"] and "no free feed" in result["basis"]
    assert result["event"] == pgc.EVENT_DEFINITION and "5,278" in pgc.EVENT_DEFINITION
    assert "coverage" in result["basis"].lower()


def test_qb_reference_override_keeps_a_neutral_qb_neutral():
    # a QB model retuned to a 0.10 base rate: 0.10 is *its* neutral and must be ours too
    plain = pgc.compute_pass_game_collapse({"qb_collapse_probability": 0.10})
    aligned = pgc.compute_pass_game_collapse(
        {"qb_collapse_probability": 0.10, "qb_collapse_reference_probability": 0.10}
    )
    assert plain["pass_game_collapse"] < BASE  # judged against 0.16 it looks like a good QB
    assert aligned["pass_game_collapse"] == pytest.approx(BASE, abs=1e-9)
    assert aligned["qb_collapse_reference"] == 0.10 and plain["qb_collapse_reference"] == pgc.QB_COLLAPSE_REFERENCE
    # direction is preserved around the supplied reference
    higher = pgc.compute_pass_game_collapse(
        {"qb_collapse_probability": 0.25, "qb_collapse_reference_probability": 0.10}
    )
    assert higher["pass_game_collapse"] > BASE


@pytest.mark.parametrize("reference", [0.0, 0.9, -0.1, 150, "n/a"])
def test_unusable_qb_reference_is_ignored(reference):
    result = pgc.compute_pass_game_collapse({"qb_collapse_probability": 0.30, "qb_collapse_reference_probability": reference})
    assert result["qb_collapse_reference"] == pgc.QB_COLLAPSE_REFERENCE
    assert result["pass_game_collapse"] == p_of({"qb_collapse_probability": 0.30})


# --- scenarios ---------------------------------------------------------------------
COLLAPSE_PRONE = {
    "qb_collapse_probability": 0.33, "ol_pass_block_win_rate": 0.63, "ol_injuries": 2.0, "opp_pressure_rate": 0.37,
    "opp_coverage_grade": 88.0, "wr_separation": 2.3, "wr_yprr_vs_man": 0.9, "wr_yprr_vs_zone": 1.2,
    "opp_man_coverage_rate": 0.5, "te_route_depth": 9.5, "te_target_share_under_pressure": 0.07,
}
RESILIENT = {
    "qb_collapse_probability": 0.09, "ol_pass_block_win_rate": 0.77, "ol_injuries": 0.0, "opp_pressure_rate": 0.25,
    "opp_coverage_grade": 15.0, "wr_separation": 3.4, "wr_yprr_vs_man": 2.1, "wr_yprr_vs_zone": 2.0,
    "opp_man_coverage_rate": 0.5, "te_route_depth": 5.0, "te_target_share_under_pressure": 0.22,
}


def test_realistic_scenarios_order_as_expected():
    bad, avg, good = p_of(COLLAPSE_PRONE), p_of(neutral()), p_of(RESILIENT)
    assert avg == pytest.approx(BASE)
    assert good < avg < bad
    assert bad < 0.45 and good > 0.04  # the spread is real but restrained


def test_single_driver_scenarios():
    only_backup_qb = p_of({"qb_collapse_probability": 0.40})
    only_shutdown_secondary = p_of({"opp_coverage_grade": 100.0})
    only_pocket_collapse = p_of({"ol_pass_block_win_rate": 0.58, "opp_pressure_rate": 0.40, "ol_injuries": 3.0})
    for p in (only_backup_qb, only_shutdown_secondary, only_pocket_collapse):
        assert BASE < p < 0.30
    assert only_backup_qb > only_shutdown_secondary  # the QB term is the heavier one


def test_drivers_name_the_largest_movers_first():
    result = pgc.compute_pass_game_collapse({"qb_collapse_probability": 0.42, "wr_separation": 2.7})
    assert result["drivers"][0].startswith("QB collapse probability")
    assert "raises" in result["drivers"][0]


def test_compute_then_apply_end_to_end():
    risky = pgc.compute_pass_game_collapse(COLLAPSE_PRONE)
    calm = pgc.compute_pass_game_collapse(RESILIENT)
    average = pgc.compute_pass_game_collapse(None)
    risky_out, _ = pgc.apply_pass_game_collapse(PROJECTION, risky)
    calm_out, _ = pgc.apply_pass_game_collapse(PROJECTION, calm)
    average_out, _ = pgc.apply_pass_game_collapse(PROJECTION, average)
    assert average_out == PROJECTION  # league-average situation is a no-op
    for field in PROJECTION:
        assert risky_out[field] < PROJECTION[field] < calm_out[field]
    # the literal brief formula on the same result
    literal, _ = pgc.apply_pass_game_collapse(PROJECTION, risky, recenter=False)
    assert literal["wr_yards"] == pytest.approx(72.0 * (1 - risky["pass_game_collapse"] * 0.55))
    assert literal["te_td_prob"] == pytest.approx(0.22 * (1 - risky["pass_game_collapse"] * 0.35))
