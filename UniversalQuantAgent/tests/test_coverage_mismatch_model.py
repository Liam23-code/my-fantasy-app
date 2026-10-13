"""Offline tests for the coverage mismatch model (pure logic, no network, no data files)."""
from __future__ import annotations

import ast
import copy
import json
import logging
import math
from pathlib import Path

import pytest

from modules import coverage_mismatch_model as cm
from modules import nfl_pack_common as common

KEY = cm.OUTPUT_KEY
LEAGUE = cm.MAN_RATE_LEAGUE
GRADE_KEYS = (
    "wr_vs_cb_matchup_grade",
    "cb_man_coverage_grade",
    "cb_zone_coverage_grade",
    "te_vs_lb_coverage_grade",
    "slot_wr_vs_nickel_grade",
)

#: Every brief-named input at its league-average value: the model must return exactly 0.
AVERAGE = {
    "wr_vs_cb_matchup_grade": 50.0,
    "opp_cb1_shadow_rate": 0.30,
    "cb_man_coverage_grade": 50.0,
    "cb_zone_coverage_grade": 50.0,
    "wr_yprr_vs_man": cm.YPRR_MAN_MEAN,
    "wr_yprr_vs_zone": cm.YPRR_ZONE_MEAN,
    "opp_man_coverage_rate": LEAGUE,
    "te_vs_lb_coverage_grade": 50.0,
    "slot_wr_vs_nickel_grade": 50.0,
}

#: Which scored factor each split input feeds (the grades feed the factor of the same name).
SCORED_BY = {
    "cb_man_coverage_grade": "cb_coverage_grade_at_opp_mix",
    "cb_zone_coverage_grade": "cb_coverage_grade_at_opp_mix",
    "wr_yprr_vs_man": "wr_yprr_at_opp_mix",
    "wr_yprr_vs_zone": "wr_yprr_at_opp_mix",
}

#: A clearly favourable receiver situation on every dimension.
FAVOURABLE = {
    "wr_vs_cb_matchup_grade": 85.0,
    "opp_cb1_shadow_rate": 0.70,
    "cb_man_coverage_grade": 20.0,
    "cb_zone_coverage_grade": 30.0,
    "wr_yprr_vs_man": 2.50,
    "wr_yprr_vs_zone": 2.00,
    "opp_man_coverage_rate": 0.55,
    "te_vs_lb_coverage_grade": 75.0,
    "slot_wr_vs_nickel_grade": 80.0,
}


def mirror(inputs: dict) -> dict:
    """The mirror-image situation: every grade -> 100 - g, every yprr reflected about the league mean."""
    out = dict(inputs)
    for key in GRADE_KEYS:
        if key in out:
            out[key] = 100.0 - out[key]
    if "wr_yprr_vs_man" in out:
        out["wr_yprr_vs_man"] = 2 * cm.YPRR_MAN_MEAN - out["wr_yprr_vs_man"]
    if "wr_yprr_vs_zone" in out:
        out["wr_yprr_vs_zone"] = 2 * cm.YPRR_ZONE_MEAN - out["wr_yprr_vs_zone"]
    return out


def factor(inputs=None, role="wr") -> float:
    return cm.compute_coverage_mismatch(inputs, role=role)[KEY]


# ---------------------------------------------------------------------------
# Constants and structure
# ---------------------------------------------------------------------------
def test_brief_coefficients_and_neutral_reference_are_exposed():
    assert cm.WR_YARDS_COEFFICIENT == 0.25
    assert cm.TE_YARDS_COEFFICIENT == 0.20
    assert cm.BASE_RATE == 0.0  # a signed factor's neutral value, not a probability
    assert 0.0 < cm.GAIN < 1.0
    assert cm.LOG.name == "nfl_pack.coverage_mismatch"
    assert set(cm.ROLES) == {"wr", "slot", "te"}


def test_factors_are_priors_ordered_by_importance_and_wr_role_is_the_public_tuple():
    assert isinstance(cm.FACTORS, tuple) and all(isinstance(f, common.Factor) for f in cm.FACTORS)
    weights = [f.weight for f in cm.FACTORS]
    assert weights == sorted(weights, reverse=True) and weights[0] == 3.0
    assert cm.ROLE_FACTORS["wr"] is cm.FACTORS
    assert cm.FACTORS[0].key == "wr_vs_cb_matchup_grade"
    # each role's primary matchup grade is the heaviest factor and comes first
    assert cm.ROLE_FACTORS["slot"][0].key == "slot_wr_vs_nickel_grade" and cm.ROLE_FACTORS["slot"][0].weight == 3.0
    assert cm.ROLE_FACTORS["te"][0].key == "te_vs_lb_coverage_grade" and cm.ROLE_FACTORS["te"][0].weight == 3.0
    # the CB-at-mix factor is a "higher CB grade hurts the receiver" factor
    cb = next(f for f in cm.FACTORS if f.key == "cb_coverage_grade_at_opp_mix")
    assert cb.direction == -1
    assert all(f.source for f in cm.FACTORS)


def test_league_references_are_the_measured_values_documented_in_the_module():
    # Pinned on purpose: these were measured on nflverse participation + pbp 2023-2024 REG (man rate: n = 38,781
    # pass plays, 64 team-seasons; WR yprr: n = 204 WR-seasons with >= 100 routes per side). Changing one means
    # re-deriving it, not editing a constant, so the test should be updated together with the docstring.
    assert cm.MAN_RATE_LEAGUE == pytest.approx(0.448)
    assert cm.MAN_RATE_TEAM_SD == pytest.approx(0.094)
    assert (cm.YPRR_MAN_MEAN, cm.YPRR_MAN_SD) == (1.668, 0.829)
    assert (cm.YPRR_ZONE_MEAN, cm.YPRR_ZONE_SD) == (1.609, 0.660)
    assert cm.WR_MIX_Z_SD == pytest.approx(0.89)
    assert (cm.NEUTRAL_GRADE, cm.GRADE_SD) == (50.0, 28.9)  # a uniform 0-100 percentile: sd = 100 / sqrt(12)
    assert cm.GRADE_SD == pytest.approx(100 / math.sqrt(12), abs=0.05)
    assert cm.GAIN == 0.6
    # and the factors the model scores carry exactly those references
    by_key = {f.key: f for f in cm.FACTORS}
    for key in ("wr_vs_cb_matchup_grade", "cb_coverage_grade_at_opp_mix", "slot_wr_vs_nickel_grade", "te_vs_lb_coverage_grade"):
        assert (by_key[key].mean, by_key[key].sd) == (50.0, 28.9), key
        assert by_key[key].source.startswith("prior"), key  # no free feed: a prior, and it says so
    assert (by_key["wr_yprr_at_opp_mix"].mean, by_key["wr_yprr_at_opp_mix"].sd) == (0.0, 0.89)
    assert "nflverse" in by_key["wr_yprr_at_opp_mix"].source


def test_te_role_omits_the_wr_and_cb_split_factors():
    keys = {f.key for f in cm.ROLE_FACTORS["te"]}
    assert keys == {"te_vs_lb_coverage_grade", "wr_vs_cb_matchup_grade", "slot_wr_vs_nickel_grade"}


def test_feed_status_is_honest_for_every_canonical_input():
    assert set(cm.INPUT_FEEDS) == set(cm.CANONICAL_INPUTS)
    no_feed = {k for k, v in cm.INPUT_FEEDS.items() if v.startswith("no free feed")}
    assert no_feed == {
        "wr_vs_cb_matchup_grade", "opp_cb1_shadow_rate", "cb_man_coverage_grade", "cb_zone_coverage_grade",
        "te_vs_lb_coverage_grade", "slot_wr_vs_nickel_grade",
    }
    assert cm.INPUT_FEEDS["opp_man_coverage_rate"].startswith("real")
    assert cm.INPUT_FEEDS["wr_yprr_vs_man"].startswith("proxy")


def test_module_is_pure_stdlib_plus_nfl_pack_common():
    source = (Path(cm.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
            if node.module == "modules":
                assert {a.name for a in node.names} == {"nfl_pack_common"}
            elif node.module and node.module.startswith("modules."):
                assert node.module == "modules.nfl_pack_common"
    assert imported <= {"__future__", "logging", "dataclasses", "typing", "modules"}
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "nflreadpy", "numpy"}
    assert not (imported & banned)


# ---------------------------------------------------------------------------
# apply_coverage_mismatch: the brief's literal formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f", [-1.0, -0.6, -0.1, 0.0, 0.35, 0.8, 1.0])
def test_apply_is_the_briefs_literal_formula(f):
    projection = {"wr_yards": 70.0, "te_yards": 48.0}
    new, log = cm.apply_coverage_mismatch(projection, {KEY: f}, recenter=False)
    assert new["wr_yards"] == pytest.approx(70.0 * (1 + f * 0.25))
    assert new["te_yards"] == pytest.approx(48.0 * (1 + f * 0.20))
    assert [e["field"] for e in log] == ["wr_yards", "te_yards"] and all(e["applied"] for e in log)


@pytest.mark.parametrize("f", [-0.9, -0.2, 0.0, 0.45, 1.0])
def test_recenter_has_no_effect_on_a_signed_factor(f):
    projection = {"wr_yards": 66.0, "te_yards": 41.0, "qb_yards": 250.0}
    on, log_on = cm.apply_coverage_mismatch(projection, {KEY: f}, recenter=True)
    off, log_off = cm.apply_coverage_mismatch(projection, {KEY: f}, recenter=False)
    assert on == off
    assert [e["factor"] for e in log_on] == [e["factor"] for e in log_off]
    # a tampered "references" entry must not re-centre it either: the neutral value is pinned to 0
    tampered, _ = cm.apply_coverage_mismatch(projection, {KEY: f, "references": {KEY: 0.4}}, recenter=True)
    assert tampered == on


def test_apply_end_to_end_matches_the_formula_on_the_computed_factor():
    result = cm.compute_coverage_mismatch(FAVOURABLE)
    f = result[KEY]
    assert f > 0.3
    new, _ = cm.apply_coverage_mismatch({"wr_yards": 80.0, "te_yards": 50.0}, result)
    assert new["wr_yards"] == pytest.approx(80.0 * (1 + f * 0.25))
    assert new["te_yards"] == pytest.approx(50.0 * (1 + f * 0.20))
    assert result["implied_multipliers"]["wr_yards"] == pytest.approx(1 + f * 0.25, abs=1e-4)
    assert result["implied_multipliers"]["te_yards"] == pytest.approx(1 + f * 0.20, abs=1e-4)


def test_apply_never_mutates_the_input_and_returns_a_new_dict():
    projection = {"wr_yards": 70.0, "te_yards": 48.0, "qb_yards": 250.0, "player": "X"}
    snapshot = copy.deepcopy(projection)
    new, _ = cm.apply_coverage_mismatch(projection, {KEY: 0.5})
    assert projection == snapshot
    assert new is not projection and new != projection


def test_apply_records_a_missing_field_as_not_applied_and_copies_the_rest():
    projection = {"wr_yards": 70.0, "qb_yards": 250.0, "label": "keep"}
    new, log = cm.apply_coverage_mismatch(projection, {KEY: 0.5})
    by_field = {e["field"]: e for e in log}
    assert by_field["wr_yards"]["applied"] is True
    missing = by_field["te_yards"]
    assert {k: missing[k] for k in ("stage", "field", "applied", "reason")} == {
        "stage": "coverage_mismatch", "field": "te_yards", "applied": False, "reason": "field missing or not numeric",
    }
    assert missing["signal"] == KEY  # audit extra: which result key was being applied
    assert "before" not in missing and "after" not in missing  # nothing was computed for a missing field
    assert "te_yards" not in new  # a missing field is not invented
    assert new["qb_yards"] == 250.0 and new["label"] == "keep"


@pytest.mark.parametrize("bad", [None, "n/a", True, float("nan"), [], ""])
def test_apply_skips_a_non_numeric_field(bad):
    new, log = cm.apply_coverage_mismatch({"wr_yards": bad, "te_yards": 40.0}, {KEY: 0.5})
    assert new["wr_yards"] is bad or (isinstance(bad, float) and math.isnan(new["wr_yards"]))
    assert log[0]["field"] == "wr_yards" and log[0]["applied"] is False
    assert log[1]["applied"] is True and new["te_yards"] == pytest.approx(40.0 * 1.1)


@pytest.mark.parametrize("result", [None, {}, {KEY: "n/a"}, {KEY: None}, {KEY: float("nan")}, "oops"])
def test_apply_with_an_unusable_result_is_a_recorded_noop(result):
    projection = {"wr_yards": 70.0, "te_yards": 48.0}
    new, log = cm.apply_coverage_mismatch(projection, result)
    assert new == projection
    assert [e["applied"] for e in log] == [False, False]
    assert all("coverage_mismatch_factor" in e["reason"] for e in log)


def test_apply_clamps_a_hand_built_out_of_range_factor():
    new, _ = cm.apply_coverage_mismatch({"wr_yards": 100.0, "te_yards": 100.0}, {KEY: 5.0})
    assert new["wr_yards"] == pytest.approx(125.0) and new["te_yards"] == pytest.approx(120.0)
    low, _ = cm.apply_coverage_mismatch({"wr_yards": 100.0, "te_yards": 100.0}, {KEY: -9.0})
    assert low["wr_yards"] == pytest.approx(75.0) and low["te_yards"] == pytest.approx(80.0)


def test_apply_leaves_every_other_pack_field_untouched_even_probabilities_and_percentages():
    projection = {
        "wr_yards": 70.0, "te_yards": 48.0, "wr_td_prob": 0.42, "te_td_prob": 0.30, "qb_completion_pct": 66.0,
        "qb_yards": 250.0, "rb_yards": 60.0, "team_points": 24.0,
    }
    new, log = cm.apply_coverage_mismatch(projection, {KEY: 0.9})
    for field in ("wr_td_prob", "te_td_prob", "qb_completion_pct", "qb_yards", "rb_yards", "team_points"):
        assert new[field] == projection[field], field
    assert {e["field"] for e in log} == {"wr_yards", "te_yards"}
    assert new["wr_yards"] > 70.0 and new["te_yards"] > 48.0


def test_apply_touches_only_pack_projection_fields_and_audits_the_signal():
    _, log = cm.apply_coverage_mismatch({"wr_yards": 1.0, "te_yards": 1.0}, {KEY: 0.2})
    assert {e["field"] for e in log} <= set(common.PROJECTION_FIELDS)
    assert all(e["stage"] == "coverage_mismatch" and e["signal"] == KEY for e in log)
    assert {e["coefficient"] for e in log} == {0.25, 0.20}


# ---------------------------------------------------------------------------
# compute_coverage_mismatch: contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", cm.ROLES)
def test_neutral_in_reference_out(role):
    reference = cm.compute_coverage_mismatch({}, role=role)
    assert reference[KEY] == 0.0 == reference["references"][KEY] == reference["reference_probability"]
    assert factor(None, role) == 0.0
    assert factor(AVERAGE, role) == 0.0  # all-average inputs
    assert reference["inputs_used"] == [] and reference["coverage"] == 0.0 and reference["z_score"] == 0.0


@pytest.mark.parametrize("role", cm.ROLES)
def test_result_carries_every_contract_key(role):
    result = cm.compute_coverage_mismatch(FAVOURABLE, role=role)
    for key in (
        "model", KEY, "probability", "reference_probability", "references", "z_score", "coverage", "inputs_used",
        "inputs_defaulted", "contributions", "drivers", "basis",
    ):
        assert key in result, key
    assert result["model"] == "coverage_mismatch" and result["role"] == role
    assert result["probability"] == result[KEY]  # alias only
    assert result["references"] == {KEY: 0.0}
    assert isinstance(result["drivers"], list) and result["drivers"]
    assert "prior" in result["basis"].lower() and "not fitted" in result["basis"].lower()
    json.dumps(result)  # the audit dict must be JSON-serialisable


@pytest.mark.parametrize("role", cm.ROLES)
def test_output_is_bounded_even_for_absurd_but_valid_inputs(role):
    extreme_good = {
        "wr_vs_cb_matchup_grade": 100, "opp_cb1_shadow_rate": 1.0, "cb_man_coverage_grade": 0, "cb_zone_coverage_grade": 0,
        "wr_yprr_vs_man": 10.0, "wr_yprr_vs_zone": 10.0, "opp_man_coverage_rate": 1.0,
        "te_vs_lb_coverage_grade": 100, "slot_wr_vs_nickel_grade": 100,
    }
    # The worst valid situation, built directly (mirror() reflects yprr about the league mean, so it cannot be
    # used to turn yprr 10.0 into yprr 0.0): grades flipped, a 0.0 receiver, a full shadow, an all-man defense.
    extreme_bad = {
        **mirror({k: v for k, v in extreme_good.items() if k in GRADE_KEYS}),
        "opp_cb1_shadow_rate": 1.0, "wr_yprr_vs_man": 0.0, "wr_yprr_vs_zone": 0.0, "opp_man_coverage_rate": 1.0,
    }
    assert extreme_bad["wr_vs_cb_matchup_grade"] == 0 and extreme_bad["cb_man_coverage_grade"] == 100
    high = factor(extreme_good, role)
    low = factor(extreme_bad, role)
    assert 0.75 < high <= 0.95  # approaches but never reaches the bound
    assert -0.95 <= low < -0.75
    assert -1.0 <= low < 0 < high <= 1.0


@pytest.mark.parametrize("role", cm.ROLES)
def test_antisymmetry_mirrored_inputs_flip_the_sign(role):
    for inputs in (FAVOURABLE, {"wr_vs_cb_matchup_grade": 90, "opp_cb1_shadow_rate": 0.4}, {"cb_man_coverage_grade": 15},
                   {"wr_yprr_vs_man": 2.4, "wr_yprr_vs_zone": 1.9, "opp_man_coverage_rate": 0.6}):
        up, down = factor(inputs, role), factor(mirror(inputs), role)
        assert down == pytest.approx(-up, abs=1e-9)
    assert factor(FAVOURABLE, role) > 0 > factor(mirror(FAVOURABLE), role)


def test_determinism_and_no_shared_state_between_calls():
    first = cm.compute_coverage_mismatch(FAVOURABLE, role="wr")
    second = cm.compute_coverage_mismatch(copy.deepcopy(FAVOURABLE), role="wr")
    assert first == second
    first["notes"].append("mutated")
    first["contributions"].clear()
    assert cm.compute_coverage_mismatch(FAVOURABLE, role="wr") == second


def test_inputs_mapping_is_not_mutated():
    inputs = copy.deepcopy(FAVOURABLE)
    inputs["junk"] = object.__name__
    before = copy.deepcopy(inputs)
    cm.compute_coverage_mismatch(inputs, role="slot")
    assert inputs == before


# ---------------------------------------------------------------------------
# compute: roles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_role", ["rb", "qb", "", "wr_te", None, 5, ["wr"]])
def test_unknown_role_is_a_value_error(bad_role):
    with pytest.raises(ValueError):
        cm.compute_coverage_mismatch({}, role=bad_role)


def test_role_is_case_and_space_insensitive_and_normalised_in_the_result():
    assert cm.compute_coverage_mismatch(FAVOURABLE, role=" TE ")["role"] == "te"
    assert factor(FAVOURABLE, "WR") == factor(FAVOURABLE, "wr")


def test_role_picks_the_primary_matchup_grade():
    only_wr = {"wr_vs_cb_matchup_grade": 85}
    only_slot = {"slot_wr_vs_nickel_grade": 85}
    only_te = {"te_vs_lb_coverage_grade": 85}
    assert factor(only_wr, "wr") > factor(only_wr, "slot") > 0
    assert factor(only_wr, "wr") > factor(only_wr, "te") > 0
    assert factor(only_slot, "slot") > factor(only_slot, "wr") > 0
    assert factor(only_te, "te") > factor(only_te, "wr") > 0
    # the primary grade is worth six times an off-role grade (3.0 vs 0.5) before the squash
    z_primary = cm.compute_coverage_mismatch(only_te, role="te")["contributions"]["te_vs_lb_coverage_grade"]["effect"]
    z_off = cm.compute_coverage_mismatch(only_te, role="wr")["contributions"]["te_vs_lb_coverage_grade"]["effect"]
    assert z_primary > z_off > 0


def test_te_role_ignores_wr_and_cb_split_inputs_and_says_so():
    split_inputs = {
        "cb_man_coverage_grade": 10, "cb_zone_coverage_grade": 10,
        "wr_yprr_vs_man": 3.0, "wr_yprr_vs_zone": 3.0, "opp_man_coverage_rate": 0.6,
    }
    with_splits = cm.compute_coverage_mismatch({"te_vs_lb_coverage_grade": 70, **split_inputs}, role="te")
    without = cm.compute_coverage_mismatch({"te_vs_lb_coverage_grade": 70}, role="te")
    assert with_splits[KEY] == without[KEY]
    assert set(with_splits["inputs_ignored"]) == set(split_inputs)
    assert any("not scored for role 'te'" in n for n in with_splits["notes"])
    # ...whereas the WR role does use them
    assert factor({"te_vs_lb_coverage_grade": 70, **split_inputs}, "wr") > factor({"te_vs_lb_coverage_grade": 70}, "wr")


def test_inputs_defaulted_lists_only_role_relevant_keys():
    te = cm.compute_coverage_mismatch({}, role="te")
    assert set(te["inputs_defaulted"]) == {
        "te_vs_lb_coverage_grade", "wr_vs_cb_matchup_grade", "opp_cb1_shadow_rate", "slot_wr_vs_nickel_grade",
    }
    wr = cm.compute_coverage_mismatch({}, role="wr")
    assert wr["inputs_defaulted"] == list(cm.CANONICAL_INPUTS)


# ---------------------------------------------------------------------------
# compute: every brief-named input is accepted and moves the score the stated way
# ---------------------------------------------------------------------------
def test_every_brief_named_input_is_accepted_together_and_alone():
    full = cm.compute_coverage_mismatch(FAVOURABLE, role="wr")
    assert full["inputs_used"] == list(cm.CANONICAL_INPUTS)
    assert full["inputs_defaulted"] == [] and full["inputs_ignored"] == [] and full["coverage"] == 1.0
    for key, value in FAVOURABLE.items():
        alone = cm.compute_coverage_mismatch({key: value}, role="wr")
        assert key in alone["inputs_used"] or key in alone["inputs_ignored"], key  # never silently dropped
        others = [k for k in cm.CANONICAL_INPUTS if k != key]
        assert not any(k in alone["inputs_used"] for k in others), key


def test_every_grade_alone_moves_the_factor_in_its_stated_direction():
    for role, key in (("wr", "wr_vs_cb_matchup_grade"), ("slot", "slot_wr_vs_nickel_grade"), ("te", "te_vs_lb_coverage_grade")):
        assert factor({key: 80}, role) > 0 > factor({key: 20}, role)
    # a better CB (higher grade) is worse for the receiver
    assert factor({"cb_man_coverage_grade": 15}, "wr") > 0 > factor({"cb_man_coverage_grade": 90}, "wr")
    assert factor({"cb_zone_coverage_grade": 15}, "wr") > 0 > factor({"cb_zone_coverage_grade": 90}, "wr")
    # a receiver who produces more is better for him
    assert factor({"wr_yprr_vs_man": 2.6}, "wr") > 0 > factor({"wr_yprr_vs_man": 0.8}, "wr")
    assert factor({"wr_yprr_vs_zone": 2.6}, "wr") > 0 > factor({"wr_yprr_vs_zone": 0.8}, "wr")


@pytest.mark.parametrize("role", cm.ROLES)
@pytest.mark.parametrize(
    "key, ladder, sign",
    [
        ("wr_vs_cb_matchup_grade", [0, 15, 30, 50, 65, 85, 100], +1),
        ("slot_wr_vs_nickel_grade", [0, 15, 30, 50, 65, 85, 100], +1),
        ("te_vs_lb_coverage_grade", [0, 15, 30, 50, 65, 85, 100], +1),
        ("cb_man_coverage_grade", [0, 15, 30, 50, 65, 85, 100], -1),
        ("cb_zone_coverage_grade", [0, 15, 30, 50, 65, 85, 100], -1),
        ("wr_yprr_vs_man", [0.0, 0.6, 1.2, 1.67, 2.2, 3.0, 4.5], +1),
        ("wr_yprr_vs_zone", [0.0, 0.6, 1.2, 1.61, 2.2, 3.0, 4.5], +1),
    ],
)
def test_single_input_monotonicity_in_the_stated_direction(role, key, ladder, sign):
    values = [factor({**AVERAGE, key: x}, role) for x in ladder]
    steps = [(b - a) * sign for a, b in zip(values, values[1:])]
    assert all(s >= 0 for s in steps), (role, key, values)
    # an input moves the factor if and only if the factor it feeds is scored in this role
    scored_in_role = SCORED_BY.get(key, key) in {f.key for f in cm.ROLE_FACTORS[role]}
    assert (values[-1] != values[0]) == scored_in_role, (role, key, values)


@pytest.mark.parametrize("role", ["wr", "slot"])
def test_shadow_rate_amplifies_whichever_way_the_grade_points(role):
    good = [factor({"wr_vs_cb_matchup_grade": 80, "opp_cb1_shadow_rate": s}, role) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    bad = [factor({"wr_vs_cb_matchup_grade": 20, "opp_cb1_shadow_rate": s}, role) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert good == sorted(good) and good[0] < good[-1]
    assert bad == sorted(bad, reverse=True) and bad[0] > bad[-1]


def test_opp_man_rate_pushes_toward_the_mix_where_the_receiver_has_the_edge():
    # A WR who beats man vs a CB who is weak in man (but strong in zone): more man = better for the WR.
    favours_man = {"wr_yprr_vs_man": 2.6, "wr_yprr_vs_zone": 1.2, "cb_man_coverage_grade": 20, "cb_zone_coverage_grade": 80}
    rates = (0.0, 0.2, 0.45, 0.65, 1.0)
    up = [factor({**favours_man, "opp_man_coverage_rate": r}) for r in rates]
    assert up == sorted(up) and up[0] < 0 < up[-1]
    # the mirror-image receiver / corner pair wants a zone-heavy defense
    favours_zone = mirror(favours_man)
    favours_zone["wr_yprr_vs_man"], favours_zone["wr_yprr_vs_zone"] = 1.2, 2.6
    down = [factor({**favours_zone, "opp_man_coverage_rate": r}) for r in rates]
    assert down == sorted(down, reverse=True)


# ---------------------------------------------------------------------------
# compute: the documented formulas
# ---------------------------------------------------------------------------
def test_shadow_weighting_formula_is_50_plus_deviation_times_half_plus_shadow():
    for grade, shadow, expected in ((80, 0.8, 50 + 30 * 1.3), (80, 0.0, 65.0), (80, 1.0, 95.0), (20, 1.0, 5.0)):
        result = cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": grade, "opp_cb1_shadow_rate": shadow})
        assert result["contributions"]["wr_vs_cb_matchup_grade"]["value"] == pytest.approx(expected)
        assert result["shadow_multiplier"] == pytest.approx(0.5 + shadow)


def test_missing_shadow_rate_means_the_grade_stands_unamplified():
    plain = cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": 80})
    pivot = cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": 80, "opp_cb1_shadow_rate": cm.DEFAULT_SHADOW_RATE})
    assert plain[KEY] == pivot[KEY] and plain["z_score"] == pivot["z_score"]
    assert plain["contributions"]["wr_vs_cb_matchup_grade"]["value"] == 80.0
    assert plain["shadow_multiplier"] == 1.0


def test_shadow_rate_alone_has_nothing_to_amplify_and_is_reported():
    result = cm.compute_coverage_mismatch({"opp_cb1_shadow_rate": 0.9})
    assert result[KEY] == 0.0
    assert "opp_cb1_shadow_rate" in result["inputs_ignored"] and "opp_cb1_shadow_rate" not in result["inputs_used"]
    assert any("opp_cb1_shadow_rate" in n for n in result["notes"])


def test_wr_split_is_standardised_per_side_and_blended_at_the_opponent_man_rate():
    man, zone, rate = 2.6, 1.2, 0.65
    z_man = (man - cm.YPRR_MAN_MEAN) / cm.YPRR_MAN_SD
    z_zone = (zone - cm.YPRR_ZONE_MEAN) / cm.YPRR_ZONE_SD
    blended = rate * z_man + (1 - rate) * z_zone
    total = sum(f.weight for f in cm.FACTORS)
    weight = next(f.weight for f in cm.FACTORS if f.key == "wr_yprr_at_opp_mix")
    expected = math.tanh(cm.GAIN * weight * (blended / cm.WR_MIX_Z_SD) / total)
    got = cm.compute_coverage_mismatch({"wr_yprr_vs_man": man, "wr_yprr_vs_zone": zone, "opp_man_coverage_rate": rate})
    assert got[KEY] == pytest.approx(expected, abs=1e-4)
    assert got["inputs_used"] == ["wr_yprr_vs_man", "wr_yprr_vs_zone", "opp_man_coverage_rate"]


def test_an_average_wr_is_neutral_against_any_man_rate():
    for rate in (0.0, 0.15, LEAGUE, 0.7, 1.0):
        assert factor({"wr_yprr_vs_man": cm.YPRR_MAN_MEAN, "wr_yprr_vs_zone": cm.YPRR_ZONE_MEAN, "opp_man_coverage_rate": rate}) == 0.0
        assert factor({"cb_man_coverage_grade": 50, "cb_zone_coverage_grade": 50, "opp_man_coverage_rate": rate}) == 0.0


def test_a_lone_split_side_is_used_as_given_and_the_rate_is_not_credited():
    result = cm.compute_coverage_mismatch({"wr_yprr_vs_man": 2.6, "opp_man_coverage_rate": 0.6})
    assert result[KEY] > 0
    assert result["inputs_used"] == ["wr_yprr_vs_man"]
    assert result["inputs_ignored"] == ["opp_man_coverage_rate"]  # a blend weight with nothing to blend


def test_man_rate_default_is_the_measured_league_rate_not_the_helpers_pre_2023_30_percent():
    default = cm.compute_coverage_mismatch({"wr_yprr_vs_man": 2.6, "wr_yprr_vs_zone": 1.2})
    explicit = cm.compute_coverage_mismatch({"wr_yprr_vs_man": 2.6, "wr_yprr_vs_zone": 1.2, "opp_man_coverage_rate": LEAGUE})
    assert default["man_rate_used"] == LEAGUE and default[KEY] == explicit[KEY]
    assert cm.MAN_RATE_LEAGUE != 0.30


def _at_sd(sd: float) -> dict:
    """Every input ``sd`` league SDs in the receiver's favour (grades use the uniform-percentile SD)."""
    return {
        "wr_vs_cb_matchup_grade": 50 + sd * cm.GRADE_SD,  # shadow missing -> multiplier 1.0
        "cb_man_coverage_grade": 50 - sd * cm.GRADE_SD,
        "cb_zone_coverage_grade": 50 - sd * cm.GRADE_SD,
        "wr_yprr_vs_man": cm.YPRR_MAN_MEAN + sd * cm.WR_MIX_Z_SD * cm.YPRR_MAN_SD,
        "wr_yprr_vs_zone": cm.YPRR_ZONE_MEAN + sd * cm.WR_MIX_Z_SD * cm.YPRR_ZONE_SD,
        "slot_wr_vs_nickel_grade": 50 + sd * cm.GRADE_SD,
        "te_vs_lb_coverage_grade": 50 + sd * cm.GRADE_SD,
    }


def test_all_factors_at_one_sd_give_tanh_of_the_gain_and_the_sign_flips_at_minus_one_sd():
    for role in cm.ROLES:
        assert factor(_at_sd(1.0), role) == pytest.approx(math.tanh(cm.GAIN * 1.0), abs=2e-3)
        assert factor(_at_sd(-1.0), role) == pytest.approx(-math.tanh(cm.GAIN), abs=2e-3)
        # and it is a modest, documented amount of yardage: +13.4% WR / +10.7% TE
        assert 1 + factor(_at_sd(1.0), role) * cm.WR_YARDS_COEFFICIENT == pytest.approx(1.134, abs=1e-3)
        assert 1 + factor(_at_sd(1.0), role) * cm.TE_YARDS_COEFFICIENT == pytest.approx(1.107, abs=1e-3)


def test_two_sd_on_every_factor_is_tanh_of_twice_the_gain_and_well_below_95_percent():
    # A 0-100 percentile grade cannot be +2 SD (the ceiling is +1.73 SD) and the module rejects a grade of
    # 107.8 as a data error, so +2 SD is checked at the scoring layer the module's own constants define.
    assert _at_sd(2.0)["wr_vs_cb_matchup_grade"] > 100  # i.e. unreachable as a raw grade...
    assert cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": _at_sd(2.0)["wr_vs_cb_matchup_grade"]})["inputs_used"] == []
    for role in cm.ROLES:
        factors = cm.ROLE_FACTORS[role]
        at_two = {f.key: f.mean + f.direction * 2.0 * f.sd for f in factors}
        z = common.score_factors(at_two, factors)["z"]
        assert z == pytest.approx(2.0)
        assert common.signed_unit(cm.GAIN * z, 1.0) == pytest.approx(math.tanh(cm.GAIN * 2.0))
        assert math.tanh(cm.GAIN * 2.0) < 0.95
    # ...but reachable through the shadow multiplier: grade 100 with shadow 0.656 is an effective +2.0 SD
    shadow = 2.0 * cm.GRADE_SD / 50.0 - 0.5
    got = cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": 100, "opp_cb1_shadow_rate": shadow})
    assert got["contributions"]["wr_vs_cb_matchup_grade"]["z"] == pytest.approx(2.0, abs=1e-2)


def test_the_most_favourable_valid_grades_stay_inside_the_documented_band():
    best = {
        "wr_vs_cb_matchup_grade": 100, "cb_man_coverage_grade": 0, "cb_zone_coverage_grade": 0,
        "slot_wr_vs_nickel_grade": 100, "te_vs_lb_coverage_grade": 100,
    }
    assert factor(best, "wr") == pytest.approx(0.706, abs=2e-3)
    assert factor(best, "slot") == pytest.approx(0.706, abs=2e-3)
    assert factor(best, "te") == pytest.approx(0.777, abs=2e-3)
    assert factor(mirror(best), "wr") == pytest.approx(-0.706, abs=2e-3)


# ---------------------------------------------------------------------------
# compute: data hygiene
# ---------------------------------------------------------------------------
BAD_VALUES = [
    None, float("nan"), float("inf"), -float("inf"), "n/a", "", "abc", True, False, [], {}, object(),
    -5, -0.01, 1e12, 1e308,
]


@pytest.mark.parametrize("key", cm.CANONICAL_INPUTS)
@pytest.mark.parametrize("bad", BAD_VALUES, ids=repr)
def test_bad_values_are_treated_as_missing_never_raised(key, bad):
    base = {"wr_vs_cb_matchup_grade": 70.0, "cb_man_coverage_grade": 30.0, "wr_yprr_vs_man": 2.2}
    without = {k: v for k, v in base.items() if k != key}
    got = cm.compute_coverage_mismatch({**base, key: bad})
    expected = cm.compute_coverage_mismatch(without)
    assert got == expected
    assert key not in got["inputs_used"] or key in base


@pytest.mark.parametrize("junk", ["garbage", 42, [1, 2, 3], object(), 3.5])
def test_non_mapping_inputs_are_neutral(junk):
    assert factor(junk) == 0.0


def test_numeric_strings_and_numpy_style_scalars_are_read():
    assert factor({"wr_vs_cb_matchup_grade": "80"}) == factor({"wr_vs_cb_matchup_grade": 80})
    assert factor({"wr_vs_cb_matchup_grade": " 80 "}) == factor({"wr_vs_cb_matchup_grade": 80})
    assert factor({"wr_vs_cb_matchup_grade": 80.0, "opp_cb1_shadow_rate": "0.6"}) == factor(
        {"wr_vs_cb_matchup_grade": 80.0, "opp_cb1_shadow_rate": 0.6}
    )


def test_rates_accept_fraction_or_percent():
    base = {"wr_vs_cb_matchup_grade": 80}
    for shadow_pct in (60, 60.0, "60%", "60"):
        assert factor({**base, "opp_cb1_shadow_rate": shadow_pct}) == factor({**base, "opp_cb1_shadow_rate": 0.6})
    split = {"wr_yprr_vs_man": 2.6, "wr_yprr_vs_zone": 1.2}
    for man_pct in (65, "65%", 65.0):
        assert factor({**split, "opp_man_coverage_rate": man_pct}) == factor({**split, "opp_man_coverage_rate": 0.65})
    # exactly 1 is 100%, not 1%
    assert factor({**base, "opp_cb1_shadow_rate": 1}) == factor({**base, "opp_cb1_shadow_rate": 1.0})
    assert factor({**base, "opp_cb1_shadow_rate": 1}) > factor({**base, "opp_cb1_shadow_rate": 0.01})


def test_grades_are_not_rescaled_and_out_of_range_grades_are_missing():
    assert factor({"wr_vs_cb_matchup_grade": "75%"}) == factor({"wr_vs_cb_matchup_grade": 75})
    # 0.65 is the 0.65th percentile (a terrible matchup), NOT 65%
    assert factor({"wr_vs_cb_matchup_grade": 0.65}) < 0 < factor({"wr_vs_cb_matchup_grade": 65})
    for out_of_range in (101, 100.5, -1, 1000):
        assert cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": out_of_range})["inputs_used"] == []
    for edge in (0, 100):
        assert cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": edge})["inputs_used"] == ["wr_vs_cb_matchup_grade"]


def test_yprr_bounds_are_zero_to_ten():
    assert cm.compute_coverage_mismatch({"wr_yprr_vs_man": 0.0})["inputs_used"] == ["wr_yprr_vs_man"]
    assert cm.compute_coverage_mismatch({"wr_yprr_vs_man": 10.0})["inputs_used"] == ["wr_yprr_vs_man"]
    assert cm.compute_coverage_mismatch({"wr_yprr_vs_man": 10.01})["inputs_used"] == []
    assert cm.compute_coverage_mismatch({"wr_yprr_vs_man": -0.1})["inputs_used"] == []


def test_a_bad_value_does_not_poison_the_good_ones():
    clean = cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": 80, "te_vs_lb_coverage_grade": 70})
    dirty = cm.compute_coverage_mismatch(
        {"wr_vs_cb_matchup_grade": 80, "te_vs_lb_coverage_grade": 70, "cb_man_coverage_grade": "n/a",
         "wr_yprr_vs_man": float("nan"), "opp_cb1_shadow_rate": None, "slot_wr_vs_nickel_grade": object()}
    )
    assert clean[KEY] == dirty[KEY] > 0


# ---------------------------------------------------------------------------
# compute: missing data shrinks toward neutral
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", cm.ROLES)
def test_missing_data_shrinks_toward_neutral_instead_of_amplifying(role):
    primary = {"wr": "wr_vs_cb_matchup_grade", "slot": "slot_wr_vs_nickel_grade", "te": "te_vs_lb_coverage_grade"}[role]
    full = factor(FAVOURABLE, role)
    partial = factor({primary: FAVOURABLE[primary], "opp_cb1_shadow_rate": FAVOURABLE["opp_cb1_shadow_rate"]}, role)
    assert 0 < partial < full
    # and the mirror image, symmetrically
    assert 0 > factor(mirror({primary: FAVOURABLE[primary]}), role) > factor(mirror(FAVOURABLE), role)
    cov_full = cm.compute_coverage_mismatch(FAVOURABLE, role=role)["coverage"]
    cov_part = cm.compute_coverage_mismatch({primary: 80}, role=role)["coverage"]
    # coverage is the share of the ROLE's weight that was observed: the TE role scores three factors (4.0), so a
    # lone TE-vs-LB grade is 3.0 / 4.0 = 0.75 by design, versus 3.0 / 6.5 = 0.46 for the WR / slot primary
    weights = {f.key: f.weight for f in cm.ROLE_FACTORS[role]}
    assert cov_full == 1.0
    assert cov_part == pytest.approx(weights[primary] / sum(weights.values()), abs=1e-3)
    assert 0 < cov_part < 1.0


def test_adding_a_concordant_input_never_shrinks_the_effect():
    steps = [
        {"wr_vs_cb_matchup_grade": 80},
        {"wr_vs_cb_matchup_grade": 80, "cb_man_coverage_grade": 25, "cb_zone_coverage_grade": 30},
        {"wr_vs_cb_matchup_grade": 80, "cb_man_coverage_grade": 25, "cb_zone_coverage_grade": 30,
         "wr_yprr_vs_man": 2.4, "wr_yprr_vs_zone": 2.2},
    ]
    values = [factor(s) for s in steps]
    assert values == sorted(values) and values[0] > 0


# ---------------------------------------------------------------------------
# compute: logging and drivers
# ---------------------------------------------------------------------------
def test_debug_log_is_emitted_on_the_pack_logger_and_silent_above_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        result = cm.compute_coverage_mismatch(FAVOURABLE, role="slot")
    records = [r for r in caplog.records if r.name.startswith("nfl_pack") and "coverage_mismatch_factor=" in r.getMessage()]
    assert records and records[0].levelno == logging.DEBUG
    message = records[0].getMessage()
    assert f"{result[KEY]:+.4f}" in message and "slot" in message
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        cm.compute_coverage_mismatch(FAVOURABLE)
    assert not [r for r in caplog.records if r.name.startswith("nfl_pack")]


def test_drivers_name_the_factors_that_moved_the_score_most():
    result = cm.compute_coverage_mismatch({"wr_vs_cb_matchup_grade": 90, "opp_cb1_shadow_rate": 0.8, "te_vs_lb_coverage_grade": 55})
    assert result["drivers"][0].startswith("WR vs CB matchup grade (shadow-weighted)")
    assert "raises the score" in result["drivers"][0]
    contribution = result["contributions"]["wr_vs_cb_matchup_grade"]
    assert contribution["inputs"] == ["wr_vs_cb_matchup_grade", "opp_cb1_shadow_rate"] and contribution["effect"] > 0


# ---------------------------------------------------------------------------
# Realistic scenarios (expected ordering)
# ---------------------------------------------------------------------------
def test_scenario_ordering_shadowed_by_an_elite_cb_vs_average_vs_soft_matchup():
    elite_shadow = {
        "wr_vs_cb_matchup_grade": 15, "opp_cb1_shadow_rate": 0.85, "cb_man_coverage_grade": 92, "cb_zone_coverage_grade": 85,
        "wr_yprr_vs_man": 1.9, "wr_yprr_vs_zone": 1.8, "opp_man_coverage_rate": 0.55,
    }
    soft = {
        "wr_vs_cb_matchup_grade": 85, "opp_cb1_shadow_rate": 0.15, "cb_man_coverage_grade": 25, "cb_zone_coverage_grade": 35,
        "wr_yprr_vs_man": 2.0, "wr_yprr_vs_zone": 1.8, "opp_man_coverage_rate": 0.50,
    }
    low, mid, high = factor(elite_shadow), factor(AVERAGE), factor(soft)
    assert low < -0.3 and mid == 0.0 and high > 0.3
    projection = {"wr_yards": 75.0, "te_yards": 45.0}
    lo_p, _ = cm.apply_coverage_mismatch(projection, cm.compute_coverage_mismatch(elite_shadow))
    hi_p, _ = cm.apply_coverage_mismatch(projection, cm.compute_coverage_mismatch(soft))
    assert lo_p["wr_yards"] < 75.0 < hi_p["wr_yards"]
    assert lo_p["wr_yards"] >= 75.0 * 0.75  # the +-25% cap
    assert hi_p["wr_yards"] <= 75.0 * 1.25


def test_scenario_a_wr_who_beats_man_vs_a_man_heavy_defense_with_a_weak_man_cb():
    wr = {"wr_yprr_vs_man": 2.6, "wr_yprr_vs_zone": 1.5}
    mismatch = {**wr, "opp_man_coverage_rate": 0.65, "cb_man_coverage_grade": 25, "cb_zone_coverage_grade": 70}
    wrong_scheme = {**wr, "opp_man_coverage_rate": 0.20, "cb_man_coverage_grade": 25, "cb_zone_coverage_grade": 70}
    strong_man_cb = {**wr, "opp_man_coverage_rate": 0.65, "cb_man_coverage_grade": 85, "cb_zone_coverage_grade": 70}
    assert factor(mismatch) > factor(wrong_scheme)
    assert factor(mismatch) > factor(strong_man_cb)
    assert factor(mismatch) > 0


def test_scenario_a_mediocre_grade_matters_more_when_the_cb_shadows():
    grade = {"wr_vs_cb_matchup_grade": 30}  # an unfavourable but not extreme matchup
    assert factor({**grade, "opp_cb1_shadow_rate": 0.0}) > factor({**grade, "opp_cb1_shadow_rate": 1.0})
    assert factor({**grade, "opp_cb1_shadow_rate": 1.0}) < -0.2


def test_scenario_te_vs_a_soft_linebacker_group_and_slot_vs_a_weak_nickel():
    assert factor({"te_vs_lb_coverage_grade": 82}, "te") > 0.2
    assert factor({"te_vs_lb_coverage_grade": 22}, "te") < -0.2
    assert factor({"slot_wr_vs_nickel_grade": 80, "wr_yprr_vs_man": 2.2, "wr_yprr_vs_zone": 2.0}, "slot") > 0.2
    projection = {"wr_yards": 60.0, "te_yards": 60.0}
    new, _ = cm.apply_coverage_mismatch(projection, cm.compute_coverage_mismatch({"te_vs_lb_coverage_grade": 82}, role="te"))
    # the TE coefficient (0.20) is smaller than the WR one (0.25) at the same factor
    assert (new["wr_yards"] - 60.0) > (new["te_yards"] - 60.0) > 0
