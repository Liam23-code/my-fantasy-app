"""The upgrade pack's parlay layer inside ``modules.nfl_parlay_builder``. Offline.

Default off must be byte-identical; informative signals move only same-game, non-structural pairs.
"""
from __future__ import annotations

import copy
from unittest import mock

import pytest

from modules import nfl_parlay_builder as pb
from modules import nfl_upgrade_pipeline as pipe
from modules import parlay_correlation

GAME = "2026_06_BUF_KC"
OTHER_GAME = "2026_06_NYJ_MIA"

ADVERSE_CONTEXT = {
    "offense": {"ol_pass_block_win_rate": 0.60, "ol_continuity": 0.60, "ol_injuries": 2.5},
    "opponent_defense": {"opp_pressure_rate": 0.40, "opp_blitz_rate": 0.36, "opp_pass_rush_win_rate": 0.38,
                         "opp_simulated_pressure_success": 0.35, "opp_def_epa_per_play": -0.10, "opp_defense_rank": 3,
                         "opp_coverage_grade": 92.0},
    "qb": {"qb_time_to_throw": 3.2, "qb_pressure_epa": -0.9, "qb_sack_avoidance": 0.89, "qb_release_time": 2.7},
    "game": {"vegas_spread": 6.0, "vegas_total": 40.0},
}
FAVOURABLE_CONTEXT = {
    "offense": {"ol_pass_block_win_rate": 0.80, "ol_continuity": 0.96, "ol_injuries": 0.0, "team_epa_per_play": 0.15,
                "explosive_pass_rate": 0.115, "red_zone_efficiency": 0.68},
    "opponent_defense": {"opp_pressure_rate": 0.22, "opp_blitz_rate": 0.18, "opp_pass_rush_win_rate": 0.24,
                         "opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_coverage_grade": 8.0,
                         "opp_explosive_play_prevention": 6.0},
    "qb": {"qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96, "qb_release_time": 2.2},
    "game": {"vegas_spread": -3.0, "vegas_total": 52.0},
}


def signals(context, team="KC"):
    return {team: pipe.team_signals(context)}


def qb_over(**kw):
    base = dict(description="Pat Pocket over 250.5 passing yards", probability=0.5, price=-110, game_id=GAME, team="KC",
                opponent="BUF", home_team="KC", away_team="BUF", side="over", line=250.5, market="passing_yards",
                player_name="Pat Pocket", position="QB")
    base.update(kw)
    return pb.make_parlay_leg("prop", **base)


def wr_over(**kw):
    base = dict(description="Wes Receiver over 65.5 receiving yards", probability=0.5, price=-110, game_id=GAME, team="KC",
                opponent="BUF", home_team="KC", away_team="BUF", side="over", line=65.5, market="receiving_yards",
                player_name="Wes Receiver", position="WR")
    base.update(kw)
    return pb.make_parlay_leg("prop", **base)


def other_game_leg():
    return pb.make_parlay_leg(
        "prop", description="Mia back over 60.5 rush yards", probability=0.5, price=-110, game_id=OTHER_GAME, team="MIA",
        opponent="NYJ", home_team="MIA", away_team="NYJ", side="over", line=60.5, market="rushing_yards",
        player_name="Mia Back", position="RB",
    )


def ml_home(**kw):
    base = dict(description="KC moneyline", probability=0.6, price=-150, game_id=GAME, team="KC", opponent="BUF",
                home_team="KC", away_team="BUF")
    base.update(kw)
    return pb.make_parlay_leg("moneyline", **base)


def ml_away(**kw):
    base = dict(description="BUF moneyline", probability=0.45, price=+130, game_id=GAME, team="BUF", opponent="KC",
                home_team="KC", away_team="BUF")
    base.update(kw)
    return pb.make_parlay_leg("moneyline", **base)


def spread_home(**kw):
    base = dict(description="KC -3.5", probability=0.52, price=-110, game_id=GAME, team="KC", opponent="BUF",
                home_team="KC", away_team="BUF", line=-3.5)
    base.update(kw)
    return pb.make_parlay_leg("spread", **base)


STACK = [qb_over(), wr_over()]


# --- default off ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("legs", [
    STACK, [qb_over(), wr_over(), other_game_leg()], [ml_home(), qb_over()], [ml_home(), spread_home(), wr_over()],
    [qb_over(), other_game_leg()], [ml_home(), ml_away()],
])
def test_default_is_byte_identical(legs):
    plain = pb.evaluate_parlay_legs(legs)
    assert pb.evaluate_parlay_legs(legs, pack_signals=None) == plain
    assert "pack" not in plain
    assert pb.correlation_matrix(legs, pack_signals=None) == pb.correlation_matrix(legs)
    assert pb.correlation_matrix(legs, pack_signals={}) == pb.correlation_matrix(legs)


def test_empty_signals_report_but_change_no_number():
    plain = pb.evaluate_parlay_legs(STACK)
    result = pb.evaluate_parlay_legs(STACK, pack_signals={})
    pack = result.pop("pack")
    assert result == plain
    assert pack["applied"] is False and pack["pack_pairs"] == 0 and pack["pairs"] == []
    assert pack["flags"]["trap"] is False and pack["flags"]["mega"] is False


def test_a_league_average_game_leaves_the_rule_table_exactly():
    average = {"offense": {"team_epa_per_play": 0.0}, "opponent_defense": {"opp_pressure_rate": 0.30}}
    plain = pb.evaluate_parlay_legs(STACK)
    result = pb.evaluate_parlay_legs(STACK, pack_signals=signals(average))
    assert result["matrix"] == plain["matrix"]
    assert result["adjusted_probability"] == plain["adjusted_probability"]


# --- informative signals ----------------------------------------------------------------------------------
def test_an_adverse_game_flags_the_stack_as_a_trap_and_lowers_the_joint_probability():
    plain = pb.evaluate_parlay_legs(STACK)
    bad = pb.evaluate_parlay_legs(STACK, pack_signals=signals(ADVERSE_CONTEXT))
    good = pb.evaluate_parlay_legs(STACK, pack_signals=signals(FAVOURABLE_CONTEXT))
    assert bad["pack"]["flags"]["trap"] is True and bad["pack"]["flags"]["mega"] is False
    assert good["pack"]["flags"]["trap"] is False
    assert bad["pack"]["score"] < good["pack"]["score"]
    assert bad["adjusted_probability"] < good["adjusted_probability"]
    assert bad["matrix"][0][1] < plain["matrix"][0][1] < good["matrix"][0][1]
    assert bad["pack"]["applied"] and bad["pack"]["pack_pairs"] == 1
    changed = bad["pack"]["pairs"][0]
    assert changed["blended_rho"] == pytest.approx(bad["matrix"][0][1], abs=1e-4)   # reported to 4 places
    assert changed["base_rho"] == pytest.approx(plain["matrix"][0][1], abs=1e-4)
    assert any(w.startswith(pb.PACK_NOTE_PREFIX) and "TRAP" not in w and "work against" in w for w in bad["warnings"])


def test_a_trap_adds_risk_and_never_lowers_it():
    plain = pb.evaluate_parlay_legs(STACK)
    bad = pb.evaluate_parlay_legs(STACK, pack_signals=signals(ADVERSE_CONTEXT))
    assert bad["risk_score"] >= plain["risk_score"]
    assert bad["risk_breakdown"].get("pack_trap") == pb.PACK_TRAP_RISK_POINTS
    good = pb.evaluate_parlay_legs(STACK, pack_signals=signals(FAVOURABLE_CONTEXT))
    assert good["risk_breakdown"].get("pack_trap") is None
    assert good["risk_score"] >= 0


def test_favourable_regimes_make_the_stack_mega_eligible_with_a_cited_line():
    good = pb.evaluate_parlay_legs(STACK, pack_signals=signals(FAVOURABLE_CONTEXT))
    flags = good["pack"]["flags"]
    assert flags["mega"] is True and flags["trap"] is False
    assert any("MEGA" in line for line in good["risk_factors"])
    assert good["pack"]["reasons"]


def test_only_same_game_non_structural_pairs_are_touched():
    legs = [qb_over(), wr_over(), other_game_leg()]
    plain = pb.correlation_matrix(legs)
    packed = pb.correlation_matrix(legs, pack_signals=signals(ADVERSE_CONTEXT))
    assert packed[0][2] == plain[0][2] and packed[1][2] == plain[1][2]       # different games: untouched
    assert packed[0][1] != plain[0][1]
    for i in range(3):
        assert packed[i][i] == 1.0
        for j in range(3):
            assert packed[i][j] == pytest.approx(packed[j][i])
            assert -1.0 <= packed[i][j] <= 1.0


def test_structural_ties_and_the_contradiction_sentinel_survive():
    tied = [ml_home(), spread_home()]
    base = pb.correlation_matrix(tied)
    after = pb.correlation_matrix(tied, pack_signals=signals(ADVERSE_CONTEXT))
    assert after == base                                                      # an ML and its own spread are one variable
    contradictory = [ml_home(), ml_away()]
    result = pb.evaluate_parlay_legs(contradictory, pack_signals=signals(ADVERSE_CONTEXT))
    assert result["has_contradiction"] is True and result["adjusted_probability"] == 0.0
    assert min(min(row) for row in result["matrix"]) == pb.CONTRADICTORY_RHO
    assert result["risk_score"] == 100.0


def test_a_team_leg_and_a_prop_are_blended_without_breaking_the_tie_groups():
    legs = [ml_home(), spread_home(), wr_over()]
    out = pb.evaluate_parlay_legs(legs, pack_signals=signals(FAVOURABLE_CONTEXT))
    m = out["matrix"]
    assert m[0][1] == pytest.approx(1.0) or abs(m[0][1]) == pytest.approx(1.0)  # the tie is preserved
    assert m[0][2] == pytest.approx(m[1][2], abs=1e-9)                         # the third leg sees ONE variable
    assert 0.0 < out["adjusted_probability"] < 1.0


def test_over_and_under_move_in_opposite_directions():
    over_under = [qb_over(), wr_over(side="under", description="Wes Receiver under 65.5 receiving yards")]
    plain = pb.correlation_matrix(over_under)
    bad = pb.correlation_matrix(over_under, pack_signals=signals(ADVERSE_CONTEXT))
    stack_plain = pb.correlation_matrix(STACK)
    stack_bad = pb.correlation_matrix(STACK, pack_signals=signals(ADVERSE_CONTEXT))
    assert (bad[0][1] - plain[0][1]) * (stack_bad[0][1] - stack_plain[0][1]) < 0


# --- failure ---------------------------------------------------------------------------------------------
def test_a_defect_in_the_pack_leaves_pricing_exactly_as_the_rule_table_has_it():
    plain = pb.evaluate_parlay_legs(STACK)
    with mock.patch.object(parlay_correlation, "build_correlation_matrix", side_effect=RuntimeError("boom")):
        result = pb.evaluate_parlay_legs(STACK, pack_signals=signals(ADVERSE_CONTEXT))
    pack = result.pop("pack")
    assert pack["applied"] is False and "boom" in (pack["error"] or "")
    assert result["matrix"] == plain["matrix"]
    assert result["adjusted_probability"] == plain["adjusted_probability"]


def test_inputs_are_not_mutated():
    legs = copy.deepcopy(STACK)
    sig = copy.deepcopy(signals(ADVERSE_CONTEXT))
    snapshot = copy.deepcopy(sig)
    pb.evaluate_parlay_legs(legs, pack_signals=sig)
    assert legs == STACK and sig == snapshot


def test_evaluate_still_returns_every_existing_key_when_the_pack_is_on():
    plain = pb.evaluate_parlay_legs(STACK)
    packed = pb.evaluate_parlay_legs(STACK, pack_signals=signals(ADVERSE_CONTEXT))
    assert set(plain) | {"pack"} == set(packed)


# --- ranking -----------------------------------------------------------------------------------------------
def test_rank_parlay_opportunities_puts_the_regime_aligned_stack_first_and_is_stable():
    sig = {"KC": pipe.team_signals(FAVOURABLE_CONTEXT), "BUF": pipe.team_signals(ADVERSE_CONTEXT)}
    kc_stack = [qb_over(), wr_over()]
    buf_stack = [
        qb_over(description="Rob QB over 230.5 passing yards", team="BUF", opponent="KC", player_name="Rob QB", line=230.5),
        wr_over(description="Buff WR over 60.5 receiving yards", team="BUF", opponent="KC", player_name="Buff WR", line=60.5),
    ]
    ranked = pb.rank_parlay_opportunities([buf_stack, kc_stack], pack_signals=sig)
    assert [r["rank"] for r in ranked] == [1, 2]
    assert ranked[0]["candidate"] == 1 and ranked[1]["candidate"] == 0
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["evaluation"]["num_legs"] == 2 and "pack" in ranked[0]["evaluation"]
    again = pb.rank_parlay_opportunities([buf_stack, kc_stack], pack_signals=sig)
    assert [r["candidate"] for r in again] == [r["candidate"] for r in ranked]


def test_rank_parlay_opportunities_drops_what_cannot_be_priced():
    sig = signals(FAVOURABLE_CONTEXT)
    ranked = pb.rank_parlay_opportunities([[qb_over()], STACK, []], pack_signals=sig)
    assert len(ranked) == 1 and ranked[0]["candidate"] == 1
    assert pb.rank_parlay_opportunities([], pack_signals=sig) == []


def test_public_surface():
    assert "rank_parlay_opportunities" in pb.__all__ and "PACK_TRAP_RISK_POINTS" in pb.__all__
