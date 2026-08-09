"""Tests for fantasy.room_brain: the ESPN-style behavior for non-user teams."""

from __future__ import annotations

import numpy as np

from fantasy.room_brain import (
    REACH_TOLERANCE_PICKS,
    is_round_one_chalk,
    room_brain_weight,
    round_one_pick,
)


def _player(name: str, position: str, adp: float | None = None, vor: float = 0.0, volatility: float = 0.0) -> dict:
    return {"name": name, "position": position, "adp": adp, "vor": vor, "volatility": volatility}


# --- ADP is the dominant timing signal --------------------------------------


def test_a_faller_outweighs_a_reach_of_equal_vor():
    """A player past their ADP should heavily outweigh one drafted well before it."""
    faller = _player("Faller", "RB", adp=10.0, vor=50.0)
    reach = _player("Reach", "RB", adp=90.0, vor=50.0)
    assert room_brain_weight(faller, overall_pick=40, round_number=5) > room_brain_weight(reach, overall_pick=40, round_number=5)


def test_reach_beyond_tolerance_is_heavily_suppressed():
    close_reach = _player("CloseReach", "RB", adp=42.0, vor=50.0)  # 2 picks early, within tolerance
    far_reach = _player("FarReach", "RB", adp=90.0, vor=50.0)  # 50 picks early
    assert room_brain_weight(far_reach, 40, 5) < room_brain_weight(close_reach, 40, 5)


def test_missing_adp_is_neutral_not_penalized():
    with_adp = _player("HasADP", "WR", adp=200.0, vor=10.0)  # a big reach
    no_adp = _player("NoADP", "WR", adp=None, vor=10.0)
    # No ADP means no timing opinion -- shouldn't be punished like a confirmed reach.
    assert room_brain_weight(no_adp, 10, 2) > room_brain_weight(with_adp, 10, 2)


def test_adp_dominates_over_raw_vor_ties():
    """Section 3 fix: previously pure-VOR weighting let a low-ADP star fall for rounds.

    Multiplicative timing means this holds regardless of how large VOR gets --
    an additive combination would let a big enough VOR edge swamp even a
    heavily-suppressed reach penalty, silently reintroducing the same bug.
    """
    high_vor_reach = _player("Superstar", "RB", adp=90.0, vor=300.0)
    modest_vor_on_time = _player("OnTime", "RB", adp=40.0, vor=50.0)
    assert room_brain_weight(modest_vor_on_time, 40, 5) > room_brain_weight(high_vor_reach, 40, 5)


# --- positional bias ---------------------------------------------------------


def test_rb_is_favored_early_and_te_qb_are_patient():
    rb = _player("EarlyRB", "RB", adp=5.0, vor=50.0)
    te = _player("EarlyTE", "TE", adp=5.0, vor=50.0)
    qb = _player("EarlyQB", "QB", adp=5.0, vor=50.0)
    early_rb_weight = room_brain_weight(rb, overall_pick=5, round_number=1)
    early_te_weight = room_brain_weight(te, overall_pick=5, round_number=1)
    early_qb_weight = room_brain_weight(qb, overall_pick=5, round_number=1)
    assert early_rb_weight > early_te_weight
    assert early_rb_weight > early_qb_weight


def test_te_patience_relaxes_in_late_rounds():
    te = _player("LateTE", "TE", adp=100.0, vor=20.0)
    early_weight = room_brain_weight(te, overall_pick=100, round_number=2)
    late_weight = room_brain_weight(te, overall_pick=100, round_number=12)
    assert late_weight > early_weight


# --- risk aversion in early rounds ------------------------------------------


def test_early_rounds_downweight_volatility():
    stable = _player("Stable", "RB", adp=10.0, vor=50.0, volatility=0.1)
    volatile = _player("Volatile", "RB", adp=10.0, vor=50.0, volatility=1.5)
    assert room_brain_weight(stable, 10, round_number=1) > room_brain_weight(volatile, 10, round_number=1)


def test_risk_aversion_does_not_apply_in_late_rounds():
    volatile = _player("Volatile", "RB", adp=10.0, vor=50.0, volatility=1.5)
    early = room_brain_weight(volatile, 10, round_number=1)
    late = room_brain_weight(dict(volatile), 10, round_number=10)
    assert late >= early  # no early-round risk discount applied this late


# --- reach tolerance scaling (mock_data_ingestion integration) --------------


def test_reach_tolerance_scale_widens_or_narrows_the_penalty_free_zone():
    pick = 5
    reach_player = _player("Reach", "TE", adp=pick + REACH_TOLERANCE_PICKS + 9.0, vor=10.0)
    narrow = room_brain_weight(reach_player, pick, round_number=5, reach_tolerance_scale=0.5)
    wide = room_brain_weight(reach_player, pick, round_number=5, reach_tolerance_scale=2.0)
    assert wide > narrow


def test_weight_is_always_positive():
    assert room_brain_weight(_player("X", "K", adp=300.0, vor=-50.0, volatility=5.0), 1, 1) > 0


# --- round 1 chalk ------------------------------------------------------------


def test_is_round_one_chalk_only_true_for_round_one():
    assert is_round_one_chalk(1) is True
    assert is_round_one_chalk(2) is False
    assert is_round_one_chalk(15) is False


def test_round_one_pick_takes_the_single_candidate():
    only = _player("Only", "RB", adp=1.0)
    assert round_one_pick([only], np.random.default_rng(0)) is only


def test_round_one_pick_mostly_takes_the_best_adp_candidate():
    best = _player("Best", "RB", adp=1.0)
    second = _player("Second", "WR", adp=2.0)
    picks = [round_one_pick([best, second], np.random.default_rng(seed))["name"] for seed in range(200)]
    best_share = picks.count("Best") / len(picks)
    assert best_share > 0.7  # mostly chalk, not a coin flip


def test_round_one_pick_is_reproducible_by_seed():
    best = _player("Best", "RB", adp=1.0)
    second = _player("Second", "WR", adp=2.0)
    first_run = [round_one_pick([best, second], np.random.default_rng(7))["name"] for _ in range(5)]
    second_run = [round_one_pick([best, second], np.random.default_rng(7))["name"] for _ in range(5)]
    assert first_run == second_run


# --- ESPN-style behavioral overrides (Section 2) ----------------------------


def test_rb_premium_rounds_1_to_3():
    from fantasy.room_brain import _position_round_multiplier

    assert _position_round_multiplier("RB", 1) > _position_round_multiplier("RB", 4)
    assert _position_round_multiplier("RB", 3) == _position_round_multiplier("RB", 1)


def test_wr_waves_rounds_3_to_7():
    from fantasy.room_brain import _position_round_multiplier

    assert _position_round_multiplier("WR", 5) > _position_round_multiplier("WR", 1)
    assert _position_round_multiplier("WR", 5) > _position_round_multiplier("WR", 9)


def test_te_patience_until_round_5():
    from fantasy.room_brain import _position_round_multiplier

    assert _position_round_multiplier("TE", 4) < _position_round_multiplier("TE", 5)


def test_qb_patience_until_round_7():
    from fantasy.room_brain import _position_round_multiplier

    assert _position_round_multiplier("QB", 6) < _position_round_multiplier("QB", 7)


def test_kicker_and_dst_suppressed_before_round_9():
    from fantasy.room_brain import _position_round_multiplier

    assert _position_round_multiplier("K", 8) < _position_round_multiplier("K", 9)
    assert _position_round_multiplier("DST", 1) < 0.5


def test_value_trap_is_dampened_even_when_falling_far_past_adp():
    """A big ADP fall on a below-replacement player is a trap, not a steal."""
    trap = _player("Trap", "WR", adp=10.0, vor=-40.0)
    real_steal = _player("RealSteal", "WR", adp=10.0, vor=40.0)
    pick = 60
    assert room_brain_weight(real_steal, pick, round_number=5) > room_brain_weight(trap, pick, round_number=5)


def test_value_trap_dampening_does_not_affect_positive_vor_players():
    stud = _player("Stud", "WR", adp=10.0, vor=40.0)
    from fantasy.room_brain import VALUE_TRAP_DAMPENING

    assert VALUE_TRAP_DAMPENING < 1.0  # sanity: the constant is a real dampening factor
    assert room_brain_weight(stud, 60, round_number=5) > 0


def test_injury_risk_suppressed_in_early_rounds():
    healthy = _player("Healthy", "RB", adp=10.0, vor=50.0)
    hurt = dict(healthy, name="Hurt", injury_status="OUT")
    assert room_brain_weight(healthy, 10, round_number=1) > room_brain_weight(hurt, 10, round_number=1)


def test_injury_risk_not_suppressed_in_late_rounds():
    hurt = dict(_player("Hurt", "RB", adp=10.0, vor=50.0), injury_status="OUT")
    early = room_brain_weight(hurt, 10, round_number=1)
    late = room_brain_weight(dict(hurt), 10, round_number=10)
    assert late >= early


def test_round_one_is_a_chalk_lock_with_no_upset_chance():
    from fantasy.room_brain import ROUND_ONE_UPSET_CHANCE

    assert ROUND_ONE_UPSET_CHANCE == 0.0
    best = _player("Best", "RB", adp=1.0)
    second = _player("Second", "WR", adp=2.0)
    picks = {round_one_pick([best, second], np.random.default_rng(seed))["name"] for seed in range(50)}
    assert picks == {"Best"}  # never the 2nd-best, now that it's a lock


# --- fused run_pressure integration (this turn) ------------------------------


def test_run_pressure_boosts_weight():
    baseline = _player("Baseline", "WR", adp=10.0, vor=50.0)
    pressured = dict(baseline, name="Pressured", run_pressure=1.0)
    assert room_brain_weight(pressured, 10, round_number=5) > room_brain_weight(baseline, 10, round_number=5)


def test_run_pressure_absent_is_neutral():
    no_field = _player("NoPressure", "WR", adp=10.0, vor=50.0)
    zero_field = dict(no_field, name="ZeroPressure", run_pressure=0.0)
    assert room_brain_weight(no_field, 10, round_number=5) == room_brain_weight(zero_field, 10, round_number=5)


# --- ADP-anchoring: the falling-players fix ---------------------------------


def test_vor_modifier_is_bounded_and_never_zeroes_a_player():
    from fantasy.room_brain import VOR_MODIFIER_MAX, VOR_MODIFIER_MIN, _vor_modifier

    assert _vor_modifier(-1000.0) == VOR_MODIFIER_MIN
    assert _vor_modifier(1000.0) == VOR_MODIFIER_MAX
    assert _vor_modifier(0.0) == 1.0
    assert VOR_MODIFIER_MIN > 0  # a bad VORP must never make a player undraftable


def test_high_adp_negative_vor_player_stays_competitive():
    """Regression: Garrett Wilson (ADP 30, VOR -71) once fell 135 picks past ADP.

    An unbounded `vor * timing` base gave him a measured 46x weight penalty
    against a positive-VOR comparable. Bounded VOR keeps a real discount
    without making a market-favoured player effectively undraftable.
    """
    fallen_star = _player("FallenStar", "WR", adp=30.0, vor=-71.0)
    healthy = _player("Healthy", "WR", adp=78.0, vor=76.0)
    # At pick 60 the star has fallen 30 past ADP; the healthy player is a reach.
    star_weight = room_brain_weight(fallen_star, 60, round_number=5)
    healthy_weight = room_brain_weight(healthy, 60, round_number=5)
    assert star_weight > healthy_weight / 5  # within striking distance, not 46x behind


def test_a_big_reach_is_still_suppressed_despite_bounded_vor():
    """The bounded modifier must not reintroduce the Saquon-style reach bug."""
    huge_vor_reach = _player("Reach", "RB", adp=150.0, vor=300.0)
    on_time = _player("OnTime", "RB", adp=40.0, vor=20.0)
    assert room_brain_weight(on_time, 40, 5) > room_brain_weight(huge_vor_reach, 40, 5)


def test_room_candidate_pool_orders_by_market_not_vorp():
    from fantasy.room_brain import room_candidate_pool

    # `eligible` arrives VORP-sorted, as simulate_draft produces it.
    eligible = [
        _player("HighVorLateAdp", "WR", adp=90.0, vor=200.0),
        _player("LowVorEarlyAdp", "RB", adp=3.0, vor=-50.0),
    ]
    pool = room_candidate_pool(eligible, size=2)
    assert pool[0]["name"] == "LowVorEarlyAdp"  # the market's board leads


def test_room_candidate_pool_puts_players_without_adp_last_keeping_vorp_order():
    from fantasy.room_brain import room_candidate_pool

    eligible = [
        _player("NoAdpBest", "WR", adp=None, vor=300.0),
        _player("NoAdpWorse", "WR", adp=None, vor=10.0),
        _player("HasAdp", "RB", adp=50.0, vor=1.0),
    ]
    pool = room_candidate_pool(eligible, size=3)
    assert pool[0]["name"] == "HasAdp"
    assert [p["name"] for p in pool[1:]] == ["NoAdpBest", "NoAdpWorse"]  # stable VORP order


def test_room_candidate_pool_always_returns_at_least_one():
    from fantasy.room_brain import room_candidate_pool

    assert len(room_candidate_pool([_player("Only", "RB", adp=1.0)], size=0)) == 1


# --- round_curve --------------------------------------------------------------


def test_round_curve_boosts_the_rounds_a_player_actually_went_in():
    from fantasy.room_brain import _round_curve_multiplier

    curve = {3: 1.0}
    assert _round_curve_multiplier(curve, 3) > 1.0
    assert _round_curve_multiplier(curve, 8) < 1.0


def test_round_curve_missing_or_empty_is_neutral():
    from fantasy.room_brain import _round_curve_multiplier

    assert _round_curve_multiplier(None, 3) == 1.0
    assert _round_curve_multiplier({}, 3) == 1.0
    assert _round_curve_multiplier("not a dict", 3) == 1.0


def test_round_curve_changes_room_weight_end_to_end():
    in_window = dict(_player("InWindow", "WR", adp=30.0, vor=20.0), round_curve={3: 1.0})
    out_window = dict(_player("OutWindow", "WR", adp=30.0, vor=20.0), round_curve={9: 1.0})
    assert room_brain_weight(in_window, 30, 3) > room_brain_weight(out_window, 30, 3)
