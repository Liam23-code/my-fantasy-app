"""Tests for fantasy.breakout_model -- the conservative, age-aware breakout estimate.

The whole point of this module is that an established veteran can never read as
a 70%+ breakout again, so the assertions are mostly about *bands* and
*ordering*, not exact numbers.
"""

from __future__ import annotations

import copy

from fantasy.breakout_model import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    compute_breakout_prob,
    compute_breakout_prob_batch,
)


def _wr(
    pid,
    *,
    age=None,
    years_experience=None,
    projection=220.0,
    prior=200.0,
    prior_games=17,
    adp=40.0,
    target_share=None,
    floor=None,
    ceiling=None,
    prior_injury=None,
):
    row = {
        "player_id": pid,
        "name": pid,
        "position": "WR",
        "projection": projection,
        "expected_fantasy_points": projection,
        "prior_season_points": prior,
        "prior_games_played": prior_games,
        "expected_games": 16.5,
        "adp": adp,
    }
    if age is not None:
        row["age"] = age
    if years_experience is not None:
        row["years_experience"] = years_experience
    if target_share is not None:
        row["target_share"] = target_share
    if floor is not None:
        row["floor"] = floor
    if ceiling is not None:
        row["ceiling"] = ceiling
    if prior_injury is not None:
        row["prior_injury_status"] = prior_injury
    return row


# --- bounds ---------------------------------------------------------------


def test_probability_is_always_bounded():
    extreme_low = compute_breakout_prob(_wr("old", age=35, projection=90.0, prior=260.0, adp=6.0))
    extreme_high = compute_breakout_prob(
        _wr("rookieish", age=22, projection=300.0, prior=40.0, prior_games=6, adp=None, target_share=0.30, floor=150.0, ceiling=360.0)
    )
    for result in (extreme_low, extreme_high):
        assert MIN_PROBABILITY <= result["breakout_probability"] <= MAX_PROBABILITY
    assert extreme_high["breakout_probability"] <= 0.65
    assert extreme_low["breakout_probability"] <= 0.12


def test_never_mutates_its_input():
    row = _wr("keep", age=27, target_share=0.2, floor=120.0, ceiling=260.0)
    before = copy.deepcopy(row)
    compute_breakout_prob(row)
    assert row == before


# --- the headline bug: old established vets ------------------------------


def test_established_thirty_one_year_old_wr1_is_not_a_breakout():
    star = _wr("aged-star", age=31, projection=250.0, prior=255.0, adp=9.0)
    result = compute_breakout_prob(star)
    assert result["breakout_probability"] <= 0.15
    assert result["breakout_tier"] == "unlikely"


def test_young_ascending_player_scores_far_above_an_identical_older_one():
    young = compute_breakout_prob(_wr("young", age=23, projection=230.0, prior=120.0, adp=55.0))
    old = compute_breakout_prob(_wr("old", age=30, projection=230.0, prior=120.0, adp=55.0))
    assert young["breakout_probability"] > old["breakout_probability"] + 0.10


def test_a_genuine_breakout_profile_lands_in_the_candidate_band():
    candidate = _wr(
        "riser",
        age=24,
        projection=245.0,
        prior=110.0,
        adp=70.0,
        target_share=0.25,
        floor=150.0,
        ceiling=300.0,
    )
    result = compute_breakout_prob(candidate)
    assert 0.30 <= result["breakout_probability"] <= 0.60
    assert result["breakout_tier"] in {"strong", "elite"}


# --- degradation paths -------------------------------------------------------


def test_experience_stands_in_when_age_is_missing_and_caps_tighter():
    rookie_ish = compute_breakout_prob(_wr("exp1", years_experience=1, projection=200.0, prior=70.0, adp=90.0))
    vet = compute_breakout_prob(_wr("exp9", years_experience=9, projection=200.0, prior=190.0, adp=30.0))
    assert rookie_ish["breakout_probability"] > vet["breakout_probability"]
    # the experience-only path is capped at 0.50, not 0.65
    assert rookie_ish["breakout_probability"] <= 0.50


def test_established_starter_is_damped_on_the_experience_only_path():
    # age never joined from the pool, so the youth curve runs off
    # years_experience alone. An established top-5 player with 8 seasons is a
    # plateaued starter, not a breakout -- the established-star damper must fire
    # here too. (Regression: the branch guard was previously an impossible
    # `age < 20.0 and experience >= 6.0`, so it never ran.)
    established = _wr("rank3-no-age", years_experience=8, projection=250.0, prior=180.0, adp=60.0)
    established["position_rank"] = 3
    plain = _wr("plain-no-age", years_experience=8, projection=250.0, prior=180.0, adp=60.0)
    damped = compute_breakout_prob(established)
    undamped = compute_breakout_prob(plain)
    assert damped["breakout_probability"] < undamped["breakout_probability"] - 0.01
    assert any("limited leap room" in d for d in damped["drivers"])


def test_no_age_and_no_experience_sits_near_the_base_rate():
    result = compute_breakout_prob(_wr("blank", projection=180.0, prior=175.0, adp=60.0))
    assert 0.05 <= result["breakout_probability"] <= 0.30


def test_holdout_or_suspension_floors_the_probability():
    row = _wr("holdout-guy", age=24, projection=240.0, prior=110.0, adp=45.0, target_share=0.26)
    assert compute_breakout_prob(row, live_status="HOLDOUT")["breakout_probability"] == MIN_PROBABILITY
    assert compute_breakout_prob(row, live_status="SUSPENDED")["breakout_probability"] == MIN_PROBABILITY


def test_injury_bounce_back_lifts_a_returning_player():
    healthy = compute_breakout_prob(_wr("healthy", age=25, projection=210.0, prior=205.0, prior_games=17, adp=40.0))
    returning = compute_breakout_prob(_wr("returning", age=25, projection=210.0, prior=95.0, prior_games=6, adp=40.0))
    assert returning["breakout_probability"] > healthy["breakout_probability"]


def test_chronic_injury_designation_damps_the_number():
    clean = compute_breakout_prob(_wr("clean", age=24, projection=230.0, prior=110.0, prior_games=8, adp=50.0))
    chronic = compute_breakout_prob(_wr("chronic", age=24, projection=230.0, prior=110.0, prior_games=6, adp=50.0, prior_injury="IR"))
    assert chronic["breakout_probability"] < clean["breakout_probability"]


# --- batch ---------------------------------------------------------------


def test_batch_keys_by_player_id_and_applies_status_map():
    rows = [
        _wr("a", age=24, projection=230.0, prior=110.0, target_share=0.24),
        _wr("b", age=31, projection=250.0, prior=255.0, adp=9.0),
        {"name": "no-id", "position": "WR", "projection": 100.0},
    ]
    out = compute_breakout_prob_batch(rows, status_by_id={"a": "HOLDOUT"})
    assert set(out) == {"a", "b"}
    assert out["a"]["breakout_probability"] == MIN_PROBABILITY  # holdout floor
    assert out["b"]["breakout_probability"] <= 0.15
