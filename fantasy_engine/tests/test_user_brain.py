"""Tests for fantasy.user_brain: the user's analytics-driven pick during a sim."""

from __future__ import annotations

from fantasy.user_brain import user_brain_pick

SETTINGS = {
    "n_teams": 2,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 3},
    "flex_eligible": [],
}


def _p(name, position, projection, **extra):
    return {"player_id": f"id-{name}", "name": name, "position": position, "team": "SF", "projection": projection, **extra}


def _pool():
    return [_p("RB1", "RB", 300.0), _p("RB2", "RB", 100.0), _p("WR1", "WR", 280.0), _p("QB1", "QB", 320.0)]


def test_returns_the_actual_player_object_from_pool():
    pool = _pool()
    picked = user_brain_pick(pool, SETTINGS, my_roster=[], next_pick_overall=1, picks_until_next=10)
    assert picked is not None
    assert any(picked is player for player in pool)  # same object, not a rebuilt copy


def test_picks_the_best_value_player():
    pool = _pool()
    picked = user_brain_pick(pool, SETTINGS, my_roster=[], next_pick_overall=1, picks_until_next=10)
    # RB1 has both the highest raw value and a weak replacement behind it (only RB2 backing it up).
    assert picked["name"] == "RB1"


def test_respects_roster_caps_like_the_assistant_does():
    pool = [_p("QB1", "QB", 300.0), _p("QB2", "QB", 250.0), _p("WR1", "WR", 100.0)]
    roster = [{"position": "QB"}, {"position": "QB"}]  # QB cap is (1, 2): already maxed
    picked = user_brain_pick(pool, SETTINGS, my_roster=roster, next_pick_overall=1, picks_until_next=10)
    assert picked["position"] != "QB"


def test_empty_pool_returns_none():
    assert user_brain_pick([], SETTINGS, my_roster=[], next_pick_overall=1, picks_until_next=10) is None


def test_synthetic_only_pool_returns_none():
    synthetic = [{"player_id": "synthetic:rb:0", "name": "Synthetic RB 0", "position": "RB", "team": "XX", "projection": 999.0}]
    assert user_brain_pick(synthetic, SETTINGS, my_roster=[], next_pick_overall=1, picks_until_next=10) is None


def test_picks_change_with_roster_need():
    # Real position depth so VOR is non-degenerate (a 1-per-position pool makes
    # everyone's VOR exactly 0, and a need multiplier can't flip a 0-vs-0 tie).
    pool = [
        _p("RB1", "RB", 210.0), _p("RB2", "RB", 150.0), _p("RB3", "RB", 90.0),
        _p("WR1", "WR", 200.0), _p("WR2", "WR", 140.0), _p("WR3", "WR", 85.0),
    ]
    no_roster_pick = user_brain_pick(pool, SETTINGS, my_roster=[], next_pick_overall=1, picks_until_next=10)
    stacked_rb_pick = user_brain_pick(
        pool, SETTINGS, my_roster=[{"position": "RB"}] * 3, next_pick_overall=1, picks_until_next=10
    )
    assert no_roster_pick["name"] == "RB1"
    assert stacked_rb_pick["position"] != "RB"  # RB need already filled, something else should win now
