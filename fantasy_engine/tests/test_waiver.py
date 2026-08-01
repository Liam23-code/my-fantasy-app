"""Unit tests for fantasy.waiver."""
from __future__ import annotations

import pytest

from fantasy.models import LeagueSettings
from fantasy.waiver import waiver_recommendations

BASE_SETTINGS = {
    "n_teams": 1,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 0},
    "flex_eligible": [],
}

# For tests isolating one multiplier's effect, two equal-value subjects need a
# *third*, distinctly worse, filler player at the same position -- otherwise
# with only 2 players and 1 dedicated slot, the worse of the two becomes its
# own "replacement" and both end up with replacement_value == 0, masking any
# multiplier difference (0 * anything is still 0). Requiring 2 dedicated RB
# slots means both equal-value subjects are starters and the filler sets a
# real, distinct replacement baseline.
TWO_RB_SETTINGS = {
    "n_teams": 1,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 0, "RB": 2, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 0},
    "flex_eligible": [],
}


def _fa(pid, name, position, points_field, value, **extra):
    return {"player_id": pid, "name": name, "position": position, points_field: value, **extra}


def _rb_filler(pid="filler", name="Filler", yards=10):
    return _fa(pid, name, "RB", "rushing_yards", yards)


def test_ranked_by_composite_score_descending():
    state = {"league_settings": BASE_SETTINGS, "my_roster": []}
    players = [
        _fa("rb1", "RB Big", "RB", "rushing_yards", 80),
        _fa("rb2", "RB Small", "RB", "rushing_yards", 20),
    ]
    ranked = waiver_recommendations(state, players, "ppr")
    assert [p["name"] for p in ranked] == ["RB Big", "RB Small"]
    assert ranked[0]["waiver_rank"] == 1
    assert ranked[1]["waiver_rank"] == 2


def test_need_multiplier_can_flip_ranking_between_close_players():
    # RB's dedicated slot is already filled (neutral 1.0x); WR's is empty
    # (1.25x boost). A weak filler at each position gives both a real,
    # comparable replacement-value baseline so the need multiplier -- not a
    # zero-VOR artifact -- is what decides the ranking.
    state = {"league_settings": BASE_SETTINGS, "my_roster": [{"position": "RB"}]}
    players = [
        _fa("rb1", "RB Guy", "RB", "rushing_yards", 55),
        _fa("rb2", "RB Filler", "RB", "rushing_yards", 10),
        _fa("wr1", "WR Guy", "WR", "receiving_yards", 50),
        _fa("wr2", "WR Filler", "WR", "receiving_yards", 10),
    ]
    ranked = waiver_recommendations(state, players, "ppr")
    assert ranked[0]["name"] == "WR Guy"  # WR is the position of need, RB already has 1/1


def test_ownership_pct_boosts_composite_score():
    state = {"league_settings": TWO_RB_SETTINGS, "my_roster": []}
    low_owned = _fa("a", "Low Owned", "RB", "rushing_yards", 50, ownership_pct=5)
    high_owned = _fa("b", "High Owned", "RB", "rushing_yards", 50, ownership_pct=80)
    ranked = waiver_recommendations(state, [low_owned, high_owned, _rb_filler()], "ppr")
    by_name = {p["name"]: p for p in ranked}
    assert by_name["Low Owned"]["replacement_value"] == by_name["High Owned"]["replacement_value"] > 0
    assert by_name["High Owned"]["composite_score"] > by_name["Low Owned"]["composite_score"]


THREE_RB_SETTINGS = {
    **TWO_RB_SETTINGS,
    "roster_requirements": {"QB": 0, "RB": 3, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 0},
}


def test_injury_status_penalizes_composite_score():
    state = {"league_settings": THREE_RB_SETTINGS, "my_roster": []}
    healthy = _fa("h", "Healthy", "RB", "rushing_yards", 50)
    out = _fa("o", "Hurt Out", "RB", "rushing_yards", 50, status="out")
    questionable = _fa("q", "Iffy", "RB", "rushing_yards", 50, status="questionable")
    ranked = waiver_recommendations(state, [healthy, out, questionable, _rb_filler()], "ppr")
    by_name = {p["name"]: p for p in ranked}
    assert by_name["Healthy"]["replacement_value"] == by_name["Iffy"]["replacement_value"] == by_name["Hurt Out"]["replacement_value"] > 0
    assert by_name["Healthy"]["composite_score"] > by_name["Iffy"]["composite_score"] > by_name["Hurt Out"]["composite_score"]
    assert "injury risk: OUT" in by_name["Hurt Out"]["rationale"]
    assert "questionable for this week" in by_name["Iffy"]["rationale"]


def test_bye_week_conflict_only_penalizes_matching_current_week():
    state = {"league_settings": TWO_RB_SETTINGS, "my_roster": [], "current_week": 9}
    on_bye_now = _fa("a", "On Bye Now", "RB", "rushing_yards", 50, bye_week=9)
    on_bye_later = _fa("b", "On Bye Later", "RB", "rushing_yards", 50, bye_week=12)
    ranked = waiver_recommendations(state, [on_bye_now, on_bye_later, _rb_filler()], "ppr")
    by_name = {p["name"]: p for p in ranked}
    assert by_name["On Bye Later"]["composite_score"] > by_name["On Bye Now"]["composite_score"]
    assert by_name["On Bye Now"]["bye_week_conflict"] is True
    assert "on bye this week" in by_name["On Bye Now"]["rationale"]
    assert by_name["On Bye Later"]["bye_week_conflict"] is False


def test_schedule_difficulty_direction():
    state = {"league_settings": TWO_RB_SETTINGS, "my_roster": []}
    easy = _fa("e", "Easy Slate", "RB", "rushing_yards", 50, schedule_difficulty=20)
    hard = _fa("h", "Hard Slate", "RB", "rushing_yards", 50, schedule_difficulty=90)
    ranked = waiver_recommendations(state, [easy, hard, _rb_filler()], "ppr")
    by_name = {p["name"]: p for p in ranked}
    assert by_name["Easy Slate"]["composite_score"] > by_name["Hard Slate"]["composite_score"]
    assert "favorable upcoming schedule" in by_name["Easy Slate"]["rationale"]
    assert "tough upcoming schedule" in by_name["Hard Slate"]["rationale"]


def test_no_budget_means_no_bid_suggestions():
    state = {"league_settings": BASE_SETTINGS, "my_roster": []}
    ranked = waiver_recommendations(state, [_fa("a", "A", "RB", "rushing_yards", 50)], "ppr")
    assert ranked[0]["suggested_faab_bid"] is None
    assert ranked[0]["suggested_auction_bid"] is None


def test_faab_budget_produces_bids_capped_and_proportional():
    state = {"league_settings": BASE_SETTINGS, "my_roster": []}
    players = [
        _fa("a", "Top Target", "RB", "rushing_yards", 90),
        _fa("b", "Depth Add", "RB", "rushing_yards", 10),
    ]
    ranked = waiver_recommendations(state, players, "ppr", budget=100)
    top, depth = ranked[0], ranked[1]
    assert top["suggested_faab_bid"] is not None
    assert 0 <= top["suggested_faab_bid"] <= 100
    assert top["suggested_faab_bid"] > depth["suggested_faab_bid"]
    assert top["suggested_auction_bid"] is None


def test_auction_league_produces_auction_bids_not_faab():
    settings = {**BASE_SETTINGS, "is_auction": True}
    state = {"league_settings": settings, "my_roster": []}
    ranked = waiver_recommendations(state, [_fa("a", "A", "RB", "rushing_yards", 90)], "ppr", budget=50)
    assert ranked[0]["suggested_auction_bid"] is not None
    assert ranked[0]["suggested_faab_bid"] is None
    assert ranked[0]["suggested_auction_bid"] <= 50


def test_non_positive_composite_score_gets_zero_bid():
    state = {"league_settings": BASE_SETTINGS, "my_roster": []}  # RB: 1 dedicated slot
    players = [
        _fa("a", "Good", "RB", "rushing_yards", 90),  # 9.0 pts, the starter
        _fa("b", "Replacement Level", "RB", "rushing_yards", 50),  # 5.0 pts, becomes the replacement
        _fa("c", "Below Replacement", "RB", "rushing_yards", 1),  # 0.1 pts, worse than replacement
    ]
    ranked = waiver_recommendations(state, players, "ppr", budget=100)
    below_replacement = next(p for p in ranked if p["name"] == "Below Replacement")
    assert below_replacement["replacement_value"] < 0
    assert below_replacement["suggested_faab_bid"] == 0.0


def test_scoring_mode_argument_overrides_league_settings_default():
    settings = {**BASE_SETTINGS, "scoring_mode": "standard"}
    state = {"league_settings": settings, "my_roster": []}
    player = _fa("a", "Pass Catcher", "RB", "rushing_yards", 0, receptions=5, receiving_yards=0)
    ppr_ranked = waiver_recommendations(state, [player], "ppr")
    standard_ranked = waiver_recommendations(state, [player], "standard")
    assert ppr_ranked[0]["points"] > standard_ranked[0]["points"]


def test_rationale_mentions_need_when_multiplier_boosted():
    state = {"league_settings": BASE_SETTINGS, "my_roster": []}
    ranked = waiver_recommendations(state, [_fa("a", "WR Guy", "WR", "receiving_yards", 50)], "ppr")
    assert "fills a starting need" in ranked[0]["rationale"]


def test_empty_available_players_returns_empty_list():
    state = {"league_settings": BASE_SETTINGS, "my_roster": []}
    assert waiver_recommendations(state, [], "ppr") == []


def test_accepts_league_settings_model_instance_directly():
    state = {"league_settings": LeagueSettings(**BASE_SETTINGS), "my_roster": []}
    ranked = waiver_recommendations(state, [_fa("a", "A", "RB", "rushing_yards", 50)], "ppr")
    assert len(ranked) == 1


def test_doubtful_injury_status_penalty_between_questionable_and_out():
    state = {"league_settings": THREE_RB_SETTINGS, "my_roster": []}
    questionable = _fa("q", "Iffy", "RB", "rushing_yards", 50, status="questionable")
    doubtful = _fa("d", "Dicey", "RB", "rushing_yards", 50, status="doubtful")
    out = _fa("o", "Hurt", "RB", "rushing_yards", 50, status="out")
    ranked = waiver_recommendations(state, [questionable, doubtful, out, _rb_filler()], "ppr")
    by_name = {p["name"]: p for p in ranked}
    assert by_name["Iffy"]["composite_score"] > by_name["Dicey"]["composite_score"] > by_name["Hurt"]["composite_score"]


def test_rationale_mentions_position_already_stocked():
    state = {"league_settings": BASE_SETTINGS, "my_roster": [{"position": "RB"}, {"position": "RB"}]}
    ranked = waiver_recommendations(state, [_fa("a", "RB Depth", "RB", "rushing_yards", 50), _rb_filler()], "ppr")
    depth = next(p for p in ranked if p["name"] == "RB Depth")
    assert depth["need_multiplier"] < 1.0
    assert "position already well-stocked" in depth["rationale"]
