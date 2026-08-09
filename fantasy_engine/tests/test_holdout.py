"""Tests for fantasy.holdout: out-of-sample scoring that breaks the circularity.

The network-backed end-to-end path (:func:`evaluate_draft_on_holdout` against
real nflverse seasons) is deliberately not exercised here -- these cover the
scoring contract, which is where a silent bug would quietly flatter one team
over another.
"""

from __future__ import annotations

from fantasy.holdout import MISSING_PLAYER_POINTS, score_roster_on_holdout


def _rostered(player_id: str, name: str = "X", position: str = "RB") -> dict:
    return {"player_id": player_id, "name": name, "position": position}


def test_scores_a_roster_against_holdout_actuals():
    roster = [_rostered("a"), _rostered("b")]
    result = score_roster_on_holdout(roster, {"a": 100.0, "b": 50.0})
    assert result["total"] == 150.0
    assert result["scored"] == 2
    assert result["missing"] == 0


def test_players_absent_from_the_holdout_season_score_zero_not_dropped():
    """Drafting someone who never played again is a real cost, not a freebie.

    Dropping them would silently flatter whichever team drafted more of them.
    """
    roster = [_rostered("played"), _rostered("retired")]
    result = score_roster_on_holdout(roster, {"played": 100.0})
    assert result["total"] == 100.0 + MISSING_PLAYER_POINTS
    assert result["missing"] == 1
    assert result["scored"] == 1


def test_a_roster_of_entirely_absent_players_is_worth_almost_nothing():
    roster = [_rostered(str(i)) for i in range(5)]
    result = score_roster_on_holdout(roster, {})
    assert result["total"] == 5 * MISSING_PLAYER_POINTS
    assert result["missing"] == 5


def test_empty_roster_scores_zero():
    result = score_roster_on_holdout([], {"a": 100.0})
    assert result["total"] == 0.0
    assert result["scored"] == 0


def test_per_player_breakdown_is_returned_for_inspection():
    roster = [_rostered("a", "Real Player", "WR")]
    result = score_roster_on_holdout(roster, {"a": 42.5})
    assert result["players"] == [{"name": "Real Player", "position": "WR", "holdout_points": 42.5}]


def test_scoring_joins_on_player_id_not_name():
    """Cross-season joins must use the stable nflverse id, never a display name."""
    roster = [_rostered("stable-id", name="Name Changed Since")]
    result = score_roster_on_holdout(roster, {"stable-id": 77.0})
    assert result["total"] == 77.0


def test_missing_points_constant_is_zero_not_a_sentinel():
    # A non-zero sentinel would silently distort every holdout total.
    assert MISSING_PLAYER_POINTS == 0.0
