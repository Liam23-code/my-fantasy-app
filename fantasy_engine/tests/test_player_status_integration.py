"""Tests for fantasy.player_status -- the read-only overlay helper."""

from __future__ import annotations

import json

import pytest

from fantasy import player_status as ps


@pytest.fixture()
def status_path(tmp_path, monkeypatch):
    """Point the whole module at a temp status file."""
    path = tmp_path / "player_status.json"
    monkeypatch.setattr(ps, "STATUS_PATH", path)
    ps.clear_cache()
    yield path
    ps.clear_cache()


def _write(path, mapping):
    path.write_text(json.dumps(mapping), encoding="utf-8")
    ps.clear_cache()


# --- normalize_status -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Out", "OUT"), ("out", "OUT"), ("IR", "OUT"), ("PUP", "OUT"), ("Inactive", "OUT"),
        ("Doubtful", "DOUBTFUL"),
        ("Questionable", "QUESTIONABLE"), ("GTD", "QUESTIONABLE"), ("day-to-day", "QUESTIONABLE"),
        ("Holdout", "HOLDOUT"), ("hold out", "HOLDOUT"),
        ("Sus", "SUSPENDED"), ("Suspended", "SUSPENDED"),
        (None, "HEALTHY"), ("", "HEALTHY"), ("Active", "HEALTHY"), ("nonsense", "HEALTHY"),
    ],
)
def test_normalize_status(raw, expected):
    assert ps.normalize_status(raw) == expected


# --- effective_status / flags --------------------------------------------------


def test_effective_status_prefers_the_live_feed_then_the_pool_row(status_path):
    _write(status_path, {"00-0039075": {"status": "OUT", "last_updated": "t", "source": "s"}})
    assert ps.effective_status({"player_id": "00-0039075", "injury_status": "QUESTIONABLE"}) == "OUT"
    # not in the feed -> falls back to the pool row's own field
    assert ps.effective_status({"player_id": "x", "injury_status": "Doubtful"}) == "DOUBTFUL"
    # neither -> HEALTHY
    assert ps.effective_status({"player_id": "y"}) == "HEALTHY"


def test_live_status_is_the_overlay_only_never_the_pool_row(status_path):
    """The hard rules key on live_status, so a stale pool injury_status can
    never be *escalated* by this feature -- only genuine feed data acts."""
    _write(status_path, {"00-0039075": {"status": "OUT"}})
    assert ps.live_status({"player_id": "00-0039075"}) == "OUT"  # feed says OUT
    assert ps.live_status({"player_id": "x", "injury_status": "OUT"}) == "HEALTHY"  # pool row ignored
    assert ps.is_out({"player_id": "x", "injury_status": "OUT"}) is False
    assert ps.availability_flag({"player_id": "x", "injury_status": "OUT"}) == "HEALTHY"


def test_name_fallback_key_matches_when_there_is_no_id(status_path):
    _write(status_path, {"nm:joeholdout": {"status": "HOLDOUT", "last_updated": "t", "source": "s"}})
    assert ps.is_holdout({"name": "Joe  Holdout!"}) is True
    assert ps.availability_flag({"name": "Joe Holdout"}) == "HOLDOUT"


def test_availability_flag_helpers(status_path):
    _write(
        status_path,
        {
            "a": {"status": "OUT"}, "b": {"status": "HOLDOUT"},
            "c": {"status": "SUSPENDED"}, "d": {"status": "QUESTIONABLE"},
        },
    )
    assert ps.is_out({"player_id": "a"}) and ps.is_unavailable({"player_id": "a"})
    assert ps.is_holdout({"player_id": "b"}) and ps.is_unavailable({"player_id": "b"})
    assert ps.is_suspended({"player_id": "c"}) and ps.is_unavailable({"player_id": "c"})
    assert not ps.is_unavailable({"player_id": "d"})  # questionable can still play


def test_empty_file_is_a_total_no_op(status_path):
    _write(status_path, {})
    assert ps.has_status_data() is False
    assert ps.effective_status({"player_id": "a", "injury_status": "OUT"}) == "OUT"  # pool row still read
    assert ps.live_status({"player_id": "a", "injury_status": "OUT"}) == "HEALTHY"  # nothing to overlay
    assert ps.is_out({"player_id": "a", "injury_status": "OUT"}) is False
    assert ps.overlay_pool_status([{"player_id": "a", "projection": 100.0}]) == [
        {"player_id": "a", "projection": 100.0}
    ]


# --- projection / score adjustment -----------------------------------------


def test_adjust_projection_for_status():
    assert ps.adjust_projection_for_status(200, "OUT") == 0.0
    assert ps.adjust_projection_for_status(200, "HOLDOUT") == 0.0
    assert ps.adjust_projection_for_status(200, "SUSPENDED") == 0.0
    assert ps.adjust_projection_for_status(200, "DOUBTFUL") == 90.0
    assert ps.adjust_projection_for_status(200, "QUESTIONABLE") == 176.0
    assert ps.adjust_projection_for_status(200, "HEALTHY") == 200.0


def test_adjust_score_for_status_is_sign_safe():
    # positive score: a downweight lowers it
    assert ps.adjust_score_for_status(10, "DOUBTFUL") == 6.0
    assert 0 < ps.adjust_score_for_status(10, "QUESTIONABLE") < 10
    # negative score: a downweight must make it *more* negative, never toward zero
    assert ps.adjust_score_for_status(-10, "DOUBTFUL") < -10
    assert ps.adjust_score_for_status(-10, "QUESTIONABLE") < -10
    # OUT zeroes it either way; HEALTHY leaves it alone
    assert ps.adjust_score_for_status(-10, "OUT") == 0.0
    assert ps.adjust_score_for_status(-10, "HEALTHY") == -10.0


# --- overlay_pool_status --------------------------------------------------------


def test_overlay_zeroes_cannot_play_and_canonicalizes_the_rest(status_path):
    _write(
        status_path,
        {
            "hold": {"status": "HOLDOUT"},
            "doubt": {"status": "DOUBTFUL"},
            "fine": {"status": "HEALTHY"},
        },
    )
    pool = [
        {"player_id": "hold", "projection": 300.0, "expected_fantasy_points": 300.0},
        {"player_id": "doubt", "projection": 250.0, "injury_status": None},
        {"player_id": "fine", "projection": 200.0},
    ]
    out = ps.overlay_pool_status(pool)
    by_id = {row["player_id"]: row for row in out}
    assert by_id["hold"]["projection"] == 0.0 and by_id["hold"]["injury_status"] == "HOLDOUT"
    assert by_id["doubt"]["projection"] == 250.0 and by_id["doubt"]["injury_status"] == "DOUBTFUL"
    assert by_id["fine"]["projection"] == 200.0
    # original list is not mutated
    assert pool[0]["projection"] == 300.0


def test_status_last_updated_and_flagged_count(status_path):
    assert ps.status_last_updated() is None
    _write(
        status_path,
        {
            "00-1": {"status": "OUT", "last_updated": "2026-08-30T00:00:00Z", "name": "Player AA"},
            "nm:playeraa": {"status": "OUT", "last_updated": "2026-08-30T00:00:00Z", "name": "Player AA"},
            "00-2": {"status": "DOUBTFUL", "last_updated": "2026-08-31T12:00:00Z", "name": "Player BB"},
        },
    )
    assert ps.status_last_updated() == "2026-08-31T12:00:00Z"
    assert ps.flagged_count() == 2  # the id key and its nm: fallback collapse via name
    assert set(ps.load_player_status()) == {"00-1", "nm:playeraa", "00-2"}
