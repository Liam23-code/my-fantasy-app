"""MLB slice of the multi-sport availability overlay: the ESPN fetcher and the
``fantasy.multi_sport_status`` helper. The network is never touched -- every
test injects a fake ``fetch``.

The engine-integration side (``modules.mlb_prop_model`` /
``modules.mlb_moneyline_model`` honouring this overlay) lives in the app test
suite, ``UniversalQuantAgent/tests/test_multi_sport_ui.py``, since those modules
are not importable from the engine package.
"""

from __future__ import annotations

import json

import pytest

from fantasy import multi_sport_status as mss
from fantasy.online import player_status_fetcher_mlb as fetcher

# A trimmed ESPN baseball/mlb/injuries payload: one IL pitcher with an id in a
# link, one day-to-day bat with an explicit id, one active (dropped) player,
# one suspended player on another team, one junk node.
ESPN_PAYLOAD = {
    "timestamp": "2026-09-05T23:00Z",
    "status": "success",
    "injuries": [
        {
            "displayName": "Arizona Diamondbacks",
            "abbreviation": "ARI",
            "injuries": [
                {
                    "status": "60-Day-IL",
                    "athlete": {
                        "displayName": "Zac Gallen",
                        "links": [{"href": "https://www.espn.com/mlb/player/_/id/39910/zac-gallen"}],
                    },
                },
                {"status": "Day-To-Day", "athlete": {"displayName": "Ketel Marte", "id": "33288"}},
                {"status": "Active", "athlete": {"displayName": "Corbin Carroll", "id": "42403"}},
            ],
        },
        {
            "displayName": "Los Angeles Dodgers",
            "abbreviation": "LAD",
            "injuries": [
                {"status": "Suspension", "athlete": {"displayName": "Banned Bat", "id": "999"}},
            ],
        },
        "not-a-dict",
    ],
}


@pytest.fixture()
def status_file(tmp_path, monkeypatch):
    path = tmp_path / "multi_sport_status.json"
    monkeypatch.setattr(mss, "STATUS_PATH", path)
    mss.clear_cache()
    yield path
    mss.clear_cache()


# --- normalize_feed ---------------------------------------------------------


def test_normalize_feed_maps_espn_statuses_and_dual_keys():
    mapping = fetcher.normalize_feed(ESPN_PAYLOAD, now="2026-09-05T18:00:00Z")
    assert mapping["nm:zacgallen"] == {
        "status": "OUT",
        "last_updated": "2026-09-05T18:00:00Z",
        "source": "espn",
        "name": "Zac Gallen",
        "team": "ARI",
    }
    # id pulled out of the player-card link
    assert mapping["id:39910"]["status"] == "OUT"
    # day-to-day -> QUESTIONABLE, explicit athlete id
    assert mapping["nm:ketelmarte"]["status"] == "QUESTIONABLE"
    assert mapping["id:33288"]["status"] == "QUESTIONABLE"
    # suspension -> SUSPENDED
    assert mapping["nm:bannedbat"]["status"] == "SUSPENDED"
    assert mapping["nm:bannedbat"]["team"] == "LAD"
    # active player and junk node are gone
    assert not any("carroll" in key for key in mapping)


def test_normalize_feed_accepts_a_flat_list_and_rejects_junk():
    flat = [
        {"player_name": "Flat Guy", "team": "SEA", "status": "out"},
        {"name": "", "status": "out"},  # no name -> skipped
    ]
    mapping = fetcher.normalize_feed(flat, now="2026-09-05T18:00:00Z")
    assert mapping == {
        "nm:flatguy": {
            "status": "OUT",
            "last_updated": "2026-09-05T18:00:00Z",
            "source": "espn",
            "name": "Flat Guy",
            "team": "SEA",
        }
    }
    assert fetcher.normalize_feed(None) == {}
    assert fetcher.normalize_feed("nope") == {}
    assert fetcher.normalize_feed(123) == {}


# --- refresh_player_status ------------------------------------------------------


def test_refresh_writes_only_the_mlb_slice_and_reports_a_summary(status_file):
    # seed an unrelated NHL slice that must survive an MLB refresh
    status_file.write_text(
        json.dumps({"nhl": {"nm:someskater": {"status": "OUT", "name": "Some Skater", "source": "espn"}}}),
        encoding="utf-8",
    )
    mss.clear_cache()

    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    assert result["ok"] is True
    assert result["written"] is True
    assert result["sport"] == "mlb"
    assert result["count"] == 3  # Gallen, Marte, Banned Bat (nm/id keys collapse by name)
    assert result["source"] == "espn"

    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {"mlb", "nhl"}
    assert on_disk["mlb"]["nm:zacgallen"]["status"] == "OUT"
    assert on_disk["nhl"]["nm:someskater"]["status"] == "OUT"  # untouched


def test_refresh_keeps_the_old_file_byte_for_byte_when_the_fetch_fails(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    before = status_file.read_text(encoding="utf-8")

    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert result["ok"] is False
    assert result["written"] is False
    assert "kept the existing" in result["error"]
    assert result["count"] == 3  # counted from the slice left in place
    assert status_file.read_text(encoding="utf-8") == before


def test_refresh_creates_an_empty_file_if_missing_even_on_failure(status_file):
    assert not status_file.exists()
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert status_file.exists()
    assert json.loads(status_file.read_text(encoding="utf-8")) == {}
    assert result["ok"] is False


def test_refresh_re_reads_under_the_lock_so_a_concurrent_slice_is_not_lost(status_file):
    """The read-merge-write is locked and the read happens *after* the fetch, so
    another sport's slice written while this refresh's feed was in flight is
    carried forward rather than reverted."""
    status_file.write_text("{}", encoding="utf-8")
    mss.clear_cache()

    def _fetch_and_race(_url):
        # simulate a concurrent NHL refresh landing mid-flight
        raw = json.loads(status_file.read_text(encoding="utf-8"))
        raw["nhl"] = {"nm:raceskater": {"status": "OUT", "name": "Race Skater", "source": "espn"}}
        status_file.write_text(json.dumps(raw), encoding="utf-8")
        return ESPN_PAYLOAD

    result = fetcher.refresh_player_status(status_file, fetch=_fetch_and_race)
    assert result["ok"] is True
    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {"mlb", "nhl"}
    assert on_disk["nhl"]["nm:raceskater"]["status"] == "OUT"  # not lost
    assert on_disk["mlb"]["nm:zacgallen"]["status"] == "OUT"


# --- the helper reads --------------------------------------------------------


def test_helper_is_a_no_op_until_the_mlb_slice_has_data(status_file):
    assert mss.has_status_data("mlb") is False
    assert mss.live_status("mlb", {"player_name": "Zac Gallen"}) == "HEALTHY"
    assert mss.is_out("mlb", {"player_name": "Zac Gallen"}) is False


def test_helper_live_and_effective_status(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    mss.clear_cache()

    assert mss.has_status_data("mlb") is True
    assert mss.has_status_data("nhl") is False
    assert mss.live_status("mlb", {"player_name": "Zac Gallen"}) == "OUT"
    assert mss.live_status("mlb", "Ketel Marte") == "QUESTIONABLE"
    assert mss.live_status("mlb", {"player_name": "Nobody Here"}) == "HEALTHY"

    assert mss.is_out("mlb", {"player_name": "Zac Gallen"}) is True
    assert mss.is_unavailable("mlb", {"player_name": "Banned Bat"}) is True
    assert mss.is_suspended("mlb", {"player_name": "Banned Bat"}) is True

    # effective_status falls back to a row's own field only when the feed is silent
    assert mss.effective_status("mlb", {"player_name": "Nobody", "status": "questionable"}) == "QUESTIONABLE"
    assert mss.effective_status("mlb", {"player_name": "Zac Gallen", "status": "healthy"}) == "OUT"


def test_team_status_penalty_is_opt_in_and_empty_without_a_roster_or_data(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    mss.clear_cache()
    assert mss.team_status_penalty("mlb", None) == {}
    roster = {"ARI": [{"player_name": "Zac Gallen", "position": "SP"}, {"player_name": "Ketel Marte", "position": "2B"}]}
    penalty = mss.team_status_penalty("mlb", roster, points_by_position={"SP": 4.0, "2B": 1.0})
    # SP OUT (4.0 * 1.0) + 2B questionable (1.0 * 0.15)
    assert penalty == {"ARI": pytest.approx(4.15)}
    assert mss.team_status_penalty("nhl", roster, points_by_position={"SP": 4.0}) == {}  # no NHL data
