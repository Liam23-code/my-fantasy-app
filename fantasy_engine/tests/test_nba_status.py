"""NBA slice of the multi-sport availability overlay: the ESPN fetcher and the
``fantasy.multi_sport_status`` helper. The network is never touched.

Engine integration (``modules.nba_prop_model`` / ``modules.nba_moneyline_model``)
is covered in ``UniversalQuantAgent/tests/test_multi_sport_ui.py``.
"""

from __future__ import annotations

import json

import pytest

from fantasy import multi_sport_status as mss
from fantasy.online import player_status_fetcher_nba as fetcher

ESPN_PAYLOAD = {
    "status": "success",
    "injuries": [
        {
            "displayName": "Denver Nuggets",
            "abbreviation": "DEN",
            "injuries": [
                {"status": "Out", "athlete": {"displayName": "Jamal Murray", "id": "3936299"}},
                {"status": "Questionable", "athlete": {"displayName": "Aaron Gordon", "id": "4278"}},
                {"status": "Active", "athlete": {"displayName": "Nikola Jokic", "id": "3112335"}},
            ],
        },
        {
            "displayName": "Boston Celtics",
            "abbreviation": "BOS",
            "injuries": [
                {"status": "Doubtful", "athlete": {"displayName": "Jaylen Brown", "id": "3917376"}},
            ],
        },
    ],
}


@pytest.fixture()
def status_file(tmp_path, monkeypatch):
    path = tmp_path / "multi_sport_status.json"
    monkeypatch.setattr(mss, "STATUS_PATH", path)
    mss.clear_cache()
    yield path
    mss.clear_cache()


def test_fetcher_points_at_the_nba_injuries_endpoint():
    assert fetcher.SPORT == "nba"
    assert fetcher.ESPN_INJURIES_URL.endswith("/basketball/nba/injuries")


def test_normalize_feed_maps_statuses_and_dual_keys():
    mapping = fetcher.normalize_feed(ESPN_PAYLOAD, now="2026-09-05T18:00:00Z")
    assert mapping["nm:jamalmurray"]["status"] == "OUT"
    assert mapping["id:3936299"]["status"] == "OUT"
    assert mapping["nm:aarongordon"]["status"] == "QUESTIONABLE"
    assert mapping["nm:jaylenbrown"]["status"] == "DOUBTFUL"
    assert mapping["nm:jaylenbrown"]["team"] == "BOS"
    assert not any("jokic" in key for key in mapping)


def test_refresh_writes_only_the_nba_slice(status_file):
    status_file.write_text(json.dumps({"cfb": {"nm:x": {"status": "OUT", "name": "X"}}}), encoding="utf-8")
    mss.clear_cache()
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    assert result["ok"] and result["sport"] == "nba" and result["count"] == 3
    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {"cfb", "nba"}
    assert on_disk["cfb"]["nm:x"]["status"] == "OUT"


def test_refresh_degrades_gracefully(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    before = status_file.read_text(encoding="utf-8")
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert result["ok"] is False and status_file.read_text(encoding="utf-8") == before


def test_helper_reads_the_nba_slice_only(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    mss.clear_cache()
    assert mss.live_status("nba", {"player": "Jamal Murray"}) == "OUT"  # NBA rows carry "player"
    assert mss.is_out("nba", {"player": "Jamal Murray"}) is True
    assert mss.live_status("nba", {"player": "Aaron Gordon"}) == "QUESTIONABLE"
    assert mss.live_status("nba", {"player": "Nikola Jokic"}) == "HEALTHY"
    assert mss.has_status_data("cbb") is False
