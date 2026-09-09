"""CBB slice of the multi-sport availability overlay: the ESPN fetcher and the
``fantasy.multi_sport_status`` helper. The network is never touched.

Engine integration (``modules.cbb_prop_model`` / ``modules.cbb_moneyline_model``)
is covered in ``UniversalQuantAgent/tests/test_multi_sport_ui.py``.
"""

from __future__ import annotations

import json

import pytest

from fantasy import multi_sport_status as mss
from fantasy.online import player_status_fetcher_cbb as fetcher

ESPN_PAYLOAD = {
    "status": "success",
    "injuries": [
        {
            "displayName": "Duke Blue Devils",
            "abbreviation": "DUKE",
            "injuries": [
                {"status": "Out", "athlete": {"displayName": "Star Wing", "id": "6001"}},
                {"status": "Day-To-Day", "athlete": {"displayName": "Backup Guard", "id": "6002"}},
                {"status": "Active", "athlete": {"displayName": "Healthy Center", "id": "6003"}},
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


def test_fetcher_points_at_the_cbb_injuries_endpoint():
    assert fetcher.SPORT == "cbb"
    assert fetcher.ESPN_INJURIES_URL.endswith("/basketball/mens-college-basketball/injuries")


def test_normalize_feed_maps_statuses():
    mapping = fetcher.normalize_feed(ESPN_PAYLOAD, now="2026-09-05T18:00:00Z")
    assert mapping["nm:starwing"]["status"] == "OUT"
    assert mapping["nm:backupguard"]["status"] == "QUESTIONABLE"
    assert not any("healthycenter" in key for key in mapping)


def test_refresh_writes_only_the_cbb_slice(status_file):
    status_file.write_text(json.dumps({"mlb": {"nm:x": {"status": "OUT", "name": "X"}}}), encoding="utf-8")
    mss.clear_cache()
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    assert result["ok"] and result["sport"] == "cbb" and result["count"] == 2
    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {"mlb", "cbb"}


def test_refresh_degrades_gracefully(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    before = status_file.read_text(encoding="utf-8")
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert result["ok"] is False and status_file.read_text(encoding="utf-8") == before


def test_helper_reads_the_cbb_slice_only(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    mss.clear_cache()
    assert mss.live_status("cbb", {"player_name": "Star Wing"}) == "OUT"
    assert mss.live_status("cbb", {"player_name": "Backup Guard"}) == "QUESTIONABLE"
    assert mss.has_status_data("nba") is False
    # the shared canonical maths comes straight from fantasy.player_status
    assert mss.adjust_projection_for_status(10.0, "OUT") == 0.0
    assert mss.adjust_projection_for_status(10.0, "QUESTIONABLE") == 8.8
