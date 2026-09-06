"""CFB slice of the multi-sport availability overlay: the ESPN fetcher and the
``fantasy.multi_sport_status`` helper. The network is never touched.

Engine integration (``modules.cfb_prop_model`` / ``modules.cfb_moneyline_model``)
is covered in ``UniversalQuantAgent/tests/test_multi_sport_ui.py``.
"""

from __future__ import annotations

import json

import pytest

from fantasy import multi_sport_status as mss
from fantasy.online import player_status_fetcher_cfb as fetcher

ESPN_PAYLOAD = {
    "status": "success",
    "injuries": [
        {
            "displayName": "Georgia Bulldogs",
            "abbreviation": "UGA",
            "injuries": [
                {"status": "Out", "athlete": {"displayName": "Star Quarterback", "id": "5001"}},
                {"status": "Questionable", "athlete": {"displayName": "Lead Back", "id": "5002"}},
            ],
        },
        # a flat row -- college feeds sometimes come back ungrouped
        {"player_name": "Suspended Corner", "team": "ALA", "status": "suspension"},
    ],
}


@pytest.fixture()
def status_file(tmp_path, monkeypatch):
    path = tmp_path / "multi_sport_status.json"
    monkeypatch.setattr(mss, "STATUS_PATH", path)
    mss.clear_cache()
    yield path
    mss.clear_cache()


def test_fetcher_points_at_the_cfb_injuries_endpoint():
    assert fetcher.SPORT == "cfb"
    assert fetcher.ESPN_INJURIES_URL.endswith("/football/college-football/injuries")


def test_normalize_feed_handles_grouped_and_flat_rows():
    mapping = fetcher.normalize_feed(ESPN_PAYLOAD, now="2026-09-05T18:00:00Z")
    assert mapping["nm:starquarterback"]["status"] == "OUT"
    assert mapping["nm:leadback"]["status"] == "QUESTIONABLE"
    assert mapping["nm:suspendedcorner"]["status"] == "SUSPENDED"
    assert mapping["nm:suspendedcorner"]["team"] == "ALA"
    assert fetcher.normalize_feed([]) == {}


def test_refresh_writes_only_the_cfb_slice(status_file):
    status_file.write_text(json.dumps({"nba": {"nm:x": {"status": "OUT", "name": "X"}}}), encoding="utf-8")
    mss.clear_cache()
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    assert result["ok"] and result["sport"] == "cfb" and result["count"] == 3
    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {"nba", "cfb"}


def test_refresh_degrades_gracefully(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    before = status_file.read_text(encoding="utf-8")
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert result["ok"] is False and status_file.read_text(encoding="utf-8") == before


def test_helper_reads_the_cfb_slice_only(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    mss.clear_cache()
    assert mss.live_status("cfb", {"player_name": "Star Quarterback"}) == "OUT"
    assert mss.is_unavailable("cfb", {"player_name": "Suspended Corner"}) is True
    assert mss.live_status("cfb", {"player_name": "Lead Back"}) == "QUESTIONABLE"
    assert mss.has_status_data("cbb") is False
