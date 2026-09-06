"""NHL slice of the multi-sport availability overlay: the ESPN fetcher and the
``fantasy.multi_sport_status`` helper. The network is never touched.

Engine integration (``modules.nhl_prop_model`` / ``modules.nhl_moneyline_model``)
is covered in ``UniversalQuantAgent/tests/test_multi_sport_ui.py``.
"""

from __future__ import annotations

import json

import pytest

from fantasy import multi_sport_status as mss
from fantasy.online import player_status_fetcher_nhl as fetcher

ESPN_PAYLOAD = {
    "status": "success",
    "injuries": [
        {
            "displayName": "Colorado Avalanche",
            "abbreviation": "COL",
            "injuries": [
                {"status": "Out", "athlete": {"displayName": "Gabriel Landeskog", "id": "5160"}},
                {"status": "Day-To-Day", "athlete": {"displayName": "Valeri Nichushkin", "id": "3900"}},
                {"status": "Active", "athlete": {"displayName": "Nathan MacKinnon", "id": "3041"}},
            ],
        },
        {
            "displayName": "Anaheim Ducks",
            "abbreviation": "ANA",
            "injuries": [
                {"status": "Injured Reserve", "athlete": {"displayName": "Injured Winger", "id": "77"}},
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


def test_fetcher_points_at_the_nhl_injuries_endpoint():
    assert fetcher.SPORT == "nhl"
    assert fetcher.ESPN_INJURIES_URL.endswith("/hockey/nhl/injuries")


def test_normalize_feed_maps_statuses_and_dual_keys():
    mapping = fetcher.normalize_feed(ESPN_PAYLOAD, now="2026-09-05T18:00:00Z")
    assert mapping["nm:gabriellandeskog"]["status"] == "OUT"
    assert mapping["id:5160"]["status"] == "OUT"
    assert mapping["nm:valerinichushkin"]["status"] == "QUESTIONABLE"
    assert mapping["nm:injuredwinger"]["status"] == "OUT"  # "Injured Reserve" -> OUT
    assert mapping["nm:injuredwinger"]["team"] == "ANA"
    assert not any("mackinnon" in key for key in mapping)  # Active dropped
    assert fetcher.normalize_feed({"garbage": 1}) == {}


def test_refresh_writes_only_the_nhl_slice(status_file):
    status_file.write_text(json.dumps({"mlb": {"nm:x": {"status": "OUT", "name": "X"}}}), encoding="utf-8")
    mss.clear_cache()
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    assert result["ok"] and result["sport"] == "nhl" and result["count"] == 3
    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {"mlb", "nhl"}
    assert on_disk["mlb"]["nm:x"]["status"] == "OUT"  # untouched


def test_refresh_degrades_gracefully(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    before = status_file.read_text(encoding="utf-8")
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert result["ok"] is False and "kept the existing" in result["error"]
    assert status_file.read_text(encoding="utf-8") == before


def test_helper_reads_the_nhl_slice_only(status_file):
    assert mss.has_status_data("nhl") is False
    fetcher.refresh_player_status(status_file, fetch=lambda _url: ESPN_PAYLOAD)
    mss.clear_cache()
    assert mss.has_status_data("nhl") is True
    assert mss.has_status_data("mlb") is False
    assert mss.live_status("nhl", {"player_name": "Gabriel Landeskog"}) == "OUT"
    assert mss.is_unavailable("nhl", {"player_name": "Gabriel Landeskog"}) is True
    assert mss.live_status("nhl", {"player_name": "Nathan MacKinnon"}) == "HEALTHY"
    assert mss.live_status("mlb", {"player_name": "Gabriel Landeskog"}) == "HEALTHY"  # wrong sport
