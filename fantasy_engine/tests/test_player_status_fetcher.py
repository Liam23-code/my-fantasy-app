"""Tests for fantasy.online.player_status_fetcher -- the one online touch point.

The network is never hit: every test injects a fake fetch or monkeypatches
``urllib.request.urlopen`` (the same seam fantasy.draft_ingestion uses).
"""

from __future__ import annotations

import json

import pytest

from fantasy.online import player_status_fetcher as fetcher
from fantasy.player_status import clear_cache

# A trimmed Sleeper /v1/players/nfl payload: one OUT with a gsis id, one
# doubtful without one, one healthy (dropped), one retired free agent
# (dropped -- no team), one junk row.
SLEEPER_PAYLOAD = {
    "4034": {"full_name": "Puka Nacua", "gsis_id": "00-0039075", "injury_status": "Out", "team": "LA", "active": True},
    "9999": {"first_name": "No", "last_name": "Gsis", "injury_status": "Doubtful", "team": "SF", "active": True},
    "1": {"full_name": "Healthy Harry", "gsis_id": "00-0000001", "injury_status": None, "team": "KC", "active": True},
    "2": {"full_name": "Retired Legend", "gsis_id": "00-0000002", "injury_status": "IR", "team": None, "active": False},
    "bad": "not-a-dict",
}


@pytest.fixture()
def status_file(tmp_path):
    path = tmp_path / "player_status.json"
    clear_cache()
    yield path
    clear_cache()


def test_normalize_feed_keeps_only_flagged_players_and_dual_keys():
    mapping = fetcher.normalize_feed(SLEEPER_PAYLOAD, now="2026-08-31T18:00:00Z")
    assert mapping["00-0039075"] == {
        "status": "OUT",
        "last_updated": "2026-08-31T18:00:00Z",
        "source": "sleeper",
        "name": "Puka Nacua",
    }
    assert mapping["nm:pukanacua"]["status"] == "OUT"
    assert mapping["nm:nogsis"]["status"] == "DOUBTFUL"
    # Healthy, retired-free-agent, and junk rows are all gone.
    assert "00-0000001" not in mapping
    assert "00-0000002" not in mapping
    assert not any("healthyharry" in key or "retiredlegend" in key for key in mapping)


def test_normalize_feed_returns_empty_on_junk():
    assert fetcher.normalize_feed(None) == {}
    assert fetcher.normalize_feed([1, 2, 3]) == {}
    assert fetcher.normalize_feed("nope") == {}


def test_refresh_writes_the_file_and_reports_a_summary(status_file):
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: SLEEPER_PAYLOAD)
    assert result["ok"] is True
    assert result["written"] is True
    assert result["count"] == 2  # two flagged players (nm: keys not double counted)
    assert result["source"] == "sleeper"
    assert result["error"] is None
    on_disk = json.loads(status_file.read_text(encoding="utf-8"))
    assert on_disk["00-0039075"]["status"] == "OUT"


def test_refresh_keeps_the_old_file_when_the_fetch_fails(status_file):
    fetcher.refresh_player_status(status_file, fetch=lambda _url: SLEEPER_PAYLOAD)
    before = status_file.read_text(encoding="utf-8")

    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert result["ok"] is False
    assert result["written"] is False
    assert "kept the existing" in result["error"]
    assert result["count"] == 2  # counted from the file that was left in place
    assert status_file.read_text(encoding="utf-8") == before  # untouched


def test_refresh_creates_the_file_if_missing_even_on_failure(status_file):
    assert not status_file.exists()
    result = fetcher.refresh_player_status(status_file, fetch=lambda _url: None)
    assert status_file.exists()
    assert json.loads(status_file.read_text(encoding="utf-8")) == {}
    assert result["ok"] is False


def test_ensure_status_file_is_idempotent(status_file):
    fetcher.ensure_status_file(status_file)
    fetcher.ensure_status_file(status_file)
    assert json.loads(status_file.read_text(encoding="utf-8")) == {}


def test_http_get_json_swallows_network_errors(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("no network in tests")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert fetcher._http_get_json("https://example.invalid/x") is None


def test_refresh_routes_through_urlopen_by_default(status_file, monkeypatch):
    class _Resp:
        def read(self):
            return json.dumps(SLEEPER_PAYLOAD).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _Resp())
    result = fetcher.refresh_player_status(status_file)
    assert result["ok"] is True
    assert result["count"] == 2
