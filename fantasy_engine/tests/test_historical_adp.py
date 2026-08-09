"""Tests for fantasy.historical_adp: real pre-season snapshots, not simulations."""

from __future__ import annotations

from fantasy.historical_adp import (
    VERIFIED_SEASONS,
    apply_historical_adp,
    fetch_historical_adp,
    historical_adp_coverage,
)


def _fake_ffc(monkeypatch, records):
    monkeypatch.setattr("fantasy.historical_adp.fetch_ffcalculator_adp", lambda **kwargs: records)


def _captured_ffc(monkeypatch, captured, records):
    def _fetch(**kwargs):
        captured.update(kwargs)
        return records

    monkeypatch.setattr("fantasy.historical_adp.fetch_ffcalculator_adp", _fetch)


def _player(name, position="RB", adp=50.0):
    return {"player_id": f"id-{name}", "name": name, "position": position, "adp": adp}


def setup_function():
    """Each test starts with a cold memo so monkeypatched fetches actually run."""
    from fantasy import historical_adp

    historical_adp._MEMO.clear()


# --- fetching -----------------------------------------------------------------


def test_fetch_requests_the_requested_season(monkeypatch):
    captured: dict = {}
    _captured_ffc(monkeypatch, captured, [{"name": "X", "adp": 3.0, "stdev": 1.0, "high": 1, "low": 6}])
    fetch_historical_adp(2024)
    assert captured["season"] == 2024


def test_fetch_keys_by_normalized_name_not_ffc_player_id(monkeypatch):
    _fake_ffc(monkeypatch, [{"name": "Ja'Marr Chase", "adp": 3.0, "stdev": 1.0, "high": 1, "low": 6}])
    table = fetch_historical_adp(2024)
    assert "jamarrchase" in table


def test_fetch_returns_empty_when_the_season_has_no_data(monkeypatch):
    _fake_ffc(monkeypatch, [])
    assert fetch_historical_adp(1999) == {}


def test_fetch_skips_records_missing_a_name_or_adp(monkeypatch):
    _fake_ffc(monkeypatch, [{"name": "NoADP"}, {"adp": 5.0}])
    assert fetch_historical_adp(2024) == {}


def test_verified_seasons_are_documented():
    assert set(VERIFIED_SEASONS) == {2023, 2024, 2025}


# --- applying to a pool --------------------------------------------------------


def test_apply_replaces_present_day_adp_and_preserves_the_original(monkeypatch):
    _fake_ffc(monkeypatch, [{"name": "Star", "adp": 4.0, "stdev": 1.0, "high": 1, "low": 8}])
    updated = apply_historical_adp([_player("Star", adp=99.0)], season=2025)
    assert updated[0]["adp"] == 4.0
    assert updated[0]["adp_present_day"] == 99.0
    assert updated[0]["adp_season"] == 2025


def test_players_absent_from_the_snapshot_get_no_adp_rather_than_an_invented_one(monkeypatch):
    """They genuinely were not on that year's board -- inventing a number would be fabrication."""
    _fake_ffc(monkeypatch, [{"name": "Known", "adp": 4.0, "stdev": 1.0, "high": 1, "low": 8}])
    updated = apply_historical_adp([_player("Unknown", adp=50.0)], season=2025)
    assert updated[0]["adp"] is None
    assert updated[0]["adp_present_day"] == 50.0


def test_apply_does_not_mutate_the_input(monkeypatch):
    _fake_ffc(monkeypatch, [{"name": "Star", "adp": 4.0, "stdev": 1.0, "high": 1, "low": 8}])
    players = [_player("Star", adp=99.0)]
    apply_historical_adp(players, season=2025)
    assert players[0]["adp"] == 99.0
    assert "adp_present_day" not in players[0]


def test_apply_passes_the_pool_through_untouched_when_no_snapshot_exists(monkeypatch):
    _fake_ffc(monkeypatch, [])
    players = [_player("Star", adp=99.0)]
    updated = apply_historical_adp(players, season=1999)
    assert updated[0]["adp"] == 99.0  # unchanged, not blanked


def test_name_matching_survives_punctuation_and_suffix_differences(monkeypatch):
    _fake_ffc(monkeypatch, [{"name": "Marvin Harrison Jr.", "adp": 20.0, "stdev": 2.0, "high": 10, "low": 30}])
    updated = apply_historical_adp([_player("Marvin Harrison")], season=2025)
    assert updated[0]["adp"] == 20.0


# --- coverage reporting ---------------------------------------------------------


def test_coverage_reports_how_much_of_the_pool_the_snapshot_covers(monkeypatch):
    _fake_ffc(monkeypatch, [{"name": "A", "adp": 1.0, "stdev": 1.0, "high": 1, "low": 2}])
    coverage = historical_adp_coverage([_player("A"), _player("B")], 2025)
    assert coverage["matched"] == 1
    assert coverage["pool_size"] == 2
    assert coverage["coverage"] == 0.5


def test_coverage_of_an_empty_pool_is_zero_not_a_crash(monkeypatch):
    _fake_ffc(monkeypatch, [])
    assert historical_adp_coverage([], 2025)["coverage"] == 0.0
