"""Tests for fantasy.draft_ingestion: one real live source, six honest stubs/parsers."""

from __future__ import annotations

import json
import urllib.error

from fantasy.draft_ingestion.ffcalculator import fetch_ffcalculator_adp
from fantasy.draft_ingestion.ffpc import parse_ffpc_draft_board_html
from fantasy.draft_ingestion.fp_analyzer import parse_fp_analyzer_csv
from fantasy.draft_ingestion.github_datasets import load_github_dataset
from fantasy.draft_ingestion.reddit_dumps import load_reddit_dump
from fantasy.draft_ingestion.sleeper_api import fetch_sleeper_adp, fetch_sleeper_draft_boards
from fantasy.draft_ingestion.underdog import parse_underdog_draft_json

# --- ffcalculator: the one real, live source --------------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_fetch_ffcalculator_adp_parses_a_successful_response(monkeypatch):
    payload = {
        "status": "Success",
        "meta": {"total_drafts": 100},
        "players": [{"player_id": 1, "name": "X", "position": "RB", "adp": 5.0, "high": 1, "low": 10, "stdev": 2.0}],
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    records = fetch_ffcalculator_adp()
    assert records == payload["players"]


def test_fetch_ffcalculator_adp_returns_empty_on_error_status(monkeypatch):
    payload = {"status": "Error", "errors": ["No ADP data found."]}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(payload))
    assert fetch_ffcalculator_adp() == []


def test_fetch_ffcalculator_adp_returns_empty_on_network_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    assert fetch_ffcalculator_adp() == []


def test_fetch_ffcalculator_adp_returns_empty_on_malformed_json(monkeypatch):
    class _BadResponse:
        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _BadResponse())
    assert fetch_ffcalculator_adp() == []


def test_fetch_ffcalculator_adp_sends_a_user_agent_header(monkeypatch):
    """This host returns HTTP 403 without one -- verified live this session."""
    captured = {}

    def _capture(request, timeout=None):
        captured["headers"] = request.headers
        return _FakeResponse({"status": "Success", "players": []})

    monkeypatch.setattr("urllib.request.urlopen", _capture)
    fetch_ffcalculator_adp()
    assert "User-agent" in captured["headers"]


# --- sleeper_api: confirmed no real data -------------------------------------


def test_sleeper_always_returns_empty():
    assert fetch_sleeper_draft_boards() == []
    assert fetch_sleeper_adp() == []


# --- fp_analyzer: real CSV parser --------------------------------------------


def test_fp_analyzer_parses_standard_columns():
    csv_text = "Player,Position,Team,ADP\nJahmyr Gibbs,RB,DET,3.3\n"
    records = parse_fp_analyzer_csv(csv_text)
    assert records == [{"name": "Jahmyr Gibbs", "position": "RB", "team": "DET", "adp": 3.3}]


def test_fp_analyzer_is_case_insensitive_and_tolerates_column_variants():
    csv_text = "player name,pos,tm,avg pick\nX,WR,SF,10.5\n"
    records = parse_fp_analyzer_csv(csv_text)
    assert records[0]["name"] == "X"
    assert records[0]["adp"] == 10.5


def test_fp_analyzer_returns_empty_without_name_or_adp_columns():
    assert parse_fp_analyzer_csv("Foo,Bar\n1,2\n") == []


def test_fp_analyzer_returns_empty_for_blank_input():
    assert parse_fp_analyzer_csv("") == []
    assert parse_fp_analyzer_csv("   ") == []


def test_fp_analyzer_skips_rows_missing_a_name():
    csv_text = "Player,ADP\n,5.0\nRealPlayer,3.0\n"
    records = parse_fp_analyzer_csv(csv_text)
    assert len(records) == 1
    assert records[0]["name"] == "RealPlayer"


# --- ffpc: real HTML table parser --------------------------------------------


def test_ffpc_parses_a_simple_draft_table():
    html = "<table><tr><th>Player</th><th>Pick</th></tr><tr><td>Ja'Marr Chase</td><td>1</td></tr></table>"
    assert parse_ffpc_draft_board_html(html) == [("Ja'Marr Chase", 1)]


def test_ffpc_returns_empty_without_a_recognizable_header():
    html = "<table><tr><th>Foo</th><th>Bar</th></tr><tr><td>X</td><td>1</td></tr></table>"
    assert parse_ffpc_draft_board_html(html) == []


def test_ffpc_returns_empty_for_blank_input():
    assert parse_ffpc_draft_board_html("") == []


def test_ffpc_skips_rows_with_no_parseable_pick_number():
    html = "<table><tr><th>Player</th><th>Pick</th></tr><tr><td>X</td><td>n/a</td></tr></table>"
    assert parse_ffpc_draft_board_html(html) == []


# --- underdog / github_datasets / reddit_dumps: generic user-supplied parsers


def test_underdog_parses_a_picks_dict():
    assert parse_underdog_draft_json({"picks": [{"name": "X", "pick": 5}]}) == [("X", 5)]


def test_underdog_parses_a_bare_list():
    assert parse_underdog_draft_json([{"player": "X", "overall_pick": 7}]) == [("X", 7)]


def test_underdog_returns_empty_for_unrecognized_shape():
    assert parse_underdog_draft_json({"not_picks": []}) == []
    assert parse_underdog_draft_json("garbage") == []


def test_github_datasets_parses_csv():
    assert load_github_dataset("name,pick\nY,7\n", fmt="csv") == [("Y", 7)]


def test_github_datasets_parses_json():
    assert load_github_dataset('[{"name": "Y", "pick": 7}]', fmt="json") == [("Y", 7)]


def test_github_datasets_rejects_unsupported_format():
    assert load_github_dataset("anything", fmt="xml") == []


def test_github_datasets_returns_empty_for_blank_input():
    assert load_github_dataset("", fmt="csv") == []


def test_reddit_dumps_delegates_to_github_datasets():
    assert load_reddit_dump('[{"name": "Z", "pick": 9}]', fmt="json") == [("Z", 9)]


# --- ffcalculator synthetic board reconstruction -----------------------------


def _ffc_records(n=50):
    return [
        {"name": f"P{i}", "position": "RB", "adp": float(i + 1), "high": max(1, i - 2), "low": i + 5, "stdev": 2.0}
        for i in range(n)
    ]


def test_synthesize_draft_boards_produces_valid_permutations():
    from fantasy.draft_ingestion.ffcalculator import synthesize_draft_boards

    boards = synthesize_draft_boards(_ffc_records(20), n_boards=3, seed=1)
    assert len(boards) == 3
    for board in boards:
        picks = [pick for _, pick in board]
        assert sorted(picks) == list(range(1, len(board) + 1))  # a real 1..N ordering
        assert len({name for name, _ in board}) == len(board)  # no duplicate players


def test_synthesize_draft_boards_is_reproducible_by_seed():
    from fantasy.draft_ingestion.ffcalculator import synthesize_draft_boards

    first = synthesize_draft_boards(_ffc_records(15), n_boards=2, seed=7)
    second = synthesize_draft_boards(_ffc_records(15), n_boards=2, seed=7)
    assert first == second


def test_synthesize_draft_boards_round_trips_back_to_real_adp():
    """Aggregating many synthetic boards must recover the real ADP they came from."""
    import statistics

    from fantasy.draft_ingestion.ffcalculator import synthesize_draft_boards

    records = _ffc_records(40)
    boards = synthesize_draft_boards(records, n_boards=60, seed=3)
    picks: dict[str, list[int]] = {}
    for board in boards:
        for name, pick in board:
            picks.setdefault(name, []).append(pick)
    by_name = {record["name"]: record for record in records}
    errors = [abs(statistics.mean(v) - by_name[n]["adp"]) for n, v in picks.items()]
    assert statistics.mean(errors) < 5.0  # reconstruction stays close to the real input


def test_synthesize_draft_boards_caps_at_board_size():
    """Rank-assigning every listed player stretched the tail badly (33-pick error)."""
    from fantasy.draft_ingestion.ffcalculator import synthesize_draft_boards

    boards = synthesize_draft_boards(_ffc_records(200), n_boards=1, seed=1, board_size=50)
    assert len(boards[0]) == 50


def test_synthesize_draft_boards_handles_empty_and_unusable_input():
    from fantasy.draft_ingestion.ffcalculator import synthesize_draft_boards

    assert synthesize_draft_boards([], n_boards=5) == []
    assert synthesize_draft_boards(_ffc_records(5), n_boards=0) == []
    assert synthesize_draft_boards([{"name": "NoADP"}], n_boards=2) == []


# --- unified to_draft_boards interface ----------------------------------------


def test_every_ingestion_module_exposes_to_draft_boards():
    from fantasy.draft_ingestion import ffcalculator, ffpc, fp_analyzer, github_datasets, reddit_dumps, underdog

    for module in (ffcalculator, ffpc, fp_analyzer, github_datasets, reddit_dumps, underdog):
        assert callable(module.to_draft_boards), module.__name__


def test_to_draft_boards_returns_a_list_of_boards_not_a_bare_board():
    from fantasy.draft_ingestion.ffpc import to_draft_boards as ffpc_boards
    from fantasy.draft_ingestion.underdog import to_draft_boards as underdog_boards

    html = "<table><tr><th>Player</th><th>Pick</th></tr><tr><td>X</td><td>1</td></tr></table>"
    assert ffpc_boards(html) == [[("X", 1)]]
    assert underdog_boards({"picks": [{"name": "Y", "pick": 2}]}) == [[("Y", 2)]]


def test_fp_analyzer_to_draft_boards_orders_by_adp():
    from fantasy.draft_ingestion.fp_analyzer import to_draft_boards

    csv_text = "Player,ADP\nLate,50.0\nEarly,1.0\n"
    assert to_draft_boards(csv_text) == [[("Early", 1), ("Late", 2)]]


def test_to_draft_boards_returns_empty_when_nothing_parses():
    from fantasy.draft_ingestion.ffpc import to_draft_boards as ffpc_boards
    from fantasy.draft_ingestion.fp_analyzer import to_draft_boards as fp_boards
    from fantasy.draft_ingestion.github_datasets import to_draft_boards as gh_boards

    assert ffpc_boards("") == []
    assert fp_boards("") == []
    assert gh_boards("", fmt="csv") == []
