"""Tests for safe, multi-league fantasy team persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fantasy.my_team_manager as manager


@pytest.fixture(autouse=True)
def isolated_team_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    storage = tmp_path / "user_teams"
    monkeypatch.setattr(manager, "USER_TEAMS_DIR", storage)
    return storage


def _team(player_name: str = "Example Runner") -> dict:
    return {
        "league_settings": {"scoring_mode": "ppr"},
        "players": [
            {
                "player_id": "runner-1",
                "name": player_name,
                "position": "RB",
                "team": "DEN",
            }
        ],
    }


def test_create_load_and_list_multiple_team_saves(isolated_team_storage: Path):
    alpha = manager.create_new_team_save(
        _team("Alpha Runner"),
        team_id="alpha",
        name="Alpha Team",
        league="Office League",
        metadata={"season": 2026},
    )
    beta = manager.create_new_team_save(
        _team("Beta Runner"),
        team_id="beta",
        name="Beta Team",
        league="Family League",
    )

    assert alpha["team_id"] == "alpha"
    assert beta["team_id"] == "beta"
    assert manager.load_saved_team("alpha")["players"][0]["name"] == "Alpha Runner"
    assert [team["team_id"] for team in manager.list_saved_teams()] == ["alpha", "beta"]
    assert manager.list_saved_teams()[0] == {
        "team_id": "alpha",
        "name": "Alpha Team",
        "league": "Office League",
        "created_at": alpha["created_at"],
        "updated_at": alpha["updated_at"],
        "player_count": 1,
        "metadata": {"season": 2026},
        "is_valid": True,
    }
    assert (isolated_team_storage / "team_alpha.json").is_file()
    assert (isolated_team_storage / "team_beta.json").is_file()


def test_create_without_arguments_returns_empty_self_describing_record():
    saved = manager.create_new_team_save()

    assert len(saved["team_id"]) == 12
    assert saved["players"] == []
    assert saved["name"]
    assert manager.load_saved_team(saved["team_id"]) == saved


def test_string_positional_argument_is_a_name_and_names_get_unique_ids():
    first = manager.create_new_team_save("Sunday Stars")
    second = manager.create_new_team_save("Sunday Stars")

    assert first["team_id"] == "sunday-stars"
    assert second["team_id"] == "sunday-stars-2"
    assert first["name"] == second["name"] == "Sunday Stars"


def test_update_preserves_creation_time_and_merges_metadata():
    created = manager.create_new_team_save(
        _team(),
        team_id="dynasty",
        name="Old Name",
        metadata={"season": 2026, "keeper": True},
    )

    updated = manager.save_saved_team(
        "dynasty",
        _team("Updated Runner"),
        name="New Name",
        metadata={"season": 2027},
    )

    assert updated["created_at"] == created["created_at"]
    assert updated["updated_at"] >= created["updated_at"]
    assert updated["name"] == "New Name"
    assert updated["league"] == created["league"]
    assert updated["metadata"] == {"season": 2027, "keeper": True}
    assert manager.load_saved_team("dynasty")["players"][0]["name"] == "Updated Runner"


def test_create_refuses_to_overwrite_but_save_helper_updates():
    manager.create_new_team_save(_team(), team_id="work")

    with pytest.raises(FileExistsError):
        manager.create_new_team_save(_team("Replacement"), team_id="work")

    result = manager.save_saved_team("work", _team("Replacement"))
    assert result["players"][0]["name"] == "Replacement"


@pytest.mark.parametrize("team_id", ["../escape", "nested/team", r"nested\team", "..", "\x00bad"])
def test_team_ids_cannot_traverse_outside_storage(team_id: str, tmp_path: Path):
    with pytest.raises(ValueError):
        manager.create_new_team_save(_team(), team_id=team_id)
    with pytest.raises(ValueError):
        manager.load_saved_team(team_id)
    with pytest.raises(ValueError):
        manager.delete_team_save(team_id)

    assert not (tmp_path / "team_escape.json").exists()


def test_friendly_team_id_is_sanitized_consistently():
    saved = manager.create_new_team_save(_team(), team_id="  Mile High Heroes!  ")

    assert saved["team_id"] == "mile-high-heroes"
    assert manager.load_saved_team("Mile High Heroes!") == saved


def test_delete_reports_whether_a_save_existed():
    manager.create_new_team_save(_team(), team_id="delete-me")

    assert manager.delete_team_save("delete-me") is True
    assert manager.delete_team_save("delete-me") is False
    with pytest.raises(FileNotFoundError):
        manager.load_saved_team("delete-me")


def test_corrupt_save_is_listed_but_cannot_be_loaded(isolated_team_storage: Path):
    isolated_team_storage.mkdir(parents=True)
    (isolated_team_storage / "team_broken.json").write_text("{not-json", encoding="utf-8")

    summary = manager.list_saved_teams()

    assert len(summary) == 1
    assert summary[0]["team_id"] == "broken"
    assert summary[0]["is_valid"] is False
    assert "error" in summary[0]
    with pytest.raises(ValueError, match="not valid JSON"):
        manager.load_saved_team("broken")


def test_writes_are_complete_json_and_leave_no_temporary_files(isolated_team_storage: Path):
    saved = manager.create_new_team_save(_team(), team_id="atomic")

    path = isolated_team_storage / "team_atomic.json"
    assert json.loads(path.read_text(encoding="utf-8")) == saved
    assert list(isolated_team_storage.glob("*.tmp")) == []


def test_legacy_single_team_api_keeps_original_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    legacy_path = tmp_path / "data" / "user_team.json"
    monkeypatch.setattr(manager, "USER_TEAM_PATH", legacy_path)
    roster = _team()["players"]

    saved = manager.save_user_team(roster)

    assert saved == roster
    assert manager.load_user_team() == roster
    assert isinstance(manager.load_user_team(), list)
