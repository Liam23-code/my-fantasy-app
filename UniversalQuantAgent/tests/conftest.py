"""Shared fixtures for the app-level test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _isolated_player_status(tmp_path_factory, monkeypatch):
    """Point ``fantasy.player_status`` at an empty per-test file so a page test
    never depends on whatever a local ``player_status.json`` holds (a user may
    have clicked Refresh Player Status)."""
    from fantasy import player_status

    empty = tmp_path_factory.mktemp("player_status") / "player_status.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player_status, "STATUS_PATH", empty)
    player_status.clear_cache()
    yield
    player_status.clear_cache()
