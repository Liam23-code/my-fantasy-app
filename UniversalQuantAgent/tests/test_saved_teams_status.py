"""Live player-status overlay on the Saved Teams page (26).

OUT players stay on a saved roster (you drafted them) but are flagged red
with a zeroed projection; DOUBTFUL / QUESTIONABLE are trimmed and flagged
yellow. Runs the real Streamlit script through AppTest against a temp
saved-team directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fantasy import my_team_manager, player_status
from streamlit.testing.v1 import AppTest

PAGE = Path(_PROJECT_ROOT) / "app" / "pages" / "26_Fantasy_Saved_Teams.py"

ROSTER = [
    {"player_id": "00-0000001", "name": "Healthy Star", "position": "WR", "projection": 240.0},
    {"player_id": "00-0000002", "name": "Ruled Out Guy", "position": "RB", "projection": 200.0},
    {"player_id": "00-0000003", "name": "Maybe Playing", "position": "TE", "projection": 120.0},
]


@pytest.fixture()
def saved_dir(tmp_path, monkeypatch):
    teams_dir = tmp_path / "user_teams"
    teams_dir.mkdir()
    monkeypatch.setattr(my_team_manager, "USER_TEAMS_DIR", teams_dir)
    document = {
        "team_id": "test1",
        "name": "Overlay Test Team",
        "league": "Test League",
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00Z",
        "metadata": {},
        "players": ROSTER,
    }
    (teams_dir / "team_test1.json").write_text(json.dumps(document), encoding="utf-8")
    return teams_dir


@pytest.fixture()
def status(tmp_path, monkeypatch):
    path = tmp_path / "player_status.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player_status, "STATUS_PATH", path)
    player_status.clear_cache()

    def _set(mapping):
        path.write_text(json.dumps(mapping), encoding="utf-8")
        player_status.clear_cache()

    yield _set
    player_status.clear_cache()


def _run():
    app = AppTest.from_file(str(PAGE), default_timeout=120)
    app.run()
    raised = [str(e.value) for e in app.exception]
    assert raised == [], f"page raised: {raised}"
    return app


def _text(app):
    blocks = [str(e.value) for e in app.markdown] + [str(e.value) for e in app.caption]
    blocks += [str(e.value) for e in app.success] + [str(e.value) for e in app.warning]
    return "\n".join(blocks)


# --- renders + controls -----------------------------------------------------


def test_page_renders_with_the_refresh_button_and_still_the_create_button(saved_dir, status):
    app = _run()
    assert any(b.key == "saved_teams_refresh_player_status" for b in app.button)
    assert any(b.label == "Create New Team Save" for b in app.button)
    assert "Player status overlay not loaded" in _text(app)


def test_last_updated_strip_matches_the_draft_room_wording(saved_dir, status):
    status({"00-0000002": {"status": "OUT", "name": "Ruled Out Guy", "last_updated": "2026-09-01T09:30:00Z"}})
    text = _text(_run())
    assert "Player status overlay ·" in text
    assert "1 player(s) flagged" in text
    assert "updated 2026-09-01 09:30:00 UTC" in text


# --- roster badges + adjusted projections --------------------------------------


def test_out_player_stays_on_the_roster_but_is_flagged_and_zeroed(saved_dir, status):
    status(
        {
            "00-0000002": {"status": "OUT", "name": "Ruled Out Guy"},
            "00-0000003": {"status": "QUESTIONABLE", "name": "Maybe Playing"},
        }
    )
    text = _text(_run())
    # OUT: still listed, red badge, projection shown as 0
    assert "Ruled Out Guy" in text
    assert "\U0001f534 OUT **Ruled Out Guy**" in text
    assert "**Ruled Out Guy** (RB) · proj 0.0" in text
    # QUESTIONABLE: yellow badge, trimmed projection (120 * 0.88 = 105.6), still listed
    assert "\U0001f7e1 Questionable **Maybe Playing**" in text
    assert "proj 105.6" in text
    # HEALTHY: green badge, untouched projection
    assert "\U0001f7e2 Healthy **Healthy Star**" in text
    assert "proj 240.0" in text


def test_team_availability_risk_line_summarises_the_flags(saved_dir, status):
    status(
        {
            "00-0000002": {"status": "OUT", "name": "Ruled Out Guy"},
            "00-0000003": {"status": "DOUBTFUL", "name": "Maybe Playing"},
        }
    )
    text = _text(_run())
    assert "Availability risk: 2 of 3 flagged" in text
    assert "1 out" in text and "1 doubtful" in text


def test_no_overlay_shows_stored_projections_and_a_prompt(saved_dir, status):
    text = _text(_run())
    assert "\U0001f7e2 Healthy **Ruled Out Guy**" in text  # no overlay -> everyone reads healthy
    assert "proj 200.0" in text
    assert "Refresh player status above to check this roster" in text
