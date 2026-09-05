"""Phase A "Refresh Player Status" strip on the eight display-only pages.

One shared component (``app.status_ui.render_status_strip``) renders the same
timestamp / flagged-count caption and the same button on every page. These
tests pin that it is actually reachable on all eight, that a click rewrites
``player_status.json``, and that a dead network leaves the existing file
exactly as it was rather than blanking it.

Nothing here touches the real feed: the fetcher's HTTP getter is replaced, so
the real ``normalize_feed`` and the real atomic write still run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app import betting_shared, fantasy_shared
from fantasy import my_team_manager, player_status
from fantasy.online import player_status_fetcher
from streamlit.testing.v1 import AppTest

PAGES = Path(_PROJECT_ROOT) / "app" / "pages"

#: page file -> the button key it passes to the shared strip.
STRIP_PAGES: dict[str, str] = {
    "21_NFL_Player_Analysis.py": "nfl_player_analysis_refresh_player_status",
    "23_NFL_Projections.py": "nfl_projections_refresh_player_status",
    "24_NFL_Slate.py": "nfl_slate_refresh_player_status",
    "25_Fantasy_Hub.py": "fantasy_hub_refresh_player_status",
    "27_Fantasy_Season_Tools.py": "season_tools_refresh_player_status",
    "28_Fantasy_My_Team.py": "my_team_refresh_player_status",
    "29_Graph_Lab.py": "graph_lab_refresh_player_status",
    "30_NFL_Betting.py": "nfl_betting_refresh_player_status",
}

#: A minimal Sleeper /v1/players/nfl payload: one currently-rostered injured
#: player, one retired player who must be ignored.
SLEEPER_PAYLOAD = {
    "4034": {
        "gsis_id": "00-0033553",
        "full_name": "Feed Injured Guy",
        "team": "SF",
        "active": True,
        "injury_status": "Out",
    },
    "9999": {
        "gsis_id": "00-0000009",
        "full_name": "Long Retired Guy",
        "team": None,
        "active": False,
        "injury_status": "Out",
    },
}

EXISTING_FILE = {
    "00-0011111": {
        "status": "QUESTIONABLE",
        "name": "Already Known Guy",
        "last_updated": "2026-08-20T08:00:00Z",
        "source": "sleeper",
    }
}


@pytest.fixture()
def status_file(tmp_path, monkeypatch):
    """Point the status module at a temp file. The pages resolve
    ``player_status.STATUS_PATH`` at click time, so a refresh lands here."""
    path = tmp_path / "player_status.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player_status, "STATUS_PATH", path)
    player_status.clear_cache()
    yield path
    player_status.clear_cache()


@pytest.fixture()
def offline_pool(monkeypatch):
    """The Fantasy pages' pool loader, stubbed so no test reaches nflverse."""
    players = [
        {
            "player_id": f"strip-{index}",
            "name": f"Strip Player {index}",
            "position": position,
            "team": "SF",
            "projection": projection,
            "expected_fantasy_points": projection,
            "points_per_game": round(projection / 16, 2),
            "games_played": 16,
            "adp": float(index + 1),
            "scoring_mode": "ppr",
            "projection_basis": "refresh strip contract pool",
        }
        for index, (position, projection) in enumerate(
            (("QB", 300.0), ("RB", 240.0), ("WR", 230.0), ("TE", 180.0))
        )
    ]
    monkeypatch.setattr(fantasy_shared, "load_pool", lambda *_a, **_k: (list(players), "strip pool"))
    return players


def prop_row(player_id: str, name: str) -> dict:
    """One prop evaluation in the exact shape ``betting_shared`` renders."""
    return {
        "player_id": player_id,
        "name": name,
        "team": "SF",
        "market": "rushing_yards",
        "line": 62.5,
        "distribution": {"mean": 68.0, "std": 22.0},
        "recommended_side": "over",
        "recommended_edge": 0.04,
        "recommended_ev": 3.1,
        "model_probability_over": 0.57,
        "model_probability_under": 0.43,
        "market_fair_probability_over": 0.53,
        "market_fair_probability_under": 0.47,
        "over_price": -110,
        "under_price": -110,
        "confidence": 0.55,
        "risk_tier": "medium",
        "odds_source": "default",
    }


@pytest.fixture()
def offline_odds(monkeypatch):
    rows = [prop_row("strip-1", "Strip Player 1")]
    monkeypatch.setattr(
        betting_shared, "load_nfl_evaluations", lambda _key: ([dict(r) for r in rows], [], {})
    )


@pytest.fixture()
def feed_ok(monkeypatch):
    """A working feed, without the network."""
    monkeypatch.setattr(player_status_fetcher, "_http_get_json", lambda *_a, **_k: SLEEPER_PAYLOAD)


@pytest.fixture()
def feed_down(monkeypatch):
    """A dead feed: ``_http_get_json`` returns ``None`` on every failure mode."""
    monkeypatch.setattr(player_status_fetcher, "_http_get_json", lambda *_a, **_k: None)


@pytest.fixture()
def saved_team(tmp_path, monkeypatch):
    teams_dir = tmp_path / "user_teams"
    teams_dir.mkdir()
    monkeypatch.setattr(my_team_manager, "USER_TEAMS_DIR", teams_dir)
    (teams_dir / "team_strip1.json").write_text(
        json.dumps(
            {
                "team_id": "strip1",
                "name": "Strip Team",
                "league": "Test League",
                "created_at": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-01T00:00:00Z",
                "metadata": {},
                "players": [
                    {"player_id": "strip-1", "name": "Strip Player 1", "position": "RB", "projection": 240.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    return "strip1"


#: ``st.page_link`` needs a real page registry, so these two are driven through
#: the app shell the same way tests/test_ui_layout.py drives them.
SHELL_PAGES = frozenset({"25_Fantasy_Hub.py", "28_Fantasy_My_Team.py"})
APP_ENTRY = Path(_PROJECT_ROOT) / "app" / "app.py"


def _run(filename, session=None, timeout=180):
    if filename in SHELL_PAGES:
        app = AppTest.from_file(str(APP_ENTRY), default_timeout=timeout)
        for key, value in (session or {}).items():
            app.session_state[key] = value
        app.run()
        app.switch_page(f"pages/{filename}").run()
    else:
        app = AppTest.from_file(str(PAGES / filename), default_timeout=timeout)
        for key, value in (session or {}).items():
            app.session_state[key] = value
        app.run()
    raised = [str(element.value) for element in app.exception]
    assert raised == [], f"{filename} raised: {raised}"
    return app


def _text(app):
    blocks = [str(element.value) for element in app.markdown]
    blocks += [str(element.value) for element in app.caption]
    blocks += [str(element.value) for element in app.success]
    blocks += [str(element.value) for element in app.warning]
    blocks += [str(element.value) for element in app.info]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# The strip is on all eight pages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page", sorted(STRIP_PAGES))
def test_every_page_calls_the_one_shared_strip(page):
    """Ten hand-rolled copies is what this component exists to prevent."""
    source = (PAGES / page).read_text(encoding="utf-8")
    assert "from app.status_ui import" in source
    assert "render_status_strip(" in source


@pytest.mark.parametrize("page,key", sorted(STRIP_PAGES.items()))
def test_the_refresh_button_renders_on_every_page(
    page, key, status_file, offline_pool, offline_odds, saved_team
):
    session = {"fantasy_selected_team_id": saved_team} if page.startswith("28_") else None
    app = _run(page, session=session)
    button = next((b for b in app.button if b.key == key), None)
    assert button is not None, f"{page} has no Refresh Player Status button"
    assert button.label == "Refresh Player Status"


@pytest.mark.parametrize("page,key", sorted(STRIP_PAGES.items()))
def test_the_not_loaded_caption_renders_on_every_page(
    page, key, status_file, offline_pool, offline_odds, saved_team
):
    session = {"fantasy_selected_team_id": saved_team} if page.startswith("28_") else None
    assert "Player status overlay not loaded" in _text(_run(page, session=session))


@pytest.mark.parametrize("page,key", sorted(STRIP_PAGES.items()))
def test_the_last_updated_strip_renders_on_every_page(
    page, key, status_file, offline_pool, offline_odds, saved_team
):
    status_file.write_text(json.dumps(EXISTING_FILE), encoding="utf-8")
    player_status.clear_cache()
    session = {"fantasy_selected_team_id": saved_team} if page.startswith("28_") else None
    text = _text(_run(page, session=session))
    assert "Player status overlay ·" in text
    assert "1 player(s) flagged" in text
    assert "updated 2026-08-20 08:00:00 UTC" in text


def test_the_button_key_is_unique_per_page():
    """Two pages sharing a widget key collide the moment both are in one
    session's page registry."""
    assert len(set(STRIP_PAGES.values())) == len(STRIP_PAGES)


def test_the_28_no_team_branch_still_offers_the_strip(status_file):
    """My Team stops early with no save selected; the strip is above the stop."""
    app = _run("28_Fantasy_My_Team.py")
    assert any(b.key == "my_team_refresh_player_status" for b in app.button)


# ---------------------------------------------------------------------------
# The click actually rewrites player_status.json
# ---------------------------------------------------------------------------
def test_clicking_refresh_writes_the_status_file(status_file, feed_ok):
    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)
    assert [str(e.value) for e in app.exception] == []

    written = json.loads(status_file.read_text(encoding="utf-8"))
    assert written["00-0033553"]["status"] == "OUT"
    assert written["00-0033553"]["source"] == "sleeper"
    assert written["nm:feedinjuredguy"]["status"] == "OUT"
    # The retired player is not a current injury and must not be written.
    assert "00-0000009" not in written


def test_the_strip_reports_the_new_count_after_a_refresh(status_file, feed_ok):
    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)
    text = _text(app)
    assert "Player status overlay ·" in text
    assert "1 player(s) flagged" in text


def test_a_refresh_feeds_the_badges_on_the_same_page(status_file, feed_ok):
    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)
    assert "\U0001f534 OUT **Feed Injured Guy**" in _text(app)


def test_the_refresh_writes_to_the_configured_path_not_the_shipped_one(status_file, feed_ok):
    """The page resolves ``player_status.STATUS_PATH`` at click time, so a
    developer's local refresh can never be what a test asserts against."""
    real_path = Path(player_status.__file__).resolve().parents[1] / "data" / "player_status.json"
    before = real_path.read_text(encoding="utf-8") if real_path.exists() else None

    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)

    assert json.loads(status_file.read_text(encoding="utf-8"))
    after = real_path.read_text(encoding="utf-8") if real_path.exists() else None
    assert after == before


# ---------------------------------------------------------------------------
# Graceful degradation: a dead feed keeps the old JSON
# ---------------------------------------------------------------------------
def test_a_failed_fetch_keeps_the_existing_file_byte_for_byte(status_file, feed_down):
    status_file.write_text(json.dumps(EXISTING_FILE), encoding="utf-8")
    player_status.clear_cache()
    before = status_file.read_text(encoding="utf-8")

    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)

    assert [str(e.value) for e in app.exception] == []
    assert status_file.read_text(encoding="utf-8") == before


def test_a_failed_fetch_warns_instead_of_erroring(status_file, feed_down):
    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "kept the existing" in warnings
    assert [str(e.value) for e in app.error] == []


def test_a_failed_fetch_leaves_the_page_rendering_normally(status_file, feed_down):
    status_file.write_text(json.dumps(EXISTING_FILE), encoding="utf-8")
    player_status.clear_cache()
    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)
    text = _text(app)
    assert "\U0001f7e1 Questionable **Already Known Guy**" in text
    assert "Availability watchlist" in [str(e.label) for e in app.expander]


def test_an_unwritable_status_file_warns_rather_than_crashing_the_page(status_file, monkeypatch):
    def _explode(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("app.status_ui.refresh_player_status", _explode)
    app = _run("24_NFL_Slate.py")
    app.button(key="nfl_slate_refresh_player_status").click().run(timeout=180)
    assert [str(e.value) for e in app.exception] == []
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "kept the existing data" in warnings
