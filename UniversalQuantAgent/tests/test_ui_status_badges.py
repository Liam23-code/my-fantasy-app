"""Phase A availability badges on the eight display-only pages.

Pages 21, 23, 24, 25, 27, 28, 29 and 30 show stored projections or historical
data. Phase A adds a status badge next to the players they already show -- and
nothing else. These tests pin both halves of that: the badge renders with the
right colour for every canonical status, and **no row disappears** because of
it. A page that started filtering on availability would fail here.

The badge vocabulary itself is shared (``app.status_ui``), so the colour
mapping is asserted once as a unit and then spot-checked through the real
Streamlit scripts via ``AppTest``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app import betting_shared, fantasy_shared, status_ui
from fantasy import my_team_manager, player_status
from streamlit.testing.v1 import AppTest

PAGES = Path(_PROJECT_ROOT) / "app" / "pages"

RED = "\U0001f534"  # OUT
ORANGE = "\U0001f7e0"  # HOLDOUT / SUSPENDED
YELLOW = "\U0001f7e1"  # DOUBTFUL / QUESTIONABLE
GREEN = "\U0001f7e2"  # HEALTHY

#: One player per canonical status, plus one whose flag comes from the pool's
#: own ``injury_status`` rather than the live feed.
POOL_SHAPE = [
    ("st-out", "Out Guy", "RB", 240.0, None),
    ("st-holdout", "Holdout Guy", "WR", 230.0, None),
    ("st-suspended", "Suspended Guy", "WR", 220.0, None),
    ("st-doubtful", "Doubtful Guy", "TE", 210.0, None),
    ("st-questionable", "Questionable Guy", "QB", 200.0, None),
    ("st-healthy", "Healthy Guy", "RB", 190.0, None),
    ("st-stored", "Stored Flag Guy", "WR", 180.0, "Questionable"),
]

LIVE_STATUS = {
    "st-out": {"status": "OUT", "name": "Out Guy", "last_updated": "2026-09-01T10:00:00Z"},
    "st-holdout": {"status": "HOLDOUT", "name": "Holdout Guy", "last_updated": "2026-09-01T10:00:00Z"},
    "st-suspended": {"status": "SUSPENDED", "name": "Suspended Guy", "last_updated": "2026-09-01T10:00:00Z"},
    "st-doubtful": {"status": "DOUBTFUL", "name": "Doubtful Guy", "last_updated": "2026-09-01T10:00:00Z"},
    "st-questionable": {
        "status": "QUESTIONABLE",
        "name": "Questionable Guy",
        "last_updated": "2026-09-01T10:00:00Z",
    },
}

ALL_NAMES = [row[1] for row in POOL_SHAPE]


def _pool() -> list[dict]:
    players = []
    for index, (player_id, name, position, projection, stored_flag) in enumerate(POOL_SHAPE):
        row = {
            "player_id": player_id,
            "name": name,
            "position": position,
            "team": ["SF", "KC", "BUF", "DAL"][index % 4],
            "projection": projection,
            "expected_fantasy_points": projection,
            "points_per_game": round(projection / 16, 2),
            "games_played": 16,
            "expected_games": 15.6,
            "floor": round(projection * 0.7, 2),
            "median": round(projection * 0.95, 2),
            "ceiling": round(projection * 1.35, 2),
            "adp": float(index + 1),
            "scoring_mode": "ppr",
            "projection_basis": "status badge contract pool",
        }
        if stored_flag:
            row["injury_status"] = stored_flag
        players.append(row)
    return players


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def status(tmp_path, monkeypatch):
    """Point the whole status module at a temp file and hand back a setter."""
    path = tmp_path / "player_status.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player_status, "STATUS_PATH", path)
    player_status.clear_cache()

    def _set(mapping):
        path.write_text(json.dumps(mapping), encoding="utf-8")
        player_status.clear_cache()

    yield _set
    player_status.clear_cache()


@pytest.fixture()
def pool(monkeypatch):
    players = _pool()
    monkeypatch.setattr(fantasy_shared, "load_pool", lambda *_a, **_k: (list(players), "status badge pool"))
    return players


@pytest.fixture()
def saved_team(tmp_path, monkeypatch):
    teams_dir = tmp_path / "user_teams"
    teams_dir.mkdir()
    monkeypatch.setattr(my_team_manager, "USER_TEAMS_DIR", teams_dir)
    document = {
        "team_id": "badges1",
        "name": "Badge Test Team",
        "league": "Test League",
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00Z",
        "metadata": {},
        "players": _pool(),
    }
    (teams_dir / "team_badges1.json").write_text(json.dumps(document), encoding="utf-8")
    return "badges1"


def _prop_row(player_id: str, name: str) -> dict:
    """One prop evaluation in the exact shape ``betting_shared`` renders."""
    return {
        "player_id": player_id,
        "name": name,
        "team": "SF",
        "market": "passing_yards",
        "line": 250.5,
        "distribution": {"mean": 262.0, "std": 45.0},
        "recommended_side": "over",
        "recommended_edge": 0.05,
        "recommended_ev": 4.2,
        "model_probability_over": 0.58,
        "model_probability_under": 0.42,
        "market_fair_probability_over": 0.53,
        "market_fair_probability_under": 0.47,
        "over_price": -110,
        "under_price": -110,
        "confidence": 0.6,
        "risk_tier": "medium",
        "odds_source": "default",
    }


@pytest.fixture()
def stub_nfl_odds(monkeypatch):
    rows = [_prop_row(player_id, name) for player_id, name, *_rest in POOL_SHAPE]
    monkeypatch.setattr(
        betting_shared, "load_nfl_evaluations", lambda _key: ([dict(r) for r in rows], [], {})
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
# The shared badge vocabulary
# ---------------------------------------------------------------------------
def test_every_canonical_status_maps_to_its_documented_colour():
    assert status_ui.status_badge("OUT") == f"{RED} OUT"
    assert status_ui.status_badge("HOLDOUT") == f"{ORANGE} Holdout"
    assert status_ui.status_badge("SUSPENDED") == f"{ORANGE} Suspended"
    assert status_ui.status_badge("DOUBTFUL") == f"{YELLOW} Doubtful"
    assert status_ui.status_badge("QUESTIONABLE") == f"{YELLOW} Questionable"
    assert status_ui.status_badge("HEALTHY") == f"{GREEN} Healthy"


def test_an_unknown_or_missing_status_reads_healthy_rather_than_erroring():
    for value in (None, "", "  ", "not-a-status", 17):
        assert status_ui.status_badge(value) == f"{GREEN} Healthy"


def test_the_badge_vocabulary_matches_the_pages_that_shipped_it_first():
    """Pages 25/26 hard-code the same six badges. A change here would make one
    player read differently on two screens, which is the bug this shares."""
    saved_teams = (PAGES / "26_Fantasy_Saved_Teams.py").read_text(encoding="utf-8")
    for badge in status_ui.STATUS_BADGES.values():
        assert badge in saved_teams, f"{badge!r} drifted from the Saved Teams vocabulary"


def test_badges_prefer_the_live_feed_and_fall_back_to_the_stored_flag(status):
    status(LIVE_STATUS)
    assert status_ui.player_status_badge({"player_id": "st-out"}) == f"{RED} OUT"
    # No live record; the pool's own injury_status is what we know.
    stored = {"player_id": "st-stored", "injury_status": "Questionable"}
    assert status_ui.player_status_badge(stored) == f"{YELLOW} Questionable"
    # live_badge is the engines' view: the stored flag is never escalated.
    assert status_ui.live_badge(stored) == f"{GREEN} Healthy"


def test_a_bare_player_id_is_accepted_as_well_as_a_row(status):
    status(LIVE_STATUS)
    assert status_ui.player_status_badge("st-holdout") == f"{ORANGE} Holdout"


# ---------------------------------------------------------------------------
# Page 25 -- Fantasy Hub: every status on one screen, nothing dropped
# ---------------------------------------------------------------------------
def test_hub_shows_all_six_badges_and_drops_nobody(pool, status):
    status(LIVE_STATUS)
    text = _text(_run("25_Fantasy_Hub.py"))
    assert f"{RED} OUT **Out Guy (RB)**" in text
    assert f"{ORANGE} Holdout **Holdout Guy (WR)**" in text
    assert f"{ORANGE} Suspended **Suspended Guy (WR)**" in text
    assert f"{YELLOW} Doubtful **Doubtful Guy (TE)**" in text
    assert f"{YELLOW} Questionable **Questionable Guy (QB)**" in text
    assert f"{GREEN} Healthy **Healthy Guy (RB)**" in text
    # The stored-flag player reads from his own injury_status.
    assert f"{YELLOW} Questionable **Stored Flag Guy (WR)**" in text
    for name in ALL_NAMES:
        assert name in text, f"{name} was filtered off the hub"


def test_hub_keeps_the_stored_projection_next_to_a_flagged_player(pool, status):
    """No math changes: an OUT player still shows the pool's own number."""
    status(LIVE_STATUS)
    text = _text(_run("25_Fantasy_Hub.py"))
    assert "**Out Guy (RB)** · proj 240.0" in text
    assert "**Healthy Guy (RB)** · proj 190.0" in text


def test_hub_summarises_the_flags_without_removing_anyone(pool, status):
    status(LIVE_STATUS)
    text = _text(_run("25_Fantasy_Hub.py"))
    assert "Availability: 6 of 7 flagged" in text
    assert "1 out" in text and "1 holdout" in text and "1 suspended" in text
    assert "1 doubtful" in text and "2 questionable" in text


def test_hub_with_no_overlay_badges_everyone_green(pool, status):
    text = _text(_run("25_Fantasy_Hub.py"))
    assert f"{GREEN} Healthy **Out Guy (RB)**" in text
    assert "Refresh player status above to check live availability" in text


# ---------------------------------------------------------------------------
# Page 28 -- My Team: badges on the saved roster, nothing removed
# ---------------------------------------------------------------------------
def test_my_team_badges_the_saved_roster_and_keeps_every_player(pool, status, saved_team):
    status(LIVE_STATUS)
    app = _run("28_Fantasy_My_Team.py", session={"fantasy_selected_team_id": saved_team})
    text = _text(app)
    assert f"{RED} OUT **Out Guy (RB)**" in text
    assert f"{ORANGE} Suspended **Suspended Guy (WR)**" in text
    assert f"{YELLOW} Doubtful **Doubtful Guy (TE)**" in text
    assert f"{GREEN} Healthy **Healthy Guy (RB)**" in text
    for name in ALL_NAMES:
        assert name in text, f"{name} was dropped from the saved roster"


def test_my_team_roster_cards_carry_the_badge_as_their_health_line(pool, status, saved_team):
    status(LIVE_STATUS)
    text = _text(_run("28_Fantasy_My_Team.py", session={"fantasy_selected_team_id": saved_team}))
    # stacked_card_html renders "Health" as a card stat -- the badge lands there.
    assert "Health" in text
    assert f"{RED} OUT" in text


def test_my_team_shows_the_stored_projection_beside_a_flagged_player(pool, status, saved_team):
    status(LIVE_STATUS)
    text = _text(_run("28_Fantasy_My_Team.py", session={"fantasy_selected_team_id": saved_team}))
    assert "**Out Guy (RB)** · proj 240.0" in text


# ---------------------------------------------------------------------------
# Page 29 -- Graph Lab: the selected player and his comps
# ---------------------------------------------------------------------------
def test_graph_lab_badges_the_selected_player(pool, status):
    status(LIVE_STATUS)
    app = _run("29_Graph_Lab.py")
    text = _text(app)
    assert f"**Availability:** {RED} OUT · Out Guy (RB)" in text
    # Nobody left the selector.
    options = " || ".join(app.selectbox(key="graph_lab_player").options)
    for name in ALL_NAMES:
        assert name in options, f"{name} was filtered out of the Graph Lab selector"


# ---------------------------------------------------------------------------
# Page 27 -- Weekly Tools: the selected weekly player and the roster list
# ---------------------------------------------------------------------------
def test_season_tools_badges_the_selected_weekly_player(pool, status):
    status(LIVE_STATUS)
    app = _run("27_Fantasy_Season_Tools.py")
    text = _text(app)
    assert f"**Availability:** {RED} OUT · Out Guy (RB)" in text
    assert len(app.selectbox(key="fantasy_weekly_player").options) == len(POOL_SHAPE)


def test_season_tools_badges_the_loaded_roster_without_benching_anyone(pool, status):
    status(LIVE_STATUS)
    app = _run("27_Fantasy_Season_Tools.py", session={"fantasy_roster": _pool()})
    text = _text(app)
    assert f"{RED} OUT **Out Guy (RB)**" in text
    assert f"{GREEN} Healthy **Healthy Guy (RB)**" in text
    # The optimizer's own exclusion control is still the only way to sit a player.
    assert app.multiselect(key="fantasy_excluded_players").value == []


# ---------------------------------------------------------------------------
# Page 30 -- NFL Betting: badges beside the priced props, no rows removed
# ---------------------------------------------------------------------------
def test_nfl_betting_badges_every_priced_player(stub_nfl_odds, status):
    status(LIVE_STATUS)
    app = _run("30_NFL_Betting.py")
    text = _text(app)
    assert f"{RED} OUT **Out Guy**" in text
    assert f"{ORANGE} Holdout **Holdout Guy**" in text
    assert f"{YELLOW} Questionable **Questionable Guy**" in text
    assert f"{GREEN} Healthy **Healthy Guy**" in text
    assert len(app.tabs) == 3  # no new tab


def test_nfl_betting_leaves_the_props_table_untouched(stub_nfl_odds, status):
    """The prop model already dropped live-OUT rows upstream. This page must
    not drop a second time -- every stubbed row still reaches the table."""
    status(LIVE_STATUS)
    app = _run("30_NFL_Betting.py")
    props = next(df.value for df in app.dataframe if "Market" in df.value.columns)
    assert len(props) == len(POOL_SHAPE)
    assert set(props["Player"]) == set(ALL_NAMES)


# ---------------------------------------------------------------------------
# Page 24 -- NFL Slate: a league-wide watchlist, no player rows of its own
# ---------------------------------------------------------------------------
def test_slate_watchlist_badges_every_flagged_player(status):
    status(LIVE_STATUS)
    text = _text(_run("24_NFL_Slate.py"))
    assert f"{RED} OUT **Out Guy**" in text
    assert f"{ORANGE} Holdout **Holdout Guy**" in text
    assert f"{ORANGE} Suspended **Suspended Guy**" in text
    assert f"{YELLOW} Doubtful **Doubtful Guy**" in text
    assert f"{YELLOW} Questionable **Questionable Guy**" in text
    assert "5 player(s) flagged league-wide" in text


def test_slate_watchlist_reads_healthy_when_the_overlay_is_empty(status):
    text = _text(_run("24_NFL_Slate.py"))
    assert f"{GREEN} Healthy — no players are flagged" in text


def test_slate_watchlist_caps_the_list_but_still_counts_everyone(status):
    """A real preseason pull is ~675 names. Printed flat, it buries the slate
    the page is actually about -- so the list is capped and the count is not."""
    status(
        {
            f"gsis-{index:04d}": {"status": "OUT", "name": f"Flagged Player {index:04d}"}
            for index in range(120)
        }
    )
    text = _text(_run("24_NFL_Slate.py"))
    assert "Showing the first 50 of 120 flagged players." in text
    assert "120 player(s) flagged league-wide — 120 out" in text
    assert f"{RED} OUT **Flagged Player 0000**" in text
    assert f"{RED} OUT **Flagged Player 0119**" not in text  # past the cap


# ---------------------------------------------------------------------------
# Pages 21 / 23 -- one analysed player, one badge
# ---------------------------------------------------------------------------
ANALYSIS_RESULT = {
    "player": "Out Guy",
    "team": "SF",
    "position": "RB",
    "season": 2025,
    "player_id": "st-out",
    "identity_summary": "Volume back with a stable role.",
    "usage_profile": {"snap_share": 72.5, "volume_per_game": 18.4, "role_stability": 81.0},
    "volatility_score": 34.0,
    "attributes": [
        {"metric": "Rushing", "blended_percentile": 88.0},
        {"metric": "Receiving", "blended_percentile": 61.0},
    ],
    "strengths": [{"trait": "Goal-line work", "percentile": 91.0}],
    "weaknesses": [{"trait": "Explosive runs", "percentile": 33.0}],
    "matchup": {"opponent": "KC", "difficulty": "hard"},
    "pace": {"seconds_per_play": 27.1},
    "weather": {"conditions": "clear"},
    "offensive_line": {"rank": 9},
    "warnings": [],
}

PROJECTION_RESULT = {
    "player": "Doubtful Guy",
    "team": "BUF",
    "position": "TE",
    "opponent": "DAL",
    "player_id": "st-doubtful",
    "projection": {
        "expected_fantasy_points": 11.4,
        "targets": 6.1,
        "receptions": 4.3,
        "receiving_yards": 48.2,
        "receiving_tds": 0.34,
    },
    "confidence": {"label": "Solid sample", "score": 72.0, "low": 7.2, "high": 15.9},
    "volatility_score": 41.0,
    "drivers": ["Target share trending up", "Neutral pass funnel"],
    "analysis": {
        "matchup": {"opponent": "DAL", "difficulty": "neutral"},
        "weather": {"conditions": "dome"},
        "pace": {"seconds_per_play": 26.4},
    },
    "warnings": [],
}


def test_player_analysis_badges_the_analysed_player(status):
    status(LIVE_STATUS)
    text = _text(_run("21_NFL_Player_Analysis.py", session={"nfl_player_analysis": ANALYSIS_RESULT}))
    assert f"**Availability:** {RED} OUT · Out Guy (RB)" in text
    assert "actuals and are unaffected by it" in text


def test_projections_page_badges_the_projected_player(status):
    status(LIVE_STATUS)
    text = _text(_run("23_NFL_Projections.py", session={"nfl_projection": PROJECTION_RESULT}))
    assert f"**Availability:** {YELLOW} Doubtful · Doubtful Guy (TE)" in text


def test_projections_page_does_not_rescale_the_projection_for_a_flagged_player(status):
    """No math changes: the doubtful player's stored 11.4 FP is still 11.4."""
    status(LIVE_STATUS)
    app = _run("23_NFL_Projections.py", session={"nfl_projection": PROJECTION_RESULT})
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Fantasy points"] == "11.4"


# ---------------------------------------------------------------------------
# No page filters, and none of them reach the network directly
# ---------------------------------------------------------------------------
BADGED_PAGES = (
    "21_NFL_Player_Analysis.py",
    "23_NFL_Projections.py",
    "24_NFL_Slate.py",
    "25_Fantasy_Hub.py",
    "27_Fantasy_Season_Tools.py",
    "28_Fantasy_My_Team.py",
    "29_Graph_Lab.py",
    "30_NFL_Betting.py",
)


@pytest.mark.parametrize("page", BADGED_PAGES)
def test_no_page_filters_on_availability(page):
    """The engines own every availability filter. A page that started calling
    ``is_out`` / ``is_unavailable`` / ``exclude_out`` would be making its own
    decision about which rows a user gets to see."""
    source = (PAGES / page).read_text(encoding="utf-8")
    for banned in ("is_out(", "is_unavailable(", "exclude_out(", "adjust_projection_for_status("):
        assert banned not in source, f"{page} filters or rescales on availability: {banned}"


@pytest.mark.parametrize("page", BADGED_PAGES)
def test_no_page_reaches_the_network_itself(page):
    source = (PAGES / page).read_text(encoding="utf-8")
    for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup"):
        assert banned not in source, f"{page} contains a direct network call: {banned}"
