"""Phase 5: the multi-sport (MLB/NHL/NBA/CFB/CBB) availability overlay end to end.

Three concerns, one file:

* **Engine integration** -- ``modules.<sport>_prop_model`` drops live-OUT players
  and zeroes / trims the rest; ``modules.<sport>_moneyline_model`` grew an opt-in
  ``team_status_penalty`` seam that is byte-identical to the historical result
  until a caller passes it.
* **UI** -- the shared ``app.status_ui`` strip and badges render on the five
  sport betting pages plus Cross-Sport Tools; a click rewrites
  ``multi_sport_status.json``; a dead feed keeps the file.
* **No NFL regression** -- the NFL ``fantasy.player_status`` overlay and its file
  are completely independent of this one.

Nothing here touches the network: the ESPN getter is replaced, so the real
``normalize_espn_injuries`` and the real atomic write still run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fantasy import multi_sport_status as mss
from fantasy import player_status
from fantasy.online import _multi_sport_common

PAGES = Path(_PROJECT_ROOT) / "app" / "pages"

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def status(tmp_path, monkeypatch):
    """Point ``multi_sport_status`` at a writable temp file and hand back a
    ``_set(sport, {name: status})`` helper."""
    path = tmp_path / "multi_sport_status.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mss, "STATUS_PATH", path)
    mss.clear_cache()

    def _set(sport: str, players: dict[str, str]) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.setdefault(sport, {})
        for name, flag in players.items():
            key = "nm:" + "".join(ch for ch in name.lower() if ch.isalpha())
            raw[sport][key] = {"status": flag, "name": name, "source": "espn", "last_updated": "2026-09-05T18:00:00Z"}
        path.write_text(json.dumps(raw), encoding="utf-8")
        mss.clear_cache()

    yield _set
    mss.clear_cache()


ESPN_ONE_OUT = {
    "injuries": [
        {
            "displayName": "Test Team",
            "abbreviation": "TST",
            "injuries": [{"status": "Out", "athlete": {"displayName": "Feed Out Guy", "id": "1"}}],
        }
    ]
}


@pytest.fixture()
def feed_ok(monkeypatch):
    monkeypatch.setattr(_multi_sport_common, "http_get_json", lambda *_a, **_k: ESPN_ONE_OUT)


@pytest.fixture()
def feed_down(monkeypatch):
    monkeypatch.setattr(_multi_sport_common, "http_get_json", lambda *_a, **_k: None)


def _prop(player_name="Test Player", *, category="hits", line=1.5):
    return {
        "player_name": player_name,
        "team": "TST",
        "category": category,
        "line": line,
        "over_price": -110.0,
        "under_price": -110.0,
    }


PROP_MODELS = {
    "mlb": ("modules.mlb_prop_model", "hits"),
    "nhl": ("modules.nhl_prop_model", "shots"),
    "cfb": ("modules.cfb_prop_model", "passing_yards"),
    "cbb": ("modules.cbb_prop_model", "points"),
}


# --------------------------------------------------------------------------- #
# Engine: <sport>_prop_model honours the overlay (mlb / nhl / cfb / cbb)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sport", sorted(PROP_MODELS))
def test_no_overlay_leaves_the_prop_board_unchanged(sport, status):
    module_name, category = PROP_MODELS[sport]
    module = __import__(module_name, fromlist=["evaluate_props"])
    rows = module.evaluate_props([_prop("Healthy One", category=category), _prop("Healthy Two", category=category)])
    assert {r["player_name"] for r in rows} == {"Healthy One", "Healthy Two"}
    assert all(r["status"] == "HEALTHY" for r in rows)


@pytest.mark.parametrize("sport", sorted(PROP_MODELS))
def test_out_players_are_removed_from_the_prop_board(sport, status):
    module_name, category = PROP_MODELS[sport]
    module = __import__(module_name, fromlist=["evaluate_props"])
    status(sport, {"Feed Out Guy": "OUT"})
    rows = module.evaluate_props([_prop("Feed Out Guy", category=category), _prop("Healthy Guy", category=category)])
    assert {r["player_name"] for r in rows} == {"Healthy Guy"}


@pytest.mark.parametrize("sport", sorted(PROP_MODELS))
@pytest.mark.parametrize("flag", ["HOLDOUT", "SUSPENDED"])
def test_holdout_and_suspended_collapse_the_over_probability(sport, flag, status):
    """The model's mean goes to 0 (projection = 0) while the posted line stays,
    so the chance of clearing any positive line collapses toward 0."""
    module_name, category = PROP_MODELS[sport]
    module = __import__(module_name, fromlist=["evaluate_prop"])
    healthy = module.evaluate_prop(_prop("Shaky Guy", category=category, line=2.5))
    status(sport, {"Shaky Guy": flag})
    row = module.evaluate_prop(_prop("Shaky Guy", category=category, line=2.5))
    assert row["status"] == flag
    assert row["line"] == 2.5  # the posted line is untouched
    assert row["model_probability_over"] < 0.05
    assert row["model_probability_over"] < healthy["model_probability_over"]


@pytest.mark.parametrize("sport", sorted(PROP_MODELS))
def test_doubtful_and_questionable_trim_the_over_probability(sport, status):
    module_name, category = PROP_MODELS[sport]
    module = __import__(module_name, fromlist=["evaluate_prop"])
    healthy = module.evaluate_prop(_prop("Trim Guy", category=category, line=2.5))
    status(sport, {"Trim Guy": "DOUBTFUL"})
    doubtful = module.evaluate_prop(_prop("Trim Guy", category=category, line=2.5))
    status(sport, {"Trim Guy": "QUESTIONABLE"})
    questionable = module.evaluate_prop(_prop("Trim Guy", category=category, line=2.5))
    assert doubtful["status"] == "DOUBTFUL"
    assert doubtful["line"] == 2.5 and questionable["line"] == 2.5
    # both trim the over probability, doubtful harder than questionable
    assert doubtful["model_probability_over"] < questionable["model_probability_over"] < healthy["model_probability_over"]


# --------------------------------------------------------------------------- #
# Engine: nba_prop_model honours the overlay (price_prop_comparison path)
# --------------------------------------------------------------------------- #


def _nba_row(player="Nikola Jokic", **overrides):
    row = {
        "player": player,
        "team": "DEN",
        "category": "points",
        "minutes_adjusted_projection": 27.0,
        "confidence_low": 23.0,
        "confidence_high": 31.0,
        "sportsbook_line": 25.5,
    }
    row.update(overrides)
    return row


def test_nba_out_player_removed_and_holdout_zeroed(status):
    from modules.nba_prop_model import index_props_by_player_and_category, price_aware_evaluations, price_prop_comparison

    odds = {"over_price": -110.0, "under_price": -110.0}

    # no overlay -> untouched
    base = price_prop_comparison(_nba_row(), odds)
    assert base["status"] == "HEALTHY"

    status("nba", {"Nikola Jokic": "SUSPENDED"})
    zeroed = price_prop_comparison(_nba_row(), odds)
    assert zeroed["status"] == "SUSPENDED"
    assert zeroed["model_probability_over"] == 0.0

    status("nba", {"Nikola Jokic": "OUT", "Jamal Murray": "HEALTHY"})
    props = [
        {"player_name": "Nikola Jokic", "category": "points", **odds},
        {"player_name": "Jamal Murray", "category": "points", **odds},
    ]
    rows = price_aware_evaluations(
        [_nba_row("Nikola Jokic"), _nba_row("Jamal Murray")],
        index_props_by_player_and_category(props),
    )
    assert {r["player"] for r in rows} == {"Jamal Murray"}


# --------------------------------------------------------------------------- #
# Engine: <sport>_moneyline_model team_status_penalty seam is opt-in
# --------------------------------------------------------------------------- #

_AVERAGES = {
    "KC": {"points_scored_avg": 28.0, "points_allowed_avg": 20.0, "games_played": 6},
    "DEN": {"points_scored_avg": 18.0, "points_allowed_avg": 24.0, "games_played": 6},
}


@pytest.mark.parametrize("module_name", ["modules.nba_moneyline_model", "modules.cfb_moneyline_model", "modules.cbb_moneyline_model"])
def test_project_game_moneyline_penalty_is_byte_identical_by_default(module_name):
    module = __import__(module_name, fromlist=["fair_moneyline"])
    base = module.fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5)
    assert module.fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5, team_status_penalty=None) == base
    assert module.fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5, team_status_penalty={}) == base
    assert module.fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5, team_status_penalty={"KC": 0.0}) == base


@pytest.mark.parametrize("module_name", ["modules.nba_moneyline_model", "modules.cfb_moneyline_model", "modules.cbb_moneyline_model"])
def test_project_game_moneyline_penalty_shades_the_favoured_team(module_name):
    module = __import__(module_name, fromlist=["fair_moneyline"])
    base = module.fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5)
    shaded = module.fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5, team_status_penalty={"KC": 7.0})
    assert shaded["spread"] == round(base["spread"] - 7.0, 2)
    assert shaded["home_win_probability"] < base["home_win_probability"]
    assert shaded["status_penalty"] == {"home": 7.0, "away": 0.0}


@pytest.mark.parametrize("module_name", ["modules.mlb_moneyline_model", "modules.nhl_moneyline_model"])
def test_composite_moneyline_penalty_is_byte_identical_by_default(module_name):
    module = __import__(module_name, fromlist=["fair_moneyline"])
    base = module.fair_moneyline("NYY", "BOS")
    assert module.fair_moneyline("NYY", "BOS", team_status_penalty=None) == base
    assert module.fair_moneyline("NYY", "BOS", team_status_penalty={}) == base
    assert "status_penalty" not in base


@pytest.mark.parametrize("module_name", ["modules.mlb_moneyline_model", "modules.nhl_moneyline_model"])
def test_composite_moneyline_penalty_drops_the_penalised_team(module_name):
    module = __import__(module_name, fromlist=["fair_moneyline"])
    base = module.fair_moneyline("NYY", "BOS")
    shaded = module.fair_moneyline("NYY", "BOS", team_status_penalty={"NYY": 0.3})
    assert shaded["home_composite_rating"] < base["home_composite_rating"]
    assert shaded["home_win_probability"] < base["home_win_probability"]
    assert shaded["status_penalty"] == {"home": 0.3, "away": 0.0}


def test_team_status_penalty_helper_feeds_the_moneyline_models(status):
    from modules.cfb_moneyline_model import fair_moneyline

    status("cfb", {"Star QB": "OUT"})
    roster = {"KC": [{"player_name": "Star QB", "position": "QB"}]}
    penalty = mss.team_status_penalty("cfb", roster, points_by_position={"QB": 7.0})
    assert penalty == {"KC": 7.0}
    shaded = fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5, team_status_penalty=penalty)
    base = fair_moneyline("KC", "DEN", averages=_AVERAGES, margin_stdev=13.5)
    assert shaded["spread"] < base["spread"]


# --------------------------------------------------------------------------- #
# UI: the shared strip + badges on the five sport pages + Cross-Sport Tools
# --------------------------------------------------------------------------- #

#: page file -> (selected-sport session state, refresh-button key)
STRIP_PAGES = {
    "31_NBA_Betting.py": (None, "nba_betting_refresh_player_status"),
    "32_CFB_Betting.py": (None, "cfb_betting_refresh_player_status"),
    "33_CBB_Betting.py": (None, "cbb_betting_refresh_player_status"),
    "35_MLB_Betting.py": (None, "mlb_betting_refresh_player_status"),
    "36_NHL_Betting.py": (None, "nhl_betting_refresh_player_status"),
    "34_Cross_Sport_Tools.py": ({"cross_sport_tools_compare_sport": "MLB"}, "cross_sport_mlb_refresh_player_status"),
}


def _run(filename, session=None, timeout=180):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PAGES / filename), default_timeout=timeout)
    for key, value in (session or {}).items():
        app.session_state[key] = value
    app.run()
    raised = [str(e.value) for e in app.exception]
    assert raised == [], f"{filename} raised: {raised}"
    return app


def _text(app):
    blocks = [str(e.value) for e in app.markdown]
    blocks += [str(e.value) for e in app.caption]
    blocks += [str(e.value) for e in app.success]
    blocks += [str(e.value) for e in app.warning]
    return "\n".join(blocks)


@pytest.mark.parametrize("page", sorted(STRIP_PAGES))
def test_every_multi_sport_page_uses_the_shared_strip(page):
    source = (PAGES / page).read_text(encoding="utf-8")
    assert "from app.status_ui import" in source
    assert "render_multi_sport_status_strip(" in source


@pytest.mark.parametrize("page", sorted(STRIP_PAGES))
def test_the_refresh_button_renders_with_its_key(page, status):
    session, key = STRIP_PAGES[page]
    app = _run(page, session=session)
    button = next((b for b in app.button if b.key == key), None)
    assert button is not None, f"{page} has no multi-sport Refresh button ({key})"
    assert button.label == "Refresh Player Status"


@pytest.mark.parametrize("page", sorted(STRIP_PAGES))
def test_the_not_loaded_caption_renders(page, status):
    session, _key = STRIP_PAGES[page]
    assert "player status overlay not loaded" in _text(_run(page, session=session))


def test_button_keys_are_unique():
    assert len({key for _s, key in STRIP_PAGES.values()}) == len(STRIP_PAGES)


def test_clicking_refresh_writes_the_multi_sport_file(status, feed_ok):
    app = _run("35_MLB_Betting.py")
    app.button(key="mlb_betting_refresh_player_status").click().run(timeout=180)
    assert [str(e.value) for e in app.exception] == []
    written = json.loads(mss.STATUS_PATH.read_text(encoding="utf-8"))
    assert written["mlb"]["nm:feedoutguy"]["status"] == "OUT"
    assert written["mlb"]["nm:feedoutguy"]["source"] == "espn"


def test_clicking_refresh_then_reports_the_new_count(status, feed_ok):
    app = _run("36_NHL_Betting.py")
    app.button(key="nhl_betting_refresh_player_status").click().run(timeout=180)
    text = _text(app)
    assert "NHL player status overlay ·" in text
    assert "1 player(s) flagged" in text


def test_a_dead_feed_keeps_the_existing_file_and_warns(status, feed_down):
    mss.STATUS_PATH.write_text(json.dumps({"mlb": {"nm:known": {"status": "QUESTIONABLE", "name": "Known", "last_updated": "2026-08-20T08:00:00Z"}}}), encoding="utf-8")
    mss.clear_cache()
    before = mss.STATUS_PATH.read_text(encoding="utf-8")
    app = _run("35_MLB_Betting.py")
    app.button(key="mlb_betting_refresh_player_status").click().run(timeout=180)
    assert [str(e.value) for e in app.exception] == []
    assert mss.STATUS_PATH.read_text(encoding="utf-8") == before
    assert [str(e.value) for e in app.error] == []


def test_badges_read_from_the_multi_sport_overlay(status):
    from app.status_ui import multi_sport_live_badge, multi_sport_status_badge

    status("mlb", {"Flagged Bat": "DOUBTFUL"})
    assert "Doubtful" in multi_sport_live_badge("MLB", {"player_name": "Flagged Bat"})
    assert "Healthy" in multi_sport_live_badge("MLB", {"player_name": "Healthy Bat"})
    # effective_status also falls back to a row's own field when the feed is silent
    assert "OUT" in multi_sport_status_badge("MLB", {"player_name": "Nobody", "status": "out"})
    # ...but a different sport's slice never leaks in
    assert "Healthy" in multi_sport_live_badge("NHL", {"player_name": "Flagged Bat"})


def test_player_availability_expander_is_on_every_sport_page():
    for page in ("31_NBA_Betting.py", "32_CFB_Betting.py", "33_CBB_Betting.py", "35_MLB_Betting.py", "36_NHL_Betting.py"):
        source = (PAGES / page).read_text(encoding="utf-8")
        assert 'st.expander("Player availability")' in source
        assert "render_multi_sport_player_badges(" in source


# --------------------------------------------------------------------------- #
# No NFL regression
# --------------------------------------------------------------------------- #


def test_the_two_overlays_use_separate_files_and_caches():
    assert player_status.STATUS_PATH != mss.STATUS_PATH
    assert player_status.STATUS_PATH.name == "player_status.json"
    assert mss.STATUS_PATH.name == "multi_sport_status.json"
    assert player_status._cache is not mss._cache


def test_multi_sport_data_does_not_move_the_nfl_overlay(status):
    status("nba", {"Some NBA Guy": "OUT"})
    # the NFL helper is keyed on its own (empty, isolated) file
    assert player_status.has_status_data() is False
    assert player_status.live_status({"name": "Some NBA Guy"}) == "HEALTHY"


def test_nfl_betting_prop_model_is_untouched():
    """betting.prop_model (NFL) still filters on fantasy.player_status only."""
    import inspect

    from betting import prop_model

    source = inspect.getsource(prop_model)
    assert "multi_sport_status" not in source
    assert "status_utils" in source  # still the NFL player_status_utils adapter
