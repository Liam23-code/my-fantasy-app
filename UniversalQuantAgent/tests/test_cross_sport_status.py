"""Live player-status overlay on the Cross-Sport Tools page (34).

Runs the real Streamlit script through AppTest with the NFL evaluation loader
stubbed, so the OUT-filter and the badge row are exercised deterministically
without depending on the shipped odds.json contents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app import betting_shared
from fantasy import player_status
from streamlit.testing.v1 import AppTest

PAGE = Path(_PROJECT_ROOT) / "app" / "pages" / "34_Cross_Sport_Tools.py"


def _row(player_id, name, *, edge=0.05):
    return {
        "player_id": player_id,
        "name": name,
        "market": "passing_yards",
        "line": 250.5,
        "recommended_side": "over",
        "recommended_edge": edge,
        "recommended_ev": 4.2,
        "confidence": 0.6,
        "risk_tier": "medium",
    }


NFL_ROWS = [
    _row("00-0000001", "Healthy Star", edge=0.09),
    _row("00-0000002", "Ruled Out Guy", edge=0.07),
    _row("00-0000003", "Maybe Playing", edge=0.05),
]


@pytest.fixture()
def stub_nfl(monkeypatch):
    monkeypatch.setattr(
        betting_shared, "load_nfl_evaluations", lambda _key: ([dict(r) for r in NFL_ROWS], [], {})
    )


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


def test_page_renders_and_exposes_the_refresh_button(stub_nfl, status):
    app = _run()
    assert len(app.tabs) == 2  # no new tab was added
    assert any(b.key == "cross_sport_refresh_player_status" for b in app.button)
    assert "Player status overlay not loaded" in _text(app)  # empty-file default


def test_last_updated_strip_appears_once_the_file_is_populated(stub_nfl, status):
    status({"00-0000002": {"status": "OUT", "name": "Ruled Out Guy", "last_updated": "2026-09-01T12:00:00Z"}})
    text = _text(_run())
    assert "Player status overlay ·" in text
    assert "1 player(s) flagged" in text
    assert "updated 2026-09-01 12:00:00 UTC" in text


# --- OUT filtering + badges ------------------------------------------------


def test_out_players_are_dropped_from_the_comparison_selectors(stub_nfl, status):
    status({"00-0000002": {"status": "OUT", "name": "Ruled Out Guy"}})
    app = _run()
    joined = " || ".join(app.selectbox(key="NFL_compare_left").options)
    assert "Ruled Out Guy" not in joined
    assert "Healthy Star" in joined
    assert "Maybe Playing" in joined


def test_availability_row_and_badges_appear_in_the_comparison_table(stub_nfl, status):
    status({"00-0000003": {"status": "QUESTIONABLE", "name": "Maybe Playing"}})
    app = _run()
    tables = [df.value for df in app.dataframe if "Metric" in df.value.columns]
    assert tables
    table = tables[0]
    assert "Availability" in list(table["Metric"].values)
    assert "Line" in list(table["Metric"].values)  # existing rows preserved
    assert len(table.columns) == 3
    avail = table[table["Metric"] == "Availability"].iloc[0]
    cells = {str(avail[c]) for c in table.columns if c != "Metric"}
    assert any("\U0001f7e2" in c for c in cells)  # 🟢 Healthy Star is prop A by default


def test_all_out_leaves_a_neutral_message_not_a_crash(stub_nfl, status):
    status(
        {
            "00-0000001": {"status": "OUT", "name": "Healthy Star"},
            "00-0000002": {"status": "OUT", "name": "Ruled Out Guy"},
            "00-0000003": {"status": "OUT", "name": "Maybe Playing"},
        }
    )
    assert "Not enough available props to compare" in _text(_run())


def test_no_scraping_or_network_imports_in_the_page_source():
    source = PAGE.read_text(encoding="utf-8")
    for banned in ("requests.get", "requests.post", "urlopen", "BeautifulSoup"):
        assert banned not in source
