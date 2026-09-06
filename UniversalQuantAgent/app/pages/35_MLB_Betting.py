import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

_FANTASY_ENGINE_ROOT = Path(__file__).resolve().parents[3] / "fantasy_engine"
if _FANTASY_ENGINE_ROOT.is_dir() and str(_FANTASY_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FANTASY_ENGINE_ROOT))

# MLB Betting: offline, deterministic fair-line analysis -- Money Lines,
# Props, Parlays -- for MLB only. Flagship sport of this build cycle (see
# mlb_pipeline.md): a full season-average model + five DFS matchup
# modules + a fusion layer live in modules/mlb_*.py as an optional overlay
# on top of the three tabs below, which price whatever line a props file
# carries directly (the same contract-required pattern CFB/CBB use).
#
# No live or keyless MLB stats/odds source was integrated this cycle --
# data/mlb_*.json ship empty by design; upload a file below to use this
# page. Nothing on this page ever fetches a sportsbook, and nothing here
# places a bet.

import streamlit as st

from app.betting_shared import load_mlb_evaluations, render_offline_moneylines_tab, render_offline_props_tab, render_parlay_builder
from app.page_runtime import apply_global_theme, page_header
from app.status_ui import render_multi_sport_player_badges, render_multi_sport_status_strip

apply_global_theme()

page_header(
    "MLB Betting",
    "Offline fair-line analysis for MLB props, money lines, and parlays -- odds come only from a file you "
    "upload here, never a sportsbook. Nothing on this page places a bet.",
    eyebrow="Betting · MLB",
)

render_multi_sport_status_strip(
    "MLB",
    "mlb_betting_refresh_player_status",
    hint="prop pricing uses the odds file exactly as loaded. ",
)

moneylines_tab, props_tab, parlays_tab = st.tabs(["Money Lines", "Props", "Parlays"])

mlb_prop_evaluations, mlb_game_evaluations, mlb_extra = load_mlb_evaluations("betting_props_upload_mlb", "betting_game_odds_upload_mlb")

with moneylines_tab:
    render_offline_moneylines_tab("MLB", mlb_game_evaluations, icon="⚾")

with props_tab:
    render_offline_props_tab("MLB", mlb_prop_evaluations, mlb_extra["raw_props"], icon="⚾")

with parlays_tab:
    render_parlay_builder(prop_evaluations=mlb_prop_evaluations, sport_key="MLB", empty_icon="⚾")

# One badge per priced player. Live-OUT players are already dropped by the MLB
# prop model, so anything flagged here is still on the board -- the badge is
# context, not a filter (mirrors the NFL betting page exactly).
with st.expander("Player availability"):
    st.caption(
        "Live availability for every player priced above. Ruled-out players are already excluded by the "
        "prop model, so anything flagged here is still bettable — the badge is context, not a filter."
    )
    _seen: set[str] = set()
    _priced_players: list[dict] = []
    for _row in mlb_prop_evaluations:
        _name = str(_row.get("player_name") or "")
        if _name and _name not in _seen:
            _seen.add(_name)
            _priced_players.append({"player_name": _name, "team": _row.get("team")})
    render_multi_sport_player_badges(
        "MLB",
        _priced_players,
        limit=40,
        empty_note="No priced props loaded, so there is nothing to check.",
    )
