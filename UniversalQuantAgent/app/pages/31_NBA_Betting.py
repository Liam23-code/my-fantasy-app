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

# NBA Betting: offline, deterministic fair-line analysis -- Money Lines,
# Props, Parlays -- for NBA only. One of six sport-specific betting pages
# split out of the former single 30_Betting_Engine.py (see
# ui_betting_tabs.md). Player Comparison and Cross-Sport Parlay live on
# the separate 34_Cross_Sport_Tools.py page.
#
# Odds come from exactly two places, both offline: our own default file
# (data/nba_props.json, generated from real per-game rates) and an
# optional file you upload here, which overrides matching entries and
# leaves everything else untouched. Game odds have no fixed default
# (NBA's daily schedule changes every day). Nothing on this page ever
# fetches a sportsbook, and nothing here places a bet.
#
# Props here go through the matchup-aware engine (see matchup_engine.md):
# real opponent-specific game logs, real defensive difficulty (rim/
# perimeter, from player-tracking defense data), real teammate on/off
# splits, and real opponent-injury context all feed the same underlying
# fuse_projection/compare_props pipeline this page already priced from --
# nothing here was refactored to add them; see modules/nba_matchup_engine.py.

import streamlit as st

from app.betting_shared import load_nba_evaluations, render_nba_moneylines_tab, render_nba_props_tab, render_parlay_builder
from app.page_runtime import apply_global_theme, page_header
from app.status_ui import render_multi_sport_player_badges, render_multi_sport_status_strip

apply_global_theme()

page_header(
    "NBA Betting",
    "Offline fair-line analysis for NBA props, money lines, and parlays -- odds come only from our own "
    "default file or a file you upload here, never a sportsbook. Nothing on this page places a bet.",
    eyebrow="Betting · NBA",
)

render_multi_sport_status_strip(
    "NBA",
    "nba_betting_refresh_player_status",
    hint="prop pricing uses the matchup engine's projections exactly as computed. ",
)

moneylines_tab, props_tab, parlays_tab = st.tabs(["Money Lines", "Props", "Parlays"])

nba_priced_props, nba_raw_props, nba_game_evaluations, nba_extra = load_nba_evaluations(
    "betting_props_upload_nba", "betting_game_odds_upload_nba"
)
nba_season = nba_extra["season"]
nba_todays_games = nba_extra["todays_games"]

with moneylines_tab:
    render_nba_moneylines_tab(nba_todays_games, nba_game_evaluations, nba_season)

with props_tab:
    render_nba_props_tab(nba_priced_props, nba_raw_props, nba_season)

with parlays_tab:
    render_parlay_builder(prop_evaluations=nba_priced_props, sport_key="NBA", empty_icon="🏀")

# One badge per priced player. Live-OUT players are already dropped by the NBA
# prop model, so anything flagged here is still on the board -- the badge is
# context, not a filter (mirrors the NFL betting page exactly).
with st.expander("Player availability"):
    st.caption(
        "Live availability for every player priced above. Ruled-out players are already excluded by the "
        "prop model, so anything flagged here is still bettable — the badge is context, not a filter."
    )
    _seen: set[str] = set()
    _priced_players: list[dict] = []
    for _row in nba_priced_props:
        _name = str(_row.get("player") or _row.get("player_name") or "")
        if _name and _name not in _seen:
            _seen.add(_name)
            _priced_players.append({"player_name": _name, "team": _row.get("team")})
    render_multi_sport_player_badges(
        "NBA",
        _priced_players,
        limit=40,
        empty_note="No priced props loaded, so there is nothing to check.",
    )
