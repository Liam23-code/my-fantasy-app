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

# CFB Betting: offline, deterministic fair-line analysis -- Money Lines,
# Props, Parlays -- for College Football only. One of four sport-specific
# betting pages split out of the former single 30_Betting_Engine.py (see
# ui_betting_tabs.md). Player Comparison and Cross-Sport Parlay live on
# the separate 34_Cross_Sport_Tools.py page.
#
# Real data needs a College Football Data API key (set CFBD_API_KEY) --
# ships empty-by-design otherwise; see cfb_pipeline.md. Nothing on this
# page ever fetches a sportsbook, and nothing here places a bet.

import streamlit as st

from app.betting_shared import load_cfb_evaluations, render_college_moneylines_tab, render_college_props_tab, render_parlay_builder
from app.page_runtime import apply_global_theme, page_header

apply_global_theme()

page_header(
    "CFB Betting",
    "Offline fair-line analysis for CFB props, money lines, and parlays -- odds come only from our own "
    "default file or a file you upload here, never a sportsbook. Nothing on this page places a bet.",
    eyebrow="Betting · CFB",
)

moneylines_tab, props_tab, parlays_tab = st.tabs(["Money Lines", "Props", "Parlays"])

with st.expander("CFB week", expanded=False):
    cfb_season = st.number_input("Season", min_value=2015, max_value=2030, value=2025, key="cfb_season_input")
    cfb_week = st.number_input("Week", min_value=1, max_value=15, value=1, key="cfb_week_input")

cfb_prop_evaluations, cfb_game_evaluations, cfb_extra = load_cfb_evaluations(
    "betting_props_upload_cfb", "betting_game_odds_upload_cfb", season=int(cfb_season), week=int(cfb_week)
)
cfb_extra["season"] = int(cfb_season)

with moneylines_tab:
    render_college_moneylines_tab("CFB", cfb_game_evaluations, cfb_extra)

with props_tab:
    render_college_props_tab("CFB", cfb_prop_evaluations, cfb_extra["raw_props"])

with parlays_tab:
    render_parlay_builder(prop_evaluations=cfb_prop_evaluations, sport_key="CFB", empty_icon="🏈")
