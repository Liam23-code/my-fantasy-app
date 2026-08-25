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

# CBB Betting: offline, deterministic fair-line analysis -- Money Lines,
# Props, Parlays -- for College Basketball only. One of four sport-
# specific betting pages split out of the former single
# 30_Betting_Engine.py (see ui_betting_tabs.md). Player Comparison and
# Cross-Sport Parlay live on the separate 34_Cross_Sport_Tools.py page.
#
# Real data comes from ESPN's public JSON API -- no key required; see
# cbb_pipeline.md. Nothing on this page ever fetches a sportsbook, and
# nothing here places a bet.

import streamlit as st

from app.betting_shared import load_cbb_evaluations, render_college_moneylines_tab, render_college_props_tab, render_parlay_builder
from app.page_runtime import apply_global_theme, page_header

apply_global_theme()

page_header(
    "CBB Betting",
    "Offline fair-line analysis for CBB props, money lines, and parlays -- odds come only from our own "
    "default file or a file you upload here, never a sportsbook. Nothing on this page places a bet.",
    eyebrow="Betting · CBB",
)

moneylines_tab, props_tab, parlays_tab = st.tabs(["Money Lines", "Props", "Parlays"])

cbb_prop_evaluations, cbb_game_evaluations, cbb_extra = load_cbb_evaluations("betting_props_upload_cbb", "betting_game_odds_upload_cbb")

with moneylines_tab:
    render_college_moneylines_tab("CBB", cbb_game_evaluations, cbb_extra)

with props_tab:
    render_college_props_tab("CBB", cbb_prop_evaluations, cbb_extra["raw_props"])

with parlays_tab:
    render_parlay_builder(prop_evaluations=cbb_prop_evaluations, sport_key="CBB", empty_icon="🏀")
