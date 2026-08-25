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

apply_global_theme()

page_header(
    "MLB Betting",
    "Offline fair-line analysis for MLB props, money lines, and parlays -- odds come only from a file you "
    "upload here, never a sportsbook. Nothing on this page places a bet.",
    eyebrow="Betting · MLB",
)

moneylines_tab, props_tab, parlays_tab = st.tabs(["Money Lines", "Props", "Parlays"])

mlb_prop_evaluations, mlb_game_evaluations, mlb_extra = load_mlb_evaluations("betting_props_upload_mlb", "betting_game_odds_upload_mlb")

with moneylines_tab:
    render_offline_moneylines_tab("MLB", mlb_game_evaluations, icon="⚾")

with props_tab:
    render_offline_props_tab("MLB", mlb_prop_evaluations, mlb_extra["raw_props"], icon="⚾")

with parlays_tab:
    render_parlay_builder(prop_evaluations=mlb_prop_evaluations, sport_key="MLB", empty_icon="⚾")
