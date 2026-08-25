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

# NHL Betting: offline, deterministic fair-line analysis -- Money Lines,
# Props, Parlays -- for NHL only. Lightweight secondary sport this cycle
# (see nhl_pipeline.md): no season/matchup/fusion layer, the same minimal
# shape CFB/CBB started from.
#
# No live or keyless NHL stats/odds source was integrated this cycle --
# data/nhl_*.json ship empty by design; upload a file below to use this
# page. Nothing on this page ever fetches a sportsbook, and nothing here
# places a bet.

import streamlit as st

from app.betting_shared import load_nhl_evaluations, render_offline_moneylines_tab, render_offline_props_tab, render_parlay_builder
from app.page_runtime import apply_global_theme, page_header

apply_global_theme()

page_header(
    "NHL Betting",
    "Offline fair-line analysis for NHL props, money lines, and parlays -- odds come only from a file you "
    "upload here, never a sportsbook. Nothing on this page places a bet.",
    eyebrow="Betting · NHL",
)

moneylines_tab, props_tab, parlays_tab = st.tabs(["Money Lines", "Props", "Parlays"])

nhl_prop_evaluations, nhl_game_evaluations, nhl_extra = load_nhl_evaluations("betting_props_upload_nhl", "betting_game_odds_upload_nhl")

with moneylines_tab:
    render_offline_moneylines_tab("NHL", nhl_game_evaluations, icon="🏒")

with props_tab:
    render_offline_props_tab("NHL", nhl_prop_evaluations, nhl_extra["raw_props"], icon="🏒")

with parlays_tab:
    render_parlay_builder(prop_evaluations=nhl_prop_evaluations, sport_key="NHL", empty_icon="🏒")
