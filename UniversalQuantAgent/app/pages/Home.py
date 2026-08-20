"""Minimal Sports Hub landing page for Universal Quant Agent."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

import streamlit as st
from app.page_runtime import apply_global_theme
from app.style import stacked_card_html

apply_global_theme()

st.markdown('<p class="quant-eyebrow">Universal Quant Agent</p>', unsafe_allow_html=True)
st.title("Sports Hub")
st.caption("Explainable signals, projections, and matchup intelligence.")

destinations = (
    {
        "title": "NFL",
        "kicker": "Football intelligence",
        "body": "Team efficiency, player projections, weekly slates, and matchup edges.",
        "page": "pages/5_NFL_Analysis.py",
        "label": "Open NFL",
        "accent": "cyan-accent",
    },
    {
        "title": "NBA",
        "kicker": "Basketball intelligence",
        "body": "Player form, fused projections, daily slates, and visual shot analysis.",
        "page": "pages/1_NBA_Analysis.py",
        "label": "Open NBA",
        "accent": "gold-accent",
    },
    {
        "title": "Fantasy",
        "kicker": "Season command center",
        "body": "Draft, manage, and optimize a roster with matchup-aware weekly forecasts.",
        "page": "pages/25_Fantasy_Hub.py",
        "label": "Open Fantasy Hub",
        "accent": "cyan-accent",
    },
    {
        "title": "Betting",
        "kicker": "Market intelligence",
        "body": "Transparent prop, parlay, and edge research with visible model context.",
        "page": "pages/8_Prop_Analyzer.py",
        "label": "Open Betting",
        "accent": "gold-accent",
    },
)

for destination in destinations:
    st.markdown(
        stacked_card_html(
            destination["title"],
            destination["body"],
            kicker=destination["kicker"],
            extra_class=f"sports-hub-card {destination['accent']}",
        ),
        unsafe_allow_html=True,
    )
    st.page_link(destination["page"], label=destination["label"], use_container_width=True)

st.caption("Research and educational analysis only — not financial or wagering advice.")
