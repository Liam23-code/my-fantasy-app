"""Streamlit dashboard for Universal Quant Agent.

Run with ``streamlit run app/app.py`` from the UniversalQuantAgent directory.
The page functions only handle presentation; all calculations remain inside the
domain modules so a future API, mobile app, or ML pipeline can reuse them.
"""

from __future__ import annotations

from pathlib import Path
from html import escape
import re
import sys
from typing import Any, Callable

# Streamlit executes this file from app/. Add the project root for domain imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nba_api.stats.static import players as nba_players, teams as nba_teams

from modules.analyzer import rank_opportunities
from modules.badge_graph import (
    render_badge_graph,
    render_spider_badge_graph,
)
from modules.radar_chart import RADAR_WINDOWS, render_efficiency_radar
from modules.shot_chart import SHOT_MODES, render_shot_chart
from modules.trend_graphs import (
    render_momentum_line,
    render_performance_timeline,
    render_usage_efficiency_scatter,
)
from modules.correlation_engine import (
    compute_player_correlations,
    compute_slate_correlations,
    compute_team_correlations,
)
from modules.edge_heatmap import EDGE_CATEGORIES, prepare_edge_heatmap
from modules.data_quality import as_dict, safe_number
from modules.finance import analyze_stock
from modules.insights_engine import (
    generate_badge_insights,
    generate_correlation_insights,
    generate_edge_insights,
    generate_player_insights,
    generate_similarity_insights,
)
from modules.nfl import compare_nfl_teams, latest_completed_nfl_season
from modules.nba_advanced import compare_players, latest_season
from modules.nba_cache import (
    ENHANCED_FALLBACK_WARNING,
    FALLBACK_WARNING,
)
from modules.player_finder import (
    high_efficiency_players,
    high_usage_players,
    top_assist_players,
    top_rebounders,
    top_scorers,
)
from modules.reliability import get_reliability_score
from modules.similarity_engine import compute_player_similarity
from modules.sports import compare_teams, fetch_nba_player_stats


st.set_page_config(
    page_title="Universal Quant Agent",
    page_icon="📊",
    layout="wide",
)


TEAM_COLORS = {
    "ATL":"#E03A3E","BOS":"#007A33","BKN":"#111111","CHA":"#1D1160",
    "CHI":"#CE1141","CLE":"#860038","DAL":"#00538C","DEN":"#0E2240",
    "DET":"#C8102E","GSW":"#1D428A","HOU":"#CE1141","IND":"#002D62",
    "LAC":"#C8102E","LAL":"#552583","MEM":"#5D76A9","MIA":"#98002E",
    "MIL":"#00471B","MIN":"#0C2340","NOP":"#0C2340","NYK":"#006BB6",
    "OKC":"#007AC1","ORL":"#0077C0","PHI":"#006BB6","PHX":"#1D1160",
    "POR":"#E03A3E","SAC":"#5A2D81","SAS":"#2F2F2F","TOR":"#CE1141",
    "UTA":"#002B5C","WAS":"#002B5C",
}


TEAM_IDS = {
    str(team["abbreviation"]).upper(): int(team["id"])
    for team in nba_teams.get_teams()
}
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def inject_app_css() -> None:
    """Apply one modern visual system across every native Streamlit page."""
    st.markdown("""
    <style>
      :root {
        --ink: #10213a;
        --muted: #64748b;
        --surface: rgba(255,255,255,.94);
        --line: #dbe5f0;
        --accent: #ff5a36;
        --accent-2: #5b6cff;
      }
      .stApp {
        background:
          radial-gradient(circle at 82% 3%, rgba(91,108,255,.12), transparent 28rem),
          radial-gradient(circle at 9% 12%, rgba(255,90,54,.10), transparent 24rem),
          linear-gradient(145deg, #f8fafc 0%, #eef4fa 100%);
        color: var(--ink);
      }
      .block-container {padding-top: 2.1rem; padding-bottom: 4rem; max-width: 1480px;}
      [data-testid="stSidebar"],
      [data-testid="stSidebar"] > div,
      [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0d1728 0%, #17243a 100%) !important;
      }
      [data-testid="stSidebar"] a,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {color: #f8fafc !important;}
      [data-testid="stSidebarNav"] a {
        border-radius: 10px;
        margin: .12rem .45rem;
        transition: background .15s ease, transform .15s ease;
      }
      [data-testid="stSidebarNav"] a:hover {background: rgba(255,255,255,.08);}
      [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, #ff5a36, #ff8a3d) !important;
        box-shadow: 0 8px 22px rgba(255,90,54,.25);
      }
      [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        padding: 1rem 1.05rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,.055);
        min-height: 112px;
      }
      [data-testid="stMetricLabel"] {color: var(--muted); font-weight: 650;}
      [data-testid="stMetricValue"] {color: var(--ink); letter-spacing: -.035em;}
      [data-testid="stForm"] {
        background: rgba(255,255,255,.84);
        border: 1px solid var(--line);
        padding: 1.25rem;
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(15,23,42,.045);
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
      }
      .stButton button, [data-testid="stFormSubmitButton"] button {
        border: 0;
        border-radius: 11px;
        font-weight: 700;
        padding: .55rem 1rem;
        background: linear-gradient(90deg, #ff5a36, #ff7d3b);
        color: white;
        box-shadow: 0 7px 18px rgba(255,90,54,.20);
      }
      .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        color: white;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        background: rgba(226,232,240,.72);
        border-radius: 13px;
        padding: .3rem;
      }
      .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: .45rem .85rem;
      }
      .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 3px 10px rgba(15,23,42,.08);
      }
      .team-card, .profile-hero {
        background: linear-gradient(135deg, white 0%, color-mix(in srgb, var(--team) 8%, white) 100%);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        border: 1px solid var(--line);
        border-left: 7px solid var(--team);
        box-shadow: 0 9px 26px rgba(15,23,42,.075);
        margin-bottom: .9rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        min-height: 104px;
      }
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.94);
        border-color: var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 9px 26px rgba(15,23,42,.065);
      }
      .profile-copy {
        border-left: 6px solid var(--team);
        padding: .38rem 0 .38rem 1rem;
        min-height: 68px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }
      .profile-copy h3 {margin: 0; color: #111827; font-size: 1.35rem;}
      .profile-copy p {margin: .25rem 0 0; color: #64748b;}
      .feature-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem;
        min-height: 154px;
        box-shadow: 0 9px 26px rgba(15,23,42,.055);
      }
      .feature-card h3 {margin:.1rem 0 .5rem;color:#0f172a;}
      .feature-card p {color:#64748b;line-height:1.55;}
      .rank-chip {
        display:inline-block;
        background:#eef2ff;
        color:#312e81;
        border:1px solid #dfe3ff;
        border-radius:999px;
        padding:.3rem .62rem;
        margin:.16rem .16rem .16rem 0;
        font-size:.79rem;
        font-weight:700;
      }
      .section-kicker {
        color:#ff5a36;
        font-size:.76rem;
        text-transform:uppercase;
        letter-spacing:.13em;
        font-weight:800;
        margin-bottom:.15rem;
      }
      h1, h2, h3 {letter-spacing: -.025em; color: var(--ink);}
      h1 {font-weight: 820; margin-bottom: .35rem;}
      h2 {margin-top: 1.7rem;}
      hr {border-color: var(--line); margin: 1.7rem 0;}
      [data-testid="stMetric"], .feature-card, [data-testid="stForm"],
      [data-testid="stDataFrame"], [data-testid="stExpander"] {
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
      }
      [data-testid="stMetric"]:hover, .feature-card:hover,
      [data-testid="stDataFrame"]:hover, [data-testid="stExpander"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 13px 32px rgba(15,23,42,.09);
        border-color: #c8d5e5;
      }
      [data-testid="stExpander"] {border-radius: 14px !important; overflow: hidden;}
      [data-testid="stExpanderDetails"] {animation: premiumReveal .18s ease-out;}
      @keyframes premiumReveal {from {opacity:.35; transform:translateY(-3px)} to {opacity:1; transform:none}}
      .stButton button, [data-testid="stFormSubmitButton"] button,
      .stTabs [data-baseweb="tab"] {transition: transform .16s ease, box-shadow .16s ease, background .16s ease;}
      .stTabs [data-baseweb="tab"]:hover {transform: translateY(-1px); background: rgba(255,255,255,.72);}
      input:focus, textarea:focus {box-shadow: 0 0 0 3px rgba(91,108,255,.12) !important;}
      .gradient-section {
        background: linear-gradient(110deg, #10213a 0%, #273e66 62%, #5b6cff 100%);
        color: white; border-radius: 16px; padding: .82rem 1.05rem;
        margin: 1.35rem 0 .85rem; box-shadow: 0 9px 24px rgba(16,33,58,.14);
      }
      .gradient-section h3 {color:white; margin:0; font-size:1.04rem; letter-spacing:-.01em;}
      .gradient-section p {margin:.18rem 0 0; color:#dbe7ff; font-size:.86rem;}
      .insights-panel {
        background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(238,242,255,.9));
        border: 1px solid #d7e0f2; border-left: 5px solid #5b6cff;
        border-radius: 16px; padding: .95rem 1.05rem .75rem; margin: .75rem 0 1.2rem;
        box-shadow: 0 8px 22px rgba(38,58,89,.055);
      }
      .insights-panel h4 {margin:0 0 .55rem; color:#17233b; font-size:.98rem;}
      .insights-panel ul {margin:.15rem 0 .2rem; padding-left:1.2rem;}
      .insights-panel li {margin:.46rem 0; color:#43536c; line-height:1.48;}
      .confidence-card {
        border:1px solid color-mix(in srgb, var(--confidence) 35%, #dbe5f0);
        background:linear-gradient(135deg, white, color-mix(in srgb, var(--confidence) 8%, white));
        border-radius:16px; padding:.9rem 1rem; min-height:92px;
        box-shadow:0 8px 22px rgba(15,23,42,.055); transition:transform .18s ease;
      }
      .confidence-card:hover {transform:translateY(-2px);}
      .confidence-card .label {color:#64748b; font-size:.78rem; font-weight:750; text-transform:uppercase; letter-spacing:.07em;}
      .confidence-card .score {color:var(--confidence); font-size:1.75rem; font-weight:820; letter-spacing:-.04em; margin-top:.2rem;}
      .confidence-track {height:11px; background:#e7edf5; border-radius:999px; overflow:hidden; margin:.55rem 0 .2rem;}
      .confidence-fill {height:100%; width:var(--score); background:linear-gradient(90deg, color-mix(in srgb, var(--confidence) 72%, white), var(--confidence)); border-radius:999px; transition:width .45s ease;}
      .confidence-caption {display:flex; justify-content:space-between; color:#64748b; font-size:.79rem; font-weight:650;}
      [data-testid="stImage"] img {transition:transform .18s ease, filter .18s ease, box-shadow .18s ease;}
      [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,.045) !important;
        border: 1px solid rgba(255,255,255,.10) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        margin: .34rem .15rem;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"]:hover {
        transform: none;
        border-color: rgba(255,255,255,.18) !important;
        background: rgba(255,255,255,.065) !important;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background: transparent !important;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
      [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
        font-weight: 700;
      }
      [data-testid="stSidebar"] [data-testid="stPageLink"] a {
        border-radius: 9px;
        color: #e5edf8 !important;
        transition: background .15s ease, transform .15s ease;
      }
      [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
        background: rgba(255,255,255,.08);
        transform: translateX(2px);
      }      @media (max-width: 760px) {
        .block-container {padding-top: 1.2rem;}
        .profile-hero img {width: 62px; height: 62px;}
      }
    </style>
    """, unsafe_allow_html=True)


def team_color(abbreviation: str) -> str:
    return TEAM_COLORS.get(str(abbreviation).upper(), "#334155")


def team_logo_url(abbreviation: str) -> str:
    team_id = TEAM_IDS.get(str(abbreviation).upper())
    return (
        f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
        if team_id else ""
    )


def player_id_for_name(player_name: str) -> int | None:
    query = str(player_name).strip().lower()
    if not query:
        return None
    exact = [
        player for player in nba_players.get_players()
        if str(player.get("full_name", "")).lower() == query
    ]
    if exact:
        return int(exact[0]["id"])
    partial = [
        player for player in nba_players.get_players()
        if query in str(player.get("full_name", "")).lower()
    ]
    return int(partial[0]["id"]) if len(partial) == 1 else None


def player_headshot_url(player_name: str) -> str:
    player_id = player_id_for_name(player_name)
    return (
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        if player_id else ""
    )


def rank_label(rank: Any) -> str:
    if not isinstance(rank, dict) or not rank.get("rank"):
        return "Rank unavailable"
    return f"#{rank['rank']} of {rank.get('out_of', '—')}"


def profile_header(
    name: str,
    abbreviation: str,
    subtitle: str = "",
    prefer_player: bool = True,
) -> None:
    """Render a lightweight identity row with a premium 60px headshot."""
    color = team_color(abbreviation)
    headshot_url = player_headshot_url(name) if prefer_player else ""
    logo_url = team_logo_url(abbreviation)
    meta = str(abbreviation).upper()
    if subtitle:
        meta += f" · {subtitle}"
    identity = re.sub(
        r"[^a-z0-9]+", "-", f"{name}-{abbreviation}-{subtitle}".lower()
    ).strip("-")[:72] or "profile"
    container_key = f"profile-{identity}"
    image_size = 60 if headshot_url else 40
    radius = "50%" if headshot_url else "13px"
    st.markdown(
        f"""
        <style>
          .st-key-{container_key} {{
            padding: .42rem .18rem .55rem;
            margin: .15rem 0 .75rem;
            border-bottom: 1px solid color-mix(in srgb, {color} 18%, #dbe5f0);
          }}
          .st-key-{container_key} [data-testid="stImage"] img {{
            width: {image_size}px !important;
            height: {image_size}px !important;
            object-fit: cover;
            object-position: center 12%;
            border-radius: {radius};
            border: 3px solid {color};
            background: linear-gradient(145deg, white, color-mix(in srgb, {color} 12%, white));
            box-shadow: 0 0 0 4px color-mix(in srgb, {color} 15%, transparent),
                        0 9px 20px rgba(15,23,42,.16);
          }}
          .st-key-{container_key}:hover [data-testid="stImage"] img {{
            transform: translateY(-2px) scale(1.035);
            filter: saturate(1.08);
            box-shadow: 0 0 0 5px color-mix(in srgb, {color} 20%, transparent),
                        0 12px 25px rgba(15,23,42,.20);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key=container_key):
        image_column, text_column = st.columns(
            [0.68, 8], vertical_alignment="center"
        )
        with image_column:
            if headshot_url:
                st.image(headshot_url, width=60)
            elif logo_url:
                st.image(logo_url, width=40)
        with text_column:
            st.markdown(
                f'<div class="profile-copy" style="--team:{color}">'
                f'<h3>{escape(str(name))}</h3>'
                f'<p>{escape(meta)}</p></div>',
                unsafe_allow_html=True,
            )


def team_banner(name: str, abbreviation: str, subtitle: str = "") -> None:
    profile_header(name, abbreviation, subtitle, prefer_player=player_id_for_name(name) is not None)


def nfl_profile_header(name: str, abbreviation: str, subtitle: str = "") -> None:
    """Render an NFL identity row with team color and a compact team logo."""
    from modules.nfl_stats import team_palette as nfl_team_palette

    team = str(abbreviation).upper()
    color, _ = nfl_team_palette(team)
    logo = f"https://a.espncdn.com/i/teamlogos/nfl/500/{team.lower()}.png" if team else ""
    identity = re.sub(r"[^a-z0-9]+", "-", f"nfl-{name}-{team}".lower()).strip("-")[:72]
    with st.container(key=f"profile-{identity}"):
        image_column, text_column = st.columns([0.55, 8], vertical_alignment="center")
        with image_column:
            if logo:
                st.image(logo, width=40)
        with text_column:
            st.markdown(
                f'<div class="profile-copy" style="--team:{color}">'
                f'<h3 style="border-left:4px solid {color};padding-left:.7rem">{escape(str(name))}</h3>'
                f'<p>{escape(team)} · {escape(str(subtitle))}</p></div>',
                unsafe_allow_html=True,
            )


def apply_global_theme() -> None:
    """Emit the shared theme on every Streamlit page rerun."""
    inject_app_css()


def friendly_label(name: str) -> str:
    """Turn dictionary keys into labels that are easy to read."""
    replacements = {"epa": "EPA", "pct": "%", "sma": "SMA", "20d": "20D"}
    words = name.replace("_", " ").split()
    return " ".join(replacements.get(word.lower(), word.title()) for word in words)


def run_analysis(label: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    """Run a provider-backed analysis with consistent progress and errors."""
    try:
        with st.spinner(f"Loading {label} data..."):
            return operation()
    except Exception as error:
        st.error(f"{label} analysis could not be completed: {error}")
        st.info("Check the team/ticker, internet connection, and data-provider availability.")
        return None


def metric_row(values: dict[str, Any], keys: list[str]) -> None:
    """Display selected numeric values as responsive Streamlit metrics."""
    columns = st.columns(len(keys))
    for column, key in zip(columns, keys):
        value = values.get(key, "—")
        column.metric(friendly_label(key), value)


def records_table(records: list[dict[str, Any]], index: str | None = None) -> pd.DataFrame:
    """Build a presentation-friendly table from dictionary records."""
    table = pd.DataFrame(records)
    if table.empty:
        return table
    table = table.rename(columns={column: friendly_label(column) for column in table.columns})
    friendly_index = friendly_label(index) if index else None
    return table.set_index(friendly_index) if friendly_index in table.columns else table


def style_figure(figure: go.Figure, height: int = 430) -> go.Figure:
    """Apply the shared Plotly style used throughout the dashboard."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=52, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.65)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#24344d"),
        title_font=dict(size=18, color="#10213a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font_color="#10213a"),
    )
    return figure


def _confidence_value(value: Any) -> float:
    """Normalize display-only confidence scores to the documented 0–100 scale."""
    return max(0.0, min(100.0, safe_number(value)))


def confidence_color(value: Any) -> str:
    """Return the shared green/yellow/red confidence color."""
    score = _confidence_value(value)
    if score >= 80:
        return "#138a55"
    if score >= 50:
        return "#d6a700"
    return "#d94a3a"


def confidence_ring(
    value: Any,
    title: str = "Confidence",
    key: str | None = None,
) -> go.Figure:
    """Render a compact Plotly confidence ring and return its figure."""
    score = _confidence_value(value)
    color = confidence_color(score)
    figure = go.Figure(
        go.Pie(
            values=[score, 100.0 - score],
            hole=.76,
            sort=False,
            direction="clockwise",
            rotation=0,
            marker=dict(colors=[color, "#e7edf5"], line=dict(width=0)),
            textinfo="none",
            hovertemplate=f"{escape(title)}: %{{value:.1f}}<extra></extra>",
        )
    )
    figure.update_layout(
        height=225,
        margin=dict(l=10, r=10, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(text=title, x=.5, xanchor="center", font=dict(size=15)),
        annotations=[dict(
            text=f"<b>{score:.0f}</b><br><span style='font-size:11px'>out of 100</span>",
            x=.5, y=.5, showarrow=False, font=dict(color=color, size=24),
        )],
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=key,
    )
    return figure


def reliability_bar(value: Any, label: str = "Reliability") -> None:
    """Render a lightweight horizontal score bar."""
    score = _confidence_value(value)
    color = confidence_color(score)
    st.markdown(
        f"""
        <div style="--confidence:{color};--score:{score:.1f}%">
          <div class="confidence-caption"><span>{escape(label)}</span><span>{score:.1f}/100</span></div>
          <div class="confidence-track"><div class="confidence-fill"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_card(value: Any, label: str = "Confidence") -> None:
    """Render a color-coded confidence metric card."""
    score = _confidence_value(value)
    color = confidence_color(score)
    tier = "High" if score >= 80 else "Moderate" if score >= 50 else "Low"
    st.markdown(
        f"""
        <div class="confidence-card" style="--confidence:{color}">
          <div class="label">{escape(label)}</div>
          <div class="score">{score:.1f}</div>
          <div class="confidence-caption"><span>{tier}</span><span>/ 100</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    """Render a compact gradient section heading used across native pages."""
    st.markdown(
        f'<div class="gradient-section"><h3>{escape(title)}</h3>'
        f'<p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_insight_list(
    insights: list[str] | Any,
    title: str = "Quant insights",
) -> None:
    """Render short analytics notes in a premium, skimmable panel."""
    clean = [str(item).strip() for item in insights if str(item).strip()] if isinstance(insights, list) else []
    if not clean:
        return
    bullets = "".join(f"<li>{escape(item)}</li>" for item in clean)
    st.markdown(
        f'<div class="insights-panel"><h4>{escape(title)}</h4><ul>{bullets}</ul></div>',
        unsafe_allow_html=True,
    )

def render_graph_lab_figure(figure: go.Figure | None, key: str) -> None:
    """Render one Graph Lab figure and its provider notes consistently."""
    if figure is None:
        return
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key=key)
    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    for warning in meta.get("warnings", []):
        st.caption(f"Data note: {warning}")

def render_correlation_heatmap(
    payload: dict[str, Any],
    selected_stats: list[str] | None = None,
    key: str = "correlation_heatmap",
) -> None:
    """Render a correlation matrix and gracefully explain partial data."""
    matrix = pd.DataFrame.from_dict(
        as_dict(payload.get("correlation_matrix")), orient="index"
    )
    if selected_stats:
        usable = [
            stat for stat in selected_stats
            if stat in matrix.index and stat in matrix.columns
        ]
        matrix = matrix.loc[usable, usable] if usable else pd.DataFrame()
    if matrix.empty:
        st.info("No variable statistics are available for this heatmap yet.")
    else:
        labels = [friendly_label(column) for column in matrix.columns]
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix.to_numpy(dtype=float),
                x=labels,
                y=[friendly_label(index) for index in matrix.index],
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=[
                    [0.0, "#b42318"],
                    [0.25, "#f7a89b"],
                    [0.5, "#f8fafc"],
                    [0.75, "#87d3bc"],
                    [1.0, "#087f5b"],
                ],
                colorbar=dict(title="Correlation"),
                text=matrix.round(2).to_numpy(),
                texttemplate="%{text:.2f}",
                hovertemplate="%{y} × %{x}<br>r=%{z:.3f}<extra></extra>",
            )
        )
        figure.update_layout(title=f"{payload.get('subject', 'Correlation')} relationships")
        st.plotly_chart(
            style_figure(figure, max(390, 66 * len(matrix) + 130)),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=key,
        )
    render_insight_list(
        generate_correlation_insights(payload), "Correlation highlights"
    )
    for warning in payload.get("warnings", []):
        st.caption(f"Data note: {warning}")


def render_edge_heatmap_view(
    props: list[dict[str, Any]],
    key_prefix: str = "slate_edges",
) -> None:
    """Render controls, heatmap, and detail table for normalized prop edges."""
    available = sorted({
        str(row.get("category", "")).lower()
        for row in props if isinstance(row, dict) and row.get("category")
    })
    defaults = [category for category in EDGE_CATEGORIES if category in available]
    control_one, control_two = st.columns([3, 2])
    categories = control_one.multiselect(
        "Prop categories",
        list(EDGE_CATEGORIES),
        default=defaults or list(EDGE_CATEGORIES[:4]),
        key=f"{key_prefix}_categories",
    )
    sort_label = control_two.selectbox(
        "Sort players by",
        ["Edge", "Reliability", "Confidence"],
        key=f"{key_prefix}_sort",
    )
    payload = prepare_edge_heatmap(props, categories, sort_label.lower())
    matrix = pd.DataFrame.from_dict(as_dict(payload.get("matrix")), orient="index")
    if matrix.empty:
        st.info("No comparable projections and sportsbook lines matched these filters.")
    else:
        extent = max(
            1.0,
            abs(safe_number(payload.get("edge_range", {}).get("min"))),
            abs(safe_number(payload.get("edge_range", {}).get("max"))),
        )
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix.to_numpy(dtype=float),
                x=[friendly_label(column) for column in matrix.columns],
                y=matrix.index.tolist(),
                zmin=-extent,
                zmax=extent,
                zmid=0,
                colorscale=[
                    [0.0, "#b42318"],
                    [0.42, "#f6b3a8"],
                    [0.5, "#f7d154"],
                    [0.58, "#b8dfc4"],
                    [1.0, "#138a55"],
                ],
                colorbar=dict(title="Edge"),
                text=matrix.round(2).to_numpy(),
                texttemplate="%{text:.2f}",
                hovertemplate="%{y}<br>%{x}<br>Edge %{z:+.2f}<extra></extra>",
            )
        )
        figure.update_layout(title="Projection edge versus sportsbook line")
        st.plotly_chart(
            style_figure(figure, max(430, 32 * len(matrix) + 150)),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"{key_prefix}_chart",
        )
    records = payload.get("records", [])
    if records:
        confidence_values = [
            safe_number(row.get("confidence")) for row in records
            if row.get("confidence") is not None
        ]
        reliability_values = [
            safe_number(row.get("reliability")) for row in records
            if row.get("reliability") is not None
        ]
        average_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else 0.0
        )
        average_reliability = (
            sum(reliability_values) / len(reliability_values)
            if reliability_values else 0.0
        )
        ring_column, score_column, reliability_column = st.columns([1, 1, 1])
        with ring_column:
            confidence_ring(
                average_confidence, "Slate confidence", f"{key_prefix}_confidence_ring"
            )
        with score_column:
            confidence_card(average_confidence, "Average confidence")
            reliability_bar(average_confidence, "Confidence")
        with reliability_column:
            confidence_card(average_reliability, "Average reliability")
            reliability_bar(average_reliability, "Reliability")
        render_insight_list(generate_edge_insights(records), "Edge explanations")
        st.dataframe(
            records_table(records),
            use_container_width=True,
            hide_index=True,
        )
    for warning in payload.get("warnings", []):
        st.caption(f"Data note: {warning}")


def render_similarity_view(
    result: dict[str, Any],
    key_prefix: str = "similarity",
) -> None:
    """Display a target profile, ranked matches, and dimension breakdown."""
    target = as_dict(result.get("target"))
    profile_header(
        target.get("player", "Player"),
        target.get("team", ""),
        f"{target.get('role', 'role unavailable')} · {result.get('season', '')}",
    )
    matches = result.get("similar_players", [])
    if not matches:
        st.info("No qualified comparison players were available for this season.")
    else:
        render_insight_list(
            generate_similarity_insights(result), "Similarity context"
        )
        featured = st.columns(min(3, len(matches)))
        for column, match in zip(featured, matches[:3]):
            with column:
                profile_header(
                    match.get("player", "Player"),
                    match.get("team", ""),
                    f"{safe_number(match.get('similarity_score')):.0f}% similar",
                )
        table_rows = [
            {
                "rank": position,
                "player": row.get("player"),
                "team": row.get("team"),
                "role": row.get("role"),
                "similarity_score": row.get("similarity_score"),
            }
            for position, row in enumerate(matches, start=1)
        ]
        left, right = st.columns([1.2, 1])
        with left:
            st.dataframe(
                records_table(table_rows),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            chart = pd.DataFrame(table_rows).sort_values("similarity_score")
            figure = px.bar(
                chart,
                x="similarity_score",
                y="player",
                orientation="h",
                color="similarity_score",
                color_continuous_scale=["#dbe5f0", "#5b6cff", "#ff5a36"],
                range_color=[0, 100],
                labels={"similarity_score": "Similarity", "player": ""},
                title="Top profile matches",
            )
            figure.update_coloraxes(showscale=False)
            st.plotly_chart(
                style_figure(figure, 430),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key=f"{key_prefix}_bar",
            )

        selected = st.selectbox(
            "Inspect similarity dimensions",
            [row.get("player") for row in matches],
            key=f"{key_prefix}_player",
        )
        selected_row = next(
            row for row in matches if row.get("player") == selected
        )
        breakdown = as_dict(selected_row.get("dimension_scores"))
        if breakdown:
            detail = pd.DataFrame(
                {
                    "Dimension": [friendly_label(key) for key in breakdown],
                    "Similarity": list(breakdown.values()),
                }
            ).sort_values("Similarity")
            figure = px.bar(
                detail,
                x="Similarity",
                y="Dimension",
                orientation="h",
                range_x=[0, 100],
                color="Similarity",
                color_continuous_scale=["#f2b7ad", "#f7d154", "#1c9a67"],
                title=f"Why {selected} is similar",
            )
            figure.update_coloraxes(showscale=False)
            st.plotly_chart(
                style_figure(figure, 430),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key=f"{key_prefix}_dimensions",
            )
    for warning in result.get("warnings", []):
        st.caption(f"Data note: {warning}")


def render_home() -> None:
    """Display project purpose and a quick dashboard guide."""
    st.title("Universal Quant Agent")
    st.subheader("Explainable analysis across markets and sports")
    st.write(
        "Explore finance, NBA, and NFL signals through one modular interface. "
        "Every score uses transparent rules today and is structured for future "
        "prediction models, sentiment data, APIs, and portfolio tools."
    )
    st.info("Educational research only — not financial, wagering, or investment advice.")

    first, second, third, fourth = st.columns(4)
    first.markdown('<div class="feature-card"><div class="section-kicker">Markets</div><h3>Finance</h3><p>Trends, volatility, moving averages, and transparent correlations.</p></div>',unsafe_allow_html=True)
    second.markdown('<div class="feature-card"><div class="section-kicker">Basketball</div><h3>NBA</h3><p>League ranks, player form, fused projections, and matchup context.</p></div>',unsafe_allow_html=True)
    third.markdown('<div class="feature-card"><div class="section-kicker">Football</div><h3>NFL</h3><p>EPA, success rate, efficiency, schedule strength, and mismatches.</p></div>',unsafe_allow_html=True)
    fourth.markdown('<div class="feature-card"><div class="section-kicker">Graph Lab</div><h3>Visualize</h3><p>Shot heatmaps, radar profiles, badge wheels, trends, difficulty, and pace curves.</p></div>',unsafe_allow_html=True)

    st.markdown("### How to use the dashboard")
    st.markdown(
        "1. Choose an analysis page from the sidebar.\n"
        "2. Enter teams or a ticker and run the analysis.\n"
        "3. Visit **Opportunities** to rank every completed result together."
    )


def render_nba_basic(result: dict[str, Any]) -> None:
    """Render color-coded NBA team cards, comparisons, and league ranks."""
    st.subheader(f"NBA matchup · {result['subject']}")
    matchup = result["matchup"]
    first, second = st.columns([1,1])
    first.metric("Season lean",matchup["favored_team"])
    second.metric("Net-rating edge",matchup["net_rating_edge"])
    st.caption(matchup.get("description",""))

    team_columns = st.columns(2)
    for column,team in zip(team_columns,result["teams"]):
        with column:
            team_banner(team["name"],team.get("abbreviation",""),
                        f"{team.get('wins',0)}–{team.get('losses',0)}")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Net rating",team.get("net_rating","—"))
            metric_columns[1].metric("Offense",team.get("offensive_rating","—"))
            metric_columns[2].metric("Defense",team.get("defensive_rating","—"))
            ranks = team.get("league_ranks",{})
            chips = "".join(
                f'<span class="rank-chip">{friendly_label(key)} {rank_label(value)}</span>'
                for key,value in ranks.items())
            st.markdown(chips,unsafe_allow_html=True)

    rows=[]
    for team in result["teams"]:
        row={key:value for key,value in team.items()
             if key not in {"efficiency","league_ranks","abbreviation"}}
        row.update(team.get("efficiency",{}))
        rows.append(row)
    overview_tab, rank_tab, correlation_tab = st.tabs(
        ["Team Comparison", "League Ranks", "Correlation Heatmaps"]
    )
    with overview_tab:
        table = records_table(rows, "name")
        st.dataframe(table, use_container_width=True)
        chart_columns = [
            column for column in
            ["Offensive Rating", "Defensive Rating", "Net Rating"]
            if column in table.columns
        ]
        if chart_columns:
            chart = (
                table[chart_columns]
                .reset_index()
                .melt(id_vars=table.index.name or "Name", var_name="Metric", value_name="Rating")
            )
            team_column = table.index.name or "Name"
            figure = px.bar(
                chart,
                x="Metric",
                y="Rating",
                color=team_column,
                barmode="group",
                title="Team efficiency comparison",
            )
            st.plotly_chart(
                style_figure(figure),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="nba_team_comparison",
            )
    with rank_tab:
        rank_rows = []
        for team in result["teams"]:
            for metric, rank in team.get("league_ranks", {}).items():
                rank_rows.append(
                    {
                        "team": team["name"],
                        "category": friendly_label(metric),
                        "rank": rank.get("rank"),
                        "out_of": rank.get("out_of"),
                    }
                )
        st.dataframe(
            records_table(rank_rows), use_container_width=True, hide_index=True
        )
    with correlation_tab:
        options = {
            team.get("name", team.get("abbreviation", "Team")): team.get("abbreviation", "")
            for team in result["teams"]
        }
        selected_team = st.selectbox(
            "Team profile",
            list(options),
            key="nba_correlation_team",
        )
        if st.button("Build team correlations", key="load_team_correlations"):
            correlations = run_analysis(
                "team correlations",
                lambda: compute_team_correlations(options[selected_team]),
            )
            if correlations:
                st.session_state["team_correlations"] = correlations
        correlations = as_dict(st.session_state.get("team_correlations"))
        if correlations:
            available = correlations.get("available_stats", [])
            selected = st.multiselect(
                "Statistics",
                available,
                default=available,
                key="team_correlation_stats",
            )
            render_correlation_heatmap(
                correlations, selected, "nba_team_correlations"
            )

def render_nba_advanced(advanced: dict[str, Any]) -> None:
    """Render the most useful parts of the nested advanced NBA response."""
    st.markdown("### Advanced matchup")
    matchup = advanced.get("matchup", {})
    first, second, third = st.columns(3)
    first.metric("Adjusted lean", matchup.get("favored_team", "—"))
    second.metric("Adjusted net-rating edge",
                  matchup.get("opponent_adjusted_net_rating_edge", "—"))
    third.metric("Matchup difficulty", f"{matchup.get('difficulty_score', 0):.1f}/100")
    st.caption(matchup.get("explanation", ""))

    rows = []
    for team in advanced.get("teams", []):
        row = {"team": team.get("team")}
        row.update(team.get("adjusted_ratings", {}))
        row.update(team.get("strength_of_schedule", {}))
        row.update({f"clutch_{key}": value for key, value in team.get("clutch", {}).items()
                    if isinstance(value, (int, float))})
        row["playstyle"] = team.get("playstyle", {}).get("cluster", "—")
        rows.append(row)
    if rows:
        st.dataframe(records_table(rows, "team"), use_container_width=True)

    st.markdown("#### Key insights")
    for team in advanced.get("teams", []):
        streaks = team.get("trends", {}).get("streaks", {})
        recent = streaks.get("last_10", {})
        style = team.get("playstyle", {}).get("cluster", "unknown style")
        st.write(
            f"- **{team.get('team', 'Team')}**: {recent.get('label', 'neutral')} over "
            f"the last {recent.get('games', 0)} games ({recent.get('win_pct', 0):.1f}% wins); "
            f"playstyle: {style}."
        )

    with st.expander("Lineups and player on/off splits"):
        for team in advanced.get("teams", []):
            st.markdown(f"**{team.get('team', 'Team')}**")
            lineups = team.get("top_lineups", [])
            on_off = team.get("top_player_on_off_splits", [])
            if lineups:
                st.caption("Top lineups by minutes")
                st.dataframe(records_table(lineups), use_container_width=True)
            if on_off:
                st.caption("Largest player on/off swings")
                st.dataframe(records_table(on_off), use_container_width=True)
            if not lineups and not on_off:
                st.write("No lineup or on/off records were returned.")


def render_nba_page() -> None:
    """Collect NBA inputs, run analysis, and display the saved result."""
    st.title("NBA Analysis")
    st.write("Compare season performance and optionally load deeper matchup context.")
    with st.form("nba_form"):
        left, right = st.columns(2)
        team_one = left.text_input("First NBA team", "DEN")
        team_two = right.text_input("Second NBA team", "BOS")
        include_advanced = st.checkbox(
            "Include advanced analysis",
            help="Adds schedule, clutch, trends, lineups, and player on/off calls.",
        )
        submitted = st.form_submit_button("Analyze NBA matchup", type="primary")
    if submitted:
        result = run_analysis(
            "NBA", lambda: compare_teams(team_one, team_two,
                                         include_advanced=include_advanced))
        if result:
            st.session_state["nba_insights"] = result
    result = st.session_state.get("nba_insights")
    if result:
        render_nba_basic(result)
        if result.get("advanced"):
            render_nba_advanced(result["advanced"])


def render_nfl_page() -> None:
    """Collect NFL inputs and display efficiency plus mismatch insights."""
    st.title("NFL Analysis")
    st.write("Compare NFL teams using public nflverse play-by-play data.")
    with st.form("nfl_form"):
        first, second, third = st.columns([2, 2, 1])
        team_one = first.text_input("First NFL team", "KC")
        team_two = second.text_input("Second NFL team", "BUF")
        season = third.number_input("Season", min_value=1999,
                                    max_value=latest_completed_nfl_season(),
                                    value=latest_completed_nfl_season(), step=1)
        submitted = st.form_submit_button("Analyze NFL matchup", type="primary")
    if submitted:
        result = run_analysis(
            "NFL", lambda: compare_nfl_teams(team_one, team_two, int(season)))
        if result:
            st.session_state["nfl_insights"] = result
    result = st.session_state.get("nfl_insights")
    if not result:
        return

    st.subheader(f"NFL matchup: {result['subject']}")
    matchup = result["matchup"]
    left, middle, right = st.columns(3)
    left.metric("Season lean", matchup["favored_team"])
    middle.metric("Net-efficiency edge", matchup["net_efficiency_edge"])
    right.metric("Matchup difficulty", f"{matchup['difficulty_score']:.1f}/100")

    fields = ["offensive_epa_per_play", "offensive_success_rate",
              "net_efficiency", "turnover_margin", "strength_of_schedule",
              "pace_seconds_per_play", "red_zone_efficiency",
              "third_down_conversion_rate"]
    rows = [{"team": team["name"],
             **{field: team["metrics"].get(field, 0) for field in fields}}
            for team in result["teams"]]
    table = records_table(rows, "team")
    st.dataframe(table, use_container_width=True)
    chart_fields = [friendly_label(field) for field in
                    ["offensive_success_rate", "net_efficiency"]]
    st.bar_chart(table[chart_fields])

    st.markdown("### Key mismatches")
    mismatches = matchup.get("key_mismatches", [])
    if mismatches:
        st.dataframe(records_table(mismatches), use_container_width=True,
                     hide_index=True)
    else:
        st.info("No mismatch crossed the Version 1 reporting threshold.")


def render_finance_page() -> None:
    """Collect a ticker and display market summary, patterns, and correlations."""
    st.title("Finance Analysis")
    st.write("Review explainable price, trend, volatility, and correlation signals.")
    with st.form("finance_form"):
        ticker = st.text_input("Stock ticker", "AAPL").strip().upper()
        submitted = st.form_submit_button("Analyze stock", type="primary")
    if submitted:
        result = run_analysis("finance", lambda: analyze_stock(ticker))
        if result:
            st.session_state["finance_insights"] = result
    result = st.session_state.get("finance_insights")
    if not result:
        return

    st.subheader(f"Market summary: {result['subject']}")
    summary = result["summary"]
    metric_row(summary, ["latest_price", "period_return_pct",
                         "annualized_volatility_pct", "average_volume_20d"])
    st.dataframe(records_table([summary]), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Patterns")
        st.dataframe(records_table(result.get("patterns", [])),
                     use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Correlations")
        correlations = [{"relationship": key, "correlation": value}
                        for key, value in result.get("correlations", {}).items()]
        st.dataframe(records_table(correlations), use_container_width=True,
                     hide_index=True)
        if correlations:
            chart = pd.DataFrame(correlations).set_index("relationship")
            st.bar_chart(chart)


def render_player_page() -> None:
    """Analyze one NBA player or compare two advanced player profiles."""
    st.title("Player Analysis")
    st.write("Review one player's efficiency or compare two league-percentile profiles.")
    with st.form("player_form"):
        first, second = st.columns(2)
        player_one = first.text_input("First NBA player", "Nikola Jokic")
        player_two = second.text_input("Second NBA player (optional)", placeholder="Example: Giannis Antetokounmpo")
        season = st.text_input("Season", latest_season(), help="NBA season format, for example 2025-26.")
        submitted = st.form_submit_button("Analyze player", type="primary")

    if submitted:
        if not player_one.strip():
            st.warning("Enter at least the first player's name.")
        elif player_two.strip():
            result = run_analysis("player comparison", lambda: compare_players(player_one, player_two, season or None))
            if result:
                result["analysis_type"] = "comparison"
                st.session_state["player_insights"] = result
        else:
            result = run_analysis("player", lambda: fetch_nba_player_stats(player_one, season or None))
            if result:
                result["analysis_type"] = "single"
                st.session_state["player_insights"] = result

    result = st.session_state.get("player_insights")
    if not result:
        return
    if result.get("analysis_type") == "comparison":
        render_player_comparison(result)
    else:
        render_single_player(result)


def render_single_player(result: dict[str, Any]) -> None:
    """Display a polished player profile, trends, ranks, and related models."""
    player_name = result.get("player", "Player")
    team = result.get("team", "")
    start_year = int(safe_number(result.get("season"), 0))
    season_label = (
        f"{start_year}-{str(start_year + 1)[-2:]}" if start_year else latest_season()
    )
    profile_header(player_name, team, f"Season {season_label}")
    render_insight_list(
        generate_player_insights(result, {}), "Player insights"
    )

    season_avg = as_dict(result.get("season_avg"))
    last5 = as_dict(result.get("last5_avg"))
    last10 = as_dict(result.get("last10_avg"))
    advanced = as_dict(result.get("advanced"))
    ranks = as_dict(result.get("league_ranks"))
    (
        overview, recent_tab, rank_tab, totals_tab, shot_tab, efficiency_tab,
        badge_tab, trends_tab, similar_tab, correlation_tab,
    ) = st.tabs(
        [
            "Overview",
            "Recent Form",
            "NBA Rankings",
            "Season Totals",
            "Shot Profile",
            "Efficiency",
            "Badge Profile",
            "Trends",
            "Similar Players",
            "Correlation Heatmaps",
        ]
    )

    with overview:
        st.markdown('<div class="section-kicker">Production</div>', unsafe_allow_html=True)
        st.markdown("### Season averages")
        columns = st.columns(4)
        for column, key, label in zip(
            columns,
            ("ppg", "rpg", "apg", "mpg"),
            ("Points", "Rebounds", "Assists", "Minutes"),
        ):
            column.metric(label, season_avg.get(key, "—"))
            column.caption(rank_label(ranks.get(key)))
        st.markdown('<div class="section-kicker">Efficiency</div>', unsafe_allow_html=True)
        advanced_columns = st.columns(3)
        advanced_columns[0].metric("PER estimate", advanced.get("per_estimate", "—"))
        advanced_columns[0].caption(rank_label(ranks.get("per_estimate")))
        advanced_columns[1].metric(
            "True shooting", f"{safe_number(advanced.get('ts_pct')):.1f}%"
        )
        advanced_columns[1].caption(rank_label(ranks.get("ts_pct")))
        advanced_columns[2].metric(
            "Usage rate", f"{safe_number(advanced.get('usage_pct')):.1f}%"
        )
        advanced_columns[2].caption(rank_label(ranks.get("usage_pct")))

    with recent_tab:
        st.markdown("### Season versus recent form")
        comparison = pd.DataFrame(
            {
                "Metric": ["Points", "Rebounds", "Assists", "Minutes"],
                "Season": [season_avg.get(key, 0) for key in ("ppg", "rpg", "apg", "mpg")],
                "Last 5": [last5.get(key, 0) for key in ("ppg", "rpg", "apg", "mpg")],
                "Last 10": [last10.get(key, 0) for key in ("ppg", "rpg", "apg", "mpg")],
            }
        ).melt(id_vars="Metric", var_name="Window", value_name="Value")
        figure = px.bar(
            comparison,
            x="Metric",
            y="Value",
            color="Window",
            barmode="group",
            color_discrete_map={
                "Season": "#263a59",
                "Last 5": "#ff5a36",
                "Last 10": "#5b6cff",
            },
            title="Production by sample window",
        )
        st.plotly_chart(
            style_figure(figure),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="player_recent_comparison",
        )

        trends = pd.DataFrame(result.get("trend_series", []))
        if not trends.empty:
            left, right = st.columns(2)
            with left:
                usage_minutes = trends.melt(
                    id_vars=["game"],
                    value_vars=[column for column in ("usage_pct", "minutes") if column in trends],
                    var_name="Metric",
                    value_name="Value",
                )
                figure = px.line(
                    usage_minutes,
                    x="game",
                    y="Value",
                    color="Metric",
                    markers=True,
                    color_discrete_map={"usage_pct": "#ff5a36", "minutes": "#263a59"},
                    title="Usage versus minutes trend",
                )
                st.plotly_chart(
                    style_figure(figure, 360),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key="player_usage_minutes",
                )
            with right:
                efficiency = trends.melt(
                    id_vars=["game"],
                    value_vars=[column for column in ("ts_pct", "per_estimate") if column in trends],
                    var_name="Metric",
                    value_name="Value",
                )
                figure = px.line(
                    efficiency,
                    x="game",
                    y="Value",
                    color="Metric",
                    markers=True,
                    color_discrete_map={"ts_pct": "#138a55", "per_estimate": "#5b6cff"},
                    title="Efficiency trend",
                )
                st.plotly_chart(
                    style_figure(figure, 360),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key="player_efficiency_trend",
                )
        else:
            st.caption("Game-level trend lines appear when the provider returns game logs.")

    with rank_tab:
        rank_rows = []
        values = {**season_avg, **advanced}
        for category, rank in ranks.items():
            rank = as_dict(rank)
            out_of = int(safe_number(rank.get("out_of")))
            position = int(safe_number(rank.get("rank")))
            percentile = (
                round((out_of - position + 1) / out_of * 100, 1)
                if out_of and position else None
            )
            rank_rows.append(
                {
                    "category": friendly_label(category),
                    "value": values.get(category),
                    "nba_rank": position or None,
                    "out_of": out_of or None,
                    "percentile": percentile,
                }
            )
        st.dataframe(
            records_table(rank_rows), use_container_width=True, hide_index=True
        )
        chart = pd.DataFrame(rank_rows)
        if "percentile" in chart:
            chart = chart.dropna(subset=["percentile"])
        if not chart.empty:
            figure = px.bar(
                chart.sort_values("percentile"),
                x="percentile",
                y="category",
                orientation="h",
                range_x=[0, 100],
                color="percentile",
                color_continuous_scale=["#dbe5f0", "#5b6cff", "#ff5a36"],
                title="NBA percentile by category",
            )
            figure.update_coloraxes(showscale=False)
            st.plotly_chart(
                style_figure(figure),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="player_rank_percentiles",
            )

    with totals_tab:
        st.dataframe(
            records_table([as_dict(result.get("season_totals"))]),
            use_container_width=True,
            hide_index=True,
        )

    with shot_tab:
        section_header(
            "Shot profile",
            "Half-court density with team-color shading and location-level hover detail.",
        )
        shot_mode = st.segmented_control(
            "Shot measure", SHOT_MODES, default="Attempts only",
            key="player_shot_mode",
        )
        if st.button("Build shot profile", key="load_player_shot_profile"):
            figure = run_analysis(
                "player shot profile",
                lambda: render_shot_chart(
                    player_name, season_label, shot_mode or "Attempts only"
                ),
            )
            if figure is not None:
                st.session_state["player_shot_profile"] = figure
        render_graph_lab_figure(
            st.session_state.get("player_shot_profile"),
            "player_analysis_shot_profile",
        )

    with efficiency_tab:
        section_header(
            "Efficiency profile",
            "Seven league-relative dimensions with season, last-10, and last-5 context.",
        )
        radar_window = st.segmented_control(
            "Sample window", list(RADAR_WINDOWS), default="Season",
            key="player_radar_window",
        )
        if st.button("Build efficiency radar", key="load_player_radar"):
            figure = run_analysis(
                "player efficiency radar",
                lambda: render_efficiency_radar(
                    player_name, season_label, radar_window or "Season"
                ),
            )
            if figure is not None:
                st.session_state["player_efficiency_radar"] = figure
        render_graph_lab_figure(
            st.session_state.get("player_efficiency_radar"),
            "player_analysis_efficiency_radar",
        )

    with badge_tab:
        section_header(
            "Badge profile",
            "Eight identity axes use one dilution-aware percentile model; dunking contributes only to Finishing.",
        )
        view_column, mode_column, comparison_column, filter_column = st.columns(4)
        badge_view = view_column.segmented_control(
            "View", ["Spider", "Wheel"], default="Spider",
            key="player_badge_view",
        )
        badge_display = mode_column.segmented_control(
            "Rating mode", ["Raw", "Adjusted"], default="Adjusted",
            key="player_badge_display_mode",
        )
        badge_compare = comparison_column.segmented_control(
            "Comparison marker", ["Entire league", "Same position"],
            default="Entire league", key="player_badge_comparison_mode",
        )
        badge_filter = filter_column.toggle(
            "Filter by minimum sample size", value=True,
            key="player_badge_sample_filter",
        )
        opponent_column, gradient_column = st.columns([2, 1])
        badge_opponent = opponent_column.text_input(
            "Opponent context (optional)", placeholder="Example: BOS",
            key="player_badge_opponent",
        )
        badge_gradient = gradient_column.toggle(
            "Spider gradient depth", value=True,
            key="player_badge_gradient",
        )
        if st.button("Build badge profile", key="load_player_badge", type="primary"):
            if (badge_view or "Spider") == "Spider":
                renderer = lambda: render_spider_badge_graph(
                    player_name, badge_display or "Adjusted",
                    badge_compare or "Entire league", season=season_label,
                    filter_minimum_samples=badge_filter,
                    opponent_team=badge_opponent or None,
                    gradient_fill=badge_gradient,
                )
            else:
                renderer = lambda: render_badge_graph(
                    player_name, season_label, badge_display or "Adjusted",
                    badge_compare or "Entire league", badge_filter,
                    badge_opponent or None,
                )
            figure = run_analysis("player badge profile", renderer)
            if figure is not None:
                st.session_state["player_badge_profile"] = figure
        badge_figure = st.session_state.get("player_badge_profile")
        if badge_figure is not None:
            badge_meta = badge_figure.layout.meta if isinstance(badge_figure.layout.meta, dict) else {}
            render_insight_list(generate_badge_insights(badge_meta), "Player identity")
            skill_labels = badge_meta.get("skill_labels", [])
            if skill_labels:
                st.markdown("**Skill labels**  ")
                st.write(" · ".join(str(label) for label in skill_labels))
            if "Low sample size: ratings may be unstable." in badge_meta.get("warnings", []):
                st.warning("Low sample size: ratings may be unstable.")
        render_graph_lab_figure(
            badge_figure,
            "player_analysis_badge_profile",
        )
        if badge_figure is not None:
            with st.expander("Percentile, sample, and dilution details"):
                st.dataframe(
                    records_table((badge_figure.layout.meta or {}).get("attributes", [])),
                    width="stretch", hide_index=True,
                )

    with trends_tab:
        section_header(
            "Player trends",
            "Rolling momentum, season progression, and league usage-efficiency context.",
        )
        momentum_view, timeline_view, scatter_view = st.tabs(
            ["Momentum", "Timeline", "Usage vs Efficiency"]
        )
        with momentum_view:
            if st.button("Build momentum line", key="load_player_momentum"):
                figure = run_analysis(
                    "player momentum",
                    lambda: render_momentum_line(player_name, season_label),
                )
                if figure is not None:
                    st.session_state["player_momentum_figure"] = figure
            render_graph_lab_figure(
                st.session_state.get("player_momentum_figure"),
                "player_analysis_momentum",
            )
        with timeline_view:
            if st.button("Build performance timeline", key="load_player_timeline"):
                figure = run_analysis(
                    "player performance timeline",
                    lambda: render_performance_timeline(player_name, season_label),
                )
                if figure is not None:
                    st.session_state["player_timeline_figure"] = figure
            render_graph_lab_figure(
                st.session_state.get("player_timeline_figure"),
                "player_analysis_timeline",
            )
        with scatter_view:
            if st.button("Build league scatter", key="load_player_usage_scatter"):
                figure = run_analysis(
                    "league usage and efficiency",
                    lambda: render_usage_efficiency_scatter(
                        season_label, player_name
                    ),
                )
                if figure is not None:
                    st.session_state["player_usage_scatter"] = figure
            render_graph_lab_figure(
                st.session_state.get("player_usage_scatter"),
                "player_analysis_usage_scatter",
            )
    with similar_tab:
        st.write(
            "Find players with comparable role, efficiency, pace, scoring, "
            "rebounding, assisting, defense, recent form, and matchup context."
        )
        if st.button("Find similar players", key="load_similar_players"):
            similarity = run_analysis(
                "player similarity",
                lambda: compute_player_similarity(player_name, season_label),
            )
            if similarity:
                st.session_state["similarity_insights"] = similarity
        similarity = as_dict(st.session_state.get("similarity_insights"))
        if similarity and as_dict(similarity.get("target")).get("player") == player_name:
            render_similarity_view(similarity, "player_analysis_similarity")

    with correlation_tab:
        st.write("Measure how production, role, efficiency, pace, and opponent context move together.")
        if st.button("Build player correlations", key="load_player_correlations"):
            correlations = run_analysis(
                "player correlations",
                lambda: compute_player_correlations(player_name, season_label),
            )
            if correlations:
                st.session_state["player_correlations"] = correlations
        correlations = as_dict(st.session_state.get("player_correlations"))
        if correlations and correlations.get("subject") == player_name:
            options = correlations.get("available_stats", [])
            selected = st.multiselect(
                "Statistics",
                options,
                default=options,
                key="player_correlation_stats",
            )
            render_correlation_heatmap(
                correlations, selected, "player_analysis_correlations"
            )


def render_player_comparison(result: dict[str, Any]) -> None:
    """Display similarity, strengths, weaknesses, and metric comparisons."""
    players = result.get("players", [])
    st.subheader(" vs ".join(player.get("player", "Unknown") for player in players))
    first, second = st.columns(2)
    first.metric("Similarity score", f"{result.get('similarity_score', 0):.1f}/100")
    second.metric("Matchup difficulty", f"{result.get('matchup_difficulty', 0):.1f}/100")
    st.caption(f"Season {result.get('season', '—')} • {result.get('explanation', '')}")
    comparison_rows = [{"player": player.get("player"), "team": player.get("team"), **player.get("metrics", {})} for player in players]
    st.markdown("### Comparison")
    table = records_table(comparison_rows, "player")
    st.dataframe(table, use_container_width=True)
    numeric = table.select_dtypes(include="number")
    if not numeric.empty:
        st.bar_chart(numeric.T)
    st.markdown("### Strengths and weaknesses")
    columns = st.columns(max(len(players), 1))
    for column, player in zip(columns, players):
        with column:
            team_banner(player.get("player","Player"),player.get("team",""))
            st.caption("Strengths")
            st.dataframe(records_table(player.get("strengths", [])), use_container_width=True, hide_index=True)
            st.caption("Weaknesses")
            st.dataframe(records_table(player.get("weaknesses", [])), use_container_width=True, hide_index=True)

def render_projection_page(page_title: str = "Player Projections") -> None:
    """Display fused projections with polished model and context visuals."""
    st.title(page_title)
    st.write("Fuse regression, minutes, pace, matchup, injury, and recent form.")
    with st.form("projection_form"):
        first, second, third = st.columns([2, 1, 1])
        player_name = first.text_input(
            "NBA player", "Nikola Jokic", key="projection_player"
        )
        opponent = second.text_input(
            "Opponent team", "BOS", key="projection_opponent"
        )
        season = third.text_input(
            "Season", latest_season(), key="projection_season"
        )
        submitted = st.form_submit_button("Build projection", type="primary")
    if submitted:
        if not player_name.strip() or not opponent.strip():
            st.warning("Enter both a player and an opponent.")
        else:
            result = run_analysis(
                "fused player projection",
                lambda: get_reliability_score(
                    player_name, opponent, season or None
                ),
            )
            if result:
                st.session_state["projection_insights"] = result

    bundle = as_dict(st.session_state.get("projection_insights"))
    fusion = as_dict(bundle.get("fusion"))
    if not fusion:
        return
    base = as_dict(fusion.get("base_projection"))
    context = as_dict(fusion.get("context"))
    team = fusion.get("team", "")
    accent = team_color(team)
    profile_header(
        fusion.get("player", "Player"),
        team,
        f"vs {fusion.get('opponent', 'Opponent')} · "
        f"{bundle.get('rating', 'unknown').title()} reliability",
    )
    st.caption(f"Season {fusion.get('season', '—')} · Explainable fusion model")
    model_warnings = []
    for source in (
        base.get("warnings", []),
        context.get("warnings", []),
        bundle.get("warnings", []),
    ):
        if isinstance(source, list):
            model_warnings.extend(str(warning) for warning in source if warning)
    fallback_status = as_dict(fusion.get("fallback_status"))
    fallback_active = bool(fallback_status.get("active")) or any(
        warning in {FALLBACK_WARNING, ENHANCED_FALLBACK_WARNING}
        for warning in model_warnings
    )
    if fallback_active:
        st.warning(ENHANCED_FALLBACK_WARNING)
        st.caption(
            f"Fallback source: {fallback_status.get('source', 'cached history')}"
        )

    projection_matchup = as_dict(
        as_dict(context.get("components")).get("matchup")
    )
    render_insight_list(
        generate_player_insights(bundle, projection_matchup),
        "Projection insights",
    )
    projection = as_dict(fusion.get("final_projection"))
    ranges = as_dict(fusion.get("confidence_range"))
    columns = st.columns(4)
    for column, stat in zip(columns, ["points", "rebounds", "assists", "pra"]):
        interval = as_dict(ranges.get(stat))
        column.metric(
            friendly_label(stat),
            projection.get(stat, "—"),
            help=f"Range: {interval.get('low', '—')} to {interval.get('high', '—')}",
        )
    reliability_column, fallback_column = st.columns(2)
    reliability_column.metric(
        "Reliability", f"{safe_number(bundle.get('score')):.1f}/100"
    )
    fallback_column.metric(
        "Fallback status",
        "Enhanced fallback" if fallback_active else "Live data",
        help=f"Source: {fallback_status.get('source', 'live')}",
    )
    reliability_bar(bundle.get("score"), "Overall model reliability")

    projection_tab, drivers_tab, reliability_tab, context_tab = st.tabs(
        ["Projection", "Model Drivers", "Reliability", "Player Context"]
    )
    with projection_tab:
        range_rows = [
            {
                "stat": stat,
                "projection": projection.get(stat),
                "low": as_dict(ranges.get(stat)).get("low"),
                "high": as_dict(ranges.get(stat)).get("high"),
            }
            for stat in projection
        ]
        recent5 = as_dict(base.get("recent_5_game_averages"))
        recent10 = as_dict(base.get("recent_10_game_averages"))
        season_averages = as_dict(base.get("season_averages"))
        if not season_averages:
            season_averages = as_dict(
                as_dict(base.get("feature_summary")).get("season_average")
            )
        chart = pd.DataFrame(
            [
                {
                    "Stat": friendly_label(stat),
                    "Window": window,
                    "Value": values.get(stat),
                }
                for stat in ("points", "rebounds", "assists", "pra")
                for window, values in (
                    ("Projection", projection),
                    ("Season", season_averages),
                    ("Last 10", recent10),
                    ("Last 5", recent5),
                )
            ]
        )
        figure = px.bar(
            chart,
            x="Stat",
            y="Value",
            color="Window",
            barmode="group",
            color_discrete_map={
                "Projection": accent,
                "Season": "#10213a",
                "Last 10": "#5b6cff",
                "Last 5": "#ff8a3d",
            },
            title="Projection versus recent production",
        )
        figure.update_layout(legend_title_text="")
        st.plotly_chart(
            style_figure(figure),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="projection_recent_chart",
        )
        st.dataframe(
            records_table(range_rows), use_container_width=True, hide_index=True
        )

    with drivers_tab:
        section_header(
            "Model drivers",
            "How minutes, role, efficiency, and matchup context shape the result.",
        )
        reliability_bar(bundle.get("score"), "Driver confidence")
        st.markdown("### Minutes, usage, and efficiency inputs")
        model_inputs = as_dict(fusion.get("input_breakdown"))
        minutes_inputs = as_dict(model_inputs.get("minutes"))
        usage_inputs = as_dict(model_inputs.get("usage"))
        efficiency_inputs = as_dict(model_inputs.get("efficiency"))
        input_columns = st.columns(3)
        input_columns[0].metric(
            "Projected minutes",
            minutes_inputs.get("projection", "—"),
            help=(
                f"Baseline: {minutes_inputs.get('selected_baseline', '—')} "
                f"from {minutes_inputs.get('selected_source', 'unknown')}"
            ),
        )
        input_columns[1].metric(
            "Blended usage",
            usage_inputs.get("adjusted", "—"),
            help=(
                f"Season {usage_inputs.get('season_average', '—')}; "
                f"L10 {usage_inputs.get('last_10', '—')}; "
                f"L5 {usage_inputs.get('last_5', '—')}"
            ),
        )
        input_columns[2].metric(
            "Adjusted TS%",
            efficiency_inputs.get("adjusted_ts_pct", "—"),
            help=(
                f"Season {efficiency_inputs.get('season_average_ts_pct', '—')}; "
                f"L10 {efficiency_inputs.get('last_10_ts_pct', '—')}; "
                f"L5 {efficiency_inputs.get('last_5_ts_pct', '—')}"
            ),
        )
        input_rows = []
        for component_name in ("minutes", "usage", "efficiency", "context"):
            component = as_dict(model_inputs.get(component_name))
            input_rows.append({"component": component_name, **component})
        st.dataframe(
            records_table(input_rows), use_container_width=True, hide_index=True
        )
        weights = as_dict(fusion.get("fusion_weights"))
        if weights:
            st.caption(
                "Normalized fusion weights · "
                + " · ".join(
                    f"{friendly_label(key)} {safe_number(value) * 100:.1f}%"
                    for key, value in weights.items()
                )
            )

        breakdown = as_dict(fusion.get("driver_breakdown"))
        driver_rows = [
            {"stat": friendly_label(stat), **as_dict(drivers)}
            for stat, drivers in breakdown.items()
        ]
        if driver_rows:
            driver_table = pd.DataFrame(driver_rows)
            value_columns = [
                column for column in driver_table.columns if column != "stat"
            ]
            melted = driver_table.melt(
                id_vars="stat",
                value_vars=value_columns,
                var_name="Driver",
                value_name="Impact",
            )
            figure = px.bar(
                melted,
                x="stat",
                y="Impact",
                color="Driver",
                barmode="relative",
                title="How each model component changes the projection",
            )
            st.plotly_chart(
                style_figure(figure),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="projection_driver_chart",
            )
            st.dataframe(
                records_table(driver_rows), use_container_width=True, hide_index=True
            )
        for driver in fusion.get("key_drivers", []):
            st.write(f"- {driver}")

    with reliability_tab:
        section_header(
            "Reliability",
            "A transparent view of data quality and model stability.",
        )
        ring_column, card_column = st.columns([1, 1.45], vertical_alignment="center")
        with ring_column:
            confidence_ring(
                bundle.get("score"), "Reliability", "projection_confidence_ring"
            )
        with card_column:
            confidence_card(bundle.get("score"), "Quant reliability")
            reliability_bar(bundle.get("score"), "Overall reliability")
        reliability_rows = [
            {"component": friendly_label(key), "score": value}
            for key, value in as_dict(bundle.get("components")).items()
        ]
        if reliability_rows:
            reliability_frame = pd.DataFrame(reliability_rows).sort_values("score")
            figure = px.bar(
                reliability_frame,
                x="score",
                y="component",
                orientation="h",
                range_x=[0, 100],
                color="score",
                color_continuous_scale=["#d94a3a", "#f7d154", "#138a55"],
                title="Reliability component scores",
            )
            figure.update_coloraxes(showscale=False)
            st.plotly_chart(
                style_figure(figure),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="projection_reliability_chart",
            )
            st.dataframe(
                records_table(reliability_rows),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(bundle.get("explanation", ""))

    with context_tab:
        pace = as_dict(as_dict(context.get("components")).get("pace"))
        matchup = as_dict(as_dict(context.get("components")).get("matchup"))
        injury = as_dict(context.get("injury_impact"))
        context_metrics = st.columns(4)
        context_metrics[0].metric(
            "Role stability", f"{safe_number(context.get('role_stability')):.1f}/100"
        )
        context_metrics[1].metric(
            "Blowout risk", f"{safe_number(context.get('blowout_risk')):.1f}/100"
        )
        context_metrics[2].metric(
            "Expected pace",
            safe_number(pace.get("pace_projection", pace.get("value")), 100),
        )
        context_metrics[3].metric(
            "Opponent difficulty",
            f"{safe_number(matchup.get('difficulty_score', matchup.get('value')), 50):.1f}/100",
        )
        projected_pace = safe_number(
            pace.get("pace_projection", pace.get("value")), 100
        )
        recent_pace = safe_number(pace.get("recent_pace"), projected_pace)
        difficulty = safe_number(
            matchup.get("difficulty_score", matchup.get("value")), 50
        )
        scatter = pd.DataFrame(
            [
                {
                    "Context": "League baseline",
                    "Pace": 100.0,
                    "Opponent Difficulty": 50.0,
                    "Size": 18,
                },
                {
                    "Context": "Recent pace",
                    "Pace": recent_pace,
                    "Opponent Difficulty": difficulty,
                    "Size": 22,
                },
                {
                    "Context": "Projected matchup",
                    "Pace": projected_pace,
                    "Opponent Difficulty": difficulty,
                    "Size": 28,
                },
            ]
        )
        figure = px.scatter(
            scatter,
            x="Pace",
            y="Opponent Difficulty",
            size="Size",
            color="Context",
            text="Context",
            range_y=[0, 100],
            color_discrete_map={
                "League baseline": "#94a3b8",
                "Recent pace": "#5b6cff",
                "Projected matchup": accent,
            },
            title="Pace versus opponent difficulty",
        )
        figure.update_traces(textposition="top center")
        st.plotly_chart(
            style_figure(figure),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="projection_context_scatter",
        )
        st.caption(
            f"Availability: {injury.get('status', 'unknown')} · "
            f"Usage trend: {safe_number(as_dict(context.get('usage_trend')).get('change')):+.2f} · "
            f"Minutes trend: {safe_number(as_dict(context.get('minutes_trend')).get('change')):+.2f}"
        )
        with st.expander("Base-model validation details"):
            st.dataframe(
                records_table(
                    [
                        {"stat": stat, **as_dict(quality)}
                        for stat, quality in as_dict(base.get("model_quality")).items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.json(as_dict(base.get("data_splits")))
            st.json(as_dict(base.get("data_quality")))


def render_player_finder_page() -> None:
    """Display configurable NBA season leaderboards."""
    st.title("Player Finder")
    st.write("Explore qualified NBA leaders using season-level base and advanced stats.")
    categories = {
        "Top scorers": (top_scorers, "points_per_game"),
        "Top rebounders": (top_rebounders, "rebounds_per_game"),
        "Top assist players": (top_assist_players, "assists_per_game"),
        "High usage": (high_usage_players, "usage_rate"),
        "High efficiency": (high_efficiency_players, "true_shooting_pct"),
    }
    with st.form("player_finder_form"):
        category = st.selectbox("Leaderboard category", list(categories))
        first, second, third = st.columns(3)
        limit = first.number_input("Players", min_value=5, max_value=50, value=20, step=5)
        min_minutes = second.number_input("Minimum minutes", min_value=0.0,
                                          max_value=48.0, value=20.0, step=1.0)
        season = third.text_input("Season", latest_season(), key="finder_season")
        submitted = st.form_submit_button("Load leaderboard", type="primary")
    if submitted:
        function, metric = categories[category]
        def load_leaderboard() -> dict[str, Any]:
            if category == "High efficiency":
                records = function(int(limit), float(min_minutes), 20, season or None)
            else:
                records = function(int(limit), float(min_minutes), season or None)
            return {"records": records, "category": category,
                    "metric": metric, "season": season}

        result = run_analysis("player finder", load_leaderboard)
        if result:
            st.session_state["player_finder_insights"] = result

    result = st.session_state.get("player_finder_insights")
    if not result:
        return
    records = result.get("records", [])
    st.subheader(f"{result.get('category')} • {result.get('season')}")
    st.dataframe(records_table(records), use_container_width=True, hide_index=True)
    if records:
        metric = result.get("metric", "points_per_game")
        chart = pd.DataFrame(records).set_index("player_name")[[metric]]
        st.bar_chart(chart)

def render_opportunities_page() -> None:
    """Rank every analysis currently saved in this Streamlit session."""
    st.title("Ranked Opportunities")
    st.write("This page combines analyses you have already run in the dashboard.")
    finance = st.session_state.get("finance_insights")
    nba = st.session_state.get("nba_insights")
    nfl = st.session_state.get("nfl_insights")
    opportunities = rank_opportunities(finance, nba, nfl)
    if not opportunities:
        st.info("Run at least one Finance, NBA, or NFL analysis first.")
        return

    table = records_table(opportunities)
    st.dataframe(table, use_container_width=True, hide_index=True)
    chart = pd.DataFrame(opportunities).set_index("opportunity")[["score"]]
    st.bar_chart(chart)
    st.caption("Scores rank signal strength across domains; they are not probabilities.")


def available_slate_edges() -> list[dict[str, Any]]:
    """Collect saved prop comparisons or derive edges from the daily slate."""
    saved = (
        st.session_state.get("prop_recommendations")
        or st.session_state.get("prop_insights")
        or []
    )
    if saved:
        return [row for row in saved if isinstance(row, dict)]
    slate = as_dict(st.session_state.get("daily_slate"))
    projections = slate.get("projections", [])
    projection_map = {
        (str(row.get("player", "")).lower(), str(category).lower()): value
        for row in projections if isinstance(row, dict)
        for category, value in as_dict(row.get("projection")).items()
    }
    rows = []
    for prop in slate.get("player_props", []):
        prop = as_dict(prop)
        key = (
            str(prop.get("player", "")).lower(),
            str(prop.get("category", "")).lower(),
        )
        projection = projection_map.get(key)
        if projection is None:
            continue
        line = safe_number(prop.get("line"))
        rows.append(
            {
                "player": prop.get("player"),
                "team": prop.get("team"),
                "category": prop.get("category"),
                "projection": projection,
                "minutes_adjusted_projection": projection,
                "sportsbook_line": line,
                "edge": safe_number(projection) - line,
                "reliability": 50.0,
                "confidence": 50.0,
                "sportsbook": prop.get("sportsbook"),
            }
        )
    return rows


def render_visual_lab_page() -> None:
    """Central workspace for correlations, edge maps, and player similarity."""
    st.title("Visual Analytics Lab")
    st.write(
        "Explore relationships and comparable profiles with filters designed "
        "to stay readable even when public data is incomplete."
    )
    correlation_tab, edge_tab, similarity_tab = st.tabs(
        ["Correlation Heatmaps", "Slate Edge Heatmap", "Similar Players"]
    )

    with correlation_tab:
        scope = st.segmented_control(
            "Correlation scope",
            ["Player", "Team", "Slate"],
            default="Player",
            key="visual_correlation_scope",
        )
        if scope == "Player":
            with st.form("visual_player_correlation_form"):
                first, second = st.columns([2, 1])
                player = first.text_input(
                    "NBA player", "Nikola Jokic", key="visual_corr_player"
                )
                season = second.text_input(
                    "Season", latest_season(), key="visual_corr_player_season"
                )
                submitted = st.form_submit_button(
                    "Build player heatmap", type="primary"
                )
            if submitted:
                payload = run_analysis(
                    "player correlations",
                    lambda: compute_player_correlations(player, season),
                )
                if payload:
                    st.session_state["visual_correlations"] = payload
        elif scope == "Team":
            with st.form("visual_team_correlation_form"):
                first, second = st.columns([2, 1])
                team = first.text_input(
                    "NBA team", "DEN", key="visual_corr_team"
                )
                season = second.text_input(
                    "Season", latest_season(), key="visual_corr_team_season"
                )
                submitted = st.form_submit_button(
                    "Build team heatmap", type="primary"
                )
            if submitted:
                payload = run_analysis(
                    "team correlations",
                    lambda: compute_team_correlations(team, season),
                )
                if payload:
                    st.session_state["visual_correlations"] = payload
        else:
            slate = as_dict(st.session_state.get("daily_slate"))
            slate_rows = slate.get("projections", []) or available_slate_edges()
            st.caption(
                "Slate mode uses projections already loaded on Daily Slate, "
                "Prop Analyzer, or Prop Recommendations."
            )
            if st.button("Build slate correlations", key="visual_slate_corr"):
                payload = compute_slate_correlations(slate_rows)
                st.session_state["visual_correlations"] = payload

        payload = as_dict(st.session_state.get("visual_correlations"))
        if payload:
            options = payload.get("available_stats", [])
            selected = st.multiselect(
                "Statistics to display",
                options,
                default=options,
                key="visual_correlation_stats",
            )
            render_correlation_heatmap(
                payload, selected, "visual_lab_correlations"
            )

    with edge_tab:
        props = available_slate_edges()
        if props:
            render_edge_heatmap_view(props, "visual_lab_edges")
        else:
            st.info(
                "Run Prop Analyzer, Prop Recommendations, or Daily Slate first "
                "to create projection-versus-line edges."
            )

    with similarity_tab:
        with st.form("visual_similarity_form"):
            first, second = st.columns([2, 1])
            target = first.text_input(
                "Target NBA player", "Nikola Jokic", key="visual_similarity_player"
            )
            season = second.text_input(
                "Season", latest_season(), key="visual_similarity_season"
            )
            submitted = st.form_submit_button(
                "Find similar players", type="primary"
            )
        if submitted:
            result = run_analysis(
                "player similarity",
                lambda: compute_player_similarity(target, season),
            )
            if result:
                st.session_state["visual_similarity"] = result
        result = as_dict(st.session_state.get("visual_similarity"))
        if result:
            render_similarity_view(result, "visual_lab_similarity")


def _home_route() -> None:
    """Render the dashboard home route inside the grouped navigation shell."""
    apply_global_theme()
    render_home()


def _custom_sidebar(page_groups: dict[str, list[st.Page]]) -> None:
    """Render collapsible, icon-led navigation groups."""
    st.sidebar.markdown("## Universal Quant")
    st.sidebar.caption("Explainable sports, market, and model research.")
    for section, pages in page_groups.items():
        with st.sidebar.expander(section, expanded=section in {"NBA Tools", "Graph Lab"}):
            for page in pages:
                st.page_link(page)


def main() -> None:
    """Run the app with a compact, grouped, collapsible sidebar."""
    # Navigation pages run as transient `__main__` modules. Keep a stable reference.
    sys.modules["_uqa_app_runtime"] = sys.modules[__name__]
    home_page = st.Page(_home_route, title="Home", icon=":material/home:", url_path="home", default=True)
    nba_pages = [
        st.Page("pages/1_NBA_Analysis.py", title="NBA Analysis", icon=":material/sports_basketball:"),
        st.Page("pages/2_Player_Analysis.py", title="Player Analysis", icon=":material/person_search:"),
        st.Page("pages/3_Player_Projections.py", title="Player Projections", icon=":material/query_stats:"),
        st.Page("pages/4_Player_Finder.py", title="Player Finder", icon=":material/leaderboard:"),
        st.Page("pages/11_Daily_Slate.py", title="Daily Slate", icon=":material/calendar_today:"),
    ]
    graph_pages = [
        st.Page("pages/14_Shot_Chart_Heatmap.py", title="Shot-Chart Heatmap", icon=":material/blur_on:"),
        st.Page("pages/15_Efficiency_Radar.py", title="Efficiency Radar", icon=":material/radar:"),
        st.Page("pages/16_Badge_Graph.py", title="Badge Graph", icon=":material/bubble_chart:"),
        st.Page("pages/17_Trend_Graphs.py", title="Trend Graphs", icon=":material/show_chart:"),
    ]
    betting_pages = [
        st.Page("pages/8_Prop_Analyzer.py", title="Prop Analyzer", icon=":material/compare_arrows:"),
        st.Page("pages/19_Edge_Heatmap.py", title="Edge Heatmap", icon=":material/grid_on:"),
        st.Page("pages/9_Prop_Recommendations.py", title="Prop Recommendations", icon=":material/recommend:"),
        st.Page("pages/10_Parlay_Builder.py", title="Parlay Builder", icon=":material/account_tree:"),
    ]
    advanced_pages = [
        st.Page("pages/18_Similarity_Engine.py", title="Similarity Engine", icon=":material/group_work:"),
        st.Page("pages/13_Visual_Analytics_Lab.py", title="Correlation Lab", icon=":material/scatter_plot:"),
        st.Page("pages/20_Model_Drivers.py", title="Model Drivers", icon=":material/tune:"),
        st.Page("pages/12_Model_Performance.py", title="Model Performance", icon=":material/analytics:"),
    ]
    football_pages = [
        st.Page("pages/5_NFL_Analysis.py", title="Team Analysis", icon=":material/sports_football:"),
        st.Page("pages/21_NFL_Player_Analysis.py", title="Player Analysis", icon=":material/person_search:"),
        st.Page("pages/22_NFL_Graph_Lab.py", title="Graph Lab", icon=":material/monitoring:"),
        st.Page("pages/23_NFL_Projections.py", title="Projections", icon=":material/query_stats:"),
        st.Page("pages/24_NFL_Slate.py", title="Slate", icon=":material/view_week:"),
    ]
    finance_pages = [
        st.Page("pages/6_Finance.py", title="Finance Dashboard", icon=":material/candlestick_chart:"),
        st.Page("pages/7_Opportunities.py", title="Opportunities", icon=":material/insights:"),
    ]
    fantasy_pages = [
        st.Page("pages/25_Fantasy.py", title="Fantasy", icon=":material/emoji_events:")
    ]
    groups = {
        "Workspace": [home_page],
        "NBA Tools": nba_pages,
        "Graph Lab": graph_pages,
        "Betting Tools": betting_pages,
        "Advanced Models": advanced_pages,
        "NFL Tools": football_pages,
        "Finance": finance_pages,
        "Fantasy": fantasy_pages,
    }
    navigator = st.navigation(groups, position="hidden")
    st.session_state["_uqa_custom_navigation"] = True
    inject_app_css()
    _custom_sidebar(groups)
    navigator.run()


if __name__ == "__main__":
    main()