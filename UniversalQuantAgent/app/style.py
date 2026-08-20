"""Premium visual system shared by every Universal Quant Agent surface.

The module is intentionally useful without a running Streamlit session:
rarity mapping, card markup, chart construction, and drag/drop markup are pure
functions and can be contract-tested offline. Rendering helpers are thin
wrappers around those pure primitives.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
MIDNIGHT_NAVY = "#0A1A2F"
CIRCUIT_CYAN = "#00C8FF"
SIGNAL_GOLD = "#F5C542"
SLATE_BLUE = "#1F2E45"
SOFT_WHITE = "#F2F4F7"

PALETTE = {
    "midnight_navy": MIDNIGHT_NAVY,
    "circuit_cyan": CIRCUIT_CYAN,
    "signal_gold": SIGNAL_GOLD,
    "slate_blue": SLATE_BLUE,
    "soft_white": SOFT_WHITE,
}

SPACING_SCALE = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "0.875rem",
    "lg": "1.25rem",
    "xl": "1.75rem",
    "2xl": "2.5rem",
}

TYPOGRAPHY_SCALE = {
    "caption": "0.72rem",
    "body": "0.92rem",
    "label": "0.78rem",
    "card_title": "1.08rem",
    "section": "1.28rem",
    "page": "2rem",
}

CYAN_GLOW = "0 0 0 1px rgba(0,200,255,.18), 0 10px 34px rgba(0,200,255,.10)"
GOLD_GLOW = "0 0 18px rgba(245,197,66,.22)"

RARITY_TIERS = (
    {"name": "Mythic", "min_rank": 1, "max_rank": 1, "class": "rarity-mythic", "symbol": "◆"},
    {"name": "Legendary", "min_rank": 2, "max_rank": 5, "class": "rarity-legendary", "symbol": "◇"},
    {"name": "Elite", "min_rank": 6, "max_rank": 10, "class": "rarity-elite", "symbol": "⬡"},
    {"name": "Pro", "min_rank": 11, "max_rank": 20, "class": "rarity-pro", "symbol": "⬢"},
    {"name": "Starter", "min_rank": 21, "max_rank": 40, "class": "rarity-starter", "symbol": "●"},
    {"name": "Depth", "min_rank": 41, "max_rank": math.inf, "class": "rarity-depth", "symbol": "•"},
)

GOLD_GLOW_CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": MIDNIGHT_NAVY,
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": SOFT_WHITE},
    "title_font": {"color": SOFT_WHITE, "size": 17},
    "hoverlabel": {"bgcolor": SLATE_BLUE, "font_color": SOFT_WHITE, "bordercolor": CIRCUIT_CYAN},
    "colorway": [SIGNAL_GOLD, CIRCUIT_CYAN, "#7891B4", "#D8E2EF"],
}


def rarity_metadata(rank: Any) -> dict[str, Any]:
    """Return the complete HoopGrids-style rarity record for ``rank``."""
    try:
        normalized = max(1, int(float(rank)))
    except (TypeError, ValueError, OverflowError):
        normalized = 41
    for tier in RARITY_TIERS:
        if tier["min_rank"] <= normalized <= tier["max_rank"]:
            return {**tier, "rank": normalized}
    return {**RARITY_TIERS[-1], "rank": normalized}


def rarity_tier(rank: Any) -> str:
    """Map a rank to Mythic, Legendary, Elite, Pro, Starter, or Depth."""
    return str(rarity_metadata(rank)["name"])


def rarity_icon(rank: Any, *, include_label: bool = True) -> str:
    """Return accessible rarity badge HTML for use inside a player card."""
    rarity = rarity_metadata(rank)
    label = f'<span class="rarity-label">{escape(rarity["name"])}</span>' if include_label else ""
    return (
        f'<span class="rarity-badge {rarity["class"]}" '
        f'aria-label="{escape(rarity["name"])} rarity, rank {rarity["rank"]}">'
        f'<span class="rarity-symbol" aria-hidden="true">{rarity["symbol"]}</span>{label}</span>'
    )


def rarity_icon_component(rank: Any, *, include_label: bool = True) -> str:
    """Named component alias used by UI callers and tests."""
    return rarity_icon(rank, include_label=include_label)


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-") or "card"


def stacked_card_html(
    title: Any,
    body: Any = "",
    *,
    kicker: Any = "",
    stats: Mapping[str, Any] | None = None,
    rarity_rank: Any | None = None,
    href: str | None = None,
    card_id: str | None = None,
    extra_class: str = "",
) -> str:
    """Build one reusable navy/cyan stacked card as safe HTML."""
    kicker_html = f'<div class="quant-card-kicker">{escape(str(kicker))}</div>' if kicker else ""
    body_html = f'<p>{escape(str(body))}</p>' if body else ""
    rarity_html = rarity_icon(rarity_rank) if rarity_rank is not None else ""
    stat_html = ""
    if stats:
        stat_html = '<div class="quant-card-stats">' + "".join(
            f'<div><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>'
            for label, value in stats.items()
        ) + "</div>"
    content = (
        f'<div class="quant-card-head"><div>{kicker_html}<h3>{escape(str(title))}</h3></div>{rarity_html}</div>'
        f"{body_html}{stat_html}"
    )
    if href:
        content += '<div class="quant-card-link">Open <span aria-hidden="true">→</span></div>'
        content = f'<a href="{escape(href, quote=True)}">{content}</a>'
    identity = escape(card_id or _slug(title), quote=True)
    classes = f"quant-card stacked-card {escape(extra_class, quote=True)}".strip()
    return f'<article id="{identity}" class="{classes}">{content}</article>'


def render_stacked_card(*args: Any, **kwargs: Any) -> None:
    """Render :func:`stacked_card_html` in Streamlit."""
    st.markdown(stacked_card_html(*args, **kwargs), unsafe_allow_html=True)


def apply_gold_glow_theme(figure: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the navy canvas, gold signal line, and cyan accents in-place."""
    layout = dict(GOLD_GLOW_CHART_THEME)
    layout.update(
        {
            "margin": {"l": 20, "r": 20, "t": 52, "b": 28},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            "hovermode": "x unified",
        }
    )
    if height is not None:
        layout["height"] = int(height)
    figure.update_layout(**layout)
    figure.update_xaxes(
        gridcolor="rgba(242,244,247,.08)",
        linecolor="rgba(0,200,255,.24)",
        tickfont={"color": "#B8C7DA"},
        zeroline=False,
    )
    figure.update_yaxes(
        gridcolor="rgba(242,244,247,.08)",
        linecolor="rgba(0,200,255,.24)",
        tickfont={"color": "#B8C7DA"},
        zeroline=False,
    )
    return figure


def gold_glow_line_chart(
    values: Iterable[Any],
    x: Iterable[Any] | None = None,
    *,
    title: str = "",
    name: str = "Projection",
    height: int = 330,
    y_suffix: str = "",
) -> go.Figure:
    """Create the canonical thin-gold-line chart with cyan peak markers."""
    numeric_values = [safe_number(value) for value in values]
    x_values = list(x) if x is not None else list(range(1, len(numeric_values) + 1))
    if len(x_values) != len(numeric_values):
        raise ValueError("x and values must contain the same number of items")
    peak = max(numeric_values, default=0.0)
    peak_x = [label for label, value in zip(x_values, numeric_values, strict=True) if value == peak]
    peak_y = [value for value in numeric_values if value == peak]
    hover = f"%{{y:.2f}}{escape(y_suffix)}<extra>{escape(name)}</extra>"

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=numeric_values,
            mode="lines",
            name=f"{name} glow",
            line={"color": "rgba(245,197,66,.16)", "width": 10, "shape": "spline"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=numeric_values,
            mode="lines",
            name=name,
            line={"color": SIGNAL_GOLD, "width": 2.2, "shape": "spline"},
            hovertemplate=hover,
        )
    )
    if peak_x:
        figure.add_trace(
            go.Scatter(
                x=peak_x,
                y=peak_y,
                mode="markers",
                name="Peak",
                marker={
                    "color": CIRCUIT_CYAN,
                    "size": 9,
                    "line": {"color": SOFT_WHITE, "width": 1.5},
                },
                hovertemplate=hover,
            )
        )
    figure.update_layout(title=title, showlegend=False)
    return apply_gold_glow_theme(figure, height)


def safe_number(value: Any, default: float = 0.0) -> float:
    """Local finite-number coercion for visual-only inputs."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _drag_player_card(player: Mapping[str, Any], index: int) -> str:
    player_id = str(player.get("player_id") or player.get("id") or f"player-{index}")
    name = str(player.get("name") or player.get("player_name") or player_id)
    position = str(player.get("position") or "—").upper()
    team = str(player.get("team") or player.get("nfl_team") or "FA").upper()
    rank = player.get("overall_rank", player.get("rank", player.get("position_rank", 41)))
    return (
        f'<div class="drag-player" draggable="true" data-player-id="{escape(player_id, quote=True)}" '
        f'data-player-name="{escape(name, quote=True)}" tabindex="0">'
        f'{rarity_icon(rank, include_label=False)}'
        f'<div class="drag-copy"><strong>{escape(name)}</strong><span>{escape(position)} · {escape(team)}</span></div>'
        '<span class="drag-handle" aria-label="Drag player">⋮⋮</span></div>'
    )


def drag_drop_lineup_html(players: Sequence[Mapping[str, Any]], *, key: str = "quant-lineup") -> str:
    """Return an accessible HTML5 drag/drop lineup board.

    Moves persist in browser ``localStorage`` under ``key``. Keyboard users can
    focus a card and press Enter to move it between Starter and Bench lanes.
    """
    starter_cards: list[str] = []
    bench_cards: list[str] = []
    for index, player in enumerate(players):
        slot = str(player.get("slot") or "BENCH").upper()
        card = _drag_player_card(player, index)
        (bench_cards if slot in {"", "BENCH", "BN", "IR", "TAXI", "RESERVE"} else starter_cards).append(card)
    storage_key = json.dumps(f"uqa-lineup-{key}")
    board_id = escape(_slug(key), quote=True)
    return f"""
    <style>
      :root {{ color-scheme: dark; }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: {MIDNIGHT_NAVY};
        color: {SOFT_WHITE};
        font-family: Inter, "Segoe UI", sans-serif;
      }}
      .drag-lineup-board {{ display: grid; gap: 14px; padding: 2px; }}
      .drag-lane {{
        min-height: 180px;
        padding: 14px;
        border: 1px solid rgba(0, 200, 255, .35);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(31, 46, 69, .94), rgba(10, 26, 47, .98));
        box-shadow: {CYAN_GLOW};
      }}
      .drag-lane-head {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
      .drag-lane-head strong {{ color: {SIGNAL_GOLD}; font-size: 15px; }}
      .drag-lane-head span, .drag-status {{ color: #AFC0D6; font-size: 12px; }}
      .drop-zone {{ display: grid; gap: 8px; min-height: 105px; border-radius: 10px; transition: .18s ease; }}
      .drop-zone.is-over {{ background: rgba(0, 200, 255, .08); box-shadow: inset 0 0 0 1px {CIRCUIT_CYAN}; }}
      .drag-player {{
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border: 1px solid rgba(0, 200, 255, .25);
        border-radius: 10px;
        background: {SLATE_BLUE};
        cursor: grab;
        transition: transform .16s ease, border-color .16s ease, opacity .16s ease;
      }}
      .drag-player:hover, .drag-player:focus {{ border-color: {CIRCUIT_CYAN}; transform: translateY(-1px); outline: none; }}
      .drag-player.is-dragging {{ opacity: .45; }}
      .drag-copy {{ display: grid; gap: 2px; }}
      .drag-copy strong {{ font-size: 14px; }}
      .drag-copy span {{ color: #AFC0D6; font-size: 11px; }}
      .drag-handle {{ color: {CIRCUIT_CYAN}; letter-spacing: -3px; }}
      .rarity-icon {{ display: inline-flex; align-items: center; color: {SIGNAL_GOLD}; font-size: 16px; }}
      .drag-status {{ padding: 2px 4px; }}
    </style>
    <div id="{board_id}" class="drag-lineup-board" data-component="drag-drop-lineup">
      <div class="drag-lane" data-lane="starters">
        <div class="drag-lane-head"><strong>Starters</strong><span>Drop active lineup here</span></div>
        <div class="drop-zone">{''.join(starter_cards)}</div>
      </div>
      <div class="drag-lane" data-lane="bench">
        <div class="drag-lane-head"><strong>Bench</strong><span>Drop reserves here</span></div>
        <div class="drop-zone">{''.join(bench_cards)}</div>
      </div>
      <div class="drag-status" role="status" aria-live="polite">Drag cards to manage the lineup.</div>
    </div>
    <script>
      (() => {{
        const board = document.getElementById({json.dumps(board_id)});
        if (!board) return;
        const storageKey = {storage_key};
        let dragged = null;
        const status = board.querySelector('.drag-status');
        const save = () => {{
          const state = {{}};
          board.querySelectorAll('.drag-lane').forEach(lane => {{
            state[lane.dataset.lane] = [...lane.querySelectorAll('.drag-player')].map(card => card.dataset.playerId);
          }});
          window.localStorage.setItem(storageKey, JSON.stringify(state));
        }};
        const move = (card, zone) => {{
          zone.appendChild(card);
          save();
          status.textContent = `${{card.dataset.playerName}} moved to ${{zone.closest('.drag-lane').dataset.lane}}.`;
        }};
        board.querySelectorAll('.drag-player').forEach(card => {{
          card.addEventListener('dragstart', event => {{
            dragged = card;
            card.classList.add('is-dragging');
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', card.dataset.playerId);
          }});
          card.addEventListener('dragend', () => {{ card.classList.remove('is-dragging'); dragged = null; }});
          card.addEventListener('keydown', event => {{
            if (event.key !== 'Enter') return;
            const lane = card.closest('.drag-lane').dataset.lane;
            const target = board.querySelector(`[data-lane="${{lane === 'starters' ? 'bench' : 'starters'}}"] .drop-zone`);
            move(card, target);
          }});
        }});
        board.querySelectorAll('.drop-zone').forEach(zone => {{
          zone.addEventListener('dragover', event => {{ event.preventDefault(); zone.classList.add('is-over'); }});
          zone.addEventListener('dragleave', () => zone.classList.remove('is-over'));
          zone.addEventListener('drop', event => {{
            event.preventDefault();
            zone.classList.remove('is-over');
            if (dragged) move(dragged, zone);
          }});
        }});
        try {{
          const state = JSON.parse(window.localStorage.getItem(storageKey) || 'null');
          if (state) Object.entries(state).forEach(([lane, ids]) => {{
            const zone = board.querySelector(`[data-lane="${{lane}}"] .drop-zone`);
            if (zone) ids.forEach(id => {{
              const card = board.querySelector(`.drag-player[data-player-id="${{CSS.escape(id)}}"]`);
              if (card) zone.appendChild(card);
            }});
          }});
        }} catch (_error) {{ window.localStorage.removeItem(storageKey); }}
      }})();
    </script>
    """


def render_drag_drop_lineup(
    players: Sequence[Mapping[str, Any]],
    *,
    key: str = "quant-lineup",
    height: int = 520,
) -> None:
    """Render the HTML5 drag/drop lineup component."""
    components.html(drag_drop_lineup_html(players, key=key), height=height, scrolling=True)


GLOBAL_CSS = f"""
<style>
  :root {{
    --midnight: {MIDNIGHT_NAVY};
    --cyan: {CIRCUIT_CYAN};
    --gold: {SIGNAL_GOLD};
    --slate: {SLATE_BLUE};
    --soft-white: {SOFT_WHITE};
    --cyan-glow: {CYAN_GLOW};
    --gold-glow: {GOLD_GLOW};
  }}
  html {{color-scheme:dark;}}
  .stApp {{
    background:
      radial-gradient(circle at 86% 2%, rgba(0,200,255,.08), transparent 30rem),
      radial-gradient(circle at 10% 28%, rgba(245,197,66,.045), transparent 26rem),
      linear-gradient(160deg, #071321 0%, var(--midnight) 48%, #081727 100%) !important;
    color:var(--soft-white) !important;
  }}
  .block-container {{max-width:1420px;padding-top:1.6rem;padding-bottom:4rem;}}
  [data-testid="stMain"] p,
  [data-testid="stMain"] li,
  [data-testid="stMain"] label,
  [data-testid="stMain"] strong,
  [data-testid="stMain"] h1,
  [data-testid="stMain"] h2,
  [data-testid="stMain"] h3,
  [data-testid="stMain"] h4,
  [data-testid="stMain"] h5,
  [data-testid="stMain"] h6,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"] {{color:var(--soft-white);}}
  [data-testid="stMain"] [data-testid="stCaptionContainer"],
  [data-testid="stMain"] [data-testid="stCaptionContainer"] p,
  [data-testid="stMain"] small {{color:#9FB0C5 !important;}}
  h1,h2,h3 {{letter-spacing:-.025em !important;}}
  a {{color:var(--cyan);}}

  [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"] {{background:#06111F !important;}}
  [data-testid="stSidebar"] [data-testid="stPageLink"] a {{
    border:1px solid transparent;border-radius:10px;color:#C7D3E2 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
    background:rgba(0,200,255,.07);border-color:rgba(0,200,255,.22);transform:translateX(2px);
  }}
  .quant-top-nav-label {{color:var(--cyan);font-size:.62rem;font-weight:850;letter-spacing:.18em;margin-bottom:.15rem;}}
  [data-testid="stMain"] [data-testid="stPageLink"] a {{
    justify-content:center;border:1px solid rgba(0,200,255,.20);border-radius:9px;
    background:rgba(31,46,69,.64);color:var(--soft-white) !important;
    transition:transform .16s ease,border-color .16s ease,background .16s ease;
  }}
  [data-testid="stMain"] [data-testid="stPageLink"] a:hover {{
    transform:translateY(-1px);border-color:var(--cyan);background:rgba(0,200,255,.08);
  }}
  [data-baseweb="select"] > div,[data-baseweb="input"] > div,
  [data-baseweb="base-input"],textarea {{
    background:var(--slate) !important;color:var(--soft-white) !important;border-color:rgba(0,200,255,.22) !important;
  }}

  .page-head {{border-color:rgba(0,200,255,.20) !important;margin-bottom:1.25rem !important;}}
  .quant-eyebrow {{color:var(--cyan) !important;font-size:.68rem;font-weight:850;letter-spacing:.16em;text-transform:uppercase;}}
  .page-head .eyebrow,.section-kicker,.quant-card-kicker {{color:var(--cyan) !important;}}
  .page-head h1 {{font-size:2rem !important;color:var(--soft-white) !important;}}
  .page-head p {{color:#AFC0D3 !important;}}
  .gradient-section {{
    background:linear-gradient(105deg,rgba(31,46,69,.98),rgba(10,26,47,.98)) !important;
    border:1px solid rgba(0,200,255,.24);border-left:3px solid var(--cyan);
    box-shadow:0 10px 30px rgba(0,0,0,.18) !important;
  }}
  .gradient-section p {{color:#AFC0D3 !important;}}

  [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"],
  [data-testid="stForm"], [data-testid="stDataFrame"], [data-testid="stExpander"],
  .feature-card,.confidence-card,.empty-state {{
    background:linear-gradient(145deg,rgba(31,46,69,.95),rgba(10,26,47,.96)) !important;
    border-color:rgba(0,200,255,.20) !important;
    box-shadow:0 12px 32px rgba(0,0,0,.18) !important;
    color:var(--soft-white) !important;
  }}
  [data-testid="stMetric"]:hover,[data-testid="stVerticalBlockBorderWrapper"]:hover,
  [data-testid="stDataFrame"]:hover,[data-testid="stExpander"]:hover,.feature-card:hover {{
    border-color:rgba(0,200,255,.55) !important;box-shadow:var(--cyan-glow) !important;
    transform:translateY(-2px);
  }}
  [data-testid="stMetricLabel"],.player-card .meta,.player-card .rank {{color:#9FB0C5 !important;}}
  [data-testid="stMetricValue"],.player-card .name,.player-card .stat .v {{color:var(--soft-white) !important;}}
  [data-testid="stMetricDelta"] {{color:var(--cyan) !important;}}
  [data-testid="stDataFrame"] {{overflow:hidden;}}

  .stButton button,[data-testid="stFormSubmitButton"] button {{
    background:linear-gradient(90deg,#087CA0,var(--cyan)) !important;color:#03111E !important;
    border:1px solid rgba(0,200,255,.5) !important;box-shadow:0 8px 22px rgba(0,200,255,.13) !important;
  }}
  .stButton button:hover,[data-testid="stFormSubmitButton"] button:hover {{
    border-color:var(--gold) !important;box-shadow:0 0 20px rgba(245,197,66,.16) !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{background:#071625 !important;border:1px solid rgba(0,200,255,.16);}}
  .stTabs [data-baseweb="tab"] {{color:#9FB0C5 !important;}}
  .stTabs [aria-selected="true"] {{background:var(--slate) !important;color:var(--soft-white) !important;box-shadow:none !important;}}

  .quant-stack {{display:grid;grid-template-columns:1fr;gap:.75rem;margin:.7rem 0 1.35rem;}}
  .quant-card {{
    position:relative;display:block;background:linear-gradient(135deg,rgba(31,46,69,.96),rgba(10,26,47,.98));
    border:1px solid rgba(0,200,255,.22);border-radius:15px;padding:1rem 1.1rem;margin:.6rem 0;
    color:var(--soft-white);box-shadow:0 12px 30px rgba(0,0,0,.18);
    transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease;
  }}
  .quant-card:hover {{transform:translateY(-2px);border-color:rgba(0,200,255,.68);box-shadow:var(--cyan-glow);}}
  .quant-card a {{color:inherit;text-decoration:none;display:block;}}
  .quant-card-head {{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;}}
  .quant-card h3 {{margin:.12rem 0 .35rem;color:var(--soft-white) !important;font-size:1.08rem;}}
  .quant-card p {{margin:.1rem 0;color:#AFC0D3 !important;line-height:1.5;}}
  .quant-card-kicker {{font-size:.69rem;text-transform:uppercase;letter-spacing:.14em;font-weight:800;}}
  .quant-card-link {{margin-top:.65rem;color:var(--cyan);font-size:.78rem;font-weight:750;}}
  .quant-card-stats {{display:flex;flex-wrap:wrap;gap:.55rem 1.3rem;margin-top:.75rem;}}
  .quant-card-stats div {{display:flex;flex-direction:column;}}
  .quant-card-stats span {{color:#8FA2BA;font-size:.67rem;text-transform:uppercase;letter-spacing:.08em;}}
  .quant-card-stats strong {{font-size:1.05rem;color:var(--soft-white);}}
  .quant-card-stats div:first-child strong {{color:var(--gold);text-shadow:var(--gold-glow);}}
  .quant-card-stats div:nth-child(2) strong {{color:var(--cyan);}}

  .rarity-badge {{display:inline-flex;align-items:center;gap:.28rem;border-radius:999px;padding:.18rem .46rem;
    font-size:.66rem;font-weight:800;letter-spacing:.05em;text-transform:uppercase;white-space:nowrap;}}
  .rarity-symbol {{font-size:.78rem;line-height:1;}}
  .rarity-mythic {{color:var(--gold);border:1px solid rgba(245,197,66,.68);background:rgba(245,197,66,.08);box-shadow:var(--gold-glow);}}
  .rarity-legendary {{color:var(--gold);border:1px solid rgba(245,197,66,.62);background:transparent;}}
  .rarity-elite {{color:var(--cyan);border:1px solid rgba(0,200,255,.62);background:rgba(0,200,255,.045);}}
  .rarity-pro {{color:var(--soft-white);border:1px solid rgba(245,197,66,.48);background:var(--midnight);}}
  .rarity-starter {{color:#BED0E5;border:1px solid rgba(159,176,197,.28);background:var(--midnight);}}
  .rarity-depth {{color:#71839A;padding:.1rem .2rem;background:transparent;}}

  .player-card {{background:linear-gradient(145deg,rgba(31,46,69,.96),rgba(10,26,47,.98));border-radius:14px;padding:.9rem;
    border:1px solid rgba(0,200,255,.18);transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease;}}
  .player-card:hover {{transform:translateY(-2px);border-color:rgba(0,200,255,.62);box-shadow:var(--cyan-glow);}}
  .player-detail-title {{display:flex;align-items:center;gap:.8rem;margin:.15rem 0 .9rem;}}
  .player-detail-title h2 {{margin:0 !important;}}
  .player-detail-title p {{margin:.15rem 0 0 !important;color:#9FB0C5 !important;}}
  .sports-hub-card,.weekly-tool-card,.quick-action-card,.roster-player-card {{border-left:3px solid var(--cyan);}}
  .gold-accent {{border-left-color:var(--gold);}}

  .gold-chart-shell {{border:1px solid rgba(245,197,66,.17);border-radius:15px;padding:.2rem;background:var(--midnight);box-shadow:var(--gold-glow);}}
  .drag-lineup-board {{font-family:Inter,Segoe UI,sans-serif;background:{MIDNIGHT_NAVY};color:{SOFT_WHITE};display:grid;
    grid-template-columns:1fr 1fr;gap:12px;padding:12px;min-height:420px;}}
  .drag-lane {{background:{SLATE_BLUE};border:1px solid rgba(0,200,255,.24);border-radius:14px;padding:10px;}}
  .drag-lane-head {{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px;}}
  .drag-lane-head strong {{color:{SOFT_WHITE};font-size:14px;}} .drag-lane-head span {{color:#91A5BE;font-size:11px;}}
  .drop-zone {{min-height:340px;border:1px dashed rgba(0,200,255,.18);border-radius:10px;padding:7px;}}
  .drop-zone.is-over {{border-color:{CIRCUIT_CYAN};background:rgba(0,200,255,.06);box-shadow:{CYAN_GLOW};}}
  .drag-player {{display:flex;align-items:center;gap:8px;background:#0D2037;border:1px solid rgba(0,200,255,.2);border-radius:10px;
    padding:9px;margin:7px 0;cursor:grab;transition:border-color .14s ease,box-shadow .14s ease,opacity .14s ease;}}
  .drag-player:hover,.drag-player:focus {{outline:none;border-color:{CIRCUIT_CYAN};box-shadow:{CYAN_GLOW};}}
  .drag-player.is-dragging {{opacity:.45;}} .drag-copy {{display:flex;flex:1;flex-direction:column;}}
  .drag-copy strong {{color:{SOFT_WHITE};font-size:12px;}} .drag-copy span {{color:#91A5BE;font-size:10px;}}
  .drag-handle {{color:{CIRCUIT_CYAN};letter-spacing:-2px;}} .drag-status {{grid-column:1/-1;color:#91A5BE;font-size:11px;padding:2px 4px;}}

  @media (max-width:820px) {{.drag-lineup-board {{grid-template-columns:1fr;}}.quant-card-stats {{gap:.45rem .9rem;}}}}
  @media (prefers-reduced-motion:reduce) {{*,*::before,*::after {{transition-duration:.001ms !important;animation-duration:.001ms !important;}}}}
</style>
"""


def inject_style() -> None:
    """Emit the complete Universal Quant visual identity CSS."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


__all__ = [
    "CIRCUIT_CYAN",
    "CYAN_GLOW",
    "GLOBAL_CSS",
    "GOLD_GLOW",
    "GOLD_GLOW_CHART_THEME",
    "MIDNIGHT_NAVY",
    "PALETTE",
    "RARITY_TIERS",
    "SIGNAL_GOLD",
    "SLATE_BLUE",
    "SOFT_WHITE",
    "SPACING_SCALE",
    "TYPOGRAPHY_SCALE",
    "apply_gold_glow_theme",
    "drag_drop_lineup_html",
    "gold_glow_line_chart",
    "inject_style",
    "rarity_icon",
    "rarity_icon_component",
    "rarity_metadata",
    "rarity_tier",
    "render_drag_drop_lineup",
    "render_stacked_card",
    "safe_number",
    "stacked_card_html",
]
