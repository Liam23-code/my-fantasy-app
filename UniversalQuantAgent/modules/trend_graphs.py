"""Storytelling charts for NBA momentum, timelines, matchups, and pace."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_quality import safe_number
from modules.graph_data import (
    load_player_table, load_team_table, percentile, player_game_data,
    resolve_player, team_palette,
)
from modules.nba_advanced import latest_season

POSITIONS = ("Guards", "Wings", "Bigs")

def _column(table: pd.DataFrame, name: str, fallback: float = 0.0) -> pd.Series:
    source = (
        table[name]
        if name in table
        else pd.Series(fallback, index=table.index, dtype=float)
    )
    return pd.to_numeric(source, errors="coerce").fillna(fallback)

def _style(figure: go.Figure, title: str, height: int = 480) -> go.Figure:
    figure.update_layout(
        title=title, template="plotly_white", height=height,
        margin=dict(l=30, r=30, t=65, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.48)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#24344d"),
        hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0),
        hoverlabel=dict(bgcolor="white", font_color="#10213a"),
    )
    figure.update_xaxes(gridcolor="rgba(100,116,139,.13)")
    figure.update_yaxes(gridcolor="rgba(100,116,139,.13)")
    return figure


def _team_for_games(player: dict[str, Any], season: str) -> str:
    table = load_player_table(season)
    if table.empty or "PLAYER_ID" not in table:
        return ""
    matches = table[pd.to_numeric(table["PLAYER_ID"], errors="coerce") == int(player["id"])]
    return str(matches.iloc[0].get("TEAM_ABBREVIATION", "")) if not matches.empty else ""


def render_momentum_line(
    player_id: int | str,
    season: str | None = None,
) -> go.Figure:
    """Return rolling five-game points, usage, TS%, and minutes momentum."""
    season = season or latest_season()
    player, data, warnings = player_game_data(player_id, season)
    team = _team_for_games(player, season)
    primary, secondary = team_palette(team)
    series = [
        ("Points", "points", primary), ("Usage %", "usage_pct", secondary),
        ("TS%", "ts_pct", "#138A55"), ("Minutes", "minutes", "#5B6CFF"),
    ]
    figure = go.Figure()
    for label, column, color in series:
        values = pd.to_numeric(data.get(column, 0), errors="coerce").fillna(0)
        rolling = values.rolling(5, min_periods=1).mean()
        figure.add_trace(go.Scatter(
            x=data["game_date"], y=rolling, mode="lines+markers", name=label,
            line=dict(color=color, width=2.6), marker=dict(size=5),
            customdata=np.column_stack([values]),
            hovertemplate=f"{label} rolling %{{y:.1f}}<br>Game value %{{customdata[0]:.1f}}<extra></extra>",
        ))
    figure = _style(figure, f"{player['full_name']} · five-game momentum")
    figure.update_layout(meta={"player": player["full_name"], "season": season, "warnings": warnings})
    return figure


def render_performance_timeline(
    player_id: int | str,
    season: str | None = None,
) -> go.Figure:
    """Return a season-long production timeline with minutes context."""
    season = season or latest_season()
    player, data, warnings = player_game_data(player_id, season)
    team = _team_for_games(player, season)
    primary, secondary = team_palette(team)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for label, column, color in (
        ("Points", "points", primary), ("Rebounds", "rebounds", secondary),
        ("Assists", "assists", "#5B6CFF"),
    ):
        figure.add_trace(go.Scatter(
            x=data["game_date"], y=data[column], mode="lines+markers", name=label,
            line=dict(color=color, width=2), marker=dict(size=4),
            hovertemplate=f"{label} %{{y:.1f}}<extra></extra>",
        ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=data["game_date"], y=data["minutes"], mode="lines", name="Minutes",
        line=dict(color="#64748B", width=2, dash="dot"),
        hovertemplate="Minutes %{y:.1f}<extra></extra>",
    ), secondary_y=True)
    figure = _style(figure, f"{player['full_name']} · performance timeline", 520)
    figure.update_yaxes(title_text="Production", secondary_y=False)
    figure.update_yaxes(title_text="Minutes", secondary_y=True, showgrid=False)
    figure.update_layout(meta={"player": player["full_name"], "season": season, "warnings": warnings})
    return figure


def render_usage_efficiency_scatter(
    season: str | None = None,
    highlight_player: int | str | None = None,
) -> go.Figure:
    """Return league usage versus true-shooting context."""
    season = season or latest_season()
    table = load_player_table(season).copy()
    warnings = list(table.attrs.get("warnings", []))
    if table.empty:
        table = pd.DataFrame([{"PLAYER_NAME":"No data", "USG_PCT":.2, "TS_PCT":.55, "MIN":0, "PTS":0, "TEAM_ABBREVIATION":""}])
        warnings.append("League player table is unavailable.")
    usage = _column(table, "USG_PCT", .2)
    shooting = _column(table, "TS_PCT", .55)
    table["usage_display"] = np.where(usage.abs() <= 1, usage * 100, usage)
    table["ts_display"] = np.where(shooting.abs() <= 1, shooting * 100, shooting)
    table["minutes_display"] = _column(table, "MIN")
    table["points_display"] = _column(table, "PTS")
    table["PLAYER_NAME"] = table.get("PLAYER_NAME", pd.Series("Player", index=table.index)).astype(str)
    table["TEAM_ABBREVIATION"] = table.get("TEAM_ABBREVIATION", pd.Series("", index=table.index)).astype(str)
    sizes = 7 + 15 * table["minutes_display"].div(max(safe_number(table["minutes_display"].max()), 1))
    figure = go.Figure(go.Scatter(
        x=table["usage_display"], y=table["ts_display"], mode="markers",
        marker=dict(size=sizes, color=table["points_display"], colorscale="Viridis", opacity=.68, colorbar=dict(title="PPG"), line=dict(color="white", width=.7)),
        customdata=np.column_stack([table["PLAYER_NAME"], table["TEAM_ABBREVIATION"], table["minutes_display"], table["points_display"]]),
        hovertemplate="%{customdata[0]} · %{customdata[1]}<br>Usage %{x:.1f}%<br>TS %{y:.1f}%<br>Minutes %{customdata[2]:.1f}<br>Points %{customdata[3]:.1f}<extra></extra>",
        name="NBA players",
    ))
    if highlight_player is not None and "PLAYER_ID" in table:
        player = resolve_player(highlight_player)
        selected = table[pd.to_numeric(table["PLAYER_ID"], errors="coerce") == int(player["id"])]
        if not selected.empty:
            figure.add_trace(go.Scatter(
                x=selected["usage_display"], y=selected["ts_display"], mode="markers+text",
                text=selected["PLAYER_NAME"], textposition="top center", name="Selected",
                marker=dict(size=18, color="#FF5A36", symbol="diamond", line=dict(color="#10213A", width=2)),
                hovertemplate="%{text}<br>Usage %{x:.1f}%<br>TS %{y:.1f}%<extra></extra>",
            ))
    figure = _style(figure, "Usage versus efficiency · league landscape", 560)
    figure.update_xaxes(title="Usage rate (%)")
    figure.update_yaxes(title="True shooting (%)")
    figure.update_layout(hovermode="closest", meta={"season": season, "warnings": warnings})
    return figure


def render_opponent_difficulty_curve(
    position: str,
    season: str | None = None,
) -> go.Figure:
    """Rank team defensive difficulty for guards, wings, or bigs."""
    season = season or latest_season()
    position = position if position in POSITIONS else "Guards"
    table = load_team_table(season).copy()
    warnings = list(table.attrs.get("warnings", []))
    defense = _column(table, "DEF_RATING", 110)
    base = defense.rank(pct=True, ascending=False) * 100
    steals = _column(table, "STL", 7.5).rank(pct=True) * 100
    blocks = _column(table, "BLK", 5).rank(pct=True) * 100
    rebounds = _column(table, "REB", 44).rank(pct=True) * 100
    if position == "Guards":
        score = .7 * base + .3 * steals
    elif position == "Wings":
        score = .65 * base + .18 * steals + .17 * blocks
    else:
        score = .58 * base + .25 * blocks + .17 * rebounds
    table["difficulty"] = score.round(1)
    table = table.sort_values("difficulty")
    labels = table.get("TEAM_ABBREVIATION", table.get("TEAM_NAME", pd.Series("Team", index=table.index))).astype(str)
    figure = go.Figure(go.Scatter(
        x=labels, y=table["difficulty"], mode="lines+markers", fill="tozeroy",
        line=dict(color="#5B6CFF", width=3, shape="spline"),
        marker=dict(size=8, color=table["difficulty"], colorscale=[[0,"#138A55"],[.5,"#F7D154"],[1,"#D94A3A"]], cmin=0, cmax=100),
        customdata=np.column_stack([defense.loc[table.index]]),
        hovertemplate="%{x}<br>Difficulty %{y:.1f}/100<br>Def rating %{customdata[0]:.1f}<extra></extra>",
        name=position,
    ))
    figure = _style(figure, f"Opponent difficulty curve · {position}", 520)
    figure.update_yaxes(title="Difficulty score", range=[0, 105])
    figure.update_xaxes(title="Opponent", tickangle=-45)
    figure.update_layout(hovermode="closest", showlegend=False, meta={"position":position,"season":season,"warnings":warnings})
    return figure


def render_pace_impact_curve(season: str | None = None) -> go.Figure:
    """Show the observed league relationship between pace and box-score output."""
    season = season or latest_season()
    table = load_team_table(season).copy()
    warnings = list(table.attrs.get("warnings", []))
    pace = _column(table, "PACE", 100)
    order = np.argsort(pace.to_numpy())
    x_sorted = pace.to_numpy()[order]
    figure = go.Figure()
    for label, column, color in (
        ("Points", "PTS", "#FF5A36"), ("Rebounds", "REB", "#5B6CFF"),
        ("Assists", "AST", "#138A55"),
    ):
        values = _column(table, column).to_numpy()
        figure.add_trace(go.Scatter(
            x=pace, y=values, mode="markers", name=f"{label} teams",
            marker=dict(size=7, color=color, opacity=.25), showlegend=False,
            hovertemplate=f"Pace %{{x:.1f}}<br>{label} %{{y:.1f}}<extra></extra>",
        ))
        if len(np.unique(x_sorted)) >= 2:
            degree = 2 if len(x_sorted) >= 3 else 1
            curve = np.polyval(np.polyfit(pace.to_numpy(), values, degree), x_sorted)
        else:
            curve = values[order]
        figure.add_trace(go.Scatter(
            x=x_sorted, y=curve, mode="lines", name=label,
            line=dict(color=color, width=3, shape="spline"),
            hovertemplate=f"Pace %{{x:.1f}}<br>Expected {label.lower()} %{{y:.1f}}<extra></extra>",
        ))
    figure = _style(figure, "Pace impact curve · NBA team production", 540)
    figure.update_xaxes(title="Possessions per 48 minutes")
    figure.update_yaxes(title="Per-game output")
    figure.update_layout(meta={"season":season,"warnings":warnings})
    return figure