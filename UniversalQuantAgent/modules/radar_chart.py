"""Team-colored efficiency radar charts for NBA player profiles."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.data_quality import safe_number
from modules.graph_data import percentile, player_row, team_palette
from modules.nba_advanced import latest_season

RADAR_WINDOWS = {"Season": 0, "Last-10": 10, "Last-5": 5}

def _rgba(hex_color: str, alpha: float) -> str:
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        return f"rgba(51,65,85,{alpha})"
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"

def _column(table: pd.DataFrame, name: str, fallback: float = 0.0) -> pd.Series:
    if name in table:
        return pd.to_numeric(table[name], errors="coerce").fillna(fallback)
    return pd.Series(fallback, index=table.index, dtype=float)


def _ratio(table: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    return _column(table, numerator).div(_column(table, denominator).replace(0, np.nan)).fillna(0)


def _radar_metrics(row: pd.Series, table: pd.DataFrame) -> tuple[list[str], list[float], list[str]]:
    data = table.copy()
    if "TS_PCT" not in data:
        attempts = _column(data, "FGA") + .44 * _column(data, "FTA")
        data["TS_PCT"] = _column(data, "PTS").div(2 * attempts.replace(0, np.nan)).fillna(.55)
    if "USG_PCT" not in data:
        data["USG_PCT"] = (_column(data, "FGA") + .44 * _column(data, "FTA") + _column(data, "TOV")).div(_column(data, "MIN").replace(0, np.nan)).fillna(.2)
    data["ASSIST_RATE"] = _column(data, "AST_PCT") if "AST_PCT" in data else _ratio(data, "AST", "MIN")
    data["REBOUND_RATE"] = _column(data, "REB_PCT") if "REB_PCT" in data else _ratio(data, "REB", "MIN")
    if "TM_TOV_PCT" in data:
        data["TURNOVER_RATE"] = _column(data, "TM_TOV_PCT")
    else:
        possession_events = _column(data, "FGA") + .44 * _column(data, "FTA") + _column(data, "TOV")
        data["TURNOVER_RATE"] = _column(data, "TOV").div(possession_events.replace(0, np.nan)).fillna(0)
    data["DEFENSE"] = (
        percentile(data["STL"], row.get("STL")) if "STL" in data else 50.0
    )
    data["PACE_FIT"] = _column(data, "PACE", 100.0)

    target_id = safe_number(row.get("PLAYER_ID"), -1)
    if "PLAYER_ID" in data:
        target_rows = data[pd.to_numeric(data["PLAYER_ID"], errors="coerce") == target_id]
        target = target_rows.iloc[0] if not target_rows.empty else row
    else:
        target = row

    defensive_parts = []
    if "STL" in data:
        defensive_parts.append(percentile(data["STL"], target.get("STL")))
    if "BLK" in data:
        defensive_parts.append(percentile(data["BLK"], target.get("BLK")))
    if "DEF_RATING" in data:
        defensive_parts.append(percentile(data["DEF_RATING"], target.get("DEF_RATING"), False))
    defensive_score = sum(defensive_parts) / len(defensive_parts) if defensive_parts else 50.0
    pace_values = _column(data, "PACE", 100.0)
    league_pace = safe_number(pace_values.mean(), 100.0)
    pace_std = max(safe_number(pace_values.std(), 1.0), 1.0)
    target_pace = safe_number(target.get("PACE"), league_pace)
    pace_fit = max(0.0, min(100.0, 100.0 - abs(target_pace - league_pace) / pace_std * 15.0))

    specs = [
        ("TS%", "TS_PCT", True, "%"),
        ("Usage", "USG_PCT", True, "%"),
        ("Assist rate", "ASSIST_RATE", True, "%"),
        ("Rebound rate", "REBOUND_RATE", True, "%"),
        ("Turnover rate", "TURNOVER_RATE", False, "%"),
    ]
    labels, ratings, actual = [], [], []
    for label, column, higher, suffix in specs:
        raw = safe_number(target.get(column))
        display = raw * 100.0 if abs(raw) <= 1.0 else raw
        labels.append(label)
        ratings.append(percentile(data[column], raw, higher))
        actual.append(f"{display:.1f}{suffix}")
    labels.extend(["Defensive impact", "Pace fit"])
    ratings.extend([round(defensive_score, 1), round(pace_fit, 1)])
    actual.extend([f"{defensive_score:.0f} percentile", f"{target_pace:.1f} pace"])
    return labels, ratings, actual


def render_efficiency_radar(
    player_id: int | str,
    season: str | None = None,
    window: str = "Season",
) -> go.Figure:
    """Return an efficiency radar with auditable percentile-based axes."""
    season = season or latest_season()
    window = window if window in RADAR_WINDOWS else "Season"
    player, row, table, warnings = player_row(
        player_id, season, last_n_games=RADAR_WINDOWS[window]
    )
    labels, ratings, actual = _radar_metrics(row, table)
    team = str(row.get("TEAM_ABBREVIATION", ""))
    primary, secondary = team_palette(team)
    closed_labels = labels + [labels[0]]
    closed_ratings = ratings + [ratings[0]]
    closed_actual = actual + [actual[0]]
    figure = go.Figure(go.Scatterpolar(
        r=closed_ratings, theta=closed_labels, customdata=closed_actual,
        mode="lines+markers", fill="toself",
        line=dict(color=primary, width=3, shape="spline"),
        marker=dict(color=secondary, size=8, line=dict(color=primary, width=2)),
        fillcolor=_rgba(primary, .20),
        hovertemplate="%{theta}<br>Rating %{r:.1f}/100<br>Actual %{customdata}<extra></extra>",
        name=player["full_name"],
    ))
    figure.update_layout(
        title=f"{player['full_name']} · {window} efficiency profile",
        template="plotly_white", height=590,
        margin=dict(l=65, r=65, t=75, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,.45)",
            radialaxis=dict(range=[0, 100], tickvals=[20, 40, 60, 80, 100], ticksuffix="", gridcolor="rgba(100,116,139,.18)"),
            angularaxis=dict(gridcolor="rgba(100,116,139,.18)", linecolor=primary, linewidth=3),
        ),
        showlegend=False,
        meta={"player": player["full_name"], "team": team, "season": season, "window": window, "warnings": warnings},
    )
    return figure