"""Interactive NBA half-court shot heatmaps for Graph Lab."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nba_api.stats.endpoints import shotchartdetail

from modules.data_quality import normalize_columns, safe_number
from modules.graph_data import player_row, resolve_player, team_palette
from modules.nba_advanced import latest_season
from modules.nba_cache import NBA_API_TIMEOUT, fetch_nba_frames

SHOT_MODES = ("Makes only", "Attempts only", "Efficiency", "Volume")


def _rgba(hex_color: str, alpha: float) -> str:
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        return f"rgba(51,65,85,{alpha})"
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _court_trace(x: list[float], y: list[float], name: str = "") -> go.Scatter:
    return go.Scatter(
        x=x, y=y, mode="lines", name=name, hoverinfo="skip", showlegend=False,
        line=dict(color="rgba(51,65,85,.72)", width=2),
    )


def _arc(radius: float, start: float, end: float, center_y: float = 0.0) -> tuple[list[float], list[float]]:
    radians = np.linspace(np.deg2rad(start), np.deg2rad(end), 90)
    return (radius * np.cos(radians)).tolist(), (center_y + radius * np.sin(radians)).tolist()


def _add_half_court(figure: go.Figure) -> None:
    """Draw an NBA-style half court in shot-chart coordinates."""
    figure.add_trace(_court_trace([-250, -250, 250, 250, -250], [-52.5, 470, 470, -52.5, -52.5]))
    figure.add_trace(_court_trace([-80, -80, 80, 80], [-47.5, 142.5, 142.5, -47.5]))
    figure.add_trace(_court_trace([-30, 30], [-7.5, -7.5]))
    figure.add_trace(_court_trace([-220, -220], [-47.5, 92.5]))
    figure.add_trace(_court_trace([220, 220], [-47.5, 92.5]))
    x, y = _arc(237.5, 22.9, 157.1)
    figure.add_trace(_court_trace(x, y))
    x, y = _arc(60, 0, 360, 142.5)
    figure.add_trace(_court_trace(x, y))
    x, y = _arc(40, 0, 180)
    figure.add_trace(_court_trace(x, y))
    x, y = _arc(7.5, 0, 360)
    figure.add_trace(_court_trace(x, y))


def fetch_shot_locations(
    player_id: int | str,
    season: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch location-level shots through retry, cache, and empty fallback."""
    season = season or latest_season()
    player = resolve_player(player_id)
    frames = fetch_nba_frames(
        f"graph_shots_{player['id']}_{season}",
        lambda: shotchartdetail.ShotChartDetail(
            player_id=int(player["id"]), team_id=0,
            season_nullable=season, season_type_all_star="Regular Season",
            context_measure_simple="FGA", timeout=NBA_API_TIMEOUT,
        ).get_data_frames(),
        fallback_factory=pd.DataFrame,
    )
    data = normalize_columns(frames[0] if frames else pd.DataFrame())
    warnings = list(data.attrs.get("warnings", [])) if isinstance(data, pd.DataFrame) else []
    for column in ("loc_x", "loc_y", "shot_made_flag", "shot_attempted_flag", "shot_distance"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if not {"loc_x", "loc_y"}.issubset(data.columns):
        data = pd.DataFrame(columns=[
            "loc_x", "loc_y", "shot_made_flag", "shot_attempted_flag", "shot_type",
        ])
        warnings.append("Shot-location data is unavailable; the court remains available for orientation.")
    return data, list(dict.fromkeys(str(item) for item in warnings if item))


def _shot_grid(shots: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(-250, 250, 26)
    y_edges = np.linspace(-50, 470, 27)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    shape = (len(y_centers), len(x_centers))
    values = np.full(shape, np.nan)
    custom = np.empty(shape + (4,), dtype=object)
    custom[:] = None
    if shots.empty:
        return x_centers, y_centers, values, custom

    data = shots.dropna(subset=["loc_x", "loc_y"]).copy()
    data["x_bin"] = np.digitize(data["loc_x"], x_edges) - 1
    data["y_bin"] = np.digitize(data["loc_y"], y_edges) - 1
    data = data[
        data["x_bin"].between(0, len(x_centers) - 1)
        & data["y_bin"].between(0, len(y_centers) - 1)
    ]
    attempts_total = max(len(data), 1)
    made = pd.to_numeric(data.get("shot_made_flag", 0), errors="coerce").fillna(0)
    data["made"] = made
    shot_types = (
        data["shot_type"].astype(str)
        if "shot_type" in data
        else pd.Series("2PT Field Goal", index=data.index)
    )
    data["shot_value"] = shot_types.str.contains("3PT", case=False).map({True: 3.0, False: 2.0})
    for (x_bin, y_bin), group in data.groupby(["x_bin", "y_bin"]):
        attempts = len(group)
        makes = safe_number(group["made"].sum())
        field_goal = makes / attempts * 100.0 if attempts else 0.0
        expected_points = safe_number((group["made"] * group["shot_value"]).sum()) / attempts if attempts else 0.0
        if mode == "Makes only":
            intensity = makes
        elif mode == "Efficiency":
            intensity = field_goal
        elif mode == "Volume":
            intensity = attempts / attempts_total * 100.0
        else:
            intensity = attempts
        values[int(y_bin), int(x_bin)] = intensity
        custom[int(y_bin), int(x_bin)] = [attempts, makes, field_goal, expected_points]
    return x_centers, y_centers, values, custom


def render_shot_chart(
    player_id: int | str,
    season: str | None = None,
    mode: str = "Attempts only",
    shots: pd.DataFrame | None = None,
) -> go.Figure:
    """Return an interactive team-colored half-court density heatmap."""
    season = season or latest_season()
    mode = mode if mode in SHOT_MODES else "Attempts only"
    player, row, _, profile_warnings = player_row(player_id, season)
    if shots is None:
        shots, shot_warnings = fetch_shot_locations(int(player["id"]), season)
    else:
        shots, shot_warnings = normalize_columns(shots.copy()), []
        for column in ("loc_x", "loc_y", "shot_made_flag", "shot_attempted_flag"):
            if column in shots:
                shots[column] = pd.to_numeric(shots[column], errors="coerce")
    team = str(row.get("TEAM_ABBREVIATION", ""))
    primary, secondary = team_palette(team)
    x, y, z, custom = _shot_grid(shots, mode)
    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        x=x, y=y, z=z, customdata=custom, zsmooth="best", opacity=.88,
        colorscale=[[0, _rgba(primary, .05)], [.55, primary], [1, secondary]],
        colorbar=dict(title=mode, thickness=13),
        hovertemplate=(
            "Attempts %{customdata[0]:.0f}<br>Makes %{customdata[1]:.0f}"
            "<br>FG% %{customdata[2]:.1f}%<br>Expected points %{customdata[3]:.2f}<extra></extra>"
        ),
    ))
    _add_half_court(figure)
    figure.update_layout(
        title=f"{player['full_name']} · {mode}",
        template="plotly_white", height=690,
        margin=dict(l=18, r=18, t=58, b=18),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)",
        xaxis=dict(range=[-255, 255], visible=False, constrain="domain"),
        yaxis=dict(range=[-55, 475], visible=False, scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="white", font_color="#10213a"),
        meta={
            "player": player["full_name"], "player_id": int(player["id"]),
            "team": team, "season": season, "mode": mode,
            "attempts": int(len(shots)),
            "warnings": list(dict.fromkeys(profile_warnings + shot_warnings)),
        },
    )
    if shots.empty:
        figure.add_annotation(
            x=0, y=225, text="Shot locations unavailable", showarrow=False,
            font=dict(size=18, color="#64748b"), bgcolor="rgba(255,255,255,.78)",
        )
    return figure