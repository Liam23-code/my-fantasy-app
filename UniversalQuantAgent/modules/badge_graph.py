"""Accurate, sample-aware badge-wheel profiles for NBA players.

The engine separates measurement from presentation. ``calculate_badge_profile``
builds transparent league and position percentiles, while ``render_badge_graph``
turns that profile into a lightweight Plotly wheel. Public NBA data can be
partial, so every derived value has a documented proxy and confidence score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.data_quality import safe_number
from modules.graph_data import load_team_table, player_row, team_palette
from modules.nba_advanced import latest_season


BADGE_COLORS = {
    "Elite": "#00C800",
    "Very good": "#78DC00",
    "Average": "#FFC800",
    "Below average": "#FF8C00",
    "Weak": "#FF3C3C",
}
BADGE_RGBA = {
    "Elite": "rgba(0, 200, 0, 0.50)",
    "Very good": "rgba(120, 220, 0, 0.50)",
    "Average": "rgba(255, 200, 0, 0.50)",
    "Below average": "rgba(255, 140, 0, 0.50)",
    "Weak": "rgba(255, 60, 60, 0.50)",
}
BADGE_ICONS = {
    "3PT shooting": "◎",
    "Mid-range": "⌖",
    "Finishing": "◉",
    "Playmaking": "🎨",
    "Defense": "🛡",
    "Rebounding": "📈",
    "Pace compatibility": "⚡",
    "Efficiency": "✦",
}
DEFAULT_SAMPLE_MINIMUMS = {
    "3PT shooting": 100.0,
    "Mid-range": 75.0,
    "Finishing": 100.0,
    "Free throws": 100.0,
    "Total shots": 300.0,
}
DISPLAY_MODES = ("Raw", "Adjusted")
COMPARISON_MODES = ("Entire league", "Same position")


def _series(table: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    """Return the first available numeric column with a stable index."""
    for name in names:
        if name in table:
            return pd.to_numeric(table[name], errors="coerce").fillna(default)
    return pd.Series(default, index=table.index, dtype=float)


def _rate_rank(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Convert one metric to league percentiles without rewarding constants."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 2 or valid.nunique() < 2:
        return pd.Series(50.0, index=values.index, dtype=float)
    fill = float(valid.median())
    return numeric.fillna(fill).rank(
        pct=True, method="average", ascending=higher_is_better
    ).mul(100.0)


def _composite(*components: tuple[pd.Series, bool]) -> pd.Series:
    """Average component percentiles so unlike basketball units can combine."""
    ranked = [_rate_rank(values, higher) for values, higher in components]
    if not ranked:
        return pd.Series(dtype=float)
    return pd.concat(ranked, axis=1).mean(axis=1)


def _percentile_against(values: pd.Series, target: float) -> float:
    """Rank target against a qualified comparison pool on a 0-100 scale."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2 or numeric.nunique() < 2:
        return 50.0
    return round(float((numeric <= target).mean() * 100.0), 1)


def _tier(value: float) -> str:
    if value >= 90:
        return "Elite"
    if value >= 80:
        return "Very good"
    if value >= 70:
        return "Average"
    if value >= 60:
        return "Below average"
    return "Weak"


def _skill_label(value: float) -> str:
    tier = _tier(value)
    return {
        "Elite": "Elite",
        "Very good": "Strong",
        "Average": "Average",
        "Below average": "Below average",
        "Weak": "Weak",
    }[tier]


def _position_group(row: pd.Series) -> str:
    """Use provider position when available, otherwise infer a broad role."""
    raw = str(
        row.get("POSITION") or row.get("PLAYER_POSITION")
        or row.get("START_POSITION") or row.get("POS") or ""
    ).upper()
    if "C" in raw:
        return "Big"
    if "G" in raw and "F" not in raw:
        return "Guard"
    if "F" in raw:
        return "Wing"

    ast_rate = safe_number(row.get("AST_PCT"), 0.0)
    reb_rate = safe_number(row.get("REB_PCT"), 0.0)
    blocks = safe_number(row.get("BLK"), 0.0)
    assists = safe_number(row.get("AST"), 0.0)
    rebounds = safe_number(row.get("REB"), 0.0)
    # Interior workload wins over creation so playmaking centers are not
    # mislabeled as guards when the provider omits its position column.
    if reb_rate >= .14 or rebounds >= 8 or blocks >= 1.2:
        return "Big"
    if ast_rate >= .24 or assists >= 6:
        return "Guard"
    return "Wing"


def _dynamic_minimum(default_minimum: float, total_shots: float) -> float:
    """Apply the requested threshold rule without dividing by zero."""
    if total_shots <= 0:
        return 0.0
    return min(float(default_minimum), float(total_shots) * .35)


def _confidence(attempts: float, dynamic_minimum: float) -> float:
    if attempts <= 0 or dynamic_minimum <= 0:
        return 0.0
    return round(min(1.0, attempts / dynamic_minimum), 4)


def _prepare_metrics(table: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Build comparable skills with explicit close-shot dilution.

    When a shot-zone provider is unavailable, the two-point attempt mix uses a
    conservative partition that sums to 100%. The same dilution equations are
    then applied to every player, preserving position fairness.
    """
    data = table.copy().reset_index(drop=True)
    gp = _series(data, "GP", default=1.0).replace(0, 1.0)
    fga_raw = _series(data, "FGA")
    per_game_mode = bool(fga_raw.dropna().median() <= 40 and gp.median() > 2)

    def total(*names: str) -> pd.Series:
        values = _series(data, *names)
        return values.mul(gp) if per_game_mode else values

    def per_game(*names: str) -> pd.Series:
        values = _series(data, *names)
        return values if per_game_mode else values.div(gp.replace(0, np.nan)).fillna(0)

    total_fga = total("FGA")
    total_fg3a = total("FG3A")
    total_fta = total("FTA")
    total_fg2a = (total_fga - total_fg3a).clip(lower=0)
    total_fg2m = (total("FGM") - total("FG3M")).clip(lower=0)
    fg2_pct = total_fg2m.div(total_fg2a.replace(0, np.nan)).fillna(0)

    true_mid_attempts = total(
        "TRUE_MID_RANGE_FGA", "MID_RANGE_FGA", "MIDRANGE_FGA"
    )
    true_mid_components = (
        "PULLUP_FGA", "FADEAWAY_FGA", "ELBOW_FGA",
        "LONG_FLOATER_FGA", "SHORT_JUMPER_FGA",
    )
    if not true_mid_attempts.any():
        available_mid_components = [name for name in true_mid_components if name in data]
        if available_mid_components:
            true_mid_attempts = sum(
                (total(name) for name in available_mid_components),
                pd.Series(0.0, index=data.index),
            )
    layup_attempts = total("LAYUP_FGA", "LAYUP_ATTEMPTS")
    close_attempts = total(
        "CLOSE_SHOT_FGA", "PAINT_NON_RA_FGA", "FLOATER_FGA",
        "TOUCH_SHOT_FGA",
    )
    dunk_attempts = total("DUNK_FGA", "DUNK_ATTEMPTS")
    # Stable proxies are only used when the public league table omits zones.
    if not true_mid_attempts.any():
        true_mid_attempts = total_fg2a.mul(.35)
    if not layup_attempts.any():
        layup_attempts = total_fg2a.mul(.42)
    if not close_attempts.any():
        close_attempts = total_fg2a.mul(.15)
    if not dunk_attempts.any():
        dunk_attempts = total_fg2a.mul(.08)

    mid_pct = _series(
        data, "TRUE_MID_RANGE_FG_PCT", "MID_RANGE_FG_PCT", "MIDRANGE_FG_PCT"
    )
    if not mid_pct.any():
        component_makes = pd.Series(0.0, index=data.index)
        component_attempts = pd.Series(0.0, index=data.index)
        for attempt_name in true_mid_components:
            pct_name = attempt_name.replace("FGA", "FG_PCT")
            if attempt_name in data and pct_name in data:
                attempts_for_zone = total(attempt_name)
                component_attempts = component_attempts + attempts_for_zone
                component_makes = component_makes + attempts_for_zone * _series(data, pct_name)
        if component_attempts.any():
            mid_pct = component_makes.div(component_attempts.replace(0, np.nan)).fillna(0)
    layup_pct = _series(data, "LAYUP_FG_PCT")
    close_pct = _series(
        data, "CLOSE_SHOT_FG_PCT", "PAINT_NON_RA_FG_PCT", "FLOATER_FG_PCT"
    )
    dunk_pct = _series(data, "DUNK_FG_PCT")
    if not mid_pct.any():
        mid_pct = fg2_pct
    if not layup_pct.any():
        layup_pct = fg2_pct
    if not close_pct.any():
        close_pct = fg2_pct
    if not dunk_pct.any():
        dunk_pct = fg2_pct

    # Requested dilution equations. Contributions are makes-equivalent values,
    # so the volume and corresponding zone efficiency remain visible together.
    layup_contribution = layup_attempts * layup_pct
    close_finish_contribution = close_attempts * .60 * close_pct
    dunk_finish_contribution = dunk_attempts * .90 * dunk_pct
    finishing_contribution = (
        layup_contribution + close_finish_contribution + dunk_finish_contribution
    )
    true_mid_contribution = true_mid_attempts * mid_pct
    close_mid_contribution = close_attempts * .55 * close_pct
    midrange_contribution = true_mid_contribution + close_mid_contribution

    effective_finishing_attempts = (
        layup_attempts + close_attempts * .60 + dunk_attempts * .90
    )
    effective_mid_attempts = true_mid_attempts + close_attempts * .55

    fg3_pct = _series(data, "FG3_PCT")
    fg3_rate = total_fg3a.div(total_fga.replace(0, np.nan)).fillna(0)
    fta_rate = total_fta.div(total_fga.replace(0, np.nan)).fillna(0)
    ast_to = per_game("AST").div(per_game("TOV").replace(0, np.nan)).fillna(0)

    data["_position_group"] = data.apply(_position_group, axis=1)
    data["_total_shots"] = total_fga
    data["_free_throw_attempts"] = total_fta
    data["_3pt_attempts"] = total_fg3a
    data["_mid_attempts"] = effective_mid_attempts
    data["_finish_attempts"] = effective_finishing_attempts
    data["_mid_pct"] = mid_pct
    data["_rim_pct"] = layup_pct
    data["_layup_contribution"] = layup_contribution
    data["_close_finish_contribution"] = close_finish_contribution
    data["_dunk_finish_contribution"] = dunk_finish_contribution
    data["_true_mid_contribution"] = true_mid_contribution
    data["_close_mid_contribution"] = close_mid_contribution

    data["_metric_3pt"] = _composite(
        (fg3_pct, True), (per_game("FG3M"), True), (fg3_rate, True)
    )
    # These paint metrics intentionally use the requested contribution formulas
    # directly. Percentile normalization happens after the dilution math.
    data["_metric_mid"] = midrange_contribution.div(gp).fillna(0)
    data["_metric_finish"] = finishing_contribution.div(gp).fillna(0)
    data["_metric_play"] = _composite(
        (per_game("AST"), True), (_series(data, "AST_PCT"), True), (ast_to, True)
    )
    data["_metric_defense"] = _composite(
        (per_game("STL"), True), (per_game("BLK"), True),
        (_series(data, "DEF_RATING", default=110.0), False),
    )
    data["_metric_rebound"] = _composite(
        (per_game("REB"), True), (_series(data, "REB_PCT"), True)
    )
    data["_metric_pace"] = _composite(
        (_series(data, "PACE", default=100.0), True),
        (_series(data, "NET_RATING"), True),
    )
    ts_pct = _series(data, "TS_PCT")
    if not ts_pct.any():
        attempts = total_fga + .44 * total_fta
        ts_pct = total("PTS").div(2 * attempts.replace(0, np.nan)).fillna(.55)
    data["_metric_efficiency"] = _composite(
        (ts_pct, True), (_series(data, "PIE"), True),
        (_series(data, "NET_RATING"), True),
    )
    data["_ts_pct"] = ts_pct
    return data, per_game_mode


# Eight identity axes. Dunking is intentionally not an axis: its 0.90-weighted
# contribution belongs only to Finishing, preventing double counting.
ATTRIBUTE_SPECS = (
    ("Finishing", "_metric_finish", "_finish_attempts", 100.0),
    ("Mid-range", "_metric_mid", "_mid_attempts", 75.0),
    ("3PT shooting", "_metric_3pt", "_3pt_attempts", 100.0),
    ("Playmaking", "_metric_play", "_total_shots", 300.0),
    ("Defense", "_metric_defense", "_total_shots", 300.0),
    ("Rebounding", "_metric_rebound", "_total_shots", 300.0),
    ("Efficiency", "_metric_efficiency", "_total_shots", 300.0),
    ("Pace compatibility", "_metric_pace", "_total_shots", 300.0),
)


def _context_factors(
    target: pd.Series,
    position: str,
    opponent_team: str | None,
    season: str,
) -> dict[str, float | str]:
    """Return neutral or cached opponent/pace adjustments."""
    neutral: dict[str, float | str] = {
        "pace_factor": 1.0,
        "opponent_factor": 1.0,
        "opponent_difficulty": 50.0,
        "opponent": str(opponent_team or "League neutral"),
    }
    if not opponent_team or not str(opponent_team).strip():
        return neutral
    teams = load_team_table(season)
    if teams.empty:
        return neutral
    query = str(opponent_team).strip().casefold()
    names = pd.Series("", index=teams.index, dtype=object)
    for column in ("TEAM_ABBREVIATION", "TEAM_NAME"):
        if column in teams:
            names = names + " " + teams[column].astype(str).str.casefold()
    matches = teams[names.str.contains(query, regex=False)]
    if matches.empty:
        return neutral
    opponent = matches.iloc[0]
    league_pace = safe_number(_series(teams, "PACE", default=100.0).mean(), 100.0)
    league_def = safe_number(_series(teams, "DEF_RATING", default=110.0).mean(), 110.0)
    team_abbr = str(target.get("TEAM_ABBREVIATION", "")).upper()
    team_match = teams[
        teams.get("TEAM_ABBREVIATION", pd.Series("", index=teams.index)).astype(str).str.upper()
        == team_abbr
    ]
    team_pace = safe_number(team_match.iloc[0].get("PACE"), league_pace) if not team_match.empty else league_pace
    opponent_pace = safe_number(opponent.get("PACE"), league_pace)
    pace_factor = ((team_pace + opponent_pace) / 2.0) / league_pace if league_pace else 1.0

    defense_signal = _series(teams, "DEF_RATING", default=league_def)
    if position == "Guard":
        defense_signal = defense_signal - _rate_rank(_series(teams, "STL"), True).div(20)
    elif position == "Big":
        defense_signal = defense_signal - (
            _rate_rank(_series(teams, "BLK"), True)
            + _rate_rank(_series(teams, "REB"), True)
        ).div(40)
    else:
        defense_signal = defense_signal - (
            _rate_rank(_series(teams, "STL"), True)
            + _rate_rank(_series(teams, "BLK"), True)
        ).div(40)
    difficulty = _percentile_against(-defense_signal, -safe_number(defense_signal.loc[opponent.name], league_def))
    opponent_factor = 1.0 + (50.0 - difficulty) / 500.0
    return {
        "pace_factor": round(pace_factor, 4),
        "opponent_factor": round(opponent_factor, 4),
        "opponent_difficulty": round(difficulty, 1),
        "opponent": str(opponent.get("TEAM_ABBREVIATION") or opponent.get("TEAM_NAME") or opponent_team),
    }


def calculate_badge_profile(
    row: pd.Series,
    table: pd.DataFrame,
    *,
    display_mode: str = "Adjusted",
    comparison_mode: str = "Entire league",
    filter_minimum_samples: bool = True,
    opponent_team: str | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    """Calculate transparent badge ratings for one selected league row.

    The canonical 2K-style rating is always the requested 60/40 blend of league
    and position percentiles. ``comparison_mode`` changes the reference marker,
    not that canonical rating, so users can inspect either comparison without
    silently changing the player's identity score.
    """
    season = season or latest_season()
    data, _ = _prepare_metrics(table)
    target_id = int(safe_number(row.get("PLAYER_ID"), -1))
    matches = data[pd.to_numeric(data.get("PLAYER_ID", -1), errors="coerce") == target_id]
    target = matches.iloc[0] if not matches.empty else _prepare_metrics(pd.DataFrame([row]))[0].iloc[0]
    position = str(target.get("_position_group", "Wing"))
    position_pool = data[data["_position_group"] == position]
    context = _context_factors(target, position, opponent_team, season)
    attributes: list[dict[str, Any]] = []
    low_sample: list[str] = []

    for name, metric_column, attempts_column, default_minimum in ATTRIBUTE_SPECS:
        attempts = safe_number(target.get(attempts_column), 0.0)
        total_shots = safe_number(target.get("_total_shots"), 0.0)
        dynamic_min = _dynamic_minimum(default_minimum, total_shots)
        confidence = _confidence(attempts, dynamic_min)
        component_confidence: dict[str, float] = {}
        if name == "Finishing":
            ft_attempts = safe_number(target.get("_free_throw_attempts"), 0.0)
            ft_minimum = _dynamic_minimum(DEFAULT_SAMPLE_MINIMUMS["Free throws"], total_shots)
            ft_confidence = _confidence(ft_attempts, ft_minimum)
            component_confidence["free_throw_confidence"] = ft_confidence
            confidence = round(min(confidence, ft_confidence), 4)

        league_pool = data
        same_position_pool = position_pool
        if filter_minimum_samples:
            pool_dynamic = data["_total_shots"].apply(
                lambda value: _dynamic_minimum(default_minimum, safe_number(value))
            )
            qualified = data[attempts_column].ge(pool_dynamic) & pool_dynamic.gt(0)
            filtered = data[qualified]
            filtered_position = filtered[filtered["_position_group"] == position]
            if len(filtered) >= 2:
                league_pool = filtered
            if len(filtered_position) >= 2:
                same_position_pool = filtered_position

        target_metric = safe_number(target.get(metric_column), 50.0)
        league_percentile = _percentile_against(league_pool[metric_column], target_metric)
        position_percentile = _percentile_against(same_position_pool[metric_column], target_metric)
        blended = min(99.0, round(.6 * league_percentile + .4 * position_percentile, 1))
        selected_percentile = league_percentile if comparison_mode == "Entire league" else position_percentile

        context_factor = 1.0
        if name in {"3PT shooting", "Mid-range", "Finishing", "Playmaking", "Efficiency"}:
            context_factor = float(context["pace_factor"]) ** .5 * float(context["opponent_factor"])
        elif name == "Rebounding":
            context_factor = float(context["pace_factor"]) ** .5
        elif name == "Pace compatibility":
            context_factor = float(context["pace_factor"])

        raw_value = round(blended, 1)
        adjusted_value = round(
            min(99.0, max(0.0, blended * confidence * context_factor)), 1
        )
        badge_value = raw_value if display_mode == "Raw" else adjusted_value
        dilution_contributions: dict[str, float] = {}
        if name == "Finishing":
            dilution_contributions = {
                "Layups × efficiency": round(safe_number(target.get("_layup_contribution")), 2),
                "Close shots × 0.60": round(safe_number(target.get("_close_finish_contribution")), 2),
                "Dunks × 0.90": round(safe_number(target.get("_dunk_finish_contribution")), 2),
            }
        elif name == "Mid-range":
            dilution_contributions = {
                "True mid-range × efficiency": round(safe_number(target.get("_true_mid_contribution")), 2),
                "Close shots × 0.55": round(safe_number(target.get("_close_mid_contribution")), 2),
            }

        if confidence < 1.0:
            low_sample.append(name)
        tier = _tier(badge_value)
        attributes.append({
            "attribute": name,
            "icon": BADGE_ICONS[name],
            "league_percentile": league_percentile,
            "position_percentile": position_percentile,
            "rating": blended,
            "raw_value": raw_value,
            "adjusted_value": adjusted_value,
            "selected_percentile": selected_percentile,
            "attempts": round(attempts, 1),
            "dynamic_minimum": round(dynamic_min, 1),
            "sample_confidence": round(confidence, 3),
            "context_factor": round(context_factor, 3),
            "dilution_contributions": dilution_contributions,
            "badge_value": badge_value,
            "tier": tier,
            "color": BADGE_COLORS[tier],
            "rgba": BADGE_RGBA[tier],
            **component_confidence,
        })

    labels: list[str] = []
    for label in ("Elite", "Strong", "Average", "Below average", "Weak"):
        names = [item["attribute"] for item in attributes if _skill_label(item["badge_value"]) == label]
        if names:
            labels.append(f"{label}: {', '.join(names)}")
    warnings: list[str] = []
    if low_sample:
        warnings.append("Low sample size: ratings may be unstable.")
    if len(data) < 20:
        warnings.append("League comparison pool is limited; percentiles use the available cached players.")
    return {
        "position_group": position,
        "display_mode": display_mode,
        "comparison_mode": comparison_mode,
        "filter_minimum_samples": bool(filter_minimum_samples),
        "attributes": attributes,
        "skill_labels": labels,
        "low_sample_attributes": low_sample,
        "context": context,
        "warnings": warnings,
    }


def render_badge_graph(
    player_id: int | str,
    season: str | None = None,
    display_mode: str = "Adjusted",
    comparison_mode: str = "Entire league",
    filter_minimum_samples: bool = True,
    opponent_team: str | None = None,
) -> go.Figure:
    """Return a premium, sample-aware player identity badge wheel."""
    season = season or latest_season()
    display_mode = display_mode if display_mode in DISPLAY_MODES else "Adjusted"
    comparison_mode = comparison_mode if comparison_mode in COMPARISON_MODES else "Entire league"
    player, row, table, provider_warnings = player_row(player_id, season)
    profile = calculate_badge_profile(
        row, table,
        display_mode=display_mode,
        comparison_mode=comparison_mode,
        filter_minimum_samples=filter_minimum_samples,
        opponent_team=opponent_team,
        season=season,
    )
    attributes = profile["attributes"]
    labels = [f"{item['icon']} {item['attribute']}" for item in attributes]
    values = [item["badge_value"] for item in attributes]
    colors = [item["color"] for item in attributes]
    theta = np.linspace(0, 360, len(attributes), endpoint=False)
    team = str(row.get("TEAM_ABBREVIATION", ""))
    primary, secondary = team_palette(team)

    figure = go.Figure()
    figure.add_trace(go.Barpolar(
        r=[99] * len(attributes), theta=theta, width=[32] * len(attributes),
        marker=dict(color="rgba(100,116,139,.10)", line=dict(color="rgba(100,116,139,.12)", width=1)),
        hoverinfo="skip", showlegend=False, name="Rating scale",
    ))
    elite_values = [value if value >= 90 else 0 for value in values]
    figure.add_trace(go.Barpolar(
        r=elite_values, theta=theta, width=[35] * len(attributes),
        marker=dict(color="rgba(34,197,94,.18)", line=dict(color="rgba(34,197,94,.28)", width=7)),
        hoverinfo="skip", showlegend=False, name="Elite glow",
    ))
    custom = np.array([
        [
            item["attribute"], item["tier"], item["league_percentile"],
            item["position_percentile"], item["rating"], item["attempts"],
            item["dynamic_minimum"], item["sample_confidence"] * 100,
            item["context_factor"], item["raw_value"], item["adjusted_value"],
            "; ".join(
                f"{label}: {value:.2f}"
                for label, value in item["dilution_contributions"].items()
            ) or "No overlapping close-shot contribution",
        ]
        for item in attributes
    ], dtype=object)
    figure.add_trace(go.Barpolar(
        r=values, theta=theta, width=[29] * len(attributes),
        marker=dict(color=colors, line=dict(color=primary, width=1.8)),
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "Badge value %{r:.1f}/99<br>Canonical rating %{customdata[4]:.1f}<br>"
            "Raw %{customdata[9]:.1f} · Adjusted %{customdata[10]:.1f}<br>"
            "League %{customdata[2]:.1f}th percentile · Position %{customdata[3]:.1f}th<br>"
            "Attempts %{customdata[5]:.0f} · Dynamic minimum %{customdata[6]:.0f}<br>"
            "Sample confidence %{customdata[7]:.0f}% · Context ×%{customdata[8]:.3f}<br>"
            "Dilution: %{customdata[11]}"
            "<extra></extra>"
        ),
        opacity=.94, showlegend=False, name="Badge value",
    ))
    selected = [item["selected_percentile"] for item in attributes]
    figure.add_trace(go.Scatterpolar(
        r=selected, theta=theta, mode="markers",
        marker=dict(size=9, symbol="diamond", color=secondary, line=dict(color=primary, width=2)),
        customdata=labels,
        hovertemplate="%{customdata}<br>Selected comparison %{r:.1f}th percentile<extra></extra>",
        showlegend=False, name="Comparison percentile",
    ))
    circle_theta = np.linspace(0, 360, 97)
    figure.add_trace(go.Scatterpolar(
        r=[103] * len(circle_theta), theta=circle_theta, mode="lines",
        line=dict(color=primary, width=6), hoverinfo="skip", showlegend=False,
        name="Team ring",
    ))
    figure.add_trace(go.Scatterpolar(
        r=[108] * len(attributes), theta=theta, mode="markers+text",
        marker=dict(size=13, color=colors, line=dict(color=primary, width=1.5)),
        text=labels, textposition="top center",
        textfont=dict(color=colors, size=12),
        hoverinfo="skip", showlegend=False, name="Skill labels",
    ))
    warnings = list(dict.fromkeys(provider_warnings + profile["warnings"]))
    figure.update_layout(
        title=f"{player['full_name']} · {display_mode.lower()} identity wheel",
        template="plotly_white", height=700,
        margin=dict(l=115, r=115, t=95, b=85),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,.40)",
            radialaxis=dict(
                range=[0, 114], tickvals=[60, 70, 80, 90, 99],
                ticktext=["60", "70", "80", "90", "99"],
                gridcolor="rgba(100,116,139,.15)", tickfont=dict(color="#64748B", size=10),
            ),
            angularaxis=dict(showticklabels=False, gridcolor="rgba(100,116,139,.10)", linecolor=primary, linewidth=2),
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#10213A"),
        annotations=[dict(
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
            text=f"<b>{profile['position_group']}</b><br>{comparison_mode}",
            font=dict(color=primary, size=14), align="center",
        )],
        meta={
            "player": player["full_name"], "team": team, "season": season, "view": "Wheel",
            "ratings": {item["attribute"]: item["badge_value"] for item in attributes},
            "canonical_ratings": {item["attribute"]: item["rating"] for item in attributes},
            "attributes": attributes, "skill_labels": profile["skill_labels"],
            "position_group": profile["position_group"], "display_mode": display_mode,
            "comparison_mode": comparison_mode,
            "filter_minimum_samples": bool(filter_minimum_samples),
            "context": profile["context"], "warnings": warnings,
        },
    )
    return figure

def _rgba_alpha(value: str, alpha: float) -> str:
    """Return one tier color with a controlled transparent alpha."""
    numbers = [part.strip() for part in value[value.find("(") + 1:value.rfind(")")].split(",")]
    if len(numbers) < 3:
        return f"rgba(91, 108, 255, {alpha:.2f})"
    return f"rgba({numbers[0]}, {numbers[1]}, {numbers[2]}, {alpha:.2f})"


def _darken_hex(value: str, factor: float = .65) -> str:
    color = str(value).lstrip("#")
    if len(color) != 6:
        return "#334155"
    channels = [int(color[index:index + 2], 16) for index in (0, 2, 4)]
    return "#" + "".join(f"{max(0, min(255, round(channel * factor))):02x}" for channel in channels)


def render_spider_badge_graph(
    player_id: int | str,
    mode: str = "Adjusted",
    comparison_mode: str = "Entire league",
    *,
    season: str | None = None,
    filter_minimum_samples: bool = True,
    opponent_team: str | None = None,
    gradient_fill: bool = True,
) -> go.Figure:
    """Return the primary transparent radar/spider badge visualization.

    ``mode`` intentionally precedes ``season`` so the public contract matches
    ``render_spider_badge_graph(player_id, mode, comparison_mode)``. Season and
    matchup inputs remain keyword-only to prevent ambiguous positional calls.
    """
    mode = mode if mode in DISPLAY_MODES else "Adjusted"
    comparison_mode = comparison_mode if comparison_mode in COMPARISON_MODES else "Entire league"
    wheel = render_badge_graph(
        player_id, season or latest_season(), mode, comparison_mode,
        filter_minimum_samples, opponent_team,
    )
    meta = dict(wheel.layout.meta or {})
    attributes = list(meta.get("attributes", []))
    short_labels = {
        "3PT shooting": "3PT", "Pace compatibility": "Pace fit",
    }
    labels = [
        f"{item['icon']} {short_labels.get(item['attribute'], item['attribute'])}"
        for item in attributes
    ]
    values = [safe_number(item.get("badge_value")) for item in attributes]
    selected = [safe_number(item.get("selected_percentile")) for item in attributes]
    colors = [str(item.get("color", BADGE_COLORS["Weak"])) for item in attributes]
    if not attributes:
        return wheel

    theta = np.linspace(0, 360, len(attributes), endpoint=False)
    theta_closed = list(theta) + [float(theta[0])]
    values_closed = values + [values[0]]
    selected_closed = selected + [selected[0]]
    overall_tier = _tier(float(np.mean(values)))
    base_rgba = BADGE_RGBA[overall_tier]
    outline = _darken_hex(BADGE_COLORS[overall_tier])
    primary, secondary = team_palette(str(meta.get("team", "")))
    figure = go.Figure()

    if gradient_fill:
        # Draw lighter full geometry first, then increasingly dark translucent
        # inner layers. This creates depth without animation or canvas effects.
        layers = ((1.0, .40), (.68, .48), (.36, .58))
    else:
        layers = ((1.0, .50),)
    for index, (scale, alpha) in enumerate(layers):
        layer_values = [value * scale for value in values_closed]
        figure.add_trace(go.Scatterpolar(
            r=layer_values,
            theta=theta_closed,
            mode="lines",
            fill="toself",
            fillcolor=_rgba_alpha(base_rgba, alpha),
            line=dict(
                color=outline,
                width=2 if index == 0 else 1,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Badge profile" if index == 0 else "Depth layer",
        ))

    comparison_label = "League percentile" if comparison_mode == "Entire league" else "Position percentile"
    figure.add_trace(go.Scatterpolar(
        r=selected_closed,
        theta=theta_closed,
        mode="lines",
        line=dict(color=secondary, width=1.5, dash="dot"),
        hovertemplate=f"{comparison_label}: %{{r:.1f}}<extra></extra>",
        showlegend=False,
        name=comparison_label,
    ))

    custom = np.array([
        [
            item["attribute"], item["tier"], item["rating"],
            item["selected_percentile"], item["sample_confidence"] * 100,
            item["raw_value"], item["adjusted_value"], item["badge_value"],
            "; ".join(
                f"{label}: {value:.2f}"
                for label, value in item.get("dilution_contributions", {}).items()
            ) or "No overlapping close-shot contribution",
        ]
        for item in attributes
    ], dtype=object)
    figure.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        mode="lines+markers",
        line=dict(color=outline, width=2),
        marker=dict(size=11, color=colors, line=dict(color=primary, width=1.5)),
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "Displayed rating %{customdata[7]:.1f}/99<br>"
            "Canonical rating %{customdata[2]:.1f} · Selected percentile %{customdata[3]:.1f}<br>"
            "Sample confidence %{customdata[4]:.0f}%<br>"
            "Raw %{customdata[5]:.1f} · Adjusted %{customdata[6]:.1f}<br>"
            "Dilution: %{customdata[8]}<extra></extra>"
        ),
        showlegend=False,
        name="Skill rating",
    ))

    elite_values = [value if value >= 90 else None for value in values]
    figure.add_trace(go.Scatterpolar(
        r=elite_values,
        theta=theta,
        mode="markers",
        marker=dict(
            size=23, color="rgba(0, 200, 0, 0.16)",
            line=dict(color="rgba(0, 200, 0, 0.34)", width=5),
        ),
        hoverinfo="skip", showlegend=False, name="Elite glow",
    ))
    ring_theta = np.linspace(0, 360, 97)
    figure.add_trace(go.Scatterpolar(
        r=[103] * len(ring_theta), theta=ring_theta, mode="lines",
        line=dict(color=primary, width=5), hoverinfo="skip",
        showlegend=False, name="Team ring",
    ))
    figure.add_trace(go.Scatterpolar(
        r=[106] * len(attributes), theta=theta, mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(color=primary, width=1.2)),
        text=labels,
        textposition=[
            "middle right", "top right", "top center", "top left",
            "middle left", "bottom left", "bottom center", "bottom right",
        ],
        textfont=dict(color=colors, size=11),
        hoverinfo="skip", showlegend=False, name="Skill labels",
    ))

    meta["view"] = "Spider"
    meta["gradient_fill"] = bool(gradient_fill)
    figure.update_layout(
        title=f"{meta.get('player', 'Player')} · {mode.lower()} badge spider",
        template="plotly_white", height=710,
        margin=dict(l=120, r=120, t=90, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,.24)",
            radialaxis=dict(
                range=[0, 114], tickvals=[20, 40, 60, 70, 80, 90, 99],
                gridcolor="rgba(100,116,139,.16)",
                tickfont=dict(color="#64748B", size=9),
            ),
            angularaxis=dict(
                showticklabels=False,
                gridcolor="rgba(100,116,139,.14)",
                linecolor=primary, linewidth=1,
            ),
        ),
        annotations=[dict(
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
            text=f"<b>{meta.get('position_group', 'NBA')}</b><br>{comparison_mode}",
            font=dict(color=primary, size=13), align="center",
        )],
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#10213A"),
        showlegend=False,
        meta=meta,
    )
    return figure