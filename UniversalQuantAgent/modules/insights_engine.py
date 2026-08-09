"""Short, explainable insights for the Universal Quant Agent interface.

The functions in this module deliberately return plain strings instead of UI
objects.  Streamlit, a future API, and automated reports can therefore share
the same explanations without coupling analytics code to presentation code.
Every function accepts partial dictionaries and quietly skips unavailable
signals, which is important when a public NBA provider is offline.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Iterable

from modules.data_quality import safe_dict, safe_get, safe_list, safe_number


def _number(value: Any) -> float | None:
    """Return a finite number, preserving the difference between zero/missing."""
    if value in (None, "", "—"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _first(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(safe_get(mapping, key))
        if value is not None:
            return value
    return None


def _window(profile: dict[str, Any], name: str) -> dict[str, Any]:
    aliases = {
        "season": ("season_avg", "season_averages", "season_average"),
        "last10": ("last10_avg", "recent_10_game_averages", "last_10"),
        "last5": ("last5_avg", "recent_5_game_averages", "last_5"),
    }
    for key in aliases[name]:
        result = safe_dict(safe_get(profile, key))
        if result:
            return result
    summary = safe_dict(safe_get(profile, "feature_summary"))
    for key in aliases[name]:
        result = safe_dict(safe_get(summary, key))
        if result:
            return result
    return {}


def _trend_sentence(label: str, season: float | None, last10: float | None,
                    last5: float | None, suffix: str = "") -> str | None:
    recent = last5 if last5 is not None else last10
    if season is None or recent is None:
        return None
    delta = recent - season
    threshold = max(abs(season) * .06, .7 if label != "usage" else 1.0)
    direction = "above" if delta > 0 else "below"
    window = "last five" if last5 is not None else "last 10"
    if abs(delta) < threshold:
        return f"{label.title()} is steady: the {window} ({recent:.1f}{suffix}) is close to the season level ({season:.1f}{suffix})."
    return f"{label.title()} is trending {direction} the season level, moving from {season:.1f}{suffix} to {recent:.1f}{suffix} over the {window}."


def _append(insights: list[str], value: str | None) -> None:
    if value and value not in insights:
        insights.append(value)


def generate_player_insights(
    player: dict[str, Any] | Any,
    matchup: dict[str, Any] | Any = None,
) -> list[str]:
    """Generate compact player, projection, and matchup observations."""
    root = safe_dict(player)
    fusion = safe_dict(safe_get(root, "fusion")) or root
    base = safe_dict(safe_get(fusion, "base_projection")) or root
    context = safe_dict(safe_get(fusion, "context"))
    inputs = safe_dict(safe_get(fusion, "input_breakdown"))
    advanced = safe_dict(safe_get(root, "advanced"))
    season, last10, last5 = (
        _window(base, "season"), _window(base, "last10"), _window(base, "last5")
    )
    insights: list[str] = []

    usage = safe_dict(safe_get(inputs, "usage"))
    season_usage = _first(usage, "season_average") or _first(advanced, "usage_pct", "usg_pct")
    last10_usage = _first(usage, "last_10")
    last5_usage = _first(usage, "last_5")
    if last10_usage is None or last5_usage is None:
        trend_rows = [safe_dict(row) for row in safe_list(safe_get(root, "trend_series"))]
        usage_values = [value for row in trend_rows if (value := _first(row, "usage_pct", "usg_pct")) is not None]
        last10_usage = last10_usage if last10_usage is not None else (mean(usage_values[-10:]) if usage_values else None)
        last5_usage = last5_usage if last5_usage is not None else (mean(usage_values[-5:]) if usage_values else None)
    _append(insights, _trend_sentence("usage", season_usage, last10_usage, last5_usage, "%"))

    season_minutes = _first(safe_dict(safe_get(inputs, "minutes")), "season_average") or _first(season, "minutes", "mpg", "min")
    last10_minutes = _first(safe_dict(safe_get(inputs, "minutes")), "last_10") or _first(last10, "minutes", "mpg", "min")
    last5_minutes = _first(safe_dict(safe_get(inputs, "minutes")), "last_5") or _first(last5, "minutes", "mpg", "min")
    _append(insights, _trend_sentence("minutes", season_minutes, last10_minutes, last5_minutes))
    role_stability = _first(context, "role_stability")
    if role_stability is not None:
        label = "stable" if role_stability >= 75 else "variable" if role_stability < 50 else "moderately stable"
        _append(insights, f"The rotation profile is {label} ({role_stability:.0f}/100), which directly affects projection confidence.")

    efficiency = safe_dict(safe_get(inputs, "efficiency"))
    season_ts = _first(efficiency, "season_average_ts_pct") or _first(advanced, "ts_pct")
    last10_ts = _first(efficiency, "last_10_ts_pct")
    last5_ts = _first(efficiency, "last_5_ts_pct")
    _append(insights, _trend_sentence("true shooting", season_ts, last10_ts, last5_ts, "%"))
    per_estimate = _first(advanced, "per_estimate")
    if per_estimate is not None:
        tier = "high-impact" if per_estimate >= 22 else "solid" if per_estimate >= 15 else "below league-average"
        _append(insights, f"The {per_estimate:.1f} PER estimate indicates a {tier} per-minute production profile.")

    components = safe_dict(safe_get(context, "components"))
    pace = safe_dict(safe_get(components, "pace"))
    pace_value = _first(pace, "pace_projection", "value")
    pace_factor = _first(safe_dict(safe_get(inputs, "context")), "pace_factor")
    if pace_value is not None:
        impact = "adds possessions" if (pace_factor or pace_value / 100) > 1.005 else "reduces possessions" if (pace_factor or pace_value / 100) < .995 else "is close to neutral"
        _append(insights, f"The projected {pace_value:.1f} pace {impact} for this matchup.")

    matchup_data = safe_dict(matchup) or safe_dict(safe_get(components, "matchup"))
    difficulty = _first(matchup_data, "difficulty_score", "value")
    if difficulty is not None:
        label = "difficult" if difficulty >= 65 else "favorable" if difficulty <= 40 else "balanced"
        _append(insights, f"Opponent difficulty is {label} at {difficulty:.0f}/100, so the matchup should be treated as a contextual adjustment rather than a standalone signal.")

    similarity = safe_dict(safe_get(root, "similarity"))
    matches = safe_list(safe_get(similarity, "similar_players"))
    if matches:
        top = safe_dict(matches[0])
        _append(insights, f"The closest role-and-production comparison is {safe_get(top, 'player', 'a comparable player')} at {safe_number(safe_get(top, 'similarity_score')):.0f}% similarity.")

    edge = _first(matchup_data, "edge")
    line = _first(matchup_data, "sportsbook_line", "line")
    if edge is not None:
        lean = "over" if edge >= 0 else "under"
        line_text = f" against a {line:.1f} line" if line is not None else ""
        _append(insights, f"The model leans {lean}{line_text} because the projection differs by {abs(edge):.1f}; reliability should determine how much weight to place on that edge.")
    return insights[:7]


def generate_slate_insights(slate: dict[str, Any] | Any) -> list[str]:
    """Summarize games, projection context, and slate-wide prop edges."""
    data = safe_dict(slate)
    games = [safe_dict(row) for row in safe_list(safe_get(data, "games"))]
    props = [safe_dict(row) for row in safe_list(safe_get(data, "player_props"))]
    projections = [safe_dict(row) for row in safe_list(safe_get(data, "projections"))]
    insights: list[str] = []
    if games:
        _append(insights, f"The slate contains {len(games)} game{'s' if len(games) != 1 else ''} and {len(props)} available prop line{'s' if len(props) != 1 else ''}.")
    pace_rows = [(safe_get(row, "player", "Player"), _first(row, "pace")) for row in projections]
    pace_rows = [(name, value) for name, value in pace_rows if value is not None]
    if pace_rows:
        name, value = max(pace_rows, key=lambda item: item[1])
        _append(insights, f"{name} has the fastest projected game environment at {value:.1f} possessions, creating the slate's strongest pace tailwind.")
    difficulty_rows = [(safe_get(row, "player", "Player"), _first(row, "matchup_difficulty")) for row in projections]
    difficulty_rows = [(name, value) for name, value in difficulty_rows if value is not None]
    if difficulty_rows:
        name, value = max(difficulty_rows, key=lambda item: item[1])
        _append(insights, f"{name} faces the highest modeled opponent difficulty ({value:.0f}/100), a useful risk flag when projections are otherwise close.")
    edge_rows = [row for row in props if _first(row, "edge") is not None]
    if edge_rows:
        strongest = max(edge_rows, key=lambda row: abs(_first(row, "edge") or 0.0))
        edge = _first(strongest, "edge") or 0.0
        _append(insights, f"The largest listed edge is {edge:+.1f} for {safe_get(strongest, 'player', 'a player')} {safe_get(strongest, 'category', '')}; verify reliability before using it.")
    if not insights:
        insights.append("Slate context is limited because games, props, or projections are still loading; available sections remain safe to review.")
    return insights[:5]


def generate_similarity_insights(player: dict[str, Any] | Any) -> list[str]:
    """Explain the strongest player-comparison signals in plain language."""
    data = safe_dict(player)
    target = safe_dict(safe_get(data, "target"))
    matches = [safe_dict(row) for row in safe_list(safe_get(data, "similar_players"))]
    if not matches:
        return ["No qualified comparison profiles were available, so similarity context is not being inferred from incomplete data."]
    first = matches[0]
    insights = [
        f"{safe_get(first, 'player', 'The top match')} is the closest comparison for {safe_get(target, 'player', 'this player')} at {safe_number(safe_get(first, 'similarity_score')):.0f}% similarity."
    ]
    dimensions = safe_dict(safe_get(first, "dimension_scores"))
    ranked = sorted(((str(key), safe_number(value)) for key, value in dimensions.items()), key=lambda item: item[1], reverse=True)
    if ranked:
        strong = ", ".join(name.replace("_", " ") for name, _ in ranked[:2])
        weak = ranked[-1][0].replace("_", " ")
        insights.append(f"The strongest shared dimensions are {strong}; {weak} is the clearest point of separation.")
    return insights[:3]


def generate_correlation_insights(player_or_team: dict[str, Any] | Any) -> list[str]:
    """Surface the strongest positive and negative off-diagonal relationships."""
    data = safe_dict(player_or_team)
    matrix = safe_dict(safe_get(data, "correlation_matrix"))
    pairs: list[tuple[float, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for first, row in matrix.items():
        for second, raw in safe_dict(row).items():
            if first == second:
                continue
            pair = tuple(sorted((str(first), str(second))))
            value = _number(raw)
            if pair in seen or value is None:
                continue
            seen.add(pair)
            pairs.append((value, pair[0], pair[1]))
    if not pairs:
        return ["No stable multi-stat relationships are available yet; the heatmap will expand when more variable observations are loaded."]
    strongest_positive = max(pairs, key=lambda item: item[0])
    strongest_negative = min(pairs, key=lambda item: item[0])
    insights = [
        f"The strongest positive relationship is {strongest_positive[1].replace('_', ' ')} with {strongest_positive[2].replace('_', ' ')} (r={strongest_positive[0]:.2f})."
    ]
    if strongest_negative[0] < -.15:
        insights.append(f"The clearest inverse relationship is {strongest_negative[1].replace('_', ' ')} versus {strongest_negative[2].replace('_', ' ')} (r={strongest_negative[0]:.2f}).")
    else:
        insights.append("No meaningful inverse relationship stands out in the selected sample.")
    return insights


def generate_edge_insights(props: Iterable[dict[str, Any]] | Any) -> list[str]:
    """Explain a normalized set of projection-versus-line edges."""
    rows = [safe_dict(row) for row in props] if isinstance(props, (list, tuple)) else []
    rows = [row for row in rows if _first(row, "edge") is not None]
    if not rows:
        return ["No comparable projection-versus-line edges are available for the current filters."]
    best = max(rows, key=lambda row: _first(row, "edge") or 0.0)
    weakest = min(rows, key=lambda row: _first(row, "edge") or 0.0)
    reliable = [safe_number(safe_get(row, "reliability")) for row in rows if safe_get(row, "reliability") is not None]
    insights = [
        f"The strongest over edge is {safe_number(safe_get(best, 'edge')):+.1f} for {safe_get(best, 'player', 'a player')} {safe_get(best, 'category', '')}; the strongest under edge is {safe_number(safe_get(weakest, 'edge')):+.1f}."
    ]
    if reliable:
        insights.append(f"Average reliability is {mean(reliable):.0f}/100 across these edges, so prioritize the green cells with above-average reliability.")
    return insights



def generate_badge_insights(profile: dict[str, Any] | Any) -> list[str]:
    """Describe a badge wheel using the exact values visible in the chart."""
    data = safe_dict(profile)
    attributes = [safe_dict(item) for item in safe_list(safe_get(data, "attributes"))]
    attributes = [item for item in attributes if safe_get(item, "attribute")]
    if not attributes:
        return ["Badge identity is unavailable until a qualified player profile is loaded."]

    ranked = sorted(
        attributes,
        key=lambda item: safe_number(safe_get(item, "badge_value")),
        reverse=True,
    )
    strengths = [item for item in ranked if safe_number(safe_get(item, "badge_value")) >= 80][:3]
    weaknesses = [item for item in reversed(ranked) if safe_number(safe_get(item, "badge_value")) < 70][:2]
    player = str(safe_get(data, "player", "This player"))
    mode = str(safe_get(data, "display_mode", "Adjusted")).lower()
    insights: list[str] = []

    if strengths:
        strength_text = ", ".join(
            f"{safe_get(item, 'attribute')} ({safe_number(safe_get(item, 'badge_value')):.0f})"
            for item in strengths
        )
        sentence = f"{player}'s clearest {mode} strengths are {strength_text}."
    else:
        best = ranked[0]
        sentence = (
            f"{player}'s strongest {mode} attribute is {safe_get(best, 'attribute')} "
            f"at {safe_number(safe_get(best, 'badge_value')):.0f}, with no skill currently clearing the strong tier."
        )
    if weaknesses:
        weak_text = ", ".join(
            f"{safe_get(item, 'attribute')} ({safe_number(safe_get(item, 'badge_value')):.0f})"
            for item in weaknesses
        )
        sentence += f" The weakest areas are {weak_text}."
    insights.append(sentence)

    low_sample = [
        str(safe_get(item, "attribute")) for item in attributes
        if safe_number(safe_get(item, "sample_confidence"), 1.0) < 1.0
    ]
    if low_sample:
        insights.append(
            "Low sample size affects " + ", ".join(low_sample)
            + "; adjusted ratings discount those skills until the dynamic thresholds are met."
        )
    return insights[:2]