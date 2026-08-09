"""Prepare robust slate-wide prop edge matrices for any UI."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from modules.data_quality import safe_dict, safe_get, safe_list, safe_number

EDGE_CATEGORIES = ("points", "rebounds", "assists", "pra", "fantasy", "minutes")
SORT_FIELDS = {"edge": "edge", "reliability": "reliability", "confidence": "confidence"}


def normalize_prop_category(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    aliases = {
        "pts": "points",
        "point": "points",
        "reb": "rebounds",
        "rebs": "rebounds",
        "ast": "assists",
        "asts": "assists",
        "points rebounds assists": "pra",
        "fantasy points": "fantasy",
        "fantasy score": "fantasy",
        "mins": "minutes",
    }
    return aliases.get(text, text)


def _driver_score(row: dict[str, Any], label: str) -> float | None:
    pattern = re.compile(rf"{label}[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.I)
    for driver in safe_list(safe_get(row, "key_drivers")):
        match = pattern.search(str(driver))
        if match:
            return safe_number(match.group(1))
    return None


def _confidence(row: dict[str, Any], projection: float) -> float:
    direct = safe_get(row, "confidence")
    if direct is not None:
        return max(0.0, min(100.0, safe_number(direct)))
    driver = _driver_score(row, "confidence")
    if driver is not None:
        return max(0.0, min(100.0, driver))
    low = safe_get(row, "confidence_low")
    high = safe_get(row, "confidence_high")
    if low is not None and high is not None:
        width = max(0.0, safe_number(high) - safe_number(low))
        return max(0.0, min(100.0, 100.0 - width / max(abs(projection), 1.0) * 100.0))
    return 50.0


def _reliability(row: dict[str, Any]) -> float:
    direct = safe_get(row, "reliability", safe_get(row, "reliability_score"))
    if direct is not None:
        return max(0.0, min(100.0, safe_number(direct)))
    driver = _driver_score(row, "reliability")
    return max(0.0, min(100.0, driver if driver is not None else 50.0))


def normalize_edge_props(props: Any) -> list[dict[str, Any]]:
    """Normalize mixed prop schemas while preserving verified sportsbook names."""
    clean: list[dict[str, Any]] = []
    for raw in safe_list(props):
        row = safe_dict(raw)
        player = str(
            safe_get(row, "player", safe_get(row, "player_name", ""))
        ).strip()
        category = normalize_prop_category(safe_get(row, "category"))
        if not player or category not in EDGE_CATEGORIES:
            continue
        projection = safe_number(
            safe_get(
                row,
                "minutes_adjusted_projection",
                safe_get(row, "projection"),
            )
        )
        line = safe_number(
            safe_get(row, "sportsbook_line", safe_get(row, "line"))
        )
        raw_edge = safe_get(row, "edge")
        edge = safe_number(raw_edge, projection - line)
        if raw_edge is None:
            edge = projection - line
        clean.append(
            {
                "player": player,
                "team": str(safe_get(row, "team", "")),
                "category": category,
                "projection": round(projection, 2),
                "sportsbook_line": round(line, 2),
                "edge": round(edge, 2),
                "reliability": round(_reliability(row), 1),
                "confidence": round(_confidence(row, projection), 1),
                "sportsbook": str(safe_get(row, "sportsbook", safe_get(row, "best_sportsbook", "Unknown"))),
                "lean": "over" if edge >= 0 else "under",
            }
        )
    return clean


def prepare_edge_heatmap(
    props: Any,
    categories: list[str] | None = None,
    sort_by: str = "edge",
) -> dict[str, Any]:
    """Return filtered records and a player-by-category edge matrix."""
    requested = [
        normalize_prop_category(category)
        for category in (categories or list(EDGE_CATEGORIES))
    ]
    requested = [category for category in requested if category in EDGE_CATEGORIES]
    sort_key = SORT_FIELDS.get(str(sort_by).lower(), "edge")
    records = [
        row for row in normalize_edge_props(props)
        if row["category"] in requested
    ]
    records.sort(key=lambda row: safe_number(row.get(sort_key)), reverse=True)

    if not records:
        return {
            "records": [],
            "matrix": {},
            "categories": requested,
            "sort_by": sort_key,
            "edge_range": {"min": 0.0, "max": 0.0},
            "warnings": ["No props matched the selected edge-heatmap filters."],
        }

    table = pd.DataFrame(records)
    # When several books quote the same prop, keep the row with the strongest
    # absolute edge so the heatmap remains one readable cell per prop.
    table["absolute_edge"] = table["edge"].abs()
    strongest = (
        table.sort_values("absolute_edge", ascending=False)
        .drop_duplicates(["player", "category"])
        .copy()
    )
    order = (
        table.groupby("player")[sort_key]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    matrix = strongest.pivot(index="player", columns="category", values="edge")
    matrix = matrix.reindex(index=order)
    matrix = matrix.reindex(columns=[c for c in requested if c in matrix.columns])
    finite = table["edge"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "records": records,
        "matrix": {
            player: {
                category: (
                    None if pd.isna(value) else round(safe_number(value), 2)
                )
                for category, value in values.items()
            }
            for player, values in matrix.to_dict(orient="index").items()
        },
        "categories": list(matrix.columns),
        "sort_by": sort_key,
        "edge_range": {
            "min": round(safe_number(finite.min()), 2) if not finite.empty else 0.0,
            "max": round(safe_number(finite.max()), 2) if not finite.empty else 0.0,
        },
        "warnings": [],
    }


def build_edge_heatmap(
    props: Any,
    categories: list[str] | None = None,
    sort_by: str = "edge",
) -> dict[str, Any]:
    """Backward-friendly name for callers that think in visualization terms."""
    return prepare_edge_heatmap(props, categories, sort_by)