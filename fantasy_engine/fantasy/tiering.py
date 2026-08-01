"""Player tiering (by position, via clustering) and cheat sheet export.

Usage::

    from fantasy.tiering import tier_players, generate_printable_cheatsheet, export_cheatsheet_csv

    tiered = tier_players(ranked_players, max_tiers=5)
    print(generate_printable_cheatsheet(tiered))
    export_cheatsheet_csv(tiered, "cheatsheet.csv")

Clustering is a small, dependency-free k-means over each position's
(median points, volatility) pairs -- deliberately not scikit-learn, since
this package's only numeric dependency is numpy (see ``pyproject.toml``).
Initialization is deterministic (evenly spaced across the value-sorted
players) rather than random, so tiering is fully reproducible run to run.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np

DEFAULT_MAX_TIERS = 5


def _value_and_volatility(player: dict[str, Any]) -> tuple[float, float]:
    value = player.get("median")
    if value is None:
        value = player.get("points", player.get("vor", 0.0))
    volatility = player.get("volatility", 0.0) or 0.0
    return float(value), float(volatility)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def _seed_centroids(points: np.ndarray, k: int) -> np.ndarray:
    """Deterministically seed k centroids via farthest-first traversal.

    Spacing seeds by the primary dimension's value range works when that
    dimension is the one separating players, but a cluster can just as
    easily be separated on volatility instead (two players with the same
    median but very different risk profiles). Farthest-first seeding -- pick
    the highest-value point, then repeatedly pick whichever remaining point
    is farthest from every centroid chosen so far -- spreads seeds across
    whichever dimensions actually have separation, without any randomness.
    """
    chosen = [int(np.argmax(points[:, 0]))]
    while len(chosen) < k:
        distances_to_nearest_centroid = np.min(np.linalg.norm(points[:, None, :] - points[None, chosen, :], axis=2), axis=1)
        distances_to_nearest_centroid[chosen] = -1.0  # never re-pick an existing centroid
        chosen.append(int(np.argmax(distances_to_nearest_centroid)))
    return points[chosen].copy()


def _kmeans_labels(points: np.ndarray, k: int, iterations: int = 100) -> np.ndarray:
    """Deterministic k-means over ``points`` (n_samples, n_features)."""
    n = len(points)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)

    centroids = _seed_centroids(points, k)
    labels = np.full(n, -1, dtype=int)  # sentinel: no iteration can legitimately produce -1
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return labels


def _cluster_position_group(players: list[dict[str, Any]], max_tiers: int) -> None:
    """Mutate each player dict in-place, adding a 1-indexed 'tier' key.

    ``players`` is always non-empty: it's a value from ``tier_players``'s
    ``by_position`` grouping, which never creates an entry with zero players.
    """
    if len(players) == 1:
        players[0]["tier"] = 1
        return

    raw = np.array([_value_and_volatility(p) for p in players])
    standardized = _standardize(raw)
    k = min(max_tiers, len(players))
    labels = _kmeans_labels(standardized, k)

    cluster_mean_value = {cluster: raw[labels == cluster, 0].mean() for cluster in set(labels)}
    ranked_clusters = sorted(cluster_mean_value, key=lambda cluster: cluster_mean_value[cluster], reverse=True)
    tier_by_cluster = {cluster: tier for tier, cluster in enumerate(ranked_clusters, start=1)}

    for player, label in zip(players, labels, strict=True):
        player["tier"] = tier_by_cluster[label]


def tier_players(players: list[dict[str, Any]], max_tiers: int = DEFAULT_MAX_TIERS) -> list[dict[str, Any]]:
    """Annotate each player with a position-relative 'tier' (1 = best).

    Tiers are clustered independently per position on (median points,
    volatility) so a WR6 and an RB6 are never compared against each other.
    Input order is preserved; each returned dict is a shallow copy of the
    input plus ``tier`` and ``tier_label``.
    """
    if max_tiers < 1:
        raise ValueError("max_tiers must be >= 1")

    by_position: dict[str, list[dict[str, Any]]] = {}
    working_copies = [dict(player) for player in players]
    for player in working_copies:
        position = str(player.get("position", "")).strip().upper()
        by_position.setdefault(position, []).append(player)

    for group in by_position.values():
        _cluster_position_group(group, max_tiers)

    for player in working_copies:
        player["tier_label"] = f"Tier {player['tier']}"
    return working_copies


def generate_printable_cheatsheet(tiered_players: list[dict[str, Any]]) -> str:
    """Render a plain-text cheat sheet grouped by position, then tier, then rank."""
    by_position: dict[str, list[dict[str, Any]]] = {}
    for player in tiered_players:
        position = str(player.get("position", "")).strip().upper()
        by_position.setdefault(position, []).append(player)

    lines: list[str] = []
    for position in sorted(by_position):
        lines.append(f"=== {position} ===")
        group = sorted(by_position[position], key=lambda p: (p.get("tier", 1), -_value_and_volatility(p)[0]))
        current_tier = None
        for player in group:
            if player.get("tier") != current_tier:
                current_tier = player.get("tier")
                lines.append(f"-- Tier {current_tier} --")
            value, _volatility = _value_and_volatility(player)
            team = f" ({player['team']})" if player.get("team") else ""
            lines.append(f"  {player.get('name', 'Unknown')}{team} - {value:.1f} pts")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


CSV_COLUMNS = [
    "tier",
    "position",
    "position_rank",
    "overall_rank",
    "name",
    "team",
    "points",
    "vor",
    "median",
    "floor",
    "ceiling",
    "volatility",
    "rationale",
]


def export_cheatsheet_csv(tiered_players: list[dict[str, Any]], file_path: str | None = None) -> str:
    """Return CSV text for the tiered cheat sheet, optionally writing it to ``file_path``."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for player in tiered_players:
        writer.writerow({column: player.get(column, "") for column in CSV_COLUMNS})
    csv_text = buffer.getvalue()
    if file_path:
        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            handle.write(csv_text)
    return csv_text
