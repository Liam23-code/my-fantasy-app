"""Deterministic player-similarity and archetype clustering utilities.

The fantasy application consumes player records from several sources.  Some
records are dictionaries, while API callers often pass Pydantic models.  This
module intentionally accepts both, normalizes a compact set of football
features, and never mutates the supplied objects.

No machine-learning dependency is required.  Nearest-player comparisons use
cosine similarity over feature-wise scaled vectors, and archetypes use a
small deterministic k-means implementation with farthest-first seeding.  The
same input therefore produces the same clusters on every platform and run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_SIMILARITY_FEATURES: tuple[str, ...] = (
    "projection",
    "points_per_game",
    "usage_rate",
    "efficiency_score",
    "targets",
    "carries",
    "receptions",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    "total_touchdowns",
    "volatility",
)

_NESTED_STAT_KEYS = ("stats", "season_stats", "metrics", "projection_data")
_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "projection": (
        "projection",
        "projected_points",
        "expected_fantasy_points",
        "fantasy_points",
        "points",
        "median",
    ),
    "points_per_game": ("points_per_game", "fantasy_points_per_game", "fppg", "ppg"),
    "usage_rate": ("usage_rate", "usage", "opportunity_share", "touch_share"),
    "efficiency_score": (
        "efficiency_score",
        "efficiency",
        "fantasy_points_per_opportunity",
        "points_per_touch",
    ),
    "targets": ("targets", "target_volume"),
    "carries": ("carries", "rushing_attempts", "rush_attempts"),
    "receptions": ("receptions", "catches"),
    "receiving_yards": ("receiving_yards", "rec_yards"),
    "rushing_yards": ("rushing_yards", "rush_yards"),
    "passing_yards": ("passing_yards", "pass_yards"),
    "total_touchdowns": ("total_touchdowns", "touchdowns", "total_tds", "tds"),
    "volatility": ("volatility", "weekly_volatility", "standard_deviation", "std_dev"),
}

_ARCHETYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "Production": ("projection", "points_per_game", "total_touchdowns"),
    "Volume": ("usage_rate", "targets", "carries", "receptions"),
    "Efficiency": ("efficiency_score", "points_per_game"),
    "Passing": ("passing_yards",),
    "Rushing": ("rushing_yards", "carries"),
    "Receiving": ("receiving_yards", "targets", "receptions"),
}


@dataclass(slots=True)
class _PreparedPlayer:
    """Internal immutable-enough representation used by clustering."""

    row: dict[str, Any]
    key: str
    player_id: str
    name: str
    position: str
    team: str
    vector: tuple[float, ...]


def _as_mapping(player: Any) -> dict[str, Any]:
    if isinstance(player, Mapping):
        return dict(player)
    model_dump = getattr(player, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(player, "__dict__"):
        return dict(vars(player))
    raise TypeError("player must be a mapping or model-like object")


def _finite_number(value: Any, *, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _strict_vector(vector: Iterable[Any], *, label: str) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an iterable of finite numbers")
    try:
        raw = tuple(vector)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of finite numbers") from exc
    if not raw:
        raise ValueError(f"{label} must not be empty")

    result: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise TypeError(f"{label}[{index}] must be numeric, not bool")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _validated_features(features: Sequence[str] | None) -> tuple[str, ...]:
    if features is None:
        return DEFAULT_SIMILARITY_FEATURES
    if isinstance(features, (str, bytes, bytearray)):
        raise TypeError("features must be a sequence of field names")
    normalized = tuple(str(feature).strip() for feature in features)
    if not normalized or any(not feature for feature in normalized):
        raise ValueError("features must contain at least one non-empty field name")
    if len(set(normalized)) != len(normalized):
        raise ValueError("features must not contain duplicates")
    return normalized


def _sources(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    nested = tuple(row[key] for key in _NESTED_STAT_KEYS if isinstance(row.get(key), Mapping))
    return (row, *nested)


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for source in _sources(row):
        for key in keys:
            if source.get(key) is not None:
                return source[key]
    return None


def _feature_value(row: Mapping[str, Any], feature: str) -> float:
    aliases = _FEATURE_ALIASES.get(feature, (feature,))
    direct = _first_value(row, aliases)
    if direct is not None:
        number = _finite_number(direct)
        if feature == "usage_rate" and 1.0 < number <= 100.0:
            number /= 100.0
        return number

    if feature == "total_touchdowns":
        touchdown_fields = (
            "passing_tds",
            "passing_touchdowns",
            "rushing_tds",
            "rushing_touchdowns",
            "receiving_tds",
            "receiving_touchdowns",
        )
        return sum(_finite_number(_first_value(row, (field,))) for field in touchdown_fields)

    if feature == "points_per_game":
        games = _finite_number(_first_value(row, ("games_played", "games")))
        projection = _finite_number(_first_value(row, _FEATURE_ALIASES["projection"]))
        return projection / games if games > 0.0 else 0.0

    if feature == "efficiency_score":
        points = _finite_number(_first_value(row, _FEATURE_ALIASES["projection"]))
        opportunities = _opportunities(row)
        return points / opportunities if opportunities > 0.0 else 0.0

    if feature == "usage_rate":
        opportunities = _opportunities(row)
        team_opportunities = _finite_number(
            _first_value(row, ("team_opportunities", "team_touches", "team_plays"))
        )
        if team_opportunities > 0.0:
            return opportunities / team_opportunities
        return opportunities

    return 0.0


def _opportunities(row: Mapping[str, Any]) -> float:
    explicit = _finite_number(_first_value(row, ("opportunities", "touches")))
    if explicit > 0.0:
        return explicit
    carries = _finite_number(_first_value(row, _FEATURE_ALIASES["carries"]))
    targets = _finite_number(_first_value(row, _FEATURE_ALIASES["targets"]))
    attempts = _finite_number(_first_value(row, ("passing_attempts", "pass_attempts")))
    return carries + targets + attempts


def cosine_similarity(vector_a: Iterable[Any], vector_b: Iterable[Any]) -> float:
    """Return the cosine similarity between two finite numeric vectors.

    General vectors can produce a value from ``-1`` to ``1``.  Football stat
    vectors are non-negative, so their practical range is ``0`` to ``1``.
    Two all-zero vectors are treated as identical (``1``); one zero and one
    informative vector are treated as unrelated (``0``).
    """

    left = _strict_vector(vector_a, label="vector_a")
    right = _strict_vector(vector_b, label="vector_b")
    if len(left) != len(right):
        raise ValueError("vectors must have the same number of dimensions")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    result = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    # Floating-point accumulation can exceed the mathematical bounds by a few
    # ulps for nearly identical high-dimensional vectors.
    return round(max(-1.0, min(1.0, result)), 6)


def build_stat_vector(player: Any, features: Sequence[str] | None = None) -> tuple[float, ...]:
    """Build a finite stat vector from a player record without mutating it."""

    row = _as_mapping(player)
    selected = _validated_features(features)
    return tuple(_feature_value(row, feature) for feature in selected)


def similarity_score(player_a: Any, player_b: Any, features: Sequence[str] | None = None) -> float:
    """Convenience wrapper for cosine similarity between two player records."""

    selected = _validated_features(features)
    return cosine_similarity(build_stat_vector(player_a, selected), build_stat_vector(player_b, selected))


def _player_key(row: Mapping[str, Any], index: int, used: set[str]) -> tuple[str, str]:
    player_id = str(row.get("player_id") or row.get("id") or row.get("name") or f"player-{index + 1}").strip()
    player_id = player_id or f"player-{index + 1}"
    key = player_id
    suffix = 2
    while key in used:
        key = f"{player_id}#{suffix}"
        suffix += 1
    used.add(key)
    return key, player_id


def _prepare_players(players: Iterable[Any], features: tuple[str, ...]) -> list[_PreparedPlayer]:
    if isinstance(players, (str, bytes, bytearray, Mapping)):
        raise TypeError("players must be an iterable of player records")
    try:
        supplied = list(players)
    except TypeError as exc:
        raise TypeError("players must be an iterable of player records") from exc

    used: set[str] = set()
    prepared: list[_PreparedPlayer] = []
    for index, player in enumerate(supplied):
        row = _as_mapping(player)
        key, player_id = _player_key(row, index, used)
        prepared.append(
            _PreparedPlayer(
                row=row,
                key=key,
                player_id=player_id,
                name=str(row.get("name") or row.get("player_name") or player_id).strip(),
                position=str(row.get("position") or "").strip().upper(),
                team=str(row.get("team") or row.get("nfl_team") or "").strip().upper(),
                vector=build_stat_vector(row, features),
            )
        )
    return prepared


def _scaled_vectors(players: Sequence[_PreparedPlayer]) -> list[tuple[float, ...]]:
    if not players:
        return []
    dimensions = len(players[0].vector)
    scales = [max(abs(player.vector[index]) for player in players) for index in range(dimensions)]
    return [
        tuple(value / scales[index] if scales[index] > 0.0 else 0.0 for index, value in enumerate(player.vector))
        for player in players
    ]


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _mean_vector(vectors: Sequence[Sequence[float]], dimensions: int) -> tuple[float, ...]:
    if not vectors:
        return (0.0,) * dimensions
    return tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions))


def _initial_centroids(vectors: Sequence[tuple[float, ...]], keys: Sequence[str], count: int) -> list[tuple[float, ...]]:
    first = min(range(len(vectors)), key=lambda index: (-sum(value * value for value in vectors[index]), keys[index]))
    chosen = [first]
    while len(chosen) < count:
        remaining = [index for index in range(len(vectors)) if index not in chosen]
        next_index = min(
            remaining,
            key=lambda index: (
                -min(_squared_distance(vectors[index], vectors[centroid]) for centroid in chosen),
                keys[index],
            ),
        )
        chosen.append(next_index)
    return [vectors[index] for index in chosen]


def _assign_vectors(vectors: Sequence[tuple[float, ...]], centroids: Sequence[tuple[float, ...]]) -> list[int]:
    return [
        min(range(len(centroids)), key=lambda cluster: (_squared_distance(vector, centroids[cluster]), cluster))
        for vector in vectors
    ]


def _repair_empty_clusters(
    assignments: list[int],
    vectors: Sequence[tuple[float, ...]],
    centroids: Sequence[tuple[float, ...]],
    count: int,
    keys: Sequence[str],
) -> None:
    sizes = [assignments.count(cluster) for cluster in range(count)]
    for empty in (cluster for cluster, size in enumerate(sizes) if size == 0):
        donors = [index for index, cluster in enumerate(assignments) if sizes[cluster] > 1]
        if not donors:
            donors = list(range(len(assignments)))
        donor = min(
            donors,
            key=lambda index: (
                -_squared_distance(vectors[index], centroids[assignments[index]]),
                keys[index],
            ),
        )
        sizes[assignments[donor]] -= 1
        assignments[donor] = empty
        sizes[empty] += 1


def _cluster_label(centroid: Sequence[float], features: Sequence[str]) -> str:
    index_by_feature = {feature: index for index, feature in enumerate(features)}
    scores: dict[str, float] = {}
    for group, group_features in _ARCHETYPE_GROUPS.items():
        values = [centroid[index_by_feature[feature]] for feature in group_features if feature in index_by_feature]
        if values:
            scores[group] = sum(values) / len(values)
    if not scores or max(scores.values(), default=0.0) <= 0.0:
        return "Balanced Archetype"
    dominant = min(scores, key=lambda group: (-scores[group], group))
    return f"{dominant} Archetype"


def _cluster_prepared(
    players: Sequence[_PreparedPlayer],
    features: tuple[str, ...],
    count: int,
    max_iterations: int,
) -> dict[str, Any]:
    vectors = _scaled_vectors(players)
    centroids = _initial_centroids(vectors, [player.key for player in players], count)
    prior_assignments: list[int] | None = None

    for _ in range(max_iterations):
        assignments = _assign_vectors(vectors, centroids)
        _repair_empty_clusters(assignments, vectors, centroids, count, [player.key for player in players])
        updated = [
            _mean_vector(
                [vector for vector, assignment in zip(vectors, assignments, strict=True) if assignment == cluster],
                len(features),
            )
            for cluster in range(count)
        ]
        shift = max(_squared_distance(old, new) for old, new in zip(centroids, updated, strict=True))
        centroids = updated
        if assignments == prior_assignments or shift <= 1e-14:
            break
        prior_assignments = assignments

    assignments = _assign_vectors(vectors, centroids)
    _repair_empty_clusters(assignments, vectors, centroids, count, [player.key for player in players])
    centroids = [
        _mean_vector(
            [vector for vector, assignment in zip(vectors, assignments, strict=True) if assignment == cluster],
            len(features),
        )
        for cluster in range(count)
    ]

    groups = [
        (cluster, centroids[cluster], [index for index, assignment in enumerate(assignments) if assignment == cluster])
        for cluster in range(count)
    ]
    groups.sort(
        key=lambda item: (
            -sum(item[1]),
            tuple(-value for value in item[1]),
            tuple(players[index].key for index in item[2]),
        )
    )

    cluster_rows: list[dict[str, Any]] = []
    assignment_rows: dict[str, str] = {}
    centroid_rows: dict[str, dict[str, float]] = {}
    label_counts: dict[str, int] = {}
    for cluster_id, (_, centroid, member_indexes) in enumerate(groups):
        base_label = _cluster_label(centroid, features)
        label_counts[base_label] = label_counts.get(base_label, 0) + 1
        label = base_label if label_counts[base_label] == 1 else f"{base_label} {label_counts[base_label]}"
        centroid_row = {feature: round(centroid[index], 6) for index, feature in enumerate(features)}
        members = []
        for index in sorted(member_indexes, key=lambda value: players[value].key):
            player = players[index]
            assignment_rows[player.key] = label
            members.append(
                {
                    "player_id": player.player_id,
                    "name": player.name,
                    "position": player.position,
                    "team": player.team,
                }
            )
        centroid_rows[label] = centroid_row
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "archetype": label,
                "size": len(members),
                "centroid": centroid_row,
                "members": members,
            }
        )

    return {
        "features": list(features),
        "n_clusters": count,
        "assignments": assignment_rows,
        "centroids": centroid_rows,
        "clusters": cluster_rows,
    }


def cluster_archetypes(
    players: Iterable[Any],
    *,
    features: Sequence[str] | None = None,
    n_clusters: int | None = None,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """Cluster players into deterministic, human-readable archetypes.

    Vectors are scaled per feature before clustering so passing yards cannot
    overwhelm rates or shares merely because they use larger units.
    """

    selected = _validated_features(features)
    prepared = _prepare_players(players, selected)
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not prepared:
        if n_clusters not in (None, 0):
            raise ValueError("n_clusters cannot be positive when players is empty")
        return {"features": list(selected), "n_clusters": 0, "assignments": {}, "centroids": {}, "clusters": []}

    if n_clusters is None:
        count = min(4, max(1, math.ceil(math.sqrt(len(prepared) / 2.0))))
    elif not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise TypeError("n_clusters must be an integer")
    else:
        count = n_clusters
    if not 1 <= count <= len(prepared):
        raise ValueError("n_clusters must be between 1 and the number of players")
    return _cluster_prepared(prepared, selected, count, max_iterations)


def _same_identity(target: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    target_id = str(target.get("player_id") or target.get("id") or "").strip()
    candidate_id = str(candidate.get("player_id") or candidate.get("id") or "").strip()
    if target_id and candidate_id:
        return target_id == candidate_id
    target_name = str(target.get("name") or target.get("player_name") or "").strip().casefold()
    candidate_name = str(candidate.get("name") or candidate.get("player_name") or "").strip().casefold()
    target_team = str(target.get("team") or target.get("nfl_team") or "").strip().upper()
    candidate_team = str(candidate.get("team") or candidate.get("nfl_team") or "").strip().upper()
    return bool(target_name) and target_name == candidate_name and (not target_team or not candidate_team or target_team == candidate_team)


def _similarity_context(
    player: Any,
    player_pool: Iterable[Any],
    *,
    limit: int,
    features: Sequence[str] | None,
    same_position: bool,
) -> tuple[dict[str, Any], tuple[str, ...], str, list[dict[str, Any]]]:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not isinstance(same_position, bool):
        raise TypeError("same_position must be a boolean")

    target = _as_mapping(player)
    selected = _validated_features(features)
    target_position = str(target.get("position") or "").strip().upper()
    if isinstance(player_pool, (str, bytes, bytearray, Mapping)):
        raise TypeError("player_pool must be an iterable of player records")

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in player_pool:
        row = _as_mapping(candidate)
        if _same_identity(target, row):
            continue
        position = str(row.get("position") or "").strip().upper()
        if same_position and target_position and position != target_position:
            continue
        identity = (
            str(row.get("player_id") or row.get("id") or "").strip(),
            str(row.get("name") or row.get("player_name") or "").strip().casefold(),
            str(row.get("team") or row.get("nfl_team") or "").strip().upper(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(row)

    prepared = _prepare_players([target, *candidates], selected)
    scaled = _scaled_vectors(prepared)
    clustering = _cluster_prepared(
        prepared,
        selected,
        min(4, max(1, math.ceil(math.sqrt(len(prepared) / 2.0)))),
        100,
    )
    archetypes = clustering["assignments"]
    target_archetype = archetypes[prepared[0].key]

    comparisons: list[dict[str, Any]] = []
    for index, candidate in enumerate(prepared[1:], start=1):
        score = max(0.0, min(1.0, cosine_similarity(scaled[0], scaled[index])))
        comparisons.append(
            {
                "player_id": candidate.player_id,
                "name": candidate.name,
                "position": candidate.position,
                "team": candidate.team,
                "similarity": round(score, 4),
                "similarity_percent": round(score * 100.0, 1),
                "archetype": archetypes[candidate.key],
            }
        )
    comparisons.sort(key=lambda row: (-row["similarity"], row["name"].casefold(), row["player_id"]))
    return target, selected, target_archetype, comparisons[:limit]


def nearest_player_comps(
    player: Any,
    player_pool: Iterable[Any],
    *,
    limit: int = 5,
    features: Sequence[str] | None = None,
    same_position: bool = True,
) -> list[dict[str, Any]]:
    """Return the nearest comparable players ordered by cosine similarity."""

    return _similarity_context(
        player,
        player_pool,
        limit=limit,
        features=features,
        same_position=same_position,
    )[3]


def compute_player_similarity(
    player: Any,
    player_pool: Iterable[Any],
    *,
    limit: int = 5,
    features: Sequence[str] | None = None,
    same_position: bool = True,
) -> dict[str, Any]:
    """Return a Graph Lab / Player Detail-ready similarity payload."""

    target, selected, archetype, comparisons = _similarity_context(
        player,
        player_pool,
        limit=limit,
        features=features,
        same_position=same_position,
    )
    player_id = str(target.get("player_id") or target.get("id") or target.get("name") or "").strip()
    return {
        "player_id": player_id,
        "name": str(target.get("name") or target.get("player_name") or player_id).strip(),
        "position": str(target.get("position") or "").strip().upper(),
        "features": list(selected),
        "vector": list(build_stat_vector(target, selected)),
        "archetype": archetype,
        "comparisons": comparisons,
    }


# Clear, discoverable aliases for callers that use the terminology from the
# product specification rather than the shorter implementation names.
calculate_cosine_similarity = cosine_similarity
create_stat_vector = build_stat_vector
archetype_clustering = cluster_archetypes
nearest_neighbors = nearest_player_comps


__all__ = [
    "DEFAULT_SIMILARITY_FEATURES",
    "archetype_clustering",
    "build_stat_vector",
    "calculate_cosine_similarity",
    "cluster_archetypes",
    "compute_player_similarity",
    "cosine_similarity",
    "create_stat_vector",
    "nearest_neighbors",
    "nearest_player_comps",
    "similarity_score",
]
