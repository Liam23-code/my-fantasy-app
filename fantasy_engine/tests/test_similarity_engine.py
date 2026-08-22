from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from quant.similarity_engine import (
    DEFAULT_SIMILARITY_FEATURES,
    archetype_clustering,
    build_stat_vector,
    cluster_archetypes,
    compute_player_similarity,
    cosine_similarity,
    nearest_neighbors,
    nearest_player_comps,
    similarity_score,
)


def _player(
    player_id: str,
    name: str,
    position: str,
    *,
    team: str = "TST",
    projection: float = 200.0,
    carries: float = 0.0,
    targets: float = 0.0,
    receptions: float = 0.0,
) -> dict:
    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": team,
        "projection": projection,
        "points_per_game": projection / 17.0,
        "usage_rate": (carries + targets) / 500.0,
        "efficiency_score": projection / max(1.0, carries + targets),
        "carries": carries,
        "targets": targets,
        "receptions": receptions,
    }


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0], [-1, 0], -1.0),
        ([0, 0], [0, 0], 1.0),
        ([0, 0], [1, 2], 0.0),
    ],
)
def test_cosine_similarity_known_vectors(left, right, expected):
    assert cosine_similarity(left, right) == expected


def test_cosine_similarity_validates_shape_and_values():
    with pytest.raises(ValueError, match="same number"):
        cosine_similarity([1, 2], [1])
    with pytest.raises(ValueError, match="must not be empty"):
        cosine_similarity([], [])
    with pytest.raises(TypeError, match="numeric"):
        cosine_similarity([True], [1])
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([float("nan")], [1])


def test_build_stat_vector_supports_aliases_nested_stats_and_derived_touchdowns():
    player = {
        "expected_fantasy_points": 275,
        "games_played": 17,
        "stats": {
            "target_volume": 120,
            "rush_attempts": 8,
            "passing_tds": 0,
            "rushing_tds": 1,
            "receiving_tds": 8,
        },
    }
    before = copy.deepcopy(player)

    vector = build_stat_vector(player, ["projection", "points_per_game", "targets", "carries", "total_touchdowns"])

    assert vector == pytest.approx((275.0, 275 / 17, 120.0, 8.0, 9.0))
    assert player == before


def test_build_stat_vector_accepts_model_like_objects_and_default_features():
    player = SimpleNamespace(player_id="one", name="One", position="RB", projection=170, carries=200)
    vector = build_stat_vector(player)
    assert len(vector) == len(DEFAULT_SIMILARITY_FEATURES)
    assert vector[0] == 170.0


def test_build_stat_vector_normalizes_percentage_usage_rates():
    assert build_stat_vector({"usage_rate": 72}, ["usage_rate"]) == (0.72,)


def test_similarity_score_is_symmetric_and_identical_players_score_one():
    left = _player("a", "A", "RB", projection=220, carries=220, targets=45, receptions=35)
    right = dict(left, player_id="b", name="B")
    assert similarity_score(left, right) == 1.0
    assert similarity_score(left, right) == similarity_score(right, left)


def test_cluster_archetypes_is_deterministic_and_separates_player_styles():
    rushers = [
        _player("r1", "Runner One", "RB", carries=280, targets=20),
        _player("r2", "Runner Two", "RB", carries=250, targets=15),
        _player("r3", "Runner Three", "RB", carries=230, targets=25),
    ]
    receivers = [
        _player("w1", "Receiver One", "WR", carries=2, targets=160, receptions=115),
        _player("w2", "Receiver Two", "WR", carries=4, targets=145, receptions=105),
        _player("w3", "Receiver Three", "WR", carries=0, targets=130, receptions=95),
    ]
    players = rushers + receivers

    result = cluster_archetypes(players, features=["carries", "targets", "receptions"], n_clusters=2)
    reversed_result = cluster_archetypes(reversed(players), features=["carries", "targets", "receptions"], n_clusters=2)

    assert result["n_clusters"] == 2
    assert result["assignments"] == reversed_result["assignments"]
    assert sum(cluster["size"] for cluster in result["clusters"]) == 6
    assert result["assignments"]["r1"] == result["assignments"]["r2"] == result["assignments"]["r3"]
    assert result["assignments"]["w1"] == result["assignments"]["w2"] == result["assignments"]["w3"]
    assert result["assignments"]["r1"] != result["assignments"]["w1"]
    assert set(result["centroids"]) == {cluster["archetype"] for cluster in result["clusters"]}


def test_cluster_archetypes_alias_and_empty_pool():
    assert archetype_clustering([]) == {
        "features": list(DEFAULT_SIMILARITY_FEATURES),
        "n_clusters": 0,
        "assignments": {},
        "centroids": {},
        "clusters": [],
    }
    with pytest.raises(ValueError, match="cannot be positive"):
        archetype_clustering([], n_clusters=1)


def test_nearest_player_comps_filters_position_self_and_duplicates():
    target = _player("target", "Target", "RB", team="AAA", projection=260, carries=240, targets=70, receptions=55)
    close = _player("close", "Close Comp", "RB", projection=250, carries=230, targets=68, receptions=52)
    distant = _player("far", "Distant Comp", "RB", projection=95, carries=60, targets=5, receptions=2)
    wrong_position = _player("wr", "Identical WR", "WR", projection=260, carries=240, targets=70, receptions=55)
    duplicate = dict(close)

    result = nearest_player_comps(
        target,
        [target, distant, wrong_position, close, duplicate],
        features=["projection", "carries", "targets", "receptions"],
    )

    assert [row["player_id"] for row in result] == ["close", "far"]
    assert result[0]["similarity"] > result[1]["similarity"]
    assert 0.0 <= result[0]["similarity"] <= 1.0
    assert result[0]["similarity_percent"] == pytest.approx(result[0]["similarity"] * 100, abs=0.1)


def test_nearest_neighbors_can_compare_across_positions_and_respects_limit():
    target = _player("target", "Target", "RB", projection=220, carries=10, targets=100, receptions=80)
    wr = _player("wr", "Wideout", "WR", projection=220, carries=10, targets=100, receptions=80)
    other = _player("other", "Other", "RB", projection=180, carries=150, targets=20, receptions=15)

    result = nearest_neighbors(target, [wr, other], limit=1, same_position=False)

    assert len(result) == 1
    assert result[0]["player_id"] == "wr"
    with pytest.raises(ValueError, match="non-negative"):
        nearest_neighbors(target, [wr], limit=-1)


def test_compute_player_similarity_returns_ui_ready_payload_without_mutation():
    target = _player("target", "Target", "RB", projection=210, carries=205, targets=50, receptions=42)
    pool = [
        _player("one", "One", "RB", projection=205, carries=200, targets=48, receptions=40),
        _player("two", "Two", "RB", projection=160, carries=130, targets=20, receptions=15),
    ]
    target_before = copy.deepcopy(target)
    pool_before = copy.deepcopy(pool)

    result = compute_player_similarity(target, pool, limit=2)

    assert result["player_id"] == "target"
    assert result["name"] == "Target"
    assert result["position"] == "RB"
    assert result["features"] == list(DEFAULT_SIMILARITY_FEATURES)
    assert len(result["vector"]) == len(DEFAULT_SIMILARITY_FEATURES)
    assert result["archetype"].endswith("Archetype")
    assert [row["player_id"] for row in result["comparisons"]] == ["one", "two"]
    assert target == target_before
    assert pool == pool_before


def test_feature_and_cluster_validation_is_explicit():
    player = _player("one", "One", "RB")
    with pytest.raises(TypeError, match="sequence"):
        build_stat_vector(player, "projection")
    with pytest.raises(ValueError, match="duplicates"):
        build_stat_vector(player, ["projection", "projection"])
    with pytest.raises(ValueError, match="between 1"):
        cluster_archetypes([player], n_clusters=2)
    with pytest.raises(ValueError, match="positive integer"):
        cluster_archetypes([player], max_iterations=0)
