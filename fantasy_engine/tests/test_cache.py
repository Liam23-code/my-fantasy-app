"""Unit tests for fantasy.cache."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from fantasy.cache import (
    LRUCache,
    RedisCache,
    cached_calculate_fantasy_points,
    cached_normalize_projection,
    get_cache,
)


class FakeRedisClient:
    """Minimal in-memory stand-in for redis.Redis, used to test RedisCache without a server."""

    def __init__(self, fail_ping: bool = False):
        self._store: dict[str, str] = {}
        self._fail_ping = fail_ping

    def ping(self) -> bool:
        if self._fail_ping:
            raise ConnectionError("no server")
        return True

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    def scan_iter(self, pattern: str):
        prefix = pattern.rstrip("*")
        return [key for key in self._store if key.startswith(prefix)]

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


def test_lru_cache_basic_get_set():
    cache = LRUCache(maxsize=10)
    assert cache.get("missing") is None
    assert cache.get("missing", "default") == "default"
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert "a" in cache
    assert len(cache) == 1


def test_lru_cache_evicts_least_recently_used():
    cache = LRUCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # touch "a" so "b" becomes the least recently used
    cache.set("c", 3)  # should evict "b", not "a"
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache


def test_lru_cache_clear():
    cache = LRUCache()
    cache.set("a", 1)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_lru_cache_rejects_invalid_maxsize():
    with pytest.raises(ValueError):
        LRUCache(maxsize=0)


def test_get_cache_defaults_to_lru():
    cache = get_cache()
    assert isinstance(cache, LRUCache)


def test_get_cache_redis_backend_with_working_client():
    client = FakeRedisClient()
    cache = get_cache(backend="redis", client=client)
    assert isinstance(cache, RedisCache)
    cache.set("k", {"x": 1})
    assert cache.get("k") == {"x": 1}


def test_get_cache_redis_backend_falls_back_to_memory_on_ping_failure():
    client = FakeRedisClient(fail_ping=True)
    cache = get_cache(backend="redis", client=client)
    assert isinstance(cache, LRUCache)


def test_get_cache_redis_backend_falls_back_when_package_missing():
    # The `redis` package is not installed in this environment, so omitting
    # `client` forces the internal `import redis` to fail and fall back.
    cache = get_cache(backend="redis")
    assert isinstance(cache, LRUCache)


def test_redis_cache_clear_deletes_only_prefixed_keys():
    client = FakeRedisClient()
    cache = RedisCache(client, prefix="fantasy:")
    cache.set("a", 1)
    client.set("other:b", "2")  # not under our prefix
    cache.clear()
    assert cache.get("a") is None
    assert client.get("other:b") == "2"


def test_redis_cache_missing_key_returns_default():
    cache = RedisCache(FakeRedisClient())
    assert cache.get("nope", "fallback") == "fallback"


def test_redis_cache_without_ttl_uses_plain_set():
    client = FakeRedisClient()
    cache = RedisCache(client, ttl_seconds=None)
    cache.set("k", {"x": 1})
    assert cache.get("k") == {"x": 1}


def test_get_cache_redis_backend_constructs_client_when_package_available(monkeypatch):
    # Fake the `redis` module so get_cache's `import redis` + `redis.Redis(...)`
    # construction path can be exercised without the real package installed.
    fake_client = FakeRedisClient()
    fake_module = types.SimpleNamespace(Redis=lambda **kwargs: fake_client)
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    cache = get_cache(backend="redis", host="example.invalid")
    assert isinstance(cache, RedisCache)
    cache.set("k", 1)
    assert cache.get("k") == 1


def test_cached_normalize_projection_hits_cache_by_player_id():
    cache = LRUCache()
    source = {"player_id": "p1", "name": "A", "position": "RB", "rushing_yards": 50}
    first = cached_normalize_projection(source, cache)
    source["rushing_yards"] = 999  # mutate after first call
    second = cached_normalize_projection(source, cache)
    assert second == first  # cache hit returned the stale (correct) first result
    assert second["rushing_yards"] == 50


def test_cached_normalize_projection_hits_cache_by_bare_id_with_loader():
    cache = LRUCache()
    calls = []

    def loader(name):
        calls.append(name)
        return {"name": name, "position": "WR", "receiving_yards": 80}

    cached_normalize_projection("Puka Nacua", cache, loader=loader)
    cached_normalize_projection("Puka Nacua", cache, loader=loader)
    assert len(calls) == 1  # second call was a cache hit, loader not invoked again


def test_cached_normalize_projection_recognizes_id_aliases_beyond_player_id():
    # The pre-check covers every id alias the adapter itself resolves
    # (player_id/id/playerId/...), not just a literal "player_id" key, so a
    # source keyed under "playerId" still round-trips through the cache.
    cache = LRUCache()
    source = {"playerId": "p9", "name": "X", "position": "RB", "rushing_yards": 50}
    first = cached_normalize_projection(source, cache)
    source["rushing_yards"] = 999
    second = cached_normalize_projection(source, cache)
    assert second == first
    assert second["rushing_yards"] == 50


def test_cached_normalize_projection_recognizes_id_on_object_attributes():
    # The pre-check also reads object attributes, not just dict keys, so a
    # non-dict player object round-trips through the cache too.
    @dataclass
    class FakePlayer:
        player_id: str
        name: str
        position: str
        rushing_yards: float

    cache = LRUCache()
    player = FakePlayer(player_id="p7", name="X", position="RB", rushing_yards=50)
    first = cached_normalize_projection(player, cache)
    player.rushing_yards = 999
    second = cached_normalize_projection(player, cache)
    assert second == first
    assert second["rushing_yards"] == 50


def test_cached_normalize_projection_without_resolvable_id_still_works_uncached():
    cache = LRUCache()
    source = {"name": "", "position": "RB", "rushing_yards": 50}  # no player_id or name to key on
    first = cached_normalize_projection(source, cache)
    source["rushing_yards"] = 90
    second = cached_normalize_projection(source, cache)
    assert first["rushing_yards"] == 50
    assert second["rushing_yards"] == 90  # not cached, reflects the mutation


def test_cached_calculate_fantasy_points_hits_cache_by_player_id_mode_and_rules():
    cache = LRUCache()
    projection = {"player_id": "p1", "passing_yards": 250}
    first = cached_calculate_fantasy_points(projection, cache, mode="ppr")
    projection["passing_yards"] = 999
    second = cached_calculate_fantasy_points(projection, cache, mode="ppr")
    assert second == first


def test_cached_calculate_fantasy_points_different_modes_do_not_collide():
    cache = LRUCache()
    projection = {"player_id": "p1", "receptions": 5}
    ppr = cached_calculate_fantasy_points(projection, cache, mode="ppr")
    standard = cached_calculate_fantasy_points(projection, cache, mode="standard")
    assert ppr["total_points"] != standard["total_points"]


def test_cached_calculate_fantasy_points_different_custom_rules_do_not_collide():
    cache = LRUCache()
    projection = {"player_id": "p1", "passing_yards": 100}
    default_result = cached_calculate_fantasy_points(projection, cache, mode="ppr")
    custom_result = cached_calculate_fantasy_points(
        projection, cache, mode="ppr", custom_rules={"multipliers": {"passing_yards": 1.0}}
    )
    assert default_result["total_points"] != custom_result["total_points"]


def test_cached_calculate_fantasy_points_without_player_id_is_not_cached():
    cache = LRUCache()
    projection = {"passing_yards": 100}
    first = cached_calculate_fantasy_points(projection, cache, mode="ppr")
    projection["passing_yards"] = 200
    second = cached_calculate_fantasy_points(projection, cache, mode="ppr")
    assert first["total_points"] != second["total_points"]
