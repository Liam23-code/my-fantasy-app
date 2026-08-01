"""Caching layer for repeated scoring and projection normalization.

Usage::

    from fantasy.cache import get_cache, cached_normalize_projection, cached_calculate_fantasy_points

    cache = get_cache()  # in-memory LRU by default
    canonical = cached_normalize_projection(source, cache)
    result = cached_calculate_fantasy_points(canonical, cache, mode="ppr")

``get_cache(backend="redis", ...)`` returns a Redis-backed cache when the
``redis`` package and a reachable server are both available, and silently
falls back to the in-memory LRU otherwise -- callers never need an
``if redis_available`` branch of their own.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Callable, Protocol

from fantasy.adapter import IDENTITY_ALIASES, normalize_projection
from fantasy.scoring import calculate_fantasy_points

_PLAYER_ID_ALIASES = IDENTITY_ALIASES["player_id"]


class Cache(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def clear(self) -> None: ...


class LRUCache:
    """A tiny, dependency-free least-recently-used cache."""

    def __init__(self, maxsize: int = 2048):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data


class RedisCache:
    """A thin JSON-serializing wrapper around a redis-py client."""

    def __init__(self, client: Any, prefix: str = "fantasy:", ttl_seconds: int | None = 3600):
        self._client = client
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _namespaced(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._client.get(self._namespaced(key))
        if raw is None:
            return default
        return json.loads(raw)

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        namespaced = self._namespaced(key)
        if self._ttl:
            self._client.setex(namespaced, self._ttl, payload)
        else:
            self._client.set(namespaced, payload)

    def clear(self) -> None:
        for key in self._client.scan_iter(f"{self._prefix}*"):
            self._client.delete(key)


def get_cache(backend: str = "memory", **options: Any) -> Cache:
    """Build a cache. ``backend="redis"`` degrades to in-memory on any failure.

    Options: ``maxsize`` (memory), ``client``/``host``/``port``/``db``/
    ``prefix``/``ttl_seconds`` (redis).
    """
    if backend == "redis":
        try:
            client = options.get("client")
            if client is None:
                import redis  # local import: optional dependency

                client = redis.Redis(
                    host=options.get("host", "localhost"),
                    port=options.get("port", 6379),
                    db=options.get("db", 0),
                    socket_connect_timeout=options.get("socket_connect_timeout", 1),
                )
            client.ping()
            return RedisCache(client, prefix=options.get("prefix", "fantasy:"), ttl_seconds=options.get("ttl_seconds", 3600))
        except Exception:
            pass  # any import/connection failure: fall through to in-memory
    return LRUCache(maxsize=options.get("maxsize", 2048))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _player_id_hint(source: Any) -> Any:
    """Cheaply extract a player id from a dict key or object attribute, without full normalization."""
    if isinstance(source, dict):
        return next((source[alias] for alias in _PLAYER_ID_ALIASES if source.get(alias)), None)
    return next((getattr(source, alias) for alias in _PLAYER_ID_ALIASES if getattr(source, alias, None)), None)


def cached_normalize_projection(
    source: Any,
    cache: Cache,
    loader: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """``normalize_projection`` memoized by player id (or bare id/name) when one is cheaply known.

    The pre-check uses the same id aliases the adapter itself resolves
    (``player_id``, ``id``, ``playerId``, ...) against either dict keys or
    object attributes, so a source identified by any of those round-trips
    through the cache on repeated calls -- not just sources already using
    the canonical ``player_id`` key or attribute.
    """
    key: str | None = None
    if isinstance(source, (str, int)) and not isinstance(source, bool):
        key = f"proj:{source}"
    else:
        player_id = _player_id_hint(source)
        if player_id:
            key = f"proj:{player_id}"

    if key is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    # `key` is derived from the exact same alias tuple the adapter resolves
    # `player_id` from, so if the pre-check above missed it, normalization
    # cannot find one either -- there's no second chance to key this result.
    result = normalize_projection(source, loader=loader)
    if key is not None:
        cache.set(key, result)
    return result


def cached_calculate_fantasy_points(
    projection: dict[str, Any],
    cache: Cache,
    mode: str = "ppr",
    custom_rules: dict[str, Any] | None = None,
    bonuses: bool = True,
) -> dict[str, Any]:
    """``calculate_fantasy_points`` memoized by (player_id, mode, rules hash, bonuses)."""
    player_id = projection.get("player_id")
    if not player_id:
        return calculate_fantasy_points(projection, mode, custom_rules, bonuses)

    rules_hash = _stable_hash(custom_rules) if custom_rules else "none"
    key = f"score:{player_id}:{mode}:{rules_hash}:{bonuses}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    result = calculate_fantasy_points(projection, mode, custom_rules, bonuses)
    cache.set(key, result)
    return result
