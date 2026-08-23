"""Unit tests for betting.cache_utils.ttl_cache.

Distinct from fantasy.cache (see that module's test_cache.py): fantasy.cache
memoizes deterministic, caller-keyed pure functions (scoring/normalization
math with no time dimension); ttl_cache is a transparent decorator for
time-sensitive I/O (file reads, live API calls) where staleness matters.
"""
from __future__ import annotations

import time

from betting.cache_utils import ttl_cache


def test_repeated_calls_within_ttl_return_cached_result_without_recomputing():
    calls = []

    @ttl_cache(seconds=60)
    def loader(x):
        calls.append(x)
        return x * 2

    assert loader(3) == 6
    assert loader(3) == 6
    assert calls == [3]  # second call served from cache, not recomputed


def test_different_arguments_are_cached_independently():
    @ttl_cache(seconds=60)
    def loader(x):
        return x * 2

    assert loader(3) == 6
    assert loader(4) == 8


def test_kwargs_participate_in_the_cache_key():
    calls = []

    @ttl_cache(seconds=60)
    def loader(x, *, flag=False):
        calls.append((x, flag))
        return (x, flag)

    loader(1, flag=True)
    loader(1, flag=False)
    assert calls == [(1, True), (1, False)]


def test_expired_entry_is_recomputed():
    calls = []

    @ttl_cache(seconds=0.05)
    def loader():
        calls.append(1)
        return "value"

    loader()
    time.sleep(0.1)
    loader()
    assert calls == [1, 1]


def test_cache_clear_forces_recomputation():
    calls = []

    @ttl_cache(seconds=60)
    def loader():
        calls.append(1)
        return "value"

    loader()
    loader.cache_clear()
    loader()
    assert calls == [1, 1]


def test_a_raised_call_is_not_cached():
    attempts = {"count": 0}

    @ttl_cache(seconds=60)
    def loader():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("first call fails")
        return "recovered"

    try:
        loader()
    except RuntimeError:
        pass
    assert loader() == "recovered"
    assert attempts["count"] == 2
