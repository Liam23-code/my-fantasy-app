"""A tiny, generic, thread-safe TTL cache decorator -- no sport, no odds, no data in it.

Shared infrastructure, not betting logic: lives in ``betting/`` alongside
:mod:`betting.odds_math` because that's this project's established home for
"genuinely sport-agnostic, reused directly rather than duplicated" code
(see betting_engine.md). Both the NFL and NBA sides import this decorator
directly rather than each rolling their own.

Deliberately minimal: in-memory, per-process, keyed by the decorated
function's arguments (which must be hashable -- this is not meant for
caching calls with file-like/upload arguments, only calls with simple,
repeatable arguments like a default file path or no arguments at all).
"""
from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(seconds: float) -> Callable[[F], F]:
    """Cache a function's return value per distinct argument set for ``seconds``.

    Thread-safe (a single lock per decorated function serializes reads and
    writes to that function's cache -- fine at this cache's scale: a
    handful of entries, refreshed at most every few minutes). Raises the
    same way the undecorated call would; a raised call is never cached.

    Like ``functools.lru_cache``, a cache hit returns the *same object* a
    previous call returned, not a copy -- callers must treat the result as
    read-only. Every function this decorator is applied to in this
    codebase already returns a freshly built dict/list per call and none
    of its callers mutate the result in place, so this holds today; it
    would stop holding if a decorated function's result were ever mutated
    by a caller.
    """

    def decorator(func: F) -> F:
        cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                cached = cache.get(key)
                if cached is not None and now - cached[0] < seconds:
                    return cached[1]
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (now, result)
            return result

        def cache_clear() -> None:
            with lock:
                cache.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
