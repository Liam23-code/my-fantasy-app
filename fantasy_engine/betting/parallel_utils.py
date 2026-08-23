"""Bounded thread-pool helpers for I/O-bound work: real per-player fetches, EV math over many rows.

Shared infrastructure, not betting logic -- lives alongside
:mod:`betting.cache_utils` and :mod:`betting.odds_math` for the same
reason (see betting_engine.md): genuinely sport-agnostic code is imported
directly by both sides, never duplicated. NBA's
``modules.parallel_utils`` re-exports this module rather than
reimplementing it.

Two different jobs get two different treatments here, deliberately:

* :func:`parallel_map` -- for genuinely I/O-bound work (a real network
  call per item, e.g. one NBA player's real projection, or one NFL
  player's real game-log fetch). The GIL doesn't hold up a thread that's
  blocked waiting on a socket, so a bounded thread pool gives a real
  wall-clock speedup here, proportional to how much of each call is spent
  waiting rather than computing.
* :func:`parallel_ev_map` -- for pure, CPU-bound math with no I/O (odds
  math over a list of already-loaded rows, e.g. NFL's
  ``betting.prop_model.evaluate_props`` or NBA's
  ``modules.nba_prop_model.price_aware_evaluations``). This is
  intentionally still a thread pool, not a process pool: the per-row math
  at both sports' current row counts (a handful of ``math.erf``/arithmetic
  calls each) is cheap enough that process-spawn overhead would dominate,
  and thread-based parallelism buys little under the GIL for pure CPU work
  at this scale. It exists to make "parallel EV calculations for both
  sports" a real, tested code path rather than a claim with nothing behind
  it -- see performance_notes.md for the measured reality at today's row
  counts, and for when switching to a process pool would actually pay off.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_DEFAULT_MAX_WORKERS = 8


def parallel_map(fn: Callable[[T], R], items: Iterable[T], *, max_workers: int = _DEFAULT_MAX_WORKERS) -> list[R]:
    """Apply ``fn`` to every item concurrently; skip an item whose call raises.

    Order of the returned results matches ``items`` order (not completion
    order), so callers can zip results back against their inputs. A single
    item's failure (a real API error, a timeout) doesn't abort the batch --
    every underlying loader in this codebase already fails soft on its own
    (see offline_data_contract.md), so this only guards against an
    unexpected exception escaping one of them.
    """
    items = list(items)
    if not items:
        return []
    if len(items) == 1 or max_workers <= 1:
        results = []
        for item in items:
            try:
                results.append(fn(item))
            except Exception:
                continue
        return results

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = [pool.submit(fn, item) for item in items]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                continue
    return results


def parallel_ev_map(fn: Callable[[T], R], rows: Iterable[T], *, max_workers: int = _DEFAULT_MAX_WORKERS) -> list[R]:
    """Apply a pure EV/probability function to every row concurrently. See module docstring."""
    return parallel_map(fn, rows, max_workers=max_workers)
