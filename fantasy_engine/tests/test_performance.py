"""Performance test: batch score 10k projections within a documented budget.

The 10-second budget mirrors the spec's "10k players in under 10s on modest
hardware" target. ``calculate_fantasy_points`` itself is pure-Python dict
arithmetic with no I/O and no heavy per-call library work, so in practice
this runs orders of magnitude under budget on any machine that can run the
rest of the test suite -- the assertion is intentionally generous so it
never flakes on a loaded CI runner, not because the real number is close.

This is honest, measured timing from whatever machine runs the suite, not a
claim about a specific reference machine (see the perf note in README.md for
the actual number observed during development).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from fantasy.adapter import normalize_projections
from fantasy.scoring import batch_calculate_fantasy_points

BATCH_SIZE = 10_000
TIME_BUDGET_SECONDS = 10.0
POSITIONS = ["QB", "RB", "WR", "TE"]


def _synthetic_projections(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    projections = []
    for i in range(n):
        position = POSITIONS[i % len(POSITIONS)]
        projections.append(
            {
                "player_id": f"perf:{i}",
                "name": f"Player {i}",
                "position": position,
                "passing_yards": float(rng.normal(200, 50)) if position == "QB" else 0.0,
                "passing_tds": float(rng.normal(1.5, 0.6)) if position == "QB" else 0.0,
                "interceptions": float(rng.normal(0.5, 0.3)),
                "rushing_yards": float(rng.normal(40, 30)),
                "rushing_tds": float(rng.normal(0.3, 0.2)),
                "receptions": float(rng.normal(3, 2)),
                "receiving_yards": float(rng.normal(40, 25)),
                "receiving_tds": float(rng.normal(0.25, 0.2)),
                "fumbles_lost": float(rng.normal(0.05, 0.05)),
            }
        )
    return projections


def test_batch_score_10k_players_within_time_budget():
    projections = _synthetic_projections(BATCH_SIZE)

    started = time.perf_counter()
    results = batch_calculate_fantasy_points(projections, mode="ppr")
    elapsed = time.perf_counter() - started

    assert len(results) == BATCH_SIZE
    assert all("total_points" in result for result in results)
    assert elapsed < TIME_BUDGET_SECONDS, f"Batch-scored {BATCH_SIZE} players in {elapsed:.2f}s, budget was {TIME_BUDGET_SECONDS}s"
    print(f"\n[perf] batch_calculate_fantasy_points: {BATCH_SIZE} players in {elapsed * 1000:.1f}ms ({BATCH_SIZE / elapsed:.0f} players/sec)")


def test_batch_normalize_10k_projections_within_time_budget():
    """Normalization is the other half of a real request (score() alone isn't the whole story)."""
    projections = _synthetic_projections(BATCH_SIZE)

    started = time.perf_counter()
    canonical = normalize_projections(projections)
    elapsed = time.perf_counter() - started

    assert len(canonical) == BATCH_SIZE
    assert elapsed < TIME_BUDGET_SECONDS, f"Normalized {BATCH_SIZE} players in {elapsed:.2f}s, budget was {TIME_BUDGET_SECONDS}s"
    print(f"\n[perf] normalize_projections: {BATCH_SIZE} players in {elapsed * 1000:.1f}ms ({BATCH_SIZE / elapsed:.0f} players/sec)")


@pytest.mark.parametrize("mode", ["standard", "half-ppr", "ppr"])
def test_batch_scoring_scales_linearly_not_quadratically(mode):
    """A regression guard: doubling the batch should not roughly quadruple the time."""
    small = _synthetic_projections(1000, seed=1)
    large = _synthetic_projections(8000, seed=2)

    start_small = time.perf_counter()
    batch_calculate_fantasy_points(small, mode=mode)
    small_elapsed = max(time.perf_counter() - start_small, 1e-6)

    start_large = time.perf_counter()
    batch_calculate_fantasy_points(large, mode=mode)
    large_elapsed = max(time.perf_counter() - start_large, 1e-6)

    ratio = large_elapsed / small_elapsed
    # 8x the players should cost roughly 8x the time for O(n) work; allow
    # generous headroom (up to 25x) before treating it as a real regression.
    assert ratio < 25, f"8x the batch size took {ratio:.1f}x as long for mode={mode!r}; possible non-linear regression"
