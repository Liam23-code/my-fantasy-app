"""Unit tests for betting.parallel_utils -- the canonical implementation NBA re-exports."""
from __future__ import annotations

import time

from betting.parallel_utils import parallel_ev_map, parallel_map


def test_preserves_input_order_not_completion_order():
    def slow_for_first(x):
        if x == 1:
            time.sleep(0.05)
        return x * 10

    assert parallel_map(slow_for_first, [1, 2, 3], max_workers=4) == [10, 20, 30]


def test_a_single_failing_item_is_skipped_not_fatal():
    def maybe_fail(x):
        if x == 2:
            raise RuntimeError("simulated failure")
        return x * 10

    assert parallel_map(maybe_fail, [1, 2, 3], max_workers=4) == [10, 30]


def test_empty_input_returns_empty_list():
    assert parallel_map(lambda x: x, []) == []


def test_parallel_ev_map_applies_pure_function_to_every_row():
    rows = [{"line": 10.0}, {"line": 20.0}]
    assert sorted(parallel_ev_map(lambda row: row["line"] * 2, rows)) == [20.0, 40.0]
