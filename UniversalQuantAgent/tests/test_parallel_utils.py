"""Unit tests for the bounded thread-pool helpers used for I/O-bound fetches and EV math."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.parallel_utils import parallel_ev_map, parallel_map


class ParallelMapTests(unittest.TestCase):
    def test_preserves_input_order_not_completion_order(self):
        def slow_for_first(x):
            if x == 1:
                time.sleep(0.05)  # first item finishes last
            return x * 10

        result = parallel_map(slow_for_first, [1, 2, 3], max_workers=4)
        self.assertEqual(result, [10, 20, 30])

    def test_a_single_failing_item_is_skipped_not_fatal(self):
        def maybe_fail(x):
            if x == 2:
                raise RuntimeError("simulated failure")
            return x * 10

        result = parallel_map(maybe_fail, [1, 2, 3], max_workers=4)
        self.assertEqual(result, [10, 30])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(parallel_map(lambda x: x, []), [])

    def test_single_item_does_not_spin_up_a_thread_pool(self):
        calls = []

        def record(x):
            calls.append(x)
            return x

        result = parallel_map(record, [42], max_workers=8)
        self.assertEqual(result, [42])
        self.assertEqual(calls, [42])

    def test_max_workers_of_one_runs_serially(self):
        order = []

        def record(x):
            order.append(x)
            return x

        parallel_map(record, [1, 2, 3], max_workers=1)
        self.assertEqual(order, [1, 2, 3])

    def test_genuine_concurrency_is_faster_than_serial_for_io_bound_work(self):
        def blocking_call(x):
            time.sleep(0.05)
            return x

        items = list(range(6))
        started = time.monotonic()
        parallel_map(blocking_call, items, max_workers=6)
        elapsed = time.monotonic() - started
        # 6 x 0.05s serial would be ~0.3s; concurrent should comfortably clear well under that.
        self.assertLess(elapsed, 0.2)


class ParallelEvMapTests(unittest.TestCase):
    def test_applies_pure_function_to_every_row(self):
        rows = [{"line": 10.0}, {"line": 20.0}]
        result = parallel_ev_map(lambda row: row["line"] * 2, rows)
        self.assertEqual(sorted(result), [20.0, 40.0])


if __name__ == "__main__":
    unittest.main()
