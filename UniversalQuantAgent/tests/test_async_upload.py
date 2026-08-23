"""Unit tests for concurrent upload parsing."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.async_upload import parse_uploads_concurrently


class ParseUploadsConcurrentlyTests(unittest.TestCase):
    def test_empty_tasks_returns_empty_dict(self):
        self.assertEqual(parse_uploads_concurrently({}), {})

    def test_single_task_runs_directly(self):
        result = parse_uploads_concurrently({"props": lambda: [1, 2, 3]})
        self.assertEqual(result, {"props": [1, 2, 3]})

    def test_multiple_tasks_all_complete_and_are_keyed_by_name(self):
        result = parse_uploads_concurrently({"props": lambda: "props-result", "odds": lambda: "odds-result"})
        self.assertEqual(result, {"props": "props-result", "odds": "odds-result"})

    def test_multiple_tasks_actually_run_concurrently_not_serially(self):
        def slow():
            time.sleep(0.05)
            return "done"

        started = time.monotonic()
        result = parse_uploads_concurrently({"a": slow, "b": slow, "c": slow})
        elapsed = time.monotonic() - started
        self.assertEqual(result, {"a": "done", "b": "done", "c": "done"})
        # 3 x 0.05s serial would be ~0.15s; concurrent should clear well under that.
        self.assertLess(elapsed, 0.12)

    def test_a_failing_task_raises_to_the_caller(self):
        def failing():
            raise ValueError("bad upload")

        with self.assertRaises(ValueError):
            parse_uploads_concurrently({"props": failing, "odds": lambda: "ok"})


if __name__ == "__main__":
    unittest.main()
