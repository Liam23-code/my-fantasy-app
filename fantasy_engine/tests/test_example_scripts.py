"""Smoke-test the example scripts themselves by actually running them.

Nothing else in the test suite imports examples/quickstart.py or
examples/validate_matrix.py, so without this they could silently rot as the
rest of the package changes underneath them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_validate_matrix_script_passes():
    result = _run("validate_matrix.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL CHECKS PASSED" in result.stdout


def test_quickstart_script_runs_end_to_end():
    result = _run("quickstart.py")
    assert result.returncode == 0, result.stdout + result.stderr
    for marker in ["Draft assistant", "Optimize this week's lineup", "Waiver-wire recommendations", "Trade analysis", "Tiering and cheat sheet export"]:
        assert marker in result.stdout
    (EXAMPLES_DIR / "cheatsheet_output.csv").unlink(missing_ok=True)
