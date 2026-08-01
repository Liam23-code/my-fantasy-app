"""Validation matrix: assert sane, non-zero fantasy output for six named players.

Mirrors the validation matrix used in the sibling NFL projection engine
(``UniversalQuantAgent/tests/test_nfl_engine.py``) but at the fantasy-scoring
layer: given each player's canonical projection, ``calculate_fantasy_points``
must produce non-zero rushing/receiving/TD contributions exactly where a real
player of that profile would have them, and zero everywhere else.

Run directly::

    python examples/validate_matrix.py

Exits 0 and prints "ALL CHECKS PASSED" on success; exits 1 and prints which
assertion failed otherwise, so it's usable as a CI smoke test too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fantasy.scoring import calculate_fantasy_points  # noqa: E402

EXAMPLES_DIR = Path(__file__).parent


def _load_named_players() -> dict[str, dict]:
    players = json.loads((EXAMPLES_DIR / "sample_projections.json").read_text(encoding="utf-8"))
    named = {"Lamar Jackson", "Josh Allen", "Saquon Barkley", "Christian McCaffrey", "Puka Nacua", "Travis Kelce"}
    return {p["name"]: p for p in players if p["name"] in named}


def main() -> int:
    players = _load_named_players()
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    lamar = calculate_fantasy_points(players["Lamar Jackson"], mode="ppr")
    check(lamar["breakdown"]["rushing_yards"] > 0, "Lamar Jackson: rushing_yards contribution should be > 0")
    check(lamar["breakdown"]["passing_tds"] > 0, "Lamar Jackson: passing_tds contribution should be > 0")
    check(lamar["breakdown"]["receiving_yards"] == 0, "Lamar Jackson: receiving_yards contribution should be 0 (QB)")

    allen = calculate_fantasy_points(players["Josh Allen"], mode="ppr")
    check(allen["breakdown"]["passing_yards"] > 0, "Josh Allen: passing_yards contribution should be > 0")
    check(allen["breakdown"]["passing_tds"] > 0, "Josh Allen: passing_tds contribution should be > 0")
    check(allen["breakdown"]["rushing_yards"] > 0, "Josh Allen: rushing_yards contribution should be > 0")

    barkley = calculate_fantasy_points(players["Saquon Barkley"], mode="ppr")
    check(barkley["breakdown"]["rushing_yards"] > 0, "Saquon Barkley: rushing_yards contribution should be > 0")
    check(barkley["breakdown"]["receptions"] > 0, "Saquon Barkley: receptions contribution should be > 0")
    check(barkley["breakdown"]["receiving_yards"] > 0, "Saquon Barkley: receiving_yards contribution should be > 0")
    check(barkley["total_points"] > 15, f"Saquon Barkley: fantasy_points should be realistic RB1 range, got {barkley['total_points']}")

    cmc = calculate_fantasy_points(players["Christian McCaffrey"], mode="ppr")
    check(cmc["breakdown"]["rushing_yards"] > 0, "Christian McCaffrey: rushing_yards contribution should be > 0")
    check(cmc["breakdown"]["receiving_yards"] > 0, "Christian McCaffrey: receiving_yards contribution should be > 0")

    nacua = calculate_fantasy_points(players["Puka Nacua"], mode="ppr")
    check(nacua["breakdown"]["receiving_yards"] > 0, "Puka Nacua: receiving_yards contribution should be > 0")
    check(nacua["breakdown"]["rushing_yards"] == 0, "Puka Nacua: rushing_yards contribution should be 0 (pure WR)")

    kelce = calculate_fantasy_points(players["Travis Kelce"], mode="ppr")
    check(kelce["breakdown"]["receiving_yards"] > 0, "Travis Kelce: receiving_yards contribution should be > 0")
    check(kelce["breakdown"]["rushing_yards"] == 0, "Travis Kelce: rushing_yards contribution should be 0 (TE)")

    if failures:
        print("VALIDATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("ALL CHECKS PASSED")
    for name, result in [
        ("Lamar Jackson", lamar), ("Josh Allen", allen), ("Saquon Barkley", barkley),
        ("Christian McCaffrey", cmc), ("Puka Nacua", nacua), ("Travis Kelce", kelce),
    ]:
        print(f"  {name:<22} {result['total_points']:>6.2f} pts (ppr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
