"""End-to-end tour of the fantasy super-engine using the bundled sample data.

Run directly::

    python examples/quickstart.py

Touches every major subsystem in order: scoring, the projection adapter,
draft ranking + cheat sheets + mock draft, lineup optimization + start/sit,
waiver recommendations, trade analysis, and tiering + CSV export. Nothing
here talks to a network -- it's all local JSON, so it also doubles as a
smoke test you can read top to bottom.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fantasy.draft import generate_cheatsheet, rank_players_for_draft, simulate_draft, suggest_picks  # noqa: E402
from fantasy.optimizer import optimize_lineup, start_sit_advice  # noqa: E402
from fantasy.scoring import calculate_fantasy_points  # noqa: E402
from fantasy.tiering import export_cheatsheet_csv, generate_printable_cheatsheet, tier_players  # noqa: E402
from fantasy.trade import evaluate_trade  # noqa: E402
from fantasy.waiver import waiver_recommendations  # noqa: E402

EXAMPLES_DIR = Path(__file__).parent


def _load(name: str):
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    projections = _load("sample_projections.json")
    league_settings = _load("sample_league_settings.json")
    roster = _load("sample_roster.json")
    available = _load("sample_available_players.json")

    section("1. Scoring a single projection")
    lamar = next(p for p in projections if p["name"] == "Lamar Jackson")
    score = calculate_fantasy_points(lamar, mode="ppr")
    print(f"Lamar Jackson: {score['total_points']} pts ({', '.join(f'{k}={v:+.1f}' for k, v in score['breakdown'].items() if v)})")

    section("2. Draft assistant: VOR ranking + cheat sheet")
    ranked = rank_players_for_draft(projections, league_settings)
    for player in ranked[:5]:
        print(f"  #{player['overall_rank']:>3} {player['name']:<22} {player['position']:<3} VOR {player['vor']:+.1f}")
    cheatsheet = generate_cheatsheet(projections, league_settings, top_n=50)
    print(f"Cheat sheet: {len(cheatsheet)} players")

    section("3. Suggest a pick mid-draft")
    draft_state = {"league_settings": league_settings, "my_roster": [{"position": "RB"}, {"position": "RB"}]}
    pick = suggest_picks(draft_state, ranked[:30], {"risk_tolerance": "balanced"})
    print(f"Best available pick: {pick['best_pick']['name']} ({pick['best_pick']['position']})")

    section("4. Simulate a full mock draft")
    mock_draft = simulate_draft(projections, league_settings, rounds=3, seed=42)
    print(f"Simulated {mock_draft['rounds']} rounds across {mock_draft['n_teams']} teams; first pick: {mock_draft['picks'][0]['name']}")

    section("5. Optimize this week's lineup")
    lineup = optimize_lineup(roster, projections, league_settings)
    print(f"Optimized lineup: {lineup['total_points']} pts ({lineup['solver']} solver)")
    for starter in lineup["starters"]:
        print(f"  {starter['slot']:<6} {starter['name']:<22} {starter['points']:>5.1f} pts")

    section("6. Start/sit advice")
    advice = start_sit_advice(roster, projections, league_settings)
    for entry in advice[:3]:
        print(f"  {entry['start_or_bench']:<6} {entry['player']:<22} {entry['reason']}")

    section("7. Waiver-wire recommendations")
    league_state = {"league_settings": league_settings, "my_roster": roster, "current_week": 5}
    waivers = waiver_recommendations(league_state, available, "ppr", budget=100)
    for candidate in waivers[:3]:
        print(f"  {candidate['name']:<22} composite {candidate['composite_score']:>5.1f}  FAAB bid ${candidate['suggested_faab_bid']}")

    section("8. Trade analysis")
    trade = evaluate_trade(
        team_a_players=["Saquon Barkley"],
        team_b_players=["Puka Nacua"],
        league_settings=league_settings,
        projections=projections,
        monte_carlo_iterations=2000,
        seed=1,
    )
    print(f"{trade['recommendation']} (fair_value={trade['fair_value']:+.1f}, win_prob_delta={trade['win_prob_delta']:+.2%})")

    section("9. Tiering and cheat sheet export")
    tiered = tier_players(ranked, max_tiers=5)
    print(generate_printable_cheatsheet(tiered[:15]))
    csv_path = EXAMPLES_DIR / "cheatsheet_output.csv"
    export_cheatsheet_csv(tiered, file_path=str(csv_path))
    print(f"CSV cheat sheet written to {csv_path}")


if __name__ == "__main__":
    main()
