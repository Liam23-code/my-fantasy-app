"""Command-line interface for the fantasy super-engine.

Usage::

    fantasy update-week --week 1 --projections-file projections.json
    fantasy run-draft-sim --seed 42 --projections-file projections.json
    fantasy score --projection-file lamar.json --mode ppr

Every command reads plain JSON files so it's trivially scriptable and
mockable in tests -- there's no hidden network call anywhere in this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from fantasy.draft import simulate_draft
from fantasy.pipeline import update_weekly
from fantasy.scoring import calculate_fantasy_points


def _load_json(path: str | Path | None, default: Any = None) -> Any:
    if path is None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_json(data: Any, path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


@click.group()
def main() -> None:
    """Fantasy Super-Engine command-line interface."""


@main.command("update-week")
@click.option("--week", type=int, required=True, help="Week number to process.")
@click.option("--projections-file", type=click.Path(exists=True), required=True, help="JSON list of this week's projections.")
@click.option("--settings-file", type=click.Path(exists=True), default=None, help="JSON league settings (defaults applied if omitted).")
@click.option("--roster-file", type=click.Path(exists=True), default=None, help="JSON list of your roster's players.")
@click.option("--available-file", type=click.Path(exists=True), default=None, help="JSON list of available free agents.")
@click.option("--snapshot-dir", type=click.Path(), default="data/snapshots", help="Directory for weekly snapshot files.")
@click.option("--output", type=click.Path(), default=None, help="Optional path to write the full JSON result.")
def update_week(
    week: int,
    projections_file: str,
    settings_file: str | None,
    roster_file: str | None,
    available_file: str | None,
    snapshot_dir: str,
    output: str | None,
) -> None:
    """Recompute rankings, lineup, and waiver targets for one week."""
    projections = _load_json(projections_file, [])
    settings = _load_json(settings_file, {})
    roster = _load_json(roster_file, None)
    available = _load_json(available_file, None)

    result = update_weekly(
        projection_source=lambda: projections,
        league_settings=settings,
        week=week,
        my_roster=roster,
        available_players_source=(lambda: available) if available is not None else None,
        snapshot_dir=snapshot_dir,
    )

    click.echo(f"Week {week}: scored {len(result['ranked_players'])} players.")
    top = result["ranked_players"][:5]
    for player in top:
        click.echo(f"  #{player['overall_rank']:>3} {player['name']:<24} {player['position']:<3} {player['points']:>6.1f} pts (VOR {player['vor']:+.1f})")
    if "lineup" in result:
        click.echo(f"Optimized lineup: {result['lineup']['total_points']:.1f} projected points ({result['lineup']['solver']} solver).")
    if "waiver_recommendations" in result and result["waiver_recommendations"]:
        best = result["waiver_recommendations"][0]
        click.echo(f"Top waiver target: {best['name']} ({best['position']}), composite score {best['composite_score']:.1f}.")
    if result["movers"]:
        click.echo(f"Biggest mover vs last week: {result['movers'][0]['name']} ({result['movers'][0]['delta']:+.1f} pts).")
    click.echo(f"Snapshot written to {result['snapshot_path']}")
    _dump_json(result, output)


@main.command("run-draft-sim")
@click.option("--seed", type=int, default=None, help="Random seed for reproducible mock drafts.")
@click.option("--projections-file", type=click.Path(exists=True), required=True, help="JSON list of draftable player projections.")
@click.option("--settings-file", type=click.Path(exists=True), default=None, help="JSON league settings (defaults applied if omitted).")
@click.option("--rounds", type=int, default=None, help="Draft rounds (defaults to one team's full roster size).")
@click.option("--output", type=click.Path(), default=None, help="Optional path to write the full JSON result.")
def run_draft_sim(seed: int | None, projections_file: str, settings_file: str | None, rounds: int | None, output: str | None) -> None:
    """Simulate a snake mock draft."""
    projections = _load_json(projections_file, [])
    settings = _load_json(settings_file, {})

    result = simulate_draft(projections, settings, rounds=rounds, seed=seed)
    click.echo(f"Simulated {result['rounds']}-round draft across {result['n_teams']} teams (seed={result['seed']}).")
    for pick in result["picks"][:10]:
        click.echo(f"  Pick {pick['overall_pick']:>3} (R{pick['round']}) {pick['team']:<8} -> {pick['name']} ({pick['position']})")
    if len(result["picks"]) > 10:
        click.echo(f"  ... and {len(result['picks']) - 10} more picks.")
    _dump_json(result, output)


@main.command("score")
@click.option("--projection-file", type=click.Path(exists=True), required=True, help="JSON projection dict to score.")
@click.option("--mode", type=str, default="ppr", help="standard, half-ppr, ppr, or custom.")
@click.option("--custom-rules-file", type=click.Path(exists=True), default=None, help="JSON custom scoring rules (required for mode=custom).")
@click.option("--output", type=click.Path(), default=None, help="Optional path to write the full JSON result.")
def score(projection_file: str, mode: str, custom_rules_file: str | None, output: str | None) -> None:
    """Score a single projection under one scoring mode."""
    projection = _load_json(projection_file, {})
    custom_rules = _load_json(custom_rules_file, None)
    result = calculate_fantasy_points(projection, mode=mode, custom_rules=custom_rules)
    click.echo(f"Total: {result['total_points']:.2f} points ({mode}).")
    for stat, points in result["breakdown"].items():
        if points:
            click.echo(f"  {stat}: {points:+.2f}")
    for bonus in result["bonuses_applied"]:
        click.echo(f"  bonus: +{bonus['points']} for {bonus['stat']} >= {bonus['threshold']}")
    _dump_json(result, output)


if __name__ == "__main__":  # pragma: no cover
    main()
