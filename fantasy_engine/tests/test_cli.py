"""Unit tests for fantasy.cli."""

from __future__ import annotations

import json

from click.testing import CliRunner

from fantasy.cli import main

SETTINGS = {
    "n_teams": 4,
    "scoring_mode": "ppr",
    "roster_requirements": {"RB": 1, "WR": 1, "FLEX": 0, "QB": 0, "TE": 0, "DST": 0, "K": 0, "BENCH": 1},
    "flex_eligible": [],
}

PROJECTIONS = [
    {"player_id": "rb1", "name": "RB One", "position": "RB", "rushing_yards": 90},
    {"player_id": "wr1", "name": "WR One", "position": "WR", "receiving_yards": 60},
]


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_main_help_exits_cleanly():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "update-week" in result.output
    assert "run-draft-sim" in result.output


def test_update_week_basic_run(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    settings_file = _write_json(tmp_path, "settings.json", SETTINGS)
    result = CliRunner().invoke(
        main,
        [
            "update-week",
            "--week",
            "1",
            "--projections-file",
            projections_file,
            "--settings-file",
            settings_file,
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Week 1: scored 2 players" in result.output
    assert (tmp_path / "snapshots" / "week_01.json").exists()


def test_update_week_with_roster_shows_lineup(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    settings_file = _write_json(tmp_path, "settings.json", SETTINGS)
    roster_file = _write_json(
        tmp_path,
        "roster.json",
        [
            {"player_id": "rb1", "name": "RB One", "position": "RB", "slot": "RB"},
            {"player_id": "wr1", "name": "WR One", "position": "WR", "slot": "WR"},
        ],
    )
    result = CliRunner().invoke(
        main,
        [
            "update-week",
            "--week",
            "1",
            "--projections-file",
            projections_file,
            "--settings-file",
            settings_file,
            "--roster-file",
            roster_file,
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Optimized lineup" in result.output


def test_update_week_with_available_players_shows_waiver_target(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    settings_file = _write_json(tmp_path, "settings.json", SETTINGS)
    available_file = _write_json(
        tmp_path,
        "available.json",
        [
            {"player_id": "fa1", "name": "Free Agent", "position": "WR", "receiving_yards": 40},
        ],
    )
    result = CliRunner().invoke(
        main,
        [
            "update-week",
            "--week",
            "1",
            "--projections-file",
            projections_file,
            "--settings-file",
            settings_file,
            "--available-file",
            available_file,
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Top waiver target: Free Agent" in result.output


def test_update_week_writes_output_json(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    output_file = tmp_path / "result.json"
    result = CliRunner().invoke(
        main,
        [
            "update-week",
            "--week",
            "1",
            "--projections-file",
            projections_file,
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--output",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert "ranked_players" in payload


def test_update_week_reports_movers_on_second_week(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    snapshot_dir = str(tmp_path / "snapshots")
    CliRunner().invoke(main, ["update-week", "--week", "1", "--projections-file", projections_file, "--snapshot-dir", snapshot_dir])
    week2_projections = _write_json(
        tmp_path,
        "week2.json",
        [
            {"player_id": "rb1", "name": "RB One", "position": "RB", "rushing_yards": 150},
            {"player_id": "wr1", "name": "WR One", "position": "WR", "receiving_yards": 60},
        ],
    )
    result = CliRunner().invoke(main, ["update-week", "--week", "2", "--projections-file", week2_projections, "--snapshot-dir", snapshot_dir])
    assert result.exit_code == 0, result.output
    assert "Biggest mover vs last week: RB One" in result.output


def test_run_draft_sim_basic(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    settings_file = _write_json(tmp_path, "settings.json", SETTINGS)
    result = CliRunner().invoke(
        main,
        ["run-draft-sim", "--seed", "42", "--projections-file", projections_file, "--settings-file", settings_file, "--rounds", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Simulated 1-round draft across 4 teams (seed=42)" in result.output


def test_run_draft_sim_writes_output_json(tmp_path):
    projections_file = _write_json(tmp_path, "projections.json", PROJECTIONS)
    output_file = tmp_path / "draft.json"
    result = CliRunner().invoke(
        main,
        ["run-draft-sim", "--seed", "1", "--projections-file", projections_file, "--rounds", "1", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert "picks" in payload


def test_run_draft_sim_shows_truncation_notice_for_long_drafts(tmp_path):
    big_pool = [{"player_id": f"p{i}", "name": f"P{i}", "position": "RB", "rushing_yards": 100 - i} for i in range(30)]
    projections_file = _write_json(tmp_path, "big.json", big_pool)
    result = CliRunner().invoke(
        main,
        ["run-draft-sim", "--seed", "1", "--projections-file", projections_file, "--rounds", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "more picks" in result.output


def test_score_command_basic(tmp_path):
    projection_file = _write_json(tmp_path, "proj.json", {"passing_yards": 250, "passing_tds": 2})
    result = CliRunner().invoke(main, ["score", "--projection-file", projection_file, "--mode", "ppr"])
    assert result.exit_code == 0, result.output
    assert "Total:" in result.output
    assert "passing_yards" in result.output


def test_score_command_with_bonus_shown(tmp_path):
    projection_file = _write_json(tmp_path, "proj.json", {"passing_yards": 310})
    result = CliRunner().invoke(main, ["score", "--projection-file", projection_file])
    assert result.exit_code == 0, result.output
    assert "bonus:" in result.output


def test_score_command_custom_mode_with_rules_file(tmp_path):
    projection_file = _write_json(tmp_path, "proj.json", {"passing_yards": 100})
    rules_file = _write_json(tmp_path, "rules.json", {"multipliers": {"passing_yards": 0.1}})
    result = CliRunner().invoke(main, ["score", "--projection-file", projection_file, "--mode", "custom", "--custom-rules-file", rules_file])
    assert result.exit_code == 0, result.output
    assert "Total: 10.00" in result.output


def test_score_command_custom_mode_without_rules_fails():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("proj.json", "w") as handle:
            json.dump({"passing_yards": 100}, handle)
        result = runner.invoke(main, ["score", "--projection-file", "proj.json", "--mode", "custom"])
        assert result.exit_code != 0
