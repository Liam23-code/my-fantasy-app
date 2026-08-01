"""Unit tests for fantasy.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fantasy.models import CanonicalProjection, LeagueSettings, Roster, RosterPlayer, RosterRequirements


def test_canonical_projection_defaults_and_coercion():
    projection = CanonicalProjection(player_id=123, name="X", position="wr", team="la")
    assert projection.player_id == "123"
    assert projection.position == "WR"
    assert projection.team == "LA"
    assert projection.passing_yards == 0.0
    assert projection.drivers == []


def test_canonical_projection_ignores_unknown_extra_fields():
    projection = CanonicalProjection(player_id="1", name="X", position="RB", some_future_stat=999)
    assert not hasattr(projection, "some_future_stat")


def test_canonical_projection_requires_player_id_and_name():
    with pytest.raises(ValidationError):
        CanonicalProjection(position="WR")


def test_roster_requirements_starting_slots_excludes_bench_and_ir():
    requirements = RosterRequirements(BENCH=7, IR=2, TAXI=1)
    slots = requirements.starting_slots()
    assert "BENCH" not in slots
    assert "IR" not in slots
    assert "TAXI" not in slots
    assert slots["QB"] == 1
    assert slots["FLEX"] == 1


def test_league_settings_defaults_and_flex_uppercasing():
    settings = LeagueSettings(flex_eligible=["rb", "wr", "te"])
    assert settings.flex_eligible == ["RB", "WR", "TE"]
    assert settings.n_teams == 12
    assert settings.roster_requirements.QB == 1


def test_roster_player_uppercases_position_and_team():
    player = RosterPlayer(player_id="1", name="X", position="rb", nfl_team="phi", slot="flex")
    assert player.position == "RB"
    assert player.nfl_team == "PHI"
    assert player.slot == "FLEX"


def test_roster_starters_bench_and_position_filters():
    roster = Roster(
        team_name="Team A",
        players=[
            RosterPlayer(player_id="1", name="QB1", position="QB", slot="QB"),
            RosterPlayer(player_id="2", name="RB1", position="RB", slot="RB"),
            RosterPlayer(player_id="3", name="RB2", position="RB", slot="BENCH"),
            RosterPlayer(player_id="4", name="Hurt", position="WR", slot="IR"),
        ],
    )
    assert {p.name for p in roster.starters()} == {"QB1", "RB1"}
    assert {p.name for p in roster.bench()} == {"RB2"}
    assert {p.name for p in roster.by_position("RB")} == {"RB1", "RB2"}


def test_roster_is_active_respects_injury_status_and_slot():
    healthy = RosterPlayer(player_id="1", name="A", position="RB", slot="RB")
    out = RosterPlayer(player_id="2", name="B", position="RB", slot="BENCH", injury_status="OUT")
    on_ir = RosterPlayer(player_id="3", name="C", position="WR", slot="IR")
    roster = Roster(team_name="T", players=[healthy, out, on_ir])
    assert roster.is_active(healthy) is True
    assert roster.is_active(out) is False
    assert roster.is_active(on_ir) is False
