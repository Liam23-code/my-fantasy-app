"""Unit tests for fantasy.adapter."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import pytest

from fantasy.adapter import normalize_projection, normalize_projections
from fantasy.models import CanonicalProjection


def test_flat_canonical_dict_passes_through(lamar_jackson_projection):
    result = normalize_projection(lamar_jackson_projection)
    assert result["name"] == "Lamar Jackson"
    assert result["position"] == "QB"
    assert result["rushing_yards"] == pytest.approx(65.0)
    assert result["passing_tds"] == pytest.approx(1.8)
    assert result["drivers"] == ["pace", "red_zone_usage", "injury_risk"]


def test_minimal_legacy_stat_dict_is_aliased_correctly():
    legacy = {
        "player": "Saquon Barkley",
        "player_position": "rb",
        "team": "phi",
        "rush_yards": 125.3,
        "rush_tds": 0.81,
        "rec_yards": 17.4,
        "receptions": 2.1,
        "ints": 0,
    }
    result = normalize_projection(legacy)
    assert result["name"] == "Saquon Barkley"
    assert result["position"] == "RB"
    assert result["team"] == "PHI"
    assert result["rushing_yards"] == pytest.approx(125.3)
    assert result["rushing_tds"] == pytest.approx(0.81)
    assert result["receiving_yards"] == pytest.approx(17.4)
    assert result["receptions"] == pytest.approx(2.1)


def test_project_nfl_player_shaped_output_is_unpacked():
    fake_engine_output = {
        "player": "Josh Allen",
        "player_id": "00-0034857",
        "team": "BUF",
        "opponent": "KC",
        "position": "QB",
        "season": 2026,
        "projection": {
            "passing_yards": 214.2,
            "passing_tds": 1.61,
            "interceptions": 0.35,
            "rushing_yards": 30.5,
            "rushing_tds": 0.69,
            "targets": 0.0,
            "receptions": 0.0,
            "receiving_yards": 0.0,
            "receiving_tds": 0.0,
            "expected_fantasy_points": 21.5,
        },
        "confidence": {"score": 62.0, "low": 15.0, "high": 28.0, "label": "Projected: 21.5 FP (15.0-28.0)"},
        "drivers": ["Averaging 214.2 pass yards...", "Matchup difficulty 50/100 vs KC."],
    }
    result = normalize_projection(fake_engine_output)
    assert result["name"] == "Josh Allen"
    assert result["position"] == "QB"
    assert result["passing_yards"] == pytest.approx(214.2)
    assert result["rushing_yards"] == pytest.approx(30.5)
    assert result["expected_fantasy_points"] == pytest.approx(21.5)
    assert result["floor"] == pytest.approx(15.0)
    assert result["ceiling"] == pytest.approx(28.0)
    assert result["median"] == pytest.approx(21.5)
    assert len(result["drivers"]) == 2


def test_player_object_with_attributes_is_normalized():
    @dataclass
    class FakePlayer:
        name: str
        position: str
        team: str
        rushing_yards: float
        receiving_yards: float

    player = FakePlayer(name="Christian McCaffrey", position="rb", team="sf", rushing_yards=91.2, receiving_yards=32.5)
    result = normalize_projection(player)
    assert result["name"] == "Christian McCaffrey"
    assert result["position"] == "RB"
    assert result["team"] == "SF"
    assert result["rushing_yards"] == pytest.approx(91.2)
    assert result["receiving_yards"] == pytest.approx(32.5)


def test_namedtuple_source_is_normalized_via_asdict():
    FakeRow = namedtuple("FakeRow", ["name", "position", "receiving_yards"])
    result = normalize_projection(FakeRow(name="Puka Nacua", position="WR", receiving_yards=92.0))
    assert result["name"] == "Puka Nacua"
    assert result["receiving_yards"] == pytest.approx(92.0)


def test_pydantic_model_source_is_normalized_via_model_dump():
    model = CanonicalProjection(player_id="1", name="Puka Nacua", position="wr", receiving_yards=92.0, receptions=6.5)
    result = normalize_projection(model)
    assert result["position"] == "WR"
    assert result["receiving_yards"] == pytest.approx(92.0)


def test_bare_id_without_loader_raises_value_error():
    with pytest.raises(ValueError):
        normalize_projection("Travis Kelce")


def test_bare_id_with_loader_resolves_projection():
    def loader(name: str) -> dict:
        assert name == "Travis Kelce"
        return {"name": name, "position": "TE", "receiving_yards": 47.4, "receptions": 5.6}

    result = normalize_projection("Travis Kelce", loader=loader)
    assert result["position"] == "TE"
    assert result["receiving_yards"] == pytest.approx(47.4)


def test_missing_fields_default_sensibly():
    result = normalize_projection({"name": "Nobody", "position": "WR"})
    assert result["passing_yards"] == 0.0
    assert result["receiving_yards"] == 0.0
    assert result["floor"] is None
    assert result["expected_fantasy_points"] is None
    assert result["drivers"] == []


def test_as_model_returns_validated_canonical_projection(lamar_jackson_projection):
    model = normalize_projection(lamar_jackson_projection, as_model=True)
    assert isinstance(model, CanonicalProjection)
    assert model.position == "QB"
    assert model.rushing_yards == pytest.approx(65.0)


def test_unsupported_source_type_raises_type_error():
    with pytest.raises(TypeError):
        normalize_projection(12345.6)  # a bare float is neither an id nor a mapping/object


def test_normalize_projections_batch(named_player_projections):
    results = normalize_projections(named_player_projections)
    assert len(results) == len(named_player_projections)
    assert {r["position"] for r in results} == {"QB", "RB", "WR", "TE"}


def test_numeric_strings_in_source_are_coerced():
    result = normalize_projection({"name": "X", "position": "QB", "passing_yards": "245.0", "season": "2026"})
    assert result["passing_yards"] == pytest.approx(245.0)
    assert result["season"] == 2026
