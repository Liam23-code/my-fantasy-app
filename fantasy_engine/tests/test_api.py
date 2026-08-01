"""Unit tests for the FastAPI surface (api.main, api.data, api.schemas)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.data import reset_projection_provider, set_projection_provider
from api.main import create_app

LAMAR_ID = "nfl:player:00-0034796"


@pytest.fixture
def client():
    return TestClient(create_app(rate_limit=10_000))


@pytest.fixture(autouse=True)
def _reset_provider():
    yield
    reset_projection_provider()


def test_get_projection_known_player(client):
    response = client.get(f"/player/{LAMAR_ID}/projection")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Lamar Jackson"
    assert body["position"] == "QB"
    assert body["rushing_yards"] > 0


def test_get_projection_unknown_player_returns_404(client):
    response = client.get("/player/nfl:player:does-not-exist/projection")
    assert response.status_code == 404


def test_get_projection_uses_overridden_provider(client):
    set_projection_provider(lambda player_id: {"player_id": player_id, "name": "Injected", "position": "WR", "receiving_yards": 42})
    response = client.get("/player/anything/projection")
    assert response.status_code == 200
    assert response.json()["name"] == "Injected"


def test_score_basic(client):
    response = client.post("/score", json={"projection": {"passing_yards": 250, "passing_tds": 2}, "mode": "ppr"})
    assert response.status_code == 200
    body = response.json()
    assert body["total_points"] == pytest.approx(250 / 25 + 8)
    assert body["mode"] == "ppr"


def test_score_with_custom_rules_overlay(client):
    payload = {
        "projection": {"passing_tds": 2},
        "mode": "ppr",
        "custom_rules": {"multipliers": {"passing_tds": 6.0}},
        "bonuses": False,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    assert response.json()["breakdown"]["passing_tds"] == pytest.approx(12.0)


def test_score_invalid_mode_returns_422(client):
    response = client.post("/score", json={"projection": {"passing_yards": 100}, "mode": "not-a-real-mode"})
    assert response.status_code == 422


def test_score_custom_mode_without_rules_returns_422(client):
    response = client.post("/score", json={"projection": {"passing_yards": 100}, "mode": "custom"})
    assert response.status_code == 422


def test_optimize_basic(client):
    payload = {
        "roster": [{"player_id": "rb1", "name": "RB One", "position": "RB", "slot": "RB"}],
        "week_projections": [{"player_id": "rb1", "name": "RB One", "position": "RB", "rushing_yards": 90}],
        "league_settings": {
            "n_teams": 10,
            "roster_requirements": {"RB": 1, "WR": 0, "TE": 0, "QB": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 0},
            "flex_eligible": [],
        },
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["starters"][0]["name"] == "RB One"
    assert body["total_points"] > 0


def test_optimize_invalid_roster_type_returns_422(client):
    response = client.post("/optimize", json={"roster": "not-a-roster", "league_settings": {}})
    assert response.status_code == 422


def test_optimize_engine_level_validation_error_returns_422(client):
    # Passes the OptimizeRequest schema (roster is a valid dict), but the
    # nested "players" entry fails RosterPlayer's own required-field
    # validation *inside* optimize_lineup, not at the request-schema layer.
    payload = {"roster": {"team_name": "T", "players": [{"position": "RB"}]}, "league_settings": {}}
    response = client.post("/optimize", json=payload)
    assert response.status_code == 422


def test_optimize_rejects_oversized_roster(client):
    oversized = [{"player_id": str(i), "name": f"P{i}", "position": "RB"} for i in range(501)]
    response = client.post("/optimize", json={"roster": oversized, "league_settings": {}})
    assert response.status_code == 422


def test_waiver_basic(client):
    payload = {
        "league_state": {"league_settings": {"n_teams": 1, "roster_requirements": {"RB": 1, "FLEX": 0}, "flex_eligible": []}, "my_roster": []},
        "available_players": [{"player_id": "fa1", "name": "Free Agent", "position": "RB", "rushing_yards": 50}],
        "budget": 100,
    }
    response = client.post("/waiver", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body[0]["name"] == "Free Agent"
    assert body[0]["suggested_faab_bid"] is not None


def test_waiver_engine_level_validation_error_returns_422(client):
    payload = {
        "league_state": {"league_settings": {"roster_requirements": "not-a-dict"}},
        "available_players": [{"player_id": "fa1", "name": "Free Agent", "position": "RB"}],
    }
    response = client.post("/waiver", json=payload)
    assert response.status_code == 422


def test_trade_eval_unresolvable_bare_name_returns_422(client):
    payload = {
        "team_a_players": ["Nonexistent Player"],
        "team_b_players": [{"player_id": "b", "name": "B", "position": "RB", "rushing_yards": 50}],
        "league_settings": {},
    }
    response = client.post("/trade-eval", json=payload)
    assert response.status_code == 422


def test_trade_eval_basic(client):
    payload = {
        "team_a_players": [{"player_id": "a", "name": "Give Guy", "position": "RB", "rushing_yards": 50}],
        "team_b_players": [{"player_id": "b", "name": "Receive Guy", "position": "RB", "rushing_yards": 90}],
        "league_settings": {"n_teams": 12},
        "monte_carlo_iterations": 200,
        "seed": 1,
    }
    response = client.post("/trade-eval", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert {"fair_value", "recommendation", "win_prob_delta", "rationale"}.issubset(body)
    assert body["fair_value"] > 0


def test_trade_eval_rejects_oversized_side(client):
    team_a = [{"player_id": str(i), "name": f"P{i}", "position": "RB"} for i in range(21)]
    payload = {"team_a_players": team_a, "team_b_players": [{"player_id": "b", "name": "B", "position": "RB"}], "league_settings": {}}
    response = client.post("/trade-eval", json=payload)
    assert response.status_code == 422


def test_trade_eval_rejects_excessive_monte_carlo_iterations(client):
    payload = {
        "team_a_players": [{"player_id": "a", "name": "A", "position": "RB", "rushing_yards": 50}],
        "team_b_players": [{"player_id": "b", "name": "B", "position": "RB", "rushing_yards": 50}],
        "league_settings": {},
        "monte_carlo_iterations": 999_999,
    }
    response = client.post("/trade-eval", json=payload)
    assert response.status_code == 422


def test_openapi_schema_documents_all_endpoints(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/player/{player_id}/projection" in paths
    assert "/score" in paths
    assert "/optimize" in paths
    assert "/waiver" in paths
    assert "/trade-eval" in paths


def test_docs_ui_is_served(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_rate_limit_returns_429_after_threshold():
    limited_client = TestClient(create_app(rate_limit=2, rate_limit_window_seconds=60.0))
    assert limited_client.get("/openapi.json").status_code == 200
    assert limited_client.get("/openapi.json").status_code == 200
    third = limited_client.get("/openapi.json")
    assert third.status_code == 429
    assert "Rate limit" in third.json()["detail"]


def test_rate_limit_window_expires_old_hits():
    # A short-but-generous window: too tight and normal request/test overhead
    # alone could exceed it before the "still within the window" assertion.
    limited_client = TestClient(create_app(rate_limit=1, rate_limit_window_seconds=1.0))
    assert limited_client.get("/openapi.json").status_code == 200
    assert limited_client.get("/openapi.json").status_code == 429  # still within the window
    time.sleep(1.1)
    assert limited_client.get("/openapi.json").status_code == 200  # window expired, hit evicted


def test_rate_limit_is_independent_per_app_instance():
    # Two separately-created apps must not share rate-limit state.
    app_one = TestClient(create_app(rate_limit=1, rate_limit_window_seconds=60.0))
    app_two = TestClient(create_app(rate_limit=1, rate_limit_window_seconds=60.0))
    assert app_one.get("/openapi.json").status_code == 200
    assert app_two.get("/openapi.json").status_code == 200  # not blocked by app_one's single hit
