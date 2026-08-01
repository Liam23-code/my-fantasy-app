"""FastAPI surface for the fantasy super-engine.

Usage::

    uvicorn api.main:app --reload
    # then open http://127.0.0.1:8000/docs for interactive OpenAPI docs

``create_app()`` is a factory (rather than a single module-level instance)
so tests -- and any deployment that wants a different rate limit -- can build
an independent app with its own middleware state instead of sharing one
process-wide rate-limit counter.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.data import get_player_projection
from api.schemas import OptimizeRequest, ScoreRequest, ScoreResponse, TradeEvalRequest, WaiverRequest
from fantasy.models import CanonicalProjection
from fantasy.optimizer import optimize_lineup
from fantasy.scoring import calculate_fantasy_points
from fantasy.trade import evaluate_trade
from fantasy.waiver import waiver_recommendations


class RateLimitMiddleware(BaseHTTPMiddleware):
    """A fixed-window request counter per client address.

    In-memory and per-process: fine for a single instance or local/dev use.
    A multi-instance deployment needs a shared store (e.g. Redis) instead --
    swap this middleware for one backed by `fantasy.cache.get_cache("redis")`
    if you scale out horizontally.
    """

    def __init__(self, app: Any, limit: int = 120, window_seconds: float = 60.0):
        super().__init__(app)
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[client_id]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.limit:
            return JSONResponse({"detail": "Rate limit exceeded. Please slow down."}, status_code=429)
        hits.append(now)
        return await call_next(request)


def create_app(rate_limit: int = 120, rate_limit_window_seconds: float = 60.0) -> FastAPI:
    app = FastAPI(
        title="Fantasy Super-Engine API",
        description=(
            "Scoring, lineup optimization, waiver recommendations, and trade analysis "
            "for fantasy football. See the root README's migration guide for wiring "
            "this up against a real projection source."
        ),
        version="0.1.0",
    )
    app.add_middleware(RateLimitMiddleware, limit=rate_limit, window_seconds=rate_limit_window_seconds)

    @app.get("/player/{player_id}/projection", response_model=CanonicalProjection, tags=["projections"])
    def get_projection(player_id: str) -> Any:
        """Look up one player's canonical projection from the configured provider."""
        projection = get_player_projection(player_id)
        if projection is None:
            raise HTTPException(status_code=404, detail=f"No projection found for player_id={player_id!r}.")
        return projection

    @app.post("/score", response_model=ScoreResponse, tags=["scoring"])
    def score(payload: ScoreRequest) -> Any:
        """Score one projection under a scoring mode, with an optional custom-rules overlay."""
        custom_rules = payload.custom_rules.model_dump(exclude_none=True) if payload.custom_rules else None
        try:
            return calculate_fantasy_points(payload.projection, mode=payload.mode, custom_rules=custom_rules, bonuses=payload.bonuses)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/optimize", tags=["optimizer"])
    def optimize(payload: OptimizeRequest) -> Any:
        """Optimize a weekly starting lineup subject to league rules and optional constraints."""
        try:
            return optimize_lineup(payload.roster, payload.week_projections, payload.league_settings, payload.constraints)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/waiver", tags=["waiver"])
    def waiver(payload: WaiverRequest) -> Any:
        """Rank free agents for waiver claims or free-agent pickups."""
        try:
            return waiver_recommendations(payload.league_state, payload.available_players, payload.scoring_mode, payload.budget)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/trade-eval", tags=["trade"])
    def trade_eval(payload: TradeEvalRequest) -> Any:
        """Evaluate a proposed trade with a Monte Carlo rest-of-season simulation."""
        try:
            return evaluate_trade(
                payload.team_a_players,
                payload.team_b_players,
                payload.league_settings,
                payload.projections,
                monte_carlo_iterations=payload.monte_carlo_iterations,
                weeks_remaining=payload.weeks_remaining,
                team_a_roster=payload.team_a_roster,
                team_b_roster=payload.team_b_roster,
                seed=payload.seed,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
