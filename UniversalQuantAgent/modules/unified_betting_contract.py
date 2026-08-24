"""Unified betting-engine contract: one dispatch interface over NFL, NBA, CFB, and CBB.

"Shared contract, separate code" (see architecture.md / betting_engine.md):
this module contains no betting logic of its own, only routing. Each
sport's underlying implementation is untouched -- calling ``load_odds("NFL")``
here does exactly what calling ``betting.odds_loader.unified_odds()``
directly already did. What this module adds is a single, uniform call
site and a uniform *output* shape across all four sports, so a caller
(the unified UI, a future integration) doesn't need four different
function names or four different result shapes to handle.

A uniform *output* shape does not mean a uniform *input* -- the four
sports' own evaluators need genuinely different real context (NFL needs a
real player pool; NBA needs already-computed matchup-adjusted comparison
rows; CFB and CBB are self-contained). :func:`compute_ev` documents
exactly what each sport needs via ``**context``, rather than hiding that
real difference behind a fake one-size-fits-all signature.
"""
from __future__ import annotations

from typing import Any, Callable

VALID_SPORTS = ("NFL", "NBA", "CFB", "CBB")

#: risk_tier -> a deterministic confidence proxy, for the sports (CFB, CBB)
#: whose evaluate_prop doesn't compute an explicit 0-1 confidence score the
#: way NFL/NBA's do -- a real function of an already-computed real
#: classification, not a fabricated number. NFL/NBA's own confidence
#: fields are used as-is; this is only a fallback.
_RISK_TIER_CONFIDENCE_PROXY = {"low": 0.8, "medium": 0.5, "high": 0.25}


def _check_sport(sport: str) -> None:
    if sport not in VALID_SPORTS:
        raise ValueError(f"sport must be one of {VALID_SPORTS}, got {sport!r}")


def load_odds(sport: str) -> dict[str, Any]:
    """Real default props + game odds for ``sport``, in one uniform shape.

    Returns ``{"props": [...], "games": {matchup_or_game_id: row}}`` for
    every sport -- NFL's own loader keeps games and props in one merged
    object; NBA/CFB/CBB each load them from two separate files. This
    function only normalizes the top-level shape; row-level fields still
    differ by sport (see each sport's own loader module for its schema).
    Never merges an upload -- uploads are handled at the UI layer, which
    already has the uploaded file in hand; this always loads the default.
    """
    _check_sport(sport)
    if sport == "NFL":
        from betting.odds_loader import unified_odds

        odds = unified_odds()
        return {"props": list(odds["player_props"].values()), "games": odds["games"]}
    if sport == "NBA":
        from modules.nba_odds_loader import unified_game_odds
        from modules.nba_props_loader import unified_props

        return {"props": unified_props(), "games": unified_game_odds()["games"]}
    if sport == "CFB":
        from modules.cfb_odds_loader import unified_game_odds
        from modules.cfb_props_loader import unified_props

        return {"props": unified_props(), "games": unified_game_odds()["games"]}
    from modules.cbb_odds_loader import unified_game_odds
    from modules.cbb_props_loader import unified_props

    return {"props": unified_props(), "games": unified_game_odds()["games"]}


def compute_ev(sport: str, props: list[dict[str, Any]], **context: Any) -> list[dict[str, Any]]:
    """Price a list of already-loaded prop rows for ``sport`` -- probability, edge, EV, risk.

    Required ``context`` by sport:

    * ``"NFL"`` -- ``players_by_id`` (``dict[str, dict]``): a real player
      pool, e.g. from ``fantasy.projections.load_forward_projections()``.
    * ``"NBA"`` -- ``comparison_rows`` (``list[dict]``): already-computed
      real matchup-adjusted rows from
      ``modules.props.compare_props``/``modules.recommendations.recommend_props``
      (NBA's props need a real opponent to project against; this contract
      doesn't resolve one on its own).
    * ``"CFB"`` -- none. Each row already carries its own real per-game
      rate as ``"line"`` (see modules.cfb_props_generator).
    * ``"CBB"`` -- ``minutes_by_player`` (``dict[str, float]``, optional):
      real per-player minutes/game, used to widen assumed variance for a
      low-minutes player (see modules.cbb_prop_model).
    """
    _check_sport(sport)
    if sport == "NFL":
        from betting.prop_model import evaluate_props

        players_by_id = context["players_by_id"]
        odds = {"player_props": {f"{row.get('player_id')}:{row.get('market')}": row for row in props}}
        return evaluate_props(players_by_id, odds)
    if sport == "NBA":
        from modules.nba_prop_model import index_props_by_player_and_category, price_aware_evaluations

        comparison_rows = context["comparison_rows"]
        return price_aware_evaluations(comparison_rows, index_props_by_player_and_category(props))
    if sport == "CFB":
        from modules.cfb_prop_model import evaluate_props

        return evaluate_props(props)
    from modules.cbb_prop_model import evaluate_props

    return evaluate_props(props, minutes_by_player=context.get("minutes_by_player"))


def compute_confidence(sport: str, priced_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A uniform confidence/risk view over already-priced rows (see :func:`compute_ev`).

    Returns ``[{"description": str, "confidence": float (0-1), "risk_tier": str}, ...]``,
    same shape for every sport. NFL/NBA's own real confidence scores are
    used directly (rescaled to 0-1 where the source is 0-100); CFB/CBB
    don't compute an explicit confidence score, so their real, already-
    computed ``risk_tier`` is mapped through a fixed, disclosed proxy
    table (see ``_RISK_TIER_CONFIDENCE_PROXY``) rather than left blank.
    """
    _check_sport(sport)
    describe: dict[str, Callable[[dict[str, Any]], str]] = {
        "NFL": lambda row: f"{row.get('name')} {row.get('market')}",
        "NBA": lambda row: f"{row.get('player')} {row.get('category')}",
        "CFB": lambda row: f"{row.get('player_name')} {row.get('category')}",
        "CBB": lambda row: f"{row.get('player_name')} {row.get('category')}",
    }[sport]

    results = []
    for row in priced_rows:
        risk_tier = row.get("risk_tier")
        if sport == "NFL":
            confidence = row.get("confidence")
        elif sport == "NBA":
            score = row.get("confidence_score")
            confidence = round(score / 100.0, 4) if score is not None else _RISK_TIER_CONFIDENCE_PROXY.get(risk_tier)
        else:
            confidence = _RISK_TIER_CONFIDENCE_PROXY.get(risk_tier)
        results.append({"description": describe(row), "confidence": confidence, "risk_tier": risk_tier})
    return results


def build_parlays(sport: str, legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Evaluate a same-sport parlay from already-constructed legs -- uniform output shape.

    ``legs`` come from that sport's own ``make_leg`` (see
    ``betting.parlay_engine.make_leg`` / ``modules.nba_parlay_engine.make_leg`` /
    ``modules.cfb_parlay_engine.make_leg`` / ``modules.cbb_parlay_engine.make_leg``
    -- all four already share one leg shape). For a parlay mixing legs from
    more than one sport, use :func:`modules.unified_parlay_engine.evaluate_cross_sport_parlay`
    directly (its legs are tagged with ``sport`` via ``make_unified_leg`` and
    it is not sport-dispatched the way this function is).
    """
    _check_sport(sport)
    if sport == "NFL":
        from betting.parlay_engine import evaluate_parlay as evaluate

        return evaluate(legs, stake=stake)
    if sport == "NBA":
        from modules.nba_parlay_engine import evaluate_parlay as evaluate

        return evaluate(legs, stake=stake)
    if sport == "CFB":
        from modules.cfb_parlay_engine import evaluate_parlay as evaluate

        return evaluate(legs, stake=stake)
    from modules.cbb_parlay_engine import evaluate_parlay as evaluate

    return evaluate(legs, stake=stake)
