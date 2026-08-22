import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

_FANTASY_ENGINE_ROOT = Path(__file__).resolve().parents[3] / "fantasy_engine"
if _FANTASY_ENGINE_ROOT.is_dir() and str(_FANTASY_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FANTASY_ENGINE_ROOT))

# Fantasy Season Tools: everything after the draft.
#
# Waivers, lineups, trades, and the scoring sandbox -- the in-season half of
# the engine, kept separate from Mock Draft, Saved Teams, and My Team.
#
# All calculations are delegated to the `fantasy` engine package
# (`fantasy_engine/fantasy`); this page only handles presentation.
#
# Deliberately a comment, not a string -- see the note in 25_Fantasy_Draft_Room.py.

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st
from app.fantasy_shared import league_setup
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    run_analysis,
    section_header,
)
from app.style import gold_glow_chart, safe_number, stacked_card_html
from fantasy.assistant import weekly_start_sit_advice
from fantasy.optimizer import optimize_lineup
from fantasy.projections import projection_season_label
from fantasy.scoring import (
    BASE_MULTIPLIERS,
    RECEPTION_MULTIPLIER_BY_MODE,
    batch_calculate_fantasy_points,
    calculate_fantasy_points,
)
from fantasy.trade import evaluate_trade
from fantasy.waiver import waiver_recommendations
from fantasy.weekly_projections import build_weekly_projection, weekly_matchups
from quant import quant_engine as quant
from quant.trade_engine import trade_fairness_score
from quant.waiver_engine import rank_waiver_priority


def _player_key(player: Mapping[str, Any]) -> str:
    """Return a stable join key for player records from different engines."""
    identifier = player.get("player_id") or player.get("id")
    if identifier:
        return str(identifier).strip().casefold()
    return str(player.get("name") or player.get("player_name") or "").strip().casefold()


def _quant_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one row from a Quant Engine metric envelope."""
    result = payload.get("result")
    if isinstance(result, Mapping):
        return dict(result)
    rows = payload.get("results")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return dict(rows[0])
    return dict(payload)


def _quant_by_player(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index metric rows by both id and normalized display name."""
    result: dict[str, dict[str, Any]] = {}
    rows = payload.get("results")
    if not isinstance(rows, list):
        return result
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        for value in (row.get("player_id"), row.get("name")):
            key = str(value or "").strip().casefold()
            if key:
                result[key] = row
    return result


def _quant_matchup_context(player: Mapping[str, Any], week: int) -> dict[str, Any]:
    """Translate the weekly engine's defense scale to Quant's strength scale."""
    matchup = weekly_matchups(dict(player))[week]
    adjustment = safe_number(matchup.get("defensive_adjustment"), 1.0)
    is_bye = str(matchup.get("opponent") or "").upper() == "BYE" or adjustment == 0.0
    # The legacy curve maps a neutral defense to 1.0 and the Quant Engine maps
    # neutral strength to 0.5.  This inverse preserves all known matchup data.
    defensive_strength = 0.5 if is_bye else max(0.0, min(1.0, 0.5 - (adjustment - 1.0) / 0.30))
    return {
        "opponent": matchup.get("opponent") or "TBD",
        "opponent_strength": defensive_strength,
        "is_bye": is_bye,
        "defensive_adjustment": adjustment,
        "has_defense_data": bool(matchup.get("has_defense_data")),
    }


def _quant_weekly_result(player: Mapping[str, Any], week: int, scoring_mode: str) -> dict[str, Any]:
    """Calculate one authoritative Quant-adjusted weekly forecast."""
    context = _quant_matchup_context(player, week)
    source = dict(player)
    # Existing weekly curves already include matchup and volatility effects.
    # Remove them here so Quant performs the adjustment exactly once from the
    # season baseline; the curve is re-attached for downstream consumers.
    source.pop("weekly_projection", None)
    row = _quant_result(
        quant.compute_weekly_matchup_score(
            source,
            week,
            matchup=context,
            scoring_mode=scoring_mode,
        )
    )
    return {
        **row,
        "week": week,
        "opponent": context["opponent"],
        "is_bye": context["is_bye"],
        "defensive_adjustment": context["defensive_adjustment"],
        "has_defense_data": context["has_defense_data"],
        "points": safe_number(row.get("weekly_projected_points", row.get("adjusted_projection"))),
        "confidence": max(0.0, min(1.0, safe_number(row.get("confidence"), 0.5))),
    }


def _quant_weekly_curve(player: Mapping[str, Any], scoring_mode: str) -> dict[int, dict[str, Any]]:
    """Produce the complete 18-week Quant projection and confidence curve."""
    return {week: _quant_weekly_result(player, week, scoring_mode) for week in range(1, 19)}


def _quant_week_pool(players: list[dict[str, Any]], week: int, scoring_mode: str) -> list[dict[str, Any]]:
    """Attach the selected Quant week to a projection pool for start/sit."""
    enriched: list[dict[str, Any]] = []
    for player in players:
        weekly = _quant_weekly_result(player, week, scoring_mode)
        curve = dict(player.get("weekly_projection") or build_weekly_projection(player, scoring_mode))
        curve[week] = {"points": weekly["points"], "confidence": weekly["confidence"]}
        enriched.append(
            {
                **player,
                "weekly_projection": curve,
                "quant_weekly_matchup_score": weekly.get("weekly_matchup_score", 0.0),
                "quant_weekly_confidence": weekly["confidence"],
            }
        )
    return enriched


def _optimizer_proxies(players: list[dict[str, Any]], week: int, scoring_mode: str) -> list[dict[str, Any]]:
    """Express Quant weekly points in the optimizer's exact point carrier."""
    proxies: list[dict[str, Any]] = []
    for player in _quant_week_pool(players, week, scoring_mode):
        value = player["weekly_projection"][week]
        proxies.append(
            {
                "player_id": player.get("player_id") or player.get("id"),
                "name": player.get("name") or player.get("player_name"),
                "position": player.get("position"),
                "team": player.get("team") or player.get("nfl_team"),
                "passing_yards": safe_number(value.get("points")) * 25.0,
                "quant_confidence": value.get("confidence"),
            }
        )
    return proxies


def _quant_waiver_recommendations(
    league_state: dict[str, Any],
    available: list[dict[str, Any]],
    scoring_mode: str,
    budget: float,
) -> list[dict[str, Any]]:
    """Preserve league recommendations and append Quant waiver intelligence."""
    recommendations = waiver_recommendations(league_state, available, scoring_mode, budget=budget)
    quant_rows = rank_waiver_priority(
        available,
        team=league_state.get("my_roster") or [],
        week=league_state.get("current_week"),
        roster_requirements=(league_state.get("league_settings") or {}).get("roster_requirements"),
    )
    by_key: dict[str, dict[str, Any]] = {}
    for row in quant_rows:
        by_key[_player_key(row)] = row
        name_key = str(row.get("name") or "").strip().casefold()
        if name_key:
            by_key[name_key] = row
    enriched = []
    for recommendation in recommendations:
        quant_row = by_key.get(_player_key(recommendation), {})
        enriched.append(
            {
                **recommendation,
                "quant_waiver_score": quant_row.get("waiver_priority_score"),
                "opportunity_score": quant_row.get("opportunity_score"),
                "breakout_probability": quant_row.get("breakout_probability"),
                "usage_trend": quant_row.get("usage_trend_direction"),
                "quant_matchup": quant_row.get("matchup_label"),
                "volatility_profile": quant_row.get("volatility_label"),
            }
        )
    enriched.sort(
        key=lambda row: (
            -safe_number(row.get("quant_waiver_score"), safe_number(row.get("composite_score"))),
            safe_number(row.get("waiver_rank"), 9999.0),
        )
    )
    for rank, row in enumerate(enriched, start=1):
        row["waiver_rank"] = rank
    return enriched

apply_global_theme()
setup = league_setup()

page_header(
    "Weekly Tools",
    "Waivers, lineups, trades, and the scoring sandbox — everything after the draft, in one place.",
    eyebrow="Fantasy · in-season",
)

league_settings = st.session_state.fantasy_league_settings
projections = setup["projections"]
roster = st.session_state.fantasy_roster
target_season = setup["target_season"]

st.caption(
    f"The player pool on this page is the same {projection_season_label(target_season)} pool the draft pages "
    "use. The Quant Engine now supplies weekly projections, confidence, matchup, volatility, scarcity, trend, "
    "waiver-priority, and trade-fairness signals. The lineup optimizer applies league roster rules to those "
    "weekly Quant forecasts, so its total matches the selected week's analytical curve."
)

tool_cards = (
    ("Weekly Projections", "18-week matchup-adjusted points and confidence curves.", "Projection signal"),
    ("Start / Sit", "Optimize the legal lineup for this week's opponent context.", "Lineup signal"),
    ("Waivers", "Find add/drop upgrades against weak roster positions.", "Acquisition signal"),
    ("Trades", "Compare rest-of-season value, fit, and weekly upside.", "Market signal"),
)
for card_index, (title, body, kicker) in enumerate(tool_cards):
    st.markdown(
        stacked_card_html(
            title,
            body,
            kicker=kicker,
            card_id=f"weekly-tool-{card_index}",
            extra_class="weekly-tool-card",
        ),
        unsafe_allow_html=True,
    )

weekly_tab, waivers_tab, lineup_tab, trades_tab, scoring_tab = st.tabs(
    ["Weekly Projections", "Waivers", "Lineup", "Trades", "Scoring"]
)

# ---------------------------------------------------------------------------
# 1. Weekly projections
# ---------------------------------------------------------------------------
with weekly_tab:
    section_header(
        "Weekly projections",
        "See the full 18-week scoring curve, opponent context, and confidence for one player.",
    )
    weekly_players = {
        str(player.get("player_id") or player.get("id") or f"player-{index}"): player
        for index, player in enumerate(projections)
    }
    if weekly_players:
        selected_player_id = st.selectbox(
            "Player",
            list(weekly_players),
            format_func=lambda player_id: (
                f"{weekly_players[player_id].get('name', player_id)} "
                f"({weekly_players[player_id].get('position', '')} · {weekly_players[player_id].get('team', '')})"
            ),
            key="fantasy_weekly_player",
        )
        selected_player = weekly_players[selected_player_id]
        scoring_mode = str(league_settings.get("scoring_mode") or "ppr")
        weekly_projection = _quant_weekly_curve(selected_player, scoring_mode)
        weekly_rows = [
            {
                "Week": week,
                "Projected points": weekly_projection[week]["points"],
                "Confidence": weekly_projection[week]["confidence"],
                "Opponent": weekly_projection[week]["opponent"],
                "Matchup multiplier": weekly_projection[week]["defensive_adjustment"],
                "Matchup score": weekly_projection[week].get("weekly_matchup_score", 0.0),
                "Defense data": "Available" if weekly_projection[week]["has_defense_data"] else "Neutral fallback",
            }
            for week in weekly_projection
        ]
        weekly_frame = pd.DataFrame(weekly_rows).set_index("Week")

        final_projection = quant.compute_final_projection(
            selected_player,
            players=projections,
            scoring_mode=scoring_mode,
        )
        volatility = _quant_result(quant.compute_volatility(selected_player))
        confidence = _quant_result(quant.compute_confidence_scores(selected_player))
        scarcity = _quant_by_player(
            quant.compute_positional_scarcity(
                projections,
                roster_slots=league_settings.get("roster_requirements"),
                teams=int(league_settings.get("n_teams") or 12),
            )
        ).get(_player_key(selected_player), {})
        rarity = _quant_by_player(quant.compute_rarity_tier(projections)).get(_player_key(selected_player), {})

        trend_source = dict(selected_player)
        if not trend_source.get("history"):
            trend_source["history"] = [
                {
                    "week": row["Week"],
                    "points": row["Projected points"],
                    "projection": safe_number(final_projection.get("final_projection")) / 17.0,
                }
                for row in weekly_rows
                if row["Opponent"] != "BYE"
            ]
        trend = _quant_result(quant.compute_trend_lines(trend_source, window=3))
        momentum = _quant_result(quant.compute_momentum(trend_source, window=3))

        selected_rank = selected_player.get(
            "overall_rank",
            selected_player.get("rank", selected_player.get("adp", 41)),
        )
        st.markdown(
            stacked_card_html(
                selected_player.get("name", "Unknown player"),
                f"{selected_player.get('position', '—')} · {selected_player.get('team') or 'FA'}",
                kicker="Selected weekly profile",
                stats={
                    "Quant projection": f"{safe_number(final_projection.get('final_projection')):.1f}",
                    "Confidence": f"{safe_number(confidence.get('confidence_score')):.0%}",
                    "Rarity": rarity.get("rarity_tier", "—"),
                    "Scarcity": f"{safe_number(scarcity.get('scarcity_score')):.1f}",
                },
                rarity_rank=selected_rank,
                extra_class="weekly-player-card",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Weekly scoring curve")
        st.plotly_chart(
            gold_glow_chart(
                weekly_frame["Projected points"].tolist(),
                x=weekly_frame.index.tolist(),
                title="Weekly scoring curve",
                name="Projected points",
                height=370,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
            key="fantasy-weekly-scoring-curve",
        )

        focus_week = int(
            st.number_input(
                "Matchup week",
                min_value=1,
                max_value=18,
                value=1,
                key="fantasy_weekly_focus_week",
            )
        )
        focus = weekly_rows[focus_week - 1]
        point_col, opponent_col, matchup_col, confidence_col = st.columns(4)
        point_col.metric("Projected points", f"{focus['Projected points']:.2f}")
        opponent_col.metric("Opponent", focus["Opponent"])
        matchup_value = focus["Matchup multiplier"]
        matchup_col.metric("Matchup", "Bye" if matchup_value == 0 else f"{matchup_value:.2f}x")
        confidence_col.metric("Confidence", f"{focus['Confidence']:.0%}")

        quant_col, volatility_col, scarcity_col, momentum_col = st.columns(4)
        quant_col.metric("Quant matchup score", f"{focus['Matchup score']:.0f}/100")
        volatility_col.metric(
            "Volatility",
            f"{safe_number(volatility.get('volatility')):.0%}",
            help=str(volatility.get("risk_level") or "Quant weekly risk"),
        )
        scarcity_col.metric("Positional scarcity", f"{safe_number(scarcity.get('scarcity_score')):.1f}/100")
        momentum_col.metric(
            "Momentum",
            f"{safe_number(momentum.get('momentum_score'), 50.0):.0f}/100",
            help=str(momentum.get("direction") or "flat").title(),
        )

        st.markdown("#### Opponent matchups")
        st.dataframe(
            weekly_frame[
                ["Opponent", "Matchup multiplier", "Matchup score", "Defense data", "Projected points"]
            ],
            width="stretch",
            height=330,
        )

        st.markdown("#### Confidence curve")
        st.plotly_chart(
            gold_glow_chart(
                (weekly_frame["Confidence"] * 100.0).tolist(),
                x=weekly_frame.index.tolist(),
                title="Weekly confidence curve",
                name="Confidence",
                height=330,
                y_suffix="%",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
            key="fantasy-weekly-confidence-curve",
        )

        rolling_points = trend.get("rolling_points") or trend.get("trend_line") or []
        trend_pairs = [
            (week, value)
            for week, value in enumerate(rolling_points, start=1)
            if isinstance(value, (int, float))
        ]
        st.markdown("#### Quant trend line")
        if trend_pairs:
            st.plotly_chart(
                gold_glow_chart(
                    [value for _week, value in trend_pairs],
                    x=[week for week, _value in trend_pairs],
                    title=f"Rolling form · {str(trend.get('trend_direction') or trend.get('direction') or 'flat').title()}",
                    name="Three-week rolling points",
                    height=330,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key="fantasy-weekly-quant-trend",
            )
        else:
            st.caption("A trend line will appear after at least three usable weekly observations are available.")
        if all(row["Defense data"] == "Neutral fallback" for row in weekly_rows):
            st.caption(
                "No week-specific defense data is attached to this player pool, so matchup multipliers are "
                "neutral. Add a schedule with opponent defense ranks/ratings to activate matchup adjustments."
            )
    else:
        empty_state(
            "No player projections loaded",
            "Load a projection pool in League & draft setup above to build weekly curves.",
            icon="📈",
        )

# ---------------------------------------------------------------------------
# 2. Waivers
# ---------------------------------------------------------------------------
with waivers_tab:
    section_header(
        "Waiver recommendations", "Rank free agents against your roster needs, with suggested FAAB/auction bids."
    )
    available = st.session_state.fantasy_available
    if available:
        col1, col2 = st.columns(2)
        current_week = col1.number_input("Current week", min_value=1, max_value=18, value=5, key="fantasy_waiver_week")
        budget = col2.number_input(
            "Remaining FAAB / auction budget", min_value=0.0, value=100.0, key="fantasy_waiver_budget"
        )
        if st.button("Get waiver recommendations", type="primary", key="fantasy_run_waivers"):
            league_state = {
                "league_settings": league_settings,
                "my_roster": roster,
                "current_week": int(current_week),
            }
            scoring_mode_override = league_settings.get("scoring_mode")
            recs = run_analysis(
                "waiver recommendations",
                lambda: _quant_waiver_recommendations(
                    league_state,
                    available,
                    str(scoring_mode_override or "ppr"),
                    float(budget),
                ),
            )
            if recs:
                st.session_state["fantasy_waiver_result"] = recs

        recs = st.session_state.get("fantasy_waiver_result")
        if recs:
            bid_col = "suggested_auction_bid" if league_settings.get("is_auction") else "suggested_faab_bid"
            columns_wanted = [
                col
                for col in [
                    "waiver_rank",
                    "name",
                    "position",
                    "quant_waiver_score",
                    "opportunity_score",
                    "breakout_probability",
                    "usage_trend",
                    "quant_matchup",
                    "volatility_profile",
                    "composite_score",
                    "replacement_value",
                    bid_col,
                    "rationale",
                ]
                if col in recs[0]
            ]
            st.dataframe(pd.DataFrame(recs)[columns_wanted], width="stretch", hide_index=True, height=360)
    else:
        empty_state(
            "No free-agent pool loaded",
            "Upload an available-players JSON in League & draft setup above to see waiver recommendations.",
            icon="🔄",
        )

# ---------------------------------------------------------------------------
# 3. Lineup
# ---------------------------------------------------------------------------
with lineup_tab:
    section_header(
        "Lineup optimizer", "Solve for the point-maximizing legal starting lineup, plus per-player start/sit calls."
    )
    if roster:
        roster_names = [player.get("name", "") for player in roster]
        id_by_name = {player.get("name"): player.get("player_id") for player in roster}
        lineup_week = int(
            st.number_input(
                "Lineup week", min_value=1, max_value=18, value=1, key="fantasy_lineup_week"
            )
        )
        solver = st.selectbox("Solver", ["auto", "ilp", "greedy"], key="fantasy_lineup_solver")
        lock_col, bench_col = st.columns(2)
        locked_names = lock_col.multiselect("Lock into starting lineup", roster_names, key="fantasy_locked_players")
        excluded_names = bench_col.multiselect(
            "Force to bench (e.g. inactive)", roster_names, key="fantasy_excluded_players"
        )
        constraints = {
            "solver": solver,
            "locked_player_ids": [id_by_name[name] for name in locked_names if id_by_name.get(name)],
            "excluded_player_ids": [id_by_name[name] for name in excluded_names if id_by_name.get(name)],
        }
        if st.button("Optimize lineup", type="primary", key="fantasy_optimize_lineup"):
            lineup = run_analysis(
                "lineup optimization",
                lambda: optimize_lineup(
                    roster,
                    _optimizer_proxies(
                        projections,
                        lineup_week,
                        str(league_settings.get("scoring_mode") or "ppr"),
                    ),
                    {**league_settings, "scoring_mode": "ppr", "custom_rules": None},
                    constraints,
                ),
            )
            if lineup:
                st.session_state["fantasy_lineup_result"] = lineup

        lineup = st.session_state.get("fantasy_lineup_result")
        if lineup:
            st.metric("Projected total points", lineup["total_points"])
            st.caption(f"Solved with the {lineup['solver']} solver.")
            st.dataframe(
                pd.DataFrame(lineup["starters"])[["slot", "name", "position", "points"]],
                width="stretch",
                hide_index=True,
            )
            for warning in lineup["warnings"]:
                st.warning(warning)
            if lineup["bench"]:
                with st.expander("Bench"):
                    st.dataframe(
                        pd.DataFrame(lineup["bench"])[["name", "position", "points"]],
                        width="stretch",
                        hide_index=True,
                    )

        st.markdown("#### Start / sit advice")
        if st.button("Get start/sit advice", key="fantasy_start_sit"):
            advice = run_analysis(
                "weekly start/sit advice",
                lambda: weekly_start_sit_advice(
                    roster,
                    _quant_week_pool(
                        projections,
                        lineup_week,
                        str(league_settings.get("scoring_mode") or "ppr"),
                    ),
                    lineup_week,
                    league_settings,
                ),
            )
            if advice:
                st.session_state["fantasy_start_sit_result"] = advice

        advice = st.session_state.get("fantasy_start_sit_result")
        if advice:
            st.dataframe(pd.DataFrame(advice), width="stretch", hide_index=True, height=320)
    else:
        empty_state(
            "No roster loaded",
            "Upload a roster JSON in League & draft setup above to optimize a lineup.",
            icon="📋",
        )

# ---------------------------------------------------------------------------
# 4. Trades
# ---------------------------------------------------------------------------
with trades_tab:
    section_header("Trade analyzer", "Monte Carlo simulate a proposed trade's rest-of-season value for both sides.")
    pool_names = sorted(
        ({player.get("name") for player in projections} | {player.get("name") for player in roster}) - {None}
    )
    if pool_names:
        col1, col2 = st.columns(2)
        team_a_gives = col1.multiselect("Team A sends", pool_names, key="fantasy_trade_a_gives")
        team_b_gives = col2.multiselect("Team B sends", pool_names, key="fantasy_trade_b_gives")
        col3, col4, col5 = st.columns(3)
        iterations = col3.number_input(
            "Monte Carlo iterations", min_value=100, max_value=20000, value=2000, step=100, key="fantasy_trade_iterations"
        )
        weeks_remaining = col4.number_input(
            "Weeks remaining", min_value=1, max_value=17, value=10, key="fantasy_trade_weeks"
        )
        trade_seed = col5.number_input("Random seed", min_value=0, value=1, key="fantasy_trade_seed")
        if st.button("Evaluate trade", type="primary", key="fantasy_evaluate_trade"):
            if not team_a_gives or not team_b_gives:
                st.warning("Pick at least one player on each side of the trade.")
            else:
                trade = run_analysis(
                    "trade evaluation",
                    lambda: {
                        **evaluate_trade(
                            team_a_players=team_a_gives,
                            team_b_players=team_b_gives,
                            league_settings=league_settings,
                            projections=projections,
                            monte_carlo_iterations=int(iterations),
                            weeks_remaining=int(weeks_remaining),
                            team_a_roster=roster,
                            seed=int(trade_seed),
                        ),
                        "quant_fairness": trade_fairness_score(
                            [player for player in [*projections, *roster] if player.get("name") in team_a_gives],
                            [player for player in [*projections, *roster] if player.get("name") in team_b_gives],
                            team_a=roster,
                            player_pool=projections,
                            roster_requirements=league_settings.get("roster_requirements"),
                        ),
                    },
                )
                if trade:
                    st.session_state["fantasy_trade_result"] = trade

        trade = st.session_state.get("fantasy_trade_result")
        if trade:
            st.subheader(trade["recommendation"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Fair value (Team A net)", f"{trade['fair_value']:+.1f} pts")
            m2.metric("Team A win-prob delta", f"{trade['win_prob_delta']:+.2%}")
            m3.metric("Team A win probability", f"{trade['team_a_win_probability']:.1%}")
            quant_fairness = trade.get("quant_fairness") or {}
            if quant_fairness:
                q1, q2, q3 = st.columns(3)
                q1.metric("Quant fairness", f"{safe_number(quant_fairness.get('fairness_score')):.0f}/100")
                q2.metric("Quant value edge", f"{safe_number(quant_fairness.get('team_a_net_value')):+.1f}")
                q3.metric("Roster fit", f"{safe_number(quant_fairness.get('team_a_fit_score')):.0f}/100")
                st.caption(f"Quant verdict: {quant_fairness.get('recommendation', 'No verdict available')}.")
            for line in trade["rationale"]:
                st.write(f"- {line}")
    else:
        empty_state(
            "Nothing to trade yet",
            "Load projections and/or a roster in League & draft setup above to evaluate a trade.",
            icon="🤝",
        )

# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------
with scoring_tab:
    section_header(
        "Fantasy scoring", "Score any player's stat line under standard, half-PPR, PPR, or a custom rule set."
    )
    scoring_mode = st.selectbox("Scoring mode", ["ppr", "half-ppr", "standard", "custom"], key="fantasy_scoring_mode")
    custom_rules = None
    if scoring_mode == "custom":
        custom_rules_text = st.text_area(
            "Custom rules — JSON with `multipliers` (points per stat unit) and optional `bonuses`",
            value='{"multipliers": {"receptions": 1.0, "receiving_yards": 0.1}, "bonuses": []}',
            key="fantasy_custom_rules_text",
        )
        try:
            custom_rules = json.loads(custom_rules_text)
        except json.JSONDecodeError as error:
            st.error(f"Invalid JSON: {error}")
    score_ready = scoring_mode != "custom" or custom_rules is not None

    with st.expander("Stat → points mapping for this mode"):
        mapping_rows = [{"Stat": stat, "Points per unit": BASE_MULTIPLIERS[stat]} for stat in BASE_MULTIPLIERS] + [
            {"Stat": "receptions", "Points per unit": RECEPTION_MULTIPLIER_BY_MODE.get(scoring_mode, 0.0)}
        ]
        st.dataframe(pd.DataFrame(mapping_rows), width="stretch", hide_index=True)
        st.caption(
            "Default multipliers shown; a custom rule set above overrides any of these per stat. "
            "Missing stats on a player default to 0 rather than erroring — every category is optional."
        )

    # Keyed by player_id, not name: two players can share a display name, and a
    # name-keyed lookup would silently score the wrong one.
    players_by_id = {player.get("player_id"): player for player in projections if player.get("player_id")}
    if players_by_id:
        chosen_id = st.selectbox(
            "Player",
            list(players_by_id),
            format_func=lambda pid: f"{players_by_id[pid].get('name', pid)} ({players_by_id[pid].get('team', '')})",
            key="fantasy_score_player",
        )
        chosen_player = players_by_id.get(chosen_id)
        if chosen_player and score_ready and st.button("Calculate points", type="primary", key="fantasy_calc_points"):
            scored = run_analysis(
                "scoring",
                lambda: calculate_fantasy_points(chosen_player, mode=scoring_mode, custom_rules=custom_rules),
            )
            if scored:
                quant_projection = quant.compute_final_projection(
                    chosen_player,
                    players=projections,
                    scoring_mode=scoring_mode,
                )
                scored["_authoritative_projection"] = quant_projection.get("final_projection")
                scored["_player_name"] = chosen_player.get("name")
                scored["_projection_basis"] = "Quant Engine weighted ensemble"
                st.session_state["fantasy_single_score"] = scored

        scored = st.session_state.get("fantasy_single_score")
        if scored:
            authoritative = scored.get("_authoritative_projection")
            if authoritative is not None:
                metric_col, note_col = st.columns([1, 2])
                metric_col.metric(
                    f"{scored.get('_player_name', 'Player')} — {target_season} projection", f"{authoritative:,.1f}"
                )
                note_col.caption(
                    f"The stat breakdown below totals {scored['total_points']:.1f}, which is the *prior season's "
                    f"actual stat line* scored under this rule set. The {target_season} projection above is that "
                    "line regressed for availability and scoring rate, then reconciled against market ADP — "
                    f"they are deliberately different numbers. Basis: {scored.get('_projection_basis', 'n/a')}"
                )
            else:
                st.metric("Total points", scored["total_points"])
            breakdown_items = sorted(scored["breakdown"].items(), key=lambda kv: -abs(kv[1]))
            breakdown_frame = pd.DataFrame(breakdown_items, columns=["Stat", "Points"])
            st.plotly_chart(
                gold_glow_chart(
                    breakdown_frame,
                    x="Stat",
                    y="Points",
                    title="Scoring contribution by stat",
                    name="Points",
                    height=340,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
                key="fantasy-scoring-breakdown",
            )
            if scored["bonuses_applied"]:
                st.write("Bonuses applied:")
                st.dataframe(pd.DataFrame(scored["bonuses_applied"]), width="stretch", hide_index=True)

        st.markdown("#### Score the full player pool")
        if score_ready and st.button("Batch score all players", key="fantasy_batch_score"):
            batch = run_analysis(
                "batch scoring",
                lambda: batch_calculate_fantasy_points(projections, mode=scoring_mode, custom_rules=custom_rules),
            )
            if batch:
                st.session_state["fantasy_batch_scores"] = batch

        batch = st.session_state.get("fantasy_batch_scores")
        if batch:
            batch_df = pd.DataFrame(
                [
                    {
                        "name": player.get("name"),
                        "position": player.get("position"),
                        "team": player.get("team"),
                        f"{target_season} projection": player.get("projection"),
                        "stat-line points": score["total_points"],
                    }
                    for player, score in zip(projections, batch, strict=True)
                ]
            ).sort_values("stat-line points", ascending=False)
            st.dataframe(batch_df, width="stretch", hide_index=True, height=320)
    else:
        empty_state("No player projections loaded", "Load a pool in League & draft setup above.", icon="📊")
