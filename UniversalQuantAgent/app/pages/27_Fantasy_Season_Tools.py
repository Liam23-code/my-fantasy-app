from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

# Fantasy Season Tools: everything after the draft.
#
# One of three Fantasy pages. Waivers, lineups, trades, and the scoring
# sandbox -- the in-season half of the engine, kept off the two draft pages so
# neither of them has to carry it.
#
# All calculations are delegated to the `fantasy` engine package
# (`fantasy_engine/fantasy`); this page only handles presentation.
#
# Deliberately a comment, not a string -- see the note in 25_Fantasy_Draft_Room.py.

import json

import pandas as pd
import streamlit as st

from app.page_runtime import apply_global_theme, empty_state, page_header, run_analysis, section_header
from app.fantasy_shared import league_setup

from fantasy.optimizer import optimize_lineup, start_sit_advice
from fantasy.projections import projection_season_label
from fantasy.scoring import (
    BASE_MULTIPLIERS,
    RECEPTION_MULTIPLIER_BY_MODE,
    batch_calculate_fantasy_points,
    calculate_fantasy_points,
)
from fantasy.trade import evaluate_trade
from fantasy.waiver import waiver_recommendations

apply_global_theme()
setup = league_setup()

page_header(
    "Season Tools",
    "Waivers, lineups, trades, and the scoring sandbox — everything after the draft, in one place.",
    eyebrow="Fantasy · in-season",
)

league_settings = st.session_state.fantasy_league_settings
projections = setup["projections"]
roster = st.session_state.fantasy_roster
target_season = setup["target_season"]

st.caption(
    f"The player pool on this page is the same {projection_season_label(target_season)} pool the draft pages "
    "use. Note that the lineup, waiver, and trade models score each player's underlying stat line directly, "
    f"so their point totals are on the raw-stat basis rather than the {target_season} projection — treat them "
    "as relative rankings, not as forecasts of the same number the draft board shows."
)

waivers_tab, lineup_tab, trades_tab, scoring_tab = st.tabs(["Waivers", "Lineup", "Trades", "Scoring"])

# ---------------------------------------------------------------------------
# 1. Waivers
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
                lambda: waiver_recommendations(league_state, available, scoring_mode_override, budget=float(budget)),
            )
            if recs:
                st.session_state["fantasy_waiver_result"] = recs

        recs = st.session_state.get("fantasy_waiver_result")
        if recs:
            bid_col = "suggested_auction_bid" if league_settings.get("is_auction") else "suggested_faab_bid"
            columns_wanted = [
                col
                for col in ["waiver_rank", "name", "position", "composite_score", "replacement_value", bid_col, "rationale"]
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
# 2. Lineup
# ---------------------------------------------------------------------------
with lineup_tab:
    section_header(
        "Lineup optimizer", "Solve for the point-maximizing legal starting lineup, plus per-player start/sit calls."
    )
    if roster:
        roster_names = [player.get("name", "") for player in roster]
        id_by_name = {player.get("name"): player.get("player_id") for player in roster}
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
                lambda: optimize_lineup(roster, projections, league_settings, constraints),
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
                "start/sit advice", lambda: start_sit_advice(roster, projections, league_settings)
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
# 3. Trades
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
                    lambda: evaluate_trade(
                        team_a_players=team_a_gives,
                        team_b_players=team_b_gives,
                        league_settings=league_settings,
                        projections=projections,
                        monte_carlo_iterations=int(iterations),
                        weeks_remaining=int(weeks_remaining),
                        team_a_roster=roster,
                        seed=int(trade_seed),
                    ),
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
            for line in trade["rationale"]:
                st.write(f"- {line}")
    else:
        empty_state(
            "Nothing to trade yet",
            "Load projections and/or a roster in League & draft setup above to evaluate a trade.",
            icon="🤝",
        )

# ---------------------------------------------------------------------------
# 4. Scoring
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
                scored["_authoritative_projection"] = chosen_player.get("projection")
                scored["_player_name"] = chosen_player.get("name")
                scored["_projection_basis"] = chosen_player.get("projection_basis", "")
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
            st.bar_chart(pd.DataFrame(breakdown_items, columns=["Stat", "Points"]).set_index("Stat"))
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
                    for player, score in zip(projections, batch)
                ]
            ).sort_values("stat-line points", ascending=False)
            st.dataframe(batch_df, width="stretch", hide_index=True, height=320)
    else:
        empty_state("No player projections loaded", "Load a pool in League & draft setup above.", icon="📊")
