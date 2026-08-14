from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

# Fantasy Draft Assistant: recommendations for a draft you're running elsewhere.
#
# One of three Fantasy pages. The Draft Room simulates a room around you; this
# page assumes the room is real and somewhere else -- you tell it who is gone
# and who you already have, and it tells you who to take, grades what you've
# built so far, and prints the cheat sheet.
#
# All calculations are delegated to the `fantasy` engine package
# (`fantasy_engine/fantasy`); this page only handles presentation.
#
# Deliberately a comment, not a string -- see the note in 25_Fantasy_Draft_Room.py.

from typing import Any

import pandas as pd
import streamlit as st

from app.page_runtime import apply_global_theme, empty_state, page_header, run_analysis, section_header
from app.fantasy_shared import league_setup, render_grade_panel, require_pool

from fantasy.assistant import capped_positions, get_best_pick_for_round, next_pick_number, picks_between
from fantasy.draft import rank_players_for_draft
from fantasy.grader import grade_team
from fantasy.projections import projection_season_label
from fantasy.tiering import combined_tier, export_cheatsheet_csv, generate_printable_cheatsheet, tier_players

apply_global_theme()
setup = league_setup()

page_header(
    "Draft Assistant",
    "For a draft you're running somewhere else — tell it who's gone and who you have, and it ranks your "
    f"next pick, grades your team, and prints the cheat sheet. All values are "
    f"{projection_season_label(setup['target_season'])}.",
    eyebrow="Fantasy · manual recommendations",
)

if require_pool(setup, "get recommendations"):
    league_settings = st.session_state.fantasy_league_settings
    projections = setup["projections"]
    n_teams, draft_pick, snake = setup["n_teams"], setup["draft_pick"], setup["snake"]

    pick_tab, sheet_tab = st.tabs(["Pick assistant", "Tiers & cheat sheets"])

    # -----------------------------------------------------------------------
    # 1. Pick assistant -- who to take right now, and how the team grades.
    # -----------------------------------------------------------------------
    with pick_tab:
        section_header(
            "Best available at your next pick",
            "Cross-position by design: if your target is gone, the right answer is rarely the next-best "
            "player at the same position — it's whoever is the best pick for this exact slot.",
        )

        all_positions = sorted({str(player.get("position", "")) for player in projections} - {""})
        players_by_id = {
            player["player_id"]: player for player in projections if player.get("player_id")
        }

        def _label(player_id: Any) -> str:
            player = players_by_id.get(player_id, {})
            return f"{player.get('name', player_id)} ({player.get('position', '')}, {player.get('team', '')})"

        manual1, manual2 = st.columns(2)
        manual_round = int(
            manual1.number_input(
                "Current round", min_value=1, max_value=setup["num_rounds"], value=1, key="fantasy_assistant_round"
            )
        )
        my_player_ids = manual2.multiselect(
            "Players already on my roster",
            list(players_by_id),
            format_func=_label,
            key="fantasy_assistant_my_players",
            help="Real players, not just positions — that lets bye-week clustering, stacking, and "
            "per-NFL-team caps actually apply, and it lets the grader score what you've built.",
        )

        manual3, manual4 = st.columns(2)
        # Options stay the full pool rather than being narrowed by the roster
        # selection above: narrowing them would drop a player out of this
        # widget's own options while still sitting in its session-state value,
        # which Streamlit rejects outright. Overlap is harmless -- both lists
        # feed the same `gone` set below.
        drafted_ids = manual3.multiselect(
            "Players taken by everyone else",
            list(players_by_id),
            format_func=_label,
            key="fantasy_assistant_drafted",
            help="Anyone off the board who isn't yours. They're removed from the suggestions below.",
        )
        position_filter = manual4.multiselect(
            "Only suggest these positions", all_positions, key="fantasy_assistant_position_filter"
        )

        manual_overall = next_pick_number(n_teams, draft_pick, manual_round, snake)
        manual_gap = picks_between(n_teams, draft_pick, manual_round, snake)
        manual_info1, manual_info2, manual_info3 = st.columns(3)
        manual_info1.metric("Your pick this round", f"#{manual_overall}")
        manual_info2.metric("Picks until your next", manual_gap)
        manual_info3.metric("Format", f"{setup['scoring_label']} · {setup['draft_type']}")

        gone = set(my_player_ids) | set(drafted_ids)
        manual_pool = [player for player in projections if player.get("player_id") not in gone]
        my_roster = [players_by_id[pid] for pid in my_player_ids]

        manual_capped = capped_positions(my_roster)
        if manual_capped:
            st.caption(f"Roster full at {', '.join(manual_capped)} — hidden from suggestions below.")

        def _manual_recommendations() -> list[dict[str, Any]]:
            # get_best_pick_for_round has no built-in position filter (it's
            # deliberately cross-position); a wide limit plus a post-filter
            # gets the same result without adding a filter the engine itself
            # never needs.
            picks = get_best_pick_for_round(
                manual_round,
                my_roster,
                manual_pool,
                league_settings,
                current_pick_overall=manual_overall,
                picks_until_next=manual_gap,
                limit=len(manual_pool) if position_filter else 12,
            )
            if position_filter:
                wanted = set(position_filter)
                picks = [entry for entry in picks if entry["position"] in wanted][:12]
            return picks

        manual_recs = run_analysis("best pick for this round", _manual_recommendations)

        if not manual_recs:
            st.info("No players match those filters — clear a filter or remove someone from the drafted list.")
        else:
            best = manual_recs[0]
            st.success(
                f"Best available at #{manual_overall}: **{best['name']}** "
                f"({best['position']}, {best['team']}) — {best['projection']:,.0f} projected pts"
            )
            st.caption(best["rationale"])

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "#": entry["rank"],
                            "Player": entry["name"],
                            "Pos": entry["position"],
                            "Team": entry["team"],
                            "Projection": entry["projection"],
                            "Value": entry["vorp"],
                            "Pos rank": entry["position_rank"],
                            "ADP": entry["adp"],
                            "ADP gap": entry["adp_proximity"],
                            "Scarcity": entry["scarcity"],
                            "Need": entry["need_label"],
                        }
                        for entry in manual_recs
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Projection": st.column_config.NumberColumn(
                        format="%.0f", help=f"Projected {setup['target_season']} fantasy points."
                    ),
                    "Value": st.column_config.NumberColumn(
                        format="%.1f", help="VORP: points above a replacement-level starter at this position."
                    ),
                    "ADP": st.column_config.NumberColumn(format="%.1f", help="FantasyPros expert consensus rank."),
                    "ADP gap": st.column_config.NumberColumn(
                        format="%.0f", help="Picks between this player's ADP and your current pick number."
                    ),
                    "Scarcity": st.column_config.NumberColumn(
                        format="%.0f",
                        help="Points of position value that disappear before your next pick, "
                        "weighted for how thin the position typically runs (TE/RB up, WR/QB down).",
                    ),
                    "Need": st.column_config.TextColumn(
                        help="Roster status at this position under your league's construction rules."
                    ),
                },
            )

        # --- live grade for the manually-entered roster ---------------------
        st.divider()
        st.markdown("### Team grade")
        if not my_roster:
            empty_state(
                "No roster to grade yet",
                'Add your own players under "Players already on my roster" and this grades every position '
                "group against your league's average team.",
                icon="📈",
            )
        else:
            manual_report = run_analysis(
                "team grade",
                lambda: grade_team(my_roster, manual_pool, league_settings),
            )
            if manual_report:
                st.caption(
                    "Graded without pick numbers, so the ADP-value component is neutral here — the Draft "
                    "Room's grade adds it because it knows exactly when each player was taken."
                )
                render_grade_panel(manual_report, detailed=True)

        with st.expander("Full VOR draft board"):
            ranked = run_analysis("draft ranking", lambda: rank_players_for_draft(manual_pool, league_settings))
            if ranked:
                board_df = pd.DataFrame(ranked)[
                    ["overall_rank", "name", "position", "team", "points", "vor", "position_rank", "rationale"]
                ]
                if position_filter:
                    board_df = board_df[board_df["position"].isin(position_filter)]
                st.dataframe(board_df, width="stretch", hide_index=True, height=320)

    # -----------------------------------------------------------------------
    # 2. Tiers & cheat sheets
    # -----------------------------------------------------------------------
    with sheet_tab:
        section_header(
            "Tiering & cheat sheets",
            "Cluster the ranked pool into tiers and export a printable or CSV cheat sheet.",
        )
        TIER_COLORS = {1: "#0f8a4c", 2: "#4c8f2f", 3: "#c9a227", 4: "#c9662a", 5: "#b3401f"}

        def _tier_background(tier: Any) -> str:
            color = TIER_COLORS.get(int(tier), "#6b7280") if pd.notna(tier) else "#6b7280"
            return f"background-color: color-mix(in srgb, {color} 22%, transparent)"

        max_tiers = st.slider("Max tiers per position", min_value=2, max_value=10, value=5, key="fantasy_max_tiers")
        if st.button("Generate tiers", type="primary", key="fantasy_generate_tiers"):
            ranked = run_analysis("draft ranking", lambda: rank_players_for_draft(projections, league_settings))
            if ranked:
                tiered = run_analysis("tiering", lambda: tier_players(ranked, max_tiers=int(max_tiers)))
                if tiered:
                    st.session_state["fantasy_tiered_players"] = tiered
                combined = run_analysis(
                    "combined tiering",
                    lambda: combined_tier(ranked, league_settings, n_teams=n_teams, max_tiers=int(max_tiers)),
                )
                if combined:
                    st.session_state["fantasy_combined_tiers"] = combined

        combined = st.session_state.get("fantasy_combined_tiers")
        if combined:
            st.markdown("#### ESPN-style cheat sheet: ADP + projection + scarcity, blended")
            st.caption(
                "**ADP tier** = replacement-level bands by real market ADP (Elite/Starter/Flex/Streamer/Deep Bench). "
                "**Scarcity tier** = VORP percentile within position. **Value tier** = the volatility-aware cluster tier. "
                "**Combined** averages all three — a player whose market ADP and projection disagree lands in between, "
                "not at either extreme."
            )
            for position in sorted({p["position"] for p in combined}):
                group = sorted(
                    (p for p in combined if p["position"] == position),
                    key=lambda p: (p["combined_tier"] if p["combined_tier"] is not None else 999, -(p.get("vor") or 0)),
                )
                with st.expander(f"{position} ({len(group)})", expanded=position in {"RB", "WR", "QB", "TE"}):
                    group_df = pd.DataFrame(
                        [
                            {
                                "Combined Tier": p["combined_tier"],
                                "Player": p["name"],
                                "Team": p.get("team", ""),
                                "ADP": p.get("adp"),
                                "ADP Tier": p.get("adp_tier_label", ""),
                                "Projection": p.get("points"),
                                "Value (VORP)": p.get("vor"),
                                "Scarcity Tier": p.get("scarcity_tier"),
                            }
                            for p in group
                        ]
                    )
                    st.dataframe(
                        group_df.style.apply(lambda row: [_tier_background(row["Combined Tier"])] * len(row), axis=1),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "ADP": st.column_config.NumberColumn(format="%.1f"),
                            "Projection": st.column_config.NumberColumn(
                                format="%.1f", help=f"Projected {setup['target_season']} fantasy points."
                            ),
                            "Value (VORP)": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )

        tiered = st.session_state.get("fantasy_tiered_players")
        if tiered:
            with st.expander("Value + volatility tiers (k-means clustering)"):
                tiers_df = pd.DataFrame(tiered)[
                    ["position", "tier", "tier_label", "overall_rank", "name", "team", "points", "vor"]
                ]
                tiers_df = tiers_df.sort_values(["position", "tier", "overall_rank"])
                st.dataframe(
                    tiers_df.style.apply(lambda row: [_tier_background(row["tier"])] * len(row), axis=1),
                    width="stretch",
                    hide_index=True,
                    height=360,
                )

            printable = generate_printable_cheatsheet(tiered)
            st.text_area(f"Printable cheat sheet — {projection_season_label(setup['target_season'])}", printable, height=240)
            st.download_button(
                "Download cheat sheet CSV",
                export_cheatsheet_csv(tiered),
                file_name=f"fantasy_cheatsheet_{setup['target_season']}.csv",
                mime="text/csv",
                key="fantasy_download_cheatsheet",
            )
        elif not combined:
            empty_state("No tiers generated yet", 'Click "Generate tiers" to build the cheat sheet.', icon="🗂️")
