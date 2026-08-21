import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

# Fantasy Mock Draft: the live, pick-by-pick draft.
#
# This page is the draft itself -- you on the clock,
# the room drafting around you, recommendations recomputed off your real
# roster after every pick, a live team grade, and the manual override for
# players the simulation took but your real draft has not.
#
# All calculations are delegated to the `fantasy` engine package
# (`fantasy_engine/fantasy`); this page only handles presentation, following the
# same split used by every other page in this app.
#
# Deliberately a comment, not a string. The sys.path preamble above has to run
# first, so a string here would not be the module docstring -- it would be a
# bare expression, which Streamlit's magic renders straight onto the page.

from typing import Any

import pandas as pd
import streamlit as st
from app.fantasy_shared import (
    drafted_card_html,
    league_setup,
    pick_card_html,
    player_card_html,
    render_grade_panel,
    require_pool,
)
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    run_analysis,
    section_header,
)
from fantasy.assistant import get_best_pick_for_round
from fantasy.grader import grade_team
from fantasy.live_draft import (
    draft_for_user,
    drafted_players,
    override_draft_for_user,
    start_live_draft,
    user_turn_context,
)
from fantasy.projections import projection_season_label

apply_global_theme()
setup = league_setup()

page_header(
    "Mock Draft",
    "Draft live, one pick at a time, with the assistant recommending your best move every turn — "
    f"and a team grade that updates after every pick. All values are {projection_season_label(setup['target_season'])}.",
    eyebrow="Fantasy · mock draft",
)

if require_pool(setup, "run a draft"):
    league_settings = st.session_state.fantasy_league_settings
    projections = setup["projections"]

    section_header(
        "Live mock draft",
        f"You're on the clock at pick {setup['draft_pick']} of a {setup['n_teams']}-team "
        f"{setup['draft_type'].lower()} draft. The room drafts automatically between your turns — take your "
        "own pick each round and every recommendation after it updates off exactly what you chose, not a "
        "pre-planned script.",
    )

    live = st.session_state.get("fantasy_live_draft")
    start_col, seed_col = st.columns([2, 1])
    seed = int(seed_col.number_input("Random seed", min_value=0, value=42, key="fantasy_draft_seed"))
    with start_col:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if st.button(
            "Restart live draft" if live else "Start live draft", type="primary", key="fantasy_start_live_draft"
        ):
            new_state = run_analysis(
                "live draft setup",
                lambda: start_live_draft(
                    projections,
                    league_settings,
                    num_teams=setup["n_teams"],
                    num_rounds=setup["num_rounds"],
                    snake=setup["snake"],
                    user_draft_slot=setup["draft_pick"],
                    seed=seed,
                ),
            )
            if new_state:
                st.session_state["fantasy_live_draft"] = new_state
                st.session_state["fantasy_live_draft_narrated"] = 0
                st.rerun()

    live = st.session_state.get("fantasy_live_draft")
    if not live:
        empty_state(
            "No live draft running",
            "Start a live draft above. You'll pick one player at a time — the room fills in every "
            "pick around you automatically, and your recommendations update the moment you pick.",
            icon="🎙️",
        )
    else:
        def _advance(operation) -> None:
            """Run a pick, store the new state, and redraw immediately.

            Streamlit only reruns the script on the next widget interaction --
            without the explicit rerun, the rest of this pass would keep
            rendering the *pre-pick* recommendations and grade computed further
            down, which is exactly the staleness the live design exists to fix.
            """
            try:
                st.session_state["fantasy_live_draft"] = operation()
            except ValueError as error:
                st.error(str(error))
                return
            st.rerun()

        # -------------------------------------------------------------------
        # What the room did since the user last saw this page.
        # -------------------------------------------------------------------
        narrated = st.session_state.get("fantasy_live_draft_narrated", 0)
        newly_resolved = live["picks"][narrated:]
        bot_picks = [pick for pick in newly_resolved if not pick["is_user_pick"]]
        if bot_picks:
            with st.container(border=True):
                st.markdown(f"**Since your last pick** — {len(bot_picks)} pick(s) happened around the league")
                st.caption(" · ".join(f"{pick['team']} took {pick['name']} ({pick['position']})" for pick in bot_picks))
        st.session_state["fantasy_live_draft_narrated"] = len(live["picks"])

        my_picks = [pick for pick in live["picks"] if pick["is_user_pick"]]
        my_roster = list(live["rosters"].get(live.get("user_team")) or [])

        summary1, summary2, summary3 = st.columns(3)
        summary1.metric("Picks made", len(live["picks"]))
        summary2.metric("Your picks", len(my_picks))
        summary3.metric(
            "Your projected total", f"{sum(pick.get('projection') or 0 for pick in my_picks):,.0f} pts"
        )
        st.caption(
            f"{live['n_teams']} teams · {live['rounds']} rounds · "
            f"{'snake' if live['snake'] else 'linear'} · seed {live['seed']}"
            + (f" · you are {live['user_team']}" if live.get("user_team") else "")
            + (f" · {len(live.get('overrides') or [])} override(s)" if live.get("overrides") else "")
        )
        for warning in live.get("warnings", []):
            st.warning(warning)

        # -------------------------------------------------------------------
        # Team grade -- recomputed from scratch on every rerun, so it is never
        # stale by construction.
        # -------------------------------------------------------------------
        report = run_analysis(
            "team grade",
            lambda: grade_team(
                my_roster,
                live["remaining"],
                league_settings,
                picks=my_picks,
                all_rosters=live["rosters"],
            ),
        )

        if live["is_complete"]:
            st.success("Draft complete.")
            st.divider()
            st.markdown("### Final draft report")
            if report:
                render_grade_panel(report, detailed=True)
        else:
            with st.container(border=True):
                st.markdown("### Team grade")
                if report:
                    render_grade_panel(report, detailed=False)

            # ---------------------------------------------------------------
            # On the clock: recommendations for this exact pick.
            # ---------------------------------------------------------------
            context = user_turn_context(live)
            info1, info2, info3 = st.columns(3)
            info1.metric("On the clock", f"R{context['round']} · #{context['overall_pick']}")
            info2.metric(
                "Picks until your next turn",
                context["picks_until_next"] if context["picks_until_next"] is not None else "—",
            )
            info3.metric("On your roster", len(context["my_roster"]))

            recommendations = run_analysis(
                "best pick for this round",
                lambda: get_best_pick_for_round(
                    context["round"],
                    context["my_roster"],
                    context["board"],
                    league_settings,
                    current_pick_overall=context["overall_pick"],
                    picks_until_next=context["picks_until_next"],
                    limit=9,
                ),
            )

            st.markdown("### Best available")
            if not recommendations:
                st.info(
                    "Every remaining position is already at your roster's cap — draft anyone still "
                    "on the board using the search below."
                )
            else:
                st.caption(
                    "Ranked by ADP proximity to this pick, then positional scarcity, then your scoring "
                    "model — not locked to one position, so a gone target never limits you to a worse "
                    "player at the same spot."
                )
                rec_columns = st.columns(3)
                for index, entry in enumerate(recommendations):
                    with rec_columns[index % 3], st.container(border=True):
                        st.markdown(player_card_html(entry, f"#{entry['rank']}"), unsafe_allow_html=True)
                        if st.button(
                            f"Draft {entry['name']}",
                            key=f"fantasy_draft_{context['overall_pick']}_{entry['player_id']}",
                            type="primary" if index == 0 else "secondary",
                            width="stretch",
                        ):
                            _advance(lambda pid=entry["player_id"]: draft_for_user(live, pid))
                with st.expander("Why these, in this order?"):
                    for entry in recommendations:
                        st.markdown(
                            f"**{entry['rank']}. {entry['name']}** ({entry['position']}) — {entry['rationale']}"
                        )

            with st.expander("Draft someone else"):
                board_by_id = {player["player_id"]: player for player in context["board"]}
                if not board_by_id:
                    st.caption("Nobody is left on the board.")
                else:
                    chosen_id = st.selectbox(
                        "Player",
                        list(board_by_id),
                        format_func=lambda pid: f"{board_by_id[pid].get('name', pid)} "
                        f"({board_by_id[pid].get('position', '')}, {board_by_id[pid].get('team', '')})",
                        key=f"fantasy_manual_pick_{context['overall_pick']}",
                    )
                    if st.button("Draft this player", key=f"fantasy_manual_draft_{context['overall_pick']}"):
                        _advance(lambda pid=chosen_id: draft_for_user(live, pid))

            # ---------------------------------------------------------------
            # Manual override -- the simulation took him, your real draft did not.
            # ---------------------------------------------------------------
            with st.expander("Override — draft a player the mock already took"):
                st.caption(
                    "The simulation is a model of a draft room, not a transcript of yours. If someone here is "
                    "still on the board in your real draft, take them: the team that had them is re-picked at "
                    "its own slot by the same room model, so the board stays complete and every later bot pick "
                    "is unchanged."
                )
                taken = drafted_players(live)
                if not taken:
                    st.caption("Nobody has been drafted by the room yet.")
                else:
                    search = st.text_input(
                        "Search drafted players",
                        key=f"fantasy_override_search_{context['overall_pick']}",
                        placeholder="Name or position…",
                    ).strip().lower()
                    matches = [
                        entry
                        for entry in taken
                        if not search
                        or search in str(entry["name"]).lower()
                        or search == str(entry["position"]).lower()
                    ]
                    if not matches:
                        st.caption("No drafted player matches that search.")
                    else:
                        st.caption(f"{len(matches)} drafted player(s) — most recent first.")
                        override_columns = st.columns(3)
                        for index, entry in enumerate(matches[:12]):
                            with override_columns[index % 3], st.container(border=True):
                                st.markdown(drafted_card_html(entry), unsafe_allow_html=True)
                                if st.button(
                                    "Override and Draft",
                                    key=f"fantasy_override_{context['overall_pick']}_{entry['player_id']}",
                                    width="stretch",
                                ):
                                    _advance(lambda pid=entry["player_id"]: override_draft_for_user(live, pid))
                        if len(matches) > 12:
                            st.caption(f"{len(matches) - 12} more — narrow the search to see them.")

        if live.get("overrides"):
            with st.expander(f"Overrides applied ({len(live['overrides'])})"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Pick": record["overall_pick"],
                                "Round": record["round"],
                                "You took": record["name"],
                                "From": record["released_from"],
                                "They re-picked": record["replaced_with"] or "— (slot vacated)",
                            }
                            for record in live["overrides"]
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

        # -------------------------------------------------------------------
        # Your team so far -- always visible, in progress or complete.
        # -------------------------------------------------------------------
        st.divider()
        st.markdown("### Your team")
        if not my_picks:
            empty_state("No picks yet", "Your first pick will appear here.", icon="🪑")
        else:
            players_by_id = {player.get("player_id"): player for player in projections if player.get("player_id")}
            roster_columns = st.columns(3)
            for index, pick in enumerate(my_picks):
                with roster_columns[index % 3], st.container(border=True):
                    st.markdown(
                        pick_card_html(pick, players_by_id.get(pick["player_id"], {})), unsafe_allow_html=True
                    )

        # -------------------------------------------------------------------
        # Reference: the board so far, and league-wide rosters.
        # -------------------------------------------------------------------
        by_round = live.get("by_round") or {}
        if by_round:
            st.divider()
            st.markdown("### Draft board so far")
            played_rounds = sorted(by_round)
            if st.session_state.get("fantasy_live_walk_round") not in played_rounds:
                st.session_state["fantasy_live_walk_round"] = played_rounds[-1]
            board_round = int(st.select_slider("Round", options=played_rounds, key="fantasy_live_walk_round"))
            st.caption(
                "**Value** = points above positional replacement (VORP). **Scarcity** = same-position "
                "players still available at that moment. **Timing** compares the pick to real ADP."
            )

            def _pick_timing_indicator(pick: dict[str, Any]) -> str:
                adp, overall_pick = pick.get("adp"), pick["pick"]
                if adp is None:
                    return "—"
                delta = overall_pick - adp
                if delta >= 15:
                    return f"📉 Steal (+{delta:.0f})"
                if delta <= -15:
                    return f"⚠️ Reach ({delta:+.0f})"
                return "On time"

            round_df = pd.DataFrame(
                [
                    {
                        "Pick": pick["pick"],
                        "Team": pick["team"],
                        "Player": pick["player_name"],
                        "Pos": pick["position"],
                        "ADP": pick.get("adp"),
                        "Value": pick.get("vor"),
                        "Scarcity": pick.get("remaining_at_position"),
                        "Timing": _pick_timing_indicator(pick),
                        "Run": f"🔥 {pick['position_run']}" if pick.get("position_run") else "",
                        "You": "★" if pick["is_user_pick"] else "",
                    }
                    for pick in by_round[board_round]
                ]
            )
            st.dataframe(
                round_df.style.apply(
                    lambda row: ["background-color: rgba(255,90,54,.14)" if row["You"] else ""] * len(row), axis=1
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "ADP": st.column_config.NumberColumn(format="%.1f"),
                    "Value": st.column_config.NumberColumn(format="%.1f", help="VORP at draft time."),
                    "Scarcity": st.column_config.NumberColumn(
                        help="Same-position players still available when this pick happened."
                    ),
                },
            )

            position_runs = live.get("clusters") or []
            if position_runs:
                with st.expander(f"Positional ADP clusters detected ({len(position_runs)})"):
                    st.caption(
                        "Tight bunches of same-position ADP that bias simulated teams toward that "
                        "position while active."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Position": run["position"],
                                    "ADP range": f"{run['start_adp']:.0f}–{run['end_adp']:.0f}",
                                    "Players": run["size"],
                                }
                                for run in position_runs
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            with st.expander("Full rosters by team"):
                for team, players in (live.get("rosters") or {}).items():
                    team_df = pd.DataFrame(players)
                    if team_df.empty:
                        continue
                    st.markdown(f"**{team}**" + (" ← you" if team == live.get("user_team") else ""))
                    columns_wanted = [c for c in ("name", "position", "team", "points", "vor") if c in team_df]
                    st.dataframe(team_df[columns_wanted], width="stretch", hide_index=True)
