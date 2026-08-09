from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

# Fantasy: draft room, scoring, lineups, waivers, trades, and tiered cheat sheets.
#
# All calculations are delegated to the `fantasy` engine package
# (`fantasy_engine/fantasy`); this page only handles presentation, following the
# same split used by every other page in this app.
#
# Deliberately a comment, not a string. The sys.path preamble above has to run
# first, so a string here would not be the module docstring -- it would be a
# bare expression, which Streamlit's magic renders straight onto the page.

import json
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    pill,
    run_analysis,
    section_header,
)

from fantasy.assistant import (
    capped_positions,
    get_replacements,
    next_pick_number,
    picks_between,
    suggest_draft_picks,
)
from fantasy.data_loader import (
    RealDataUnavailable,
    drop_synthetic,
    latest_completed_season,
    load_real_projections,
    validate_players,
)
from fantasy.draft import rank_players_for_draft, simulate_draft
from fantasy.optimizer import optimize_lineup, start_sit_advice
from fantasy.scoring import (
    BASE_MULTIPLIERS,
    RECEPTION_MULTIPLIER_BY_MODE,
    batch_calculate_fantasy_points,
    calculate_fantasy_points,
)
from fantasy.tiering import (
    combined_tier,
    export_cheatsheet_csv,
    generate_printable_cheatsheet,
    tier_players,
)
from fantasy.trade import evaluate_trade
from fantasy.waiver import waiver_recommendations

apply_global_theme()
page_header(
    "Fantasy",
    "Simulate a draft, then walk it round by round with the assistant on your shoulder — "
    "plus scoring, lineups, waivers, trades, and tiered cheat sheets.",
    eyebrow="Draft intelligence",
)

# ---------------------------------------------------------------------------
# Player pool: real NFL players only.
#
# `fantasy.data_loader` pulls actual production from nflverse and scores it
# under the league's scoring mode. The engine's bundled example JSON is not
# used as a player pool -- it is ~97% engine-generated `synthetic:*` filler,
# which produces a draft board of players who do not exist.
# ---------------------------------------------------------------------------
SCORING_LABELS = {"PPR": "ppr", "Half-PPR": "half-ppr", "Standard": "standard"}

_DEFAULT_LEAGUE_SETTINGS = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
    "max_players_per_nfl_team": None,
    "faab_budget": 100,
}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def _load_players(season: int, scoring_mode: str) -> tuple[list[dict[str, Any]], str]:
    """Fetch and score the real player pool. Cached so tab switches are instant."""
    players = load_real_projections(season=season, scoring_mode=scoring_mode)
    basis = players[0].get("projection_basis", "") if players else ""
    return players, basis


def _clean_uploaded(players: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop synthetic entries from uploaded JSON and validate the rest."""
    if not isinstance(players, list):
        return [], []
    return validate_players(drop_synthetic(players), require_projection=False)


def _init_state() -> None:
    for key, value in {
        "fantasy_league_settings": dict(_DEFAULT_LEAGUE_SETTINGS),
        "fantasy_projections": [],
        "fantasy_roster": [],
        "fantasy_available": [],
        "fantasy_draft_focus": None,
    }.items():
        st.session_state.setdefault(key, value)


_init_state()

# ---------------------------------------------------------------------------
# League, draft, and data setup -- shared by every tab below.
# ---------------------------------------------------------------------------
with st.expander("League & draft setup", expanded=not st.session_state.fantasy_projections):
    settings = dict(st.session_state.fantasy_league_settings)

    col1, col2, col3 = st.columns(3)
    n_teams = int(
        col1.number_input(
            "Number of teams", min_value=2, max_value=32, value=int(settings.get("n_teams", 12)), key="fantasy_n_teams"
        )
    )
    draft_pick = int(
        col2.number_input(
            "My draft pick",
            min_value=1,
            max_value=n_teams,
            value=min(int(st.session_state.get("fantasy_draft_pick", 1)), n_teams),
            key="fantasy_draft_pick",
            help="Your slot in round 1. In a snake draft this also determines every later pick.",
        )
    )
    num_rounds = int(
        col3.number_input("Number of rounds", min_value=1, max_value=30, value=15, key="fantasy_num_rounds")
    )

    col4, col5, col6 = st.columns(3)
    scoring_label = col4.selectbox("Scoring format", list(SCORING_LABELS), index=0, key="fantasy_scoring_format")
    scoring_mode = SCORING_LABELS[scoring_label]
    draft_type = col5.radio(
        "Draft type", ["Snake", "Linear"], horizontal=True, key="fantasy_draft_type",
        help="Snake reverses the order every round. Linear runs 1..N every round.",
    )
    snake = draft_type == "Snake"
    season = int(
        col6.number_input(
            "Stats season",
            min_value=2000,
            max_value=latest_completed_season() + 1,
            value=latest_completed_season(),
            key="fantasy_season",
        )
    )

    settings.update({"n_teams": n_teams, "scoring_mode": scoring_mode})
    st.session_state.fantasy_league_settings = settings

    # --- load the real pool -------------------------------------------------
    load_error: str | None = None
    projection_basis = ""
    try:
        with st.spinner(f"Loading real {season} NFL player data..."):
            real_players, projection_basis = _load_players(season, scoring_mode)
    except RealDataUnavailable as error:
        real_players, load_error = [], str(error)
    except Exception as error:  # network hiccup, upstream schema change, ...
        real_players, load_error = [], f"Unexpected error loading NFL data: {error}"

    st.markdown("**Custom data (optional)** — upload JSON to override any of the pools below.")
    up1, up2, up3 = st.columns(3)
    projections_file = up1.file_uploader("Projections JSON", type="json", key="fantasy_upload_projections")
    roster_file = up2.file_uploader("My roster JSON", type="json", key="fantasy_upload_roster")
    available_file = up3.file_uploader("Available players JSON", type="json", key="fantasy_upload_available")

    uploaded_rejects: list[dict[str, Any]] = []
    if projections_file is not None:
        try:
            cleaned, uploaded_rejects = _clean_uploaded(json.load(projections_file))
            st.session_state.fantasy_projections = cleaned
            projection_basis = "uploaded projections"
        except json.JSONDecodeError as error:
            st.error(f"Could not parse the projections file: {error}")
    else:
        st.session_state.fantasy_projections = real_players

    if roster_file is not None:
        try:
            st.session_state.fantasy_roster = _clean_uploaded(json.load(roster_file))[0]
        except json.JSONDecodeError as error:
            st.error(f"Could not parse the roster file: {error}")

    if available_file is not None:
        try:
            st.session_state.fantasy_available = _clean_uploaded(json.load(available_file))[0]
        except json.JSONDecodeError as error:
            st.error(f"Could not parse the available-players file: {error}")
    elif not st.session_state.fantasy_available:
        # Free agents default to everyone outside the top ~n_teams*num_rounds picks.
        st.session_state.fantasy_available = st.session_state.fantasy_projections[n_teams * num_rounds :][:150]

    projections = st.session_state.fantasy_projections
    if load_error and projections_file is None:
        st.error(load_error)
        st.info("Upload a projections JSON above to use this page offline.")
    elif projections:
        st.success(f"{len(projections)} real players loaded · {projection_basis}")
        if uploaded_rejects:
            st.warning(f"Skipped {len(uploaded_rejects)} uploaded row(s) that were synthetic or missing fields.")
        with_adp = sum(1 for player in projections if player.get("adp") is not None)
        missing_positions = sorted(
            slot
            for slot, count in (settings.get("roster_requirements") or {}).items()
            if count and slot not in {"BENCH", "IR", "TAXI", "FLEX"}
            and not any(p.get("position") == slot for p in projections)
        )
        st.caption(
            f"{with_adp} players carry FantasyPros ADP · "
            f"{len(st.session_state.fantasy_roster)} roster spots · "
            f"{len(st.session_state.fantasy_available)} free agents."
        )
        if missing_positions:
            st.caption(
                f"No {', '.join(missing_positions)} in the nflverse pool — those slots will go unfilled in a mock draft."
            )


# ---------------------------------------------------------------------------
# Shared presentation helpers for the draft room.
# ---------------------------------------------------------------------------
def _fmt(value: Any, spec: str = ",.0f", dash: str = "—") -> str:
    """Format a number for a card, degrading to a dash rather than erroring."""
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return dash


def _timing_pill(entry: dict[str, Any]) -> str:
    """The one timing signal that matters most for this player, color-coded."""
    if entry.get("must_draft_now"):
        return pill("🔥 Won't last", "danger")
    if entry.get("steal_bonus"):
        return pill(f"📉 Steal +{entry['steal_bonus']:.0f}", "success")
    if entry.get("reach_penalty"):
        return pill(f"⚠️ Reach −{entry['reach_penalty']:.0f}", "warning")
    if entry.get("adp") is not None:
        return pill("On time", "neutral")
    return pill("No ADP", "neutral")


def _scarcity_pill(entry: dict[str, Any]) -> str:
    """Positional scarcity, banded so the color carries the meaning."""
    scarcity = entry.get("scarcity")
    if scarcity is None:
        return ""
    position = entry.get("position", "")
    if scarcity >= 40:
        return pill(f"{position} scarce · {scarcity:.0f}", "danger")
    if scarcity >= 18:
        return pill(f"{position} thinning · {scarcity:.0f}", "warning")
    return pill(f"{position} deep · {scarcity:.0f}", "success")


def _need_pill(entry: dict[str, Any]) -> str:
    label = entry.get("need_label")
    if label == "Fills need":
        return pill("Fills need", "info")
    if label == "Depth":
        return pill("Depth", "neutral")
    return ""


def _player_card_html(
    entry: dict[str, Any],
    rank_label: str,
    taken_by: str | None = None,
) -> str:
    """One recommendation rendered as a premium card."""
    pills = "".join([_timing_pill(entry), _scarcity_pill(entry), _need_pill(entry)])
    if taken_by:
        pills = pill(f"🔒 Taken by {taken_by}", "danger") + pills
    stats = [
        ("Proj", _fmt(entry.get("projection"))),
        ("VORP", _fmt(entry.get("vorp"), "+.0f")),
        ("ADP", _fmt(entry.get("adp"), ".1f")),
        (f"{entry.get('position', '')} rank", _fmt(entry.get("position_rank"), ".0f")),
    ]
    stat_html = "".join(
        f'<div class="stat"><span class="k">{escape(key)}</span><span class="v">{escape(value)}</span></div>'
        for key, value in stats
    )
    classes = "player-card is-taken" if taken_by else "player-card"
    return (
        f'<div class="{classes}">'
        f'<div class="rank">{escape(rank_label)}</div>'
        f'<div class="name">{escape(str(entry.get("name", "—")))}</div>'
        f'<div class="meta">{escape(str(entry.get("position", "")))} · {escape(str(entry.get("team", "")))}</div>'
        f'<div class="stat-row">{stat_html}</div>'
        f"<div>{pills}</div>"
        f"</div>"
    )


def _pick_card_html(pick: dict[str, Any], source: dict[str, Any] | None = None) -> str:
    """One of my own drafted players, rendered as a card.

    ``source`` is the player's row from the projection pool. The pick dict's
    own ``team`` is the *fantasy* team ("Team 5"), so the NFL team has to come
    from the pool row instead.
    """
    source = source or {}
    nfl_team = str(source.get("team") or "")
    adp = pick.get("adp")
    timing = ""
    if adp is not None:
        delta = float(pick["overall_pick"]) - float(adp)
        if delta >= 15:
            timing = pill(f"📉 Steal +{delta:.0f}", "success")
        elif delta <= -15:
            timing = pill(f"⚠️ Reach {delta:+.0f}", "warning")
        else:
            timing = pill("On time", "neutral")
    stats = [
        ("Proj", _fmt(pick.get("projection"))),
        ("VORP", _fmt(pick.get("vor"), "+.0f")),
        ("ADP", _fmt(adp, ".1f")),
    ]
    stat_html = "".join(
        f'<div class="stat"><span class="k">{escape(key)}</span><span class="v">{escape(value)}</span></div>'
        for key, value in stats
    )
    return (
        '<div class="player-card">'
        f'<div class="rank">Round {pick["round"]} · Pick #{pick["overall_pick"]}</div>'
        f'<div class="name">{escape(str(pick.get("name", "—")))}</div>'
        f'<div class="meta">{escape(str(pick.get("position", "")))} · {escape(nfl_team)}</div>'
        f'<div class="stat-row">{stat_html}</div>'
        f"<div>{timing}</div>"
        "</div>"
    )


tab_labels = [
    "Draft Room",
    "Scoring",
    "Lineup",
    "Waivers",
    "Trades",
    "Cheat Sheets",
]
tabs = st.tabs(tab_labels)

# ---------------------------------------------------------------------------
# 1. Draft Room -- mock draft + the assistant, walked round by round.
# ---------------------------------------------------------------------------
with tabs[0]:
    projections = st.session_state.fantasy_projections
    if not projections:
        empty_state(
            "No player pool loaded",
            "Fix the data source in League & draft setup above, or upload a projections JSON, to run a draft.",
            icon="🏈",
        )
    else:
        section_header(
            "Mock draft",
            f"Simulate a {n_teams}-team {draft_type.lower()} draft from pick {draft_pick}, using real NFL players only.",
        )
        run_col, seed_col = st.columns([2, 1])
        seed = int(seed_col.number_input("Random seed", min_value=0, value=42, key="fantasy_draft_seed"))
        with run_col:
            st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
            if st.button("Run mock draft", type="primary", key="fantasy_run_mock_draft"):
                mock_result = run_analysis(
                    "mock draft simulation",
                    lambda: simulate_draft(
                        projections,
                        st.session_state.fantasy_league_settings,
                        num_teams=n_teams,
                        num_rounds=num_rounds,
                        snake=snake,
                        user_draft_slot=draft_pick,
                        seed=seed,
                    ),
                )
                if mock_result:
                    st.session_state["fantasy_mock_draft_result"] = mock_result
                    st.session_state["fantasy_draft_focus"] = None

        mock_result = st.session_state.get("fantasy_mock_draft_result")
        if not mock_result:
            empty_state(
                "No draft run yet",
                "Run a mock draft to walk it round by round with the assistant, see who got sniped, "
                "and pull replacement options the moment a target comes off the board.",
                icon="🎯",
            )
        else:
            by_round = mock_result.get("by_round") or {}
            user_picks = mock_result.get("user_picks") or []
            all_picks = mock_result.get("picks") or []

            summary1, summary2, summary3 = st.columns(3)
            summary1.metric("Picks made", len(all_picks))
            summary2.metric("Your picks", len(user_picks))
            summary3.metric(
                "Your projected total",
                f"{sum(pick.get('projection') or 0 for pick in user_picks):,.0f} pts",
            )
            st.caption(
                f"{mock_result['n_teams']} teams · {mock_result['rounds']} rounds · "
                f"{'snake' if mock_result.get('snake') else 'linear'} · seed {mock_result['seed']}"
                + (f" · you are {mock_result['user_team']}" if mock_result.get("user_team") else "")
            )
            for warning in mock_result.get("warnings", []):
                st.warning(warning)

            if not by_round:
                empty_state("No picks were made", "The player pool was empty.", icon="🗒️")
            else:
                players_by_id = {p.get("player_id"): p for p in projections if p.get("player_id")}

                # Read the draft's own geometry back off the result rather than
                # the live widgets: changing a setting after a run must not
                # silently renumber the picks of a draft already on screen.
                sim_teams = int(mock_result.get("n_teams") or n_teams)
                sim_snake = bool(mock_result.get("snake"))
                sim_rounds = int(mock_result.get("rounds") or num_rounds)

                # ---------------------------------------------------------------
                # Your team. These -- and only these -- are the clickable cards.
                # Tapping one asks the engine who else was on the board at that
                # exact pick, which is what `get_replacements` answers.
                # ---------------------------------------------------------------
                st.markdown("### Your team")
                st.caption(
                    "Every player the simulation drafted for you. Tap one to see who else was "
                    "available at that pick and how the alternatives score."
                )

                if not user_picks:
                    empty_state(
                        "The draft ended before your slot came up",
                        "Increase the number of rounds, or move your draft pick earlier.",
                        icon="🪑",
                    )
                else:
                    roster_columns = st.columns(3)
                    for index, pick in enumerate(user_picks):
                        source = players_by_id.get(pick["player_id"], {})
                        with roster_columns[index % 3]:
                            with st.container(border=True):
                                st.markdown(
                                    _pick_card_html(pick, source),
                                    unsafe_allow_html=True,
                                )
                                if st.button(
                                    "See who else was available",
                                    key=f"fantasy_mypick_{pick['overall_pick']}_{pick['player_id']}",
                                    width="stretch",
                                ):
                                    st.session_state["fantasy_draft_focus"] = {
                                        "player_id": pick["player_id"],
                                        "name": pick["name"],
                                        "overall_pick": int(pick["overall_pick"]),
                                        "round": int(pick["round"]),
                                    }

                    # -----------------------------------------------------------
                    # Replacement panel for whichever of my picks was tapped.
                    # -----------------------------------------------------------
                    focus = st.session_state.get("fantasy_draft_focus")
                    if focus:
                        st.divider()
                        focus_pick = int(focus["overall_pick"])
                        section_header(
                            f"Instead of {focus['name']} — round {focus['round']}, pick #{focus_pick}",
                            "Same position, similar ADP cost, scored by the same model that drafts your "
                            "team, adjusted for how contested that round actually was.",
                        )

                        # Everyone taken strictly *before* this pick is gone; the
                        # focus player is deliberately still on the board here,
                        # because get_replacements resolves the target from the
                        # pool and then filters it out itself.
                        gone_before = {
                            p["player_id"] for p in all_picks if p["overall_pick"] < focus_pick
                        }
                        board_at_pick = [
                            player
                            for player in projections
                            if player.get("player_id") not in gone_before
                        ]
                        # Look the real player objects back up rather than reusing
                        # the pick dicts: a pick's "team" is the fantasy team
                        # ("Team 5"), and the assistant's stacking and bye-week
                        # signals need the actual NFL team.
                        roster_before = [
                            players_by_id[p["player_id"]]
                            for p in user_picks
                            if p["overall_pick"] < focus_pick and p["player_id"] in players_by_id
                        ]
                        later_picks = [p for p in user_picks if p["overall_pick"] > focus_pick]
                        focus_gap = (
                            int(later_picks[0]["overall_pick"]) - focus_pick - 1
                            if later_picks
                            else picks_between(sim_teams, draft_pick, int(focus["round"]), sim_snake)
                        )

                        context1, context2, context3 = st.columns(3)
                        context1.metric("Pick", f"#{focus_pick}")
                        context2.metric("Picks until your next", focus_gap)
                        context3.metric("On your roster then", len(roster_before))

                        draft_state = {
                            "league_settings": st.session_state.fantasy_league_settings,
                            "my_roster": roster_before,
                            "available_players": board_at_pick,
                            "drafted_player_ids": sorted(gone_before),
                            "round_number": int(focus["round"]),
                            "rounds": sim_rounds,
                            "next_pick_overall": focus_pick,
                            "picks_until_next": focus_gap,
                        }
                        replacements = run_analysis(
                            "replacement options",
                            lambda: get_replacements(focus["player_id"], draft_state),
                        )
                        if not replacements:
                            empty_state(
                                "No alternative available",
                                "Nothing comparable was left at that position, or your roster was "
                                "already capped there at that point in the draft.",
                                icon="🔍",
                            )
                        else:
                            replacement_columns = st.columns(min(3, len(replacements)))
                            for index, option in enumerate(replacements):
                                with replacement_columns[index % len(replacement_columns)]:
                                    with st.container(border=True):
                                        st.markdown(
                                            _player_card_html(option, f"Option {option['rank']}"),
                                            unsafe_allow_html=True,
                                        )
                                        st.caption(
                                            f"{option['adp_gap']:+.0f} picks vs {option['replaces_name']} on ADP"
                                            if option.get("adp_gap") is not None
                                            else "next up at the position"
                                        )
                            with st.expander("Why these alternatives?"):
                                for option in replacements:
                                    st.markdown(
                                        f"**{option['rank']}. {option['name']}** "
                                        f"({option['position']}) — {option['replacement_rationale']}"
                                    )
                            if st.button("Clear selection", key="fantasy_clear_focus"):
                                st.session_state["fantasy_draft_focus"] = None

                # ---------------------------------------------------------------
                # Read-only round board.
                # ---------------------------------------------------------------
                st.divider()
                st.markdown("### Round-by-round board")
                simulated_rounds = sorted(by_round)
                # A shorter re-run can leave the stored round out of range, and
                # select_slider raises rather than clamping. Reconcile first.
                if st.session_state.get("fantasy_walk_round") not in simulated_rounds:
                    st.session_state["fantasy_walk_round"] = simulated_rounds[0]
                round_number = int(
                    st.select_slider("Round", options=simulated_rounds, key="fantasy_walk_round")
                )

                st.caption(
                    "**Value** = points above positional replacement (VORP). **Scarcity** = same-position "
                    "players still available at that moment. **Timing** compares the pick to real ADP."
                )

                def _pick_timing_indicator(pick: dict) -> str:
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
                        for pick in by_round[round_number]
                    ]
                )
                st.dataframe(
                    round_df.style.apply(
                        lambda row: ["background-color: rgba(255,90,54,.14)" if row["You"] else ""] * len(row),
                        axis=1,
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

                position_runs = mock_result.get("position_runs") or []
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
                    for team, players in (mock_result.get("rosters") or {}).items():
                        team_df = pd.DataFrame(players)
                        if team_df.empty:
                            continue
                        st.markdown(f"**{team}**" + (" ← you" if team == mock_result.get("user_team") else ""))
                        columns_wanted = [c for c in ("name", "position", "team", "points", "vor") if c in team_df]
                        st.dataframe(team_df[columns_wanted], width="stretch", hide_index=True)

        # -------------------------------------------------------------------
        # Manual assistant -- no simulation required.
        # -------------------------------------------------------------------
        st.divider()
        st.markdown("### Manual pick assistant")
        st.caption("For a draft you're running elsewhere — no simulation needed.")
        # Deliberately a toggle rather than an expander: the VOR board below is
        # itself an expander, and Streamlit refuses to nest one inside another.
        if st.toggle("Show manual assistant", key="fantasy_show_manual"):
            all_positions = sorted({str(player.get("position", "")) for player in projections} - {""})

            manual1, manual2 = st.columns(2)
            manual_round = int(
                manual1.number_input(
                    "Current round", min_value=1, max_value=num_rounds, value=1, key="fantasy_assistant_round"
                )
            )
            already_drafted = manual2.multiselect(
                "Players already off the board",
                [player.get("name") for player in projections],
                key="fantasy_assistant_drafted",
                help="Anyone taken so far — by you or anyone else. They are removed from the suggestions.",
            )

            manual3, manual4 = st.columns(2)
            my_positions = manual3.multiselect(
                "Positions already on my roster", all_positions, key="fantasy_my_drafted_positions"
            )
            position_filter = manual4.multiselect(
                "Only suggest these positions", all_positions, key="fantasy_assistant_position_filter"
            )

            manual_overall = next_pick_number(n_teams, draft_pick, manual_round, snake)
            manual_gap = picks_between(n_teams, draft_pick, manual_round, snake)
            manual_info1, manual_info2, manual_info3 = st.columns(3)
            manual_info1.metric("Your pick this round", f"#{manual_overall}")
            manual_info2.metric("Picks until your next", manual_gap)
            manual_info3.metric("Format", f"{scoring_label} · {draft_type}")

            drafted_names = set(already_drafted)
            manual_pool = [player for player in projections if player.get("name") not in drafted_names]
            manual_roster = [{"position": position} for position in my_positions]

            manual_capped = capped_positions(manual_roster)
            if manual_capped:
                st.caption(f"Roster full at {', '.join(manual_capped)} — hidden from suggestions below.")

            suggestions = run_analysis(
                "draft suggestions",
                lambda: suggest_draft_picks(
                    manual_pool,
                    st.session_state.fantasy_league_settings,
                    my_roster=manual_roster,
                    next_pick_overall=manual_overall,
                    picks_until_next=manual_gap,
                    limit=12,
                    positions=position_filter or None,
                ),
            )

            if not suggestions:
                st.info("No players match those filters — clear a filter or remove someone from the drafted list.")
            else:
                best = suggestions[0]
                st.success(
                    f"Best available at #{manual_overall}: **{best['name']}** "
                    f"({best['position']}, {best['team']}) — {best['projection']:,.0f} projected pts"
                )
                st.caption(best["rationale"])

                def _manual_timing(entry: dict) -> str:
                    if entry.get("must_draft_now"):
                        return "🔥 Won't last"
                    if entry.get("steal_bonus"):
                        return f"📉 Steal +{entry['steal_bonus']:.0f}"
                    if entry.get("reach_penalty"):
                        return f"⚠️ Reach -{entry['reach_penalty']:.0f}"
                    if entry.get("adp") is not None:
                        return "On time"
                    return "—"

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
                                "Timing": _manual_timing(entry),
                                "Scarcity": entry["scarcity"],
                                "Need": entry["need_label"],
                                "Score": entry["score"],
                            }
                            for entry in suggestions
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Projection": st.column_config.NumberColumn(
                            format="%.0f", help="Projected season fantasy points."
                        ),
                        "Value": st.column_config.NumberColumn(
                            format="%.1f", help="VORP: points above a replacement-level starter at this position."
                        ),
                        "ADP": st.column_config.NumberColumn(format="%.1f", help="FantasyPros expert consensus rank."),
                        "Timing": st.column_config.TextColumn(
                            help="Steal = fell past ADP (value). Reach = ADP says they'd likely still be there. "
                            "Won't last = their ADP says they're gone before your next turn — draft now."
                        ),
                        "Scarcity": st.column_config.NumberColumn(
                            format="%.0f",
                            help="Points of position value that disappear before your next pick, "
                            "weighted for how thin the position typically runs (TE/RB up, WR/QB down).",
                        ),
                        "Need": st.column_config.TextColumn(
                            help="Roster status at this position under your league's construction rules."
                        ),
                        "Score": st.column_config.NumberColumn(
                            format="%.1f", help="Combined value + timing + scarcity + need score."
                        ),
                    },
                )

            with st.expander("Full VOR draft board"):
                ranked = run_analysis(
                    "draft ranking",
                    lambda: rank_players_for_draft(manual_pool, st.session_state.fantasy_league_settings),
                )
                if ranked:
                    board_df = pd.DataFrame(ranked)[
                        ["overall_rank", "name", "position", "team", "points", "vor", "position_rank", "rationale"]
                    ]
                    if position_filter:
                        board_df = board_df[board_df["position"].isin(position_filter)]
                    st.dataframe(board_df, width="stretch", hide_index=True, height=320)

# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------
with tabs[1]:
    section_header("Fantasy Scoring", "Score any player projection under standard, half-PPR, PPR, or a custom rule set.")
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

    with st.expander("Stat → points mapping for this mode"):
        mapping_rows = [
            {"Stat": stat, "Points per unit": BASE_MULTIPLIERS[stat]}
            for stat in BASE_MULTIPLIERS
        ] + [{"Stat": "receptions", "Points per unit": RECEPTION_MULTIPLIER_BY_MODE.get(scoring_mode, 0.0)}]
        st.dataframe(pd.DataFrame(mapping_rows), width="stretch", hide_index=True)
        st.caption(
            "Default multipliers shown; a custom rule set above overrides any of these per stat. "
            "Missing stats on a player default to 0 rather than erroring — every category is optional."
        )

    projections = st.session_state.fantasy_projections
    # Keyed by player_id, not name: two players can share a display name, and a
    # name-keyed lookup would silently score the wrong one.
    players_by_id = {player.get("player_id"): player for player in projections if player.get("player_id")}
    if players_by_id:
        id_options = list(players_by_id)
        chosen_id = st.selectbox(
            "Player",
            id_options,
            format_func=lambda pid: f"{players_by_id[pid].get('name', pid)} ({players_by_id[pid].get('team', '')})",
            key="fantasy_score_player",
        )
        chosen_player = players_by_id.get(chosen_id)
        score_ready = scoring_mode != "custom" or custom_rules is not None
        if chosen_player and score_ready and st.button("Calculate points", type="primary", key="fantasy_calc_points"):
            scored = run_analysis(
                "scoring",
                lambda: calculate_fantasy_points(chosen_player, mode=scoring_mode, custom_rules=custom_rules),
            )
            if scored:
                scored["_authoritative_projection"] = chosen_player.get("projection") or chosen_player.get("expected_fantasy_points")
                scored["_player_name"] = chosen_player.get("name")
                st.session_state["fantasy_single_score"] = scored

        scored = st.session_state.get("fantasy_single_score")
        if scored:
            authoritative = scored.get("_authoritative_projection")
            if authoritative is not None:
                metric_col, note_col = st.columns([1, 2])
                metric_col.metric(f"{scored.get('_player_name', 'Player')} — season projection", f"{authoritative:,.1f}")
                note_col.caption(
                    f"Stat breakdown below totals {scored['total_points']:.1f} — season-total bonuses (300-yard games, etc.) "
                    "are computed on the season sum, not per game, so it can differ slightly from the season projection above, "
                    "which sums each game's own score."
                )
            else:
                st.metric("Total points", scored["total_points"])
            breakdown_items = sorted(scored["breakdown"].items(), key=lambda kv: -abs(kv[1]))
            breakdown_df = pd.DataFrame(breakdown_items, columns=["Stat", "Points"]).set_index("Stat")
            st.bar_chart(breakdown_df)
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
                        "total_points": score["total_points"],
                    }
                    for player, score in zip(projections, batch)
                ]
            ).sort_values("total_points", ascending=False)
            st.dataframe(batch_df, width="stretch", hide_index=True, height=320)
    else:
        empty_state("No player projections loaded", "Load a pool in League & draft setup above.", icon="📊")

# ---------------------------------------------------------------------------
# 3. Lineup
# ---------------------------------------------------------------------------
with tabs[2]:
    section_header("Lineup Optimizer", "Solve for the point-maximizing legal starting lineup, plus per-player start/sit calls.")
    roster = st.session_state.fantasy_roster
    projections = st.session_state.fantasy_projections
    if roster:
        roster_names = [player.get("name", "") for player in roster]
        id_by_name = {player.get("name"): player.get("player_id") for player in roster}
        solver = st.selectbox("Solver", ["auto", "ilp", "greedy"], key="fantasy_lineup_solver")
        lock_col, bench_col = st.columns(2)
        locked_names = lock_col.multiselect("Lock into starting lineup", roster_names, key="fantasy_locked_players")
        excluded_names = bench_col.multiselect("Force to bench (e.g. inactive)", roster_names, key="fantasy_excluded_players")
        constraints = {
            "solver": solver,
            "locked_player_ids": [id_by_name[name] for name in locked_names if id_by_name.get(name)],
            "excluded_player_ids": [id_by_name[name] for name in excluded_names if id_by_name.get(name)],
        }
        if st.button("Optimize lineup", type="primary", key="fantasy_optimize_lineup"):
            lineup = run_analysis(
                "lineup optimization",
                lambda: optimize_lineup(roster, projections, st.session_state.fantasy_league_settings, constraints),
            )
            if lineup:
                st.session_state["fantasy_lineup_result"] = lineup

        lineup = st.session_state.get("fantasy_lineup_result")
        if lineup:
            st.metric("Projected total points", lineup["total_points"])
            st.caption(f"Solved with the {lineup['solver']} solver.")
            starters_df = pd.DataFrame(lineup["starters"])[["slot", "name", "position", "points"]]
            st.dataframe(starters_df, width="stretch", hide_index=True)
            for warning in lineup["warnings"]:
                st.warning(warning)
            if lineup["bench"]:
                with st.expander("Bench"):
                    bench_df = pd.DataFrame(lineup["bench"])[["name", "position", "points"]]
                    st.dataframe(bench_df, width="stretch", hide_index=True)

        st.markdown("#### Start / sit advice")
        if st.button("Get start/sit advice", key="fantasy_start_sit"):
            advice = run_analysis(
                "start/sit advice",
                lambda: start_sit_advice(roster, projections, st.session_state.fantasy_league_settings),
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
# 4. Waivers
# ---------------------------------------------------------------------------
with tabs[3]:
    section_header("Waiver Recommendations", "Rank free agents against your roster needs, with suggested FAAB/auction bids.")
    available = st.session_state.fantasy_available
    if available:
        col1, col2 = st.columns(2)
        current_week = col1.number_input("Current week", min_value=1, max_value=18, value=5, key="fantasy_waiver_week")
        budget = col2.number_input("Remaining FAAB / auction budget", min_value=0.0, value=100.0, key="fantasy_waiver_budget")
        if st.button("Get waiver recommendations", type="primary", key="fantasy_run_waivers"):
            league_state = {
                "league_settings": st.session_state.fantasy_league_settings,
                "my_roster": st.session_state.fantasy_roster,
                "current_week": int(current_week),
            }
            scoring_mode_override = st.session_state.fantasy_league_settings.get("scoring_mode")
            recs = run_analysis(
                "waiver recommendations",
                lambda: waiver_recommendations(league_state, available, scoring_mode_override, budget=float(budget)),
            )
            if recs:
                st.session_state["fantasy_waiver_result"] = recs

        recs = st.session_state.get("fantasy_waiver_result")
        if recs:
            bid_col = "suggested_auction_bid" if st.session_state.fantasy_league_settings.get("is_auction") else "suggested_faab_bid"
            columns_wanted = [
                col for col in
                ["waiver_rank", "name", "position", "composite_score", "replacement_value", bid_col, "rationale"]
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
# 5. Trades
# ---------------------------------------------------------------------------
with tabs[4]:
    section_header("Trade Analyzer", "Monte Carlo simulate a proposed trade's rest-of-season value for both sides.")
    projections = st.session_state.fantasy_projections
    roster = st.session_state.fantasy_roster
    pool_names = sorted({player.get("name") for player in projections} | {player.get("name") for player in roster} - {None})
    if pool_names:
        col1, col2 = st.columns(2)
        team_a_gives = col1.multiselect("Team A sends", pool_names, key="fantasy_trade_a_gives")
        team_b_gives = col2.multiselect("Team B sends", pool_names, key="fantasy_trade_b_gives")
        col3, col4, col5 = st.columns(3)
        iterations = col3.number_input(
            "Monte Carlo iterations", min_value=100, max_value=20000, value=2000, step=100, key="fantasy_trade_iterations"
        )
        weeks_remaining = col4.number_input("Weeks remaining", min_value=1, max_value=17, value=10, key="fantasy_trade_weeks")
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
                        league_settings=st.session_state.fantasy_league_settings,
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
# 6. Cheat Sheets
# ---------------------------------------------------------------------------
with tabs[5]:
    section_header("Tiering & Cheat Sheets", "Cluster the ranked player pool into tiers and export a printable or CSV cheat sheet.")
    projections = st.session_state.fantasy_projections
    if not projections:
        empty_state("No player pool loaded", "Load player projections above to generate tiers.", icon="🗂️")
    else:
        TIER_COLORS = {1: "#0f8a4c", 2: "#4c8f2f", 3: "#c9a227", 4: "#c9662a", 5: "#b3401f"}

        def _tier_background(tier: Any) -> str:
            color = TIER_COLORS.get(int(tier), "#6b7280") if pd.notna(tier) else "#6b7280"
            return f"background-color: color-mix(in srgb, {color} 22%, transparent)"

        max_tiers = st.slider("Max tiers per position", min_value=2, max_value=10, value=5, key="fantasy_max_tiers")
        if st.button("Generate tiers", type="primary", key="fantasy_generate_tiers"):
            ranked = run_analysis(
                "draft ranking",
                lambda: rank_players_for_draft(projections, st.session_state.fantasy_league_settings),
            )
            if ranked:
                tiered = run_analysis("tiering", lambda: tier_players(ranked, max_tiers=int(max_tiers)))
                if tiered:
                    st.session_state["fantasy_tiered_players"] = tiered
                combined = run_analysis(
                    "combined tiering",
                    lambda: combined_tier(ranked, st.session_state.fantasy_league_settings, n_teams=n_teams, max_tiers=int(max_tiers)),
                )
                if combined:
                    st.session_state["fantasy_combined_tiers"] = combined

        combined = st.session_state.get("fantasy_combined_tiers")
        if combined:
            st.markdown("#### ESPN-style cheat sheet: ADP + projection + scarcity, blended")
            st.caption(
                "**ADP tier** = replacement-level bands by real market ADP (Elite/Starter/Flex/Streamer/Deep Bench). "
                "**Scarcity tier** = VORP percentile within position. **Value tier** = the volatility-aware cluster tier. "
                "**Combined** averages all three — a player whose market ADP and real production disagree lands in between, "
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
                        group_df.style.apply(
                            lambda row: [_tier_background(row["Combined Tier"])] * len(row), axis=1
                        ),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "ADP": st.column_config.NumberColumn(format="%.1f"),
                            "Projection": st.column_config.NumberColumn(format="%.1f"),
                            "Value (VORP)": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )

        tiered = st.session_state.get("fantasy_tiered_players")
        if tiered:
            with st.expander("Value + volatility tiers (k-means clustering)"):
                tiers_df = pd.DataFrame(tiered)[["position", "tier", "tier_label", "overall_rank", "name", "team", "points", "vor"]]
                tiers_df = tiers_df.sort_values(["position", "tier", "overall_rank"])
                st.dataframe(
                    tiers_df.style.apply(lambda row: [_tier_background(row["tier"])] * len(row), axis=1),
                    width="stretch",
                    hide_index=True,
                    height=360,
                )

            printable = generate_printable_cheatsheet(tiered)
            st.text_area("Printable cheat sheet", printable, height=240)
            csv_text = export_cheatsheet_csv(tiered)
            st.download_button(
                "Download cheat sheet CSV",
                csv_text,
                file_name="fantasy_cheatsheet.csv",
                mime="text/csv",
                key="fantasy_download_cheatsheet",
            )
        elif not combined:
            empty_state(
                "No tiers generated yet",
                'Click "Generate tiers" to build the cheat sheet.',
                icon="🗂️",
            )
