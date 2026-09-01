"""Shared setup, data loading, and card rendering for the Fantasy pages.

The Fantasy section is three pages -- Draft Room, Draft Assistant, and Season
Tools -- and all three need the same league settings, the same player pool, and
the same card vocabulary. That shared surface lives here rather than being
copied into each page, so a change to (say) how the pool is projected forward
lands on every page at once instead of two out of three.

Import this only *after* a page has run its ``sys.path`` preamble; it imports
``app.page_runtime`` and the ``fantasy`` engine, neither of which resolves
until that preamble has run.

The pool
--------
Every page draws its players from :func:`fantasy.projections.load_forward_projections`,
not from raw prior-season actuals. That distinction is the whole reason the UI
can print "2026 Projections" honestly: the numbers on screen are a forward
estimate for the season being drafted, with the prior season's production as
their input rather than their answer. :func:`season_context` returns both
seasons so a page can always say which is which.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from app.page_runtime import empty_state, pill

from fantasy.data_loader import (
    RealDataUnavailable,
    drop_synthetic,
    latest_completed_season,
    validate_players,
)
from fantasy.grader import letter_grade
from fantasy.projections import load_forward_projections, project_forward, projection_season_label
from quant.quant_engine import compute_breakout_probability

SCORING_LABELS = {"PPR": "ppr", "Half-PPR": "half-ppr", "Standard": "standard"}

DEFAULT_LEAGUE_SETTINGS: dict[str, Any] = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
    "max_players_per_nfl_team": None,
    "faab_budget": 100,
}

#: Session keys every Fantasy page expects to exist. Shared across the three
#: pages on purpose: a league configured in the Draft Room is the same league
#: the Season Tools page reasons about.
_STATE_DEFAULTS: dict[str, Any] = {
    "fantasy_league_settings": None,  # replaced with a copy in init_state
    "fantasy_projections": [],
    "fantasy_roster": [],
    "fantasy_available": [],
    "fantasy_live_draft": None,
    "fantasy_live_draft_narrated": 0,
}


# ---------------------------------------------------------------------------
# Player pool
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_pool(source_season: int, target_season: int, scoring_mode: str) -> tuple[list[dict[str, Any]], str]:
    """Fetch real player data and project it forward. Cached so page switches are instant."""
    players = load_forward_projections(
        season=source_season, target_season=target_season, scoring_mode=scoring_mode
    )
    basis = players[0].get("projection_basis", "") if players else ""
    return players, basis


def _clean_uploaded(players: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop synthetic entries from uploaded JSON and validate the rest."""
    if not isinstance(players, list):
        return [], []
    return validate_players(drop_synthetic(players), require_projection=False)


def init_state() -> None:
    for key, value in _STATE_DEFAULTS.items():
        if key == "fantasy_league_settings":
            st.session_state.setdefault(key, dict(DEFAULT_LEAGUE_SETTINGS))
        else:
            st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# League & data setup -- rendered identically at the top of all three pages.
# ---------------------------------------------------------------------------
def league_setup(show_uploads: bool = True) -> dict[str, Any]:
    """Render the shared league/draft/data expander and return everything it decided.

    Returns ``{"n_teams", "draft_pick", "num_rounds", "scoring_label",
    "scoring_mode", "draft_type", "snake", "source_season", "target_season",
    "projections", "projection_basis", "load_error"}``. Widget keys are shared
    across the Fantasy pages so a league configured on one page is still
    configured on the next.
    """
    init_state()
    settings = dict(st.session_state.fantasy_league_settings)

    with st.expander("League & draft setup", expanded=not st.session_state.fantasy_projections):
        col1, col2, col3 = st.columns(3)
        n_teams = int(
            col1.number_input(
                "Number of teams", min_value=2, max_value=32,
                value=int(settings.get("n_teams", 12)), key="fantasy_n_teams",
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
        source_season = int(
            col6.number_input(
                "Source season (actuals)",
                min_value=2000,
                max_value=latest_completed_season() + 1,
                value=latest_completed_season(),
                key="fantasy_source_season",
                help="The completed season whose real production feeds the model. "
                "Projections are produced for the following season.",
            )
        )
        target_season = source_season + 1

        settings.update({"n_teams": n_teams, "scoring_mode": scoring_mode})
        st.session_state.fantasy_league_settings = settings

        st.caption(
            f"Every number on the Fantasy pages is a **{projection_season_label(target_season)}** figure: "
            f"{source_season} production, regressed for availability and scoring rate, then reconciled "
            "against current market ADP. Rookies have no prior-season production and do not appear."
        )

        # --- load and project the real pool ---------------------------------
        load_error: str | None = None
        projection_basis = ""
        try:
            with st.spinner(f"Building {target_season} projections from {source_season} NFL data..."):
                real_players, projection_basis = load_pool(source_season, target_season, scoring_mode)
        except RealDataUnavailable as error:
            real_players, load_error = [], str(error)
        except Exception as error:  # network hiccup, upstream schema change, ...
            real_players, load_error = [], f"Unexpected error loading NFL data: {error}"

        uploaded_rejects: list[dict[str, Any]] = []
        upload_note = ""
        if show_uploads:
            st.markdown("**Custom data (optional)** — upload JSON to override any of the pools below.")
            up1, up2, up3 = st.columns(3)
            projections_file = up1.file_uploader("Projections JSON", type="json", key="fantasy_upload_projections")
            roster_file = up2.file_uploader("My roster JSON", type="json", key="fantasy_upload_roster")
            available_file = up3.file_uploader("Available players JSON", type="json", key="fantasy_upload_available")
        else:
            projections_file = roster_file = available_file = None

        if projections_file is not None:
            try:
                cleaned, uploaded_rejects = _clean_uploaded(json.load(projections_file))
                # An upload that already declares the season it projects is
                # taken at its word. Anything else is prior-season data as far
                # as this app can tell, and gets the same forward treatment the
                # built-in pool gets -- otherwise the board would silently mix
                # projected and unprojected players on one screen.
                already_forward = bool(cleaned) and all(
                    player.get("projection_season") == target_season for player in cleaned
                )
                if already_forward:
                    upload_note = f"used as-is (already tagged {target_season})"
                else:
                    cleaned = project_forward(cleaned, target_season=target_season)
                    upload_note = f"projected forward to {target_season}"
                st.session_state.fantasy_projections = cleaned
                projection_basis = f"uploaded projections — {upload_note}"
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
            st.success(
                f"{len(projections)} real players · {projection_season_label(target_season)} · {projection_basis}"
            )
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
                    f"No {', '.join(missing_positions)} in the nflverse pool — those slots will go "
                    "unfilled in a mock draft."
                )

    return {
        "n_teams": n_teams,
        "draft_pick": draft_pick,
        "num_rounds": num_rounds,
        "scoring_label": scoring_label,
        "scoring_mode": scoring_mode,
        "draft_type": draft_type,
        "snake": snake,
        "source_season": source_season,
        "target_season": target_season,
        "projections": st.session_state.fantasy_projections,
        "projection_basis": projection_basis,
        "load_error": load_error,
    }


def quant_breakout_by_player(players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Breakout-probability rows from the Quant Engine, keyed by player id.

    The Draft Room and Draft Assistant recommendation tables are built by
    :mod:`fantasy.assistant`'s own backtest-validated VORP/ADP scoring, not
    the Quant Engine -- that scoring stays as-is (see the module docstring
    there for why). This adds Quant's independent breakout read as a second,
    non-authoritative signal next to it, the same "preserve the primary
    ranking, append a Quant column" pattern the Season Tools page already
    uses for waiver and trade recommendations.
    """
    if not players:
        return {}
    by_player = compute_breakout_probability(players).get("by_player", {})
    return {str(key): dict(value) for key, value in by_player.items()}


def breakout_pill(entry: dict[str, Any]) -> str:
    """Quant's breakout-probability read, banded so the color carries the meaning."""
    probability = entry.get("quant_breakout_probability")
    if probability is None:
        return ""
    probability = float(probability)
    if probability >= 0.65:
        return pill(f"🚀 Breakout {probability:.0%}", "success")
    if probability >= 0.40:
        return pill(f"Breakout {probability:.0%}", "neutral")
    return ""


def require_pool(setup: dict[str, Any], what: str) -> bool:
    """Render an empty state and return ``False`` when there is no pool to work with."""
    if setup["projections"]:
        return True
    empty_state(
        "No player pool loaded",
        f"Fix the data source in League & draft setup above, or upload a projections JSON, to {what}.",
        icon="🏈",
    )
    return False


# ---------------------------------------------------------------------------
# Card vocabulary
# ---------------------------------------------------------------------------
def fmt(value: Any, spec: str = ",.0f", dash: str = "—") -> str:
    """Format a number for a card, degrading to a dash rather than erroring."""
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return dash


def proximity_pill(entry: dict[str, Any]) -> str:
    """How close this player's ADP sits to the pick being made -- color-coded."""
    proximity = entry.get("adp_proximity")
    if proximity is None:
        return pill("No ADP", "neutral")
    if proximity <= 5:
        return pill(f"On ADP · {proximity:.0f}", "success")
    if proximity <= 15:
        return pill(f"{proximity:.0f} picks off ADP", "neutral")
    return pill(f"{proximity:.0f} picks off ADP", "warning")


def scarcity_pill(entry: dict[str, Any]) -> str:
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


def need_pill(entry: dict[str, Any]) -> str:
    label = entry.get("need_label")
    if label == "Fills need":
        return pill("Fills need", "info")
    if label == "Depth":
        return pill("Depth", "neutral")
    return ""


#: Live availability -> (badge text, pill tone). HEALTHY / missing shows nothing.
_STATUS_BADGES: dict[str, tuple[str, str]] = {
    "OUT": ("🚑 OUT", "danger"),
    "HOLDOUT": ("💼 Holdout", "danger"),
    "SUSPENDED": ("🚫 Suspended", "danger"),
    "DOUBTFUL": ("⚠️ Doubtful", "warning"),
    "QUESTIONABLE": ("❓ Questionable", "warning"),
}


def status_pill(entry: dict[str, Any]) -> str:
    """A player-availability badge from the live status overlay, or ``""``."""
    badge = _STATUS_BADGES.get(str(entry.get("status") or "").strip().upper())
    if not badge:
        return ""
    text, tone = badge
    return pill(text, tone)


def timing_pill(overall_pick: Any, adp: Any) -> str:
    """Steal / reach / on-time, from the gap between a pick number and its ADP."""
    if adp is None or overall_pick is None:
        return ""
    delta = float(overall_pick) - float(adp)
    if delta >= 15:
        return pill(f"📉 Steal +{delta:.0f}", "success")
    if delta <= -15:
        return pill(f"⚠️ Reach {delta:+.0f}", "warning")
    return pill("On time", "neutral")


def _stat_row(stats: list[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="stat"><span class="k">{escape(key)}</span><span class="v">{escape(value)}</span></div>'
        for key, value in stats
    )


def player_card_html(entry: dict[str, Any], rank_label: str, taken_by: str | None = None) -> str:
    """One recommendation rendered as a premium card."""
    pills = "".join(
        [status_pill(entry), proximity_pill(entry), scarcity_pill(entry), need_pill(entry), breakout_pill(entry)]
    )
    if taken_by:
        pills = pill(f"🔒 Taken by {taken_by}", "danger") + pills
    stats = [
        ("Proj", fmt(entry.get("projection"))),
        ("VORP", fmt(entry.get("vorp"), "+.0f")),
        ("ADP", fmt(entry.get("adp"), ".1f")),
        (f"{entry.get('position', '')} rank", fmt(entry.get("position_rank"), ".0f")),
    ]
    classes = "player-card is-taken" if taken_by else "player-card"
    return (
        f'<div class="{classes}">'
        f'<div class="rank">{escape(rank_label)}</div>'
        f'<div class="name">{escape(str(entry.get("name", "—")))}</div>'
        f'<div class="meta">{escape(str(entry.get("position", "")))} · {escape(str(entry.get("team", "")))}</div>'
        f'<div class="stat-row">{_stat_row(stats)}</div>'
        f"<div>{pills}</div>"
        f"</div>"
    )


def pick_card_html(pick: dict[str, Any], source: dict[str, Any] | None = None) -> str:
    """One of my own drafted players, rendered as a card.

    ``source`` is the player's row from the projection pool. The pick dict's
    own ``team`` is the *fantasy* team ("Team 5"), so the NFL team has to come
    from the pool row instead.
    """
    source = source or {}
    stats = [
        ("Proj", fmt(pick.get("projection"))),
        ("VORP", fmt(pick.get("vor"), "+.0f")),
        ("ADP", fmt(pick.get("adp"), ".1f")),
    ]
    return (
        '<div class="player-card">'
        f'<div class="rank">Round {pick["round"]} · Pick #{pick["overall_pick"]}</div>'
        f'<div class="name">{escape(str(pick.get("name", "—")))}</div>'
        f'<div class="meta">{escape(str(pick.get("position", "")))} · {escape(str(source.get("team") or ""))}</div>'
        f'<div class="stat-row">{_stat_row(stats)}</div>'
        f'<div>{timing_pill(pick.get("overall_pick"), pick.get("adp"))}</div>'
        "</div>"
    )


def drafted_card_html(entry: dict[str, Any]) -> str:
    """A player the simulation has already taken, offered back for an override."""
    stats = [
        ("Proj", fmt(entry.get("projection"))),
        ("VORP", fmt(entry.get("vor"), "+.0f")),
        ("ADP", fmt(entry.get("adp"), ".1f")),
    ]
    taken = pill(f"🔒 {entry.get('drafted_by', '')} · #{entry.get('overall_pick', '')}", "danger")
    return (
        '<div class="player-card is-taken">'
        f'<div class="rank">Round {entry.get("round", "")} · Pick #{entry.get("overall_pick", "")}</div>'
        f'<div class="name">{escape(str(entry.get("name", "—")))}</div>'
        f'<div class="meta">{escape(str(entry.get("position", "")))} · {escape(str(entry.get("nfl_team", "")))}</div>'
        f'<div class="stat-row">{_stat_row(stats)}</div>'
        f"<div>{taken}</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Team grade rendering -- shared by the live panel and the final report.
# ---------------------------------------------------------------------------
def _grade_tone(score: float) -> str:
    # Anchored to the re-centred 0-100 scale (~50 = league-average = C):
    # green for a genuine B+, blue for average-and-up, amber for a weak team.
    if score >= 72:
        return "success"
    if score >= 46:
        return "info"
    if score >= 35:
        return "warning"
    return "danger"


def grade_headline_html(overall: dict[str, Any], caption: str = "") -> str:
    """The big score-and-letter block at the top of a grade panel."""
    score = float(overall.get("score", 0.0))
    tone = _grade_tone(score)
    caption_html = f'<div class="meta">{escape(caption)}</div>' if caption else ""
    return (
        '<div class="player-card">'
        '<div class="rank">Team grade</div>'
        f'<div class="name">{escape(overall.get("grade", letter_grade(score)))} · {score:.0f}/100</div>'
        f"{caption_html}"
        f"<div>{pill(f'{score:.0f} / 100', tone)}</div>"
        "</div>"
    )


def positional_strength_frame(positions: list[dict[str, Any]]) -> pd.DataFrame:
    """Positional scores as a chartable frame, weakest group last."""
    rows = [
        {
            "Position": group["position"],
            "Score": group["score"],
            "Grade": group["grade"],
            "Starters": len(group["starters"]),
            "Slots": group["starter_slots"],
            "Your points": group["starter_points"],
            "League average": group["league_average_points"],
            "vs average": group["points_vs_average"],
            "ADP value": group["adp_value_picks"],
            "Depth": group["depth"],
        }
        for group in positions
        if group.get("applicable")
    ]
    return pd.DataFrame(rows)


def render_grade_panel(report: dict[str, Any], detailed: bool = False) -> None:
    """Render a team grade: headline, component scores, and positional strength.

    ``detailed=True`` adds the full post-draft report -- the positional chart,
    the best and worst pick, and the ADP value summary.
    """
    overall = report["overall"]
    st.markdown(grade_headline_html(overall, overall.get("rationale", "")), unsafe_allow_html=True)

    components = overall["components"]
    weights = overall["weights"]
    columns = st.columns(len(components))
    labels = {
        "projected_points": "Projected points",
        "positional_balance": "Positional balance",
        "value_vs_adp": "Value vs ADP",
        "risk_profile": "Risk profile",
    }
    for column, (key, value) in zip(columns, components.items()):
        column.metric(labels.get(key, key), f"{value:.0f}", help=f"Weighted {weights[key]:.0%} of the overall grade.")

    frame = positional_strength_frame(report["positions"])
    if frame.empty:
        st.caption("No startable positions to grade yet.")
        return

    st.markdown("##### Positional strength")
    st.caption(
        "Each score is this position group against the *average team in your league*, measured from the "
        "pool itself, then weighted for how scarce the position is and how much ADP value you captured there. "
        "50 is exactly average."
    )
    st.bar_chart(frame.set_index("Position")["Score"], height=220)

    if not detailed:
        weakest = frame.sort_values("Score").iloc[0]
        st.caption(f"Weakest group: **{weakest['Position']}** ({weakest['Score']:.0f}/100, {weakest['Grade']}).")
        return

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.0f", help="0-100 against this league's average team."),
            "Your points": st.column_config.NumberColumn(format="%.0f"),
            "League average": st.column_config.NumberColumn(format="%.0f"),
            "vs average": st.column_config.NumberColumn(format="%+.0f"),
            "ADP value": st.column_config.NumberColumn(
                format="%+.0f", help="Picks of value captured at this position (positive = taken later than ADP)."
            ),
        },
    )
    with st.expander("Why each group scored what it did"):
        for group in report["positions"]:
            if group.get("applicable"):
                st.markdown(f"**{group['position']} — {group['score']:.0f} ({group['grade']})** — {group['rationale']}")

    best, worst = report.get("best_pick"), report.get("worst_pick")
    value = report.get("value_vs_adp") or {}
    st.markdown("##### Best and worst picks")
    if not best:
        st.caption("No picks to grade yet.")
        return
    best_col, worst_col, value_col = st.columns(3)
    best_col.metric(
        "Best pick",
        str(best.get("name", "—")),
        f"{best.get('value_picks') or 0:+.0f} picks vs ADP",
        help=f"{best['position']} · {best['projection']:,.0f} proj · {best['vorp']:+.0f} VORP",
    )
    worst_col.metric(
        "Worst pick",
        str(worst.get("name", "—")) if worst else "—",
        f"{(worst or {}).get('value_picks') or 0:+.0f} picks vs ADP",
        help=(
            f"{worst['position']} · {worst['projection']:,.0f} proj · {worst['vorp']:+.0f} VORP"
            if worst
            else "No picks recorded."
        ),
    )
    value_col.metric(
        "Value gained vs ADP",
        f"{value.get('total_picks', 0):+.0f} picks",
        f"{value.get('total_points', 0):+,.0f} pts",
        help="Summed across every pick with a market ADP. Positive means you consistently drafted "
        "players later than the market expected them to go.",
    )
