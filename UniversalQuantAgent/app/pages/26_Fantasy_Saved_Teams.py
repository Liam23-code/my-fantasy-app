"""Saved Teams library for multi-league fantasy management."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]

import streamlit as st
from app.page_runtime import (
    apply_global_theme,
    empty_state,
    page_header,
    section_header,
)
from app.style import gold_glow_chart, stacked_card_html
from fantasy.my_team_manager import (
    create_new_team_save,
    delete_team_save,
    list_saved_teams,
)


def _open_team(team_id: str) -> None:
    """Select a save for the session and navigate to its manager."""
    st.session_state["fantasy_selected_team_id"] = team_id
    st.query_params["team_id"] = team_id
    st.switch_page("pages/28_Fantasy_My_Team.py")


def _timestamp_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Not yet updated"
    return text.replace("T", " ").replace("Z", " UTC")[:20]


apply_global_theme()

page_header(
    "Saved Teams",
    "A separate save file for every redraft, keeper, dynasty, or office league.",
    eyebrow="Fantasy · team library",
)

with st.expander("Create a team save", expanded=False):
    name = st.text_input(
        "Team name",
        value="My Fantasy Team",
        key="saved-teams-new-name",
        max_chars=80,
    )
    league = st.text_input(
        "League name",
        value="Fantasy League",
        key="saved-teams-new-league",
        max_chars=100,
    )
    if st.button("Create New Team Save", type="primary", key="saved-teams-create"):
        try:
            created = create_new_team_save(name=name.strip() or None, league=league.strip() or None)
        except (FileExistsError, OSError, TypeError, ValueError) as error:
            st.error(f"The team save could not be created: {error}")
        else:
            st.session_state["fantasy_selected_team_id"] = created["team_id"]
            st.success(f"Created {created['name']}.")
            st.switch_page("pages/28_Fantasy_My_Team.py")

try:
    saved_teams = list_saved_teams()
except OSError as error:
    saved_teams = []
    st.error(f"Saved teams could not be read: {error}")

section_header("Your Team Saves", "Select a card to open that exact save in My Team.")

if not saved_teams:
    empty_state(
        "No team saves yet",
        "Create a new save above. Nothing from Mock Draft is written here automatically.",
        icon="📁",
    )
else:
    roster_counts = [team["player_count"] for team in saved_teams if team.get("is_valid")]
    roster_labels = [team["name"] for team in saved_teams if team.get("is_valid")]
    if roster_counts and any(roster_counts):
        st.plotly_chart(
            gold_glow_chart(
                roster_counts,
                x=roster_labels,
                title="Roster depth across saves",
                name="Rostered players",
                height=280,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
            key="saved-teams-roster-depth",
        )

    for index, team in enumerate(saved_teams):
        team_id = str(team["team_id"])
        valid = bool(team.get("is_valid", True))
        st.markdown(
            stacked_card_html(
                team.get("name") or team_id,
                team.get("league") or "Fantasy League",
                kicker=f"Save ID · {team_id}",
                stats={
                    "Players": team.get("player_count", 0),
                    "Updated": _timestamp_label(team.get("updated_at")),
                },
                rarity_rank=index + 1,
                card_id=f"saved-team-{team_id}",
                extra_class="saved-team-card",
            ),
            unsafe_allow_html=True,
        )
        if valid:
            if st.button(
                f"Open {team.get('name') or team_id}",
                key=f"saved-team-open-{team_id}",
                use_container_width=True,
            ):
                _open_team(team_id)
        else:
            st.error(f"This save is damaged and cannot be opened. {team.get('error', '')}")

        with st.expander(f"Manage {team.get('name') or team_id}"):
            confirmed = st.checkbox(
                "I understand this permanently deletes this save file.",
                key=f"saved-team-delete-confirm-{team_id}",
            )
            if st.button(
                "Delete team save",
                key=f"saved-team-delete-{team_id}",
                disabled=not confirmed,
            ):
                try:
                    removed = delete_team_save(team_id)
                except (OSError, ValueError) as error:
                    st.error(f"The save could not be deleted: {error}")
                else:
                    if st.session_state.get("fantasy_selected_team_id") == team_id:
                        st.session_state.pop("fantasy_selected_team_id", None)
                    if removed:
                        st.success("Team save deleted.")
                    st.rerun()
