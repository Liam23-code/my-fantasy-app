from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

# ``fantasy`` is editable-installed for the app; adding the engine root keeps a
# local/offline launch self-contained now that this page reads the live
# player-status overlay for its availability badge.
_ENGINE_ROOT = str(Path(__file__).resolve().parents[3] / "fantasy_engine")
if Path(_ENGINE_ROOT).is_dir() and _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

import pandas as pd
import streamlit as st

from app.page_runtime import apply_global_theme, confidence_ring, nfl_profile_header, reliability_bar, render_insight_list, run_analysis
from app.status_ui import player_status_badge, render_status_strip
from modules.nfl import latest_completed_nfl_season
from modules import nfl_matchup_features as nmf
from modules.nfl_projections import project_nfl_player

apply_global_theme()


def _project_with_matchup_pack(player, opponent, season, mode, comparison, use_pack):
    """The projection, optionally adjusted by the NFL upgrade pack for this opponent.

    The pack needs the player's team (known only once the stats are resolved), so the base projection runs
    first and the pack pass runs second. Everything about the pack is fail-soft: any problem hands back the
    base projection unchanged.
    """
    base = project_nfl_player(player, opponent, season, mode, comparison)
    if not use_pack or not str(opponent or "").strip() or not base.get("team") or base.get("position") not in ("QB", "RB", "WR", "TE"):
        return base
    try:
        context = nmf.build_game_context(
            base["team"], opponent, season=season,
            player={"name": base["player"], "player_id": base.get("player_id"), "team": base["team"], "position": base["position"]},
        )
        if not context or not context.get("has_features"):
            base["upgrade_pack"] = {
                "applied": False, "neutral": True, "has_signal": False, "adjustments": [], "summary": {}, "explanations": [],
                "warnings": ["No matchup features were available for this team and opponent, so nothing was adjusted."],
                "guard": [], "stage_order": [], "signals": {},
            }
            return base
        return project_nfl_player(player, opponent, season, mode, comparison, pack_context=context)
    except Exception:  # noqa: BLE001 - optional layer
        return base


st.title("NFL Player Projections")
st.write("Project volume, yardage, touchdowns, and fantasy points with matchup and weather context.")

# Display-only. The projection below is the stored model output for the chosen
# player and opponent; the badge reports availability next to it and never
# rescales, zeroes, or hides a number.
render_status_strip(
    "nfl_projections_refresh_player_status",
    hint="the projection below is the stored model output. ",
)

with st.form("nfl_projection_form"):
    one, two, three = st.columns([2, 2, 1])
    player = one.text_input("NFL player", "Saquon Barkley")
    opponent = two.text_input("Opponent", "DAL")
    season = three.number_input("Season", 1999, latest_completed_nfl_season(), latest_completed_nfl_season())
    left, right = st.columns(2)
    mode = left.radio("Mode", ["Raw", "Adjusted"], index=1, horizontal=True)
    comparison = right.radio("Comparison", ["League", "Position"], horizontal=True)
    use_pack = st.checkbox(
        "Use the matchup upgrade pack",
        value=True,
        key="nfl_projection_use_pack",
        help="Adjusts the line for this opponent: pass-rush / protection collapse risk, explosion chance and game script. "
             "Priors, not a fitted model; a league-average matchup is unchanged.",
    )
    submitted = st.form_submit_button("Project NFL player", type="primary")
if submitted:
    result = run_analysis("NFL projection", lambda: _project_with_matchup_pack(player, opponent, int(season), mode, comparison, use_pack))
    if result:
        st.session_state["nfl_projection"] = result
result = st.session_state.get("nfl_projection")
if not result:
    st.info("Choose a player and opponent to build a projection.")
else:
    nfl_profile_header(result["player"], result["team"], f"{result['position']} vs {result['opponent']}")
    projected_player = {
        "name": result["player"],
        "team": result["team"],
        "position": result["position"],
        "player_id": result.get("player_id"),
    }
    st.markdown(
        f"**Availability:** {player_status_badge(projected_player)} · "
        f"{result['player']} ({result['position']})"
    )
    st.caption(
        "Current availability, shown alongside the projection. Every number below is the model's "
        "stored output and is not adjusted by this badge."
    )
    projection, drivers, reliability, context = st.tabs(["Projection", "Model Drivers", "Reliability", "Player Context"])
    with projection:
        p = result["projection"]
        position_metrics = {
            "QB": [("Fantasy points", "expected_fantasy_points"), ("Passing yards", "passing_yards"), ("Rushing yards", "rushing_yards"), ("Interceptions", "interceptions")],
            "RB": [("Fantasy points", "expected_fantasy_points"), ("Carries", "carries"), ("Rushing yards", "rushing_yards"), ("Receiving yards", "receiving_yards")],
            "WR": [("Fantasy points", "expected_fantasy_points"), ("Targets", "targets"), ("Receptions", "receptions"), ("Receiving yards", "receiving_yards")],
            "TE": [("Fantasy points", "expected_fantasy_points"), ("Targets", "targets"), ("Receptions", "receptions"), ("Receiving yards", "receiving_yards")],
            "DEF": [("Fantasy points", "expected_fantasy_points"), ("Pressure rate", "pressure_rate"), ("Sack probability", "sack_probability"), ("EPA allowed", "epa_allowed")],
        }
        metric_specs = position_metrics[result["position"]]
        cols = st.columns(4)
        for column, (label, key) in zip(cols, metric_specs):
            column.metric(label, p.get(key, 0))
        touchdown_metrics = {
            "QB": [("Passing TDs", "passing_tds"), ("Rushing TDs", "rushing_tds")],
            "RB": [("Rushing TDs", "rushing_tds"), ("Receiving TDs", "receiving_tds")],
            "WR": [("Receiving TDs", "receiving_tds")],
            "TE": [("Receiving TDs", "receiving_tds")],
        }.get(result["position"], [])
        if touchdown_metrics:
            td_columns = st.columns(len(touchdown_metrics))
            for column, (label, key) in zip(td_columns, touchdown_metrics):
                column.metric(label, f"{p.get(key, 0):.2f}")
        st.success(result["confidence"]["label"])
        pack = result.get("upgrade_pack")
        if pack is not None:
            with st.expander("Matchup upgrade pack", expanded=bool(pack.get("applied"))):
                if pack.get("applied"):
                    for sentence in pack.get("explanations", []):
                        st.write(sentence)
                    rows = [
                        {"Stat": field.replace("_", " "), "Before": entry["before"], "After": entry["after"],
                         "Change": f"{(entry['after'] / entry['before'] - 1) * 100:+.0f}%" if entry.get("before") else ""}
                        for field, entry in pack.get("summary", {}).items()
                    ]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True)
                    st.caption(
                        "Weights are priors, not fitted; several inputs are proxies or neutral where no free feed exists; "
                        "the baseline is this model's own per-game rate, which can already lean on a matchup score."
                    )
                else:
                    st.write("The pack left this projection unchanged.")
                for warning in pack.get("warnings", [])[:4]:
                    st.caption(f"Note: {warning}")
        chart_keys = [key for _, key in metric_specs[1:]]
        st.bar_chart(pd.DataFrame({"Projection": {key: p.get(key, 0) for key in chart_keys}}))
    with drivers:
        render_insight_list(result["drivers"], "Projection drivers")
    with reliability:
        left, right = st.columns([1, 2])
        with left:
            confidence_ring(result["confidence"]["score"], "Projection confidence", "nfl_projection_confidence")
        with right:
            reliability_bar(result["confidence"]["score"], "Model reliability")
            st.metric("Confidence interval", f"{result['confidence']['low']:.1f}–{result['confidence']['high']:.1f} FP")
            st.metric("Volatility", f"{result['volatility_score']:.0f}/100")
    with context:
        st.json({"matchup": result["analysis"]["matchup"], "weather": result["analysis"]["weather"], "pace": result["analysis"]["pace"]}, expanded=True)
    for warning in result.get("warnings", []):
        st.caption(f"Data note: {warning}")
