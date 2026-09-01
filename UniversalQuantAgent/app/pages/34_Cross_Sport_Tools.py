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

# Cross-Sport Tools: Player Comparison (same-sport, pick a sport via the
# toggle) and Cross-Sport Parlay (mix legs from any of the six sports in
# one parlay). Split out of the former single 30_Betting_Engine.py so each
# of the six sport-specific betting pages (30-33, 35-36) stays to exactly
# the three tabs in ui_betting_tabs.md's spec -- these two tools apply
# across sports, so they get their own page rather than being duplicated
# six times.
#
# Nothing on this page ever fetches a sportsbook, and nothing here places
# a bet. The one network touch is the opt-in "Refresh Player Status" button,
# which reads Sleeper's public feed via fantasy.online.player_status_fetcher;
# every comparison stays offline, reading the local player_status.json.

from typing import Any

import pandas as pd
import streamlit as st

from app.betting_shared import load_cbb_evaluations, load_cfb_evaluations, load_mlb_evaluations, load_nba_evaluations, load_nfl_evaluations, load_nhl_evaluations
from app.page_runtime import apply_global_theme, empty_state, page_header, section_header

from fantasy.online.player_status_fetcher import refresh_player_status
from fantasy.player_status import flagged_count, has_status_data, live_status, status_last_updated

from modules.unified_parlay_engine import evaluate_cross_sport_parlay, make_unified_leg

apply_global_theme()

page_header(
    "Cross-Sport Tools",
    "Compare two priced props side by side, or build a parlay that mixes legs from any of the six "
    "sports -- odds come only from our own default files or files you upload, never a sportsbook. "
    "Nothing on this page places a bet.",
    eyebrow="Betting · cross-sport",
)


# ---------------------------------------------------------------------------
# Live player-status overlay (NFL only; other sports keep their own data).
# ---------------------------------------------------------------------------
_STATUS_BADGE: dict[str, str] = {
    "OUT": "🔴 OUT",
    "HOLDOUT": "🟠 Holdout",
    "SUSPENDED": "🟠 Suspended",
    "DOUBTFUL": "🟡 Doubtful",
    "QUESTIONABLE": "🟡 Questionable",
    "HEALTHY": "🟢 Healthy",
}


def _status_badge(status: str) -> str:
    return _STATUS_BADGE.get(str(status or "").strip().upper(), "🟢 Healthy")


def _row_player(row: dict[str, Any]) -> dict[str, Any]:
    """A ``{player_id, name}`` view of an evaluation row, across every sport's
    slightly different field names."""
    identifier = row.get("player_id") or row.get("player") or row.get("player_name") or ""
    name = row.get("name") or row.get("player") or row.get("player_name") or ""
    return {"player_id": identifier, "name": name}


status_info_col, status_button_col = st.columns([3, 1])
_status_updated = status_last_updated()
with status_info_col:
    if _status_updated:
        st.caption(
            f"Player status overlay · {flagged_count()} player(s) flagged · "
            f"updated {_status_updated.replace('T', ' ').replace('Z', ' UTC')} · "
            "OUT players are dropped from comparison; badges are NFL-only."
        )
    else:
        st.caption(
            "Player status overlay not loaded — comparisons use every priced prop as-is. "
            "Refresh to pull live OUT / doubtful / holdout / suspended status (NFL)."
        )
with status_button_col:
    st.markdown('<div style="height:.2rem"></div>', unsafe_allow_html=True)
    if st.button("Refresh Player Status", key="cross_sport_refresh_player_status", width="stretch"):
        result = refresh_player_status()
        if result.get("ok"):
            st.success(f"Updated {result['count']} player status flag(s) from {result['source']}.")
            st.rerun()
        else:
            st.warning(result.get("error") or "Could not refresh player status — kept the existing data.")


comparison_tab, cross_sport_tab = st.tabs(["Player Comparison", "Cross-Sport Parlay"])

# -----------------------------------------------------------------------
# Player Comparison
# -----------------------------------------------------------------------
with comparison_tab:
    section_header("Player Comparison", "Side-by-side edge, EV, and confidence for two priced props from one sport.")
    sport = st.radio("Sport", ["NFL", "NBA", "CFB", "CBB", "MLB", "NHL"], horizontal=True, key="cross_sport_tools_compare_sport")

    if sport == "NFL":
        rows_for_compare, _, _ = load_nfl_evaluations("compare_nfl_odds_upload")
        label_fn = lambda row: f"{row['name']} {row['market']} {row['line']}"
        metrics = [
            ("Market", "market"), ("Line", "line"), ("Recommended side", "recommended_side"),
            ("Edge", "recommended_edge"), ("EV / $100", "recommended_ev"), ("Confidence", "confidence"), ("Risk", "risk_tier"),
        ]
    elif sport == "NBA":
        rows_for_compare, _, _, _ = load_nba_evaluations("compare_nba_props_upload", "compare_nba_odds_upload")
        label_fn = lambda row: f"{row['player']} {row['category']} {row['sportsbook_line']}"
        metrics = [
            ("Category", "category"), ("Line", "sportsbook_line"), ("Recommended side", "recommended_priced_side"),
            ("EV / $100", "recommended_priced_ev"), ("Reliability", "confidence_score"), ("Risk", "risk_tier"),
        ]
    elif sport == "CFB":
        rows_for_compare, _, _ = load_cfb_evaluations("compare_cfb_props_upload", "compare_cfb_odds_upload", season=2025, week=1)
        label_fn = lambda row: f"{row['player_name']} {row['category']} {row['line']}"
        metrics = [
            ("Category", "category"), ("Line", "line"), ("Recommended side", "recommended_side"),
            ("Edge", "recommended_edge"), ("EV / $100", "recommended_ev"), ("Risk", "risk_tier"), ("Basis", "basis"),
        ]
    elif sport == "CBB":
        rows_for_compare, _, _ = load_cbb_evaluations("compare_cbb_props_upload", "compare_cbb_odds_upload")
        label_fn = lambda row: f"{row['player_name']} {row['category']} {row['line']}"
        metrics = [
            ("Category", "category"), ("Line", "line"), ("Recommended side", "recommended_side"),
            ("Edge", "recommended_edge"), ("EV / $100", "recommended_ev"), ("Risk", "risk_tier"), ("Basis", "basis"),
        ]
    elif sport == "MLB":
        rows_for_compare, _, _ = load_mlb_evaluations("compare_mlb_props_upload", "compare_mlb_odds_upload")
        label_fn = lambda row: f"{row['player_name']} {row['category']} {row['line']}"
        metrics = [
            ("Category", "category"), ("Line", "line"), ("Recommended side", "recommended_side"),
            ("Edge", "recommended_edge"), ("EV / $100", "recommended_ev"), ("Risk", "risk_tier"), ("Basis", "basis"),
        ]
    else:
        rows_for_compare, _, _ = load_nhl_evaluations("compare_nhl_props_upload", "compare_nhl_odds_upload")
        label_fn = lambda row: f"{row['player_name']} {row['category']} {row['line']}"
        metrics = [
            ("Category", "category"), ("Line", "line"), ("Recommended side", "recommended_side"),
            ("Edge", "recommended_edge"), ("EV / $100", "recommended_ev"), ("Risk", "risk_tier"), ("Basis", "basis"),
        ]

    # Drop live-OUT players from the comparison entirely (a no-op until the
    # status overlay is refreshed; only ever affects NFL rows).
    dropped_out = 0
    if has_status_data():
        kept = [row for row in rows_for_compare if live_status(_row_player(row)) != "OUT"]
        dropped_out = len(rows_for_compare) - len(kept)
        rows_for_compare = kept

    if len(rows_for_compare) < 2:
        if dropped_out:
            empty_state(
                "Not enough available props to compare",
                f"{dropped_out} prop(s) were dropped because the player is ruled OUT. "
                "Load more props on this sport's betting page, or refresh player status.",
                icon="🚑",
            )
        else:
            empty_state("Not enough priced props to compare", "Load more props on this sport's betting page first.", icon="🔬")
    else:
        options = {label_fn(row): row for row in rows_for_compare}
        col1, col2 = st.columns(2)
        left_label = col1.selectbox("Player / prop A", list(options), index=0, key=f"{sport}_compare_left")
        right_label = col2.selectbox("Player / prop B", list(options), index=min(1, len(options) - 1), key=f"{sport}_compare_right")
        left, right = options[left_label], options[right_label]

        # Each row mixes numbers and strings across different metrics (e.g.
        # "Line" is a float, "Recommended side" is a string) -- stringify
        # for display so the column has one consistent dtype instead of
        # letting pandas/pyarrow silently coerce a mixed-type object column.
        table_rows = [
            {
                "Metric": "Availability",
                left_label: _status_badge(live_status(_row_player(left))),
                right_label: _status_badge(live_status(_row_player(right))),
            }
        ]
        table_rows += [
            {"Metric": label, left_label: str(left.get(key, "")), right_label: str(right.get(key, ""))}
            for label, key in metrics
        ]
        comparison_table = pd.DataFrame(table_rows)
        st.dataframe(comparison_table, width="stretch", hide_index=True)

# -----------------------------------------------------------------------
# Cross-Sport Parlay
# -----------------------------------------------------------------------
with cross_sport_tab:
    section_header(
        "Cross-Sport Parlay",
        "Combine legs from any of the six sports in one parlay. Each sport's own correlation patterns "
        "are still detected within that sport's legs; legs from different sports are never treated as "
        "correlated with each other.",
    )
    st.caption("Pick which sports to load below -- each is loaded fresh here (this page doesn't reuse the sport pages' cache keys).")

    load_cols = st.columns(6)
    sports_to_load = [s for s, col in zip(("NFL", "NBA", "CFB", "CBB", "MLB", "NHL"), load_cols) if col.checkbox(s, value=True, key=f"cross_sport_include_{s}")]

    if st.button("Load selected sports for a cross-sport parlay", key="cross_sport_load_button"):
        with st.spinner("Loading selected sports..."):
            loaded: dict[str, list[dict[str, Any]]] = {}
            if "NFL" in sports_to_load:
                loaded["NFL"] = load_nfl_evaluations("cross_sport_nfl_odds_upload")[0]
            if "NBA" in sports_to_load:
                loaded["NBA"] = load_nba_evaluations("cross_sport_nba_props_upload", "cross_sport_nba_odds_upload")[0]
            if "CFB" in sports_to_load:
                loaded["CFB"] = load_cfb_evaluations("cross_sport_cfb_props_upload", "cross_sport_cfb_odds_upload", season=2025, week=1)[0]
            if "CBB" in sports_to_load:
                loaded["CBB"] = load_cbb_evaluations("cross_sport_cbb_props_upload", "cross_sport_cbb_odds_upload")[0]
            if "MLB" in sports_to_load:
                loaded["MLB"] = load_mlb_evaluations("cross_sport_mlb_props_upload", "cross_sport_mlb_odds_upload")[0]
            if "NHL" in sports_to_load:
                loaded["NHL"] = load_nhl_evaluations("cross_sport_nhl_props_upload", "cross_sport_nhl_odds_upload")[0]
        st.session_state["cross_sport_loaded"] = loaded

    loaded = st.session_state.get("cross_sport_loaded") or {}

    if len(loaded) < 1 or not any(loaded.values()):
        empty_state(
            "Load at least one sport with priced props to build a cross-sport parlay",
            "Check the sports you want above and click the load button.",
            icon="🌐",
        )
    else:
        _LABEL_FN_BY_SPORT = {
            "NFL": lambda row: f"NFL: {row['name']} {row['market']} {row['recommended_side']} {row['line']}",
            "NBA": lambda row: f"NBA: {row['player']} {row['category']} {row['recommended_priced_side']} {row['sportsbook_line']}",
            "CFB": lambda row: f"CFB: {row['player_name']} {row['category']} {row['recommended_side']} {row['line']}",
            "CBB": lambda row: f"CBB: {row['player_name']} {row['category']} {row['recommended_side']} {row['line']}",
            "MLB": lambda row: f"MLB: {row['player_name']} {row['category']} {row['recommended_side']} {row['line']}",
            "NHL": lambda row: f"NHL: {row['player_name']} {row['category']} {row['recommended_side']} {row['line']}",
        }
        combined_options: dict[str, tuple[str, dict[str, Any]]] = {}
        for sport_key, rows in loaded.items():
            for row in rows:
                combined_options[_LABEL_FN_BY_SPORT[sport_key](row)] = (sport_key, row)

        default_labels = list(combined_options)[:2]
        chosen_labels = st.multiselect("Legs (mix sports freely)", list(combined_options), default=default_labels, key="cross_sport_leg_picker")
        if len(chosen_labels) < 2:
            st.info("Pick at least 2 legs (from any of the loaded sports) to evaluate a cross-sport parlay.")
        else:
            legs = []
            for label in chosen_labels:
                sport_key, row = combined_options[label]
                if sport_key == "NFL":
                    side = row["recommended_side"]
                    legs.append(make_unified_leg("NFL", description=label, model_probability=row[f"model_probability_{side}"], price=row[f"{side}_price"], player_id=row["player_id"], team=row["team"], market=row["market"], side=side, confidence=row["confidence"]))
                elif sport_key == "NBA":
                    side = row["recommended_priced_side"]
                    legs.append(make_unified_leg("NBA", description=label, model_probability=row[f"model_probability_{side}"], price=row[f"{side}_price"], player_id=row["player"], team=row["team"], market=row["category"], side=side, confidence=row.get("confidence_score", 70.0) / 100.0))
                else:
                    side = row["recommended_side"]
                    confidence = {"low": 0.8, "medium": 0.5, "high": 0.25}.get(row.get("risk_tier"), 0.6)
                    legs.append(make_unified_leg(sport_key, description=label, model_probability=row[f"model_probability_{side}"], price=row[f"{side}_price"], player_id=row.get("player_name"), team=row["team"], market=row["category"], side=side, confidence=confidence))

            result = evaluate_cross_sport_parlay(legs)
            st.caption(f"Sports in this parlay: {', '.join(result['sports'])}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Adjusted hit probability", f"{result['adjusted_hit_probability']:.1%}")
            m2.metric("Payout / $100", f"${result['payout_per_100_stake']:,.2f}")
            m3.metric("Adjusted EV / $100", f"${result['adjusted_ev']:+,.2f}")
            m4.metric("Confidence", f"{result['confidence']:.0%}", help=f"Risk tier: {result['risk_tier']}")

            if result["correlations_detected"]:
                st.warning(f"Correlated legs detected within one sport -- hit probability adjusted from the naive {result['naive_hit_probability']:.1%} to {result['adjusted_hit_probability']:.1%}.")
                for finding in result["correlations_detected"]:
                    leg_a, leg_b = finding["legs"]
                    st.caption(f"- {legs[leg_a]['description']} + {legs[leg_b]['description']}: {finding['note']}")
            else:
                st.caption("No known correlation pattern detected between these legs (cross-sport pairs are never correlated).")
