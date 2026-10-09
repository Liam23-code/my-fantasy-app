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

# NFL Parlays: stack moneyline, spread and player-prop legs and price the whole
# ticket with correlation -- not just the leg prices multiplied together.
#
# This page is deliberately thin. Every number comes from three modules:
#   * modules.nfl_moneyline_analysis  -> win / cover probabilities for game legs
#   * modules.nfl_prop_analysis       -> over / under probabilities for prop legs
#   * modules.nfl_parlay_builder      -> correlation, combined probability, EV, risk
# Leg probabilities are the models' own; the page only collects the user's choices
# (week, game, side, line, price), converts a sportsbook-style spread into the
# nflverse sign convention, and lays the engine's output out. Nothing here
# fetches a sportsbook and nothing places a bet.

import math
import re

import streamlit as st

from app import betting_ui as bt
from app.page_runtime import apply_global_theme, page_header, section_header
from betting import player_status_utils as status_utils
from modules import nfl_moneyline_analysis as ml
from modules import nfl_parlay_builder as pb
from modules import nfl_prop_analysis as pa
from modules.odds_input import (
    evaluate_price,
    format_american,
    format_money,
    format_pct,
    format_signed_pct,
    parse_number,
    parse_odds,
)

# ---------------------------------------------------------------------------
# Constants and small formatting helpers
# ---------------------------------------------------------------------------
KIND_ML = "Moneyline"
KIND_SPREAD = "Spread"
KIND_PROP = "Player prop"
KINDS = (KIND_ML, KIND_SPREAD, KIND_PROP)

DEFAULT_LEG_PRICE = -110.0
DEFAULT_STAKE = 100.0

SRC_USER = "your sportsbook price"
SRC_CONSENSUS = "nflverse consensus moneyline"
SRC_FAIR = "model fair price (no vig)"
SRC_STANDARD = "standard -110"

_NAME_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"})


def _md(text) -> str:
    """Escape ``$`` so a caption holding two dollar amounts never turns into LaTeX."""
    return str(text).replace("$", "\\$")


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prob_text(probability) -> str:
    """A probability as text that never rounds a real long shot down to ``0.0%``."""
    number = _finite(probability)
    if number is None:
        return "—"
    if number <= 0:
        return "0%"
    pct = number * 100.0
    if pct >= 1.0:
        return f"{pct:.1f}%"
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return "<0.01%"


def _no_negative_zero(number, digits: int):
    """``number`` unchanged, except a value that rounds to zero at ``digits`` decimals of a percent becomes exactly 0.0.

    Formatting a tiny negative (a fair-priced leg's value gap is -0.00004, say) prints ``-0.0 pts`` /
    ``-0.00 pts``, which reads as a real shortfall that is not there.
    """
    return 0.0 if number is not None and round(number * 100.0, digits) == 0 else number


def _signed_percent(fraction) -> str:
    number = _no_negative_zero(_finite(fraction), 1)
    return "—" if number is None else f"{number * 100:+.1f}%"


def _signed_pts(points) -> str:
    """Signed probability points (``+3.2 pts``) that never print ``-0.0 pts``."""
    return format_signed_pct(_no_negative_zero(_finite(points), 1))


def _lift_text(fraction) -> str:
    """The correlation lift in points to two decimals, never ``-0.00``."""
    number = _no_negative_zero(_finite(fraction), 2)
    return "—" if number is None else f"{number * 100:+.2f} pts"


def _one_in_text(probability) -> str:
    number = _finite(probability)
    if number is None or number <= 0:
        return ""
    odds = 1.0 / number
    return f"about 1 in {odds:,.0f}" if odds < 1e9 else "about 1 in a billion or more"


def _spread_text(line: float) -> str:
    return "PK" if abs(line) < 1e-9 else f"{line:+g}"


def _stake_text(stake: float) -> str:
    """``$100`` for a whole stake, ``$12.50`` otherwise (never ``$12`` for a $12.50 stake)."""
    return f"${stake:,.0f}" if float(stake).is_integer() else f"${stake:,.2f}"


# The shared number parser drops commas as thousands separators, so a European decimal comma
# ("1,91", "-3,5", "249,5") would silently read as 191, -35 or 2495. Two digits or fewer after a
# lone comma is never a thousands group, so it is refused with a message instead.
_COMMA_DECIMAL = re.compile(r"^[\s+\-−–—‒]*\d+,\d{1,2}\s*$")


def _comma_problem(text, label: str) -> str | None:
    if text is not None and _COMMA_DECIMAL.match(str(text)):
        return f"Use a period for the decimal point in the {label} (for example 1.91 or -3.5), not a comma."
    return None


def _odds(text, label: str):
    """``parse_odds`` plus the decimal-comma guard: ``(price or None, error or None)``."""
    problem = _comma_problem(text, label)
    if problem:
        return None, problem
    parsed = parse_odds(text)
    return parsed.price, parsed.error


_PICKEM_WORDS = frozenset({"pk", "pick", "pickem", "pickam"})


def _parse_spread(text):
    """``(spread or None, error or None)`` for a sportsbook-style spread such as ``-3.5``, ``+7`` or ``PK``."""
    simplified = re.sub(r"[\s'’\-]", "", str(text or "").lower())
    if simplified in _PICKEM_WORDS:
        return 0.0, None
    problem = _comma_problem(text, "spread")
    if problem:
        return None, problem
    parsed = parse_number(text, allow_negative=True, minimum=-60, maximum=60, label="spread")
    if parsed.error and parsed.error.startswith("Enter the spread as a number"):
        return None, "Enter the spread as a number like -3.5 or +7, or PK for a pick'em."
    return parsed.value, parsed.error


def _leg_problem(problem) -> str:
    """A leg-building ``ValueError`` as a sentence a bettor can act on."""
    text = str(problem)
    if "strictly between 0% and 100%" in text:
        return (
            "The model rates this leg as a near-certainty either way (about 0% or 100%), so it cannot be priced as a "
            "parlay leg. Check the number you typed."
        )
    return text


def _last_name(name) -> str:
    parts = str(name or "").split()
    while len(parts) > 1 and parts[-1].lower() in _NAME_SUFFIXES:
        parts.pop()
    return parts[-1] if parts else ""


def _tone_for(verdict_tone: str) -> str:
    return {"success": "good", "danger": "bad", "warning": "warn"}.get(verdict_tone, "cyan")


def _fav_text(analysis, spread_home: float) -> str:
    """``DAL -8.5`` from the nflverse convention (positive means the home team is favored)."""
    if spread_home > 0:
        return f"{analysis['home_team']} -{spread_home:g}"
    if spread_home < 0:
        return f"{analysis['away_team']} -{abs(spread_home):g}"
    return "Pick'em"


def _notices(messages, tone: str = "warn") -> None:
    for message in messages:
        bt.render(bt.notice_html(message, tone))


# ---------------------------------------------------------------------------
# Page CSS: the app's light-canvas rules paint a segmented control white on
# white, so restyle just that widget in the Midnight Circuit palette (the same
# constants the kit uses, so nothing drifts from the theme). The kit's own
# "Risk score N / 100" line is hidden because the page passes it a stretched bar
# position (see _risk_bar_position) and prints the engine's true score instead.
# ---------------------------------------------------------------------------
_PAGE_CSS = f"""<style>
[data-testid="stMain"] [data-testid="stButtonGroup"] button[data-variant="segmented_control"]{{
  background:{bt.SLATE_BLUE}b8 !important;border:1px solid {bt.CIRCUIT_CYAN}38 !important;color:{bt.MUTED_TEXT} !important;
  border-radius:10px !important;font-weight:650;transition:border-color .16s ease,background .16s ease;
}}
[data-testid="stMain"] [data-testid="stButtonGroup"] button[data-variant="segmented_control"] p{{color:inherit !important}}
[data-testid="stMain"] [data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover{{border-color:{bt.CIRCUIT_CYAN} !important}}
[data-testid="stMain"] .stButton button:disabled{{opacity:.42;filter:saturate(.35);box-shadow:none !important;cursor:not-allowed}}
.bt-risk .bt-risk-score{{display:none}}
[data-testid="stMain"] [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"]{{
  background:{bt.CIRCUIT_CYAN}24 !important;border-color:{bt.CIRCUIT_CYAN} !important;color:{bt.SOFT_WHITE} !important;
  box-shadow:0 0 18px {bt.CIRCUIT_CYAN}30;
}}
</style>"""


# ---------------------------------------------------------------------------
# Session state, sticky widgets, callbacks
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("pl_legs", [])
    st.session_state.setdefault("pl_mem", {})
    st.session_state.setdefault("pl_stake", DEFAULT_STAKE)
    st.session_state.setdefault("pl_book", "")
    if st.session_state.get("pl_kind") not in KINDS:
        st.session_state["pl_kind"] = KIND_ML


def _sticky(key: str, options, default=None) -> None:
    """Pre-seed a select/radio ``key`` with its last valid value.

    Streamlit forgets a widget's value when the widget is not drawn for one
    run (switching the leg type hides the game picker). The value is kept in
    ``pl_mem`` and restored here -- always BEFORE the widget is created, never
    alongside a ``value=``/``index=`` default.
    """
    memory = st.session_state.setdefault("pl_mem", {})
    options = list(options)
    fallback = options[0] if options else None
    choice = next(
        (c for c in (st.session_state.get(key), memory.get(key), default) if c is not None and c in options),
        fallback,
    )
    if key not in st.session_state or st.session_state[key] != choice:
        st.session_state[key] = choice


def _remember(key: str) -> None:
    st.session_state.setdefault("pl_mem", {})[key] = st.session_state.get(key)


def _reseed(key: str, ctx_key: str, ctx, value: str) -> None:
    """Reset a text box to ``value`` whenever the thing it describes changes."""
    if key not in st.session_state or st.session_state.get(ctx_key) != ctx:
        st.session_state[ctx_key] = ctx
        st.session_state[key] = value


def _keep_kind() -> None:
    """A segmented control can be clicked off; there is always a leg type, so put the last one back."""
    if st.session_state.get("pl_kind") not in KINDS:
        last = st.session_state.get("pl_mem", {}).get("pl_kind")
        st.session_state["pl_kind"] = last if last in KINDS else KIND_ML


def _flash(tone: str, text: str) -> None:
    st.session_state["pl_flash"] = {"tone": tone, "text": text}


def _leg_name(leg) -> str:
    return str(leg.get("description") or (leg.get("ui") or {}).get("short") or "leg")


def _add_leg(leg) -> None:
    """Add ``leg`` unless it is missing, conflicts with the slip or the slip is full (always a message, never a traceback)."""
    legs = list(st.session_state.get("pl_legs", []))
    if not isinstance(leg, dict):
        _flash("warn", "Nothing was added: fix the highlighted number first.")
        return
    if len(legs) >= pb.MAX_PARLAY_LEGS:
        _flash("warn", f"A slip holds at most {pb.MAX_PARLAY_LEGS} legs. Remove one first.")
        return
    allowed, message = pb.can_add_leg(legs, leg)
    if not allowed:
        _flash("bad", message)
        return
    legs.append(leg)
    st.session_state["pl_legs"] = legs
    note = f" {message}" if message else ""
    _flash("warn" if message else "good", f"Added {_leg_name(leg)}. Your slip now has {len(legs)} leg{'s' if len(legs) != 1 else ''}.{note}")


def _remove_leg(leg_id: str) -> None:
    legs = list(st.session_state.get("pl_legs", []))
    kept = [leg for leg in legs if leg.get("leg_id") != leg_id]
    removed = [leg for leg in legs if leg.get("leg_id") == leg_id]
    st.session_state["pl_legs"] = kept
    if removed:
        _flash("info", f"Removed {_leg_name(removed[0])}.")


def _clear_slip() -> None:
    st.session_state["pl_legs"] = []
    _flash("info", "Slip cleared.")


def _request_example() -> None:
    st.session_state["pl_do_example"] = True


def _request_add(key: str) -> None:
    """Remember WHICH Add button was pressed; the leg itself is priced from the widget values of the run that follows.

    A callback's ``args`` come from the previous run, so a price typed a moment before the click (a text box only
    commits on blur) would be silently ignored if the leg were passed through them. The builder reads this flag
    after it has priced the leg from this run's fresh values, which also means the slip, the Add button and the
    results below are all drawn from one pass (no forced rerun and no half-updated first pass).
    """
    st.session_state["pl_add_request"] = key


def _retry_loads() -> None:
    st.cache_data.clear()
    ml.clear_caches()
    pa.clear_player_pool_cache()


# ---------------------------------------------------------------------------
# Cached loaders (no network at import time; failures become empty states)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_game_inputs():
    return ml.load_moneyline_inputs()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_player_pool():
    pool = pa.load_player_pool()
    if not pool:
        # Raising keeps an empty (failed) load out of the cache so Retry really retries.
        raise LookupError("The NFL player pool came back empty.")
    return pool


def _load_game_inputs():
    """``(inputs, error)`` -- the schedule, ratings and quarterback context for game legs."""
    try:
        with st.spinner("Loading the NFL schedule, team ratings and quarterbacks (about 10 seconds the first time)..."):
            inputs = _cached_game_inputs()
    except ml.ScheduleUnavailable as error:
        return None, str(error)
    except Exception as error:  # noqa: BLE001 - the UI boundary must never show a traceback
        return None, f"The game data could not be loaded ({type(error).__name__}). Check your connection and try again."
    schedule = inputs.get("schedule") if isinstance(inputs, dict) else None
    if schedule is None or len(schedule) == 0 or not ml.weeks_in_schedule(schedule):
        return None, "The NFL schedule came back empty. Try again in a moment."
    return inputs, None


def _load_pool():
    """``(pool, error)`` -- the NFL player pool for prop legs."""
    try:
        with st.spinner("Loading the NFL player pool..."):
            pool = _cached_player_pool()
    except Exception:  # noqa: BLE001
        return None, "The NFL player pool could not be loaded. Check your connection and try again."
    return pool, None


def _unavailable_card(title: str, body: str, key: str) -> None:
    bt.render(bt.empty_card_html(title, body))
    st.button("Retry", key=key, on_click=_retry_loads, help="Clears the cached NFL data and loads it again.")


# ---------------------------------------------------------------------------
# Leg construction (shared by the builders and the example parlay)
# ---------------------------------------------------------------------------
def _game_confidence(analysis) -> float:
    """How reliable the game model's inputs are for this matchup (0-1).

    The MoneyLines engine reports ``confidence_factors`` -- margin, sample, qb,
    market. ``margin`` is "how clear the favorite is", which says nothing about how
    much to trust the number for the underdog's side of a leg, so only the three
    data-reliability factors are used: games played, quarterback certainty and
    agreement with the market.
    """
    factors = analysis.get("confidence_factors") or {}
    value = 1.0
    for name in ("sample", "qb", "market"):
        number = _finite(factors.get(name))
        value *= 0.7 if number is None else number
    return min(1.0, max(0.05, value))


def _teams(analysis, side: str) -> tuple[str, str]:
    home, away = analysis["home_team"], analysis["away_team"]
    return (home, away) if side == "home" else (away, home)


def _build_moneyline_leg(game, analysis, side: str, price: float, source: str):
    team, opponent = _teams(analysis, side)
    probability = analysis["probabilities"]["home_win" if side == "home" else "away_win"]
    leg = pb.make_parlay_leg(
        "moneyline",
        description=f"{team} moneyline",
        probability=probability,
        price=price,
        game_id=game["game_id"],
        team=team,
        opponent=opponent,
        home_team=analysis["home_team"],
        away_team=analysis["away_team"],
        side=side,
        confidence=_game_confidence(analysis),
    )
    leg["ui"] = {
        "kind": KIND_ML,
        "short": f"{team} ML",
        "matchup": game["label"],
        "week": game.get("week"),
        "played": bool(game.get("played")),
        "price_source": source,
    }
    return leg


def _build_spread_leg(game, analysis, side: str, line: float, price: float, source: str):
    team, opponent = _teams(analysis, side)
    probability = analysis["probabilities"]["home_cover" if side == "home" else "away_cover"]
    leg = pb.make_parlay_leg(
        "spread",
        description=f"{team} {_spread_text(line)}",
        probability=probability,
        price=price,
        game_id=game["game_id"],
        team=team,
        opponent=opponent,
        home_team=analysis["home_team"],
        away_team=analysis["away_team"],
        side=side,
        line=line,
        confidence=_game_confidence(analysis),
    )
    leg["ui"] = {
        "kind": KIND_SPREAD,
        "short": f"{team} {_spread_text(line)}",
        "matchup": game["label"],
        "week": game.get("week"),
        "played": bool(game.get("played")),
        "price_source": source,
    }
    return leg


def _build_prop_leg(player, evaluation, side: str, game, source: str = SRC_USER):
    spec = pa.STAT_CATEGORIES[evaluation["category"]]
    line = evaluation["line"]
    priced = evaluation["pricing_over" if side == "over" else "pricing_under"]
    probability = evaluation["probability_over" if side == "over" else "probability_under"]
    team = ml.normalize_team(evaluation.get("team") or player.get("team"))
    kwargs = {}
    if game is not None:
        kwargs = {"game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"]}
    name = evaluation["name"]
    leg = pb.make_parlay_leg(
        "prop",
        description=f"{name} {side.title()} {line:g} {spec['short_label']}",
        probability=probability,
        price=priced["price"],
        team=team or None,
        side=side,
        line=line,
        market=spec["market"],
        player_id=str(player.get("player_id")) if player.get("player_id") is not None else None,
        player_name=name,
        position=evaluation.get("position"),
        confidence=evaluation["confidence"],
        **kwargs,
    )
    leg["ui"] = {
        "kind": KIND_PROP,
        "short": f"{_last_name(name)} {'O' if side == 'over' else 'U'}{line:g}",
        "matchup": game["label"] if game is not None else (f"{team} game" if team else "Game not linked"),
        "week": game.get("week") if game is not None else None,
        "played": bool(game.get("played")) if game is not None else False,
        "price_source": source,
        "position": evaluation.get("position"),
    }
    return leg


def _team_game(games, team: str):
    for game in games:
        if team in (game["away_team"], game["home_team"]):
            return game
    return None


# ---------------------------------------------------------------------------
# Game preview (moneyline and spread builders)
# ---------------------------------------------------------------------------
def _qb_line(analysis, game, side: str) -> str:
    info = analysis["teams"][side]
    name = info.get("qb_name")
    if not name:
        return "QB not identified"
    assumed = info.get("qb_source") in ("team_primary", "none") or game.get(f"{side}_qb_source") == "last_game"
    return f"QB {name}" + (" (assumed starter)" if assumed else "")


def _assumed_starters(analysis, game) -> list[str]:
    notes = []
    for side in ("away", "home"):
        info = analysis["teams"][side]
        if info.get("qb_name") and (info.get("qb_source") in ("team_primary", "none") or game.get(f"{side}_qb_source") == "last_game"):
            notes.append(
                f"{info['team']}: no starting quarterback is confirmed yet, so {info['qb_name']} is assumed. "
                "A different starter would move this number."
            )
    return notes


def _team_payload(analysis, game, side: str) -> dict:
    info = analysis["teams"][side]
    return {
        "team": info["team"],
        "name": info["team_name"],
        "subtitle": _qb_line(analysis, game, side),
        "win_probability": info["win_probability"],
        "rating": info["strength_rating"],
        "expected_points": info["expected_points"],
    }


def _game_chips(analysis, game) -> list[str]:
    chips = []
    context = analysis.get("context") or {}
    if context.get("neutral_site") or game.get("neutral_site"):
        chips.append("Neutral site")
    if context.get("div_game"):
        chips.append("Divisional")
    if context.get("roof") in ("dome", "closed"):
        chips.append("Roofed stadium")
    if game.get("played"):
        chips.append(str(game.get("status") or "Final"))
    return chips


def _market_spread_text(analysis) -> str:
    market = analysis["probabilities"].get("market_spread_home")
    return "No line posted yet" if market is None else _fav_text(analysis, market)


def _line_caption(analysis) -> str:
    model = analysis["scoring"]["model_spread_home"]
    model_text = _fav_text(analysis, round(model, 1))
    market = analysis["probabilities"].get("market_spread_home")
    if market is None:
        return f"Model spread: {model_text}. No market line is posted for this game yet, so the model number is the reference."
    return f"Model spread: {model_text}. Market spread (nflverse consensus): {_fav_text(analysis, market)}."


def _render_game_preview(game, analysis, shared_warnings=()) -> None:
    total = analysis["scoring"].get("market_total")
    bt.render(
        bt.matchup_card_html(
            away=_team_payload(analysis, game, "away"),
            home=_team_payload(analysis, game, "home"),
            kickoff=game.get("kickoff_label") or "",
            chips=_game_chips(analysis, game),
            favored=analysis["favored_team"],
            spread_text=_market_spread_text(analysis),
            total_text="" if total is None else f"O/U {total:g}",
            compact=True,
        )
    )
    st.caption(_md(_line_caption(analysis)))
    _notices(_assumed_starters(analysis, game), "steel")
    own = [
        w for w in analysis.get("warnings") or []
        if w not in shared_warnings and "is assumed" not in w and "already been played" not in w
    ]
    _notices(own[:3], "warn")
    if game.get("played"):
        _notices(
            [
                f"This game is already final ({game.get('status')}). You can still add it, but the ratings already include "
                "its result, so the numbers are hindsight rather than a forecast."
            ],
            "warn",
        )


def _price_tiles(probability: float, price: float, source: str, label: str) -> None:
    priced = evaluate_price(probability, price)
    tone = _tone_for(priced["verdict_tone"])
    bt.render(
        bt.stat_grid_html(
            [
                {"label": label, "value": format_pct(priced["model_probability"]), "sub": "The model's chance this leg wins", "tone": "cyan"},
                {"label": "Price used", "value": format_american(price), "sub": f"Source: {source}", "tone": "cyan"},
                {"label": "Break-even needed", "value": format_pct(priced["implied_probability"]), "sub": "Win rate this price requires", "tone": "cyan"},
                {"label": "Value gap", "value": _signed_pts(priced["value_gap"]), "sub": priced["verdict"], "tone": tone},
            ],
            columns=2,
        )
    )


# ---------------------------------------------------------------------------
# Builder controls
# ---------------------------------------------------------------------------
def _add_controls(candidate, key: str, hint: str, just_added: bool = False) -> None:
    """The Add button plus the pre-flight conflict check (``_add_leg`` re-checks when the click is handled).

    The click itself is handled by the caller BEFORE this runs, so the notices and the disabled state here
    always describe the slip as it is now. ``just_added`` hides the "already in your parlay" note for the
    leg that was added a moment ago (the button is disabled; the confirmation below already says what happened).
    """
    legs = st.session_state.get("pl_legs", [])
    blocked = candidate is None
    if candidate is not None:
        if len(legs) >= pb.MAX_PARLAY_LEGS:
            blocked = True
            _notices([f"A slip holds at most {pb.MAX_PARLAY_LEGS} legs. Remove one to add another."], "warn")
        else:
            allowed, message = pb.can_add_leg(legs, candidate)
            if not allowed:
                blocked = True
                if not (just_added and "already in your parlay" in message):
                    _notices([message], "bad")
            elif message:
                _notices([message], "warn")
    elif hint:
        st.caption(_md(hint))
    st.button(
        "Add leg to slip", key=f"pl_add_{key}", type="primary", use_container_width=True, disabled=blocked,
        on_click=_request_add, args=(key,),
    )


def _pick_game(inputs):
    """Week, game and side pickers shared by the moneyline and spread builders."""
    schedule = inputs["schedule"]
    games = ml.games_for_week(schedule, st.session_state["pl_week"])
    if not games:
        bt.render(bt.empty_card_html("No games this week", "Pick another week."))
        return None, None
    by_id = {game["game_id"]: game for game in games}
    upcoming = next((game for game in games if not game["played"]), games[0])
    _sticky("pl_game", list(by_id), upcoming["game_id"])

    def game_label(game_id: str) -> str:
        game = by_id[game_id]
        tail = f" - {game['status']}" if game["played"] else ""
        return f"{game['away_team']} @ {game['home_team']}  ({game['kickoff_label']}){tail}"

    game_id = st.selectbox("Game", list(by_id), key="pl_game", format_func=game_label)
    _remember("pl_game")
    game = by_id[game_id]
    _sticky("pl_side", ["away", "home"], "home")
    neutral = bool(game.get("neutral_site"))

    def side_label(side: str) -> str:
        team = game["home_team"] if side == "home" else game["away_team"]
        if side == "home":
            return f"{team} (designated home)" if neutral else f"{team} (home)"
        return f"{team} (away)"

    side = st.radio("Team to back", ["away", "home"], key="pl_side", format_func=side_label, horizontal=True)
    _remember("pl_side")
    return game, side


def _moneyline_price(analysis, side: str, text):
    """``(price, source, error)``: pasted price, else the consensus moneyline, else the model's fair price."""
    pasted, error = _odds(text, "price")
    if error:
        return None, None, error
    if pasted is not None:
        return pasted, SRC_USER, None
    market = analysis["moneyline"].get("market_home_ml" if side == "home" else "market_away_ml")
    if market is not None:
        return market, SRC_CONSENSUS, None
    return analysis["moneyline"]["home_fair_price" if side == "home" else "away_fair_price"], SRC_FAIR, None


def _render_moneyline_builder(inputs, game, side: str):
    analysis = ml.analyze_game(game, inputs["ratings"], inputs["qb_context"])
    _render_game_preview(game, analysis, inputs.get("warnings") or [])
    # A typed price belongs to the bet it was typed for: carried over to another game or side it would be added
    # to the wrong leg without a word, so the box starts blank again whenever the game or side changes.
    _reseed("pl_ml_price", "pl_ml_ctx", (game["game_id"], side), "")
    text = st.text_input(
        "Your sportsbook price (optional)",
        key="pl_ml_price",
        placeholder="e.g. -150 or +130",
        help="Leave blank to use the nflverse consensus moneyline, or the model's own fair price when no line is posted yet.",
    )
    price, source, error = _moneyline_price(analysis, side, text)
    if error:
        _notices([error], "warn")
        return None, ""
    try:
        candidate = _build_moneyline_leg(game, analysis, side, price, source)
        _price_tiles(analysis["probabilities"]["home_win" if side == "home" else "away_win"], price, source, "Model win probability")
    except ValueError as problem:
        _notices([_leg_problem(problem)], "warn")
        return None, ""
    if source == SRC_FAIR:
        st.caption("No moneyline is posted for this game, so the leg is priced at the model's own fair price. That has no edge by construction.")
    elif source == SRC_CONSENSUS:
        st.caption("Price is the nflverse consensus moneyline, a reference rather than your book's number. Paste your own price above to replace it.")
    return candidate, ""


def _spread_meaning(team: str, line: float) -> str:
    """What the typed spread asks of ``team``, in words (the sign is the easiest thing to get backwards)."""
    size = abs(line)
    whole = float(size).is_integer()
    if size < 1e-9:
        return f"{team} PK (pick'em): {team} has to win the game outright."
    if line < 0:
        need = f"{math.floor(size) + 1:g} or more" if not whole else f"a win by exactly {size:g} is a push"
        return f"{team} {line:+g} means {team} has to win by more than {size:g} points ({need})."
    keep = f"{math.floor(size):g} or fewer" if not whole else f"losing by exactly {size:g} is a push"
    return f"{team} {line:+g} means {team} covers by winning, or by losing by fewer than {size:g} points ({keep})."


def _default_spread_text(analysis, side: str) -> str:
    used = analysis["probabilities"]["spread_used_home"]  # market when posted, else the model's, rounded to .5
    line = -used if side == "home" else used
    return _spread_text(line).replace("PK", "0")


def _render_spread_builder(inputs, game, side: str):
    base = ml.analyze_game(game, inputs["ratings"], inputs["qb_context"])
    _render_game_preview(game, base, inputs.get("warnings") or [])
    _reseed("pl_sp_line", "pl_sp_ctx", (game["game_id"], side), _default_spread_text(base, side))
    team, _opponent = _teams(base, side)
    posted = base["probabilities"]["spread_source"] == "market"
    line_text = st.text_input(
        f"Spread for {team}",
        key="pl_sp_line",
        placeholder="e.g. -3.5 or +7",
        help=(
            "Type the spread the way your sportsbook shows it for this team: -3.5 means the team lays 3.5 points, "
            "+7 means it gets 7. Pre-filled with the market spread when one is posted, else the model's, rounded to a half point."
        ),
    )
    spread_value, spread_error = _parse_spread(line_text)
    half_step = spread_value is None or (spread_value * 2) == int(spread_value * 2)
    source_note = (
        "Pre-filled from the nflverse consensus spread."
        if posted
        else "No market spread is posted, so the box starts at the model spread rounded to a half point."
    )
    # The sign is the easiest thing to get backwards, so say what the typed number asks of the team.
    meaning = f" {_spread_meaning(team, spread_value)}" if spread_value is not None and half_step else ""
    st.caption(_md(source_note + meaning))
    _reseed("pl_sp_price", "pl_sp_price_ctx", (game["game_id"], side), f"{DEFAULT_LEG_PRICE:.0f}")
    price_text = st.text_input(
        "Price",
        key="pl_sp_price",
        placeholder="-110",
        help="Spreads usually pay -110. Leave blank to assume -110.",
    )
    if spread_error:
        _notices([spread_error], "warn")
        return None, ""
    if spread_value is None:
        return None, "Enter the spread to price this leg."
    if not half_step:
        _notices(["NFL spreads move in half-point steps (for example -3, -3.5 or -4). Check the number you typed."], "warn")
        return None, ""
    typed_price, price_error = _odds(price_text, "price")
    if price_error:
        _notices([price_error], "warn")
        return None, ""
    price = typed_price if typed_price is not None else DEFAULT_LEG_PRICE
    source = SRC_STANDARD if price == DEFAULT_LEG_PRICE else SRC_USER
    line = spread_value + 0.0
    # Sign adapter. The leg line is what the backed team receives (-3.5 lays 3.5). The engine's
    # user_spread_home uses the nflverse convention, where POSITIVE means the HOME team is favored.
    user_spread_home = -line if side == "home" else line
    analysis = ml.analyze_game(game, inputs["ratings"], inputs["qb_context"], user_spread_home=user_spread_home)
    try:
        candidate = _build_spread_leg(game, analysis, side, line, price, source)
        _price_tiles(analysis["probabilities"]["home_cover" if side == "home" else "away_cover"], price, source, "Model cover probability")
    except ValueError as problem:
        _notices([_leg_problem(problem)], "warn")
        return None, ""
    if float(line).is_integer():
        st.caption("A whole-number spread can push (a refund when the margin lands exactly on it). The parlay engine treats every leg as push-free.")
    return candidate, ""


def _passing_caveat(cv: float | None) -> str:
    """How the engine's quarterback passing-yard spread compares with 2025 starters (its own ``cv`` against its own constants)."""
    low, median, high = pa.OBSERVED_QB_PASSING_CV
    if cv is None:
        return "Passing-yard probabilities come from a Gaussian curve, which is a modeling choice rather than observed variance."
    base = (
        f"This passing-yard curve varies by about {cv:.2f} of the projected yards; 2025 starting quarterbacks varied about "
        f"{median:.2f} (middle half {low:.2f} to {high:.2f})."
    )
    if cv > high:
        return base + " That is wider than quarterbacks show in practice, so over and under sit closer to 50% than history supports."
    if cv < low:
        return base + " That is narrower than quarterbacks show in practice, so the probabilities are more confident than history supports."
    return base + " That is inside the usual range, but the Gaussian curve is still a modeling choice, not observed variance."


def _player_option(player) -> str:
    status = pa.player_status(player)
    label = pa.player_label(player)
    return label if status == status_utils.HEALTHY else f"{label} - {status.title()}"


def _status_tone(status: str) -> str:
    if status in status_utils.UNAVAILABLE_STATUSES:
        return "bad"
    return "warn" if status != status_utils.HEALTHY else "neutral"


def _render_prop_builder(inputs):
    pool, error = _load_pool()
    if error:
        _unavailable_card("Player props are unavailable", error, "pl_retry_pool")
        return None, ""
    teams = pa.teams_in_pool(pool)
    if not teams:
        bt.render(bt.empty_card_html("No players to price", "The player pool has no skill-position players yet."))
        return None, ""
    game_teams = []
    week_games = []
    if inputs is not None:
        week_games = ml.games_for_week(inputs["schedule"], st.session_state["pl_week"])
        # Open on a team that still has a game to play (the same rule as the game pickers), not on a finished
        # Thursday game whose props could only be priced in hindsight.
        upcoming = [g for g in week_games if not g["played"]] or week_games
        game_teams = [t for g in upcoming for t in (g["home_team"], g["away_team"]) if t in teams]
    _sticky("pl_pp_team", teams, game_teams[0] if game_teams else None)
    team = st.selectbox("Team", teams, key="pl_pp_team", format_func=pa.team_label)
    _remember("pl_pp_team")
    roster = {str(p["player_id"]): p for p in pa.players_for_team(pool, team)}
    if not roster:
        bt.render(bt.empty_card_html("No players to price", f"{team} has no players with enough volume for a prop."))
        return None, ""
    _sticky("pl_pp_player", list(roster))
    player_id = st.selectbox("Player", list(roster), key="pl_pp_player", format_func=lambda pid: _player_option(roster[pid]))
    _remember("pl_pp_player")
    player = roster[player_id]
    categories = pa.eligible_categories(player)
    _sticky("pl_pp_cat", categories)
    category = st.selectbox("Stat", categories, key="pl_pp_cat")
    _remember("pl_pp_cat")
    if category is None:
        return None, "This player has no stat that can be priced."
    try:
        projection = pa.project_stat(player, category)
    except ValueError as problem:
        _notices([str(problem)], "warn")
        return None, ""

    status = pa.player_status(player)
    game = _team_game(week_games, ml.normalize_team(projection["team"])) if week_games else None
    _reseed("pl_pp_line", "pl_pp_ctx", (player_id, category), f"{pa.suggest_line(projection):g}")

    line_col, side_col = st.columns([1, 1])
    with line_col:
        line_text = st.text_input(
            "Line",
            key="pl_pp_line",
            placeholder="e.g. 249.5",
            help="Pre-filled with the half-point line nearest the model's projection. Paste your sportsbook's line to price that instead.",
        )
    with side_col:
        side = st.radio("Side", ["over", "under"], key="pl_pp_side", format_func=str.title, horizontal=True)
    _reseed("pl_pp_over", "pl_pp_over_ctx", (player_id, category), f"{pa.DEFAULT_PRICE:.0f}")
    _reseed("pl_pp_under", "pl_pp_under_ctx", (player_id, category), f"{pa.DEFAULT_PRICE:.0f}")
    over_col, under_col = st.columns([1, 1])
    with over_col:
        over_text = st.text_input("Over price", key="pl_pp_over", placeholder="-110")
    with under_col:
        under_text = st.text_input("Under price", key="pl_pp_under", placeholder="-110")

    comma_problems = [
        problem
        for problem in (
            _comma_problem(line_text, "line"),
            _comma_problem(over_text, "over price"),
            _comma_problem(under_text, "under price"),
        )
        if problem
    ]
    if comma_problems:
        _notices(comma_problems, "warn")
        return None, ""
    checked = pa.validate_inputs(category, line_text, over_text, under_text)
    if checked["errors"]:
        _notices(checked["errors"], "warn")
        return None, ""
    if checked["blank_line"]:
        return None, checked["message"]
    _notices(checked["notes"], "steel")
    try:
        evaluation = pa.evaluate_line(player, category, checked["line"], checked["over_price"], checked["under_price"])
    except ValueError as problem:
        _notices([_leg_problem(problem)], "warn")
        return None, ""

    chips = [bt.chip_html(projection["position"] or "Player", "cyan"), bt.chip_html(status.title(), _status_tone(status))]
    chips.append(bt.chip_html(f"Evidence: {evaluation['confidence_label']}", "neutral"))
    if game is not None:
        chips.append(bt.chip_html(game["label"], "steel"))
    bt.render(f'<div class="bt bt-chips">{"".join(chips)}</div>')
    line = evaluation["line"]
    season = projection.get("data_season")
    sample = f"{season} data" if season else "last season's rate"
    bt.render(
        bt.probability_bar_html(
            f"Over {line:g}",
            evaluation["probability_over"],
            f"Under {line:g}",
            evaluation["probability_under"],
            caption=(
                f"Model projection: {projection['mean']:,.1f} {projection['unit']} per game "
                f"(spread about {projection['stdev']:,.1f}), from {projection['games']:g} games of {sample}."
            ),
        )
    )
    priced = evaluation["pricing_over" if side == "over" else "pricing_under"]
    probability = evaluation["probability_over" if side == "over" else "probability_under"]
    tiles = [
        {"label": f"Model {side} probability", "value": format_pct(probability), "sub": "The model's chance this leg wins", "tone": "cyan"},
        {"label": "Price used", "value": format_american(priced["price"]), "sub": f"Source: {SRC_STANDARD if priced['price'] == pa.DEFAULT_PRICE else SRC_USER}", "tone": "cyan"},
        {"label": "Break-even needed", "value": format_pct(priced["implied_probability"]), "sub": "Win rate this price requires", "tone": "cyan"},
        {"label": "Value gap", "value": _signed_pts(priced["value_gap"]), "sub": priced["verdict"], "tone": _tone_for(priced["verdict_tone"])},
    ]
    bt.render(bt.stat_grid_html(tiles, columns=2))

    # The page's own disclosures come first so the module's warnings can never push them off the list.
    caveats = []
    if category == "Passing Yards":
        caveats.append(_passing_caveat(_finite(evaluation.get("cv"))))
    if category.startswith("Anytime TD"):
        caveats.append(
            "Anytime TD counts rushing and receiving touchdowns (not passing). In a parlay it is linked to your other legs "
            "with the same touchdown priors as the other TD props, which are heuristics rather than fitted numbers."
        )
    caveats.extend(evaluation.get("warnings") or [])
    _notices(caveats[:5], "warn")
    if week_games and game is None:
        _notices([f"{projection['team']} has no game in week {st.session_state['pl_week']} (bye week or not scheduled). Pick another week."], "warn")
        return None, ""
    if inputs is None:
        st.caption("The schedule is unavailable, so this prop is linked to your other legs by team name only.")
    if status in status_utils.UNAVAILABLE_STATUSES:
        _notices([f"{projection['name']} is listed {status}. Sportsbooks void props for players who do not play, so this leg cannot be added."], "bad")
        return None, ""
    try:
        defaulted = "over_price" if side == "over" else "under_price"
        candidate = _build_prop_leg(player, evaluation, side, game, SRC_STANDARD if defaulted in checked["defaults_used"] or evaluation["pricing_" + side]["price"] == pa.DEFAULT_PRICE else SRC_USER)
    except ValueError as problem:
        _notices([_leg_problem(problem)], "warn")
        return None, ""
    return candidate, ""


def _render_builder(inputs, inputs_error) -> None:
    # Take the click out of session state first: if anything below raised, a leftover request would add a leg
    # on the user's NEXT interaction, which they never asked for.
    request = st.session_state.pop("pl_add_request", None)
    section_header("Build a leg", "Pick a bet type, set the numbers, add it to the slip.")
    kind = st.segmented_control(
        "Leg type", list(KINDS), key="pl_kind", selection_mode="single", label_visibility="collapsed", on_change=_keep_kind
    )
    kind = kind if kind in KINDS else KIND_ML
    _remember("pl_kind")
    if inputs is not None:
        weeks = ml.weeks_in_schedule(inputs["schedule"])
        _sticky("pl_week", weeks, ml.default_week(inputs["schedule"]))
        st.selectbox("Week", weeks, key="pl_week", format_func=lambda week: f"Week {week}")
        _remember("pl_week")
    candidate, hint = None, ""
    try:
        if kind == KIND_PROP:
            candidate, hint = _render_prop_builder(inputs)
        elif inputs is None:
            _unavailable_card("Game data is unavailable", inputs_error or "The NFL schedule could not be loaded.", "pl_retry_games")
        else:
            game, side = _pick_game(inputs)
            if game is not None:
                if kind == KIND_ML:
                    candidate, hint = _render_moneyline_builder(inputs, game, side)
                else:
                    candidate, hint = _render_spread_builder(inputs, game, side)
    except ValueError as problem:  # a malformed game row must read as a message, never a traceback
        candidate = None
        _notices([f"This selection could not be priced: {problem}"], "warn")
    add_key = {KIND_ML: "ml", KIND_SPREAD: "spread", KIND_PROP: "prop"}[kind]
    drawn = kind == KIND_PROP or inputs is not None
    # The click is handled here, after the leg was priced from this run's widget values and BEFORE the slip,
    # the Add button and the results are drawn, so all of them describe the slip as it is now.
    if drawn and request == add_key:
        _add_leg(candidate)
    flash = st.session_state.pop("pl_flash", None)
    if drawn:
        just_added = bool(flash) and str(flash["text"]).startswith("Added ")
        _add_controls(candidate, add_key, hint, just_added)
    # Feedback for the last add / remove / clear / example sits right under the Add button, where the
    # eye already is (on a phone the slip is further down the page than the button). It is shown even when
    # there is no Add button (game data unavailable), so a message such as "the example needs the schedule"
    # is never swallowed.
    if flash:
        bt.render(bt.notice_html(flash["text"], flash["tone"]))


# ---------------------------------------------------------------------------
# Example parlay
# ---------------------------------------------------------------------------
_EXAMPLE_PROP_CATEGORIES = ("Rushing Yards", "Receiving Yards", "Passing Yards")


def _example_prop(pool, game, team: str):
    """The strongest healthy over for ``team``: lead back's rushing, then a receiver's, then the QB's."""
    for category in _EXAMPLE_PROP_CATEGORIES:
        for player in pa.players_for_team(pool, team, category):
            if pa.player_status(player) in status_utils.UNAVAILABLE_STATUSES:
                continue
            try:
                projection = pa.project_stat(player, category)
                evaluation = pa.evaluate_line(player, category, pa.suggest_line(projection), pa.DEFAULT_PRICE, pa.DEFAULT_PRICE)
                return _build_prop_leg(player, evaluation, "over", game, SRC_STANDARD)
            except ValueError:
                continue
    return None


def _build_example(inputs, pool):
    """A realistic 3-leg slip from live data: a favorite's moneyline, a same-game prop, a spread elsewhere."""
    schedule, ratings, qb_context = inputs["schedule"], inputs["ratings"], inputs["qb_context"]
    games = ml.games_for_week(schedule, ml.default_week(schedule))
    games = [g for g in games if not g["played"]] or games
    rated = []
    for game in games:
        try:
            analysis = ml.analyze_game(game, ratings, qb_context)
        except ValueError:
            continue
        side = analysis["favored"]
        probability = analysis["probabilities"]["home_win" if side == "home" else "away_win"]
        rated.append((abs(probability - 0.68), game, analysis, side))
    rated.sort(key=lambda row: row[0])  # favorites near a realistic 2-to-1, not a -1200 lock
    if not rated:
        return [], "No games are available to build an example from."

    def moneyline_leg(game, analysis, side):
        price, source, _error = _moneyline_price(analysis, side, "")
        return _build_moneyline_leg(game, analysis, side, price, source)

    def spread_leg(game, analysis, side):
        used = analysis["probabilities"]["spread_used_home"]
        line = (-used if side == "home" else used) + 0.0
        return _build_spread_leg(game, analysis, side, line, DEFAULT_LEG_PRICE, SRC_STANDARD)

    anchor, prop_leg = rated[0], None
    if pool:
        for candidate in rated:
            team = candidate[2]["favored_team"]
            prop_leg = _example_prop(pool, candidate[1], team)
            if prop_leg is not None:
                anchor = candidate
                break
    others = [row for row in rated if row[1]["game_id"] != anchor[1]["game_id"]]
    legs: list = []

    def push(builder, row):
        try:
            leg = builder(row[1], row[2], row[3])
        except ValueError:
            return
        if pb.can_add_leg(legs, leg)[0]:
            legs.append(leg)

    push(moneyline_leg, anchor)
    if prop_leg is not None and pb.can_add_leg(legs, prop_leg)[0]:
        legs.append(prop_leg)
        if others:
            push(spread_leg, others[0])
    else:
        for row in others[:1]:
            push(moneyline_leg, row)
        for row in others[1:2]:
            push(spread_leg, row)
    if len(legs) < 2:
        return legs, "Not enough games or players were available to build a full example."
    return legs, ""


def _process_example_request(inputs) -> None:
    if not st.session_state.pop("pl_do_example", False):
        return
    if inputs is None:
        _flash("warn", "The example needs the NFL schedule, which could not be loaded. Try Retry in the builder.")
        return
    pool = None
    pool_result, _error = _load_pool()
    if pool_result:
        pool = pool_result
    with st.spinner("Building an example from the live NFL slate..."):
        legs, problem = _build_example(inputs, pool)
    if len(legs) < 2:
        _flash("warn", problem or "An example could not be built right now.")
        return
    st.session_state["pl_legs"] = legs
    note = "" if pool else " (Player data was unavailable, so it uses game legs only.)"
    _flash("good", f"Loaded an example with {len(legs)} legs from the live slate. Edit, remove or add legs freely.{note}")


# ---------------------------------------------------------------------------
# Slip and pricing inputs
# ---------------------------------------------------------------------------
def _leg_rows(legs, result) -> list[dict]:
    rows = []
    for index, leg in enumerate(legs):
        ui = leg.get("ui") or {}
        try:
            gap = evaluate_price(leg["probability"], leg["price"])["value_gap"]
        except ValueError:
            gap = 0.0
        tone = "good" if gap >= 0.01 else "warn" if gap <= -0.03 else "cyan"
        bits = [ui.get("matchup"), f"Week {ui['week']}" if ui.get("week") else None, f"Price: {ui.get('price_source')}" if ui.get("price_source") else None]
        if ui.get("played"):
            bits.append("Game already final")
        if result is not None and index < len(result["legs"]):
            bits.append(f"Edge vs price {_signed_pts(result['legs'][index]['edge'])}")
        rows.append(
            {
                "label": leg.get("description"),
                "sublabel": " · ".join(str(b) for b in bits if b),
                "type": ui.get("kind") or leg.get("leg_type"),
                "probability": leg.get("probability"),
                "price": leg.get("price"),
                "tone": tone,
            }
        )
    return rows


def _slip_summary(result, stake: float) -> list[tuple[str, str]]:
    if result is None:
        return []
    return [
        ("Combined price", format_american(result["american_odds"])),
        ("Hit probability", _prob_text(result["adjusted_probability"])),
        (f"Payout on {_stake_text(stake)}", format_money(result["payout"])),
        ("Expected value", format_money(result["ev_adjusted"])),
    ]


def _render_slip(legs, result, stake: float) -> None:
    section_header(
        "Your slip",
        "Every leg must win. Each leg keeps the probability and price it had when you added it. Remove any leg with its button.",
    )
    bt.render(bt.parlay_slip_html(_leg_rows(legs, result), title="Parlay slip", summary=_slip_summary(result, stake)))
    if legs:
        with st.container(horizontal=True):
            for number, leg in enumerate(legs, start=1):
                ui = leg.get("ui") or {}
                st.button(
                    f"Remove {number}: {ui.get('short') or _leg_name(leg)}",
                    key=f"pl_rm_{leg['leg_id']}",
                    on_click=_remove_leg,
                    args=(leg["leg_id"],),
                    help=f"Remove {_leg_name(leg)} from the slip",
                )
    first, second = st.columns(2)
    with first:
        st.button("Load an example parlay", key="pl_example", on_click=_request_example, use_container_width=True,
                  help="Replaces the slip with a realistic 3-leg example from the live NFL slate: a favorite's moneyline, a same-game prop and a spread elsewhere.")
    with second:
        st.button("Clear slip", key="pl_clear", on_click=_clear_slip, disabled=not legs, use_container_width=True)


def _book_quote(text) -> tuple[float | None, str | None]:
    """``(price or None, why it was ignored or None)`` for the sportsbook's quote for the whole parlay."""
    price, error = _odds(text, "parlay price")
    if error is None and price is not None and abs(price) > pb.MAX_ABS_BOOK_PRICE:
        error = f"No sportsbook quotes a parlay beyond +/-{pb.MAX_ABS_BOOK_PRICE:,.0f}, so check for a typo."
    return (None, error) if error else (price, None)


def _sync_book_quote_with_slip(legs) -> None:
    """Forget the sportsbook's parlay quote when the legs change.

    A book prices an exact slip. Kept after a leg is added, removed or swapped, the old quote would silently
    override the product of the new legs' prices and show an expected value for a ticket nobody quoted.
    """
    signature = tuple(leg.get("leg_id") for leg in legs)
    previous = st.session_state.get("pl_book_sig")
    st.session_state["pl_book_sig"] = signature
    if previous is not None and previous != signature and str(st.session_state.get("pl_book") or "").strip():
        st.session_state["pl_book"] = ""  # before the text box is drawn this run, so Streamlit allows it
        st.session_state["pl_book_cleared"] = True


def _read_pricing() -> tuple[float, float | None]:
    """Stake and optional book price from session state (widget values are synced before the script runs)."""
    stake = _finite(st.session_state.get("pl_stake"))
    price, _error = _book_quote(st.session_state.get("pl_book"))
    return (stake if stake and stake > 0 else DEFAULT_STAKE), price


def _render_pricing_inputs() -> None:
    section_header("Price this ticket", "Your stake and, ideally, the price your sportsbook quotes for the whole parlay.")
    st.number_input(
        "Stake ($)", min_value=1.0, max_value=100000.0, step=10.0, format="%.2f", key="pl_stake",
        help="Dollars risked. Payout and expected value scale with it.",
    )
    book_text = st.text_input(
        "Sportsbook parlay price (recommended)",
        key="pl_book",
        placeholder="e.g. +650",
        help=(
            "Sportsbooks reprice correlated parlays, so the product of the leg prices usually overstates what they pay. "
            "Paste the combined price your book shows for this exact slip. Leave blank to use the product of your leg prices."
        ),
    )
    price, problem = _book_quote(book_text)
    if st.session_state.pop("pl_book_cleared", False):
        _notices(
            ["Your sportsbook parlay price was cleared because the legs changed. A quote only fits the exact slip it was "
             "given for, so paste a fresh one."],
            "steel",
        )
    if problem:
        _notices([f"Sportsbook parlay price ignored: {problem}"], "warn")
    elif price is None:
        st.caption(_md("Blank means the product of your leg prices. Books reprice correlated legs, so paste their quote for an honest EV."))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _evaluate(legs, stake: float, book_price):
    try:
        return pb.evaluate_parlay_legs(legs, stake=stake, book_price=book_price), None
    except ValueError as problem:
        return None, str(problem)


def _headline_tiles(result, stake: float) -> str:
    adjusted, naive = result["adjusted_probability"], result["naive_probability"]
    book = result["price_source"] == "book_quote"
    ev = result["ev_adjusted"]
    edge = result["edge"]
    fair = format_american(result["fair_american_odds"]) if result["has_fair_price"] else "—"
    tiles = [
        {"label": "Hit probability", "value": _prob_text(adjusted), "sub": f"{_prob_text(naive)} if the legs were independent", "tone": "cyan"},
        {"label": "Combined price", "value": format_american(result["american_odds"]),
         "sub": f"{result['decimal_odds']:.2f}x return, {'your sportsbook quote' if book else 'product of your leg prices'}", "tone": "cyan"},
        {"label": "Payout if it hits", "value": format_money(result["payout"]),
         "sub": f"Profit on a {_stake_text(stake)} stake (total return ${stake + result['payout']:,.2f})", "tone": "cyan"},
        {"label": "Expected value", "value": format_money(ev),
         "sub": f"{_signed_percent(result['ev_pct'])} of stake, {format_money(result['ev_naive'])} if independent",
         "tone": "good" if ev > 0 else "bad" if ev < 0 else "neutral"},
        {"label": "Fair price", "value": fair, "sub": "What the model's hit chance is worth", "tone": "cyan"},
        {"label": "Edge vs price", "value": _signed_pts(edge), "sub": f"Break-even needed: {_prob_text(result['parlay_implied_probability'])}",
         "tone": "good" if edge > 0 else "bad" if edge < 0 else "neutral"},
    ]
    return bt.stat_grid_html(tiles, columns=3)


def _engine_notices(result, legs) -> None:
    for warning in result["warnings"]:
        contradictory = "cannot both win" in warning or "cannot all win" in warning
        bt.render(bt.notice_html(warning, "bad" if contradictory else "warn"))
    fair_priced = [leg for leg in legs if (leg.get("ui") or {}).get("price_source") == SRC_FAIR]
    if fair_priced:
        names = ", ".join(_leg_name(leg) for leg in fair_priced)
        bt.render(bt.notice_html(
            f"{names} {'is' if len(fair_priced) == 1 else 'are'} priced at the model's own fair price, which has no edge by construction. "
            "Any expected value on this ticket comes from the other legs' prices or from correlation, not from that leg.", "steel"))
    final = [leg for leg in legs if (leg.get("ui") or {}).get("played")]
    if final:
        bt.render(bt.notice_html(
            f"{', '.join(_leg_name(leg) for leg in final)} {'is' if len(final) == 1 else 'are'} from a game that is already final. "
            "The ratings include that result, so treat it as a demonstration, not a forecast.", "warn"))


def _render_probability_section(result, stake: float) -> None:
    section_header(
        "Combined probability",
        "The legs multiplied together (naive, as if independent), versus the same legs once they are allowed to move together (adjusted).",
    )
    naive, adjusted = result["naive_probability"], result["adjusted_probability"]
    scale = max(naive, adjusted, 1e-9) * 1.15
    lift = result["correlation_lift"]
    bt.render(bt.meter_html("Independent (leg probabilities multiplied)", naive, max_value=scale, display=_prob_text(naive), tone="steel",
                            caption="What you would get if no leg influenced any other."))
    bt.render(bt.meter_html("Correlation-adjusted (headline)", adjusted, max_value=scale, display=_prob_text(adjusted), tone="cyan",
                            caption=f"{_lift_text(lift)} versus independent. Positive means the legs tend to win together."))
    ev_n, ev_a = result["ev_naive"], result["ev_adjusted"]
    rows = [
        {"metric": "Hit chance", "naive": _prob_text(naive), "adj": _prob_text(adjusted), "diff": _lift_text(lift)},
        {"metric": f"Expected value on {_stake_text(stake)}", "naive": {"value": ev_n, "format": "money"},
         "adj": {"value": ev_a, "format": "money", "tone": "good" if ev_a > 0 else "bad" if ev_a < 0 else "muted"},
         "diff": {"value": ev_a - ev_n, "format": "money"}},
    ]
    columns = [
        {"key": "metric", "label": "Measure"},
        {"key": "naive", "label": "Naive"},
        {"key": "adj", "label": "Adjusted"},
        {"key": "diff", "label": "Change"},
    ]
    bt.render(bt.comparison_table_html(rows, columns, highlight_col="adj"))
    method = result["probability_method"]
    notes = []
    if method == "monte_carlo":
        notes.append(f"Estimated by a fixed-seed simulation (standard error about {result['probability_std_error'] * 100:.2f} pts), so the same slip always prices the same.")
    elif method == "closed_form":
        notes.append("Exact calculation (bivariate normal), no simulation noise.")
    elif method == "independent":
        notes.append("The legs have no modelled link, so the combined probability is the exact product.")
    bounds = result["probability_bounds"]
    notes.append(f"No dependence structure can put this outside {_prob_text(bounds['lower'])} to {_prob_text(bounds['upper'])}.")
    notes.append(
        f"Cross-check: the original parlay engine, which applies a flat +/-12% per detected prop pattern and cannot see moneyline or spread legs, "
        f"gives {_prob_text(result['engine_adjusted_probability'])}."
    )
    if result["psd_adjusted"]:
        notes.append("The pairwise rules disagreed with each other, so the matrix was nudged to the nearest valid one before pricing.")
    st.caption(_md(" ".join(notes)))


#: Where the engine puts the Medium and High tiers on its 0-100 risk score (Low is everything below Medium).
_RISK_MEDIUM_AT = float(getattr(pb, "_RISK_MEDIUM_AT", 25.0))
_RISK_HIGH_AT = float(getattr(pb, "_RISK_HIGH_AT", 50.0))


def _risk_bar_position(score: float) -> float:
    """The engine's 0-100 risk score stretched so each tier fills one third of the kit's three-segment bar.

    The kit draws three EQUAL segments and puts its marker at ``score`` percent of the bar, but the engine's tiers
    start at 25 and 50 (and the engine floors many scores at exactly 25.0 or 50.0). Unstretched, a score of 50 (High)
    puts the marker in the Medium third and 25 (Medium) in the Low third. The stretch keeps the marker inside the
    lit segment, a little in from its edge so it never straddles two segments; the true score is printed under the card.
    """
    value = min(100.0, max(0.0, score))
    third = 100.0 / 3.0
    if value < _RISK_MEDIUM_AT:
        tier_index, fraction = 0, value / _RISK_MEDIUM_AT
    elif value < _RISK_HIGH_AT:
        tier_index, fraction = 1, (value - _RISK_MEDIUM_AT) / (_RISK_HIGH_AT - _RISK_MEDIUM_AT)
    else:
        tier_index, fraction = 2, (value - _RISK_HIGH_AT) / (100.0 - _RISK_HIGH_AT)
    position = (tier_index + 0.08 + 0.84 * fraction) * third
    return max(position, 1.5)  # the kit reads 0-1 as a fraction; the bar position is on 0-100


def _render_risk_section(result) -> None:
    section_header("Risk and confidence", "A rule-of-thumb read of how fragile the ticket is. It is not a prediction.")
    risk_col, conf_col = st.columns([3, 2])
    with risk_col:
        score = float(result["risk_score"])
        bt.render(bt.risk_meter_html(result["risk_tier"], _risk_bar_position(score), factors=result["risk_factors"]))
        st.caption(
            _md(
                f"Risk score {score:.0f} / 100. Low is below {_RISK_MEDIUM_AT:g}, Medium starts at {_RISK_MEDIUM_AT:g}, "
                f"High starts at {_RISK_HIGH_AT:g}. The bar is stretched so each tier fills a third of it."
            )
        )
    with conf_col:
        bt.render(bt.meter_html("Confidence", result["confidence"], display=format_pct(result["confidence"], 0), tone="cyan",
                                caption="How much evidence stands behind the leg estimates, less 4% per leg beyond two. It is not a chance of winning."))
        one_in = _one_in_text(result["adjusted_probability"])
        bt.render(bt.stat_grid_html([
            {"label": "Legs", "value": result["num_legs"], "sub": "Every one must win", "tone": "cyan"},
            {"label": "Hit chance", "value": one_in.replace("about ", "").capitalize() if one_in else "—", "sub": "Model estimate, correlation counted", "tone": "cyan"},
        ], columns=2))


_ACRONYMS = {"ml": "ML", "qb": "QB", "qbs": "QBs", "rb": "RB", "td": "TD", "tds": "TDs"}
_KIND_NAMES = {"run_pass_tradeoff": "Run/pass tradeoff", "teammate_backs_split": "Teammate running-back split"}


def _kind_name(kind: str) -> str:
    if kind in _KIND_NAMES:
        return _KIND_NAMES[kind]
    words = [_ACRONYMS.get(word, word) for word in kind.split("_")]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def _render_correlation_section(result, legs) -> None:
    section_header("Correlation between legs", "How the legs move together. Heuristic priors, not fitted coefficients (see the method notes below).")
    shorts = [(leg.get("ui") or {}).get("short") or _leg_name(leg) for leg in legs]
    labels = list(shorts)
    grid = [[max(-1.0, min(1.0, float(value))) for value in row] for row in result["matrix"]]
    if result["has_contradiction"]:
        clashing = [
            f"{shorts[pair['legs'][0]]} and {shorts[pair['legs'][1]]}"
            for pair in result["correlation_pairs"]
            if pair["direction"] == "contradictory"
        ]
        bt.render(bt.notice_html(
            f"{'; '.join(clashing)} cannot both win. The engine stores a pair like that as -2 internally; the grid clamps it "
            "to -1 for display.", "bad"))
    bt.render(bt.correlation_grid_html(labels, grid, caption="Values are the correlations this exact ticket was priced with. +1 and -1 mark legs that are the same event or exact opposites."))
    items = []
    for pair in result["correlation_pairs"]:
        first, second = pair["legs"]
        if pair["direction"] == "contradictory":
            # No rho here: the pair's stored -0.95 is a display cap, not the "cannot both win" it really is.
            detail = f"Cannot both win. {_kind_name(pair['kind'])}: {pair['note']}"
        else:
            detail = f"{_kind_name(pair['kind'])}: {pair['strength'].lower()} link, rho {pair['rho']:+.2f}. {pair['note']}"
        items.append({
            "title": f"Leg {first + 1} ({shorts[first]}) and Leg {second + 1} ({shorts[second]})",
            "detail": detail,
            "tone": "pro" if pair["direction"] == "positive" else "con",
        })
    if items:
        bt.render(bt.reasoning_panel_html(
            items, title="How your legs connect",
            intro="Supports means the two legs tend to win or lose together, which raises the joint chance above the plain product. Caution means they work against each other."))
    else:
        bt.render(bt.notice_html("No linked pairs: these legs are in different games or have no modelled relationship, so they are priced as independent.", "steel"))


def _render_results(legs, result, error, stake: float) -> None:
    if result is None:
        if len(legs) < 2:
            bt.render(bt.empty_card_html(
                "Add at least two legs to price a parlay" if legs else "Your slip is empty",
                "With two or more legs this page shows the combined probability, expected value, correlation grid and risk tier."
                + (" One more leg and it appears." if legs else " Build a leg in the builder, or load an example parlay."),
            ))
        else:
            bt.render(bt.notice_html(error or "This parlay could not be priced.", "warn"))
        return
    bt.render(bt.verdict_banner_html(result["verdict"], result["summary"], result["verdict_tone"]))
    _engine_notices(result, legs)
    bt.render(_headline_tiles(result, stake))
    _render_probability_section(result, stake)
    _render_risk_section(result)
    _render_correlation_section(result, legs)


# ---------------------------------------------------------------------------
# Method notes
# ---------------------------------------------------------------------------
def _render_method() -> None:
    with st.expander("How this is computed, and where it can mislead", expanded=False):
        st.markdown(_md(
            "**Leg probabilities.** A moneyline leg uses the model's win probability, which converts the model spread with a "
            f"{ml.WIN_PROB_STDEV:g}-point standard deviation backtested on closing lines. A spread leg uses the cover probability, which uses the wider "
            f"{ml.MARGIN_STDEV:g}-point margin spread around your line. A prop leg uses the player engine's distribution of that stat, built from last "
            "season's per-game rate. It is not adjusted for tonight's opponent, weather or role changes.\n\n"
            "**Correlation.** Every correlation here is a heuristic prior written from football structure (game script, shared passing volume), "
            "not a coefficient fitted to joint-outcome data. The moneyline and spread pairs in one game are different: they are threshold events on "
            "the same final margin, so they are exactly nested or opposed. Legs in different games are treated as independent.\n\n"
            "**Combined probability.** A Gaussian copula glues the legs together. Legs with no link multiply exactly; a pair of linked legs uses "
            "an exact bivariate-normal calculation; a group of three or more linked legs uses a fixed-seed Monte Carlo with antithetic draws. "
            "The result is clamped to the Frechet bounds. A Gaussian copula has no tail "
            "dependence, so it will not capture a quarterback injury wiping out a whole stack.\n\n"
            "**Pushes and whole numbers.** A spread or prop line on a whole number can push. The parlay engine treats every leg as push-free, "
            "and real sportsbooks usually drop a pushed leg from the ticket. Prefer half-point lines.\n\n"
            "**Prices.** nflverse moneylines and spreads are consensus references, typically closing numbers, not the price at your sportsbook. "
            "A leg priced at the model's fair price has no edge by construction. Books reprice correlated same-game parlays, so the product of leg prices "
            "usually overstates the payout: paste your book's quoted parlay price for an honest expected value.\n\n"
            "**Risk tier.** Low, Medium or High is a rule of thumb built from the hit probability, the number of legs, the longest-odds leg and how "
            "concentrated the correlation is. Confidence is how much evidence supports the leg estimates. Neither predicts the result."
        ))
        st.markdown("**Correlation rules the engine uses**")
        rows = [
            {"rule": f"{_kind_name(row['kind'])}: {row['summary']}", "prior": row["prior"]}
            for row in pb.correlation_rule_table()
        ]
        bt.render(bt.comparison_table_html(rows, [{"key": "rule", "label": "Pattern and what it means"}, {"key": "prior", "label": "Prior correlation"}]))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
apply_global_theme()
bt.inject_betting_css()
st.markdown(_PAGE_CSS, unsafe_allow_html=True)
_init_state()

page_header(
    "Parlays",
    "Stack moneyline, spread and player-prop legs, then see the real hit chance once correlation is counted, not just the leg prices "
    "multiplied together. Nothing on this page places a bet.",
    eyebrow="Betting · NFL",
)

_inputs, _inputs_error = _load_game_inputs()
if _inputs is not None:
    _notices(_inputs.get("warnings") or [], "steel")
_process_example_request(_inputs)

_build_col, _slip_col = st.columns([1, 1], gap="large")

# Columns can be filled in any order. The builder goes first because pressing "Add leg" changes the slip:
# the slip, the price and every result below must be drawn from the slip as it is AFTER that click.
with _build_col:
    _render_builder(_inputs, _inputs_error)

_legs = list(st.session_state["pl_legs"])
_sync_book_quote_with_slip(_legs)
_stake_value, _book_price_value = _read_pricing()
_result, _result_error = (None, None)
if len(_legs) >= 2:
    _result, _result_error = _evaluate(_legs, _stake_value, _book_price_value)

with _slip_col:
    _render_slip(_legs, _result, _stake_value)
    _render_pricing_inputs()

_render_results(_legs, _result, _result_error, _stake_value)
_render_method()
bt.render(bt.disclaimer_html())
