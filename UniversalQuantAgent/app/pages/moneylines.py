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

# NFL MoneyLines: pick a week, pick a game, see what the model thinks, then paste
# the moneyline your sportsbook is offering to see whether it pays more or less
# than the model's fair price.
#
# This page is thin on purpose. Every number comes from the engine:
#   * modules.nfl_moneyline_analysis -- schedule, ratings, quarterbacks, the
#     nine-metric game analysis, and the price comparison for a pasted moneyline;
#   * modules.odds_input -- parsing pasted odds/spreads and the shared price math;
#   * app.betting_ui -- the Midnight Circuit components.
# The page only chooses what to show and writes the plain-English captions.
#
# Honest by construction: the model is a points model of team efficiency plus a
# quarterback adjustment (see the "How this is computed and its limits" panel),
# the nflverse market line is a consensus reference rather than your book's
# price, and nothing here places a bet.

import html as html_lib
import math
import re
from typing import Any

import streamlit as st

from app import betting_ui as bt
from app.page_runtime import apply_global_theme, page_header, section_header
from app.style import CIRCUIT_CYAN, MIDNIGHT_NAVY, SIGNAL_GOLD, SOFT_WHITE
from betting.odds_math import hold as book_hold
from modules import nfl_moneyline_analysis as nma
from modules import odds_input as oi

apply_global_theme()
bt.inject_betting_css()

#: Plain-English line used wherever the nflverse market columns are blank.
NO_LINE = "No line posted yet"
#: A value gap this wide (probability points) is shown with a caution: a model
#: that disagrees with the price by this much is more often missing information
#: (injuries, weather, a line that has already moved) than it is right.
LARGE_GAP = 0.10
#: A typed price whose break-even rate sits this far (probability points) from the nflverse consensus for the same
#: side is almost never a real offer from another book (books differ from the consensus by a point or two): it is
#: the wrong team picked, a typo or a stale line, and the verdict would then be nonsense.
CONSENSUS_GAP = 0.12
#: Sportsbook spreads are typed in points; anything beyond this is a typo.
MAX_SPREAD = 60.0
#: American prices past this size do not exist at any sportsbook (real NFL moneylines sit well inside
#: +/-5000), so a bigger number is a typo that would otherwise print a 20-digit "expected value".
MAX_PRICE = 100_000.0
#: id of the marker above the analysis; the "skip to the analysis" link under the week picker points at it.
_ANALYSIS_ANCHOR = "ml-analysis"
_PICKEM_WORDS = frozenset({"pk", "pick", "pickem"})
#: ``1,150`` / ``-1,150.5``: a comma used as a thousands separator. Any other comma is a decimal comma.
_THOUSANDS_TEXT = re.compile(r"[+\-−–—‒]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_ROOF_LABELS = {"dome": "Dome", "closed": "Roof closed", "open": "Roof open", "outdoors": "Outdoors"}
_VERDICT_LEAD = {
    "Strong value": "This price pays well above the model's fair price.",
    "Slight value": "This price pays slightly above the model's fair price.",
    "Fair price": "This price is close to the model's fair price.",
    "Overpriced": "This price pays less than the model's fair price, so the model sees negative expected value.",
}
_VERDICT_TILE_TONE = {"success": "good", "info": "cyan", "neutral": "neutral", "danger": "bad", "warning": "warn"}
_CONFIDENCE_TONE = {"High": "good", "Moderate": "cyan", "Low": "warn"}

st.markdown(
    f"""<style>
/* The slate is a CSS grid, so the tiles reflow with the width of the CONTAINER (Streamlit's own columns only
   stack below 640px of browser width, which left 120px-wide tiles beside an open sidebar). Each cell is
   a flex column that pins its button to the bottom, so a row of tiles of unequal height still lines up. */
.st-key-ml_slate {{display:grid !important;grid-template-columns:repeat(auto-fill,minmax(min(100%,14rem),1fr));gap:1rem;align-items:stretch;}}
.st-key-ml_slate > * {{min-width:0;}}
[class*="st-key-ml_cell_"] {{display:flex !important;flex-direction:column;height:100%;gap:0 !important;}}
[class*="st-key-ml_cell_"] > [data-testid="stElementContainer"]:has(.bt-tile) {{flex:1 1 auto;display:flex;flex-direction:column;}}
[class*="st-key-ml_cell_"] > [data-testid="stElementContainer"]:has(.bt-tile) > [data-testid="stMarkdown"] {{flex:1 1 auto;display:flex;flex-direction:column;}}
[class*="st-key-ml_cell_"] > [data-testid="stElementContainer"]:has(.bt-tile) > [data-testid="stMarkdown"] > div {{flex:1 1 auto;align-items:stretch;}}
[class*="st-key-ml_cell_"] [data-testid="stMarkdownContainer"]:has(.bt-tile) {{flex:1 1 auto;display:flex;flex-direction:column;width:100%;}}
[class*="st-key-ml_cell_"] .bt-tile {{flex:1 1 auto;}}
[class*="st-key-ml_pick_"] {{margin-top:.15rem;}}
[class*="st-key-ml_pick_"] button {{
  min-height:2.2rem;font-weight:700;letter-spacing:.02em;
  background:rgba(10,26,47,.78) !important;color:{CIRCUIT_CYAN} !important;
  border:1px solid rgba(0,200,255,.34) !important;box-shadow:none !important;
}}
[class*="st-key-ml_pick_"] button:hover {{border-color:{CIRCUIT_CYAN} !important;background:rgba(0,200,255,.09) !important;}}
[class*="st-key-ml_pick_"] button[kind="primary"] {{
  background:rgba(245,197,66,.12) !important;color:{SIGNAL_GOLD} !important;
  border:1px solid rgba(245,197,66,.62) !important;box-shadow:0 0 16px rgba(245,197,66,.14) !important;
}}
[class*="st-key-ml_pick_"] button p {{color:{CIRCUIT_CYAN} !important;}}
[class*="st-key-ml_pick_"] button[kind="primary"] p {{color:{SIGNAL_GOLD} !important;}}
.ml-jump a {{color:{CIRCUIT_CYAN} !important;font-size:.86rem;font-weight:700;text-decoration:none;letter-spacing:.01em;}}
.ml-jump a:hover {{text-decoration:underline;}}
[class*="st-key-ml_method"] [data-testid="stMarkdownContainer"] {{max-width:64rem;line-height:1.65;}}
/* The matchup card switches to its stacked layout at 820px of BROWSER width, but beside an open sidebar the card is
   only ~430-480px wide up to ~960px, too tight for three columns (team names broke mid-word). Stack it there too. */
@media (min-width:821px) and (max-width:960px) {{
  .bt-matchup:not(.bt-compact) .bt-mu-body {{grid-template-columns:repeat(2,minmax(0,1fr));}}
  .bt-matchup:not(.bt-compact) .bt-mu-mid {{grid-column:1 / -1;grid-row:2;flex-direction:row;flex-wrap:wrap;gap:.3rem 1rem;min-width:0;}}
  .bt-matchup:not(.bt-compact) .bt-mu-prob {{font-size:1.75rem;}}
}}
/* Same band for the stat tiles: three ~130px tiles broke "24.9 - 19.4" over two lines, so wait for 11rem per tile. */
@media (min-width:821px) and (max-width:1100px) {{
  .bt-grid {{grid-template-columns:repeat(auto-fill,minmax(max(11rem,calc((100% - (var(--bt-cols,4) - 1) * .7rem) / var(--bt-cols,4))),1fr));}}
}}
</style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Small formatting helpers (display only; no model math lives here)
# --------------------------------------------------------------------------
def _num(value: Any) -> float | None:
    """A finite float, or ``None`` for None / NaN / inf / non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _spread_label(home: str, away: str, spread_home: Any) -> str:
    """``DAL -7.2`` / ``TB -3.5`` / ``Pick'em`` from the nflverse convention (positive = home favored)."""
    value = _num(spread_home)
    if value is None:
        return ""
    rounded = round(value, 1)
    if rounded == 0:
        return "Pick'em"
    return f"{home} -{rounded:g}" if rounded > 0 else f"{away} -{abs(rounded):g}"


def _spread_reading(home: str, away: str, sportsbook_spread: float) -> str:
    """Echo a typed sportsbook spread back in words (``-3.5`` -> ``DAL favored by 3.5``) so the sign cannot be misread."""
    points = round(abs(sportsbook_spread), 2)
    if points == 0:
        return "pick'em, neither team favored."
    if sportsbook_spread < 0:
        return f"{home} favored by {points:g}."
    return f"{away} favored by {points:g} ({home} +{points:g})."


def _signed(value: Any, digits: int = 1) -> str | None:
    """``+1.4`` / ``-0.6`` / ``+0.0``: a value that rounds to zero never prints as the odd-looking ``-0.0``."""
    number = _num(value)
    if number is None:
        return None
    return f"{0.0 if round(number, digits) == 0 else number:+.{digits}f}"


def _pts(value: Any, digits: int = 1) -> str | None:
    return _signed(value, digits)


def _signed_pct(points: Any) -> str:
    """:func:`oi.format_signed_pct` (``+3.2 pts``) without a ``-0.0 pts`` for a gap that rounds to nothing."""
    number = _num(points)
    if number is not None and round(number * 100, 1) == 0:
        return oi.format_signed_pct(0.0)
    return oi.format_signed_pct(points)


def _week_label(week: int) -> str:
    return f"Week {week}"


def _stake_text(stake: float) -> str:
    """``$100`` for a whole-dollar stake, ``$25.50`` otherwise."""
    return f"${stake:,.0f}" if float(stake).is_integer() else f"${stake:,.2f}"


def _has_decimal_comma(text: str) -> bool:
    """True for a comma that is not a thousands separator (``1,91``, ``-3,5``).

    :mod:`modules.odds_input` drops every comma so ``1,150`` reads as 1150, which
    would silently turn the European ``-3,5`` into -35 and ``1,91`` into +191.
    """
    squeezed = re.sub(r"\s", "", text)
    return "," in squeezed and _THOUSANDS_TEXT.fullmatch(squeezed) is None


def _parse_price(text: Any) -> oi.ParsedOdds:
    """Pasted moneyline text -> American price, with the page's own guards on top of ``oi.parse_odds``.

    Blank is "no price yet" (no error). A decimal comma and a price too large to
    exist are friendly errors instead of a silently different number.
    """
    raw = "" if text is None else str(text)
    if _has_decimal_comma(raw):
        return oi.ParsedOdds(None, "Use a period for decimal odds (1.91, not 1,91). American odds like -150 or +120 need no comma.", "invalid")
    parsed = oi.parse_odds(raw)
    if parsed.price is not None and abs(parsed.price) > MAX_PRICE:
        return oi.ParsedOdds(None, f"No sportsbook offers a price beyond +/-{MAX_PRICE:,.0f}. Check the number for a typo.", "invalid")
    return parsed


def _parse_sportsbook_spread(text: Any) -> oi.ParsedNumber:
    """Sportsbook spread text -> number in sportsbook notation (``-3.5`` = the home team is favored).

    ``PK`` / ``pick'em`` (and nothing else) are read as 0. Everything else goes
    through :func:`modules.odds_input.parse_number`; its generic wording is
    replaced by one that explains the sign convention.
    """
    raw = "" if text is None else str(text)
    if re.sub(r"[\s.'’`\-]", "", raw.lower()) in _PICKEM_WORDS:
        return oi.ParsedNumber(0.0, None)
    if "," in raw:  # a spread never has a thousands separator, so a comma is a decimal comma ("-3,5")
        return oi.ParsedNumber(None, "Use a period for the decimal point (-3.5, not -3,5).")
    parsed = oi.parse_number(raw, allow_negative=True, minimum=-MAX_SPREAD, maximum=MAX_SPREAD, label="spread")
    if parsed.error:
        return oi.ParsedNumber(
            None,
            f"Enter the spread as a number between -{MAX_SPREAD:g} and +{MAX_SPREAD:g}, "
            "like -3.5 (home team favored) or +3.5 (home team underdog). PK means pick'em.",
        )
    return parsed


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _load_inputs(season: int) -> dict[str, Any]:
    """Schedule, blended ratings and quarterback context for ``season`` (about 7-12 s cold)."""
    return nma.load_moneyline_inputs(season)


def _get_inputs(season: int) -> dict[str, Any]:
    """:func:`_load_inputs`, but a load that came back with warnings is not pinned for an hour.

    The engine fails soft (a missing play-by-play file becomes league-average
    ratings plus a warning) and deliberately remembers such a degraded result
    for only about a minute, so a network blip heals itself. ``st.cache_data``
    would otherwise freeze that degraded result for its whole hour. Dropping the
    page-level entry whenever there are warnings hands the timing back to the
    engine, whose own memo makes the repeat call cost milliseconds.
    """
    inputs = _load_inputs(season)
    if inputs.get("warnings"):
        _load_inputs.clear()
    return inputs


def _clear_all_caches() -> None:
    st.cache_data.clear()
    nma.clear_caches()


def _render_load_failure(title: str, body: str) -> None:
    bt.render(bt.empty_card_html(title, body))
    st.button("Retry", key="ml_retry", on_click=_clear_all_caches, help="Clears the cached data and loads it again.")


# --------------------------------------------------------------------------
# Selection state
# --------------------------------------------------------------------------
def _set_grid_open(is_open: bool) -> None:
    """Fold the tile grid up (a game was picked) or open it again (a new week); see :func:`_render_game_grid`."""
    st.session_state["ml_grid_open"] = is_open
    st.session_state["ml_grid_epoch"] = int(st.session_state.get("ml_grid_epoch", 0)) + 1


def _on_week_change() -> None:
    st.session_state.pop("ml_game", None)
    _set_grid_open(True)


def _pick_game(game_id: str) -> None:
    st.session_state["ml_game"] = game_id
    st.session_state["ml_scroll_pending"] = True  # see _scroll_to_analysis: the folded slate would otherwise leave the viewport mid-analysis
    _set_grid_open(False)


def _scroll_to_analysis() -> None:
    """Bring the matchup card into view after a tile was picked (runs once per pick).

    Folding the slate shortens the page by up to several screens. A tile tapped
    far down the grid on a phone therefore left the viewport in the middle of
    the price check, and the user had to scroll back up to find the game they
    had just chosen. Streamlit has no scroll API, so this tiny script (static
    text; nothing user-typed ever reaches it) scrolls to the analysis anchor.
    The pick counter makes the text differ on every pick so the browser runs it again.
    """
    picks = int(st.session_state.get("ml_grid_epoch", 0))
    st.html(
        f"<script>/* pick {picks} */(function(){{setTimeout(function(){{var a=document.getElementById('{_ANALYSIS_ANCHOR}');"
        "if(a){a.scrollIntoView({block:'start'});}},180);})();</script>",
        unsafe_allow_javascript=True,
    )


def _selected_game(games: list[dict[str, Any]]) -> dict[str, Any]:
    """The picked game, else the first upcoming game of the week, else the first game."""
    by_id = {game["game_id"]: game for game in games}
    picked = st.session_state.get("ml_game")
    if picked in by_id:
        return by_id[picked]
    chosen = next((game for game in games if not game["played"]), games[0])
    st.session_state["ml_game"] = chosen["game_id"]
    return chosen


def _sync_inputs_to_game(game_id: str) -> None:
    """Price boxes belong to one game: a new game starts blank, and the side resets to the model's pick.

    Runs before the input widgets exist this run, so session state may be
    written (Streamlit forbids it once a widget with that key is drawn).
    """
    if st.session_state.get("ml_inputs_game") == game_id:
        return
    st.session_state["ml_inputs_game"] = game_id
    for key in ("ml_price", "ml_price_other", "ml_spread"):
        st.session_state[key] = ""
    st.session_state.pop("ml_side", None)


# --------------------------------------------------------------------------
# Game grid
# --------------------------------------------------------------------------
def _market_line_text(game: dict[str, Any]) -> str:
    label = _spread_label(game["home_team"], game["away_team"], game.get("market_spread_home"))
    if not label:
        return NO_LINE
    total = _num(game.get("market_total"))
    return f"Consensus {label}" + (f" · O/U {total:g}" if total is not None else "")


def _tile_lean(game: dict[str, Any], ratings: Any, qb_context: Any) -> str:
    if game["played"]:
        return ""
    try:
        quick = nma.analyze_game(game, ratings, qb_context)
    except Exception:  # noqa: BLE001 - a tile must never take the page down
        return ""
    favored = quick["favored"]
    return f"Model: {quick['favored_team']} {oi.format_pct(quick['teams'][favored]['win_probability'], 0)}"


def _week_summary_html(season: Any, week: int, games: list[dict[str, Any]]) -> str:
    upcoming = sum(1 for game in games if not game["played"])
    final = len(games) - upcoming
    chips = [bt.chip_html(f"{season} season" if season else "NFL season", "cyan"), bt.chip_html(_week_label(week), "gold")]
    chips.append(bt.chip_html(f"{len(games)} game{'s' if len(games) != 1 else ''}", "steel"))
    chips.append(bt.chip_html(f"{upcoming} upcoming", "good" if upcoming else "neutral"))
    chips.append(bt.chip_html(f"{final} final", "neutral"))
    return f'<div class="bt bt-chips" style="margin:0 0 .5rem">{"".join(chips)}</div>'


def _render_game_grid(games: list[dict[str, Any]], selected_id: str, ratings: Any, qb_context: Any, week: int) -> None:
    open_grid = bool(st.session_state.get("ml_grid_open", True))
    # The key carries an epoch that changes on every pick / week change. The browser keeps an expander open
    # or shut across reruns, so a new key is what makes the grid fold up after a game is picked (even when
    # the user had opened it by hand) and open again after a week change, instead of leaving the analysis
    # a full screen below the tiles.
    plural = "s" if len(games) != 1 else ""
    if open_grid:
        label = f"{_week_label(week)} slate: {len(games)} game{plural}. Pick one to analyze."
    else:
        picked = next((game["label"] for game in games if game["game_id"] == selected_id), "")
        label = f"{_week_label(week)} slate: {len(games)} game{plural}. {picked} is selected. Open to pick another."
    with st.expander(label, expanded=open_grid, key=f"ml_grid_{int(st.session_state.get('ml_grid_epoch', 0))}"):
        with st.container(key="ml_slate"):  # a CSS grid (see the style block): auto-fits as many tiles per row as the width allows
            for game in games:
                is_selected = game["game_id"] == selected_id
                with st.container(key=f"ml_cell_{game['game_id']}"):
                    subtitle = _market_line_text(game) + (" · Neutral site" if game.get("neutral_site") else "")
                    bt.render(
                        bt.game_tile_html(
                            away_team=game["away_team"],
                            home_team=game["home_team"],
                            kickoff=game["kickoff_label"],
                            status=game["status"],
                            selected=is_selected,
                            subtitle=subtitle,
                            model_lean=_tile_lean(game, ratings, qb_context),
                        )
                    )
                    st.button(
                        "Selected" if is_selected else "Analyze this game",
                        key=f"ml_pick_{game['game_id']}",
                        on_click=_pick_game,
                        args=(game["game_id"],),
                        type="primary" if is_selected else "secondary",
                        width="stretch",
                    )


# --------------------------------------------------------------------------
# Matchup hero
# --------------------------------------------------------------------------
def _qb_note(team: dict[str, Any], game: dict[str, Any], side: str) -> str:
    """``Dak Prescott (assumed)`` / ``Case Keenum (backup)``: who is under center and how sure we are."""
    name = team.get("qb_name")
    if not name:
        return "not listed"
    flags = []
    if game.get(f"{side}_qb_source") == "last_game" or team.get("qb_source") in ("team_primary", "none"):
        flags.append("assumed")
    if team.get("qb_role") == "backup":
        flags.append("backup")
    elif team.get("qb_role") == "unknown":
        flags.append("unproven")
    return f"{name} ({', '.join(flags)})" if flags else str(name)


def _context_chips(analysis: dict[str, Any]) -> list[str]:
    ctx = analysis["context"]
    home, away = analysis["home_team"], analysis["away_team"]
    chips: list[str] = []
    if ctx.get("neutral_site"):
        chips.append("Neutral site")
    if ctx.get("div_game"):
        chips.append("Divisional")
    roof = ctx.get("roof")
    if roof:
        chips.append(_ROOF_LABELS.get(str(roof).lower(), str(roof).title()))
    wind, temp = _num(ctx.get("wind")), _num(ctx.get("temp"))
    if wind is not None and str(roof).lower() not in ("dome", "closed"):
        chips.append(f"Wind {wind:.0f} mph")
    if temp is not None and str(roof).lower() not in ("dome", "closed"):
        chips.append(f"{temp:.0f}°F")
    home_rest, away_rest = _num(ctx.get("home_rest")), _num(ctx.get("away_rest"))
    if home_rest is not None and away_rest is not None:
        chips.append(f"Rest: {home} {home_rest:.0f}d / {away} {away_rest:.0f}d")
    return chips


def _market_texts(analysis: dict[str, Any]) -> tuple[str, str]:
    """``(spread_text, total_text)`` for the matchup card: the consensus line, or the plain "no line" wording."""
    spread = analysis["probabilities"].get("market_spread_home")
    label = _spread_label(analysis["home_team"], analysis["away_team"], spread)
    total = _num(analysis["scoring"].get("market_total"))
    spread_text = f"Consensus: {label}" if label else NO_LINE
    total_text = f"O/U {total:g}" if total is not None else ""
    return spread_text, total_text


def _render_hero(analysis: dict[str, Any], game: dict[str, Any]) -> None:
    teams = analysis["teams"]
    spread_text, total_text = _market_texts(analysis)
    kickoff = analysis["kickoff_label"]
    if analysis["played"]:
        kickoff = f"{kickoff} · {analysis['status']}"
    bt.render(
        bt.matchup_card_html(
            away={
                "team": analysis["away_team"],
                "name": teams["away"]["team_name"],
                "subtitle": f"QB {_qb_note(teams['away'], game, 'away')}",
                "win_probability": teams["away"]["win_probability"],
                "rating": teams["away"]["strength_rating"],
                "expected_points": teams["away"]["expected_points"],
            },
            home={
                "team": analysis["home_team"],
                "name": teams["home"]["team_name"],
                "subtitle": f"QB {_qb_note(teams['home'], game, 'home')}",
                "win_probability": teams["home"]["win_probability"],
                "rating": teams["home"]["strength_rating"],
                "expected_points": teams["home"]["expected_points"],
            },
            kickoff=kickoff,
            chips=_context_chips(analysis),
            favored=analysis["favored_team"],
            spread_text=spread_text,
            total_text=total_text,
        )
    )

    if analysis["played"]:
        away_score, home_score = game.get("away_score"), game.get("home_score")
        score = (
            f"Final: {analysis['away_team']} {away_score}, {analysis['home_team']} {home_score}. "
            if away_score is not None and home_score is not None
            else "Final. "
        )
        bt.render(
            bt.notice_html(
                score + "This game has already been played, so the ratings already include its result. "
                "The numbers below show how the model sees the matchup, not a live bet.",
                "gold",
            )
        )
    if analysis["context"].get("neutral_site"):
        bt.render(
            bt.notice_html(
                f"Neutral-site game: {analysis['home_team']} is only the designated home team, so the model gives "
                f"no home-field edge (it normally adds {nma.HOME_FIELD_POINTS:g} points).",
                "info",
            )
        )

    notes = [w for w in analysis["warnings"] if not w.startswith("This game has already been played")]
    for note in notes[:3]:
        bt.render(bt.notice_html(note, "warn"))
    if len(notes) > 3:
        with st.expander(f"{len(notes) - 3} more note{'s' if len(notes) - 3 != 1 else ''} from the model", key="ml_notes"):
            for note in notes[3:]:
                bt.render(bt.notice_html(note, "warn"))


def _headline_tiles(analysis: dict[str, Any], game: dict[str, Any]) -> list[dict[str, Any]]:
    home, away = analysis["home_team"], analysis["away_team"]
    teams, scoring, pace = analysis["teams"], analysis["scoring"], analysis["pace"]
    favored = analysis["favored"]
    market_spread = analysis["probabilities"].get("market_spread_home")
    market_label = _spread_label(home, away, market_spread)
    market_total, total_gap = _num(scoring.get("market_total")), _num(scoring.get("total_vs_market"))

    qb_swing = _num(scoring["adjustments"].get("qb_spread")) or 0.0
    qb_team = home if qb_swing >= 0 else away
    if abs(qb_swing) < 0.05:
        qb_value, qb_sub = "Even", "Quarterbacks roughly cancel out"
    else:
        qb_value = f"{qb_team} {abs(qb_swing):+.1f} pts"
        qb_sub = f"{_qb_note(teams['away'], game, 'away')} at {_qb_note(teams['home'], game, 'home')}: net effect on the spread"

    return [
        {
            "label": "Model favorite",
            "value": analysis["favored_team"],
            "sub": f"{oi.format_pct(teams[favored]['win_probability'])} to win, by about {analysis['margin']:.1f} points",
            "tone": "gold",
        },
        {
            "label": "Model spread",
            "value": _spread_label(home, away, scoring["model_spread_home"]),
            "sub": f"Consensus line: {market_label}" if market_label else NO_LINE,
            "tone": "cyan",
        },
        {
            "label": "Model total",
            "value": f"{scoring['model_total']:.1f}",
            "sub": (
                f"Consensus total {market_total:g}" + (f" (model {_signed(total_gap)})" if total_gap is not None else "")
                if market_total is not None
                else NO_LINE
            ),
            "tone": "cyan",
        },
        {
            "label": "Projected score",
            "value": f"{scoring['away_expected_points']:.1f} – {scoring['home_expected_points']:.1f}",
            "sub": f"{away} – {home} expected points",
            "tone": "cyan",
        },
        {
            "label": "Pace projection",
            "value": f"{pace['plays_projection']:.0f} plays",
            "sub": f"{pace['label']} · {pace['seconds_per_play']:.1f} sec/play (league {pace['league_plays']:.0f} plays)",
            "tone": "cyan",
        },
        {"label": "Quarterback edge", "value": qb_value, "sub": qb_sub, "tone": "cyan"},
    ]


# --------------------------------------------------------------------------
# Price check
# --------------------------------------------------------------------------
def _side_label_for(analysis: dict[str, Any]):
    def label(side: str) -> str:
        team = analysis["teams"][side]
        return f"{team['team']} · {team['team_name']}"

    return label


def _render_inputs(analysis: dict[str, Any]) -> tuple[str, oi.ParsedOdds, oi.ParsedOdds, float, oi.ParsedNumber]:
    """Draw the price inputs; return the chosen side and the parsed values."""
    home, away = analysis["home_team"], analysis["away_team"]
    if "ml_side" not in st.session_state:
        st.session_state["ml_side"] = analysis["favored"]
    side = st.radio(
        "The price is for",
        ["away", "home"],
        format_func=_side_label_for(analysis),
        key="ml_side",
        horizontal=True,
        help="Defaults to the team the model favors. Switch it to price the other side.",
    )
    # Bottom-aligned: beside an open sidebar the labels wrap to different line counts, and top-aligned boxes then staggered.
    price_col, other_col, stake_col = st.columns(3, vertical_alignment="bottom")
    with price_col:
        st.text_input(
            "Sportsbook moneyline",
            key="ml_price",
            placeholder="-150 or +120",
            help="American odds as your sportsbook shows them. Decimal odds like 1.91 also work.",
        )
    with other_col:
        st.text_input(
            "Other team's moneyline (optional)",
            key="ml_price_other",
            placeholder="optional",
            help="The opposite side's price at the same book. It lets the page remove the sportsbook's margin (vig) and show the market's true probability.",
        )
    with stake_col:
        stake = st.number_input(
            "Stake in dollars",
            min_value=1.0,
            max_value=1_000_000.0,
            value=100.0,
            step=10.0,
            key="ml_stake",
            help="Used only to express expected value in dollars.",
        )
    spread_col, note_col = st.columns([1, 2], vertical_alignment="center")
    with spread_col:
        spread_text = st.text_input(
            f"Sportsbook spread for {home} (optional)",
            key="ml_spread",
            placeholder="-3.5",
            help=f"In sportsbook notation for the home team: -3.5 means {home} is favored by 3.5, +3.5 means {away} is favored by 3.5. PK is a pick'em.",
        )
    parsed_spread = _parse_sportsbook_spread(spread_text)
    with note_col:
        st.caption(
            f"Optional. Type the spread your book shows for {home}, the home team: -3.5 means {home} is favored, "
            f"+3.5 means {away} is favored. It updates the spread probability below."
            + (f" Read as: {_spread_reading(home, away, parsed_spread.value)}" if parsed_spread.value is not None else "")
        )
    parsed_price = _parse_price(st.session_state.get("ml_price", ""))
    parsed_other = _parse_price(st.session_state.get("ml_price_other", ""))
    return side, parsed_price, parsed_other, float(stake), parsed_spread


def _price_tiles(ev: dict[str, Any], stake: float, no_vig: tuple[float, float] | None, book_margin: float | None, analysis: dict[str, Any], side: str) -> list[dict[str, Any]]:
    team = ev["team"]
    gap_tone = _VERDICT_TILE_TONE.get(ev["verdict_tone"], "neutral")
    ev_value = ev["expected_value"]
    tiles = [
        {
            "label": "Implied probability",
            "value": oi.format_pct(ev["implied_probability"]),
            "sub": f"Break-even win rate at {oi.format_american(ev['price'])}, sportsbook margin included",
            "tone": "cyan",
        },
        {
            "label": "Model probability",
            "value": oi.format_pct(ev["model_probability"]),
            "sub": f"The model's chance that {team} wins",
            "tone": "gold" if analysis["favored"] == side else "cyan",
        },
        {
            "label": "Value gap",
            "value": _signed_pct(ev["value_gap"]),
            "sub": "Model minus implied. Positive means the price pays more than the model thinks is fair",
            "tone": gap_tone,
        },
        {
            "label": f"Expected value per {_stake_text(stake)}",
            "value": oi.format_money(ev_value),
            "sub": f"Average profit per bet if the model's {oi.format_pct(ev['model_probability'])} is the true chance ({_signed(ev['ev_pct'] * 100)}% of the stake). Not a promise",
            # A "Fair price" verdict must not sit beside a red -$0.07: inside the fair band the dollars are neutral.
            "tone": "neutral" if ev["verdict"] == "Fair price" else "good" if ev_value > 0 else "bad" if ev_value < 0 else "neutral",
        },
        {
            "label": "Model fair price",
            "value": oi.format_american(ev["fair_price"]),
            "sub": "The price at which the model sees no edge either way",
            "tone": "cyan",
        },
    ]
    if no_vig is not None:
        tiles.append(
            {
                "label": "Your book, margin removed",
                "value": oi.format_pct(no_vig[0]),
                "sub": f"{team} after removing the book's {oi.format_pct(book_margin)} margin from your two prices",
                "tone": "steel",
            }
        )
    elif ev.get("market_probability_novig") is not None:
        tiles.append(
            {
                "label": "Consensus, margin removed",
                "value": oi.format_pct(ev["market_probability_novig"]),
                "sub": f"nflverse consensus chance for {team}, a reference only",
                "tone": "steel",
            }
        )
    return tiles


def _model_vs_market_rows(
    analysis: dict[str, Any], side: str, parsed_price: oi.ParsedOdds, ev: dict[str, Any] | None,
    no_vig: tuple[float, float] | None, parsed_spread: oi.ParsedNumber,
) -> list[dict[str, Any]]:
    home, away = analysis["home_team"], analysis["away_team"]
    team = analysis["teams"][side]
    ml = analysis["moneyline"]
    model_prob = team["win_probability"]
    consensus_price = ml.get(f"market_{side}_ml")
    consensus_novig = ml.get(f"market_{side}_prob_novig")
    scoring = analysis["scoring"]

    user_spread_label = ""
    if parsed_spread.value is not None:
        user_spread_label = _spread_label(home, away, 0.0 - parsed_spread.value)
    market_label = _spread_label(home, away, analysis["probabilities"].get("market_spread_home"))
    market_total = _num(scoring.get("market_total"))

    return [
        {
            "metric": f"{team['team']} moneyline",
            "model": {"value": team["fair_moneyline"], "format": "american"},
            "book": {"value": parsed_price.price if ev else None, "format": "american"},
            "consensus": {"value": consensus_price, "format": "american"},
        },
        {
            "metric": "Win probability (margin included)",
            "model": {"value": model_prob, "format": "pct"},
            "book": {"value": ev["implied_probability"] if ev else None, "format": "pct"},
            "consensus": None,
        },
        {
            "metric": "Win probability (margin removed)",
            "model": {"value": model_prob, "format": "pct"},
            "book": {"value": no_vig[0] if no_vig else None, "format": "pct"},
            "consensus": {"value": consensus_novig, "format": "pct"},
        },
        {
            "metric": "Spread",
            "model": _spread_label(home, away, scoring["model_spread_home"]),
            "book": user_spread_label or None,
            "consensus": market_label or None,
        },
        {
            "metric": "Total points",
            "model": f"{scoring['model_total']:.1f}",
            "book": None,
            "consensus": f"{market_total:g}" if market_total is not None else None,
        },
    ]


def _render_price_check(analysis: dict[str, Any]) -> None:
    """Inputs, verdict and the model-vs-market table (the sportsbook spread was already applied upstream)."""
    section_header(
        "Price check",
        "Paste the moneyline your sportsbook is offering. The page compares it with the model's fair price.",
    )
    side, parsed_price, parsed_other, stake, parsed_spread = _render_inputs(analysis)
    team_code = analysis["teams"][side]["team"]
    other_side = "home" if side == "away" else "away"
    other_code = analysis["teams"][other_side]["team"]

    if parsed_spread.error:
        bt.render(bt.notice_html(parsed_spread.error, "bad"))

    ev: dict[str, Any] | None = None
    no_vig: tuple[float, float] | None = None
    book_margin: float | None = None

    if parsed_price.error:
        bt.render(bt.notice_html(f"Moneyline not understood. {parsed_price.error}", "bad"))
    elif parsed_price.price is None:
        bt.render(
            bt.empty_card_html(
                "Paste a sportsbook moneyline to price it",
                f"Type the price your book shows for {team_code}, like -150 or +120. "
                "Nothing is calculated until you do, and nothing is ever submitted.",
            )
        )
        if parsed_other.price is not None:
            bt.render(bt.notice_html(f"Enter the {team_code} price too, so the two sides can be compared.", "info"))
    else:
        try:
            ev = nma.evaluate_moneyline_price(analysis, side, parsed_price.price, stake=stake)
        except ValueError as error:
            bt.render(bt.notice_html(f"That price could not be priced: {error}", "bad"))

    if parsed_other.error:
        bt.render(bt.notice_html(f"Other team's moneyline not understood. {parsed_other.error}", "bad"))
    elif parsed_other.price is not None and parsed_price.price is not None and not parsed_price.error:
        try:
            book_margin = book_hold(parsed_price.price, parsed_other.price)
            no_vig = oi.no_vig_probabilities(parsed_price.price, parsed_other.price)
        except ValueError:
            no_vig = None
        if no_vig is not None and book_margin is not None and book_margin < 0:
            bt.render(
                bt.notice_html(
                    f"The {team_code} and {other_code} prices add up to less than 100%, so they cannot be the two "
                    "sides of one market. Check both numbers.",
                    "warn",
                )
            )
            no_vig = book_margin = None

    if ev is not None:
        lead = _VERDICT_LEAD.get(ev["verdict"], "")
        detail = (
            f"{lead} {team_code} at {oi.format_american(ev['price'])} needs to win {oi.format_pct(ev['implied_probability'])} "
            f"of the time to break even. The model gives {team_code} {oi.format_pct(ev['model_probability'])} "
            f"({_signed_pct(ev['value_gap'])}), worth {oi.format_money(ev['expected_value'])} on average on a "
            f"{_stake_text(stake)} stake if the model is right. The model's fair price is {oi.format_american(ev['fair_price'])}."
        )
        bt.render(bt.verdict_banner_html(ev["verdict"], detail, ev["verdict_tone"]))
        bt.render(bt.stat_grid_html(_price_tiles(ev, stake, no_vig, book_margin, analysis, side), columns=3))
        if abs(ev["value_gap"]) >= LARGE_GAP:
            bt.render(
                bt.notice_html(
                    "A gap this large is more often a sign the model is missing something (an injury, the weather, "
                    "a line that has already moved) than a real edge. Treat it as a reason to double-check.",
                    "warn",
                )
            )
        consensus = ev.get("market_probability_novig")
        typed_rate = no_vig[0] if no_vig is not None else ev["implied_probability"]
        if consensus is not None and abs(typed_rate - consensus) >= CONSENSUS_GAP:
            bt.render(
                bt.notice_html(
                    f"Your {team_code} price implies {oi.format_pct(typed_rate)}, but the nflverse consensus puts {team_code} at "
                    f"{oi.format_pct(consensus)} (margin removed). A difference this wide usually means the price is for the other "
                    f"team ({other_code}), a typo or an old line. Check the team picked above before trusting the verdict.",
                    "warn",
                )
            )
        if book_margin is not None and book_margin > 0.15:
            bt.render(
                bt.notice_html(
                    f"The two prices carry a {oi.format_pct(book_margin)} margin, which is unusually large for one "
                    f"market. Make sure the other price is for {other_code}.",
                    "warn",
                )
            )
        st.caption("A positive value gap means the model likes the price. It does not mean the bet will win.")

    section_header(
        "Model vs. market",
        "The model's numbers beside your sportsbook's price and the nflverse consensus, for the team picked above.",
    )
    rows = _model_vs_market_rows(analysis, side, parsed_price, ev, no_vig, parsed_spread)
    bt.render(
        bt.comparison_table_html(
            rows,
            [
                {"key": "metric", "label": "Measure"},
                {"key": "model", "label": "Model"},
                {"key": "book", "label": "Your sportsbook"},
                {"key": "consensus", "label": "nflverse consensus"},
            ],
            highlight_col="model",
        )
    )
    market_missing = (
        analysis["probabilities"].get("market_spread_home") is None
        and analysis["scoring"].get("market_total") is None
        and analysis["moneyline"].get("market_home_ml") is None
    )
    if market_missing:
        bt.render(
            bt.notice_html(
                f"{NO_LINE}. nflverse leaves the market columns blank until a consensus line exists, "
                "so compare the model against your own sportsbook's numbers.",
                "info",
            )
        )
    st.caption(
        "The nflverse line is a consensus reference (usually the closing line), not the price at your sportsbook. "
        "Use it to sanity-check the model, not as an offer. Margin included is the break-even rate of a price as "
        "quoted; margin removed is the market's own probability once the sportsbook's cut is taken out "
        "(it needs both sides' prices, or the nflverse moneylines)."
    )


# --------------------------------------------------------------------------
# Model breakdown, reasoning, methodology
# --------------------------------------------------------------------------
def _spread_source_text(analysis: dict[str, Any]) -> str:
    home, away = analysis["home_team"], analysis["away_team"]
    probs = analysis["probabilities"]
    label = _spread_label(home, away, probs["spread_used_home"])
    source = probs["spread_source"]
    if source == "user":
        return f"your sportsbook spread, {label}"
    if source == "market":
        return f"the nflverse consensus spread, {label}"
    return f"the model's own spread rounded to the nearest half point, {label} (no line is posted)"


def _breakdown_rows(analysis: dict[str, Any], game: dict[str, Any]) -> list[dict[str, Any]]:
    teams = analysis["teams"]
    ml = analysis["moneyline"]

    def both(label: str, key: str, fmt: str = "dec1") -> dict[str, Any]:
        return {"metric": label, **{side: {"value": teams[side][key], "format": fmt} for side in ("away", "home")}}

    def text(label: str, values: dict[str, Any]) -> dict[str, Any]:
        return {"metric": label, **values}

    return [
        both("Team strength rating (0-100)", "strength_rating"),
        both("Offensive efficiency (0-100)", "offensive_efficiency"),
        both("Defensive efficiency (0-100)", "defensive_efficiency"),
        text("Quarterback", {side: _qb_note(teams[side], game, side) for side in ("away", "home")}),
        text(
            "QB passer rating",
            {side: f"{teams[side]['qb_rating']:.0f}" + (" est." if teams[side]["qb_is_estimate"] else "") for side in ("away", "home")},
        ),
        text("QB impact (points vs an average starter)", {side: _pts(teams[side]["qb_impact_points"]) for side in ("away", "home")}),
        both("Pace (seconds per play, lower is faster)", "pace_spp"),
        both("Scoring projection (expected points)", "expected_points"),
        both("Win probability", "win_probability", "pct"),
        both("Spread probability (chance to cover)", "spread_cover_probability", "pct"),
        text(
            "Moneyline probability (market, margin removed)",
            {side: {"value": ml.get(f"market_{side}_prob_novig"), "format": "pct"} for side in ("away", "home")},
        ),
        both("Fair moneyline (model)", "fair_moneyline", "american"),
        text(
            "Games of data (this season / last)",
            {side: f"{teams[side]['games_current']} / {teams[side]['games_prior']}" for side in ("away", "home")},
        ),
    ]


def _render_breakdown(analysis: dict[str, Any], game: dict[str, Any]) -> None:
    section_header(
        "Model breakdown",
        "Strength, efficiency, quarterback, pace, scoring and the win, spread and moneyline probabilities for both "
        "teams side by side. The model's favorite is highlighted in gold.",
    )
    away, home = analysis["away_team"], analysis["home_team"]
    bt.render(
        bt.comparison_table_html(
            _breakdown_rows(analysis, game),
            [
                {"key": "metric", "label": "Measure"},
                {"key": "away", "label": away, "align": "right"},
                {"key": "home", "label": home, "align": "right"},
            ],
            highlight_col=analysis["favored"],
        )
    )
    st.caption(
        f"Strength: 50 is a league-average team, and one rating point is {1 / nma.STRENGTH_POINTS_SCALE:.1f} points of "
        "spread. Efficiency: 0-100, higher is better for offense and for defense. QB impact: points of spread versus "
        f"an average starter, capped at {nma.QB_IMPACT_CLAMP:g} either way. The net quarterback effect in the tile above can "
        "differ from the gap between the two impacts, because part of each starter's value is already built into his "
        "team's strength rating and only the rest is added to the spread. Spread probability is each team's chance to cover "
        f"{_spread_source_text(analysis)}. Moneyline probability is blank when no market moneyline is posted."
    )


def _render_reasoning(analysis: dict[str, Any]) -> None:
    section_header("Why the model leans this way", "How sure the model is, and the numbers behind each factor.")
    factors = analysis["confidence_factors"]
    label = analysis["confidence_label"]
    bt.render(
        bt.meter_html(
            "Model confidence",
            analysis["confidence"],
            display=f"{label} · score {analysis['confidence'] * 100:.0f}/100",  # not a percent: it is not a probability
            tone=_CONFIDENCE_TONE.get(label, "cyan"),
            caption=(
                f"Margin {factors['margin']:.2f} × sample size {factors['sample']:.2f} × quarterback certainty "
                f"{factors['qb']:.2f} × market agreement {factors['market']:.2f}. This rates how sure the model is "
                "that its favorite is the right side. It is not a win probability."
            ),
        )
    )
    favored_name = analysis["teams"][analysis["favored"]]["team_name"]
    bt.render(
        bt.reasoning_panel_html(
            analysis["reasoning"],
            title=f"Why the model favors {favored_name}",
            intro=(
                "Supports means a factor backs the model's pick, Caution means it works against the pick or adds "
                "risk, and Context is shown for background."
            ),
        )
    )


def _render_methodology() -> None:
    k_games, carry, blend = nma.BLEND_K_GAMES, nma.PRIOR_CARRYOVER, nma.EFFICIENCY_WEIGHT
    with st.expander("How this is computed and its limits", key="ml_method"):
        st.markdown(
            f"""
**Team ratings.** Offense, defense, pace and scoring come from nflverse play-by-play and results for this season and
last season, blended by sample size. The live season gets weight n / (n + K) with K = {k_games:g} games, so after
{k_games:g} games it counts as much as last year. Last season is first pulled toward the league average: only
{carry:.0%} of last year's gap from average carries over, because team results repeat only loosely from year to year
(offensive EPA per play correlated about 0.41, 0.46 and 0.41 across the last three year pairs, but defensive EPA per
play only 0.08, -0.11 and 0.37). Re-run on 2023 to 2025, the backtest correlation between the model and the closing
line was 0.909, 0.850 and 0.910, and it stayed flat for carry-over values anywhere from 0.3 to 0.5. A team with no
data at all gets exactly the league average.

**Expected points.** The engine's offense-versus-defense scoring projection is blended
{1 - blend:.0%} / {blend:.0%} with an efficiency estimate built from EPA per play and projected plays. Scoring is noisy
(turnover and special-teams luck, garbage time) while EPA per play is steadier, and the efficiency side is also where
**pace** enters: more projected plays means more scoring chances. Home field adds {nma.HOME_FIELD_POINTS:g} points to the
spread, and nothing at a neutral site. A small rest-day edge ({nma.REST_POINTS_PER_DAY:g} points per extra day, capped at
{nma.REST_DAYS_CAP:g} days) and a wind discount on the total ({nma.WIND_TOTAL_PER_MPH:g} points per mph above
{nma.WIND_THRESHOLD_MPH:g} mph outdoors, capped at {nma.WIND_TOTAL_CAP:g}) complete the picture.

**Quarterbacks.** The named starter's value over an average starter is converted to points
({nma.QB_POINTS_PER_EPA:g} points per EPA per dropback, shrunk toward the average for small samples and capped at
{nma.QB_IMPACT_CLAMP:g} points either way), then reduced by the value already built into the team's rating, so a
regular starter is not counted twice. A backup is priced at the measured backup level, and a listed starter with no NFL
history is priced as an unproven starter. In the 2025 backtest, dropping the quarterback term entirely cost 0.06 of
correlation with the market. When the schedule has not named a starter yet, the page says the starter is assumed.

**Strength rating.** 50 plus {nma.STRENGTH_POINTS_SCALE:g} times the team's margin against an average team at a neutral
site, so one rating point is {1 / nma.STRENGTH_POINTS_SCALE:.1f} points of spread.

**Win probability versus spread probability.** The model's spread becomes a win probability with a normal curve whose
standard deviation is {nma.WIN_PROB_STDEV:g} points. The chance to cover a given line uses {nma.MARGIN_STDEV:g} points,
the spread of the final margin around a line. They differ on purpose. A backtest on 3,649 regular-season games from
2012 to 2025 (closing spread against the actual winner) scored a log-loss of 0.6110 at 13.5 and 0.6085 at 11.1, the
minimum, with 11 to 12 essentially flat. Favorites of 6 to 8.5 points won 74.5% of the time, against 69.4% from the 13.5
curve. A de-vigged moneyline market sits right on the calibrated curve (log-loss 0.6094), so using
{nma.WIN_PROB_STDEV:g} keeps the model-versus-moneyline comparison fair instead of showing every big favorite as
overpriced.

**Price math.** Implied probability is the break-even win rate of your price, sportsbook margin included. Value gap is the
model probability minus that, in probability points. A gap of {oi.STRONG_VALUE_GAP * 100:g} points or more reads as
strong value, {oi.SLIGHT_VALUE_GAP * 100:g} point or more as slight value, anything within {oi.SLIGHT_VALUE_GAP * 100:g}
point either way as a fair price, and a gap below -{oi.SLIGHT_VALUE_GAP * 100:g} point as overpriced. Expected value is the
average profit on your stake if the model probability were the true one. Enter the other team's price to remove the
book's margin and see the market's own probability.

**Limits to keep in mind.**
- Only the quarterback is modelled as a player. Other injuries, depth-chart changes, coaching, travel and motivation are
  not in the numbers.
- EPA ratings are not adjusted for strength of opponent, and early in the season they lean on last year, which misses
  roster moves.
- Pace is the clock time between snaps and ignores game script.
- The margin model is a normal curve with no push probability. Half-point lines are modelled exactly, but whole-number
  spreads like 3 and 7 carry extra push probability in real games, and real margins cluster at 3 and 7.
- The nflverse market columns are a consensus reference, usually the closing line, not your sportsbook's price. They are
  blank for games that have no line yet, which is common for later weeks, and the page then says so.
- A model that disagrees sharply with the market is far more likely to be missing information than to have found an edge,
  so the confidence score is cut when that happens.
- Nothing here places a bet or guarantees a result.
"""
        )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def _render_page() -> None:
    page_header(
        "MoneyLines",
        "Pick an NFL game, see the model's projection for both teams, then paste your sportsbook's moneyline "
        "to see whether the price beats the model's fair price. Nothing here places a bet.",
        eyebrow="Betting · NFL",
    )

    try:
        with st.spinner("Loading the NFL schedule, team ratings and quarterbacks. The first load can take several seconds."):
            inputs = _get_inputs(nma.current_season())
    except nma.ScheduleUnavailable as error:
        _render_load_failure("The NFL schedule is unavailable", str(error))
        return
    except Exception as error:  # noqa: BLE001 - the UI boundary: a data failure must never be a traceback
        _render_load_failure(
            "The model inputs could not be loaded",
            f"Something went wrong while loading the data ({type(error).__name__}). Check your connection and try again.",
        )
        return

    schedule, ratings, qb_context = inputs["schedule"], inputs["ratings"], inputs["qb_context"]
    season = inputs.get("season")
    weeks = nma.weeks_in_schedule(schedule)
    if not weeks:
        _render_load_failure("No NFL games found", "The schedule loaded but it has no regular-season games to show.")
        return

    if st.session_state.get("ml_week") not in weeks:
        st.session_state["ml_week"] = nma.default_week(schedule)
    week_col, summary_col = st.columns([1, 3], vertical_alignment="bottom")
    with week_col:
        week = st.selectbox(
            "NFL week",
            weeks,
            format_func=_week_label,
            key="ml_week",
            on_change=_on_week_change,
            help="Opens on the first week that still has an unplayed game.",
        )
    games = nma.games_for_week(schedule, week)
    with summary_col:
        if games:
            bt.render(_week_summary_html(season, week, games))

    if not games:
        bt.render(bt.empty_card_html(f"No games in {_week_label(week)}", "Pick another week from the list above."))
        return

    scroll_to_analysis = bool(st.session_state.pop("ml_scroll_pending", False))  # consumed now: it applies to this run only
    game = _selected_game(games)
    if st.session_state.get("ml_grid_open", True):
        # With the slate open on a phone the analysis sits many screens down: offer a way straight to it.
        bt.render(
            f'<div class="ml-jump"><a href="#{_ANALYSIS_ANCHOR}">Skip to the analysis for '
            f'{html_lib.escape(game["label"])} &darr;</a></div>'
        )
    _render_game_grid(games, game["game_id"], ratings, qb_context, week)
    game = _selected_game(games)  # a tile click is applied by its callback before this run, but re-read for safety

    _sync_inputs_to_game(game["game_id"])
    user_spread_home = None
    spread_now = _parse_sportsbook_spread(st.session_state.get("ml_spread", ""))
    if spread_now.value is not None:
        user_spread_home = 0.0 - spread_now.value  # sportsbook "home -3.5" is nflverse +3.5 (positive = home favored)

    try:
        analysis = nma.analyze_game(game, ratings, qb_context, user_spread_home=user_spread_home)
    except Exception as error:  # noqa: BLE001
        bt.render(
            bt.empty_card_html(
                "This game could not be analyzed",
                f"The model could not price {game.get('label', 'this game')} ({type(error).__name__}). Pick another game.",
            )
        )
        return

    # scroll-margin-top clears the sticky app header and leaves the (folded) slate bar visible above the matchup card
    bt.render(f'<div id="{_ANALYSIS_ANCHOR}" style="scroll-margin-top:8rem"></div>')
    _render_hero(analysis, game)
    bt.render(bt.stat_grid_html(_headline_tiles(analysis, game), columns=3))
    _render_price_check(analysis)
    _render_breakdown(analysis, game)
    _render_reasoning(analysis)
    _render_methodology()
    if scroll_to_analysis:  # last, so the script's element never shifts anything above it on the next run
        _scroll_to_analysis()


_render_page()
bt.render(bt.disclaimer_html())
