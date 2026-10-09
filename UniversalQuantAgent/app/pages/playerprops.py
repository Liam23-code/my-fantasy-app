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

# NFL Player Props: pick a team, a player and a stat, paste the sportsbook's line
# (and optionally the over / under prices) and see how the model prices it.
#
# This page is deliberately thin. Every number and every sentence about the model
# comes from modules.nfl_prop_analysis (the projection, the over / under / push
# probabilities, edge, EV, confidence, volatility, the reasoning list and the
# engine warnings). The only arithmetic here is drawing the distribution curve.
# Nothing here fetches a sportsbook and nothing places a bet.

import math
import re
from html import escape
from string import Template

import plotly.graph_objects as go
import streamlit as st

from app import betting_ui as bt
from app.page_runtime import apply_global_theme, page_header, section_header
from app.status_ui import render_status_strip
from app.style import (
    CIRCUIT_CYAN,
    MIDNIGHT_NAVY,
    SIGNAL_GOLD,
    SLATE_BLUE,
    SOFT_WHITE,
    apply_gold_glow_theme,
)
from modules import nfl_prop_analysis as npa
from modules.odds_input import (
    SLIGHT_VALUE_GAP,
    STRONG_VALUE_GAP,
    format_american,
    format_money,
    format_pct,
    format_signed_pct,
)

# ---------------------------------------------------------------------------
# Page-local look: the player hero card. Everything else is the shared kit.
# ---------------------------------------------------------------------------
_CSS_TOKENS = {
    "cyan": CIRCUIT_CYAN,
    "gold": SIGNAL_GOLD,
    "navy": MIDNIGHT_NAVY,
    "slate": SLATE_BLUE,
    "text": SOFT_WHITE,
    "deep": bt.DEEP_NAVY,
    "muted": bt.MUTED_TEXT,
    "subtle": bt.SUBTLE_TEXT,
    "dim": bt.DIM_TEXT,
}


def _rgb_triplet(color: str) -> str:
    text = color.lstrip("#")
    return f"{int(text[0:2], 16)},{int(text[2:4], 16)},{int(text[4:6], 16)}"


_CSS_TOKENS.update({f"{name}_rgb": _rgb_triplet(value) for name, value in list(_CSS_TOKENS.items())})

_PAGE_CSS = Template(
    "<style>\n"
    ".pp-hero{--pp-team:$cyan;padding:1.3rem 1.4rem 1.1rem;overflow:hidden;"
    "border-color:color-mix(in srgb,var(--pp-team) 40%,transparent);"
    "background:radial-gradient(120% 160% at 0% 0%,color-mix(in srgb,var(--pp-team) 22%,transparent),transparent 58%),"
    "linear-gradient(145deg,rgba($slate_rgb,.94),rgba($navy_rgb,.97));"
    "box-shadow:0 14px 34px rgba($deep_rgb,.42),0 0 38px color-mix(in srgb,var(--pp-team) 15%,transparent)}\n"
    ".pp-hero-top{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,17rem),1fr));gap:1.1rem 1.5rem;align-items:center}\n"
    ".pp-id{display:flex;align-items:center;gap:1rem;min-width:0}\n"
    ".pp-id-text{min-width:0}\n"
    ".pp-kicker{font-size:.68rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:$muted !important;overflow-wrap:anywhere}\n"
    ".pp-name{margin-top:.12rem;font-size:1.75rem;font-weight:800;letter-spacing:-.03em;line-height:1.12;color:$text !important;overflow-wrap:anywhere}\n"
    ".pp-chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.6rem}\n"
    ".pp-proj{min-width:0;padding:.9rem 1.05rem;border-radius:14px;background:rgba($navy_rgb,.62);"
    "border:1px solid color-mix(in srgb,var(--pp-team) 36%,transparent)}\n"
    ".pp-proj-cap{font-size:.66rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:$muted !important}\n"
    ".pp-proj-val{display:flex;align-items:baseline;flex-wrap:wrap;gap:.2rem .5rem;margin-top:.2rem}\n"
    ".pp-num{font-size:3rem;font-weight:800;letter-spacing:-.045em;line-height:1;color:$text !important;font-variant-numeric:tabular-nums}\n"
    ".pp-unit{font-size:1rem;font-weight:700;color:$subtle !important}\n"
    ".pp-proj-sub{margin-top:.35rem;font-size:.78rem;line-height:1.45;color:$muted !important;overflow-wrap:anywhere}\n"
    ".pp-foot{margin-top:.95rem;padding-top:.7rem;border-top:1px solid rgba($cyan_rgb,.14);font-size:.78rem;line-height:1.55;color:$muted !important;overflow-wrap:anywhere}\n"
    ".pp-foot-k{font-weight:800;letter-spacing:.1em;text-transform:uppercase;font-size:.66rem;color:$subtle !important}\n"
    ".pp-swipe{display:none;margin:-.25rem 0 .6rem;font-size:.74rem;line-height:1.45;color:$muted !important}\n"
    "@media (max-width:900px){.pp-swipe{display:block}}\n"
    "@media (max-width:640px){.pp-hero{padding:1.05rem .95rem .95rem}.pp-name{font-size:1.4rem}.pp-num{font-size:2.5rem}"
    ".bt-table .bt-th,.bt-table .bt-td{padding:.5rem .55rem !important}}\n"
    "</style>"
).substitute(_CSS_TOKENS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_UNAVAILABLE = frozenset({"OUT", "HOLDOUT", "SUSPENDED"})
#: status -> (chip text, chip tone)
_STATUS_CHIP = {
    "HEALTHY": ("No injury flag", "good"),
    "QUESTIONABLE": ("QUESTIONABLE", "warn"),
    "DOUBTFUL": ("DOUBTFUL", "warn"),
    "OUT": ("OUT", "bad"),
    "HOLDOUT": ("HOLDOUT", "bad"),
    "SUSPENDED": ("SUSPENDED", "bad"),
}
#: The "healthy" chip when no availability feed is loaded: nobody was checked, so it must not read as a clean bill.
_NO_FEED_CHIP = ("No status data", "neutral")
_DEFAULT_CATEGORY = {"QB": "Passing Yards", "RB": "Rushing Yards", "WR": "Receiving Yards", "TE": "Receiving Yards"}
_POSITION_ALIASES = {"HB": "RB", "FB": "RB"}
_FEATURED_TEAM = "KC"
_CHART_HEIGHT = 340
#: A best side this many probability points under its break-even earns the red "no edge" banner (5 points is
#: about twice the vig on a standard -110 / -110 market).
_STEEP_VALUE_GAP = 0.05
#: Past this size the model's fair price stops being a number anyone would see on a board (the engine clamps a
#: near-certain side to a probability of 1e-6, which prices out near -99,999,900), so the tile says "beyond".
_FAIR_PRICE_CAP = 10_000
#: No sportsbook posts American odds this long; a pasted price beyond it is a typo, not a market.
_MAX_PRICE = 100_000
#: Engine warnings that question what the user typed (the line or the prices) rather than the player's data. They
#: sit directly under the verdict, because a mistyped line or price would otherwise be read as a "Strong value".
_INPUT_CHECK_MARKERS = (
    "is outside the usual",
    "standard deviations from the projection",
    "is not a half-point step",
    "The prices ",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clear_caches() -> None:
    """Retry / reload: drop the page cache and the engine's in-process pool cache."""
    st.cache_data.clear()
    npa.clear_player_pool_cache()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_pool() -> list[dict]:
    rows = npa.load_player_pool()
    if not rows:
        # Raising keeps an empty (failed) load out of the cache, so a transient outage heals on retry.
        raise RuntimeError("the NFL player pool is empty")
    return rows


def _load_pool() -> list[dict]:
    try:
        with st.spinner("Loading the NFL player pool (the first load can take about 15 seconds)..."):
            return list(_cached_pool())
    except Exception:  # noqa: BLE001 - any loader failure is a calm empty state, never a traceback
        return []


def _feed_loaded() -> bool:
    """Whether the live availability feed holds any record at all (it is empty until Refresh Player Status runs)."""
    try:
        return bool(npa.status_utils.has_status_data())
    except Exception:  # noqa: BLE001 - a status lookup must never take the page down
        return False


def _count(number: float, noun: str) -> str:
    """``1 game`` / ``14 games`` / ``2.5 games``: a count with its noun agreeing (a one-game sample is real)."""
    return f"{number:g} {noun}{'' if number == 1 else 's'}"


def _num(value: object, digits: int = 1) -> str:
    return bt.format_value(value, "dec1" if digits == 1 else "dec2")


def _stat(value: object, unit: str) -> str:
    """Yards to one decimal, counts to two (the engine's own convention); an em dash when missing."""
    return _num(value, 1 if unit == "yds" else 2)


def _quantile(value: object) -> str:
    return bt.format_value(value, "auto")


def _range_text(low: object, high: object, unit: str) -> str:
    """``63 to 449`` for yardage (whole yards: a tenth of a yard means nothing in a range, and it keeps
    the tile on one line at phone width), ``0 to 2`` for counts."""
    if unit == "yds":
        return f"{bt.format_value(low, 'int')} to {bt.format_value(high, 'int')}"
    return f"{_quantile(low)} to {_quantile(high)}"


def _standard_hold() -> str:
    """The hold of a standard -110 / -110 market, from the engine's own odds maths (``4.8%``)."""
    try:
        return format_pct(npa.hold(npa.DEFAULT_PRICE, npa.DEFAULT_PRICE))
    except Exception:  # noqa: BLE001 - a caption must never take the page down
        return "5%"


def _status_share(status: str) -> str:
    """Share of the healthy per-game rate the engine keeps for a status (``88%`` for QUESTIONABLE)."""
    try:
        return f"{npa.status_utils.adjust_projection_for_status(1.0, status):.0%}"
    except Exception:  # noqa: BLE001
        return "a reduced share"


def _qb_curve_is_wide(projection: dict) -> bool:
    """True for a quarterback passing-yards curve drawn wider than starting QBs' real weekly spread."""
    return (
        projection["market"] == "passing_yards"
        and projection["mean"] > 0
        and projection["cv"] > npa.OBSERVED_QB_PASSING_CV[2]
    )


def _position_of(player: dict) -> str:
    raw = str(player.get("position") or "").strip().upper()
    return _POSITION_ALIASES.get(raw, raw)


def _player_key(player: dict) -> str:
    pid = player.get("player_id")
    if pid is not None and str(pid).strip():
        return str(pid)
    return f"{player.get('team')}:{player.get('name')}"


def _option_label(player: dict) -> str:
    label = npa.player_label(player)
    status = npa.player_status(player)
    return label if status == "HEALTHY" else f"{label} · {status.title()}"


def _ensure_choice(key: str, options: list, fallback: object) -> None:
    """Make sure ``st.session_state[key]`` holds one of ``options``, assigning ``fallback`` when it does not.

    Assigning (rather than deleting) a stale value is what makes the browser's select box follow: see the
    note above the team / player / stat selectors. Must run before the widget with this key is created.
    """
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback


#: ``249,5`` / ``1,91``: a decimal comma (one or two digits after it, so never a thousands separator).
_DECIMAL_COMMA = re.compile(r"^(\d{1,3}),(\d{1,2})$")
#: ``O 249.5`` / ``Under 249.5`` / ``o/u 249.5``: the side marker a sportsbook cell prints in front of a line.
_SIDE_MARKER = re.compile(r"^(?:o\s*/\s*u|over|under|o|u)\s*:?\s*(?=\.?\d)", re.IGNORECASE)
#: ``249.5 yds`` / ``5.5 rec``: the unit a sportsbook cell prints after a line.
_UNIT_SUFFIX = re.compile(
    r"\s*(?:yards?|yds?|receptions?|rec|touchdowns?|tds?|points?|pts?)\.?$", re.IGNORECASE
)


def _clean_line_text(text: object) -> str:
    """Forgive what a sportsbook cell looks like when pasted: ``O 249.5``, ``249.5 yds``, ``249,5``.

    The line box keeps what the user typed; only the text handed to the engine's validator is tidied.
    Anything still unreadable afterwards reaches the validator unchanged and gets its friendly error.
    """
    raw = str(text if text is not None else "").strip()
    cleaned = _UNIT_SUFFIX.sub("", _SIDE_MARKER.sub("", raw, count=1), count=1).strip()
    if not cleaned:
        return raw  # nothing but decoration ("yds"): hand it over as typed so the validator can say what is wrong
    return _DECIMAL_COMMA.sub(r"\1.\2", cleaned)


def _clean_price_text(text: object) -> str:
    """Forgive ``(-110)`` and a decimal comma (``1,91`` would otherwise be read as +191 American)."""
    cleaned = str(text if text is not None else "").strip()
    if len(cleaned) > 2 and cleaned[0] == "(" and cleaned[-1] == ")":
        cleaned = cleaned[1:-1].strip()
    return _DECIMAL_COMMA.sub(r"\1.\2", cleaned)


def _check_price_sizes(checked: dict) -> None:
    """Turn a price no sportsbook posts (``+1000000``) into a friendly error instead of pricing it.

    The engine would accept any American price of 100 or more in size, so a typo such as an extra run of
    zeros would reach the table as a 19-digit number. The offending price is cleared (the page then prices at
    -110 for the chart only and withholds the verdict), exactly as it does for any other unusable price.
    """
    for field, name in (("over_price", "Over"), ("under_price", "Under")):
        price = checked.get(field)
        if price is not None and abs(price) > _MAX_PRICE:
            checked["errors"].append(
                f"{name} price: that is beyond any posted American price (about {_MAX_PRICE:,} at most). "
                "Check it for a typo."
            )
            checked[field] = None
            checked["ok"] = False


def _fair_price_text(price: object) -> str:
    """The model's fair American price, or ``beyond +/-10,000`` where it stops being a usable number.

    The engine clamps a near-certain side to a probability of 1e-6, which prices out at -99,999,900. That is a
    bookkeeping limit, not a price anyone would see on a board.
    """
    try:
        value = float(price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    if abs(value) > _FAIR_PRICE_CAP:
        return f"beyond {'-' if value < 0 else '+'}{_FAIR_PRICE_CAP:,}"
    return format_american(value)


def _use_suggested_line() -> None:
    suggestion = st.session_state.get("pp_suggested")
    if suggestion:
        st.session_state["pp_line"] = suggestion


def _finish() -> None:
    bt.render(bt.disclaimer_html())
    st.stop()


# ---------------------------------------------------------------------------
# Hero player card
# ---------------------------------------------------------------------------
def _hero_html(projection: dict) -> str:
    unit = projection["unit"]
    team = projection["team"]
    status = projection["status"]
    games = projection["games"]
    season = projection.get("data_season")
    season_text = str(season) if season else "last season"
    color = bt.team_display_color(team)

    chip_text, chip_tone = _STATUS_CHIP.get(status, _STATUS_CHIP["HEALTHY"])
    if status == "HEALTHY" and not _feed_loaded():
        # A green "No injury flag" would claim a check that never happened: with no feed nobody is flagged.
        chip_text, chip_tone = _NO_FEED_CHIP
    thin = games < npa.THIN_SAMPLE_GAMES
    chips = "".join(
        (
            bt.chip_html(chip_text, chip_tone),
            bt.chip_html(projection["category"], "cyan"),
            bt.chip_html(
                f"{'Thin sample · ' if thin else ''}{_count(games, 'game')} · {season_text}",
                "warn" if thin else "steel",
            ),
        )
    )

    sub = f"{escape(projection['category'])} · one game"
    if status != "HEALTHY":
        sub += (
            f"<br>Raw {escape(season_text)} rate {escape(_stat(projection['per_game_rate'], unit))} {escape(unit)} "
            f"x {projection['status_factor']:.2f} for {escape(status.title())} availability"
        )
    else:
        sub += f" · from {escape(season_text)} actuals"

    foot_parts = []
    basis = str(projection.get("projection_basis") or "").strip()
    if basis:
        foot_parts.append(f'<span class="pp-foot-k">Fantasy projection basis</span> {escape(basis)}.')
    rate_text = (
        "the raw per-game rate from last season"
        if status == "HEALTHY"
        else "last season's per-game rate, scaled for his availability flag"
    )
    foot_parts.append(
        '<span class="pp-foot-k">What this is</span> The model\'s own estimate for a single game: '
        f"{rate_text}, not adjusted for the opponent, game script or weather."
    )
    foot = f'<div class="pp-foot">{"<br>".join(foot_parts)}</div>'

    return (
        f'<div class="bt bt-card pp-hero" style="--pp-team:{color}" role="group" '
        f'aria-label="{escape(projection["name"], quote=True)} projection">'
        '<div class="pp-hero-top">'
        f'<div class="pp-id">{bt.team_logo_html(team, 68)}<div class="pp-id-text">'
        f'<div class="pp-kicker">{escape(npa.team_label(team))} · {escape(projection["position"] or "Player")}</div>'
        f'<div class="pp-name">{escape(projection["name"])}</div>'
        f'<div class="pp-chips">{chips}</div></div></div>'
        '<div class="pp-proj"><div class="pp-proj-cap">Model projection · per game</div>'
        f'<div class="pp-proj-val"><span class="pp-num">{escape(_stat(projection["mean"], unit))}</span>'
        f'<span class="pp-unit">{escape(unit)}</span></div>'
        f'<div class="pp-proj-sub">{sub}</div></div>'
        "</div>"
        f"{foot}</div>"
    )


def _hero_tiles(projection: dict, baseline: dict | None) -> list[dict]:
    unit = projection["unit"]
    family = projection["family"]
    season = projection.get("data_season")
    season_text = str(season) if season else "last season"
    p10, p90 = (baseline["p10"], baseline["p90"]) if baseline else (None, None)
    if family == "gaussian":
        spread_sub = (
            f"One standard deviation (cv {projection['cv']:.2f}). Under the model's normal curve, about two games "
            "in three land within this distance of the projection."
        )
        range_sub = "The middle 80% of games under the model: one game in ten lands below it, one in ten above."
        if _qb_curve_is_wide(projection):
            median = npa.OBSERVED_QB_PASSING_CV[1]
            spread_sub += (
                f" This curve is wider than starting quarterbacks' real weekly spread (cv about {median:.2f}): see the note below."
            )
            range_sub += " For a quarterback's passing yards it reads wider than real games do."
    else:
        # A count stat is a whole number and its spread is fixed by its rate, so the "two in three" and
        # "one in ten" rules of thumb for a normal curve would be wrong here.
        spread_sub = (
            f"Square root of the mean (cv {projection['cv']:.2f}): for a count stat the spread is set by the rate itself."
        )
        range_sub = "At least 80% of games land in this range under the model (counts come in whole numbers)."
    fantasy = projection["projected_fantasy_points"]
    fantasy_value = f"{fantasy:,.1f} pts" if fantasy else "—"
    fantasy_sub = (
        f"{projection['fantasy_points_per_game']:.1f} per game over {_count(projection['expected_games'], 'expected game')}. "
        "A regressed fantasy projection, shown for context only."
        if fantasy
        else "No fantasy projection is available for this player."
    )
    return [
        {
            "label": "Spread (1 st. dev.)",
            "value": f"±{_stat(projection['stdev'], unit)} {unit}",
            "sub": spread_sub,
            "tone": "cyan",
        },
        {
            "label": f"Typical range, p10 to p90 ({unit})",
            "value": f"{_range_text(p10, p90, unit)}" if baseline else "—",
            "sub": range_sub,
            "tone": "cyan",
        },
        {
            "label": "Sample behind the rate",
            "value": _count(projection["games"], "game"),
            "sub": (
                f"{projection['season_total']:,.0f} {unit} in {season_text} = {_stat(projection['per_game_rate'], unit)} per game, "
                f"before any availability adjustment. Confidence {projection['confidence_label'].lower()}."
            ),
            "tone": "warn" if projection["games"] < npa.THIN_SAMPLE_GAMES else "steel",
        },
        {"label": "Fantasy projection", "value": fantasy_value, "sub": fantasy_sub, "tone": "steel"},
    ]


# ---------------------------------------------------------------------------
# Distribution chart (drawing only: every probability printed on it is the engine's)
# ---------------------------------------------------------------------------
def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def _pmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _rgba(color: str, alpha: float) -> str:
    return f"rgba({_rgb_triplet(color)},{alpha:g})"


def _distribution_figure(
    evaluation: dict, *, line: float | None, p_over: float | None, p_under: float | None, favored: str | None
) -> go.Figure | None:
    """The one-game outcome curve. ``line`` is ``None`` when no valid line is entered (no verdict is implied)."""
    family = evaluation["family"]
    mean, stdev = float(evaluation["mean"]), float(evaluation["stdev"])
    unit = evaluation["unit"]
    integer_line = bool(evaluation["integer_line"])
    p10, p90 = float(evaluation["p10"]), float(evaluation["p90"])
    over_color = SIGNAL_GOLD if favored == "over" else CIRCUIT_CYAN
    under_color = SIGNAL_GOLD if favored == "under" else bt.STEEL_BLUE
    neutral = CIRCUIT_CYAN

    fig = go.Figure()
    if family == "poisson":
        lam = max(mean, 0.0)
        top = math.ceil(lam + 5.0 * math.sqrt(lam)) + 2
        if line is not None:
            top = max(top, math.ceil(line) + 2)
        top = max(top, 5)
        ks = list(range(top + 1))
        probs = [_pmf(k, lam) for k in ks]
        colors = []
        for k in ks:
            if line is None:
                colors.append(_rgba(neutral, 0.78 if p10 <= k <= p90 else 0.28))
            elif k > line:
                colors.append(_rgba(over_color, 0.82))
            elif k < line:
                colors.append(_rgba(under_color, 0.82))
            else:
                colors.append(_rgba(SOFT_WHITE, 0.45))
        fig.add_trace(
            go.Bar(
                x=ks,
                y=probs,
                marker={"color": colors, "line": {"width": 0}},
                text=[f"{p:.0%}" if p >= 0.04 else "" for p in probs],
                textposition="outside",
                textfont={"color": bt.SUBTLE_TEXT, "size": 11},
                cliponaxis=False,
                hovertemplate=f"{evaluation['category']} = %{{x}}<br>Chance %{{y:.1%}}<extra></extra>",
                showlegend=False,
            )
        )
        x_range = [-0.6, top + 0.6]
        y_max = max(probs) if probs else 1.0
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", range=[0, y_max * 1.28], showgrid=True)
        fig.update_xaxes(range=x_range, dtick=1 if top <= 16 else None, tick0=0)
        under_edge = over_edge = line
    else:
        if stdev <= 0.0 or mean <= 0.0:
            return None
        low = max(0.0, mean - 4.0 * stdev)
        high = mean + 4.0 * stdev
        if line is not None:
            low = min(low, max(0.0, line - stdev))
            high = max(high, line + stdev)

        def pdf(x: float) -> float:
            return math.exp(-0.5 * ((x - mean) / stdev) ** 2) / (stdev * math.sqrt(2.0 * math.pi))

        # One shared sample grid, dense where the curve lives (mean +- 4 stdev) and coarse in the rest of the
        # range, so a stray line far from the mean (which widens the axis) cannot turn the peak into a polygon.
        grid = sorted(
            set(_linspace(low, high, 240))
            | set(_linspace(max(low, mean - 4.0 * stdev), min(high, mean + 4.0 * stdev), 240))
        )

        def region(a: float, b: float) -> tuple[list[float], list[float]]:
            xs = [a, *(x for x in grid if a < x < b), b]
            return xs, [pdf(x) for x in xs]

        def fill(a: float, b: float, color: str, alpha: float, name: str) -> None:
            if b - a <= 1e-9:
                return
            xs, ys = region(a, b)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", line={"width": 0}, fill="tozeroy",
                    fillcolor=_rgba(color, alpha), hoverinfo="skip", showlegend=False, name=name,
                )
            )

        if line is None:
            fill(low, high, neutral, 0.16, "Outcomes")
            fill(max(low, p10), min(high, p90), neutral, 0.42, "Middle 80%")
            under_edge = over_edge = None
        else:
            under_edge = line - 0.5 if integer_line else line
            over_edge = line + 0.5 if integer_line else line
            fill(low, min(under_edge, high), under_color, 0.6 if favored == "under" else 0.5, "Under")
            if integer_line:
                fill(max(under_edge, low), min(over_edge, high), SOFT_WHITE, 0.35, "Push")
            fill(max(over_edge, low), high, over_color, 0.6 if favored == "over" else 0.5, "Over")
        xs, ys = region(low, high)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", line={"width": 2.2, "color": SOFT_WHITE}, showlegend=False,
                hovertemplate=f"%{{x:,.1f}} {unit}<extra></extra>", name="Model outcome curve",
            )
        )
        y_max = max(ys)
        fig.update_yaxes(visible=False, range=[0, y_max * 1.3])
        fig.update_xaxes(range=[low, high])

    apply_gold_glow_theme(fig, height=_CHART_HEIGHT)
    fig.update_layout(
        margin={"l": 16, "r": 16, "t": 34, "b": 48},
        hovermode="closest" if family == "poisson" else "x",
        bargap=0.18,
        xaxis_title=f"{evaluation['category']} in one game ({unit})",
        title_text="",
        showlegend=False,
    )
    fig.update_xaxes(title_font={"color": bt.MUTED_TEXT, "size": 12})

    shapes = [
        {
            "type": "line", "xref": "x", "yref": "paper", "x0": mean, "x1": mean, "y0": 0, "y1": 1,
            "line": {"color": _rgba(SOFT_WHITE, 0.55), "width": 1.4, "dash": "dot"},
        }
    ]
    # Both marker labels sit in the top margin; whichever marker is further left anchors its text to
    # its left, the other to its right, so the two labels can never run into each other.
    mean_left = line is None or mean <= line
    annotations = [
        {
            "x": mean, "y": 1.0, "xref": "x", "yref": "paper", "text": f"Model mean {_stat(mean, unit)}",
            "showarrow": False, "xanchor": "right" if mean_left and line is not None else "left" if line is not None else "center",
            "yanchor": "bottom", "xshift": -6 if mean_left and line is not None else 6 if line is not None else 0,
            "font": {"size": 11, "color": bt.MUTED_TEXT},
        }
    ]
    if line is not None:
        shapes.append(
            {
                "type": "line", "xref": "x", "yref": "paper", "x0": line, "x1": line, "y0": 0, "y1": 1,
                "line": {"color": SOFT_WHITE, "width": 2, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "x": line, "y": 1.0, "xref": "x", "yref": "paper", "text": f"Book line {line:g}", "showarrow": False,
                "xanchor": "left" if mean_left else "right", "yanchor": "bottom",
                "xshift": 6 if mean_left else -6, "font": {"size": 12, "color": SOFT_WHITE},
            }
        )
        if p_under is not None:
            annotations.append(
                {
                    "x": under_edge, "y": 0.93, "xref": "x", "yref": "paper", "text": f"Under {format_pct(p_under)}",
                    "showarrow": False, "xanchor": "right", "yanchor": "top", "xshift": -8,
                    "font": {"size": 13, "color": under_color if favored == "under" else bt.SUBTLE_TEXT},
                }
            )
        if p_over is not None:
            annotations.append(
                {
                    "x": over_edge, "y": 0.93, "xref": "x", "yref": "paper", "text": f"Over {format_pct(p_over)}",
                    "showarrow": False, "xanchor": "left", "yanchor": "top", "xshift": 8,
                    "font": {"size": 13, "color": over_color if favored == "over" else bt.SUBTLE_TEXT},
                }
            )
    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _verdict_banner(evaluation: dict) -> str:
    side = evaluation["recommended_side"]
    pricing = evaluation[f"pricing_{side}"]
    line, unit, stake = evaluation["line"], evaluation["unit"], evaluation["stake"]
    push = evaluation["push_probability"]
    verdict = evaluation["verdict"]

    if evaluation["status"] in _UNAVAILABLE:
        return bt.verdict_banner_html(
            f"{evaluation['name']} is {evaluation['status']}: these numbers are a what-if",
            "Most sportsbooks refund a prop when the player does not play, so there is no live bet to price. "
            "The model zeroes his stat inputs, which is why the under looks near-certain and the over near-impossible.",
            "warning",
        )

    decided_note = " when the bet is decided" if push > 0 else ""
    numbers = (
        f"The model gives the {side} {format_pct(pricing['model_probability'])}{decided_note} against a "
        f"break-even of {format_pct(pricing['implied_probability'])} at {format_american(pricing['price'])} "
        f"(no-vig fair {format_pct(evaluation[f'market_fair_probability_{side}'])}). "
        f"Expected value {format_money(evaluation['recommended_ev'])} on a ${stake:,.0f} stake."
    )
    if verdict in ("Strong value", "Slight value"):
        return bt.verdict_banner_html(
            f"Lean {side.upper()} {line:g} {unit}", f"{verdict}. {numbers}", evaluation["verdict_tone"]
        )
    if evaluation["has_edge"]:
        return bt.verdict_banner_html(
            f"{side.title()} {line:g} {unit} is priced about fairly", f"{verdict}. {numbers}", "neutral"
        )
    # Every fair coin flip at -110 / -110 sits about 2.4 points under break-even (that is just the vig), so
    # "no edge" is the ordinary outcome and only reads as a warning when the price is clearly steep.
    steep = evaluation["recommended_value_gap"] <= -_STEEP_VALUE_GAP
    return bt.verdict_banner_html(
        "No edge at these prices",
        f"Neither side beats its price; the closer one is the {side}. {numbers}",
        "danger" if steep else "neutral",
    )


def _result_tiles(evaluation: dict, *, lean: bool) -> list[dict]:
    """The six headline tiles. ``lean`` is True only when the page is actually backing a side (gold value tone)."""
    side = evaluation["recommended_side"]
    pricing = evaluation[f"pricing_{side}"]
    unit, stake, line = evaluation["unit"], evaluation["stake"], evaluation["line"]
    ev = evaluation["recommended_ev"]
    gap = evaluation["recommended_value_gap"]
    valuable = lean
    diff = evaluation["line_minus_mean"]
    book_hold = evaluation["hold"]
    tiles = [
        {
            "label": f"Expected value · {side.title()}",
            "value": format_money(ev),
            "sub": f"Average profit on a ${stake:,.0f} stake if the model's probability is right ({pricing['ev_pct']:+.1%} of the stake).",
            "tone": "good" if ev > 0 else "bad",
        },
        {
            "label": "Value gap",
            "value": format_signed_pct(gap),
            "sub": "Model win chance minus the price's break-even, in probability points. Strong value starts at "
            f"{_count(STRONG_VALUE_GAP * 100, 'point')}, slight at {SLIGHT_VALUE_GAP * 100:.0f}.",
            "tone": "gold" if valuable else "neutral",
        },
        {
            "label": "Edge vs no-vig fair",
            "value": format_signed_pct(evaluation["recommended_edge"]),
            "sub": f"Model chance minus the market's de-vigged {format_pct(evaluation[f'market_fair_probability_{side}'])} for the {side}.",
            "tone": "cyan",
        },
        {
            "label": "Line vs projection",
            "value": f"{diff:+,.1f} {unit}" if unit == "yds" else f"{diff:+,.2f} {unit}",
            "sub": (
                f"The {line:g} line sits right on the model's mean."
                if abs(evaluation["z_score"]) < 0.005
                else f"The {line:g} line sits {abs(evaluation['z_score']):.2f} standard deviations "
                f"{'above' if diff > 0 else 'below'} the model's mean."
            ),
            "tone": "steel",
        },
        {
            "label": "Book hold",
            "value": format_pct(book_hold),
            "sub": f"The margin built into the two prices. A standard -110 / -110 market holds about {_standard_hold()}.",
            "tone": "steel",
        },
    ]
    if evaluation["integer_line"]:
        tiles.append(
            {
                "label": "Push probability",
                "value": format_pct(evaluation["push_probability"]),
                "sub": f"The chance the stat lands exactly on {line:g}. A push refunds the stake.",
                "tone": "steel",
            }
        )
    else:
        tiles.append(
            {
                "label": f"Model fair price · {side.title()}",
                "value": _fair_price_text(pricing["fair_price"]),
                "sub": "The American price at which this side would break even if the model is right.",
                "tone": "steel",
            }
        )
    return tiles


def _side_table(evaluation: dict, *, lean: str | None) -> str:
    """Over / under rows. ``lean`` is the side the page backs ("over" / "under"), or ``None`` for no gold row."""
    line, stake = evaluation["line"], evaluation["stake"]
    rows = []
    for side in ("over", "under"):
        pricing = evaluation[f"pricing_{side}"]
        gap = pricing["value_gap"]
        ev = pricing["expected_value"]
        rows.append(
            {
                "side": {
                    "value": f"{side.title()} {line:g}" + (" (lean)" if side == lean else ""),
                    "tone": "gold" if side == lean else "",
                },
                "price": {"value": pricing["price"], "format": "american"},
                "model": {"value": pricing["model_probability"], "format": "pct"},
                "breakeven": {"value": pricing["implied_probability"], "format": "pct"},
                "gap": {
                    "value": gap,
                    "format": "signed_pct",
                    "tone": "good" if gap >= SLIGHT_VALUE_GAP else "bad" if gap <= -SLIGHT_VALUE_GAP else "",
                },
                "ev": {"value": ev, "format": "money", "tone": "good" if ev > 0 else "bad" if ev < 0 else ""},
                "read": pricing["verdict"],
            }
        )
    columns = [
        {"key": "side", "label": "Side"},
        {"key": "price", "label": "Price"},
        {"key": "model", "label": "Model win chance" + (" (if decided)" if evaluation["integer_line"] else "")},
        {"key": "breakeven", "label": "Break-even"},
        {"key": "gap", "label": "Value gap"},
        {"key": "ev", "label": f"EV on {stake:,.0f}"},
        {"key": "read", "label": "Read"},
    ]
    return bt.comparison_table_html(rows, columns)


_QUALITY_TONE = {"High": "good", "Moderate": "cyan", "Low": "warn"}
_VOLATILITY_TONE = {"High": "warn", "Moderate": "cyan", "Low": "good"}


def _render_quality(evaluation: dict, projection: dict) -> None:
    confidence, volatility = evaluation["confidence"], evaluation["volatility"]
    conf_label, vol_label = evaluation["confidence_label"], evaluation["volatility_label"]
    cv = evaluation["cv"]
    season = projection.get("data_season")
    left, middle, right = st.columns(3)
    with left:
        bt.render(
            bt.meter_html(
                "Confidence",
                confidence,
                display=f"{conf_label} · {confidence:.0%}",
                tone=_QUALITY_TONE.get(conf_label, "cyan"),
                caption="How much evidence backs the projection: sample size and the model ensemble's own confidence. "
                "It is not the chance this bet wins.",
            )
        )
    with middle:
        count_note = (
            " For a count stat the spread is set by the mean, so this reading is informational."
            if evaluation["family"] == "poisson"
            else ""
        )
        bt.render(
            bt.meter_html(
                "Volatility",
                volatility,
                display=f"{vol_label} · cv {cv:.2f}",
                tone=_VOLATILITY_TONE.get(vol_label, "cyan"),
                caption="How widely single games swing around the average (cv = standard deviation divided by the mean). "
                f"The wider the swing, the easier a small edge is swamped.{count_note}",
            )
        )
    with right:
        bt.render(
            bt.risk_meter_html(
                evaluation["risk_tier"],
                factors=[
                    f"Coefficient of variation {cv:.2f}",
                    f"{_count(projection['games'], 'game')} behind the rate ({season or 'last season'})",
                ],
            )
        )


def _is_input_check(warning: str) -> bool:
    """True for an engine warning that questions the typed line or prices (see ``_INPUT_CHECK_MARKERS``)."""
    return any(marker in warning for marker in _INPUT_CHECK_MARKERS)


def _heads_up_notices(evaluation: dict, projection: dict, *, input_checks: bool = False) -> list[str]:
    """Engine warnings as notice cards: the ones about the typed line / prices (``input_checks=True``) or the rest."""
    notices = []
    for warning in evaluation["warnings"]:
        if warning == projection["status_note"] and projection["status"] != "HEALTHY":
            continue  # already shown, with its own tone, directly under the player card
        if _is_input_check(warning) != input_checks:
            continue
        tone = "info" if warning.startswith("Integer line") else "warn"
        notices.append(bt.notice_html(warning, tone))
    return [html for html in notices if html]


def _qb_width_notice(projection: dict) -> str:
    """The quarterback passing-yards width caveat, shown next to the spread it qualifies (not just with a verdict)."""
    if not _qb_curve_is_wide(projection):
        return ""
    low, median, high = npa.OBSERVED_QB_PASSING_CV
    return bt.notice_html(
        f"Passing-yards spread: this curve uses a coefficient of variation of {projection['cv']:.2f}, wider than "
        f"the roughly {median:.2f} (middle half {low:.2f} to {high:.2f}) that starting quarterbacks showed week to week in 2025. "
        "A curve that wide pulls over / under probabilities toward 50% for lines away from the mean and shrinks "
        "apparent edges there. That only holds if the projected mean is right: it is last season's raw average, "
        "so a line far from it may just mean the projection is out of date.",
        "warn",
    )


def _render_methodology() -> None:
    cv_low, cv_high = getattr(npa, "CV_LOW", None), getattr(npa, "CV_HIGH", None)
    band = f"{cv_low:.2f} to {cv_high:.2f}" if cv_low is not None and cv_high is not None else "a fixed range"
    cv_top = f"{cv_high:.2f}" if cv_high is not None else "its maximum"
    q_low, q_med, q_high = npa.OBSERVED_QB_PASSING_CV
    bullets = [
        (
            "**Where the projection comes from.** The per-game mean is the player's raw total for the stat in the last "
            "completed season divided by the games he played. It is a plain historical rate, not a forecast of "
            "this week: there is **no opponent, game-script, weather, role-change or in-season adjustment**, and it "
            "does not use any games from the current season. The fantasy projection on the card is a separate, "
            "regressed and market-reconciled number shown for context; the stat mean does not use that regression."
        ),
        (
            "**Yardage uses a normal (Gaussian) curve.** The standard deviation is the coefficient of variation times the "
            f"mean, with the coefficient scaled across {band} by the engine's 0 to 1 volatility read (derived from the "
            "fantasy floor / ceiling band). That spread is a modelling choice, not measured weekly variance, and a "
            "normal curve leaves a little probability below zero yards. A player with almost no history can read "
            '"low" volatility simply for lack of data, so the page flags thin samples '
            f"(under {npa.THIN_SAMPLE_GAMES:g} games)."
        ),
        (
            "**Quarterback passing yards are the known weak spot.** Most quarterbacks saturate the engine's volatility "
            f"read, giving a coefficient of about {cv_top}, while starting quarterbacks' weekly passing yards in 2025 "
            f"varied with a median of about {q_med:.2f} (middle half {q_low:.2f} to {q_high:.2f}). The curve is therefore "
            "wider than history: lines away from the mean land closer to 50% than they should and apparent edges are "
            "understated (if the projected mean is right, which for a line far from it is the open question). "
            "Skill-position yardage runs the other way: real weekly spread is at or above the top of "
            "the engine's range, so tails are likelier than drawn."
        ),
        (
            "**Counts use a Poisson model** (receptions, touchdown counts): the variance equals the mean, so the spread "
            "is fixed by the rate. Real weekly counts run slightly over-dispersed (variance about 1.0 to 1.2 times the "
            "mean), so extreme games are a bit likelier than shown. **Anytime TD** is a Poisson count with the rushing "
            "plus receiving touchdown rate; passing touchdowns do not count, and the line is always 0.5."
        ),
        (
            "**Odds math.** Break-even is the win rate a price needs, vig included. The no-vig fair probability removes "
            "the book's margin from the two prices. The value gap is the model's win chance minus break-even "
            f"(slight value starts at {_count(SLIGHT_VALUE_GAP * 100, 'point')}, strong at {_count(STRONG_VALUE_GAP * 100, 'point')}). Expected "
            "value is the average profit on your stake if the model's probability were the true probability. "
            "On a whole-number line the stat can push; a push refunds the stake, so the page prices over, under and "
            "push together and compares the market with the win chance given the bet is decided."
        ),
        (
            "**Confidence and volatility.** Confidence reflects how much evidence is behind the projection (sample size and "
            "the ensemble's own confidence); it is not the probability that the pick wins. Volatility shows how widely "
            "single games swing."
        ),
        (
            f"**Availability.** The live injury feed trims the per-game rate for QUESTIONABLE (to about {_status_share('QUESTIONABLE')} "
            f"of normal) and DOUBTFUL (about {_status_share('DOUBTFUL')}) players and zeroes it for OUT, HOLDOUT and SUSPENDED "
            "players, who stay selectable so you can see why they price the way they do (their numbers are a what-if, "
            "never a pick). The model trims a rate; it does not model a game a player misses, "
            "and most books refund a prop if the player does not play. Check the final injury report before relying on "
            "any number. With no feed loaded, everyone is treated as healthy."
        ),
        (
            "**What this is not.** It is research, not advice: a model probability above break-even is not a prediction "
            "that the bet wins, small samples are noisy, and nothing on this page places a bet."
        ),
    ]
    with st.expander("How this is computed and its limits"):
        st.markdown("\n".join(f"- {bullet}" for bullet in bullets))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
apply_global_theme()
bt.inject_betting_css()
st.markdown(_PAGE_CSS, unsafe_allow_html=True)

page_header(
    "Player Props",
    "Choose a team, a player and a stat. The model projects one game; paste the sportsbook's line and prices to see "
    "the chance of the over and the under, the expected value, and why.",
    eyebrow="Betting · NFL",
)

# Same live-availability strip as the NFL Betting page: OUT / doubtful / questionable flags change a prop's price,
# so the one button that refreshes them belongs where the prop is priced.
render_status_strip("pp_refresh_player_status", hint="props are priced from last season's per-game rates. ")

pool = _load_pool()
if not pool:
    bt.render(
        bt.empty_card_html(
            "Player data is unavailable",
            "The NFL player pool could not be loaded. You may be offline or the data provider may be down. "
            "Check your connection and retry.",
        )
    )
    st.button("Retry", key="pp_retry", on_click=_clear_caches, help="Clear the cached data and load it again.")
    _finish()

teams = npa.teams_in_pool(pool)
if not teams:
    bt.render(
        bt.empty_card_html(
            "No props-eligible players found",
            "The player pool loaded but has no quarterbacks, running backs, receivers or tight ends with a stat line to price.",
        )
    )
    st.button("Retry", key="pp_retry_empty", on_click=_clear_caches, help="Clear the cached data and load it again.")
    _finish()

# --- 1. Team -> player -> stat category ---------------------------------------
section_header("Choose a player", "The player list follows the team, and the stat list follows the player.")

# Selection state is always ASSIGNED, never deleted. The browser keeps a select box's displayed value
# for as long as its key survives a rerun, so deleting a stale key (the old approach) left the Player and
# Stat category boxes showing the previous team's player while everything below had moved on. Setting
# the value in session state before the widget is created tells the browser to adopt it. For the same
# reason none of these three boxes passes ``index=``.
_ensure_choice("pp_team", teams, _FEATURED_TEAM if _FEATURED_TEAM in teams else teams[0])
team_col, player_col, category_col = st.columns([1.1, 1.4, 1.1])
team = team_col.selectbox("Team", teams, format_func=npa.team_label, key="pp_team")

team_players = npa.players_for_team(pool, team)
players_by_key = {_player_key(player): player for player in team_players}
if not players_by_key:
    bt.render(
        bt.empty_card_html(
            "No props-eligible players on this team",
            "None of this team's players has a stat line the model can price. Pick another team.",
        )
    )
    _finish()

player_keys = list(players_by_key)
_ensure_choice("pp_player", player_keys, player_keys[0])  # a team change lands on that team's first player
player_key = player_col.selectbox(
    "Player",
    player_keys,
    format_func=lambda key: _option_label(players_by_key[key]),
    key="pp_player",
    help="Ordered by fantasy projection. Players the live feed lists as OUT, DOUBTFUL or QUESTIONABLE stay selectable and are tagged.",
)
player = players_by_key[player_key]

eligible = npa.eligible_categories(player)
if not eligible:
    bt.render(
        bt.empty_card_html(
            "No stat market for this player",
            "The model only prices a stat when a player has real volume in it. Pick another player.",
        )
    )
    _finish()

# A different player always starts on his own default market (a QB on passing yards, a back on rushing
# yards, a receiver on receiving yards), even if the previous market is also open to him.
wanted_category = _DEFAULT_CATEGORY.get(_position_of(player))
default_category = wanted_category if wanted_category in eligible else eligible[0]
if st.session_state.get("pp_category_for") != player_key or st.session_state.get("pp_category") not in eligible:
    st.session_state["pp_category"] = default_category
    st.session_state["pp_category_for"] = player_key
category = category_col.selectbox(
    "Stat category",
    eligible,
    key="pp_category",
    help="Only the markets this player realistically has are offered.",
)
st.caption(npa.STAT_CATEGORIES[category]["blurb"])
if not _feed_loaded():
    st.caption(
        "No live availability data is loaded (or nobody is currently flagged), so every player is treated as "
        "healthy. Use Refresh Player Status at the top of this page to pull OUT, DOUBTFUL and "
        "QUESTIONABLE flags."
    )

# --- 2. The model's projection --------------------------------------------------
try:
    projection = npa.project_stat(player, category)
except ValueError as error:
    bt.render(bt.empty_card_html("No projection for this player", str(error)))
    _finish()

suggested = npa.suggest_line(projection)
suggested_text = f"{suggested:g}"
st.session_state["pp_suggested"] = suggested_text

# A new player or stat means the old line no longer applies: pre-fill the suggestion (the user can still edit it).
# The prices go with it: a price quoted for one player's market must not silently carry over to the next one
# (the stake is the user's own preference and stays).
# The box is also refilled when its value is simply gone: Streamlit drops a widget's state when the page is
# left (or when an earlier run stopped before reaching it), while the context key below survives, so
# coming back to the same player and stat would otherwise show an empty line.
_line_context = f"{player_key}|{category}"
if st.session_state.get("pp_line_ctx") != _line_context or "pp_line" not in st.session_state:
    st.session_state["pp_line_ctx"] = _line_context
    st.session_state["pp_line"] = suggested_text
    st.session_state["pp_over_price"] = ""
    st.session_state["pp_under_price"] = ""

# The spread and the p10-p90 range do not depend on the line, so the player card reads them from an
# evaluation at the suggested line. It is only a source of those two numbers: no verdict is drawn from it.
try:
    baseline = npa.evaluate_line(player, category, suggested, stake=100.0)
except Exception:  # noqa: BLE001 - the card then simply shows an em dash for the range
    baseline = None

section_header("The model's projection", "One game, from last season's per-game rate. This is the model's estimate, not the book's.")
bt.render(_hero_html(projection))
if projection["status"] != "HEALTHY":
    unavailable = projection["status"] in _UNAVAILABLE
    bt.render(
        bt.notice_html(
            f"Availability: {projection['status_note']}"
            + (
                ""
                if unavailable
                else " The model trims his per-game rate rather than modelling a game he misses; check the final injury report."
            ),
            "bad" if unavailable else "warn",
        )
    )
bt.render(bt.stat_grid_html(_hero_tiles(projection, baseline), columns=4))
bt.render(_qb_width_notice(projection))

# --- 3. The sportsbook line, prices and stake -----------------------------------
section_header("Price the line", "Paste the number from your sportsbook. Prices and stake are optional.")
line_col, over_col, under_col, stake_col = st.columns([1.2, 1, 1, 1])
line_text = line_col.text_input(
    f"Sportsbook line ({projection['unit']})",
    key="pp_line",
    placeholder=suggested_text,
    help="Paste the book's number, for example 249.5. A half-point (x.5) line cannot push; a whole number can.",
)
over_text = over_col.text_input(
    "Over price (optional)",
    key="pp_over_price",
    placeholder="-110",
    help="American odds such as -110 or +105 (decimal odds like 1.91 also work). Leave blank to assume -110.",
)
under_text = under_col.text_input(
    "Under price (optional)",
    key="pp_under_price",
    placeholder="-110",
    help="American odds such as -110 or +105 (decimal odds like 1.91 also work). Leave blank to assume -110.",
)
stake_value = stake_col.number_input(
    "Stake ($)",
    min_value=1,
    max_value=1_000_000,
    value=100,
    step=10,
    key="pp_stake",
    help="The dollar amount that expected value is reported on.",
)
stake = float(stake_value) if stake_value else 100.0
st.button(
    f"Use the model's suggested line ({suggested_text})",
    key="pp_use_suggested",
    on_click=_use_suggested_line,
    help="The half-point line nearest the projection, which is where a sportsbook would typically post it.",
)

# What a sportsbook cell looks like when pasted ("O 249.5", "249.5 yds", "249,5", "1,91") is tidied first; the
# boxes themselves keep exactly what was typed.
checked = npa.validate_inputs(
    category, _clean_line_text(line_text), _clean_price_text(over_text), _clean_price_text(under_text)
)
_check_price_sizes(checked)
line_valid = checked["line"] is not None

# Input feedback: friendly, never a traceback. A blank price is silent (it means -110).
for message in checked["errors"]:
    bt.render(bt.notice_html(message, "bad"))
for note in checked["notes"]:
    st.caption(note)
if checked["ok"] and len(checked["defaults_used"]) == 2:
    st.caption(f"No prices entered, so both sides are priced at {format_american(npa.DEFAULT_PRICE)}.")

evaluation = None
pricing_failed = False
if line_valid:
    # A usable line with an unusable price is still evaluated at -110 so the chart can mark the line,
    # but no verdict is shown until the prices are fixed.
    eval_over = checked["over_price"] if checked["over_price"] is not None else npa.DEFAULT_PRICE
    eval_under = checked["under_price"] if checked["under_price"] is not None else npa.DEFAULT_PRICE
    try:
        evaluation = npa.evaluate_line(
            player, category, checked["line"], eval_over, eval_under, stake=stake, pool=pool
        )
    except ValueError as error:
        pricing_failed = True
        bt.render(bt.notice_html(str(error), "bad"))
    except Exception as error:  # noqa: BLE001 - the page must degrade to a message, not a traceback
        pricing_failed = True
        bt.render(bt.notice_html(f"The engine could not price this selection ({error}).", "bad"))
results_ready = bool(checked["ok"]) and evaluation is not None

# --- 4. Results -----------------------------------------------------------------
favored = None
if results_ready:
    side = evaluation["recommended_side"]
    # Gold means "the model sees value here". An OUT / HOLDOUT / SUSPENDED player is a what-if (the book
    # refunds the prop), so nothing on the page is allowed to look like a recommendation for him.
    favored = (
        side
        if evaluation["verdict"] in ("Strong value", "Slight value")
        and evaluation["has_edge"]
        and evaluation["status"] not in _UNAVAILABLE
        else None
    )
    section_header("Your read", f"{projection['name']} · {projection['category']} · line {evaluation['line']:g}")
    bt.render(_verdict_banner(evaluation))
    # A line far from the projection, an odd hold or a line outside the usual range is as likely a typo (or the
    # wrong player's number) as a mispricing, so those checks sit right under the verdict, not at the bottom.
    for notice in _heads_up_notices(evaluation, projection, input_checks=True):
        bt.render(notice)

    push = evaluation["push_probability"]
    bt.render(
        bt.probability_bar_html(
            f"Over {evaluation['line']:g}",
            evaluation["probability_over"],
            f"Under {evaluation['line']:g}",
            evaluation["probability_under"],
            left_color=SIGNAL_GOLD if favored == "over" else CIRCUIT_CYAN,
            right_color=SIGNAL_GOLD if favored == "under" else bt.STEEL_BLUE,
            caption=(
                f"Push {format_pct(push)}: the stake is refunded if the stat lands exactly on {evaluation['line']:g}. "
                "Over, under and push add up to 100%; the bar compares over with under."
                if evaluation["integer_line"]
                else "The model's chance that each side wins. A half-point line cannot push, so the two add up to 100%."
            ),
        )
    )
    bt.render(bt.stat_grid_html(_result_tiles(evaluation, lean=favored is not None), columns=3))
    bt.render(_side_table(evaluation, lean=favored))
    # Phones show about three of the table's seven columns; say so (CSS hides this line on wide screens).
    bt.render(
        '<div class="pp-swipe">Swipe the table sideways for the break-even, value gap, expected value and the '
        "read on each side.</div>"
    )

    section_header("Quality of the read", "How much to trust the number, and how widely this stat swings.")
    _render_quality(evaluation, projection)
elif not checked["errors"] and not pricing_failed:
    bt.render(
        bt.empty_card_html(
            "Paste a line to see the verdict",
            f"The model projects {_stat(projection['mean'], projection['unit'])} {projection['unit']} for one game. "
            f"Enter the sportsbook's line above (the model suggests {suggested_text}) to price the over and the under.",
        )
    )

# --- 5. Distribution --------------------------------------------------------------
chart_source = evaluation if line_valid and evaluation is not None else baseline
if chart_source is not None:
    marked = line_valid and evaluation is not None
    section_header(
        "The distribution of outcomes",
        "How the model spreads one game's result. The dashed line is the sportsbook's number; the shaded regions are the over and the under."
        if marked
        else "How the model spreads one game's result. Enter a line above to split it into over and under.",
    )
    figure = _distribution_figure(
        chart_source,
        line=chart_source["line"] if marked else None,
        p_over=chart_source["probability_over"] if marked else None,
        p_under=chart_source["probability_under"] if marked else None,
        favored=favored,
    )
    if figure is None:
        bt.render(
            bt.empty_card_html(
                "No curve to draw",
                "The model projects no production for this stat, so there is no spread of outcomes to show.",
            )
        )
    else:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False}, key="pp_distribution")
        if not marked:
            share = "the middle 80%" if chart_source["family"] == "gaussian" else "at least 80%"
            st.caption(
                f"The brighter band is where {share} of games land under the model "
                f"({_range_text(chart_source['p10'], chart_source['p90'], projection['unit'])} {projection['unit']})."
            )

# --- 6. Reasoning + engine warnings ---------------------------------------------
if results_ready:
    section_header("Why the model prices it this way", "Every factor below carries the numbers behind it.")
    for notice in _heads_up_notices(evaluation, projection):
        bt.render(notice)
    bt.render(
        bt.reasoning_panel_html(
            evaluation["reasoning"],
            title="What drives this read",
            intro="Supports means the factor backs the model's lean (or your trust in it); Caution argues against it; Context just explains.",
        )
    )

# --- 7. Method and limits -----------------------------------------------------------
_render_methodology()

footer_left, footer_right = st.columns([3, 1])
footer_left.caption(
    "Rates come from the last completed season's player totals and are cached for an hour. "
    "Availability comes from the live player-status feed when it is loaded."
)
footer_right.button("Reload data", key="pp_reload", on_click=_clear_caches, help="Clear the cached data and load it again.")
bt.render(bt.disclaimer_html())
