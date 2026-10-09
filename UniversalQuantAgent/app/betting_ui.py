"""Midnight Circuit component kit shared by the MoneyLines, Player Props and Parlays pages.

Every public ``*_html`` function here is a **pure HTML-string builder**: no
Streamlit session, no network, no engine call. A page computes its numbers
with the betting engine, hands plain dicts/floats to these builders, and gets
back one compact line of markup to pass to :func:`render`. Keeping the
builders pure is what lets ``tests/test_betting_ui.py`` pin their behaviour
offline.

Design rules (and why)
----------------------
* **Native look.** Dark navy glass cards with a cyan hairline border and a
  soft cyan glow, soft-white text, muted ``#9FB0C5`` secondary text. Gold is
  reserved for the single "this is the highlight" signal (the model's favored
  side, the highlighted comparison column). Green / red / amber exist only as
  *semantic* tints (positive EV, negative EV, caution) and are declared once
  as module constants so no component invents its own.
* **Streamlit-markdown safe.** ``st.markdown`` turns a blank line, or a line
  indented four spaces, into a Markdown code block. Every builder therefore
  returns a single line with no embedded newlines (user text is
  whitespace-collapsed first) and starts with a block element (``<div>``), so
  Streamlit parses it as one raw HTML block instead of a paragraph -- inside a
  paragraph, ``$4.20 ... $5`` would be run through the LaTeX renderer.
* **Escaped and finite.** Every dynamic string goes through :func:`html.escape` (imported as ``_escape``);
  every number goes through a helper that turns ``None``/NaN/inf into an em
  dash, so a missing model input can never print ``nan`` on a card.
* **Specificity.** ``app.app.inject_app_css`` first paints a *light* canvas
  with rules such as ``[data-testid="stMain"] p {color: ink}`` and
  ``style.inject_style`` then repaints dark. The kit avoids ``p``/``strong``/
  ``em``/``li``/headings entirely (those are what the light rules target),
  sets its own text colors with ``!important`` on namespaced ``bt-`` classes,
  and passes dynamic colors through CSS custom properties (``--bt-c``) rather
  than inline ``color:`` so an inline value can never fight a stylesheet
  ``!important``.
* **Motion is restrained** (a hover lift, a gauge fill, a pulsing "live" dot)
  and switched off under ``prefers-reduced-motion``.

Team logos
----------
ESPN serves ``https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png``. nflverse
codes map to the ESPN slug by lower-casing, except ``LA`` -> ``lar`` and
``WAS`` -> ``wsh``. Streamlit 1.60 *strips* ``onerror``/``onload`` attributes
from ``st.markdown`` HTML (verified against the rendered DOM), so an
``onerror`` fallback alone would leave a broken-image icon in the real app.
The fallback is therefore CSS: a broken ``<img>`` renders its ``::after``
pseudo-element in Chromium and Firefox, and that pseudo-element paints a team
colored monogram badge over the broken image. An always-present monogram
underlay (revealed by ``onerror`` wherever handlers do run) is covered by the
logo's opaque chip background, so a transparent logo never shows letters
through it. Safari does not paint ``::after`` on broken images, so there the
failure case degrades to an empty team-colored chip -- still no broken icon.

Layout
------
CSS media queries (820px / 360px) see the *browser window*, not the column a
component sits in, and Streamlit's main area is often far narrower than the
window (an open sidebar, ``st.columns``). So:

* :func:`stat_grid_html` sizes its columns from its own container
  (``auto-fill`` with a minimum tile width) and needs nothing from the page.
* :func:`matchup_card_html` has a three-column default that needs about 420px;
  pass ``compact=True`` for a card inside ``st.columns`` / a narrow container.
* No CSS container query is used. ``container-type: inline-size`` on a card
  collapsed it to its 44px of padding inside ``st.container(horizontal=True)``
  (Streamlit 1.60 sizes those children to fit their content), so a regression
  test forbids it.

Teams
-----
Anywhere a team is expected the kit accepts an nflverse code, a historical or
vendor alias (``KAN``, ``WSH``, ``OAK``), a full name or a nickname, in any
case (the same spellings as ``modules.nfl.lookup_team``). Anything else that is
a 2-4 letter token is shown as a monogram with no logo request.

Limits
------
These components display numbers; they do not validate betting logic. They
cannot tell whether a probability is well calibrated or a line is still
available. Colors such as green/amber/red describe *relative* model output,
never a recommendation to wager. A heat-grid cell's text is whichever of soft
white / navy reads better on its fill; on mid-strength fills neither can beat
about 3.8:1, so those few cells fall short of WCAG AA for small text.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from html import escape as _escape
from typing import Any

import streamlit as st

from app.style import (
    CIRCUIT_CYAN,
    MIDNIGHT_NAVY,
    SIGNAL_GOLD,
    SLATE_BLUE,
    SOFT_WHITE,
)
from modules.odds_input import format_american, format_money, format_pct, format_signed_pct

try:  # pragma: no cover - exercised implicitly; the fallback only matters if a provider import breaks
    from modules.nfl_stats import NFL_TEAM_COLORS as _NFL_TEAM_COLORS
    from modules.nfl_stats import team_palette as _team_palette
except Exception:  # noqa: BLE001 - a UI kit must import even if the stats provider cannot
    _NFL_TEAM_COLORS = {}

    def _team_palette(team: str) -> tuple[str, str]:
        return SLATE_BLUE, CIRCUIT_CYAN

try:  # pragma: no cover
    from modules.nfl import NFL_TEAMS as _NFL_TEAMS
except Exception:  # noqa: BLE001
    _NFL_TEAMS = {}

# ---------------------------------------------------------------------------
# Color constants. The five brand colors come from app.style so this kit can
# never drift from the theme; the rest are declared once, here.
# ---------------------------------------------------------------------------
#: Darker end of the app canvas gradient (``#071321`` in ``style.GLOBAL_CSS``).
DEEP_NAVY = "#071321"
#: Secondary / tertiary / quiet text, the exact values ``style.GLOBAL_CSS`` uses.
MUTED_TEXT = "#9FB0C5"
SUBTLE_TEXT = "#AFC0D3"
DIM_TEXT = "#8396AE"
#: Cool mid-blue already in ``style.GOLD_GLOW_CHART_THEME['colorway']``; the
#: second series color (e.g. "Under", "Spread") so gold stays reserved.
STEEL_BLUE = "#7891B4"
#: Semantic tints that sit well on navy. Positive EV / negative EV / caution.
GOOD_GREEN = "#3DDC97"
BAD_RED = "#FF6B6B"
WARN_AMBER = "#FF9F5A"

#: Every non-brand hex literal the kit may emit. ``BETTING_CSS`` is tested to
#: contain no hex outside ``PALETTE_HEX | set(BETTING_COLOR_CONSTANTS.values())``.
BETTING_COLOR_CONSTANTS: dict[str, str] = {
    "DEEP_NAVY": DEEP_NAVY,
    "MUTED_TEXT": MUTED_TEXT,
    "SUBTLE_TEXT": SUBTLE_TEXT,
    "DIM_TEXT": DIM_TEXT,
    "STEEL_BLUE": STEEL_BLUE,
    "GOOD_GREEN": GOOD_GREEN,
    "BAD_RED": BAD_RED,
    "WARN_AMBER": WARN_AMBER,
}
PALETTE_HEX: frozenset[str] = frozenset(
    value.upper() for value in (MIDNIGHT_NAVY, CIRCUIT_CYAN, SIGNAL_GOLD, SLATE_BLUE, SOFT_WHITE)
)

TONES = ("cyan", "gold", "good", "bad", "warn", "steel", "neutral")
VERDICT_TONES = ("success", "info", "neutral", "danger", "warning")

_EM_DASH = "—"
_MISSING_TEXT = frozenset(
    {"nan", "none", "null", "nat", "<na>", "inf", "+inf", "-inf", "infinity", "-infinity", "undefined"}
)
_HEX_RE = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")


# ---------------------------------------------------------------------------
# Value hygiene: escaping, finite numbers, safe attributes
# ---------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    """True for ``None``, NaN/inf, pandas NA/NaT, and the strings ``"nan"``/``"None"``/``"inf"``."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TEXT
    if type(value).__name__ in ("NAType", "NaTType"):
        return True
    if hasattr(value, "__float__") and not isinstance(value, bytes):
        try:
            return not math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


#: A non-finite number that a caller's own f-string printed into text:
#: ``"O/U nan"``, ``"DAL +nan"``, ``"-inf"``, ``"nan%"``, ``"<NA>"``. Only the
#: exact tokens Python / numpy / pandas print are matched (``nan`` / ``NaN`` and
#: ``inf``, ``NaT``, ``<NA>``) and only as whole words, so "finance", "info" and
#: a capitalised "None" in a sentence are never touched.
_NONFINITE_TOKEN = re.compile(r"(?<![\w.])[+-]?(?:nan|NaN|inf|NaT|<NA>)%?(?!\w)")


def _clean(value: Any) -> str:
    """Whitespace-collapsed ``str(value)`` with printed NaN/inf tokens replaced by an em dash.

    Collapsing whitespace is load-bearing: a stray ``\\n\\n`` inside a caller's
    sentence would become a blank line and break the Markdown HTML block. The
    token scrub is the string-side twin of the numeric helpers: the schedule's
    market columns are NaN for unlined games, so a page that builds
    ``f"O/U {total}"`` would otherwise show the user ``O/U nan``.
    """
    return _NONFINITE_TOKEN.sub(_EM_DASH, " ".join(str(value).split()))


def _text(value: Any, default: str = "") -> str:
    """Escaped, whitespace-collapsed single-line text; ``default`` when missing or blank."""
    if _is_missing(value):
        return _escape(default, quote=True)
    collapsed = _clean(value)
    if not collapsed:
        return _escape(default, quote=True)
    return _escape(collapsed, quote=True)


def _num(value: Any) -> float | None:
    """A finite float, or ``None`` for None/NaN/inf/bool/non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


#: Float-noise allowance for probabilities. ``1 - (1 - p)`` or a sum of
#: complements can land a hair outside ``[0, 1]`` (``1.0000000000000002``); that
#: is the same probability, not a missing one, so it is clamped rather than
#: dropped. Anything beyond this (``1.001``, ``54.0``) is still treated as bad
#: input and shown as missing, never rescaled.
_PROB_TOLERANCE = 1e-9


def _prob(value: Any) -> float | None:
    """A probability in ``[0, 1]`` or ``None`` (out-of-range is treated as missing, never rescaled)."""
    number = _num(value)
    if number is None or number < -_PROB_TOLERANCE or number > 1.0 + _PROB_TOLERANCE:
        return None
    return min(1.0, max(0.0, number))


def _opt(value: Any) -> str:
    """Whitespace-collapsed ``str(value)``, or ``""`` when missing.

    Used instead of ``value or default`` on caller-supplied objects: calling
    ``bool()`` on a pandas ``NA`` raises, and a NaN float is truthy.
    """
    return "" if _is_missing(value) else _clean(value)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    number = _num(value)
    if number is None:
        return default
    return max(low, min(high, int(round(number))))


def _hex(value: Any, default: str) -> str:
    """A validated ``#rgb``/``#rrggbb`` color (safe inside a style attribute) or ``default``."""
    candidate = _opt(value)
    return candidate if _HEX_RE.match(candidate) else default


def _tone(value: Any, default: str = "cyan") -> str:
    candidate = _opt(value).lower()
    return candidate if candidate in TONES else default


def _seq(value: Any) -> list[Any]:
    """Treat ``None`` as empty and a bare string/mapping as one item (never iterate a string's characters)."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """``value`` as a mapping (a pandas row ``Series`` is converted), or ``None`` when it is not row-like."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and not hasattr(value, "columns"):  # a Series, not a DataFrame
        try:
            converted = value.to_dict()
        except Exception:  # noqa: BLE001 - a UI helper must not raise on an odd row object
            return None
        return converted if isinstance(converted, Mapping) else None
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    """:func:`_as_mapping`, with ``{}`` for anything that is not row-like."""
    converted = _as_mapping(value)
    return {} if converted is None else converted


def _records(value: Any) -> list[Any]:
    """Like :func:`_seq`, but a DataFrame becomes its list of row dicts (not its column names)."""
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        try:
            return list(value.to_dict("records"))
        except Exception:  # noqa: BLE001
            return []
    return _seq(value)


def _flag(value: Any) -> bool:
    """Truthiness that never raises (``pd.NA``) and never reads a NaN as ``True``."""
    if _is_missing(value):
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):  # e.g. a multi-element array
        return False


def _rgb(color: str) -> tuple[int, int, int]:
    text = color.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = _rgb(color)
    return f"rgba({red},{green},{blue},{alpha:g})"


def _to_hex(rgb: Sequence[float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, int(round(part)))) for part in rgb))


def _mix(color_a: str, color_b: str, weight_b: float) -> str:
    """Linear sRGB mix; ``weight_b`` 0 -> ``color_a``, 1 -> ``color_b``."""
    a, b = _rgb(color_a), _rgb(color_b)
    return _to_hex([a[i] * (1 - weight_b) + b[i] * weight_b for i in range(3)])


def _luminance(color: str) -> float:
    """WCAG relative luminance."""
    channels = []
    for part in _rgb(color):
        scaled = part / 255.0
        channels.append(scaled / 12.92 if scaled <= 0.03928 else ((scaled + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_on_navy(color: str) -> float:
    return (_luminance(color) + 0.05) / (_luminance(MIDNIGHT_NAVY) + 0.05)


def _distance(color_a: str, color_b: str) -> float:
    a, b = _rgb(color_a), _rgb(color_b)
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _truncate(value: Any, limit: int) -> str:
    """Whitespace-collapsed text cut to ``limit`` characters with an ellipsis (NOT yet escaped)."""
    collapsed = _clean(value) if not _is_missing(value) else ""
    return collapsed if len(collapsed) <= limit else collapsed[: max(1, limit - 1)].rstrip() + "…"


#: Words a page author is likely to write for a :func:`format_value` kind.
_KIND_ALIASES = {
    "percent": "pct", "percentage": "pct", "probability": "pct",
    "odds": "american", "price": "american", "moneyline": "american",
    "usd": "money", "dollar": "money", "dollars": "money",
    "integer": "int", "decimal": "dec2",
}


def format_value(value: Any, kind: str = "auto") -> str:
    """Format ``value`` for display (NOT escaped) using the shared odds helpers.

    ``kind``: ``auto`` (integers plain, other numbers up to 2 decimals, text
    as-is), ``pct`` (0-1 -> ``54.3%``), ``signed_pct`` (``+3.2 pts``),
    ``american`` (``-110``), ``money`` (``+$4.20``), ``int``, ``dec1``,
    ``dec2``. ``None``/NaN/inf always give an em dash; ``kind`` is
    case-insensitive and an unknown kind reads as ``auto``. Text that is not a
    number (``"76%"``, ``"N/A"``) is shown as given under every kind -- a typed
    kind only formats numbers, it never swallows a caller's own label.
    """
    if _is_missing(value):
        return _EM_DASH
    kind = str(kind).strip().lower()
    kind = _KIND_ALIASES.get(kind, kind)
    number = _num(value)
    if number is None:
        return _clean(value) or _EM_DASH
    if kind == "pct":
        return format_pct(number)
    if kind == "signed_pct":
        return format_signed_pct(number)
    if kind == "american":
        return format_american(number)
    if kind == "money":
        return format_money(number)
    if kind == "int":
        return f"{int(round(number)):,}"
    if kind == "dec1":
        return f"{number:,.1f}"
    if kind == "dec2":
        return f"{number:,.2f}"
    if isinstance(value, str):  # numeric-looking text the caller formatted themselves
        return _clean(value)
    if abs(number - round(number)) < 1e-9 and abs(number) < 1e15:
        return f"{int(round(number)):,}"
    text = f"{number:,.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _value_text(value: Any, kind: str = "auto") -> str:
    """Escaped :func:`format_value`."""
    return _escape(format_value(value, kind), quote=True)


# ---------------------------------------------------------------------------
# Teams: logos, names, colors
# ---------------------------------------------------------------------------
#: Historical / alternate codes -> the nflverse code the rest of the app uses.
_CANON_CODES = {
    "LAR": "LA", "STL": "LA", "WSH": "WAS", "ARZ": "ARI", "JAC": "JAX", "OAK": "LV", "SD": "LAC",
}
#: nflverse code -> ESPN logo slug where lower-casing is not enough.
_ESPN_SLUGS = {"LA": "lar", "WAS": "wsh"}
_ESPN_LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
_CODE_RE = re.compile(r"^[A-Z]{2,4}$")


def _build_team_lookup() -> dict[str, str]:
    """Lower-case token -> nflverse code for every code, alias, full name and nickname the app knows.

    Built from ``modules.nfl.NFL_TEAMS`` so the kit accepts exactly what
    ``modules.nfl.lookup_team`` accepts (``KAN``, ``Bucs``, ``Washington``,
    ``Dallas Cowboys``...), plus the historical aliases in ``_CANON_CODES``.
    """
    lookup: dict[str, str] = {}
    for code, entry in _NFL_TEAMS.items():
        full_name, nickname, aliases = entry
        for token in (code, full_name, nickname, *aliases):
            lookup[" ".join(str(token).split()).lower()] = code
    for alias, code in _CANON_CODES.items():
        lookup.setdefault(alias.lower(), code)
    return lookup


_TEAM_LOOKUP: dict[str, str] = _build_team_lookup()


def _team_code(team: Any) -> str:
    """Canonical nflverse code for ``team``, or ``""`` when it names no team and is not code-shaped.

    Resolution order: a known code / alias / full name / nickname (any case,
    so ``"Jets"`` is ``NYJ`` and not an unknown four-letter code), then any
    other 2-4 letter token is kept upper-cased as an *unknown* code (it gets a
    monogram and no logo request), then ``""``.
    """
    if _is_missing(team):
        return ""
    text = " ".join(str(team).split())
    known = _TEAM_LOOKUP.get(text.lower())
    if known:
        return known
    code = text.upper()
    return code if _CODE_RE.match(code) else ""


def _is_known_team(code: str) -> bool:
    return bool(code) and code in _NFL_TEAM_COLORS


def team_logo_url(team: Any) -> str:
    """ESPN logo URL for an NFL team code, or ``""`` for an unknown/blank code.

    Accepts nflverse codes in any case (``LA`` -> ``lar``, ``WAS`` -> ``wsh``)
    plus common aliases (``LAR``, ``WSH``, ``JAC``, ...). An unknown code gets
    no URL rather than a guessed one that would 404.
    """
    code = _team_code(team)
    if not _is_known_team(code):
        return ""
    return _ESPN_LOGO_URL.format(slug=_ESPN_SLUGS.get(code, code.lower()))


def team_name(team: Any) -> str:
    """Full team name (``Dallas Cowboys``) for a known code, else the code, else ``""``. Not escaped."""
    code = _team_code(team)
    entry = _NFL_TEAMS.get(code)
    if entry:
        return str(entry[0])
    return code


@lru_cache(maxsize=256)
def _team_colors(code: str) -> tuple[str, str, str]:
    """``(primary, fill, alt)`` for a team code.

    ``primary`` is the real team color (used for the monogram badge),
    ``fill`` is a version guaranteed visible on navy (3:1 contrast) for bars
    and glows, and ``alt`` is a second visible option used when two teams'
    fills collide. Unknown teams get the brand slate/cyan.
    """
    if not _is_known_team(code):
        return SLATE_BLUE, CIRCUIT_CYAN, STEEL_BLUE
    primary, secondary = (_hex(color, SLATE_BLUE) for color in _team_palette(code))

    def visible(color: str) -> str:
        # Mix toward soft white just far enough to clear 3:1 on navy; this keeps
        # the hue (a dark Cowboys blue becomes a lighter blue, not a gray).
        adjusted = color
        for step in range(0, 24):
            adjusted = _mix(color, SOFT_WHITE, step * 0.04)
            if _contrast_on_navy(adjusted) >= 3.0:
                break
        return adjusted

    # Near-black primaries (Raiders, Bears, Texans, Seahawks...) turn muddy gray
    # when lightened, so lead with the secondary color instead.
    lead, other = (secondary, primary) if _luminance(primary) < 0.03 else (primary, secondary)
    return primary, visible(lead), visible(other)


def team_display_color(team: Any) -> str:
    """A team color that is clearly visible on the navy canvas (for bars, rings, glows)."""
    return _team_colors(_team_code(team))[1]


def matchup_colors(away: Any, home: Any) -> tuple[str, str]:
    """``(away_color, home_color)`` for a split bar, guaranteed visibly different.

    Two navy-ish teams (or the same code twice) would otherwise paint one
    indistinguishable bar; the home side falls back to its alternate color,
    then to brand cyan / steel blue.
    """
    away_code, home_code = _team_code(away), _team_code(home)
    away_color = _team_colors(away_code)[1]
    home_color = _team_colors(home_code)[1]
    if _distance(away_color, home_color) < 90:
        for candidate in (_team_colors(home_code)[2], CIRCUIT_CYAN, STEEL_BLUE):
            if _distance(away_color, candidate) >= 90:
                home_color = candidate
                break
    return away_color, home_color


def _contrast(color_a: str, color_b: str) -> float:
    """WCAG contrast ratio between two solid colors."""
    first, second = _luminance(color_a), _luminance(color_b)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _readable_on(color: str) -> str:
    """Soft white or navy, whichever has the higher WCAG contrast on solid ``color``.

    Compared by actual contrast rather than a luminance cut-off: orange and
    mid-blue team colors (CIN, DEN, CAR, LAC, MIA) sit in the band where navy
    text beats white by a wide margin (5.2:1 vs 3.1:1 on the Bengals orange).
    """
    return SOFT_WHITE if _contrast(color, SOFT_WHITE) >= _contrast(color, MIDNIGHT_NAVY) else MIDNIGHT_NAVY


def team_logo_html(team: Any, size: int = 44) -> str:
    """Inline team logo chip (a ``<span>``, meant to be embedded in other markup).

    Layers, bottom to top: a team-colored monogram underlay, then the ESPN
    logo on an opaque navy chip. See the module docstring for how a failed
    image degrades without ever showing a broken-image icon. Unknown or blank
    codes render the monogram only (no image request at all).
    """
    code = _team_code(team)
    size = _clamp_int(size, 16, 160, 44)
    primary, fill, _alt = _team_colors(code)
    monogram = _escape(code or "?", quote=True)
    font = max(10, int(round(size * (0.34 if len(code) >= 3 else 0.42))))
    url = team_logo_url(code)
    label = _escape(team_name(code) or "Team", quote=True)
    image = (
        f'<img class="bt-logo-img" src="{_escape(url, quote=True)}" alt="" data-code="{monogram}" '
        f'loading="lazy" decoding="async" referrerpolicy="no-referrer" draggable="false" '
        f"onerror=\"this.style.display='none'\">"
        if url
        else ""
    )
    return (
        f'<span class="bt-logo" role="img" aria-label="{label}" '
        f'style="--bt-s:{size}px;--bt-f:{font}px;--bt-p:{primary};--bt-c:{fill};--bt-fg:{_readable_on(primary)}">'
        f'<span class="bt-logo-mono" aria-hidden="true">{monogram}</span>{image}</span>'
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
# The last block of the template restyles Streamlit 1.60's own widgets (select, text/number input, radio,
# expander header). It draws them with react-aria markup that app/style.py (baseweb selectors only) does not
# reach, so on the dark canvas they came out as white fields with a red radio dot. The rules live here, not in
# style.py, so only pages that load this kit are affected.
_CSS_TEMPLATE = """<style>
.bt,.bt *,.bt *::before,.bt *::after{box-sizing:border-box}
.bt{color:$text !important;font-variant-numeric:tabular-nums;line-height:1.4;letter-spacing:normal;text-align:left}
.bt-t-cyan{--bt-t:$cyan;--bt-k:$cyan;--bt-v:$text}
.bt-t-gold{--bt-t:$gold;--bt-k:$gold;--bt-v:$gold}
.bt-t-good{--bt-t:$good;--bt-k:$good;--bt-v:$good}
.bt-t-bad{--bt-t:$bad;--bt-k:$bad;--bt-v:$bad}
.bt-t-warn{--bt-t:$warn;--bt-k:$warn;--bt-v:$warn}
.bt-t-steel{--bt-t:$steel;--bt-k:$steel;--bt-v:$text}
.bt-t-neutral{--bt-t:$steel;--bt-k:$subtle;--bt-v:$text}
.bt-card{position:relative;background:linear-gradient(145deg,rgba($slate,.94),rgba($navy,.97));border:1px solid rgba($cyan,.22);border-radius:16px;padding:1.05rem 1.15rem;margin:.6rem 0;box-shadow:0 12px 30px rgba($deep,.38);transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease}
.bt-lift:hover{transform:translateY(-2px);border-color:rgba($cyan,.6);box-shadow:0 0 0 1px rgba($cyan,.18),0 14px 38px rgba($cyan,.12)}
.bt-selected{border-color:$cyan !important;box-shadow:0 0 0 1px rgba($cyan,.6),0 0 30px rgba($cyan,.24) !important}
.bt-muted{color:$muted !important}
.bt-chip{display:inline-flex;flex:none;align-items:center;gap:.32rem;max-width:100%;overflow:hidden;text-overflow:ellipsis;padding:.15rem .58rem;border-radius:999px;font-size:.7rem;font-weight:700;letter-spacing:.035em;line-height:1.35;white-space:nowrap;color:var(--bt-k,$subtle) !important;background:color-mix(in srgb,var(--bt-t,$steel) 13%,transparent);border:1px solid color-mix(in srgb,var(--bt-t,$steel) 42%,transparent)}
.bt-chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.75rem}
.bt-dot{display:inline-block;flex:none;width:.5rem;height:.5rem;border-radius:50%;background:var(--bt-c,$cyan)}
.bt-logo{position:relative;display:inline-flex;flex:none;width:var(--bt-s,44px);height:var(--bt-s,44px);border-radius:30%;overflow:hidden;vertical-align:middle;background:$plate;border:1px solid color-mix(in srgb,var(--bt-c,$cyan) 55%,transparent);box-shadow:0 0 16px color-mix(in srgb,var(--bt-c,$cyan) 24%,transparent)}
.bt-logo-mono{position:absolute;top:0;right:0;bottom:0;left:0;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,var(--bt-p,$slate),color-mix(in srgb,var(--bt-p,$slate) 60%,$navy));color:var(--bt-fg,$text) !important;font-size:var(--bt-f,14px);font-weight:800;letter-spacing:.02em}
.bt-logo-img{position:absolute;top:0;right:0;bottom:0;left:0;display:block;width:100%;height:100%;max-width:none;margin:0;padding:12%;border:0;border-radius:0;object-fit:contain;background:$plate}
.bt-logo-img::after{content:attr(data-code);position:absolute;top:0;right:0;bottom:0;left:0;display:flex;align-items:center;justify-content:center;padding:0;background:linear-gradient(145deg,var(--bt-p,$slate),color-mix(in srgb,var(--bt-p,$slate) 60%,$navy));color:var(--bt-fg,$text);font-size:var(--bt-f,14px);font-weight:800;letter-spacing:.02em}
.bt-matchup{padding:1.2rem 1.3rem 1.05rem;overflow:hidden}
.bt-mu-body{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:stretch;gap:.5rem 1rem}
.bt-mu-team{display:flex;flex-direction:column;align-items:center;gap:.3rem;min-width:0;padding:.5rem .5rem .7rem;text-align:center;border-radius:14px;border:1px solid transparent}
.bt-mu-team.bt-fav{background:linear-gradient(180deg,rgba($gold,.09),rgba($gold,0) 80%);border-color:rgba($gold,.36);box-shadow:0 0 24px rgba($gold,.10)}
.bt-mu-flag{display:flex;align-items:center;justify-content:center;min-height:1.4rem}
.bt-badge-fav{display:inline-flex;align-items:center;justify-content:center;gap:.35rem;max-width:100%;padding:.14rem .55rem;border-radius:999px;font-size:.62rem;font-weight:800;letter-spacing:.14em;line-height:1.25;text-align:center;color:$gold !important;background:rgba($gold,.11);border:1px solid rgba($gold,.62);box-shadow:0 0 16px rgba($gold,.22)}
.bt-badge-fav::before{content:"";width:.4rem;height:.4rem;border-radius:50%;background:$gold}
.bt-mu-name{font-size:1.04rem;font-weight:760;letter-spacing:-.015em;line-height:1.2;overflow-wrap:anywhere}
.bt-mu-sub{font-size:.76rem;color:$muted !important;overflow-wrap:anywhere}
.bt-nw{white-space:nowrap}
.bt-mu-prob{margin-top:.2rem;font-size:2.1rem;font-weight:800;letter-spacing:-.04em;line-height:1}
.bt-fav .bt-mu-prob{color:$gold !important;text-shadow:0 0 20px rgba($gold,.3)}
.bt-mu-cap{font-size:.62rem;letter-spacing:.13em;text-transform:uppercase;color:$dim !important}
.bt-mu-stats{display:flex;flex-wrap:wrap;justify-content:center;gap:.15rem .85rem;margin-top:.2rem;font-size:.76rem;color:$muted !important}
.bt-mu-stats .bt-b{color:$text !important;font-weight:700}
.bt-mu-mid{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.32rem;min-width:7.5rem;text-align:center}
.bt-mu-kick{font-size:.78rem;color:$subtle !important}
.bt-mu-vs{font-size:.72rem;font-weight:700;letter-spacing:.22em;color:$dim !important}
.bt-mu-spread{font-size:1.1rem;font-weight:770;letter-spacing:-.01em}
.bt-mu-total{font-size:.8rem;color:$muted !important}
.bt-mu-bar{margin-top:.55rem}
.bt-bar{display:flex;gap:3px;height:10px;border-radius:999px;overflow:hidden;background:rgba($navy,.85)}
.bt-bar-seg{display:block;flex:var(--bt-w,1) 1 0%;min-width:2px;height:100%;background:linear-gradient(180deg,color-mix(in srgb,var(--bt-c,$cyan) 82%,$text),var(--bt-c,$cyan));animation:bt-fade .6s ease both}
.bt-bar-seg:first-child{border-radius:999px 2px 2px 999px}
.bt-bar-seg:last-child{border-radius:2px 999px 999px 2px}
.bt-bar-seg:only-child{border-radius:999px}
.bt-bar-empty{--bt-c:$slate;background:rgba($slate,.9)}
.bt-bar-labels{display:flex;justify-content:space-between;gap:1rem;margin-top:.4rem;font-size:.72rem;font-weight:700;letter-spacing:.06em;color:$muted !important}
.bt-bar-lab{display:inline-flex;align-items:center;gap:.4rem}
.bt-tile{padding:.8rem .9rem;display:flex;flex-direction:column;gap:.5rem;min-height:5.4rem}
.bt-tile-row{display:flex;align-items:center;justify-content:space-between;gap:.5rem}
.bt-tile-kick{font-size:.74rem;color:$subtle !important}
.bt-tile-teams{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.bt-tile-code{font-size:.98rem;font-weight:780;letter-spacing:-.01em}
.bt-tile-at{font-size:.72rem;color:$dim !important}
.bt-tile-sub{font-size:.76rem;color:$muted !important}
.bt-tile-lean{display:flex;align-items:center;gap:.4rem;font-size:.76rem;font-weight:700;color:$gold !important}
.bt-live-dot{width:.45rem;height:.45rem;border-radius:50%;background:$cyan;animation:bt-pulse 1.6s ease-in-out infinite}
.bt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(max(8rem,calc((100% - (var(--bt-cols,4) - 1) * .7rem) / var(--bt-cols,4))),1fr));gap:.7rem;margin:.6rem 0}
.bt-stat{position:relative;overflow:hidden;min-width:0;padding:.85rem 1rem .85rem 1.1rem;border-radius:14px;background:linear-gradient(145deg,rgba($slate,.92),rgba($navy,.96));border:1px solid rgba($cyan,.16);transition:transform .16s ease,border-color .16s ease}
.bt-stat::before{content:"";position:absolute;top:0;bottom:0;left:0;width:3px;background:var(--bt-t,$cyan)}
.bt-stat:hover{transform:translateY(-1px);border-color:rgba($cyan,.45)}
.bt-stat-label{font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:$muted !important}
.bt-stat-value{margin-top:.2rem;font-size:1.65rem;font-weight:800;letter-spacing:-.03em;line-height:1.15;color:var(--bt-v,$text) !important;overflow-wrap:anywhere}
.bt-stat-sub{margin-top:.15rem;font-size:.76rem;color:$muted !important}
.bt-meter{margin:.6rem 0}
.bt-meter-head{display:flex;justify-content:space-between;align-items:baseline;gap:.75rem;font-size:.82rem}
.bt-meter-label{font-weight:650;color:$subtle !important}
.bt-meter-value{font-weight:780;color:var(--bt-v,$text) !important}
.bt-meter-track{height:8px;margin-top:.4rem;border-radius:999px;overflow:hidden;background:rgba($navy,.9);border:1px solid rgba($cyan,.13)}
.bt-meter-fill{width:var(--bt-w,0%);height:100%;border-radius:999px;transform-origin:left center;background:linear-gradient(90deg,color-mix(in srgb,var(--bt-t,$cyan) 50%,transparent),var(--bt-t,$cyan));animation:bt-grow .7s cubic-bezier(.2,.7,.2,1) both}
.bt-meter-cap{margin-top:.3rem;font-size:.74rem;color:$muted !important}
.bt-prob{margin:.6rem 0}
.bt-prob-head{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem}
.bt-prob-side{display:flex;flex-direction:column;gap:.1rem;min-width:0}
.bt-prob-side.bt-r{align-items:flex-end;text-align:right}
.bt-prob-name{font-size:.74rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:$muted !important;overflow-wrap:anywhere}
.bt-prob-pct{font-size:1.55rem;font-weight:800;letter-spacing:-.035em;line-height:1.1;color:var(--bt-c,$text) !important}
.bt-prob .bt-bar{margin-top:.5rem}
.bt-prob-cap{margin-top:.4rem;font-size:.74rem;color:$muted !important}
.bt-panel{padding:1.1rem 1.2rem 1.15rem}
.bt-panel-head{display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
.bt-panel-title{font-size:1.04rem;font-weight:780;letter-spacing:-.012em}
.bt-panel-count{font-size:.68rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:$muted !important;white-space:nowrap}
.bt-panel-intro{margin-top:.25rem;font-size:.86rem;line-height:1.5;color:$subtle !important}
.bt-reasons{display:grid;gap:.55rem;margin-top:.8rem}
.bt-reason{display:grid;grid-template-columns:auto minmax(0,1fr);gap:.8rem;padding:.75rem .85rem;border-radius:12px;background:rgba($navy,.6);border:1px solid rgba($cyan,.11);border-left:3px solid var(--bt-t,$steel)}
.bt-reason-n{display:flex;align-items:center;justify-content:center;width:1.75rem;height:1.75rem;border-radius:50%;font-size:.78rem;font-weight:800;color:var(--bt-k,$subtle) !important;background:color-mix(in srgb,var(--bt-t,$steel) 15%,transparent);border:1px solid color-mix(in srgb,var(--bt-t,$steel) 48%,transparent)}
.bt-reason-top{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .55rem}
.bt-reason-title{font-size:.94rem;font-weight:720;line-height:1.3;overflow-wrap:anywhere}
.bt-reason-tag{font-size:.62rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:var(--bt-k,$subtle) !important}
.bt-reason-detail{margin-top:.2rem;font-size:.85rem;line-height:1.5;color:$subtle !important;overflow-wrap:anywhere}
.bt-verdict{display:flex;align-items:center;gap:1.05rem;margin:.7rem 0;padding:1.1rem 1.3rem;border-radius:16px;border:1px solid color-mix(in srgb,var(--bt-t,$steel) 40%,transparent);background:linear-gradient(110deg,color-mix(in srgb,var(--bt-t,$steel) 15%,transparent),rgba($navy,.94) 72%);box-shadow:0 12px 30px rgba($deep,.32)}
.bt-v-icon{display:flex;flex:none;align-items:center;justify-content:center;width:2.7rem;height:2.7rem;border-radius:50%;font-size:1.2rem;font-weight:800;color:var(--bt-k,$text) !important;background:color-mix(in srgb,var(--bt-t,$steel) 17%,transparent);border:1px solid color-mix(in srgb,var(--bt-t,$steel) 58%,transparent)}
.bt-v-body{min-width:0}
.bt-v-head{font-size:1.5rem;font-weight:800;letter-spacing:-.028em;line-height:1.2}
.bt-v-detail{margin-top:.25rem;font-size:.9rem;line-height:1.5;color:$subtle !important}
.bt-tablewrap{overflow-x:auto;margin:.6rem 0;border-radius:14px;border:1px solid rgba($cyan,.18);background:linear-gradient(145deg,rgba($slate,.7),rgba($navy,.92));-webkit-overflow-scrolling:touch}
.bt-table{width:100%;margin:0 !important;border-collapse:collapse !important;border-spacing:0 !important;font-size:.86rem}
.bt-table .bt-th,.bt-table .bt-td{padding:.6rem .85rem !important;border:0 !important;border-bottom:1px solid rgba($cyan,.09) !important;background:transparent !important;color:$text !important;white-space:nowrap}
.bt-table .bt-th{background:rgba($navy,.7) !important;color:$muted !important;font-size:.68rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase}
.bt-table .bt-al-l{text-align:left !important}
.bt-table .bt-al-c{text-align:center !important}
.bt-table .bt-al-r{text-align:right !important}
.bt-table .bt-td.bt-first{font-weight:680;white-space:normal}
.bt-table tbody tr:last-child .bt-td{border-bottom:0 !important}
.bt-table tbody tr:hover .bt-td{background:rgba($cyan,.05) !important}
.bt-table .bt-hl{background:rgba($gold,.075) !important;color:$gold !important;font-weight:760}
.bt-table .bt-th.bt-hl{color:$gold !important}
.bt-table .bt-na{color:$dim !important}
.bt-table .bt-c-good{color:$good !important}
.bt-table .bt-c-bad{color:$bad !important}
.bt-table .bt-c-gold{color:$gold !important}
.bt-table .bt-c-cyan{color:$cyan !important}
.bt-table .bt-c-muted{color:$muted !important}
.bt-risk{padding:.95rem 1.1rem}
.bt-risk-head{display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
.bt-risk-title{font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:$muted !important}
.bt-risk-tier{font-size:1.2rem;font-weight:800;letter-spacing:-.02em;color:var(--bt-k,$text) !important}
.bt-risk-segs{position:relative;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;margin:.7rem 0 .3rem}
.bt-risk-seg{height:10px;border-radius:999px;background:rgba($steel,.2);border:1px solid rgba($steel,.3)}
.bt-risk-seg.bt-on{background:var(--bt-t,$steel);border-color:transparent;box-shadow:0 0 14px color-mix(in srgb,var(--bt-t,$steel) 48%,transparent)}
.bt-risk-mark{position:absolute;top:-4px;left:var(--bt-pos,0%);width:3px;height:18px;margin-left:-1px;border-radius:2px;background:$text;box-shadow:0 0 8px rgba($text,.5)}
.bt-risk-labels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));font-size:.7rem;text-align:center;color:$dim !important}
.bt-risk-labels .bt-on{font-weight:800;color:var(--bt-k,$text) !important}
.bt-risk-score{margin-top:.5rem;font-size:.76rem;color:$muted !important}
.bt-factors{display:grid;gap:.3rem;margin-top:.65rem}
.bt-factor{position:relative;padding-left:1rem;font-size:.82rem;line-height:1.45;color:$subtle !important}
.bt-factor::before{content:"";position:absolute;top:.5em;left:.1rem;width:.38rem;height:.38rem;border-radius:50%;background:var(--bt-t,$steel)}
.bt-corr{padding:.95rem 1.1rem}
.bt-corr-scroll{overflow-x:auto;padding-bottom:.35rem;-webkit-overflow-scrolling:touch}
.bt-corr-grid{display:grid;grid-template-columns:minmax(5.5rem,auto) repeat(var(--bt-n,1),minmax(2.9rem,1fr));gap:3px;min-width:calc(5.5rem + var(--bt-n,1) * 3.1rem)}
.bt-cl{display:flex;align-items:center;font-size:.7rem;color:$muted !important;white-space:nowrap}
.bt-cl-col{justify-content:center;align-self:end;height:5.4rem;writing-mode:vertical-rl;transform:rotate(180deg)}
.bt-cc{display:flex;align-items:center;justify-content:center;min-height:2.3rem;border-radius:8px;font-size:.72rem;font-weight:700;background:var(--bt-bg,$slate);color:var(--bt-fg,$subtle) !important}
.bt-cc.bt-diag{font-weight:600}
.bt-corr-legend{display:flex;flex-wrap:wrap;gap:.3rem 1rem;margin-top:.6rem;font-size:.74rem;color:$muted !important}
.bt-corr-legend span{display:inline-flex;align-items:center;gap:.4rem}
.bt-corr-cap{margin-top:.4rem;font-size:.76rem;color:$subtle !important}
.bt-slip{padding:1.05rem 1.15rem 1.1rem}
.bt-slip-head{display:flex;justify-content:space-between;align-items:center;gap:1rem}
.bt-slip-title{font-size:1.04rem;font-weight:780;letter-spacing:-.012em}
.bt-legs{display:grid;gap:.55rem;margin-top:.8rem}
.bt-leg{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:.4rem 1rem;padding:.7rem .85rem;border-radius:12px;background:rgba($navy,.62);border:1px solid rgba($cyan,.12);border-left:3px solid var(--bt-t,$cyan);transition:border-color .16s ease,transform .16s ease}
.bt-leg:hover{border-color:rgba($cyan,.4);transform:translateX(2px)}
.bt-leg-main{min-width:0}
.bt-leg-top{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .55rem}
.bt-leg-label{font-size:.94rem;font-weight:720;overflow-wrap:anywhere}
.bt-leg-sub{margin-top:.2rem;font-size:.78rem;color:$muted !important;overflow-wrap:anywhere}
.bt-leg-nums{display:flex;align-items:center;gap:.8rem;text-align:right}
.bt-leg-prob{display:flex;flex-direction:column;align-items:flex-end;font-size:.96rem;font-weight:770;line-height:1.15}
.bt-leg-probcap{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:$dim !important}
.bt-leg-price{min-width:3.6rem;padding:.25rem .6rem;border-radius:9px;text-align:center;font-size:.94rem;font-weight:800;background:rgba($slate,.95);border:1px solid rgba($cyan,.3)}
.bt-slip-foot{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;margin-top:.9rem;padding-top:.8rem;border-top:1px solid rgba($cyan,.14)}
.bt-slip-sum{display:flex;flex-direction:column;gap:.1rem;min-width:0;max-width:100%}
.bt-slip-sum-label{font-size:.64rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:$muted !important}
.bt-slip-sum-value{font-size:1.15rem;font-weight:800;letter-spacing:-.02em;overflow-wrap:anywhere}
.bt-empty{margin:.6rem 0;padding:1.5rem 1.1rem;text-align:center;border-radius:16px;border:1px dashed rgba($cyan,.28);background:rgba($navy,.45)}
.bt-empty-title{font-size:.98rem;font-weight:740;color:$text !important}
.bt-empty-body{margin:.3rem auto 0;max-width:46ch;font-size:.85rem;line-height:1.5;color:$muted !important}
.bt-notice{display:flex;gap:.65rem;align-items:flex-start;margin:.6rem 0;padding:.75rem .95rem;border-radius:12px;font-size:.85rem;line-height:1.5;color:$subtle !important;background:color-mix(in srgb,var(--bt-t,$steel) 9%,transparent);border:1px solid color-mix(in srgb,var(--bt-t,$steel) 38%,transparent)}
.bt-notice::before{content:"";flex:none;width:.5rem;height:.5rem;margin-top:.5em;border-radius:50%;background:var(--bt-t,$steel)}
.bt-disclaimer{margin-top:1.2rem;padding-top:.75rem;border-top:1px solid rgba($cyan,.13);font-size:.76rem;line-height:1.55;color:$muted !important}
.bt-tile-kick,.bt-tile-sub,.bt-tile-lean,.bt-panel-title,.bt-panel-intro,.bt-slip-title,.bt-slip-sum-label,.bt-meter-label,.bt-meter-cap,.bt-prob-cap,.bt-v-head,.bt-v-detail,.bt-factor,.bt-notice-text,.bt-empty-title,.bt-empty-body,.bt-corr-cap{min-width:0;overflow-wrap:anywhere}
@keyframes bt-fade{from{opacity:0}to{opacity:1}}
@keyframes bt-grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes bt-pulse{0%,100%{opacity:1}50%{opacity:.3}}
.bt-compact{padding:1rem .8rem .9rem}
.bt-compact .bt-mu-body{grid-template-columns:repeat(2,minmax(0,1fr))}
.bt-compact .bt-mu-mid{grid-column:1 / -1;grid-row:2;flex-direction:row;flex-wrap:wrap;gap:.3rem 1rem;min-width:0}
.bt-compact .bt-mu-prob{font-size:1.75rem}
@media (max-width:820px){
.bt-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
.bt-matchup{padding:1rem .8rem .9rem}
.bt-mu-body{grid-template-columns:repeat(2,minmax(0,1fr))}
.bt-mu-mid{grid-column:1 / -1;grid-row:2;flex-direction:row;flex-wrap:wrap;gap:.3rem 1rem;min-width:0}
.bt-mu-prob{font-size:1.75rem}
.bt-leg{grid-template-columns:minmax(0,1fr)}
.bt-leg-nums{justify-content:space-between}
.bt-verdict{padding:.95rem 1rem}
.bt-v-head{font-size:1.25rem}
.bt-panel,.bt-slip,.bt-corr,.bt-risk{padding-left:.9rem;padding-right:.9rem}
}
@media (max-width:360px){
.bt-grid{grid-template-columns:minmax(0,1fr)}
.bt-stat-value{font-size:1.4rem}
.bt-mu-name{font-size:.95rem}
}
@media (prefers-reduced-motion:reduce){
.bt,.bt *,.bt *::before,.bt *::after{transition:none !important;animation:none !important}
.bt-lift:hover,.bt-stat:hover,.bt-leg:hover{transform:none !important}
}
[data-testid="stMain"] [data-testid="stSelectbox"] [role="group"],
[data-testid="stMain"] [data-testid="stTextInputRootElement"],
[data-testid="stMain"] [data-testid="stNumberInputContainer"]{background:rgba($deep,.92) !important;border:1px solid rgba($cyan,.30) !important;border-radius:10px}
[data-testid="stMain"] [data-testid="stSelectbox"] [role="group"]:focus-within,
[data-testid="stMain"] [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stMain"] [data-testid="stNumberInputContainer"]:focus-within{border-color:$cyan !important;box-shadow:0 0 0 1px rgba($cyan,.45)}
[data-testid="stMain"] input{color:$text !important;-webkit-text-fill-color:$text !important;caret-color:$cyan}
[data-testid="stMain"] input::placeholder{color:$dim !important;-webkit-text-fill-color:$dim !important;opacity:1}
[data-testid="stMain"] [data-testid="stNumberInputContainer"] button{background:transparent !important;color:$cyan !important}
[data-testid="stMain"] [data-testid="stExpander"] summary{background:rgba($navy,.96) !important;color:$text !important;border-radius:10px}
[data-testid="stMain"] [data-testid="stExpander"] summary *{color:$text !important}
[data-testid="stMain"] [data-testid="stExpander"] summary svg{color:$cyan !important}
[data-testid="stMain"] [data-testid="stRadioOption"] > div > div > div:first-child{background:rgba($cyan,.16) !important;box-shadow:inset 0 0 0 1px rgba($cyan,.55)}
[data-testid="stMain"] [data-testid="stRadioOption"] > div > div > div:first-child > div{background:rgba($deep,.95) !important}
[data-testid="stMain"] [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child{background:$gold !important;box-shadow:0 0 12px rgba($gold,.45)}
[data-testid="stMain"] [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div{background:$navy !important}
[data-testid="stSelectboxVirtualDropdown"]{background:$navy !important;border:1px solid rgba($cyan,.30);border-radius:10px}
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] *{color:$text !important}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"]{background:rgba($cyan,.14) !important}
[data-testid="stMain"] [data-testid="stCaptionContainer"]{opacity:1 !important}
</style>"""

_CSS_TOKENS: dict[str, str] = {
    "cyan": CIRCUIT_CYAN,
    "gold": SIGNAL_GOLD,
    "navy": MIDNIGHT_NAVY,
    "slate": SLATE_BLUE,
    "text": SOFT_WHITE,
    "deep": DEEP_NAVY,
    "muted": MUTED_TEXT,
    "subtle": SUBTLE_TEXT,
    "dim": DIM_TEXT,
    "steel": STEEL_BLUE,
    "good": GOOD_GREEN,
    "bad": BAD_RED,
    "warn": WARN_AMBER,
    # Opaque light-silver plate behind a logo. NFL logos are drawn for light
    # backgrounds, so a dark wordmark (Jets, Rams, Cowboys, Seahawks...) all but
    # vanishes on the navy canvas; a soft plate keeps every one legible. It is
    # opaque on purpose: it also hides the monogram underlay once the logo has
    # loaded, so a transparent logo never shows letters through it.
    "plate": (
        f"linear-gradient(145deg,color-mix(in srgb,{SOFT_WHITE} 94%,{SLATE_BLUE}),"
        f"color-mix(in srgb,{SOFT_WHITE} 80%,{SLATE_BLUE}))"
    ),
}
_RGBA_TOKEN = re.compile(r"rgba\(\$([a-z]+),\s*([0-9.]+)\)")
_HEX_TOKEN = re.compile(r"\$([a-z]+)")


def _build_css(template: str) -> str:
    """Resolve ``$name`` / ``rgba($name,.2)`` tokens so every color traces to a declared constant."""
    resolved = _RGBA_TOKEN.sub(
        lambda match: _rgba(_CSS_TOKENS[match.group(1)], float(match.group(2))), template
    )
    return _HEX_TOKEN.sub(lambda match: _CSS_TOKENS[match.group(1)], resolved)


BETTING_CSS: str = _build_css(_CSS_TEMPLATE)


def inject_betting_css() -> None:
    """Emit the component CSS. Call once per page run, after ``apply_global_theme()``."""
    st.markdown(BETTING_CSS, unsafe_allow_html=True)


def render(html: str) -> None:
    """Render any builder output (``st.markdown(html, unsafe_allow_html=True)``).

    Nothing is drawn for ``None`` / NaN or a blank string: ``st.markdown(None)``
    prints the word ``None``, and an empty markdown element still takes a row
    of vertical gap. ``chip_html(None)`` and ``notice_html("")`` legitimately
    return ``""``, so ``render`` of those is a silent no-op.
    """
    if _is_missing(html):
        return
    text = html if isinstance(html, str) else str(html)
    if not text.strip():
        return
    st.markdown(text, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------
def chip_html(label: Any, tone: str = "neutral") -> str:
    """One pill; ``tone`` in cyan|gold|good|bad|warn|steel|neutral. Returns a ``<span>`` (embed it)."""
    text = _text(label)
    if not text:
        return ""
    return f'<span class="bt-chip bt-t-{_tone(tone, "neutral")}">{text}</span>'


def empty_card_html(title: Any = "Nothing to show yet", body: Any = "") -> str:
    """Calm dashed placeholder for an empty / not-yet-run result."""
    body_html = f'<div class="bt-empty-body">{_text(body)}</div>' if _text(body) else ""
    return f'<div class="bt bt-empty"><div class="bt-empty-title">{_text(title, "Nothing to show yet")}</div>{body_html}</div>'


def notice_html(message: Any, tone: str = "info") -> str:
    """A slim inline note for engine warnings / data caveats. ``tone``: cyan|gold|good|bad|warn|steel|neutral|info.

    ``info`` is an alias for ``cyan``. Blank messages render nothing.
    """
    text = _text(message)
    if not text:
        return ""
    resolved = "cyan" if str(tone).strip().lower() == "info" else _tone(tone, "warn")
    return f'<div class="bt bt-notice bt-t-{resolved}" role="note"><div class="bt-notice-text">{text}</div></div>'


def disclaimer_html() -> str:
    """The standard research-only footer shown under every betting page."""
    return (
        '<div class="bt bt-disclaimer">Research and educational analysis only - '
        "not financial or wagering advice. Nothing here places a bet.</div>"
    )


# ---------------------------------------------------------------------------
# Probability splits (shared by the matchup card and probability_bar_html)
# ---------------------------------------------------------------------------
def _resolve_split(left: Any, right: Any) -> tuple[float | None, float | None]:
    """Bar proportions for two sides.

    Both given -> normalised to sum to 1 (so a vigged 0.52/0.50 pair still
    draws a 51/49 bar). Exactly one given -> the other is its complement
    (valid for a two-way market with no push). Neither -> ``(None, None)``.
    """
    first, second = _prob(left), _prob(right)
    if first is None and second is None:
        return None, None
    if first is None:
        first = 1.0 - (second or 0.0)
    if second is None:
        second = 1.0 - first
    total = first + second
    if total <= 0:
        return 0.5, 0.5
    return first / total, second / total


def _split_bar(left_color: str, right_color: str, left: float | None, right: float | None, label: str) -> str:
    if left is None or right is None:
        return (
            '<div class="bt-bar" role="img" aria-label="Win probability unavailable">'
            '<span class="bt-bar-seg bt-bar-empty" style="--bt-w:1"></span></div>'
        )
    return (
        f'<div class="bt-bar" role="img" aria-label="{_escape(label, quote=True)}">'
        f'<span class="bt-bar-seg" style="--bt-c:{left_color};--bt-w:{left * 100:.2f}"></span>'
        f'<span class="bt-bar-seg" style="--bt-c:{right_color};--bt-w:{right * 100:.2f}"></span></div>'
    )


# ---------------------------------------------------------------------------
# Matchup card + game tile
# ---------------------------------------------------------------------------
def _favored_side(favored: Any, away_code: str, home_code: str) -> str | None:
    """``"away"``, ``"home"`` or ``None``. Accepts a team code (any case) or the words away/home."""
    if _is_missing(favored):
        return None
    wanted = str(favored).strip().lower()
    if wanted in ("away", "home"):
        return wanted
    code = _team_code(favored)
    if not code:
        return None
    if code == away_code:  # also the tie-break when both sides share a code
        return "away"
    if code == home_code:
        return "home"
    return None


def _team_block(team: Mapping[str, Any], code: str, *, favored: bool) -> str:
    name = _text(_opt(team.get("name")) or team_name(code), "TBD")
    # Only the record is nowrap (a hyphenated "2-1-1" must not break at a hyphen).
    # The subtitle wraps at spaces: a long "QB Baker Mayfield (questionable ...)"
    # used to be nowrap too and spilled out of its column into the neighbouring
    # team and past the card edge.
    subtitle, record = _text(team.get("subtitle")), _text(team.get("record"))
    detail = " · ".join(part for part in (subtitle, f'<span class="bt-nw">{record}</span>' if record else "") if part)
    stats = []
    rating = _num(team.get("rating"))
    if rating is not None:
        stats.append(f'<span>Rating <span class="bt-b">{rating:,.1f}</span></span>')
    expected = _num(team.get("expected_points"))
    if expected is not None:
        stats.append(f'<span>Exp. <span class="bt-b">{expected:,.1f}</span> pts</span>')
    flag = '<span class="bt-badge-fav">MODEL FAVORS</span>' if favored else ""
    return (
        f'<div class="bt-mu-team{" bt-fav" if favored else ""}">'
        f'<div class="bt-mu-flag">{flag}</div>'
        f"{team_logo_html(code, 56)}"
        f'<div class="bt-mu-name">{name}</div>'
        + (f'<div class="bt-mu-sub">{detail}</div>' if detail else "")
        + f'<div class="bt-mu-prob">{format_pct(_prob(team.get("win_probability")))}</div>'
        '<div class="bt-mu-cap">Win probability</div>'
        + (f'<div class="bt-mu-stats">{"".join(stats)}</div>' if stats else "")
        + "</div>"
    )


def matchup_card_html(
    *,
    away: Mapping[str, Any],
    home: Mapping[str, Any],
    kickoff: str = "",
    chips: Sequence[str] = (),
    favored: str | None = None,
    spread_text: str = "",
    total_text: str = "",
    selected: bool = False,
    compact: bool = False,
) -> str:
    """Premium head-to-head card: two team blocks, a center column, a split win-probability bar.

    ``away`` / ``home``: ``{team (code), name?, subtitle?, record?,
    win_probability (0-1 or None), rating?, expected_points?}``. A bare string
    is read as the team code. ``favored`` is a team code (any case) or the
    words ``"away"``/``"home"``; the matching side gets exactly one gold
    ``MODEL FAVORS`` badge and gold emphasis, and ``None``/unknown shows no
    badge. The bar is proportional to the two probabilities (normalised; one
    missing side uses the other's complement; both missing draws a neutral
    bar). ``selected=True`` adds the cyan selection glow. ``chips`` is a list
    of short labels (``"Divisional"``, ``"Dome"``).

    Width: the default three-column layout (away | center | home) needs
    about 420px of container width. The page-width breakpoint (820px) only sees the
    *browser window*, so a card placed in ``st.columns(2)`` / ``st.columns(3)``
    next to an open sidebar can be far narrower than the window and gets
    cramped. Pass ``compact=True`` there: it forces the stacked layout (the two
    teams side by side, the kickoff / spread / total row underneath) at any
    width. A card in a single full-width column should leave it ``False``.
    """
    away_info = _mapping({"team": away} if isinstance(away, str) else away)
    home_info = _mapping({"team": home} if isinstance(home, str) else home)
    away_code = _team_code(away_info.get("team"))
    home_code = _team_code(home_info.get("team"))
    side = _favored_side(favored, away_code, home_code)

    away_prob, home_prob = _resolve_split(away_info.get("win_probability"), home_info.get("win_probability"))
    away_color, home_color = matchup_colors(away_code, home_code)
    away_label = away_code or "AWAY"
    home_label = home_code or "HOME"
    if away_prob is None or home_prob is None:
        bar_label = "Win probability unavailable"
        caption = '<div class="bt-bar-labels"><span>Win probability unavailable</span></div>'
    else:
        bar_label = f"Win probability: {away_label} {format_pct(away_prob)}, {home_label} {format_pct(home_prob)}"
        caption = (
            '<div class="bt-bar-labels">'
            f'<span class="bt-bar-lab"><span class="bt-dot" style="--bt-c:{away_color}"></span>{_escape(away_label)}</span>'
            f'<span class="bt-bar-lab">{_escape(home_label)}<span class="bt-dot" style="--bt-c:{home_color}"></span></span>'
            "</div>"
        )

    mid = ['<div class="bt-mu-mid">']
    if _text(kickoff):
        mid.append(f'<div class="bt-mu-kick">{_text(kickoff)}</div>')
    mid.append('<div class="bt-mu-vs">@</div>')
    if _text(spread_text):
        mid.append(f'<div class="bt-mu-spread">{_text(spread_text)}</div>')
    if _text(total_text):
        mid.append(f'<div class="bt-mu-total">{_text(total_text)}</div>')
    mid.append("</div>")

    chip_row = "".join(chip_html(chip, "neutral") for chip in _seq(chips))
    aria = _escape(
        f"{team_name(away_code) or away_label} at {team_name(home_code) or home_label}", quote=True
    )
    return (
        f'<div class="bt bt-card bt-lift bt-matchup{" bt-compact" if _flag(compact) else ""}{" bt-selected" if _flag(selected) else ""}" role="group" aria-label="{aria}">'
        '<div class="bt-mu-body">'
        + _team_block(away_info, away_code, favored=side == "away")
        + "".join(mid)
        + _team_block(home_info, home_code, favored=side == "home")
        + "</div>"
        f'<div class="bt-mu-bar">{_split_bar(away_color, home_color, away_prob, home_prob, bar_label)}{caption}</div>'
        + (f'<div class="bt-chips">{chip_row}</div>' if chip_row else "")
        + "</div>"
    )


#: Status labels that mean "being played right now": ``Live``, ``In progress``,
#: ``Q3 10:21``, ``2nd Qtr``, ``Halftime``, ``OT 2:11``, ``Overtime``. Word
#: boundaries matter: a bare ``startswith("ot")`` also lit the live dot for
#: ``Other`` / ``Otters``, and ``startswith("live")`` for ``Lively``.
_LIVE_STATUS = re.compile(
    r"^(?:live\b|in[ -]progress\b|q[1-4]\b|[1-4](?:st|nd|rd|th)\b|quarter\s*[1-4]\b|half(?:time)?\b|end of\b|overtime\b|ot\d?\b)"
)


def _status_tone(status: str) -> tuple[str, bool]:
    """``(tone, is_live)`` for a game-status label."""
    lowered = status.strip().lower()
    if not lowered:
        return "neutral", False
    if _LIVE_STATUS.match(lowered):
        return "cyan", True
    if lowered.startswith(("final", "completed")):
        return "neutral", False
    if lowered.startswith(("postponed", "delayed", "canceled", "cancelled", "suspended")):
        return "warn", False
    return "steel", False


def game_tile_html(
    *,
    away_team: str,
    home_team: str,
    kickoff: str = "",
    status: str = "",
    selected: bool = False,
    subtitle: str = "",
    model_lean: str = "",
) -> str:
    """Compact selectable tile for listing a week's slate (put a Streamlit button under it).

    ``status`` is free text (``Upcoming``, ``Final 27-24``, ``Live``); it picks
    the chip tone and a ``Live`` status gets a pulsing dot. ``model_lean`` is a
    short gold line such as ``Model: KC 61%``.
    """
    away_code, home_code = _team_code(away_team), _team_code(home_team)
    status_text = _clean(status) if not _is_missing(status) else ""
    tone, live = _status_tone(status_text)
    status_chip = ""
    if status_text:
        dot = '<span class="bt-live-dot"></span>' if live else ""
        status_chip = f'<span class="bt-chip bt-t-{tone}">{dot}{_text(status_text)}</span>'
    sub = f'<div class="bt-tile-sub">{_text(subtitle)}</div>' if _text(subtitle) else ""
    lean = (
        f'<div class="bt-tile-lean"><span class="bt-dot" style="--bt-c:{SIGNAL_GOLD}"></span>{_text(model_lean)}</div>'
        if _text(model_lean)
        else ""
    )
    return (
        f'<div class="bt bt-card bt-lift bt-tile{" bt-selected" if _flag(selected) else ""}">'
        f'<div class="bt-tile-row"><span class="bt-tile-kick">{_text(kickoff, "Time TBD")}</span>{status_chip}</div>'
        f'<div class="bt-tile-teams">{team_logo_html(away_code, 30)}<span class="bt-tile-code">{_escape(away_code or "TBD")}</span>'
        f'<span class="bt-tile-at">@</span>{team_logo_html(home_code, 30)}<span class="bt-tile-code">{_escape(home_code or "TBD")}</span></div>'
        f"{sub}{lean}</div>"
    )


# ---------------------------------------------------------------------------
# Stat tiles, meters, probability bar
# ---------------------------------------------------------------------------
def stat_tile_html(label: Any, value: Any, sub: Any = "", tone: str = "cyan") -> str:
    """One KPI tile. ``value`` may be text (shown as given) or a number (formatted).

    ``tone``: cyan | gold | good | bad | neutral (warn/steel also accepted).
    ``cyan``/``neutral`` tiles keep a soft-white value and color only the
    accent bar; gold/good/bad/warn tint the value itself.
    """
    sub_html = f'<div class="bt-stat-sub">{_text(sub)}</div>' if _text(sub) else ""
    return (
        f'<div class="bt bt-stat bt-t-{_tone(tone)}">'
        f'<div class="bt-stat-label">{_text(label)}</div>'
        f'<div class="bt-stat-value">{_value_text(value)}</div>{sub_html}</div>'
    )


def _tile_fields(tile: Any) -> tuple[Any, Any, Any, str] | None:
    if isinstance(tile, Mapping):
        sub = tile.get("sub", tile.get("caption", ""))
        return tile.get("label"), tile.get("value"), sub, tile.get("tone", "cyan")
    if isinstance(tile, (list, tuple)) and len(tile) >= 2:
        return tile[0], tile[1], tile[2] if len(tile) > 2 else "", tile[3] if len(tile) > 3 else "cyan"
    return None


def stat_grid_html(tiles: Sequence[Any], columns: int = 4) -> str:
    """Responsive grid of stat tiles (2 columns on tablets/phones, 1 on very narrow screens).

    ``tiles``: dicts ``{label, value, sub?, tone?}`` or tuples
    ``(label, value[, sub[, tone]])``. Unusable entries are skipped.
    """
    cells = []
    for tile in _records(tiles):
        fields = _tile_fields(tile)
        if fields is not None:
            cells.append(stat_tile_html(*fields))
    count = _clamp_int(columns, 1, 6, 4)
    return f'<div class="bt bt-grid" style="--bt-cols:{count}">{"".join(cells)}</div>'


def meter_html(
    label: Any,
    value: Any,
    *,
    max_value: float = 1.0,
    display: Any = None,
    tone: str = "cyan",
    caption: Any = "",
) -> str:
    """Horizontal gauge (confidence, volatility, efficiency...).

    The fill is ``value / max_value`` clamped to 0-100%. ``display`` overrides
    the right-hand text; by default a 0-1 gauge shows ``72%`` and any other
    scale shows ``72 / 100``. A missing value shows an em dash and an empty
    track.
    """
    maximum = _num(max_value)
    if maximum is None or maximum <= 0:
        maximum = 1.0
    number = _num(value)
    fraction = None if number is None else max(0.0, min(1.0, number / maximum))
    if not _is_missing(display) and str(display).strip():
        shown = _text(display)
    elif number is None:
        shown = _EM_DASH
    elif maximum == 1.0:
        shown = format_pct(fraction, 0)
    else:
        shown = f"{_value_text(number)} / {_value_text(maximum)}"
    width = "0%" if fraction is None else f"{fraction * 100:.1f}%"
    aria = (
        f' aria-valuemin="0" aria-valuemax="{maximum:.10g}" aria-valuenow="{number:.10g}"'
        if number is not None
        else ""
    )
    caption_html = f'<div class="bt-meter-cap">{_text(caption)}</div>' if _text(caption) else ""
    return (
        f'<div class="bt bt-meter bt-t-{_tone(tone)}" style="--bt-w:{width}">'
        f'<div class="bt-meter-head"><span class="bt-meter-label">{_text(label)}</span>'
        f'<span class="bt-meter-value">{shown}</span></div>'
        f'<div class="bt-meter-track" role="meter" aria-label="{_text(label, "Meter")}"{aria}>'
        f'<div class="bt-meter-fill"></div></div>{caption_html}</div>'
    )


def probability_bar_html(
    left_label: Any,
    left_prob: Any,
    right_label: Any,
    right_prob: Any,
    *,
    left_color: str | None = None,
    right_color: str | None = None,
    caption: Any = "",
) -> str:
    """Two-sided split bar (Over vs Under, Home vs Away).

    Header shows each side's label and percentage as given (missing -> em
    dash); the bar is proportional (normalised; one missing side uses the
    other's complement; both missing draws a neutral bar). Defaults: left
    cyan, right steel blue, so gold stays reserved for the favored signal.
    Colors must be ``#rgb``/``#rrggbb`` or they fall back to the defaults.
    """
    left_hex = _hex(left_color, CIRCUIT_CYAN)
    right_hex = _hex(right_color, STEEL_BLUE)
    left_share, right_share = _resolve_split(left_prob, right_prob)
    aria = (
        f"{_truncate(left_label, 40)} {format_pct(_prob(left_prob))}, "
        f"{_truncate(right_label, 40)} {format_pct(_prob(right_prob))}"
    )
    caption_html = f'<div class="bt-prob-cap">{_text(caption)}</div>' if _text(caption) else ""
    return (
        '<div class="bt bt-prob">'
        '<div class="bt-prob-head">'
        f'<div class="bt-prob-side"><span class="bt-prob-name">{_text(left_label)}</span>'
        f'<span class="bt-prob-pct" style="--bt-c:{left_hex}">{format_pct(_prob(left_prob))}</span></div>'
        f'<div class="bt-prob-side bt-r"><span class="bt-prob-name">{_text(right_label)}</span>'
        f'<span class="bt-prob-pct" style="--bt-c:{right_hex}">{format_pct(_prob(right_prob))}</span></div>'
        "</div>"
        f"{_split_bar(left_hex, right_hex, left_share, right_share, aria)}{caption_html}</div>"
    )


# ---------------------------------------------------------------------------
# Reasoning panel, verdict banner
# ---------------------------------------------------------------------------
_REASON_TONES = {
    "pro": ("cyan", "Supports"),
    "con": ("warn", "Caution"),
    "neutral": ("steel", "Context"),
}
#: Page authors reach for these words as often as ``pro``/``con``; any other
#: word still reads as neutral rather than raising or guessing.
_REASON_ALIASES = {
    "pro": "pro", "positive": "pro", "support": "pro", "supports": "pro", "for": "pro",
    "con": "con", "negative": "con", "caution": "con", "against": "con", "risk": "con",
    "neutral": "neutral", "context": "neutral", "info": "neutral",
}


def reasoning_panel_html(
    items: Sequence[Any],
    *,
    title: Any = "Why the model leans this way",
    intro: Any = "",
) -> str:
    """Numbered explanation cards.

    ``items``: dicts ``{title, detail, tone: "pro"|"con"|"neutral", team?}``
    (a bare string is read as a title). Pro items get a cyan marker, cons an
    amber one, neutral a slate one -- each also carries a text tag so meaning
    never depends on color alone. ``team`` adds that team's small logo.
    Entries with no text are skipped; an empty list renders a calm
    placeholder instead of an empty card.
    """
    cards = []
    for raw in _records(items):
        entry = {"title": raw} if isinstance(raw, str) else _mapping(raw)
        heading, detail = _text(entry.get("title")), _text(entry.get("detail"))
        if not heading and not detail:
            continue
        kind = _REASON_ALIASES.get(_opt(entry.get("tone")).lower(), "neutral")
        color, tag = _REASON_TONES[kind]
        team_code = _team_code(entry.get("team"))
        logo = team_logo_html(team_code, 22) if team_code else ""
        number = len(cards) + 1
        cards.append(
            f'<div class="bt-reason bt-t-{color}"><div class="bt-reason-n">{number}</div><div>'
            f'<div class="bt-reason-top">{logo}<span class="bt-reason-title">{heading or detail}</span>'
            f'<span class="bt-reason-tag">{tag}</span></div>'
            + (f'<div class="bt-reason-detail">{detail}</div>' if heading and detail else "")
            + "</div></div>"
        )
    head_title = _text(title, "Why the model leans this way")
    if not cards:
        return (
            f'<div class="bt bt-card bt-panel"><div class="bt-panel-head"><span class="bt-panel-title">{head_title}</span></div>'
            '<div class="bt-reasons"><div class="bt-empty"><div class="bt-empty-body">'
            "No reasoning is available for this selection yet.</div></div></div></div>"
        )
    intro_html = f'<div class="bt-panel-intro">{_text(intro)}</div>' if _text(intro) else ""
    return (
        f'<div class="bt bt-card bt-panel"><div class="bt-panel-head"><span class="bt-panel-title">{head_title}</span>'
        f'<span class="bt-panel-count">{len(cards)} factor{"s" if len(cards) != 1 else ""}</span></div>'
        f'{intro_html}<div class="bt-reasons">{"".join(cards)}</div></div>'
    )


_VERDICT_STYLE = {
    "success": ("good", "✓"),
    "info": ("cyan", "i"),
    "neutral": ("steel", "≈"),
    "warning": ("warn", "!"),
    "danger": ("bad", "✕"),
}


def verdict_banner_html(headline: Any, detail: Any = "", tone: str = "neutral") -> str:
    """Large calm banner for ``Strong value`` / ``Fair price`` / ``Overpriced`` style results.

    ``tone`` matches ``app.page_runtime.pill`` tones (success|info|neutral|
    danger|warning); anything else reads as neutral. The icon glyph repeats
    the tone in text form so the verdict is not conveyed by color alone.
    """
    key = _opt(tone).lower()
    color, glyph = _VERDICT_STYLE.get(key, _VERDICT_STYLE["neutral"])
    detail_html = f'<div class="bt-v-detail">{_text(detail)}</div>' if _text(detail) else ""
    return (
        f'<div class="bt bt-verdict bt-t-{color}" role="status">'
        f'<span class="bt-v-icon" aria-hidden="true">{glyph}</span>'
        f'<div class="bt-v-body"><div class="bt-v-head">{_text(headline, _EM_DASH)}</div>{detail_html}</div></div>'
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _normalise_columns(columns: Any, rows: Sequence[Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for column in _seq(columns):
        if isinstance(column, Mapping):
            key = _opt(column.get("key")) or _opt(column.get("label"))
            label = _opt(column.get("label")) or key
            align = _opt(column.get("align"))
        elif isinstance(column, (list, tuple)) and len(column) >= 2:
            key, label, align = str(column[0]), str(column[1]), ""
        else:
            key = label = str(column)
            align = ""
        if key:
            specs.append({"key": key, "label": label, "align": align})
    if not specs:
        first = next((row for row in rows if isinstance(row, Mapping)), None)
        if first is not None:
            specs = [{"key": str(key), "label": str(key), "align": ""} for key in first]
    return specs


def _cell_parts(cell: Any) -> tuple[Any, str, str]:
    """``(value, format kind, tone)`` for a table cell (plain value or ``{value, format?, tone?}``)."""
    if isinstance(cell, Mapping):
        value = cell.get("value", cell.get("text"))
        return value, _opt(cell.get("format")) or "auto", _opt(cell.get("tone"))
    return cell, "auto", ""


def comparison_table_html(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[Any],
    *,
    highlight_col: str | None = None,
) -> str:
    """Compact model-vs-market table.

    ``columns``: keys, ``(key, label)`` pairs, or ``{key, label, align?}``
    dicts (empty -> the first row's keys). ``rows``: dicts; a cell may be a
    plain value or ``{value, format?, tone?}`` where ``format`` is a
    :func:`format_value` kind (``pct``, ``american``, ``money``...) and
    ``tone`` is good|bad|gold|cyan|muted. Missing/NaN/inf -> a dim em dash.
    The first column is left-aligned, the rest right-aligned (override with
    ``align``). ``highlight_col`` (a column key or label) gets the gold
    highlight. The table scrolls horizontally on narrow screens.
    """
    # Rows are string-keyed copies: column specs are strings, so a row keyed by
    # ints (a DataFrame with default column labels) would otherwise show every
    # cell as missing. A DataFrame is read as its list of row dicts.
    data_rows: list[dict[str, Any]] = []
    for raw in _records(rows):
        row = _as_mapping(raw)
        if row is not None:
            data_rows.append({str(key): value for key, value in row.items()})
    specs = _normalise_columns(columns, data_rows)
    if not specs or not data_rows:
        return empty_card_html("Nothing to compare yet", "Run the analysis to see the model next to the market.")
    wanted = _opt(highlight_col).lower()
    highlight = next(
        (i for i, spec in enumerate(specs) if wanted and wanted in (spec["key"].lower(), spec["label"].lower())),
        None,
    )

    def alignment(index: int, spec: dict[str, str]) -> str:
        choice = spec["align"].strip().lower()
        if choice in ("left", "right", "center"):
            return {"left": "bt-al-l", "right": "bt-al-r", "center": "bt-al-c"}[choice]
        return "bt-al-l" if index == 0 else "bt-al-r"

    head = "".join(
        f'<th class="bt-th {alignment(i, spec)}{" bt-hl" if i == highlight else ""}" scope="col">{_text(spec["label"])}</th>'
        for i, spec in enumerate(specs)
    )
    body = []
    for row in data_rows:
        cells = []
        for i, spec in enumerate(specs):
            value, kind, tone = _cell_parts(row.get(spec["key"]))
            missing = _is_missing(value) or (isinstance(value, str) and not value.strip())
            classes = ["bt-td", alignment(i, spec)]
            if i == 0:
                classes.append("bt-first")
            if i == highlight:
                classes.append("bt-hl")
            if missing:
                classes.append("bt-na")
            elif tone in ("good", "bad", "gold", "cyan", "muted"):
                classes.append(f"bt-c-{tone}")
            cells.append(f'<td class="{" ".join(classes)}">{_EM_DASH if missing else _value_text(value, kind)}</td>')
        body.append(f'<tr>{"".join(cells)}</tr>')
    return (
        '<div class="bt bt-tablewrap"><table class="bt-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# Risk meter
# ---------------------------------------------------------------------------
_RISK_SEGMENTS = (("low", "Low", "good"), ("medium", "Medium", "warn"), ("high", "High", "bad"))


def _risk_key(tier: Any, score_pct: float | None) -> str | None:
    text = str(tier).strip().lower() if not _is_missing(tier) else ""
    if "low" in text:
        return "low"
    if "med" in text or "mod" in text:
        return "medium"
    if "high" in text or "extreme" in text:
        return "high"
    if score_pct is not None:
        return "low" if score_pct < 34 else "medium" if score_pct < 67 else "high"
    return None


def risk_meter_html(tier: Any, score: Any = None, *, factors: Sequence[Any] = ()) -> str:
    """Three-segment Low / Medium / High gauge with the active tier lit.

    ``tier`` is matched loosely (``"Low"``, ``"medium risk"``, ``"HIGH"``). If
    it is missing/unrecognised but ``score`` is given, the tier is derived
    (<34 low, <67 medium, else high). ``score`` is a risk score, higher = riskier:
    0-1 is read as a fraction (``0.42`` -> 42/100), anything above 1 as already
    on a 0-100 scale, and a marker is placed at that position. ``factors`` is a
    list of short bullet strings.
    """
    number = _num(score)
    score_pct = None if number is None else max(0.0, min(100.0, number * 100.0 if number <= 1.0 else number))
    active = _risk_key(tier, score_pct)
    segments, labels = [], []
    active_tone, active_label = "neutral", "Unrated"
    for key, label, color in _RISK_SEGMENTS:
        on = key == active
        if on:
            active_tone, active_label = color, label
        segments.append(f'<span class="bt-risk-seg bt-t-{color}{" bt-on" if on else ""}"></span>')
        labels.append(f'<span class="bt-t-{color}{" bt-on" if on else ""}">{label}</span>')
    marker = f'<span class="bt-risk-mark" style="--bt-pos:{score_pct:.1f}%"></span>' if score_pct is not None else ""
    score_html = f'<div class="bt-risk-score">Risk score {score_pct:.0f} / 100</div>' if score_pct is not None else ""
    notes = [_text(item) for item in _seq(factors)]
    factor_html = "".join(f'<div class="bt-factor">{note}</div>' for note in notes if note)
    return (
        f'<div class="bt bt-card bt-risk bt-t-{active_tone}" role="group" aria-label="Risk: {active_label}">'
        f'<div class="bt-risk-head"><span class="bt-risk-title">Risk</span><span class="bt-risk-tier">{active_label}</span></div>'
        f'<div class="bt-risk-segs">{"".join(segments)}{marker}</div>'
        f'<div class="bt-risk-labels">{"".join(labels)}</div>'
        f"{score_html}" + (f'<div class="bt-factors">{factor_html}</div>' if factor_html else "") + "</div>"
    )


# ---------------------------------------------------------------------------
# Correlation heat-grid
# ---------------------------------------------------------------------------
def _matrix_rows(matrix: Any) -> list[list[Any]]:
    """Coerce lists / numpy arrays / DataFrames to a ragged list of lists without raising."""
    if matrix is None:
        return []
    if hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    rows = []
    for row in _seq(matrix):
        rows.append([] if isinstance(row, (str, bytes)) else _seq(row))
    return rows


#: Roughly what sits behind a heat-grid cell: the card is a slate-to-navy gradient.
#: Used only to pick a legible text color for the translucent cell fill.
_CELL_BACKDROP = _mix(SLATE_BLUE, MIDNIGHT_NAVY, 0.5)


def _corr_text(number: float, signed: bool) -> str:
    """``+0.55`` / ``0.55`` with two decimals, never the ``-0.00`` that a tiny negative rounds to."""
    text = f"{number:+.2f}" if signed else f"{number:.2f}"
    if text == "-0.00":
        return "+0.00" if signed else "0.00"
    return text


def _corr_cell(value: Any, diagonal: bool) -> tuple[str, str, str]:
    """``(display text, background, text color)`` for one heat-grid cell.

    The fill gets more opaque with ``|r|``; the text color is chosen by real
    contrast against the *blended* fill. A fixed soft-white label was only
    2.7:1 on a strong cyan (``r`` near 1) -- exactly the cells a reader looks
    at first -- so strong cells switch to navy text where that reads better.
    """
    number = _num(value)
    if diagonal:
        return ("—" if number is None else _corr_text(number, False)), _rgba(SLATE_BLUE, 0.55), DIM_TEXT
    if number is None:
        return "—", _rgba(SLATE_BLUE, 0.35), DIM_TEXT
    strength = min(1.0, abs(number))
    if strength < 0.05:
        return _corr_text(number, True), _rgba(SLATE_BLUE, 0.7), MUTED_TEXT
    base = CIRCUIT_CYAN if number > 0 else BAD_RED
    alpha = 0.14 + 0.62 * strength
    return _corr_text(number, True), _rgba(base, alpha), _readable_on(_mix(_CELL_BACKDROP, base, alpha))


def correlation_grid_html(labels: Sequence[Any], matrix: Any, *, caption: Any = "") -> str:
    """N x N correlation heat-grid.

    Positive correlation is cyan (stronger = more intense), negative is red,
    values near zero are slate, and the diagonal is muted. ``matrix`` may be
    a list of lists, a numpy array or a DataFrame; ragged or short rows leave
    em-dash cells rather than raising. ``N`` is the larger of ``len(labels)``
    and the number of matrix rows; a missing label reads ``#k``. Labels are
    truncated to 12 characters (the full text stays in the tooltip). The grid
    scrolls horizontally on narrow screens. ``N == 0`` renders a calm
    placeholder.
    """
    names = _seq(labels)
    rows = _matrix_rows(matrix)
    size = max(len(names), len(rows))
    if size == 0:
        return empty_card_html("No correlation data", "Add at least two legs to see how they move together.")

    def label_at(index: int) -> str:
        return _clean(names[index]) if index < len(names) and not _is_missing(names[index]) else f"#{index + 1}"

    cells = ['<span class="bt-cl"></span>']
    for column in range(size):
        full = label_at(column)
        cells.append(
            f'<span class="bt-cl bt-cl-col" title="{_escape(full, quote=True)}">{_escape(_truncate(full, 12))}</span>'
        )
    for row in range(size):
        full = label_at(row)
        cells.append(
            f'<span class="bt-cl" title="{_escape(full, quote=True)}">{_escape(_truncate(full, 12))}</span>'
        )
        values = rows[row] if row < len(rows) else []
        for column in range(size):
            value = values[column] if column < len(values) else None
            text, background, foreground = _corr_cell(value, row == column)
            tip = f"{full} vs {label_at(column)}: {text}"
            cells.append(
                f'<span class="bt-cc{" bt-diag" if row == column else ""}" title="{_escape(tip, quote=True)}" '
                f'style="--bt-bg:{background};--bt-fg:{foreground}">{text}</span>'
            )
    legend = (
        '<div class="bt-corr-legend">'
        f'<span><span class="bt-dot" style="--bt-c:{CIRCUIT_CYAN}"></span>Move together</span>'
        f'<span><span class="bt-dot" style="--bt-c:{BAD_RED}"></span>Move opposite</span>'
        f'<span><span class="bt-dot" style="--bt-c:{SLATE_BLUE}"></span>Unrelated</span></div>'
    )
    caption_html = f'<div class="bt-corr-cap">{_text(caption)}</div>' if _text(caption) else ""
    return (
        '<div class="bt bt-card bt-corr" role="group" aria-label="Correlation matrix">'
        f'<div class="bt-corr-scroll"><div class="bt-corr-grid" style="--bt-n:{size}">{"".join(cells)}</div></div>'
        f"{legend}{caption_html}</div>"
    )


# ---------------------------------------------------------------------------
# Parlay slip
# ---------------------------------------------------------------------------
def _leg_type(value: Any) -> tuple[str, str]:
    """``(display label, chip tone)`` for a leg type."""
    text = _clean(value) if not _is_missing(value) else ""
    lowered = text.lower()
    if "money" in lowered or lowered in ("ml", "h2h"):
        return "Moneyline", "cyan"
    if "spread" in lowered or lowered == "ats":
        return "Spread", "steel"
    if "prop" in lowered:
        return "Prop", "gold"
    return (text or "Leg"), "neutral"


def parlay_slip_html(
    legs: Sequence[Mapping[str, Any]],
    *,
    title: Any = "Your slip",
    summary: Sequence[Any] = (),
) -> str:
    """Slip card listing each parlay leg; an empty list renders a calm empty slip.

    ``legs``: dicts ``{label, sublabel?, type ("Moneyline"|"Spread"|"Prop"),
    probability (0-1), price (American), tone?}``. ``tone`` (cyan|gold|good|
    bad|warn|steel|neutral) colors the leg's left rail (default cyan).
    ``summary`` is an optional footer of ``(label, value)`` pairs -- e.g.
    combined price and joint probability, which the *page* computes with the
    parlay engine; this module never combines legs itself.
    """
    rows = []
    for raw in _records(legs):
        leg = _as_mapping(raw)
        if leg is None:  # a None / bare string is not a leg; a phantom "Leg 2 -- --" row would be misleading
            continue
        index = len(rows) + 1
        type_label, type_tone = _leg_type(leg.get("type"))
        sub = f'<div class="bt-leg-sub">{_text(leg.get("sublabel"))}</div>' if _text(leg.get("sublabel")) else ""
        rows.append(
            f'<div class="bt-leg bt-t-{_tone(leg.get("tone"), "cyan")}"><div class="bt-leg-main">'
            f'<div class="bt-leg-top"><span class="bt-chip bt-t-{type_tone}">{_escape(type_label)}</span>'
            f'<span class="bt-leg-label">{_text(leg.get("label"), f"Leg {index}")}</span></div>{sub}</div>'
            f'<div class="bt-leg-nums"><div class="bt-leg-prob">{format_pct(_prob(leg.get("probability")))}'
            f'<span class="bt-leg-probcap">Model</span></div>'
            f'<span class="bt-leg-price">{format_american(_num(leg.get("price")))}</span></div></div>'
        )
    head_title = _text(title, "Your slip")
    if not rows:
        return (
            f'<div class="bt bt-card bt-slip"><div class="bt-slip-head"><span class="bt-slip-title">{head_title}</span>'
            '<span class="bt-chip bt-t-neutral">0 legs</span></div>'
            '<div class="bt-legs"><div class="bt-empty"><div class="bt-empty-title">Your slip is empty</div>'
            '<div class="bt-empty-body">Add legs to see the combined price and probability.</div></div></div></div>'
        )
    foot_items = []
    for pair in _seq(summary):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            foot_items.append(
                f'<div class="bt-slip-sum"><span class="bt-slip-sum-label">{_text(pair[0])}</span>'
                f'<span class="bt-slip-sum-value">{_value_text(pair[1])}</span></div>'
            )
    foot = f'<div class="bt-slip-foot">{"".join(foot_items)}</div>' if foot_items else ""
    return (
        f'<div class="bt bt-card bt-slip"><div class="bt-slip-head"><span class="bt-slip-title">{head_title}</span>'
        f'<span class="bt-chip bt-t-cyan">{len(rows)} leg{"s" if len(rows) != 1 else ""}</span></div>'
        f'<div class="bt-legs">{"".join(rows)}</div>{foot}</div>'
    )


__all__ = [
    "BAD_RED",
    "BETTING_COLOR_CONSTANTS",
    "BETTING_CSS",
    "DEEP_NAVY",
    "DIM_TEXT",
    "GOOD_GREEN",
    "MUTED_TEXT",
    "PALETTE_HEX",
    "STEEL_BLUE",
    "SUBTLE_TEXT",
    "TONES",
    "VERDICT_TONES",
    "WARN_AMBER",
    "chip_html",
    "comparison_table_html",
    "correlation_grid_html",
    "disclaimer_html",
    "empty_card_html",
    "format_value",
    "game_tile_html",
    "inject_betting_css",
    "matchup_card_html",
    "matchup_colors",
    "meter_html",
    "notice_html",
    "parlay_slip_html",
    "probability_bar_html",
    "reasoning_panel_html",
    "render",
    "risk_meter_html",
    "stat_grid_html",
    "stat_tile_html",
    "team_display_color",
    "team_logo_html",
    "team_logo_url",
    "team_name",
    "verdict_banner_html",
]
