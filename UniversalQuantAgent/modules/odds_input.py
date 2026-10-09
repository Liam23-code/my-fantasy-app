"""Parse and price a sportsbook number a user pastes into a betting page.

The MoneyLines, Player Props, and Parlays pages all accept free text -- an
American moneyline like ``-150`` or ``+120``, or a prop line like ``249.5`` --
and then compare it to the model's probability. This module owns that shared,
UI-free step so the three pages agree on what "implied probability", "value
gap", and "expected value" mean:

* **implied probability** -- the break-even win rate of the price, vig
  included (``1 / decimal odds``), exactly :func:`betting.odds_math.implied_probability`.
* **value gap** -- ``model probability - implied probability``, in probability
  points. Positive means the model rates the side likelier than the price
  requires; it is not a prediction that the bet wins.
* **expected value** -- expected profit on a stake (default $100) if the model
  probability were the true probability, exactly
  :func:`betting.odds_math.expected_value`.

Nothing here touches the network or Streamlit, and nothing places a bet. All
pricing math is delegated to ``betting.odds_math`` so it cannot drift from the
numbers the existing NFL Betting page already shows.
"""
from __future__ import annotations

import math
import re
from typing import Any, NamedTuple

from betting.odds_math import (
    american_to_decimal,
    decimal_to_american,
    expected_value,
    fair_price_from_probability,
    implied_probability,
    remove_vig_two_way,
)

#: Characters people paste in place of an ASCII minus sign (U+2212, en/em dash).
_MINUS_VARIANTS = "−–—‒"
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_EVEN_WORDS = frozenset({"even", "evens", "ev", "evn"})

#: value-gap cut-offs (probability points) for the verdict label.
STRONG_VALUE_GAP = 0.03
SLIGHT_VALUE_GAP = 0.01


class ParsedOdds(NamedTuple):
    """Result of :func:`parse_odds`. ``price`` is always American odds."""

    price: float | None
    error: str | None
    kind: str  # "empty" | "american" | "decimal" | "invalid"


class ParsedNumber(NamedTuple):
    """Result of :func:`parse_number` (a prop line or a spread)."""

    value: float | None
    error: str | None


def _clean(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    for variant in _MINUS_VARIANTS:
        cleaned = cleaned.replace(variant, "-")
    return re.sub(r"[\s,$]", "", cleaned)


def parse_odds(text: Any) -> ParsedOdds:
    """Parse pasted odds into an American price.

    Accepts ``-150``, ``+120``, ``120``, ``−150`` (typographic minus),
    ``EVEN``, and decimal odds such as ``1.91`` (contains a decimal point and
    is below 100). American odds must be at least 100 in magnitude, so a bare
    ``-50`` is rejected rather than silently misread. Blank input is not an
    error -- it simply means "no price entered yet".
    """
    cleaned = _clean(text)
    if not cleaned:
        return ParsedOdds(None, None, "empty")
    if cleaned in _EVEN_WORDS:
        return ParsedOdds(100.0, None, "american")
    if not _NUMBER_PATTERN.match(cleaned):
        return ParsedOdds(None, "Enter odds like -150 or +120 (decimal odds like 1.91 also work).", "invalid")
    value = float(cleaned)
    if "." in cleaned:
        if cleaned.startswith(("-", "+")) or not 1.01 <= value < 100:
            return ParsedOdds(None, "Decimal odds must be between 1.01 and 99.99 (e.g. 1.91).", "invalid")
        return ParsedOdds(float(decimal_to_american(value)), None, "decimal")
    if abs(value) < 100:
        return ParsedOdds(None, "American odds are at least 100 in size (e.g. -110, +105, -150).", "invalid")
    return ParsedOdds(value, None, "american")


def parse_number(
    text: Any,
    *,
    allow_negative: bool = True,
    minimum: float | None = None,
    maximum: float | None = None,
    label: str = "value",
) -> ParsedNumber:
    """Parse a prop line or spread such as ``249.5``, ``-3.5``, or ``0.5``."""
    cleaned = _clean(text)
    if not cleaned:
        return ParsedNumber(None, None)
    if not _NUMBER_PATTERN.match(cleaned):
        return ParsedNumber(None, f"Enter the {label} as a number, e.g. 249.5.")
    value = float(cleaned)
    if value < 0 and not allow_negative:
        return ParsedNumber(None, f"The {label} cannot be negative.")
    if minimum is not None and value < minimum:
        return ParsedNumber(None, f"The {label} must be at least {minimum:g}.")
    if maximum is not None and value > maximum:
        return ParsedNumber(None, f"The {label} must be at most {maximum:g}.")
    return ParsedNumber(value, None)


def format_american(price: float | None) -> str:
    """``+120`` / ``-150``; an em dash when there is no usable price."""
    if price is None or not math.isfinite(float(price)):
        return "—"
    value = int(round(float(price)))
    return f"+{value}" if value > 0 else str(value)


def format_pct(probability: float | None, digits: int = 1) -> str:
    """``54.3%``; an em dash for ``None``/NaN."""
    if probability is None or not math.isfinite(float(probability)):
        return "—"
    return f"{float(probability) * 100:.{digits}f}%"


def format_signed_pct(points: float | None, digits: int = 1) -> str:
    """Signed probability points, e.g. ``+3.2 pts``; an em dash for ``None``/NaN."""
    if points is None or not math.isfinite(float(points)):
        return "—"
    shown = round(float(points) * 100, digits)
    if shown == 0:  # -0.00004 would otherwise print "-0.0 pts", a shortfall that is not there
        return f"{0:.{digits}f} pts"
    return f"{shown:+.{digits}f} pts"


def format_money(amount: float | None, digits: int = 2) -> str:
    """Signed dollars, e.g. ``+$4.20`` / ``-$11.00``; an em dash for ``None``/NaN."""
    if amount is None or not math.isfinite(float(amount)):
        return "—"
    value = round(float(amount), digits)  # round first so -0.004 reads "$0.00", not "-$0.00"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{digits}f}"


def no_vig_probabilities(price_a: float, price_b: float) -> tuple[float, float]:
    """De-vigged fair probabilities for a two-way market (thin wrapper, one import path for pages)."""
    return remove_vig_two_way(price_a, price_b)


def value_verdict(value_gap: float) -> tuple[str, str]:
    """Map a value gap to ``(label, tone)``; tone matches ``app.page_runtime.pill`` tones.

    A NaN/None gap is "Unavailable", never silently "Overpriced" (every
    comparison against NaN is False, which would otherwise fall through).
    """
    if value_gap is None or not math.isfinite(float(value_gap)):
        return "Unavailable", "neutral"
    if value_gap >= STRONG_VALUE_GAP:
        return "Strong value", "success"
    if value_gap >= SLIGHT_VALUE_GAP:
        return "Slight value", "info"
    if value_gap > -SLIGHT_VALUE_GAP:
        return "Fair price", "neutral"
    return "Overpriced", "danger"


def evaluate_price(model_probability: float, price: float, *, stake: float = 100.0) -> dict[str, Any]:
    """Compare a model probability to a sportsbook American price.

    Returns the break-even (implied) probability, the model probability, the
    value gap, expected value on ``stake`` (and as a percentage of stake), the
    model's own fair American price, and a verdict label/tone.
    """
    probability = float(model_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"model_probability must be within [0, 1], got {model_probability!r}")
    # parse_odds already enforces this for pasted text; checking again here keeps
    # a hand-built price (e.g. +50, which is not a real American price) from
    # silently producing a plausible-looking EV.
    if not math.isfinite(float(price)) or abs(float(price)) < 100:
        raise ValueError(f"American odds must be finite and at least 100 in size, got {price!r}")
    implied = implied_probability(price)
    gap = probability - implied
    ev = expected_value(probability, price, stake)
    label, tone = value_verdict(gap)
    return {
        "price": float(price),
        "decimal_odds": round(american_to_decimal(price), 4),
        "implied_probability": round(implied, 4),
        "model_probability": round(probability, 4),
        "value_gap": round(gap, 4),
        "expected_value": round(ev, 2),
        "ev_pct": round(ev / stake, 4) if stake else 0.0,
        "stake": float(stake),
        "fair_price": fair_price_from_probability(probability),
        "verdict": label,
        "verdict_tone": tone,
        "is_value": gap >= SLIGHT_VALUE_GAP,
    }


__all__ = [
    "ParsedNumber",
    "ParsedOdds",
    "SLIGHT_VALUE_GAP",
    "STRONG_VALUE_GAP",
    "evaluate_price",
    "format_american",
    "format_money",
    "format_pct",
    "format_signed_pct",
    "no_vig_probabilities",
    "parse_number",
    "parse_odds",
    "value_verdict",
]
