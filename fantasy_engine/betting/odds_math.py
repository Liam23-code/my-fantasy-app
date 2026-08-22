"""American-odds conversions, vig removal, edge, and expected value.

Pure, deterministic, offline. Every function here is a closed-form transform
of numbers the caller supplies -- prices already loaded from ``odds.json``
or a user upload. No network access, no external data.
"""

from __future__ import annotations


def american_to_decimal(price: float) -> float:
    """Convert American odds (e.g. -110, +150) to decimal odds (e.g. 1.909, 2.5)."""
    price = float(price)
    if price == 0:
        raise ValueError("American odds price cannot be 0")
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / abs(price)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds back to American odds."""
    decimal_odds = float(decimal_odds)
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1.0")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    return round(-100.0 / (decimal_odds - 1.0))


def implied_probability(price: float) -> float:
    """Raw implied win probability of one American-odds price, vig included."""
    return 1.0 / american_to_decimal(price)


def remove_vig_two_way(price_a: float, price_b: float) -> tuple[float, float]:
    """De-vig a two-sided market (over/under, home/away) to fair probabilities.

    Normalizes the two raw implied probabilities -- which sum to more than
    1.0 because of the book's hold -- so they sum to exactly 1.0. This is
    the standard proportional de-vig method: each side's fair share of the
    hold-free probability mass is proportional to its own raw implied
    probability.
    """
    p_a = implied_probability(price_a)
    p_b = implied_probability(price_b)
    total = p_a + p_b
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return p_a / total, p_b / total


def hold(price_a: float, price_b: float) -> float:
    """A two-way market's total hold (vig), e.g. 0.045 for a 4.5% hold."""
    return implied_probability(price_a) + implied_probability(price_b) - 1.0


def fair_price_from_probability(probability: float) -> float:
    """The (vig-free) American odds implied by a true win probability."""
    probability = max(1e-6, min(1.0 - 1e-6, float(probability)))
    return decimal_to_american(1.0 / probability)


def expected_value(probability: float, price: float, stake: float = 100.0) -> float:
    """Expected profit on ``stake`` units at ``price``, given a true win probability."""
    decimal_odds = american_to_decimal(price)
    payout_if_win = stake * (decimal_odds - 1.0)
    return float(probability) * payout_if_win - (1.0 - float(probability)) * stake


def edge(model_probability: float, price: float) -> float:
    """The model's edge over the market: model probability minus the market's raw implied probability."""
    return float(model_probability) - implied_probability(price)


def edge_vs_fair(model_probability: float, price_a: float, price_b: float, *, side: str) -> float:
    """Edge measured against the market's *de-vigged* fair probability, not its raw (vig-included) one.

    ``side`` selects which of the two de-vigged probabilities to compare
    ``model_probability`` against -- ``"a"`` for ``price_a``'s side,
    ``"b"`` for ``price_b``'s side. Comparing to the vig-included raw
    probability (:func:`edge`) always understates the model's true edge by
    roughly half the hold; this is the more honest number for ranking bets.
    """
    fair_a, fair_b = remove_vig_two_way(price_a, price_b)
    fair = fair_a if side == "a" else fair_b
    return float(model_probability) - fair
