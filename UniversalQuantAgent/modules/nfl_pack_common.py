"""Shared scoring math, input coercion and adjustment helpers for the NFL model upgrade pack.

Every model in the pack (``qb_collapse_model``, ``pass_game_collapse``,
``run_game_collapse``, ``explosion_model``, ``explosive_play_model``,
``coverage_mismatch_model``, ``red_zone_funnel_model``, ``game_script_model``)
is a small, pure, deterministic function from a flat ``{input_name: number}``
mapping to a probability (or a signed factor) plus an audit trail. They all
share the pieces in this module so that they behave the same way when data is
missing, when a value arrives as ``"31%"`` or ``NaN``, and when a probability is
applied to a projection.

The scoring model (a transparent weighted z-score through a logistic)
----------------------------------------------------------------------
Each model declares a list of :class:`Factor` objects. A factor says *which
input*, *what a league-average value looks like* (``mean``), *how far one
standard deviation is* (``sd``), *how much it matters* (``weight``) and *which
way it pushes* (``direction``). :func:`score_factors` turns the inputs the
caller actually supplied into one number::

    z = sum(weight_i * direction_i * clip(z_i)) / sum(weight over ALL factors)

A factor the caller did not supply contributes ``z_i = 0`` (neutral) but still
counts in the denominator. So **missing data shrinks the answer toward the
league-average case instead of inflating it**, and ``coverage`` reports how much
of the weight was actually observed. :func:`logistic_probability` then maps the
score to a probability::

    p = sigmoid(logit(base_rate) + gain * z)

With every input missing (or exactly average) ``z = 0`` and ``p = base_rate``
exactly. That value is the model's :data:`reference_probability`.

Why every adjustment is "re-centered"
-------------------------------------
The historical per-game averages the projection engine starts from already
*contain* the league-average amount of collapse / explosion / script variation.
Multiplying them by ``1 - p * 0.65`` with ``p`` near its base rate would take a
permanent bite out of every quarterback regardless of matchup. So the helpers
here apply the brief's formulas in their exact form but divide by the same
formula evaluated at the neutral reference probability::

    factor = (1 - p * c) / (1 - p_ref * c)

A league-average situation therefore leaves a projection untouched and only the
*deviation* from average moves it. ``recenter=False`` gives the literal formula.

HONEST LIMITS
-------------
* Weights are **priors, not fitted coefficients**. The project has no offline
  joint-outcome data to fit them against; each model reports ``"basis"`` saying
  so. The league references (``mean`` / ``sd``) are empirical where a free
  nflverse feed exists and priors where one does not (each is labelled).
* A factor with no free public source (PFF-style pass-block win rate, CB shadow
  rate, bracket coverage) is accepted as an optional input and is neutral until
  the caller supplies it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

LOG = logging.getLogger("nfl_pack")

#: Bump when a model's inputs, weights or output keys change in a way callers could notice.
PACK_VERSION = "1.0"

#: One input is never allowed to move the score by more than this many SDs.
Z_CLIP = 3.0
#: Probabilities handed to formulas are kept strictly inside (0, 1) for logit().
_EPS = 1e-4
#: Rounding applied to every number a model returns.
ROUND = 4

#: The flat projection vocabulary the pack's apply-functions read and write.
PROJECTION_FIELDS: tuple[str, ...] = (
    "qb_yards",
    "qb_completion_pct",
    "qb_td_prob",
    "qb_int_prob",
    "qb_attempts",
    "wr_yards",
    "wr_td_prob",
    "te_yards",
    "te_td_prob",
    "rb_yards",
    "rb_td_prob",
    "rb_attempts",
    "team_points",
)

#: Net multiplier the whole pack may apply to any one field (see the pipeline's guard).
CUMULATIVE_FACTOR_BOUNDS: tuple[float, float] = (0.40, 2.00)


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    """``value`` limited to ``[low, high]`` (no NaN handling -- callers pass finite floats)."""
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    """Numerically safe logistic function."""
    if x >= 0:
        e = math.exp(-min(x, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(x, -700.0))
    return e / (1.0 + e)


def logit(p: float) -> float:
    """Inverse of :func:`sigmoid`; ``p`` is clamped away from 0 and 1."""
    p = clamp(p, _EPS, 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def signed_unit(z: float, scale: float = 1.5) -> float:
    """Squash a z-score to ``(-1, 1)`` with ``tanh(z / scale)`` (0 stays 0)."""
    return math.tanh(z / scale) if scale > 0 else 0.0


def to_float(value: Any) -> float | None:
    """A finite float, or ``None`` for anything that is not a usable number.

    Accepts ints/floats/numpy scalars and numeric strings (``"3.4"``, ``"1,204"``,
    ``"31.5%"`` -> ``0.315``). ``None``, ``bool``, ``NaN``, ``inf``, ``pd.NA`` and
    unparseable text are all ``None`` -- a model never raises on bad input.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            if text.endswith("%"):
                number = float(text[:-1]) / 100.0
            else:
                number = float(text)
        else:
            number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def to_rate(value: Any) -> float | None:
    """A rate in ``[0, 1]`` or ``None``.

    Values in ``(1, 100]`` are read as percentages (``31.5`` -> ``0.315``); a value
    of exactly ``1`` is read as 100% (a fraction), not 1%. Negatives and anything
    above 100 are rejected as data errors rather than clamped into plausibility.
    """
    number = to_float(value)
    if number is None:
        return None
    if number < 0.0 or number > 100.0:
        return None
    if number > 1.0:
        number /= 100.0
    return number


def r(value: float, digits: int = ROUND) -> float:
    """Round for output; keeps ``-0.0`` from leaking into JSON."""
    out = round(float(value), digits)
    return 0.0 if out == 0 else out


# ---------------------------------------------------------------------------
# Factor scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Factor:
    """One input of a model: where it lives, what 'average' is, how much it matters.

    ``direction`` is ``+1`` when a *higher* value raises the model's output
    (more collapse risk, more explosion), ``-1`` when a higher value lowers it.
    ``kind`` is ``"rate"`` (0-1 fraction, percentages auto-detected) or
    ``"value"`` (used as given). ``mean`` / ``sd`` are in the same units the
    caller supplies (fractions for a rate). ``source`` says where the league
    reference came from (an nflverse feed + seasons, or ``"prior"``).
    """

    key: str
    mean: float
    sd: float
    weight: float
    direction: int
    label: str
    kind: str = "value"
    unit: str = ""
    aliases: tuple[str, ...] = ()
    source: str = "prior"

    def __post_init__(self) -> None:
        if self.sd <= 0:
            raise ValueError(f"factor {self.key!r}: sd must be positive")
        if self.weight <= 0:
            raise ValueError(f"factor {self.key!r}: weight must be positive")
        if self.direction not in (-1, 1):
            raise ValueError(f"factor {self.key!r}: direction must be +1 or -1")
        if self.kind not in ("rate", "value"):
            raise ValueError(f"factor {self.key!r}: kind must be 'rate' or 'value'")


def read_input(inputs: Mapping[str, Any] | None, factor: Factor) -> tuple[float | None, str | None]:
    """``(value, key_used)`` for ``factor`` from ``inputs``, trying ``key`` then each alias."""
    if not inputs:
        return None, None
    for key in (factor.key, *factor.aliases):
        if key not in inputs:
            continue
        value = to_rate(inputs[key]) if factor.kind == "rate" else to_float(inputs[key])
        if value is not None:
            return value, key
    return None, None


def score_factors(
    inputs: Mapping[str, Any] | None,
    factors: Sequence[Factor],
    *,
    z_clip: float = Z_CLIP,
) -> dict[str, Any]:
    """Weighted, direction-signed z-score of ``inputs`` over ``factors``.

    Returns ``{"z", "coverage", "used", "defaulted", "contributions"}``. ``z`` is
    divided by the weight of *all* factors, so unsupplied factors pull the score
    toward 0 (neutral). ``contributions[key]`` carries the raw ``value``, the
    signed per-factor ``z`` (positive = pushes the output up), ``weight``,
    ``effect`` (that factor's share of ``z``) and ``label`` / ``unit``.
    """
    total_weight = sum(f.weight for f in factors)
    if total_weight <= 0:
        return {"z": 0.0, "coverage": 0.0, "used": [], "defaulted": [f.key for f in factors], "contributions": {}}
    weighted = 0.0
    used_weight = 0.0
    used: list[str] = []
    defaulted: list[str] = []
    contributions: dict[str, dict[str, Any]] = {}
    for factor in factors:
        value, via = read_input(inputs, factor)
        if value is None:
            defaulted.append(factor.key)
            continue
        z_raw = clamp((value - factor.mean) / factor.sd, -z_clip, z_clip)
        z_dir = z_raw * factor.direction
        effect = factor.weight * z_dir / total_weight
        weighted += factor.weight * z_dir
        used_weight += factor.weight
        used.append(factor.key)
        contributions[factor.key] = {
            "value": r(value, 4),
            "z": r(z_dir, 3),
            "weight": factor.weight,
            "effect": r(effect, 4),
            "label": factor.label,
            "unit": factor.unit,
            "via": via,
        }
    return {
        "z": weighted / total_weight,
        "coverage": used_weight / total_weight,
        "used": used,
        "defaulted": defaulted,
        "contributions": contributions,
    }


def logistic_probability(base_rate: float, z: float, gain: float) -> float:
    """``sigmoid(logit(base_rate) + gain * z)`` -- exactly ``base_rate`` when ``z == 0``."""
    return sigmoid(logit(base_rate) + gain * z)


def describe_drivers(contributions: Mapping[str, Mapping[str, Any]], *, top: int = 3, floor: float = 0.005) -> list[str]:
    """Plain-English lines for the factors that moved the score most (largest ``|effect|`` first)."""
    ranked = sorted(contributions.values(), key=lambda c: abs(c.get("effect", 0.0)), reverse=True)
    lines: list[str] = []
    for item in ranked:
        if len(lines) >= top or abs(item.get("effect", 0.0)) < floor:
            break
        z = float(item.get("z", 0.0))
        unit = str(item.get("unit") or "")
        value = float(item.get("value", 0.0))
        shown = f"{value * 100:.1f}%" if unit == "%" else (f"{value:.2f} {unit}".strip() if unit else f"{value:.2f}")
        direction = "raises" if z > 0 else "lowers"
        lines.append(f"{item.get('label', 'input')} {shown} ({z:+.1f} SD) {direction} the score")
    return lines


def run_factor_model(
    *,
    model: str,
    primary_key: str,
    inputs: Mapping[str, Any] | None,
    factors: Sequence[Factor],
    base_rate: float,
    gain: float,
    basis: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard probability model: score the factors, map through the logistic, build the audit dict.

    The returned mapping always contains ``model``, ``<primary_key>`` and the
    generic alias ``probability`` (same number), ``reference_probability`` (the
    value for all-neutral inputs) and ``references`` (``{output_key: reference}``,
    the form the apply-functions read -- a multi-output model adds its other
    outputs to it), ``z_score``, ``coverage``, ``inputs_used``,
    ``inputs_defaulted``, ``contributions``, ``drivers`` and ``basis``.
    """
    scored = score_factors(inputs, factors)
    probability = clamp(logistic_probability(base_rate, scored["z"], gain), 0.0, 1.0)
    result: dict[str, Any] = {
        "model": model,
        primary_key: r(probability),
        "probability": r(probability),
        "reference_probability": r(base_rate),
        "references": {primary_key: r(base_rate)},
        "z_score": r(scored["z"], 4),
        "coverage": r(scored["coverage"], 3),
        "inputs_used": scored["used"],
        "inputs_defaulted": scored["defaulted"],
        "contributions": scored["contributions"],
        "drivers": describe_drivers(scored["contributions"]),
        "basis": basis,
    }
    if extra:
        result.update(extra)
    log_result(result, primary_key)
    return result


def log_result(result: Mapping[str, Any], primary_key: str = "probability") -> None:
    """Debug-log one model result (the brief asks for the probabilities to be loggable)."""
    if not LOG.isEnabledFor(logging.DEBUG):
        return
    LOG.debug(
        "%s: %s=%.4f (reference %.4f, z=%+.3f, coverage=%.0f%%, used=%d, defaulted=%d) drivers=%s",
        result.get("model"),
        primary_key,
        float(result.get(primary_key, result.get("probability", 0.0)) or 0.0),
        float(result.get("reference_probability", 0.0) or 0.0),
        float(result.get("z_score", 0.0) or 0.0),
        100.0 * float(result.get("coverage", 0.0) or 0.0),
        len(result.get("inputs_used", ()) or ()),
        len(result.get("inputs_defaulted", ()) or ()),
        "; ".join(result.get("drivers", ()) or ()),
    )


# ---------------------------------------------------------------------------
# Applying a probability to a projection
# ---------------------------------------------------------------------------
def adjustment_factor(
    probability: float,
    coefficient: float,
    *,
    direction: int,
    reference: float = 0.0,
    recenter: bool = True,
) -> float:
    """The multiplier ``(1 +/- p*c)``, optionally divided by the same formula at ``reference``.

    ``direction=-1`` is the "haircut" form ``1 - p*c`` (collapse); ``+1`` is the
    "boost" form ``1 + p*c`` (explosion, funnels). With ``recenter=True`` and
    ``reference`` = the model's neutral probability, an average situation returns
    exactly ``1.0``. The result is never negative.
    """
    sign = -1.0 if direction < 0 else 1.0
    numerator = 1.0 + sign * probability * coefficient
    if recenter:
        denominator = 1.0 + sign * reference * coefficient
        if denominator > 1e-9:
            numerator /= denominator
    return max(0.0, numerator)


def _is_number(value: Any) -> bool:
    return to_float(value) is not None


def scale_field(
    projection: dict[str, Any],
    key: str,
    *,
    stage: str,
    probability: float,
    coefficient: float,
    direction: int,
    reference: float,
    recenter: bool,
    adjustments: list[dict[str, Any]],
    cap: float | None = None,
) -> None:
    """In-place ``projection[key] *= (1 +/- p*c)`` (re-centered) with an audit entry.

    A missing / non-numeric field is skipped and recorded as ``applied=False`` so a
    caller can see *why* a number did not move. ``cap`` clamps the result (use
    ``1.0`` for a true probability or a completion fraction).
    """
    before = to_float(projection.get(key))
    if before is None:
        adjustments.append({"stage": stage, "field": key, "applied": False, "reason": "field missing or not numeric"})
        return
    factor = adjustment_factor(probability, coefficient, direction=direction, reference=reference, recenter=recenter)
    after = before * factor
    if cap is not None:
        after = min(after, cap)
    projection[key] = after
    adjustments.append(
        {
            "stage": stage,
            "field": key,
            "applied": True,
            "before": r(before, 4),
            "after": r(after, 4),
            "factor": r(factor, 4),
            "probability": r(probability),
            "reference": r(reference),
            "coefficient": coefficient,
            "direction": "boost" if direction > 0 else "haircut",
            "recentered": bool(recenter),
        }
    )


def add_to_field(
    projection: dict[str, Any],
    key: str,
    *,
    stage: str,
    probability: float,
    amount: float,
    reference: float,
    recenter: bool,
    adjustments: list[dict[str, Any]],
    floor: float = 0.0,
) -> None:
    """In-place ``projection[key] += p * amount`` (re-centered to ``(p - ref) * amount``)."""
    before = to_float(projection.get(key))
    if before is None:
        adjustments.append({"stage": stage, "field": key, "applied": False, "reason": "field missing or not numeric"})
        return
    delta = ((probability - reference) if recenter else probability) * amount
    after = max(floor, before + delta)
    projection[key] = after
    adjustments.append(
        {
            "stage": stage,
            "field": key,
            "applied": True,
            "before": r(before, 4),
            "after": r(after, 4),
            "delta": r(after - before, 4),
            "probability": r(probability),
            "reference": r(reference),
            "amount": amount,
            "recentered": bool(recenter),
        }
    )


def guard_cumulative(
    original: Mapping[str, Any],
    adjusted: dict[str, Any],
    keys: Iterable[str],
    *,
    bounds: tuple[float, float] = CUMULATIVE_FACTOR_BOUNDS,
) -> list[dict[str, Any]]:
    """Clamp each ``adjusted[key]`` to ``bounds`` x ``original[key]``; return one note per clamp.

    Several stages stack multiplicatively (collapse x explosion x mismatch x
    funnel x script). Each is bounded by construction, but the product of five
    mild boosts is not obviously mild, so the pipeline refuses to move any field
    outside ``CUMULATIVE_FACTOR_BOUNDS`` of where it started and says so.
    """
    notes: list[dict[str, Any]] = []
    low, high = bounds
    for key in keys:
        before = to_float(original.get(key))
        after = to_float(adjusted.get(key))
        if before is None or after is None or before <= 0:
            continue
        lo, hi = before * low, before * high
        if after < lo or after > hi:
            clipped = clamp(after, lo, hi)
            adjusted[key] = clipped
            notes.append({"field": key, "unguarded": r(after, 4), "guarded": r(clipped, 4), "bounds": [low, high]})
    return notes


def blend_man_zone(man_value: Any, zone_value: Any, man_rate: Any) -> float | None:
    """Coverage-weighted blend ``man_rate*man + (1-man_rate)*zone`` of a man/zone split.

    Returns whichever side exists if only one is supplied, ``None`` if neither is.
    ``man_rate`` (how often the opponent plays man) defaults to the league-typical
    0.30 when missing or malformed.
    """
    man = to_float(man_value)
    zone = to_float(zone_value)
    if man is None and zone is None:
        return None
    if man is None:
        return zone
    if zone is None:
        return man
    rate = to_rate(man_rate)
    rate = 0.30 if rate is None else rate
    return rate * man + (1.0 - rate) * zone


def merge_inputs(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    """Left-to-right merge where a later non-``None`` value wins (``None`` never erases)."""
    merged: dict[str, Any] = {}
    for source in sources:
        if not source:
            continue
        for key, value in source.items():
            if value is not None:
                merged[key] = value
    return merged


__all__ = [
    "CUMULATIVE_FACTOR_BOUNDS",
    "Factor",
    "LOG",
    "PACK_VERSION",
    "PROJECTION_FIELDS",
    "Z_CLIP",
    "add_to_field",
    "adjustment_factor",
    "blend_man_zone",
    "clamp",
    "describe_drivers",
    "guard_cumulative",
    "log_result",
    "logistic_probability",
    "logit",
    "merge_inputs",
    "r",
    "read_input",
    "run_factor_model",
    "scale_field",
    "score_factors",
    "sigmoid",
    "signed_unit",
    "to_float",
    "to_rate",
]
