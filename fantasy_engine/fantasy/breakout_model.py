"""breakout_model: a conservative, age-aware "will this player leap?" estimate.

Why this module exists
----------------------
The draft board used to surface :func:`quant.quant_engine.compute_breakout_probability`,
which on a stripped board row has no working age signal (every player defaults
to age 27), no usage-trend signal (target/carry counts are gone), and a logit
whose dominant terms -- ``1.8 * usage + 1.2 * efficiency`` -- just measure
*current production*. The result was 70-85% "breakout" probabilities pinned to
established 30-year-old WR1s. That is precisely backwards: a player who already
finished as a positional WR1 has, by definition, already broken out.

This model starts from a low base, adds a handful of *bounded* upside drivers,
and applies a hard age gate. It is deliberately pessimistic:

======================================  ==================
profile                                 breakout_probability
======================================  ==================
established veteran / plateaued star     0.05 - 0.12
ordinary role player                     0.08 - 0.18
genuine ascending young player           0.20 - 0.38
elite young breakout profile             0.40 - 0.55
rare (young + vacated role + efficiency) 0.55 - 0.65
======================================  ==================

``breakout_probability`` is always clamped to ``[0.05, 0.65]``. There is no
path to 70%+.

What it reads (all optional -- every field degrades gracefully)
--------------------------------------------------------------
``age`` / ``years_experience``
    The dominant gate. A real ``age`` gives a per-position youth curve *and*
    an explicit decline penalty (RB/WR 27-28 mild, 29-30 moderate, 31+ heavy;
    TE/QB softer). With only ``years_experience`` the youth curve is coarser
    and the probability is capped tighter (0.50) because the read is less sure.
    With neither, a mild neutral youth prior and a 0.45 cap.
``projection`` vs ``prior_season_points``
    Usage ascension: the forward projection already sees a player whose role
    is climbing. A big positive gap is the single strongest legitimate signal
    available on an offline board.
``prior_games_played`` (or ``games_played``) vs ``expected_games``
    Injury bounce-back: a player who missed most of last season but is
    projected for a full slate has real rebound room.
``adp``
    Market headroom: an early ADP means the market has already priced the
    player in, so there is little "surprise" left; a deep or missing ADP with
    a real projection means the market has not caught up.
``target_share`` / ``air_yards_share``
    A concrete role signal for pass catchers when the pool carries it.
``floor`` / ``ceiling`` (or ``volatility``)
    A wide outcome band adds a little breakout upside (and a little bust risk).
``prior_injury_status``
    A chronic designation (IR / PUP / NFI) damps the number.

Nothing here mutates its input. This is a leaf module: it imports only
:mod:`fantasy.utils` (and, lazily, :mod:`fantasy.player_status` is *not*
imported -- callers pass a resolved ``live_status`` string instead), so
importing it never drags in the rest of the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fantasy.utils import clamp, safe_float

#: Hard bounds on every returned probability. The upper bound is the whole
#: point of the rebuild: no established veteran can read as a 70% breakout.
MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.65

#: Age at which a position's youth upside has fully decayed. Mirrors
#: :data:`projections.projection_engine.POSITION_YOUTH_CEILING` so the two
#: models agree about when "young" ends.
YOUTH_CEILING: dict[str, float] = {
    "QB": 32.0,
    "RB": 26.0,
    "WR": 28.0,
    "TE": 29.0,
    "K": 30.0,
    "DST": 30.0,
}
_DEFAULT_YOUTH_CEILING = 28.0

#: Explicit age-decline penalties applied on top of the youth curve. RB/WR
#: fall off a cliff; TE/QB decline later and slower.
_SKILL_AGE_PENALTY: tuple[tuple[float, float], ...] = ((31.0, -0.18), (29.0, -0.10), (27.0, -0.04))
_PASSER_AGE_PENALTY: tuple[tuple[float, float], ...] = ((36.0, -0.14), (34.0, -0.08), (32.0, -0.03))

#: Typical season-average target share for a clear #1 option, per position --
#: the yardstick a real ``target_share`` is compared against.
_TARGET_SHARE_NORM: dict[str, float] = {"WR": 0.18, "TE": 0.14, "RB": 0.12}


def _as_mapping(player: Any) -> Mapping[str, Any]:
    if isinstance(player, Mapping):
        return player
    if hasattr(player, "model_dump") and callable(player.model_dump):
        dumped = player.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(player, "__dict__"):
        return vars(player)
    raise TypeError("player must be a mapping or an object with fields")


def _num(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] is not None:
            return safe_float(row[key], default)
    return default


def _position(row: Mapping[str, Any], override: str | None) -> str:
    position = str(override or row.get("position") or row.get("player_position") or "").strip().upper()
    return "DST" if position in {"DEF", "D/ST"} else position


def _age_penalty(position: str, age: float) -> float:
    bands = _PASSER_AGE_PENALTY if position in {"QB", "TE"} else _SKILL_AGE_PENALTY
    for threshold, penalty in bands:
        if age >= threshold:
            return penalty
    return 0.0


def _tier(probability: float) -> str:
    if probability >= 0.45:
        return "elite"
    if probability >= 0.30:
        return "strong"
    if probability >= 0.18:
        return "possible"
    return "unlikely"


def _result(probability: float, drivers: list[str], components: dict[str, float]) -> dict[str, Any]:
    probability = round(clamp(probability, MIN_PROBABILITY, MAX_PROBABILITY), 4)
    return {
        "breakout_probability": probability,
        "breakout_tier": _tier(probability),
        "drivers": drivers,
        "components": {key: round(value, 4) for key, value in components.items()},
    }


def compute_breakout_prob(
    player: Any,
    *,
    position: str | None = None,
    live_status: str | None = None,
) -> dict[str, Any]:
    """Estimate one player's breakout probability, bounded to ``[0.05, 0.65]``.

    ``player`` is any mapping (or object with fields) from the projection pool
    or the draft board. ``position`` overrides ``player["position"]`` when the
    row does not carry one. ``live_status`` is a resolved
    :mod:`fantasy.player_status` flag (``"OUT"`` / ``"HOLDOUT"`` / ...); when it
    is ``HOLDOUT`` or ``SUSPENDED`` the player's role for the coming season is
    in flux and the probability is floored.

    Returns ``{"breakout_probability", "breakout_tier", "drivers", "components"}``.
    Never raises on a sparse row and never mutates ``player``.
    """
    row = _as_mapping(player)
    pos = _position(row, position)

    status = str(live_status or "").strip().upper()
    if status in {"HOLDOUT", "SUSPENDED"}:
        return _result(
            MIN_PROBABILITY,
            [f"{status.title()} — role and touches in flux for the coming season"],
            {"status_floor": 1.0},
        )

    projection = _num(row, "projection", "expected_fantasy_points", "final_projection")
    prior_points = _num(row, "prior_season_points")
    prior_games = _num(row, "prior_games_played", "games_played", default=0.0)
    expected_games = _num(row, "expected_games", default=17.0)
    adp = _num(row, "adp")
    floor = _num(row, "floor")
    ceiling = _num(row, "ceiling")
    target_share = _num(row, "target_share", "air_yards_share")
    position_rank = _num(row, "position_rank")
    prior_injury = str(row.get("prior_injury_status") or "").strip().upper()

    age = _num(row, "age")
    experience = _num(row, "years_experience", "experience", "years_exp")
    drivers: list[str] = []
    components: dict[str, float] = {}

    # --- youth / experience gate --------------------------------------------
    ceiling_age = YOUTH_CEILING.get(pos, _DEFAULT_YOUTH_CEILING)
    if age >= 20.0:
        youth = clamp((ceiling_age - age) / 6.0, 0.0, 1.0)
        age_penalty = _age_penalty(pos, age)
        hard_cap = MAX_PROBABILITY
        if youth >= 0.5:
            drivers.append(f"{age:.0f} years old — still inside the {pos or 'position'} growth window")
        elif age_penalty <= -0.08:
            drivers.append(f"{age:.0f} years old — past the {pos or 'position'} decline curve")
    elif experience > 0.0:
        youth = {0.0: 0.9, 1.0: 0.9, 2.0: 0.72}.get(float(int(experience)), 0.5 if experience <= 4 else 0.22 if experience <= 6 else 0.05)
        age_penalty = 0.0
        hard_cap = 0.50
        if experience <= 2:
            drivers.append(f"only {experience:.0f} year(s) of NFL experience — role can still climb")
    else:
        youth = 0.35
        age_penalty = 0.0
        hard_cap = 0.45
    components["youth"] = youth
    components["age_penalty"] = age_penalty
    youth_bonus = 0.20 * youth

    # --- usage ascension: the forward projection already sees them rising ---
    if prior_points > 20.0:
        ascension = clamp((projection - prior_points) / max(prior_points, 40.0), -0.6, 1.0)
    else:
        # No meaningful prior season (limited role, or missed most of it) but a
        # real projection now -- a role has opened up.
        ascension = 0.6 if projection > 150.0 else 0.3 if projection > 100.0 else 0.0
    ascension_term = clamp(0.22 * max(ascension, 0.0), 0.0, 0.22)
    if ascension_term >= 0.10:
        drivers.append(f"projected {projection:.0f} pts vs {prior_points:.0f} last season — role trending up")

    if target_share > 0.0 and pos in _TARGET_SHARE_NORM:
        norm = _TARGET_SHARE_NORM[pos]
        share_edge = clamp(0.10 * (target_share / norm - 1.0), -0.06, 0.10)
        ascension_term = clamp(ascension_term + share_edge, 0.0, 0.28)
        if share_edge >= 0.03:
            drivers.append(f"{target_share:.0%} target share — a lead-option role")
    components["ascension"] = ascension_term

    # --- injury bounce-back -----------------------------------------------------
    bounce = 0.0
    if 0.0 < prior_games < 13.0 and expected_games >= 15.0:
        bounce = clamp(0.12 * (13.0 - prior_games) / 7.0, 0.0, 0.12)
        drivers.append(f"missed time last season ({prior_games:.0f} games) but projected for a full slate")
    if prior_injury in {"IR", "PUP", "NFI", "OUT"} and 0.0 < prior_games < 8.0:
        bounce = min(bounce, 0.06)
    components["injury_bounce"] = bounce

    # --- market headroom: has the market already priced the leap in? ----------
    if adp <= 0.0:
        headroom = 0.10 if projection > 130.0 else 0.04 if projection > 80.0 else 0.0
        if headroom >= 0.08:
            drivers.append("no market ADP yet — the field has not caught up to the projection")
    elif adp >= 90.0:
        headroom = 0.08
    elif adp >= 48.0:
        headroom = 0.04
    else:
        headroom = 0.0
    components["market_headroom"] = headroom

    # --- volatility kicker ---------------------------------------------------
    if ceiling > 0.0 and floor >= 0.0 and projection > 0.0:
        volatility = clamp((ceiling - floor) / max(projection, 1.0), 0.0, 1.2)
    else:
        volatility = clamp(_num(row, "volatility"), 0.0, 1.2)
    vol_term = clamp(0.04 * volatility, 0.0, 0.05)
    components["volatility"] = vol_term

    raw = 0.06 + youth_bonus + ascension_term + bounce + headroom + vol_term + age_penalty

    # --- established-star damper: already elite AND past the youth ceiling ---
    established = (0.0 < position_rank <= 5.0) or (0.0 < adp <= 24.0)
    if established and age >= ceiling_age > 0.0:
        raw *= 0.55
        drivers.append("already an established positional starter — limited leap room")
    elif established and age < 20.0 and experience >= 6.0:
        raw *= 0.70

    # --- chronic-injury damper --------------------------------------------------
    if prior_injury in {"IR", "PUP", "NFI"} and 0.0 < prior_games < 10.0:
        raw *= 0.78
        drivers.append(f"{prior_injury} designation last season — durability overhang")

    if not drivers:
        drivers.append("no distinguishing breakout signal — sits near the positional base rate")

    components["hard_cap"] = hard_cap
    probability = clamp(raw, MIN_PROBABILITY, hard_cap)
    return _result(probability, drivers, components)


def compute_breakout_prob_batch(
    players: Any,
    *,
    status_by_id: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run :func:`compute_breakout_prob` over a pool, keyed by ``str(player_id)``.

    ``status_by_id`` is an optional ``{player_id: live_status}`` map (e.g.
    :func:`fantasy.draft_state.build_draft_state`'s ``status_by_id``) so a
    holdout/suspension floors the number without this module importing the
    status overlay itself.
    """
    status_by_id = status_by_id or {}
    out: dict[str, dict[str, Any]] = {}
    for player in players or []:
        try:
            row = _as_mapping(player)
        except TypeError:
            continue
        key = str(row.get("player_id") or row.get("id") or "")
        if not key:
            continue
        out[key] = compute_breakout_prob(row, live_status=status_by_id.get(key))
    return out
