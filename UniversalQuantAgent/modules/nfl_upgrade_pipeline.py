"""NFL upgrade pipeline: compute every pack signal for one player and apply them in the brief's order.

This is the "main projection engine" seam for the NFL model upgrade pack. It is pure logic (no
Streamlit, no network): a caller supplies a *context* of canonical features (built by
:mod:`modules.nfl_matchup_features`, or hand-written) and a player's per-game stat line, and gets back the
adjusted line plus a full audit of what moved and why.

Order (as the brief requires)
-----------------------------
1. **Collapse** -- ``qb_collapse`` for a QB; ``pass_game_collapse`` (which consumes the QB collapse
   probability) for a WR / TE; ``run_game_collapse`` for an RB.
2. **Explosion** -- ``explosion_model`` (QB / WR / RB yards, team points).
3. **Coverage mismatch** then **explosive plays** -- WR / TE yards; WR / RB yards.
4. **Red-zone funnel** -- QB / RB / TE touchdown rates.
5. **Game script** -- QB and RB attempts.
6. **Parlay correlation** -- recalculated by the pages from :func:`team_signals` (see
   :mod:`modules.parlay_correlation`).

The game-script *probabilities* are computed before step 1 because ``run_game_collapse`` consumes the
trailing-script probability; their *application* stays at step 5.

What "neutral" means
--------------------
Every model returns exactly its reference probability for neutral or missing inputs, and every apply
function divides by the reference (``recenter=True``), so with no context -- or a context with no usable
features -- the returned line equals the input line. That is what keeps every existing number on every
existing page unchanged until a context is supplied.

Two things the brief does not specify, made explicit here
---------------------------------------------------------
* **Volume pass-through** (``propagate_volume=True``). The brief's game-script step changes ``qb_attempts``
  and ``rb_attempts`` only. Nothing downstream bets on attempts, so by itself that step would be
  cosmetic. When on (default), the *team pass-volume factor* (the ``qb_attempts`` multiplier) also scales a
  QB's passing yards / TDs / interceptions and every receiver's receiving yards / TDs / receptions, and the
  *team rush-volume factor* (the ``rb_attempts`` multiplier) scales a RB's rushing yards / TDs, at
  elasticity 1 (a yard is attempts x yards-per-attempt). It is audited as ``volume_passthrough`` rows and
  switched off with ``propagate_volume=False``.
* **Cumulative guard** (``guard=True``). Five mild multiplicative stages compound. No field is allowed to
  leave ``CUMULATIVE_FACTOR_BOUNDS`` (0.40x-2.00x) of where it started; a clamp is reported in ``guard``.

HONEST LIMITS
-------------
* The brief's coefficients are applied as written; they are priors, not fitted (see each model).
* The engine's baseline already contains average collapse / explosion / script variation, which is why
  everything is re-centred. It does *not* contain opponent-specific pressure or line information, so the
  pack adds signal there -- but a baseline that already leans on a matchup score (``context_factor``) can
  overlap with the coverage / explosion stages.
* ``team_points`` is moved by the explosion stage only (the brief gives no collapse coefficient for points).
* ``rb_yards`` is rushing yards; an RB's receiving line moves only through volume pass-through.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from modules import (
    coverage_mismatch_model,
    explosion_model,
    explosive_play_model,
    game_script_model,
    pass_game_collapse,
    qb_collapse_model,
    red_zone_funnel_model,
    run_game_collapse,
)
from modules.nfl_pack_common import (
    CUMULATIVE_FACTOR_BOUNDS,
    PACK_VERSION,
    guard_cumulative,
    merge_inputs,
    r,
    to_float,
)

LOG = logging.getLogger("nfl_pack.pipeline")

POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")
_POSITION_ALIASES = {"HB": "RB", "FB": "RB", "TB": "RB", "WR/TE": "WR"}

#: Stage names in the order the pipeline applies them (collapse names are position-specific).
STAGE_ORDER: tuple[str, ...] = (
    "collapse",
    "explosion",
    "coverage_mismatch",
    "explosive_play",
    "red_zone_funnel",
    "game_script",
)

#: Engine stat-line field -> pack field, per position (docs/nfl_model_upgrade_pack.md section 8).
LINE_TO_PACK: dict[str, dict[str, str]] = {
    "QB": {
        "passing_yards": "qb_yards",
        "completion_pct": "qb_completion_pct",
        "passing_tds": "qb_td_prob",
        "interceptions": "qb_int_prob",
        "attempts": "qb_attempts",
        "team_points": "team_points",
    },
    "RB": {
        "rushing_yards": "rb_yards",
        "rushing_tds": "rb_td_prob",
        "carries": "rb_attempts",
        "team_points": "team_points",
    },
    "WR": {"receiving_yards": "wr_yards", "receiving_tds": "wr_td_prob", "team_points": "team_points"},
    "TE": {"receiving_yards": "te_yards", "receiving_tds": "te_td_prob", "team_points": "team_points"},
}

#: Fields volume pass-through scales, per position: ``(field, "pass" | "rush")``.
VOLUME_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "QB": (("passing_yards", "pass"), ("passing_tds", "pass"), ("interceptions", "pass")),
    "RB": (
        ("rushing_yards", "rush"), ("rushing_tds", "rush"),
        ("receiving_yards", "pass"), ("receiving_tds", "pass"), ("receptions", "pass"), ("targets", "pass"),
    ),
    "WR": (("receiving_yards", "pass"), ("receiving_tds", "pass"), ("receptions", "pass"), ("targets", "pass")),
    "TE": (("receiving_yards", "pass"), ("receiving_tds", "pass"), ("receptions", "pass"), ("targets", "pass")),
}

#: Context sections merged (left to right, later wins) into the flat model inputs.
CONTEXT_SECTIONS: tuple[str, ...] = ("offense", "opponent_defense", "qb", "game", "player")

#: Flat probabilities the pages and the parlay layer read, by the brief's names.
SIGNAL_KEYS: tuple[str, ...] = (
    "collapse_probability",
    "pass_game_collapse",
    "run_collapse",
    "explosion_probability",
    "coverage_mismatch_factor",
    "explosive_play_prob_pass",
    "explosive_play_prob_run",
    "rz_pass_funnel",
    "rz_run_funnel",
    "positive_script_prob",
    "negative_script_prob",
    "neutral_script_prob",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def normalize_position(position: Any) -> str | None:
    """``QB`` / ``RB`` / ``WR`` / ``TE`` for any common spelling, else ``None``."""
    text = str(position or "").strip().upper()
    text = _POSITION_ALIASES.get(text, text)
    return text if text in POSITIONS else None


def model_inputs(context: Mapping[str, Any] | None, *, player_features: Mapping[str, Any] | None = None,
                 overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Flat canonical inputs for the models from a context (or an already-flat mapping).

    A context is a mapping with any of the sections in :data:`CONTEXT_SECTIONS`; a mapping with none of them
    is treated as already flat. ``player_features`` and ``overrides`` (user-supplied numbers) win over the
    context, in that order. ``None`` / non-numeric values never erase a value.
    """
    if not context and not player_features and not overrides:
        return {}
    flat: dict[str, Any] = {}
    if context:
        if any(isinstance(context.get(section), Mapping) for section in CONTEXT_SECTIONS):
            flat = merge_inputs(*[context.get(section) for section in CONTEXT_SECTIONS if isinstance(context.get(section), Mapping)])
        else:
            flat = dict(context)
    return merge_inputs(flat, player_features, overrides)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _flat_probabilities(results: Mapping[str, Mapping[str, Any] | None]) -> dict[str, float]:
    out: dict[str, float] = {}
    for result in results.values():
        if not result:
            continue
        for key in SIGNAL_KEYS:
            if key in result and key not in out:
                value = to_float(result[key])
                if value is not None:
                    out[key] = value
    return out


def compute_game_signals(
    context: Mapping[str, Any] | None,
    *,
    position: str | None = None,
    player_features: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    wr_role: str = "wr",
) -> dict[str, Any]:
    """Every model's result for one player in one game, in dependency order.

    ``position`` may be ``None`` for a team-level read (:func:`team_signals`). ``wr_role`` is the
    coverage-mismatch role for a WR (``"wr"`` or ``"slot"``). Never raises on bad data.
    """
    pos = normalize_position(position)
    inputs = model_inputs(context, player_features=player_features, overrides=overrides)

    # script first: run_game_collapse consumes the trailing probability, but it is APPLIED last.
    script = game_script_model.compute_game_script(inputs)
    qb = qb_collapse_model.compute_qb_collapse(inputs)
    results: dict[str, Mapping[str, Any] | None] = {"game_script": script, "qb_collapse": qb}

    # A derived input is handed on only when the model it came from had real evidence: a reference value
    # passed through would be counted as "observed" and make an empty context look like it carried signal.
    pass_inputs = dict(inputs)
    if (qb.get("coverage") or 0.0) > 0.0:
        pass_inputs["qb_collapse_probability"] = qb["collapse_probability"]
    run_inputs = dict(inputs)
    if (script.get("coverage") or 0.0) > 0.0:
        run_inputs["trailing_script_prob"] = script["trailing_script_prob"]
    results["pass_game_collapse"] = pass_game_collapse.compute_pass_game_collapse(pass_inputs)
    results["run_game_collapse"] = run_game_collapse.compute_run_game_collapse(run_inputs)
    results["explosion"] = explosion_model.compute_explosion(inputs)
    results["explosive_play"] = explosive_play_model.compute_explosive_play(inputs)
    results["red_zone_funnel"] = red_zone_funnel_model.compute_red_zone_funnel(inputs)
    role = "te" if pos == "TE" else ("slot" if str(wr_role).lower() == "slot" else "wr")
    results["coverage_mismatch"] = coverage_mismatch_model.compute_coverage_mismatch(inputs, role=role)

    probabilities = _flat_probabilities(results)
    defaulted = sorted({k for res in results.values() if res for k in res.get("inputs_defaulted", ())})
    coverage = {name: res.get("coverage") for name, res in results.items() if res}
    LOG.debug(
        "signals pos=%s collapse=%.3f pass_game=%.3f run=%.3f explosion=%.3f mismatch=%+.3f pass_explosive=%.3f "
        "run_explosive=%.3f script(+/-/0)=%.3f/%.3f/%.3f",
        pos, probabilities.get("collapse_probability", float("nan")), probabilities.get("pass_game_collapse", float("nan")),
        probabilities.get("run_collapse", float("nan")), probabilities.get("explosion_probability", float("nan")),
        probabilities.get("coverage_mismatch_factor", float("nan")), probabilities.get("explosive_play_prob_pass", float("nan")),
        probabilities.get("explosive_play_prob_run", float("nan")), probabilities.get("positive_script_prob", float("nan")),
        probabilities.get("negative_script_prob", float("nan")), probabilities.get("neutral_script_prob", float("nan")),
    )
    return {
        "position": pos,
        "pack_version": PACK_VERSION,
        "inputs": inputs,
        "results": results,
        "probabilities": probabilities,
        "coverage": coverage,
        "inputs_defaulted": defaulted,
        "has_signal": any(res and (res.get("coverage") or 0.0) > 0.0 for res in results.values()),
    }


def team_signals(context: Mapping[str, Any] | None, *, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Position-agnostic read for one offence in one game (what the parlay layer correlates).

    Returns ``{"probabilities", "references", "inputs_defaulted", "has_signal", "vegas_spread"}`` where
    ``probabilities`` carries the brief's names (:data:`SIGNAL_KEYS`) and ``references`` the matching
    league-average values, so a consumer can read each signal relative to neutral.
    """
    signals = compute_game_signals(context, position=None, overrides=overrides)
    references: dict[str, float] = {}
    for result in signals["results"].values():
        if not result:
            continue
        for key, value in (result.get("references") or {}).items():
            number = to_float(value)
            if number is not None and key in SIGNAL_KEYS and key not in references:
                references[key] = number
    # the script model references its three outputs; a signed factor's neutral is 0
    for key in ("coverage_mismatch_factor", "rz_pass_funnel", "rz_run_funnel"):
        references.setdefault(key, 0.0)
    return {
        "probabilities": signals["probabilities"],
        "references": references,
        "inputs_defaulted": signals["inputs_defaulted"],
        "has_signal": signals["has_signal"],
        "vegas_spread": to_float(signals["inputs"].get("vegas_spread")),
    }


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------
def _to_pack(line: Mapping[str, Any], position: str) -> dict[str, float]:
    pack: dict[str, float] = {}
    for engine_key, pack_key in LINE_TO_PACK[position].items():
        value = to_float(line.get(engine_key))
        if value is not None:
            pack[pack_key] = value
    if position == "QB" and "qb_completion_pct" not in pack:
        completions, attempts = to_float(line.get("completions")), to_float(line.get("attempts"))
        if completions is not None and attempts is not None and attempts > 0:
            pack["qb_completion_pct"] = completions / attempts
    return pack


def _applied(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(entry) for entry in entries if entry.get("applied")]


def apply_upgrade_pack(
    line: Mapping[str, Any],
    signals: Mapping[str, Any],
    *,
    position: str,
    recenter: bool = True,
    guard: bool = True,
    propagate_volume: bool = True,
) -> dict[str, Any]:
    """Apply every stage, in the brief's order, to an engine stat line.

    ``line`` uses engine field names (``passing_yards``, ``receiving_yards``, ``carries``, ``team_points``,
    ...); fields it lacks are skipped. Returns ``{"line", "original", "adjustments", "guard", "summary",
    "stage_order", "recentered", "propagate_volume", "warnings", "neutral"}``; the input mapping is never
    mutated.
    """
    pos = normalize_position(position)
    original = dict(line)
    if pos is None:
        return _result(original, original, [], [], [], recenter, propagate_volume,
                       [f"unknown position {position!r}: nothing applied"])
    results = signals.get("results") or {}
    pack = _to_pack(line, pos)
    adjustments: list[dict[str, Any]] = []
    stage_order: list[str] = []

    def run(stage: str, fn, result_name: str) -> None:
        nonlocal pack
        result = results.get(result_name)
        if not result:
            return
        pack, entries = fn(pack, result, recenter=recenter)
        kept = _applied(entries)
        adjustments.extend(kept)
        stage_order.append(stage)
        moved = {e["field"]: e.get("factor", e.get("delta")) for e in kept}
        if moved and LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("stage %-18s %s", stage, moved)

    # 1. collapse
    if pos == "QB":
        run("qb_collapse", qb_collapse_model.apply_qb_collapse, "qb_collapse")
    elif pos in ("WR", "TE"):
        run("pass_game_collapse", pass_game_collapse.apply_pass_game_collapse, "pass_game_collapse")
    else:
        run("run_game_collapse", run_game_collapse.apply_run_game_collapse, "run_game_collapse")
    # 2. explosion
    run("explosion", explosion_model.apply_explosion, "explosion")
    # 3. coverage mismatch, then explosive plays
    if pos in ("WR", "TE"):
        run("coverage_mismatch", coverage_mismatch_model.apply_coverage_mismatch, "coverage_mismatch")
    run("explosive_play", explosive_play_model.apply_explosive_play, "explosive_play")
    # 4. red-zone funnel
    run("red_zone_funnel", red_zone_funnel_model.apply_red_zone_funnel, "red_zone_funnel")
    # 5. game script
    run("game_script", game_script_model.apply_game_script, "game_script")

    out = dict(original)
    inverse = {pack_key: engine_key for engine_key, pack_key in LINE_TO_PACK[pos].items()}
    for pack_key, value in pack.items():
        engine_key = inverse.get(pack_key)
        if engine_key is None:
            continue
        if engine_key == "completion_pct" and "completion_pct" not in original:
            continue  # derived input: written back below as completions
        out[engine_key] = value
    if pos == "QB" and "qb_completion_pct" in pack:
        if "completion_pct" in original:
            out["completion_pct"] = pack["qb_completion_pct"]
        attempts = to_float(out.get("attempts"))
        if "completions" in original and attempts is not None:
            out["completions"] = pack["qb_completion_pct"] * attempts

    # 5b. volume pass-through (not in the brief; see module docstring)
    if propagate_volume and results.get("game_script"):
        factors, _ = game_script_model.apply_game_script({"qb_attempts": 1.0, "rb_attempts": 1.0}, results["game_script"],
                                                          recenter=recenter)
        scale = {"pass": factors["qb_attempts"], "rush": factors["rb_attempts"]}
        for field, channel in VOLUME_FIELDS[pos]:
            value = to_float(out.get(field))
            if value is None or abs(scale[channel] - 1.0) < 1e-12:
                continue
            out[field] = value * scale[channel]
            adjustments.append({
                "stage": "volume_passthrough", "field": field, "applied": True, "before": r(value, 4),
                "after": r(value * scale[channel], 4), "factor": r(scale[channel], 4), "channel": channel,
            })
        stage_order.append("volume_passthrough")

    # cumulative guard on the engine fields that moved
    guard_notes: list[dict[str, Any]] = []
    warnings: list[str] = []
    if guard:
        movable = [e for e in dict.fromkeys(entry["field"] for entry in adjustments)]
        engine_fields = [inverse.get(f, f) for f in movable]
        guard_notes = guard_cumulative(original, out, engine_fields, bounds=CUMULATIVE_FACTOR_BOUNDS)
        for note in guard_notes:
            warnings.append(
                f"{note['field']}: combined adjustments would have moved it to {note['unguarded']}; "
                f"limited to {note['guarded']} ({note['bounds'][0]:.2f}x-{note['bounds'][1]:.2f}x of the starting value)."
            )
    return _result(original, out, adjustments, guard_notes, stage_order, recenter, propagate_volume, warnings)


def _result(original, out, adjustments, guard_notes, stage_order, recenter, propagate_volume, warnings) -> dict[str, Any]:
    summary: dict[str, dict[str, float | None]] = {}
    for key, new in out.items():
        old = to_float(original.get(key))
        new_f = to_float(new)
        if old is None or new_f is None or old == new_f:
            continue
        summary[key] = {"before": r(old, 4), "after": r(new_f, 4), "factor": r(new_f / old, 4) if old else None}
    return {
        "line": out,
        "original": original,
        "adjustments": adjustments,
        "guard": guard_notes,
        "summary": summary,
        "stage_order": stage_order,
        "recentered": bool(recenter),
        "propagate_volume": bool(propagate_volume),
        "warnings": warnings,
        "neutral": not summary,
    }


def project_with_pack(
    line: Mapping[str, Any],
    *,
    position: str,
    context: Mapping[str, Any] | None,
    player_features: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    recenter: bool = True,
    guard: bool = True,
    propagate_volume: bool = True,
    wr_role: str = "wr",
) -> dict[str, Any]:
    """Convenience: :func:`compute_game_signals` then :func:`apply_upgrade_pack`.

    The result is :func:`apply_upgrade_pack`'s dict plus ``signals`` and ``explanations``. With ``context``
    ``None`` (or empty) the line comes back unchanged and ``neutral`` is ``True``.
    """
    signals = compute_game_signals(context, position=position, player_features=player_features,
                                   overrides=overrides, wr_role=wr_role)
    applied = apply_upgrade_pack(line, signals, position=position, recenter=recenter, guard=guard,
                                 propagate_volume=propagate_volume)
    applied["signals"] = signals
    applied["explanations"] = explain(signals, position=position)
    return applied


# ---------------------------------------------------------------------------
# Plain-English explanation
# ---------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def explain(signals: Mapping[str, Any], *, position: str | None = None, top_drivers: int = 2) -> list[str]:
    """One short sentence per relevant signal, with the league-average comparison and the top drivers."""
    pos = normalize_position(position)
    results = signals.get("results") or {}
    lines: list[str] = []

    def drivers(name: str) -> str:
        found = (results.get(name) or {}).get("drivers") or []
        return (" Biggest drivers: " + "; ".join(found[:top_drivers]) + ".") if found else ""

    def prob_line(name: str, key: str, label: str, relevant: bool = True) -> None:
        res = results.get(name)
        if not res or not relevant:
            return
        value, ref = to_float(res.get(key)), to_float((res.get("references") or {}).get(key))
        if value is None:
            return
        gap = "" if ref is None else f" (league average {_pct(ref)})"
        lines.append(f"{label}: {_pct(value)}{gap}.{drivers(name)}")

    prob_line("qb_collapse", "collapse_probability", "QB collapse risk", pos in (None, "QB", "WR", "TE"))
    prob_line("pass_game_collapse", "pass_game_collapse", "Pass-game collapse risk", pos in (None, "WR", "TE"))
    prob_line("run_game_collapse", "run_collapse", "Run-game collapse risk", pos in (None, "RB"))
    prob_line("explosion", "explosion_probability", "Explosion chance", True)
    mismatch = results.get("coverage_mismatch")
    if mismatch and pos in ("WR", "TE") and to_float(mismatch.get("coverage_mismatch_factor")) is not None:
        factor = float(mismatch["coverage_mismatch_factor"])
        if abs(factor) > 0.005:
            lines.append(f"Coverage matchup: {factor:+.2f} (-1 worst, +1 best).{drivers('coverage_mismatch')}")
    script = results.get("game_script")
    if script:
        lines.append(
            f"Game script: {_pct(script.get('trailing_script_prob'))} of plays trailing (pass-heavy), "
            f"{_pct(script.get('leading_script_prob'))} leading (run-heavy), {_pct(script.get('neutral_script_prob'))} neutral."
        )
    if not signals.get("has_signal"):
        lines.append("No matchup features were available, so the upgrade pack left this projection unchanged.")
    return lines


__all__ = [
    "CONTEXT_SECTIONS",
    "LINE_TO_PACK",
    "POSITIONS",
    "SIGNAL_KEYS",
    "STAGE_ORDER",
    "VOLUME_FIELDS",
    "apply_upgrade_pack",
    "compute_game_signals",
    "explain",
    "model_inputs",
    "normalize_position",
    "project_with_pack",
    "team_signals",
]

