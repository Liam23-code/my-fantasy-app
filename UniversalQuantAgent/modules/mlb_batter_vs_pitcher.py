"""MLB batter-vs-pitcher matchup adjustments: handedness, pitch-type effectiveness, career BvP, contact quality.

One of five DFS matchup modules feeding modules/mlb_fusion_model.py (see
mlb_matchup_engine.md). Every function here is pure math over a real,
already-known player-profile dict the caller supplies (from an uploaded
lineup/BvP file, or a future generator) -- nothing here fetches or
fabricates a statistic. Every adjustment is a disclosed, bounded
multiplier around 1.0 (neutral), not a full reprojection.
"""
from __future__ import annotations

from typing import Any

#: Platoon advantage: a batter facing an opposite-handed pitcher has a
#: real, well-documented edge (the ball's break is easier to read); same
#: handed match up favors the pitcher. A switch-hitter always bats from
#: the advantageous side, so is treated as neutral-to-favorable. Modest,
#: disclosed, capped constants -- not fitted coefficients.
_PLATOON_ADVANTAGE = 1.08
_PLATOON_DISADVANTAGE = 0.94
_NEUTRAL = 1.0

#: Below this many real BvP at-bats, the sample is too small to trust on
#: its own -- heavily regressed toward neutral (classic small-sample BvP
#: caution; see e.g. any sabermetric treatment of "clutch" BvP narratives).
_BVP_FULL_SAMPLE_AB = 40.0


def handedness_matchup_multiplier(batter_hand: str, pitcher_hand: str) -> float:
    """Platoon-advantage multiplier for a batter's plate appearance against this pitcher's hand."""
    batter_hand = (batter_hand or "").strip().upper()[:1]
    pitcher_hand = (pitcher_hand or "").strip().upper()[:1]
    # A switch-hitter always bats from the advantageous side; an unknown
    # hand on either side leaves the matchup un-modeled -- both are
    # treated as neutral rather than guessed.
    if batter_hand == "S" or not batter_hand or not pitcher_hand:
        return _NEUTRAL
    return _PLATOON_ADVANTAGE if batter_hand != pitcher_hand else _PLATOON_DISADVANTAGE


def pitch_type_effectiveness(pitcher_pitch_mix: dict[str, float], batter_whiff_rates: dict[str, float]) -> float:
    """A matchup whiff-rate index: the batter's own whiff rate against each pitch type, weighted by usage.

    ``pitcher_pitch_mix`` is ``{"fastball": 0.55, "slider": 0.25, ...}``
    (usage shares, real or assumed to sum to ~1.0); ``batter_whiff_rates``
    is the batter's own real whiff rate against each pitch type. Returns
    the batter's real weighted-average whiff rate against *this* pitcher's
    actual mix -- higher means a tougher matchup for the batter.
    """
    total_usage = sum(pitcher_pitch_mix.values()) or 1.0
    weighted = sum(usage * float(batter_whiff_rates.get(pitch, 0.25)) for pitch, usage in pitcher_pitch_mix.items())
    return round(weighted / total_usage, 4)


def career_bvp_adjustment(bvp_record: dict[str, Any]) -> float:
    """A small, sample-size-regressed multiplier from real career batter-vs-pitcher history.

    ``bvp_record`` carries real ``"at_bats"`` and ``"hits"`` (plus
    optionally ``"home_runs"``). BvP samples are almost always tiny (a few
    dozen at-bats over years), so this is heavily regressed toward neutral
    -- even a real .400 BvP average over 10 at-bats should barely move a
    projection, per standard sabermetric practice.
    """
    at_bats = float(bvp_record.get("at_bats") or 0.0)
    if at_bats <= 0:
        return _NEUTRAL
    hits = float(bvp_record.get("hits") or 0.0)
    bvp_avg = hits / at_bats
    reliability = min(1.0, at_bats / _BVP_FULL_SAMPLE_AB)
    league_avg = 0.250
    blended_avg = reliability * bvp_avg + (1.0 - reliability) * league_avg
    return round(max(0.85, min(1.15, blended_avg / league_avg)), 4)


def hard_hit_barrel_index(hard_hit_rate: float, barrel_rate: float) -> float:
    """A composite quality-of-contact multiplier from real hard-hit% and barrel% (league-average = 1.0).

    League-average hard-hit rate is ~35%, barrel rate is ~8% (published
    Statcast norms) -- used only as the neutral center point for this
    multiplier, not asserted as this player's own rate.
    """
    league_hard_hit, league_barrel = 0.35, 0.08
    hard_hit_component = (float(hard_hit_rate) / league_hard_hit) if league_hard_hit else 1.0
    barrel_component = (float(barrel_rate) / league_barrel) if league_barrel else 1.0
    return round(max(0.7, min(1.4, 0.5 * hard_hit_component + 0.5 * barrel_component)), 4)


def matchup_multiplier(batter_profile: dict[str, Any], pitcher_profile: dict[str, Any]) -> dict[str, Any]:
    """Combine every batter-vs-pitcher signal into one bounded multiplier plus its components.

    ``batter_profile``: ``{"hand", "whiff_rates", "hard_hit_rate", "barrel_rate", "bvp": {...}}``.
    ``pitcher_profile``: ``{"hand", "pitch_mix"}``. Any missing signal
    contributes a neutral 1.0 rather than raising -- a partial profile is
    a normal state, not an error.
    """
    handedness = handedness_matchup_multiplier(batter_profile.get("hand", ""), pitcher_profile.get("hand", ""))
    pitch_mix = pitcher_profile.get("pitch_mix")
    whiff_rates = batter_profile.get("whiff_rates")
    contact_difficulty = pitch_type_effectiveness(pitch_mix, whiff_rates) if pitch_mix and whiff_rates else 0.25
    # A higher matchup whiff rate is *bad* for the batter -- invert around
    # the league-average ~0.25 whiff rate so the combined multiplier stays
    # in the same "higher = better for the batter" direction as the rest.
    contact_component = round(max(0.8, min(1.2, 0.25 / max(contact_difficulty, 0.05))), 4)
    bvp = career_bvp_adjustment(batter_profile.get("bvp") or {})
    quality = hard_hit_barrel_index(batter_profile.get("hard_hit_rate", 0.35), batter_profile.get("barrel_rate", 0.08))

    combined = round(handedness * contact_component * bvp * quality, 4)
    combined = max(0.6, min(1.6, combined))
    return {
        "combined_multiplier": combined,
        "handedness_multiplier": handedness,
        "matchup_whiff_rate": contact_difficulty,
        "contact_component": contact_component,
        "bvp_adjustment": bvp,
        "contact_quality_index": quality,
    }
