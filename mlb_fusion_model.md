# MLB fusion model: blending the season baseline with the DFS matchup layer

How `modules/mlb_fusion_model.py` combines [mlb_pipeline.md](mlb_pipeline.md)'s season-average baseline with [mlb_matchup_engine.md](mlb_matchup_engine.md)'s five matchup modules into one final per-category projection.

## The six named weights

The build spec asked for six specific weighting factors; `WEIGHTS` (in `modules/mlb_fusion_model.py`) names all six and each plays a distinct, disclosed role:

```python
WEIGHTS = {
    "reliability": 0.6,          # \\ together set how much of the blended
    "sample_size": 0.4,          # / matchup signal is trusted at all
    "matchup_difficulty": 0.40,  # \\
    "park_factor": 0.25,         #  \  relative importance of each of the
    "pitcher_quality": 0.25,     #  /  four matchup-layer multipliers
    "lineup_protection": 0.10,   # /
}
```

**`matchup_difficulty` / `park_factor` / `pitcher_quality` / `lineup_protection`** (sum to `1.0`) are the relative importance of the four matchup-layer inputs when they're blended into one signed adjustment delta (`combined_matchup_adjustment`). Each input is centered at `1.0` (neutral); its distance from `1.0`, scaled by its weight, is its contribution — a `1.2` matchup-difficulty multiplier with weight `0.40` contributes `+0.08` to the raw delta, a `0.9` park factor with weight `0.25` contributes `-0.025`, and so on.

**`reliability` / `sample_size`** (sum to `1.0`) set `confidence_scale`: how much of that raw blended delta is actually trusted enough to move the season baseline. A player with a highly reliable, large-sample season baseline (see `modules/mlb_season_model.py::reliability_score`, itself a function of real games played against a category-specific stabilization point) is adjusted by nearly the full matchup signal; a player with a thin season sample gets a damped adjustment. This is a documented modeling choice, not a fitted coefficient — stacking an uncertain matchup read on top of an already-uncertain season baseline would compound noise rather than add real information, so this fusion model chooses to damp rather than amplify in that case. A deployment with a different view can override `WEIGHTS` directly.

## The formula

For each of the 7 modeled categories:

```
adjustment_delta_raw = 0.40*(matchup_difficulty - 1) + 0.25*(park_factor - 1)
                      + 0.25*(pitcher_quality - 1)   + 0.10*(lineup_protection - 1)

confidence_scale = 0.6 * reliability + 0.4 * sample_size_confidence(games_played)

final_projection = stabilized_season_mean * (1 + adjustment_delta_raw * confidence_scale)
```

All-neutral matchup inputs (every multiplier `= 1.0`) always leave `adjustment_delta_raw = 0`, so the final projection exactly equals the season baseline regardless of confidence — a fully-untested code path is inert, not silently wrong. `sample_size_confidence` is a plain `games_played / 60` read (capped at `1.0`), deliberately independent of any one category's own stabilization point (see `mlb_season_model.py`'s per-category `STABILIZATION_GAMES`) — it's a second, coarser sample-size signal, not a duplicate of the season model's own reliability.

## Risk tier

Each category's output also gets a `risk_tier`, derived from a `pseudo_cv = 1.0 - overall_confidence` fed through the same `betting.prop_model._risk_tier` classification every other sport's engine reuses directly (not re-implemented) — a low-confidence projection (thin sample, unreliable matchup context) is `"high"` risk the same way a high-coefficient-of-variation prop is elsewhere in this codebase.

## Where this fits: optional overlay, not the contract

`mlb_fusion_model.fuse_projection` is never called by `modules/unified_betting_contract.py::compute_ev` directly. `modules/mlb_prop_model.py` (the contract-required evaluator) prices whatever `"line"` a prop row already carries; `compute_ev("MLB", props, matchup_multipliers={...})` accepts an *optional* per-player composite multiplier (typically the output of a `fuse_projection` call, collapsed to one number) the same way CBB's `minutes_by_player` context is optional. This mirrors NBA's `modules.nba_matchup_engine.compute_matchup_context` — an on-demand, real, bounded adjustment layer, not a hidden requirement every prop must pass through.

## Testing

`UniversalQuantAgent/tests/test_mlb_fusion.py` — the six named weights are present and the two weight groups each sum to `1.0`; neutral inputs produce a zero adjustment; favorable/unfavorable matchup inputs move the projection in the correct direction; a low-reliability player's adjustment is damped relative to a high-reliability player's under the identical matchup input; every category always returns a valid risk tier and a `0-1` confidence.
