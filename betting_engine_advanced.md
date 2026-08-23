# Betting engine, advanced: correlation, risk tiers, cross-sport parlays

Builds on [betting_engine.md](betting_engine.md) (read that first for the shared-vs-sport-specific architecture). This covers what's new in the performance/polish cycle: the full correlation pattern list, unified risk tiering, and cross-sport parlays.

## Correlation patterns, complete list

Detected by `betting.parlay_engine.detect_correlations` (NFL), `modules.nba_parlay_engine.nba_detect_correlations` (NBA -- calls the NFL detector internally for the one shared pattern, then adds its own), and `modules.unified_parlay_engine.detect_cross_sport_correlations` (routes a mixed leg list to each sport's own detector, partitioned by sport -- see below).

| Pattern | Sport | Same... | Direction required | Rationale |
|---|---|---|---|---|
| `qb_pass_catcher_stack` | NFL | team | both `"over"` | same team, same passing-game script |
| `rb_volume_and_touchdown` | NFL | player | both `"over"` | same player, shared rushing volume drives both |
| `rb_volume_and_game_total` | NFL | game | both `"over"` | a team leaning on the run reflects a positive, clock-controlling game script, correlating with the total |
| `favorite_and_game_total_over` / `_under` | **shared** (generic market names) | game | moneyline favorite + total over/under | a favorite correlates with pace/control, which correlates with the total |
| `overlapping_stat_categories` | NBA | player | both `"over"` | `PRA` structurally contains `points`/`rebounds`/`assists` -- not independent |
| `teammate_scoring_stack` | NBA | team (different players) | both `"over"` | same team, same game pace drives both players' scoring |

Every adjustment uses the same fixed, disclosed, capped `CORRELATION_ADJUSTMENT = 0.12` (a modest relative bump/discount, not a fitted coefficient -- there's no offline source of real joint-outcome data to calibrate one against; see `betting.parlay_engine`'s module docstring). Adding a new pattern means adding a new `_pair_correlation`/`_nba_pair_correlation` branch and a same-shape "why" note -- not inventing a new adjustment magnitude.

### Why `favorite_and_game_total_*` needed no NBA-specific work

It's the one pattern in the table with no sport name in its `kind` string or its market check (`market_a == "moneyline"`, `market_b == "total"`) -- purely structural. NBA legs built with `market="moneyline"`/`market="total"` (the same market names NFL's game-evaluation layer uses) trigger it unmodified. `tests/test_nfl_nba_offline_market_parity.py::SharedEngineReuseTests::test_nba_parlay_correlation_detector_finds_the_same_generic_pattern_nfl_does` asserts this identity directly, not just similar behavior.

## Risk tiers, unified

One classification, reused everywhere: `betting.prop_model._risk_tier(cv)`, a coefficient-of-variation (stdev / mean) threshold table --

```python
_RISK_TIERS = ((0.35, "low"), (0.50, "medium"), (float("inf"), "high"))
```

- NFL: `betting.prop_model.evaluate_prop` computes `cv` from its Gaussian/Poisson distribution's own stdev/mean.
- NBA: `modules.nba_prop_model.price_prop_comparison` computes the same `cv` from its confidence-band-derived stdev divided by the model projection, then calls the *same imported function* -- `from betting.prop_model import _risk_tier`. This was a genuine gap closed this cycle (NBA's priced props previously had no risk tier at all); `tests/test_nba_prop_model.py::PricePropComparisonTests::test_risk_tier_present_and_uses_shared_nfl_classification` asserts the NBA result matches what calling `_risk_tier` directly on the same `cv` would produce.
- Parlays: both sports' `evaluate_parlay` use `betting.parlay_engine._risk_tier(num_legs, hit_probability)` -- a different, num-legs-and-hit-probability-based classification (not the same function as the prop-level one above; the name collision is between two genuinely different "risk" concepts -- a single prop's variance vs. a parlay's compounding leg count). `modules.unified_parlay_engine.evaluate_cross_sport_parlay` imports and calls this exact function too.

## Cross-sport parlays

`modules/unified_parlay_engine.py`. A leg is tagged with its sport via `make_unified_leg("NFL", **kwargs)` / `make_unified_leg("NBA", **kwargs)`, which forwards to that sport's own `make_leg` and adds a `"sport"` key -- the leg's shape is otherwise identical to a same-sport parlay leg, so all the existing generic math (`parlay_decimal_odds`, `correlation_adjusted_probability`, `_parlay_ev`, `_risk_tier`) keeps working unmodified on a mixed list.

The one new piece is `detect_cross_sport_correlations`: it partitions the leg list into NFL-tagged and NBA-tagged index groups, runs each sport's own detector against *only its own subset*, then remaps the returned local pair-indices back to the original mixed-list indices. An NFL leg and an NBA leg are never checked against each other by any pattern -- there is no cross-sport correlation pattern in this codebase, which is the *correct* assumption for two genuinely independent real-world events (an NFL game and an NBA game don't share a game script). This is asserted directly: `tests/test_unified_parlay_engine.py::DetectCrossSportCorrelationsTests::test_an_nfl_leg_and_an_nba_leg_are_never_correlated`.

```python
from modules.unified_parlay_engine import make_unified_leg, evaluate_cross_sport_parlay

legs = [
    make_unified_leg("NFL", description="...", model_probability=0.55, price=-110.0, market="passing_yards", side="over", team="KC"),
    make_unified_leg("NBA", description="...", model_probability=0.6, price=-120.0, market="points", side="over", player_id="p1"),
]
result = evaluate_cross_sport_parlay(legs)
# result["sports"] == ["NBA", "NFL"]; everything else matches either sport's own evaluate_parlay output shape
```

The Betting Engine page's **Cross-Sport Parlay** tab lazy-loads whichever sport isn't already loaded for the current sport-toggle position (behind a button, not on every render -- see [ui_design.md](ui_design.md) and [performance_notes.md](performance_notes.md) for why eager-loading both sports on every render was rejected).
