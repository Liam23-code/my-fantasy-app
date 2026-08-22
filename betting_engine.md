# Betting engine reference: odds loader, EV, confidence, parlay

Covers both sports' fair-line evaluation stack: what's genuinely shared (imported directly, not duplicated) versus what's sport-specific and why. See [offline_data_contract.md](offline_data_contract.md) for the data-sourcing rules this engine follows, and the [Betting Engine page](UniversalQuantAgent/app/pages/30_Betting_Engine.py) for the UI that surfaces all of this with a sport toggle.

## Why NFL and NBA aren't one codebase

They share a *contract* (same shape of loader, same real-basis disclosure, same odds math) but not a codebase. NFL's engine (`fantasy_engine/betting/`) was built first, is fully tested, and its prop model is a from-scratch Gaussian/Poisson fit on raw season rates. NBA already had a materially richer, pre-existing per-player model (`fuse_projection` + `get_reliability_score`, with minutes/pace/matchup/injury context) built for a different purpose (season-long fantasy projections) before any of this betting work started. Forcing NBA onto NFL's simpler prop model would have been a real quality regression, not just a style choice — so NBA's props are priced by wrapping its *existing* richer model, not by reimplementing NFL's.

What genuinely is shared: anything with no sport in it is imported directly, never copy-pasted. What isn't shared: anything that encodes a real, sport-specific fact (a scoring-margin constant, which stat categories overlap, which stats indicate a shared game script).

## Odds loader

| | NFL | NBA |
|---|---|---|
| Player props | `betting.odds_loader.unified_odds()` | `modules.nba_props_loader.unified_props()` |
| Game odds (moneyline/spread/total) | same `unified_odds()` (games + props in one file) | `modules.nba_odds_loader.unified_game_odds()` (separate from props — see below) |
| Default file | `fantasy_engine/data/odds.json` (pre-generated, full week) | `data/nba_props.json` (pre-generated, season-long rates); `data/nba_game_odds.json` (empty — no fixed schedule to pre-populate) |
| Keying | `game_id`, `player_id:market` | `(player_name, category)`; games keyed by `away@home` matchup string (JSON-serializable form of the `(home, away)` pair) |

NBA game odds are a separate loader from NBA props (unlike NFL, which loads both from one file) because they have different lifetimes: props are season-long real per-game rates, stable enough to pre-generate once; a specific day's NBA matchups aren't known far enough in advance to pre-generate the way a full NFL week is — see [nba_pipeline.md](nba_pipeline.md).

## EV engine

**Fully shared, not duplicated.** `betting.odds_math` — `american_to_decimal`, `implied_probability`, `remove_vig_two_way`, `expected_value`, `edge_vs_fair`, `fair_price_from_probability` — has no sport-specific content; it is American-odds math on numbers the caller supplies. Both `betting.prop_model`/`betting.moneyline_model` (NFL) and `modules.nba_prop_model`/`modules.nba_moneyline_model` (NBA) import these functions directly from `fantasy_engine`'s `betting` package (importable from anywhere in this venv — see `fantasy_engine/pyproject.toml`'s editable install).

The one place NBA's EV computation genuinely differs: NBA props didn't originally carry over/under prices at all (`modules/nba_props_loader.py`'s row schema was extended to add `over_price`/`under_price`, defaulting to `-110.0` — same convention as NFL's `_DEFAULT_PRICE`), and NBA's probability distribution comes from `modules.nba_prop_model` treating `compare_props`'s existing `confidence_low`/`confidence_high` band as an approximate 90% interval rather than fitting a fresh Gaussian/Poisson model — see that module's docstring for why, and [offline_data_contract.md](offline_data_contract.md)'s rule 5 on disclosing the approximation.

## Confidence engine

**No new module was built for NBA — its existing, more sophisticated confidence infrastructure was reused instead of duplicated.** NFL's confidence is a simple inline blend in `betting.prop_model._confidence` (real-sample-size confidence, averaged with the ensemble's own confidence read). NBA already had `modules/reliability.py::get_reliability_score` (a full reliability model over recent variance, matchup difficulty, and injury impact) and `modules/props.py::prop_confidence` (confidence-band-width-based scoring) before this work started — both real signals, both richer than NFL's blend. `modules.nba_prop_model`'s priced rows pass through `compare_props`/`recommend_props`'s existing `confidence_score`/`reliability_score` fields unchanged; the parlay UI feeds `reliability_score / 100` in as each leg's `confidence`.

## Parlay engine

`betting.parlay_engine`'s leg construction, combined odds, correlation-adjusted probability math, and risk tiering (`make_leg`, `parlay_decimal_odds`, `correlation_adjusted_probability`, `_parlay_ev`, `_risk_tier`) are sport-agnostic and imported directly by `modules.nba_parlay_engine`, not duplicated. Only the *correlation pattern detector* has sport-specific content:

- **Reused unmodified for NBA:** the favorite/total pattern (`market == "moneyline"` correlating with `market == "total"`) — fully generic, fires for NBA moneyline+total legs exactly as built for NFL.
- **NFL-only, doesn't apply to NBA:** QB/pass-catcher stacks, RB volume+touchdown stacks (NFL stat-category sets).
- **NBA-only, new:** `modules.nba_parlay_engine.nba_detect_correlations` adds two NBA-specific patterns: same-player overlapping stat categories (a `PRA` leg structurally contains `points`/`rebounds`/`assists` — not independent), and same-team same-direction scoring stacks (two teammates' points legs, driven by the same game pace).

`modules.nba_parlay_engine.evaluate_parlay` calls `betting.parlay_engine.correlation_adjusted_probability(legs, correlations=nba_detect_correlations(legs))` — the generic engine's own top-level `evaluate_parlay` doesn't expose a way to inject a custom detector, so this is a thin thirty-line orchestration function, not a reimplementation of the underlying math.

## Team / moneyline model

Not shared, because the underlying facts genuinely differ: NBA scores are higher and more variable than NFL's, so NFL's published margin/total standard deviations (`NFL_MARGIN_STDEV = 13.5`, `NFL_TOTAL_STDEV = 10.0`) would be a real modeling error if reused for NBA. `modules.nba_team_model.real_margin_and_total_volatility` computes NBA's own standard deviations directly from this season's completed games (`margin_stdev`, `total_stdev` — see [nba_pipeline.md](nba_pipeline.md)) rather than assuming a published constant, which is more rigorous than the NFL side, not less.

What *is* reused: `betting.team_model.project_game` (the "your real scoring rate vs. opponent's real rate allowed" blend) takes no sport-specific constant at all — just team codes and an `averages` dict — so `modules.nba_moneyline_model.fair_moneyline` calls it directly, passing `modules.nba_team_model.team_scoring_averages`'s output (built to the identical `{"points_scored_avg", "points_allowed_avg", "games_played"}` shape NFL's `team_scoring_averages` already returns).
