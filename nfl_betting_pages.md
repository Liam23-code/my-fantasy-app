# NFL betting pages: MoneyLines, Player Props, Parlays

Three Streamlit pages that price an NFL market against a sportsbook number the user pastes, in the Midnight Circuit theme. They sit first in the **Betting** nav group (top rail "Betting" and the Home card both land on MoneyLines) and sit *beside* the older pages (`30_NFL_Betting.py` and friends), not instead of them. Nothing here fetches a sportsbook and nothing places a bet. See [ui_betting_tabs.md](ui_betting_tabs.md) for the older per-sport pages and [betting_engine.md](betting_engine.md) for the `fantasy_engine/betting` math these pages call.

## Where things live

| Piece | File | Role |
|---|---|---|
| MoneyLines page | `UniversalQuantAgent/app/pages/moneylines.py` | Week dropdown, slate of game tiles, nine-measure analysis, price check |
| Player Props page | `UniversalQuantAgent/app/pages/playerprops.py` | Team -> player -> stat, projection, line check, distribution chart |
| Parlays page | `UniversalQuantAgent/app/pages/parlays.py` | Moneyline / spread / prop legs, correlation, combined probability, EV, risk tier |
| Navigation | `UniversalQuantAgent/app/app.py` (`main()`, `betting_pages`) | The three `st.Page` entries; `pages/Home.py` links to MoneyLines |
| Paste-a-price helper | `modules/odds_input.py` | Parses `-150`, `+120`, `1.91`, `EVEN`; implied probability, value gap, EV, verdict |
| MoneyLines engine | `modules/nfl_moneyline_analysis.py` | Schedule, blended team ratings, QB impact, `analyze_game`, `evaluate_moneyline_price` |
| Player Props engine | `modules/nfl_prop_analysis.py` | Eligibility, `project_stat`, `evaluate_line`, `validate_inputs`, reasoning |
| Parlay engine | `modules/nfl_parlay_builder.py` | Leg specs, pairwise correlation rules, Gaussian-copula joint probability, risk tier |
| Component kit | `app/betting_ui.py` | Matchup card, tiles, meters, reasoning panel, slip, correlation grid, widget theming |

The pages are thin: every number comes from a module, none is recomputed in the page. The `main.py` at the project root is the old CLI; the Streamlit entry and its navigation are `app/app.py`.

## What reuses the existing engine

- `betting.moneyline_model.win_probability_from_spread` and `betting.odds_math.fair_price_from_probability` produce every win probability and fair price.
- `betting.team_model.project_game` supplies base expected points (fed a NaN-free, blended averages dict, see below).
- `betting.prop_model.evaluate_prop / over_under_probability / stat_distribution` price the six native prop markets, byte-identical to the NFL Betting page (including the player-status overlay). Anytime TD composes the rushing and receiving Poisson rates.
- `betting.parlay_engine` (`make_leg`, `naive_joint_probability`, `parlay_decimal_odds`, `detect_correlations`, `correlation_adjusted_probability`) supplies leg shape, odds math and a cross-check number.
- `modules.nfl.fetch_play_by_play / build_team_season_stats` supply efficiency, EPA and pace.

## Conventions that are easy to get wrong

- **Spread sign.** nflverse `spread_line` and `user_spread_home` are positive when the **home** team is favored. Sportsbook text is the opposite ("home -3.5" = home favored). The page negates what the user types before calling `analyze_game`. A parlay spread leg uses `line = -spread_line` for the home team and `+spread_line` for the away team.
- **Value gap** = model probability minus the price's break-even (vig included), in probability points. **EV** is per stake (default $100) if the model were right. Both are model outputs, not predictions that a bet wins.
- **Market columns are blank later in the season.** nflverse spread, moneyline and total are NaN from about week 6 of 2026 onward; the pages say "No line posted yet" and fall back to the model's own fair price.

## MoneyLines model

Base expected points come from the engine's `project_game`, then adjustments in points: an EPA-per-play blend (`EFFICIENCY_WEIGHT` 0.4, which is also where pace enters the total), home field (`HOME_FIELD_POINTS` 2.0, zero at neutral venues), the named starting QB's value over an average starter (capped at +/-6, net of what is already in the team rating), and a small rest and wind term. Current-season ratings are blended with the prior season by `n / (n + 6)`, and prior-season deviations are carried at `PRIOR_CARRYOVER` 0.4.

- **Win probability uses `WIN_PROB_STDEV` = 11.5, cover probability uses the engine's 13.5.** The engine default under-prices favorites when read as a win rate. Over 3,649 regular-season games (2012-2025), closing spread vs actual winner: log-loss 0.6110 at 13.5, 0.6085 at 11.1 (the minimum), 0.6094 for the de-vigged moneyline. Favorites of 6-8.5 points won 74.5% of the time against 69.4% on the 13.5 curve. Converting the model's fair spread on the calibrated curve keeps the comparison with a pasted moneyline apples to apples.
- Backtest of the pre-game model spread against the closing line (2025): correlation about 0.91, mean absolute difference about 2 points. It still loses to the market on actual results, and only the QB is modelled (no other injuries, coaching, travel).

## Player Props model

One game's mean is last season's per-game rate (the same data season as the NFL Betting page). Yardage is a volatility-scaled Gaussian (cv 0.28-0.65); counts (TDs, receptions) are Poisson. Anytime TD is `1 - exp(-lambda)` with `lambda` = rushing + receiving TD rate (priced off the engine's 3-dp-rounded rates so it matches the Rushing TDs market exactly when there are no receiving TDs). **Known limit:** the engine's volatility saturates for most QBs, so passing-yards spread (cv about 0.59-0.65) is wider than 2025's observed weekly cv (median about 0.33), which pulls over/under probabilities toward 50% for lines away from the mean. The page says so next to the number; the engine was left untouched so figures match the older page. No opponent, weather or role adjustment.

## Parlays model

Each leg carries a probability and a price (pasted price > nflverse consensus > the model's fair price, and the page labels which). Pairwise correlations are a documented rule table of **heuristic priors, not fitted coefficients** (`correlation_rule_table()`; also shown in the page's methodology expander). The adjusted hit probability is a Gaussian copula over those correlations (closed-form exact whenever no correlated cluster has three or more legs, otherwise seeded Monte Carlo so results are repeatable; always clamped to the Frechet bounds, with the standard error reported); contradictory legs are blocked at add time. The default EV multiplies leg prices, which books reprice for correlated legs, so the page asks for the book's quoted parlay price (`book_price`) and says so. Risk tier is Low / Medium / High from a deterministic 0-100 score that honours the engine's baseline (`<=2` legs and `>=35%` is Low; `<=4` legs and `>=15%` is Medium; else High).

## Known issues found while building (not fixed here)

- `betting.team_model.team_scoring_by_week` / `team_scoring_averages(season)` return **NaN for every team in an in-progress season**: the loader tests `home_score is None`, and pandas gives NaN for unplayed games. The NFL Betting page avoids it by pricing from `upcoming_season() - 1`. The new modules compute in-season points from the schedule with `pd.notna` instead. Fixing the engine is a one-line `pd.notna` check.
- `app/style.py` `apply_gold_glow_theme` sets a title font but no title text, so a Plotly figure built without a title shows the literal word "undefined" in Streamlit 1.60 (Player Props sets `title_text=""`). Streamlit's header bar also renders near-white across the dark canvas on every page.
- Streamlit 1.60 draws selects, text/number inputs, radios and expander headers with markup `app/style.py` does not select, which rendered them white on the dark canvas. `app/betting_ui.py` restyles them for pages that load the kit; the older pages are unchanged.

## Tests

`tests/test_odds_input.py`, `tests/test_betting_ui.py`, `tests/test_nfl_moneyline_analysis.py`, `tests/test_nfl_prop_analysis.py`, `tests/test_nfl_parlay_builder.py` cover the helper, kit and engines offline from hand-built fixtures (plus a couple of live-data tests that skip when nflverse is unreachable). `tests/test_moneylines_page.py`, `tests/test_playerprops_page.py`, `tests/test_parlays_page.py` drive each real page through `AppTest` with monkeypatched loaders and assert behaviour: week/game/team/player selection, typed prices and lines changing the numbers, sign conversion, contradictory legs blocked, empty and offline states, and no `nan` / `None` / `inf` text anywhere.
