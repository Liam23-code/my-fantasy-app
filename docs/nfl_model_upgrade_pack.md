# NFL model upgrade pack

Collapse, explosion, mismatch, red-zone funnel, game-script and parlay-correlation layers for the
NFL projection and betting pages. Everything is under `UniversalQuantAgent/modules/` and is pure
logic (no Streamlit, no network) except the two feature loaders and the context assembler, which are
the only code that reads nflverse and which fail soft.

> **How to read the numbers.** The weights in every model are *priors*, not fitted coefficients, and
> several inputs are free-data *proxies* or are *neutral until you supply them* (section 6). Each model
> reports a `basis` string, the inputs it actually used, and the inputs it defaulted. A pack number is a
> disciplined adjustment for matchup context, not a measured edge. Sections 5 and 7 say where the
> brief was changed or extended and what the evidence does and does not support.

## 1. Pipeline order

```text
 context (canonical features)                    player stat line (engine field names)
        |                                                   |
        v                                                   v
 game_script probabilities (computed first: run_game_collapse needs trailing_script_prob)
        |
 1  COLLAPSE        QB: qb_collapse | WR/TE: pass_game_collapse (consumes QB collapse) | RB: run_game_collapse
 2  EXPLOSION       explosion_model          qb_yards, wr_yards, rb_yards, team_points
 3  MISMATCH        coverage_mismatch_model  wr_yards, te_yards
    EXPLOSIVE PLAY  explosive_play_model     wr_yards (+18 x p), rb_yards (+12 x p)
 4  RED ZONE        red_zone_funnel_model    qb_td_prob, rb_td_prob, te_td_prob
 5  GAME SCRIPT     game_script_model        qb_attempts, rb_attempts   (+ volume pass-through, section 5)
 6  PARLAY          parlay_correlation       correlation matrix, strength score, MEGA / TRAP flags
```

## 2. Modules

| Module | Compute function | Apply function | Output |
|---|---|---|---|
| `nfl_pack_common.py` | `Factor`, `score_factors`, `run_factor_model`, `scale_field`, `add_to_field`, `guard_cumulative` | | shared math |
| `qb_collapse_model.py` | `compute_qb_collapse(inputs)` | `apply_qb_collapse(projection, result, *, recenter=True)` | `collapse_probability` |
| `pass_game_collapse.py` | `compute_pass_game_collapse(inputs)` | `apply_pass_game_collapse(...)` | `pass_game_collapse` |
| `run_game_collapse.py` | `compute_run_game_collapse(inputs)` | `apply_run_game_collapse(...)` | `run_collapse` |
| `explosion_model.py` | `compute_explosion(inputs)` | `apply_explosion(...)` | `explosion_probability` |
| `explosive_play_model.py` | `compute_explosive_play(inputs)` | `apply_explosive_play(...)` | `explosive_play_prob_pass`, `explosive_play_prob_run` |
| `coverage_mismatch_model.py` | `compute_coverage_mismatch(inputs, *, role)` | `apply_coverage_mismatch(...)` | `coverage_mismatch_factor` (-1..+1) |
| `red_zone_funnel_model.py` | `compute_red_zone_funnel(inputs)` | `apply_red_zone_funnel(...)` | `rz_pass_funnel`, `rz_run_funnel` (-1..+1) |
| `game_script_model.py` | `compute_game_script(inputs)` | `apply_game_script(...)` | `positive/negative/neutral_script_prob` |
| `parlay_correlation.py` | `build_correlation_matrix`, `parlay_strength`, `rank_parlays` | | `correlation_matrix`, `parlay_strength_score` |
| `nfl_team_features.py` | `team_feature_table(season, *, through_week)` | | 32-team offence / defence features |
| `nfl_player_features.py` | `player_feature_table(...)`, `find_player_features(...)` | | QB / WR / TE / RB features |
| `nfl_matchup_features.py` | `build_game_context(team, opponent, ...)` | | one game's context |
| `nfl_upgrade_pipeline.py` | `compute_game_signals`, `apply_upgrade_pack`, `project_with_pack`, `team_signals` | | adjusted line + audit |

Every compute function takes a flat mapping of canonical keys (any subset), never raises on bad data
(`None`, `NaN`, `"n/a"`, `"31%"` and `31` are all handled), is deterministic, and returns its output
plus `references` (the value for neutral input), `z_score`, `coverage`, `inputs_used`,
`inputs_defaulted`, `contributions`, `drivers` and `basis`. Probabilities are logged at DEBUG on the
`nfl_pack.*` loggers (for example `logging.getLogger("nfl_pack").setLevel(logging.DEBUG)`).

## 3. How a number is produced

Each model is a transparent weighted z-score through a logistic:

```text
z = sum(weight_i * direction_i * clip(z_i)) / sum(weight over ALL factors)      (missing input -> z_i = 0)
p = sigmoid(logit(base_rate) + gain * z)
```

* A missing input contributes *neutral* but still counts in the denominator, so **missing data
  shrinks the answer toward the league-average case instead of inflating it**; `coverage` says how much
  of the weight was observed.
* With every input missing or exactly average, `p = base_rate` exactly. That is the model's reference.
* **Every adjustment is re-centred.** The engine's per-game averages already contain the league-average
  amount of collapse and explosion, so `qb_yards *= (1 - p*0.65)` at `p = 0.16` would cut every
  quarterback ~10% in every matchup. The apply functions divide by the same formula at the reference
  (`(1 - p*c) / (1 - p_ref*c)`), so a league-average situation is an exact no-op and only the *deviation*
  moves a number. `recenter=False` gives the literal formula from the brief.

## 4. What each probability means (event definitions)

A probability needs an event. Each base rate was measured on nflverse regular-season data; none was
tuned to the model's output.

| Model | Event | Base rate | Notes |
|---|---|---|---|
| `qb_collapse` | a starter game (>= 20 dropbacks) with EPA/dropback <= -0.25 | 0.16 (580 of 3,658 games, 2019-25) | gain 1.0; on 1,596 starter games 2023-25 (leave-one-game-out features, not a forward backtest) AUC 0.63, Spearman -0.21; 5 of 12 inputs show no association |
| `pass_game_collapse` | team passing yards <= 70% of the team's own average over its other games AND team passing EPA < 0 | 0.12 (625 of 5,278 team-games, 2016-25) | gain 0.6; composite AUC ~0.58 - a mild tilt, not a forecast |
| `run_game_collapse` | designed-RB rushing <= 60% of the team's own average AND negative RB rushing EPA | 0.176 (n = 3,742) | gain 0.8; **no out-of-sample discrimination on free proxies (AUC 0.51)** - prior-driven |
| `explosion` | the offence scores 35+ points | 0.117 (437 of 3,742 team-games) | gain 0.55; prior-season features AUC 0.605. The brief suggested 30+, which is 25.8% of team-games and not an "explosion" tail |
| `explosive_play` | pass: a WR with >= 4 targets has a reception of 30+; run: an RB with >= 8 carries has a carry of 30+ | pass 0.273 (n = 8,383), run 0.126 (n = 4,617) | the brief's `+p*18` / `+p*12` yards, re-centred; pass AUC 0.53 (weak) |
| `coverage_mismatch` | none - a signed factor, +-1 is the extreme of the evidence | 0 | six of nine inputs have no free feed, so with free data alone it stays near 0 |
| `red_zone_funnel` | pass share of red-zone touchdowns | 0.567 (n = 7,008) | the defensive red-zone contrast shows no predictive signal on free data; a small prior-driven lean |
| `game_script` | play-share: trailing >= 7 (pass-heavy) / leading >= 7 (run-heavy) / neutral | trailing 0.297 at a pick'em | the one *fitted* model: ordered logit on spread and total, 2018-25 (264,889 snaps); holdout log-loss 0.971 vs 1.031 for constant shares |

## 5. Where this goes beyond or against the brief

1. **Game-script semantics.** The brief's formulas (`qb_attempts *= 1 + pos*0.20 - neg*0.25`,
   `rb_attempts *= 1 + neg*0.30 - pos*0.20`) are only football-correct if *positive script = pass-heavy
   (the offence is trailing)* and *negative script = run-heavy (leading)*. The brief's run-collapse input,
   "negative game script probability", uses the colloquial sense (the team is *trailing* and abandons the
   run). The model therefore exposes unambiguous aliases `trailing_script_prob` / `leading_script_prob`;
   `run_game_collapse` consumes `trailing_script_prob`. The coefficients are applied exactly as written.
   If you meant the opposite, set `POSITIVE_SCRIPT_IS_PASS_HEAVY = False` in `game_script_model.py`.
2. **Re-centring** (section 3) - the literal formula is available with `recenter=False`.
3. **Volume pass-through** (`propagate_volume=True`, not in the brief). Game script changes
   `qb_attempts` / `rb_attempts` only, and nothing bets on attempts, so alone it would be cosmetic. The
   attempt factor also scales the player's dependent counting stats at elasticity 1 (a QB's yards, TDs,
   interceptions; every receiver's yards, TDs, receptions; a RB's rushing yards and TDs). Audited as
   `volume_passthrough` rows; switch off with `propagate_volume=False`.
4. **Cumulative guard** (`guard=True`, not in the brief). Five mild stages compound; no field may leave
   0.40x-2.00x of its starting value, and any clamp is reported.
5. **`team_points`** is moved by the explosion stage only; the brief gives no collapse coefficient for
   points, so none was invented. `rb_yards` is rushing yards; an RB's receiving line moves only through
   volume pass-through.
6. **Parlay blend** is `base + w * (pack_rho - pack_rho_reference)`, not a convex mix: the rule table
   already carries the league-average link between two markets, and the pack adds only how *this* game's
   regimes differ from average. A league-average game returns the rule table exactly; structural +-1 pairs
   and the mutually-exclusive sentinel are never touched.
7. **Probability caps.** A field that is a probability (<= 1.0) is capped at 1.0, one on a percent scale
   (> 1.5) at 100, and an expected count between 1.0 and 1.5 (a QB's 1.3 TDs/game) is left uncapped.

## 6. Data: what is real, what is a proxy, what is neutral

All data is nflverse through `nflreadpy` (game data; offline_data_contract.md rule 2). Injury data is
never fetched live (rule 1).

| Input | Status | Source |
|---|---|---|
| EPA/play, explosive rates, pace, red-zone TD rate, PROE, QB deep-attempt rate, red-zone pass rate | real | pbp |
| QB time to throw, WR separation | real | NGS |
| RB yards after contact, missed-tackle rate | real | PFR advanced |
| stacked-box rate | real | NGS rushing (`percent_attempts_gte_eight_defenders`) |
| pressure rate, blitz rate, <= 4-rusher pressure, man-coverage rate | real | participation `was_pressure` / `number_of_pass_rushers` / `defense_man_zone_type` (PFR counts mapped onto that scale for a season the feed has not published yet) |
| `ol_pass_block_win_rate`, `opp_pass_rush_win_rate` | **proxy** | 1 - pressure-or-sack share allowed; pressures per dropback (not PFF/ESPN win rates) |
| `ol_run_block_win_rate` | **proxy** | PFR yards-before-contact and rush EPA, placed on a 0.70 / 0.04 scale (a relative standing) |
| `opp_coverage_grade`, `opp_run_defense_grade`, `opp_explosive_play_prevention`, red-zone grades | **proxy** | team percentiles of EPA / 20+ yard plays allowed |
| `ol_continuity` | **proxy** | top-five OL share of OL snaps |
| `ol_injuries` | optional | only from the project's own offline player-status file, and **ignored when older than 7 days** (it was 36 days old on 2026-10-12) |
| CB1 shadow rate, bracket coverage, WR-vs-CB / slot-vs-nickel / TE-vs-LB grades, coverage bust rate | **no free feed** | neutral until you pass them in `overrides` |

Definition breaks the loaders handle: nflverse started coding sacks and scrambles as pressure in 2023
(references use 2023-25), `defenders_in_box` breaks in 2024 (NGS is used), and the man-coverage label
regime shifted in 2023. Early in a season a team feature is shrunk toward last season with weight
`g / (g + K)`; `K` was measured per metric from split-half reliability on 2018-2025 and runs from 2.8
(man-coverage rate, PROE) through ~5 (EPA/play) to 40 (red-zone TD rate, red-zone EPA).

## 7. Honest limits

* Weights are priors. Several measured associations are weak (above); `coverage_mismatch` and the
  red-zone funnel are mostly inert on free data. Treat the output as a matchup *lean*.
* The collapse events are team- or QB-level; how much of one lands on a single receiver is decided by the
  brief's coefficients, not by data.
* The Player Props baseline is the player's *prior-season* per-game rate; the pack adds the first
  opponent-specific adjustment but not a role change, weather or in-season form.
* The engine baseline can already lean on a matchup score, so coverage / explosion can partly overlap it.
* Gaussian-copula parlays have no tail dependence; sportsbooks reprice same-game parlays; the parlay
  strength score is a rule of thumb for how fragile a ticket is, not a prediction.
* The regime model's own correlations (~+0.16 for a QB-yards / WR-yards stack) are far below the rule
  table (0.45) and observed same-team correlations (~+0.5): it carries the *game-environment* part only,
  which is why it is blended as a deviation.

## 8. Using it

```python
from modules import nfl_matchup_features as feats, nfl_upgrade_pipeline as pipe

ctx = feats.build_game_context("KC", "BUF", season=2026, week=7)            # I/O, fail-soft, cached
out = pipe.project_with_pack({"passing_yards": 255.0, "passing_tds": 1.8, "attempts": 36.0},
                             position="QB", context=ctx)
out["line"]          # adjusted stat line        out["adjustments"]  # stage, field, before, after, factor
out["explanations"]  # plain-English sentences   out["guard"]        # any cumulative clamp
```

With no context (or one with no usable features) `line` equals the input and `neutral` is `True`.
User numbers go in `overrides=` (same canonical keys) and always win.

## 9. Pages

Every page has a **"Use the matchup upgrade pack"** toggle, default **on**. Engine functions take the
pack as an opt-in keyword (`pack_context=` / `pack_contexts=` / `pack_signals=`) whose default leaves every
existing key and number byte-identical (pinned by tests); only the page builds a context, through one
cached, fail-soft helper, and passes it only when it exists. When features cannot be loaded the page says so
in one quiet line and shows the unchanged number.

| Page | What changed | Where it is applied |
|---|---|---|
| **Player Props** (`pages/playerprops.py`, `modules/nfl_prop_analysis.py`) | new **Opponent** selector (defaults to the team's game in the schedule's current week) and a **Matchup signals** card | the market's per-game mean is adjusted after the availability overlay; a Gaussian market keeps its cv, a count market uses `sqrt(mean)`; probabilities are priced from the adjusted distribution |
| **MoneyLines** (`pages/moneylines.py`, `modules/nfl_moneyline_analysis.py`) | pack card: explosion chance, QB-collapse risk and points before to after per team | each team's expected points get the explosion stage *before* spread, win probability, cover probability and fair price are derived, so every number stays consistent; slate tiles show the base model |
| **Parlays** (`pages/parlays.py`, `modules/nfl_parlay_builder.py`) | leg probabilities come from the pack-adjusted projections; a **Pack read** section (strength 0-100, MEGA / TRAP / contradiction / boosted-overs flags, plain-English reasons, the correlations the pack moved) and a ranked "stack opportunities" expander | the copula matrix; the TRAP flag adds 5 risk points and never lowers risk |
| **NFL Projections** (`pages/23_NFL_Projections.py`, `modules/nfl_projections.py`) | checkbox in the form; adjustments table in the Projection tab; drivers gain one sentence per material change | the finished per-game line, then fantasy points and the confidence band are recomputed |

Not changed on purpose: **NFL Betting (Offline)** (`30_NFL_Betting.py`) prices the engine's own generated
odds file against the engine's own un-adjusted projections, so a matchup overlay there would only
manufacture edges against a number built without it. Slate tiles on MoneyLines and the earlier
Parlay Builder are untouched.

Tests: `test_nfl_moneyline_pack_integration.py`, `test_nfl_props_pack_integration.py`,
`test_nfl_projections_pack.py`, `test_nfl_parlay_pack_integration.py` (engine level) and
`test_nfl_moneyline_pack_page.py`, `test_nfl_props_pack_page.py`, `test_nfl_parlays_pack_page.py`
(`AppTest`, offline). The three pre-existing page test modules disable the pack with a module-level
`setUpModule` because the toggle is on by default and its builder does I/O; their assertions are the
base model's, unchanged except one pinned widget list (the new Opponent selector).

## 10. Tests, environment, launch

```powershell
# from the repo root (C:\Users\churc\OneDrive\Documents\AI Quant)
.\UniversalQuantAgent\.venv\Scripts\python.exe -m pytest tests UniversalQuantAgent\tests fantasy_engine\tests -q
.\UniversalQuantAgent\.venv\Scripts\python.exe run_local.py
```

`UniversalQuantAgent/tests` has **no** `__init__.py` on purpose: `fantasy_engine/tests` is a package named
`tests`, and a second one makes pytest raise `ImportPathMismatchError` for the combined run (verified).
