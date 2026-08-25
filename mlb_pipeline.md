# MLB data pipeline

Flagship sport of this build cycle. See [offline_data_contract.md](offline_data_contract.md) for the rules this pipeline follows, [betting_engine.md](betting_engine.md) for how MLB fits the six-sport unified contract, [mlb_matchup_engine.md](mlb_matchup_engine.md) for the five DFS matchup modules, [mlb_fusion_model.md](mlb_fusion_model.md) for how they combine with the season baseline, and [nhl_pipeline.md](nhl_pipeline.md) for the lightweight sibling sport added the same cycle.

## The real blocker: no live or keyless MLB stats source this cycle

CFB and CBB each needed a real, keyless-or-narrowly-keyed public JSON stats API to build their pipelines (`api.collegefootballdata.com`, ESPN's public site API — see [college_sports_betting.md](college_sports_betting.md)). This cycle's instructions were explicit: no live odds or scraping. Rather than reach for a package that scrapes HTML or needs an API key this environment can't obtain, MLB ships fully offline this cycle — every default data file is empty by design, the exact same documented pattern CFB's `data/cfb_props.json` already established while gated on an unset `CFBD_API_KEY`. MLB's gate is simpler: there's no key to set at all yet, just an upload path.

| Considered | Why not this cycle |
|---|---|
| `pybaseball` | Scrapes Baseball Reference / Statcast HTML/CSV endpoints under the hood for several of its functions — the same category of source this project's `bs4`/scraping ban exists to keep out (see offline_data_contract.md rule 1's spirit, even though it's not a sportsbook). |
| A hypothetical `mlb-api`/`mlb_api`-style package equivalent to `nba_api` | Not present in `requirements.txt`, and adding a new runtime dependency plus its own real-data verification is a larger, separate decision than this cycle's explicit "no live data" instruction calls for. |
| MLB's own real `statsapi.mlb.com` public endpoint | A real, public, keyless JSON API — a legitimate future candidate for the same narrow, documented `requests` exception CFB/CBB use (see offline_data_contract.md's exception list). Deliberately not added this cycle; see "What's not built" below. |

Every one of `modules/mlb_props_loader.py`, `modules/mlb_odds_loader.py`, `modules/mlb_injuries_loader.py`, `modules/mlb_lineups_loader.py` works exactly the same way CFB's loaders do without a key: real data comes from a file you upload through the MLB Betting page, and `data/mlb_*.json` ship empty with a `"note"` explaining why (enforced by `tests/test_data_contract_hardening.py::ProvenanceDisclosureTests`).

## Three-layer engine

```
modules/mlb_season_model.py          (baseline: long-term regression averages)
    -> modules/mlb_batter_vs_pitcher.py   (handedness, pitch-type, career BvP, contact quality)
    -> modules/mlb_ballpark_model.py      (real, static park factors)
    -> modules/mlb_lineup_model.py        (batting order, protection, platoon, SB environment)
    -> modules/mlb_bullpen_model.py       (fatigue, ERA rating, leverage usage)
    -> modules/mlb_defense_model.py       (efficiency, outfield arm, catcher framing, infield range)
-> modules/mlb_fusion_model.py       (reliability/sample-size-weighted blend -> final projection)
```

The season model and the five matchup modules and the fusion model are all pure, offline math over caller-supplied inputs (a game log, a batter/pitcher profile, a park's real static attributes, a bullpen's real recent-appearance log, a defense's real season metrics) — none of them fetch anything, and none of them are wired into the unified betting-engine contract as a requirement (see betting_engine.md). They're an optional overlay: a power user (or, in a future cycle with a real stats source, a generator) runs the fusion model to arrive at the number they trust enough to put in a prop's `"line"` field. Whatever `"line"` a prop actually carries — hand-entered, uploaded, or fusion-informed — is what `modules/mlb_prop_model.py`, the contract-required evaluator, prices directly, the same self-contained pattern `modules/cbb_prop_model.py` and `modules/cfb_prop_model.py` use.

## Why Poisson, not Gaussian

Every one of the 7 modeled categories (`modules/mlb_common.py::STAT_CATEGORIES` — hits, home runs, RBI, total bases, strikeouts, walks, stolen bases) is a low-mean, non-negative per-game count (a typical hits line is ~1.0, a home-run line is ~0.3). `modules/mlb_prop_model.py` models every category as Poisson rather than Gaussian — the same treatment `fantasy_engine/betting/prop_model.py` already applies to NFL's own low-mean count markets (touchdowns, receptions), for the identical documented reason: a Gaussian with `stdev = mean * cv` badly understates a low-mean count stat's real spread.

## MLB moneyline: no live team-scoring average, by design

Unlike NFL/NBA/CFB/CBB's moneyline models, `modules/mlb_moneyline_model.py` does not call `betting.team_model.project_game` — that function needs a real points-scored-average dict, which this engine deliberately never fetches (see above). Instead, `fair_moneyline` takes a composite rating per team, built from caller-supplied pitcher-quality/bullpen-strength/park-factor/defensive-efficiency/lineup-strength context (each defaulting to a neutral `1.0` when not supplied — "no fabricated ratings," the same principle `betting/team_model.py`'s own docstring states), and converts the composite-rating difference to a win probability via a logistic transform plus the real, published MLB home-field win rate (~54%) rather than a normal approximation over a scoring margin — there is no real margin distribution available without live scoring data here.

## Data files

| File | Ships with | Why |
|---|---|---|
| `data/mlb_props.json` | Empty, `"note"` explaining why | No live/keyless source integrated this cycle (see above) |
| `data/mlb_game_odds.json` | Empty, `"note"` | This app never fetches odds from any sportsbook, for any sport |
| `data/mlb_injuries.json` | Empty, `"note"` | Same rule as every sport's injury file |
| `data/mlb_lineups.json` | Empty, `"note"` | No live lineup source; feeds `mlb_lineup_model.py` / `mlb_batter_vs_pitcher.py` when populated |

## Unified integration

`modules/unified_betting_contract.py`: `VALID_SPORTS` extended to include `"MLB"`; `compute_ev("MLB", props, **context)` accepts an optional `matchup_multipliers` context (a per-player composite multiplier, e.g. from `mlb_fusion_model.fuse_projection`) — the same optional-context shape CBB's `minutes_by_player` already established, not a new pattern. `modules/unified_parlay_engine.py`'s `_MAKE_LEG_BY_SPORT`/`_DETECT_CORRELATIONS_BY_SPORT` dispatch dicts gained an `"MLB"` entry routed to `modules/mlb_parlay_engine.py`'s own new correlation detector (baseball correlations are a real, different phenomenon from football/basketball — see that module's docstring for why this couldn't be a re-export the way CFB/CBB's engines are).

## What's *not* built (by design, not oversight)

- **No live per-game stats ingestion.** Every number here traces to a file you upload or a caller-supplied context dict — see "the real blocker" above.
- **No run-line (spread) or game-total pricing.** `modules/mlb_moneyline_model.py` prices moneyline only this cycle; `modules/mlb_odds_loader.py`'s schema already carries an optional `"total"` field (forward-compatible) but nothing evaluates it yet.
- **No props generator.** Unlike NBA/CFB/CBB, there's no `modules/mlb_props_generator.py` — there's no live source for it to generate from this cycle.

## Testing

`UniversalQuantAgent/tests/test_mlb_season_model.py`, `test_mlb_matchup.py`, `test_mlb_fusion.py`, `test_mlb_prop_model.py` (loaders + prop model), `test_mlb_moneyline.py`, `test_mlb_parlay.py`, and `test_mlb_nhl_betting_pages.py` (UI). `tests/test_data_contract_hardening.py` and `tests/test_unified_betting_contract.py` (repo root and `UniversalQuantAgent/tests/`) were extended with MLB cases rather than duplicated.
