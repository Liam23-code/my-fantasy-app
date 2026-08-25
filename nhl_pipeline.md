# NHL data pipeline

Lightweight secondary sport this cycle — the same minimal shape CFB/CBB started from, without either college sport's later live-data addition. See [offline_data_contract.md](offline_data_contract.md) for the rules this pipeline follows, [betting_engine.md](betting_engine.md) for how NHL fits the six-sport unified contract, and [mlb_pipeline.md](mlb_pipeline.md) for the flagship sport added the same cycle.

## No live or keyless NHL stats source this cycle

The NHL does publish a real, public, keyless JSON API at `api-web.nhle.com` — a legitimate future candidate for the same narrow, documented `requests` exception CFB/CBB use (see offline_data_contract.md's exception list). It was deliberately not integrated this cycle: the build instructions were explicit that no live odds or scraping should be introduced, and NHL is scoped as a lightweight secondary sport rather than a second flagship needing its own live-data verification pass. `data/nhl_props.json`, `data/nhl_game_odds.json`, and `data/nhl_injuries.json` all ship empty by design, the same documented pattern MLB's default files use (see mlb_pipeline.md) — real data comes only from a file you upload through the NHL Betting page.

## Minimal engine, mirroring CFB/CBB's original shape

```
modules/nhl_common.py            (32-team alias table, 4-category vocabulary)
modules/nhl_props_loader.py      modules/nhl_odds_loader.py      modules/nhl_injuries_loader.py
modules/nhl_prop_model.py        modules/nhl_moneyline_model.py  modules/nhl_parlay_engine.py
```

No season-average model, no matchup layer, no fusion model — NHL has none of MLB's richer three-layer structure this cycle (see mlb_pipeline.md / mlb_matchup_engine.md), matching how CFB and CBB shipped before either later grew a live data source. `modules/nhl_prop_model.py` prices whatever `"line"` a prop row carries directly, the same self-contained pattern every "no pre-existing rich model to wrap" sport in this codebase uses (see betting_engine.md).

## Two distributions, not one

The four modeled categories (`modules/nhl_common.py::STAT_CATEGORIES` — shots, goals, assists, saves) split into two real, differently-shaped groups:

- **Goals and assists** are low-mean, discrete, right-skewed counts (a typical goals line is ~0.3-0.5) — modeled as Poisson, the same treatment `fantasy_engine/betting/prop_model.py` applies to NFL's own low-mean count markets and `modules/mlb_prop_model.py` applies to every one of MLB's seven categories (see mlb_pipeline.md's "Why Poisson, not Gaussian").
- **Shots and goalie saves** have a meaningfully higher real per-game mean (shots ~2-3, saves ~25-30), where the normal approximation is a reasonable fit — kept as Gaussian with a disclosed coefficient of variation (`_BASE_CV = 0.45`), the same shape CFB/CBB's simpler prop models use.

## NHL moneyline: no live team-scoring average, by design

Same reasoning as MLB (see mlb_pipeline.md): `modules/nhl_moneyline_model.py` does not call `betting.team_model.project_game`, since no live NHL scoring-average ingestion was added this cycle. `fair_moneyline` takes a composite rating per team from caller-supplied goaltending/offensive/defensive/special-teams-strength context (each defaulting to neutral `1.0` — no fabricated rating), converted to a win probability via the same logistic-transform-plus-home-advantage approach `mlb_moneyline_model.py` uses, with NHL's own real, published home-ice win rate (~55%) in place of MLB's home-field constant.

## Two correlation patterns

`modules/nhl_parlay_engine.py` adds two real hockey correlation patterns on top of the fully generic parlay math (`betting.parlay_engine`, reused directly — see betting_engine.md's "shared, not duplicated" section):

- **goals ↔ assists** (same player, same game) — a real multi-point game structurally links a player's own goal and assist totals, the same "overlapping stat category" relationship `modules/nba_parlay_engine.py`'s PRA pattern and `modules/mlb_parlay_engine.py`'s HR/total-bases pattern both model for their sports.
- **teammate goal stack** (same team, same game, different players) — two teammates' goal totals both real-ly depend on the same game's pace and power-play opportunities, the same "teammate stack, same shared cause" relationship `modules/nba_parlay_engine.py`'s teammate-scoring pattern models for basketball.

Deliberately two patterns, not MLB's four — NHL is scoped as the lightweight sport this cycle; a richer correlation set (e.g. a goalie-saves-vs-opponent-shot-volume pattern) is real, buildable future work, not built here.

## Unified integration

`modules/unified_betting_contract.py`: `VALID_SPORTS` extended to include `"NHL"`; `compute_ev("NHL", props)` needs no extra context, the same as CFB. `modules/unified_parlay_engine.py`'s dispatch dicts gained an `"NHL"` entry routed to `modules/nhl_parlay_engine.py`'s own detector (hockey correlations are a real, different phenomenon from every other sport's — not a re-export).

## What's *not* built (by design, not oversight)

- **No live per-game stats ingestion** — see "no live or keyless NHL stats source" above.
- **No puck line (spread) or game-total pricing** — moneyline only this cycle, the same scope limit as MLB's run-line.
- **No season-average/matchup/fusion layer** — see "minimal engine" above; a genuinely buildable future extension once/if NHL is promoted to a flagship sport.

## Testing

`UniversalQuantAgent/tests/test_nhl_prop_model.py` (loaders + prop model, both Poisson and Gaussian categories), `test_nhl_moneyline.py`, `test_nhl_parlay.py`, and `test_mlb_nhl_betting_pages.py` (UI, shared with MLB's page tests). `tests/test_data_contract_hardening.py` and `tests/test_unified_betting_contract.py` were extended with NHL cases.
