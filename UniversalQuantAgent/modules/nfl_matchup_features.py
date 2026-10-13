"""Assemble the canonical-feature CONTEXT for one offence against one defence in one NFL game.

``build_game_context(team, opponent, ...)`` is the single call the pages make for I/O. It composes
what the finished loaders already produce -- :mod:`modules.nfl_team_features` (team offence and the
opponent's defence), :mod:`modules.nfl_player_features` (the quarterback and the projected player),
the nflverse schedule (the market lines) -- with the project's own offline player-status overlay
(offensive-line injuries) and the user's overrides, into the dict that
:func:`modules.nfl_upgrade_pipeline.compute_game_signals` flattens and feeds to the eight models::

    {"team", "opponent", "season", "week", "through_week", "is_home",
     "offense": {...}, "opponent_defense": {...}, "qb": {...}, "player": {...},
     "game": {"vegas_spread", "vegas_total", "opp_epa_per_play", "ol_injuries", "ol_health"},
     "sources": {key: {"kind": "real"|"proxy"|"override"|"derived"|"default", "basis": str}},
     "warnings": [str], "has_features": bool,
     "qb_player": {...} | None, "player_match": {...} | None}

The last two keys are additions (not in the contract): the identity of the quarterback whose
features were used and of the matched projected player, so a page can show *who* the numbers are about.

Rules this module keeps
-----------------------
* **Fail soft, never raise.** Every loader it calls already degrades to ``{}`` / an empty frame; this
  module additionally guards each stage, so any failure becomes fewer keys plus a ``warnings`` entry.
  With nothing available the result is the neutral context (``has_features`` False), which the
  pipeline treats as an exact no-op.
* **A missing feed is neutral.** A key with no data is simply absent -- never filled with a made-up
  number. (``sources`` carries a ``"default"`` entry for the few absent ``game`` keys that were
  *looked for* -- lines and OL injuries -- so a page can say why; those keys are NOT in the sections.)
* **No leakage.** ``through_week`` defaults to ``week - 1`` (a projection for a game never sees that
  game); an explicit ``through_week`` is respected, with a warning if it is not before ``week``. The
  schedule is read only for ``spread_line`` / ``total_line`` (pre-game lines); the result and score
  columns are never selected, so they cannot leak into a feature.
* **Honest labels.** ``kind`` is what the loader said (``real`` / ``proxy``), ``override`` for a
  number the caller typed, ``derived`` for arithmetic on another entry (``ol_health``), ``default``
  for "looked for, not available". ``has_features`` is True only when at least one *real* or
  *proxy* feature made it in -- overrides and caller-supplied lines alone do not count.

Sections and precedence
-----------------------
Sections are merged by the pipeline left to right (later wins): ``offense``, ``opponent_defense``,
``qb``, ``game``, ``player``. This module keeps that order as :data:`SECTIONS`. A canonical key may
appear in two sections only when both carry the *same* number (a QB being projected sits in ``qb``
and ``player`` with identical values); where the loaders disagree (the team table's primary-QB
``qb_efficiency`` vs the QB's own player-table value) the later section wins and the earlier copy is
removed, so ``sources[key]`` always describes the number the models actually read. User overrides
are routed to a section by :data:`OVERRIDE_SPECS` and *remove* the same key from every other
section, so an override can never be shadowed.

Quarterback
-----------
``qb`` may name the quarterback (id, name or a row); otherwise the team's QB with the most
dropbacks (then most games) in the player table is used -- a season-to-date rule, so a QB who has
just replaced an injured starter needs ``qb=...``. If the projected ``player`` is himself a QB he is
the quarterback. The team table's ``qb_efficiency`` / ``qb_deep_attempt_rate`` describe the team's
primary QB; when the QB in use is someone else they are dropped (with a warning) rather than
presented as his.

Offensive-line injuries (``ol_injuries`` / ``ol_health``)
---------------------------------------------------------
There is no free live feed (``offline_data_contract.md`` rule 1). The count is derived only from the
project's own overlay file (:func:`fantasy.player_status.load_player_status`, local JSON), crossed with
the team's five most-used OL by *this season's* snap counts (``nfl_team_features``' snap loader, through
``through_week``): OUT / DOUBTFUL / HOLDOUT / SUSPENDED = 1.0, QUESTIONABLE = 0.5. Limits, all
reported in ``sources``: the overlay has no team or position, so records are matched to the five
linemen by normalised name; a player the file does not list is "not flagged", not "proven healthy",
so a team none of whose five OL appear in the file gets NO ``ol_injuries`` (neutral) rather than a
measured zero; the file is a snapshot of *now*, so it is not applied to a past season or to a
game whose week has already been played (``through_week`` before the latest week with snap data);
and a file older than :data:`STATUS_MAX_AGE_DAYS` (7) is not applied at all.

Market lines
------------
``vegas_spread`` / ``vegas_total`` come from the nflverse schedule's consensus ``spread_line`` /
``total_line`` (the pack contract, section 4, names that as the free feed). nflverse spread_line is
POSITIVE when the HOME team is favoured; the home team's ``vegas_spread`` is ``-spread_line`` and the
away team's ``+spread_line`` (sportsbook sign: -7 = this team favoured by 7). Only those two game
lines are read -- never a moneyline, price or juice field -- but note that
``offline_data_contract.md`` rules 1-2 say sportsbook odds should come only from the project's own
file or a user upload. The pack contract's choice is followed here and flagged; set
:data:`USE_SCHEDULE_LINES` to ``False`` to forbid schedule lines entirely (callers then supply
``vegas_spread`` / ``vegas_total``, e.g. from an uploaded odds file). A caller-supplied number always
beats the schedule, and an override beats both.

Honest limits
-------------
* Early in a season (either team under four games) the team table is mostly last season's values
  (it shrinks toward the prior season) and the player table is heavily shrunk toward the league mean
  with most features below their volume gates; a warning says so.
* Player features come from last-season-or-this-season rows only as the loaders define them; at
  ``week=1`` (``through_week=0``) the player table is empty by design.
* The overlay can only describe availability as of its own timestamp.
"""
from __future__ import annotations

import copy
import datetime as _dt
import logging
import math
import re
import threading
import time
import unicodedata
from typing import Any, Callable, Mapping

import pandas as pd

from modules import nfl_player_features as npf
from modules import nfl_team_features as ntf
from modules.nfl_pack_common import to_float, to_rate

LOG = logging.getLogger("nfl_pack.matchup_features")

#: Section order == ``nfl_upgrade_pipeline.CONTEXT_SECTIONS`` (later wins when the pipeline merges).
SECTIONS: tuple[str, ...] = ("offense", "opponent_defense", "qb", "game", "player")

#: Read the market lines from the nflverse schedule when the caller did not supply them. See the module
#: docstring: the pack contract asks for it, ``offline_data_contract.md`` rules 1-2 are stricter.
USE_SCHEDULE_LINES: bool = True

#: The player-status file is a snapshot of availability; older than this many days it is NOT applied
#: (injury designations turn over weekly), and ``ol_injuries`` stays neutral with the reason in ``sources``.
STATUS_MAX_AGE_DAYS = 7

_KINDS = frozenset({"real", "proxy", "override", "derived", "default"})
_MEASURED = frozenset({"real", "proxy"})

# ---------------------------------------------------------------------------
# Override routing and validation (documented table)
# ---------------------------------------------------------------------------
# key -> (section, kind, low, high). kind: "rate" (0-1, percent accepted), "grade" (0-100 percentile),
# "value" (as given), "fraction" (signed fraction; |x| in (1, 100] is read as percentage points).
OverrideSpec = tuple[str, str, float, float]


def _group(section: str, kind: str, low: float, high: float, keys: str) -> dict[str, OverrideSpec]:
    return {key: (section, kind, low, high) for key in keys.split()}


OVERRIDE_SPECS: dict[str, OverrideSpec] = {}
for _part in (
    _group("offense", "rate", 0.0, 1.0,
           "ol_pass_block_win_rate ol_run_block_win_rate ol_continuity explosive_pass_rate explosive_run_rate "
           "red_zone_efficiency team_rz_pass_rate"),
    _group("offense", "value", -1.0, 1.0, "team_epa_per_play"),
    _group("offense", "value", 15.0, 60.0, "pace_seconds_per_play"),
    _group("offense", "value", 0.0, 8.0, "wr1_separation"),
    _group("offense", "fraction", -0.5, 0.5, "proe rroe"),
    _group("opponent_defense", "rate", 0.0, 1.0,
           "opp_pressure_rate opp_blitz_rate opp_simulated_pressure_success opp_pass_rush_win_rate "
           "opp_cb1_shadow_rate opp_bracket_coverage_rate opp_man_coverage_rate opp_stacked_box_rate "
           "opp_missed_tackle_rate opp_coverage_bust_rate"),
    _group("opponent_defense", "grade", 0.0, 100.0,
           "opp_coverage_grade opp_run_defense_grade opp_explosive_play_prevention opp_rz_pass_defense_grade "
           "opp_rz_run_defense_grade"),
    _group("opponent_defense", "value", -1.0, 1.0, "opp_def_epa_per_play opp_rz_pass_funnel opp_rz_run_funnel"),
    _group("opponent_defense", "value", 1.0, 32.0, "opp_defense_rank"),
    _group("qb", "rate", 0.0, 1.0, "qb_scramble_rate qb_sack_avoidance qb_deep_attempt_rate qb_rz_efficiency"),
    _group("qb", "value", 1.0, 6.0, "qb_time_to_throw"),
    _group("qb", "value", 0.8, 6.0, "qb_release_time"),
    _group("qb", "value", -3.0, 1.5, "qb_pressure_epa"),
    _group("qb", "value", -1.5, 1.5, "qb_efficiency"),
    _group("game", "value", -30.0, 30.0, "vegas_spread"),
    _group("game", "value", 20.0, 90.0, "vegas_total"),
    _group("game", "value", -1.0, 1.0, "opp_epa_per_play"),
    _group("game", "value", 0.0, 5.0, "ol_injuries"),
    _group("game", "rate", 0.0, 1.0, "ol_health"),
    _group("player", "value", 0.0, 8.0, "wr_separation"),
    _group("player", "value", 0.0, 20.0, "te_route_depth"),
    _group("player", "value", 0.0, 6.0, "rb_yards_after_contact wr_yprr_vs_man wr_yprr_vs_zone wr_expected_yprr_vs_coverage"),
    _group("player", "value", -5.0, 5.0, "wr_yprr_at_opp_mix"),
    _group("player", "rate", 0.0, 1.0,
           "wr_deep_route_rate te_target_share_under_pressure te_rz_target_share rb_breakaway_rate rb_rz_efficiency"),
    _group("player", "grade", 0.0, 100.0,
           "wr_vs_cb_matchup_grade cb_coverage_grade_at_opp_mix slot_wr_vs_nickel_grade te_vs_lb_coverage_grade "
           "cb_man_coverage_grade cb_zone_coverage_grade"),
):
    OVERRIDE_SPECS.update(_part)

#: Computed by the pipeline from its own drivers; accepting a number here would be silently overwritten.
MODEL_OUTPUT_KEYS: frozenset[str] = frozenset({
    "qb_collapse_probability", "collapse_probability", "qb_collapse_prob", "trailing_script_prob",
    "leading_script_prob", "negative_game_script_prob", "positive_script_prob", "negative_script_prob",
    "neutral_script_prob", "explosion_probability", "pass_game_collapse", "run_collapse",
})

_PREFIX_SECTION = (("opp_", "opponent_defense"), ("qb_", "qb"), ("ol_", "offense"), ("wr_", "player"),
                   ("te_", "player"), ("rb_", "player"), ("vegas_", "game"))

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_TTL_OK = 6 * 3600.0
_TTL_EMPTY = 300.0
_CACHE: dict[Any, tuple[float, Any, float]] = {}
_LOCK = threading.RLock()


def _cached(key: Any, loader: Callable[[], Any]) -> Any:
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < hit[2]:
            return hit[1]
    value = loader()
    empty = isinstance(value, pd.DataFrame) and value.empty
    with _LOCK:
        _CACHE[key] = (now, value, _TTL_EMPTY if empty else _TTL_OK)
    return value


def clear_cache() -> None:
    """Drop this module's schedule cache and the team / player loaders' caches (a manual refresh)."""
    with _LOCK:
        _CACHE.clear()
    for loader in (ntf, npf):
        try:
            loader.clear_cache()
        except Exception as error:  # noqa: BLE001 - clearing is best effort
            LOG.debug("clear_cache of %s failed: %s", getattr(loader, "__name__", loader), error)


def _today() -> _dt.date:
    try:
        return ntf._today()
    except Exception:  # noqa: BLE001
        return _dt.date.today()


def _current_season() -> int:
    """The season in progress from September on, else the latest complete one (team loader's rule)."""
    try:
        return int(ntf._default_season())
    except Exception:  # noqa: BLE001
        today = _today()
        return today.year if today.month >= 9 else today.year - 1


def _clean_int(value: Any, low: int, high: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and value != number:
        return None
    return number if low <= number <= high else None


def _clean_bool(value: Any) -> bool | None:
    """``True`` / ``False`` for a real boolean (or 0 / 1, numpy bool); ``None`` for anything else (never ``"false"`` -> True)."""
    if isinstance(value, bool):
        return value
    if value is not None and type(value).__name__ == "bool_":
        return bool(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _mapping(value: Any) -> dict[str, Any]:
    """A shallow ``dict`` view of a mapping or attribute object (``{}`` for anything else)."""
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if value is None or isinstance(value, (str, bytes, int, float)):
        return {}
    out: dict[str, Any] = {}
    for name in ("player_id", "id", "name", "player_name", "team", "position", "opponent"):
        if hasattr(value, name):
            out[name] = getattr(value, name)
    return out


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_ID_RE = re.compile(r"^\d{2}-\d{7}$")


def _name_tokens(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[.'`]", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return [token for token in text.split() if token not in _SUFFIXES]


def _same_person(lead: str | None, name: Any, player_id: Any) -> bool | None:
    """Is the team table's ``lead=`` label (``"P.Mahomes"`` or a GSIS id) this player? ``None`` = cannot tell."""
    if not lead:
        return None
    lead = lead.strip()
    if _ID_RE.match(lead):
        pid = _text(player_id)
        return None if pid is None else (lead == pid)
    a, b = _name_tokens(lead), _name_tokens(name)
    if not a or not b:
        return None
    if a[-1] != b[-1]:
        return False
    return a[0][:1] == b[0][:1]


def _finite(value: Any) -> float | None:
    number = to_float(value)
    return None if number is None else float(number)


# ---------------------------------------------------------------------------
# The I/O boundary (small private functions; tests monkeypatch these)
# ---------------------------------------------------------------------------
_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id", "season", "game_type", "week", "gameday", "home_team", "away_team", "spread_line", "total_line",
)


def _team_table(season: int, through_week: int | None) -> dict[str, dict]:
    """``nfl_team_features.team_feature_table`` (cached and fail-soft there)."""
    return ntf.team_feature_table(season, through_week=through_week)


def _player_table(season: int, through_week: int | None) -> dict[str, dict]:
    """``nfl_player_features.player_feature_table`` (cached and fail-soft there)."""
    return npf.player_feature_table(season, through_week=through_week)


def _load_schedule(season: int) -> pd.DataFrame:
    """One season's schedule (only the id, week, team and line columns). Empty frame on any failure; TTL-cached."""
    def go() -> pd.DataFrame:
        try:
            import nflreadpy as nfl

            data = nfl.load_schedules([int(season)])
            if hasattr(data, "select") and hasattr(data, "columns"):
                keep = [c for c in _SCHEDULE_COLUMNS if c in data.columns]
                if keep:
                    data = data.select(keep)
            if hasattr(data, "to_pandas"):
                data = data.to_pandas()
            return data if isinstance(data, pd.DataFrame) else pd.DataFrame()
        except Exception as error:  # noqa: BLE001 - fail soft by contract
            LOG.debug("schedule %s unavailable: %s", season, error)
            return pd.DataFrame()

    return _cached(("schedule", int(season)), go)


def _status_overlay() -> dict[str, Any]:
    """The project's local player-status file (``fantasy.player_status``); ``{}`` when empty or unreadable."""
    try:
        from fantasy import player_status

        data = player_status.load_player_status()
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception as error:  # noqa: BLE001
        LOG.debug("player-status overlay unavailable: %s", error)
        return {}


def _snap_frame(season: int) -> pd.DataFrame:
    """Prepared regular-season snap counts of ``season`` (``nfl_team_features``' loader); empty on failure."""
    try:
        frame = ntf._prepare_snaps(ntf._load_snaps(int(season)))
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    except Exception as error:  # noqa: BLE001
        LOG.debug("snap counts %s unavailable: %s", season, error)
        return pd.DataFrame()


_OL_WEIGHT: dict[str, float] = {"OUT": 1.0, "DOUBTFUL": 1.0, "HOLDOUT": 1.0, "SUSPENDED": 1.0, "QUESTIONABLE": 0.5}


def _ol_injury_count(team: str, season: int, through_week: int | None) -> tuple[float | None, str]:
    """``(count | None, basis)``: OL starters out for ``team`` from the local overlay x the top-five OL by snaps.

    ``None`` is returned (never ``0``) whenever the overlay says nothing usable about the team's five
    most-used linemen; the basis then explains why. Never raises.
    """
    try:
        return _ol_injury_count_impl(team, season, through_week)
    except Exception as error:  # noqa: BLE001
        LOG.debug("ol injury count failed for %s: %s", team, error)
        return None, f"no player-status file result ({type(error).__name__}: {error})"


def _ol_injury_count_impl(team: str, season: int, through_week: int | None) -> tuple[float | None, str]:
    status = _status_overlay()
    if not status:
        return None, "no player-status file data (file missing or empty)"
    stamps_all = [str(r.get("last_updated")) for r in status.values()
                  if isinstance(r, Mapping) and r.get("last_updated")]
    newest = max(stamps_all) if stamps_all else None
    age = _status_age_days(newest)
    if age is not None and age > STATUS_MAX_AGE_DAYS:
        return None, (f"no player-status file fresh enough: last refreshed {newest} ({age} days ago, limit "
                      f"{STATUS_MAX_AGE_DAYS}); refresh it for current availability")
    if season != _current_season():
        return None, (f"player-status file not applied: it is a snapshot of current availability and the "
                      f"context is for the {season} season (season in progress: {_current_season()})")
    frame = _snap_frame(season)
    need = {"team", "week", "position", "player", "offense_snaps"}
    if frame is None or frame.empty or not need.issubset(frame.columns):
        return None, f"no {season} snap counts available to identify {team}'s five most-used OL"
    week = pd.to_numeric(frame["week"], errors="coerce")
    latest = int(week.max()) if week.notna().any() else 0
    if through_week is not None and int(through_week) < latest:
        return None, (f"player-status file not applied: it is a snapshot of current availability and this "
                      f"context is as of week {int(through_week)} (snap data runs to week {latest})")
    sub = frame[(frame["team"] == team) & frame["position"].astype(str).str.upper().isin(ntf._OL_POSITIONS)]
    if through_week is not None:
        sub = sub[pd.to_numeric(sub["week"], errors="coerce") <= int(through_week)]
    snaps = pd.to_numeric(sub["offense_snaps"], errors="coerce")
    keep = snaps.notna() & (snaps > 0)
    key_col = "pfr_player_id" if "pfr_player_id" in sub.columns else "player"
    sub = sub[keep].assign(_snaps=snaps[keep])
    if sub.empty:
        return None, f"no {season} OL snap counts for {team} through week {through_week}"
    sub = sub.assign(_key=sub[key_col].astype(str))
    per = (sub.groupby("_key").agg(name=("player", "first"), snaps=("_snaps", "sum"))
              .reset_index().sort_values(["snaps", "_key"], ascending=[False, True], kind="mergesort"))
    top = per.head(5)
    if len(top) < 5:
        return None, f"only {len(top)} {team} OL with {season} snaps"

    from fantasy.player_status import normalize_status
    from fantasy.utils import normalize_player_name

    by_name: dict[str, Mapping[str, Any]] = {}
    for key, record in status.items():
        if not isinstance(record, Mapping):
            continue
        norm = key[3:] if str(key).startswith("nm:") else normalize_player_name(record.get("name"))
        if norm:
            by_name.setdefault(norm, record)
    count = 0.0
    flagged: list[str] = []
    for name in top["name"]:
        record = by_name.get(normalize_player_name(name))
        if record is None:
            continue
        label = normalize_status(record.get("status"))
        weight = _OL_WEIGHT.get(label)
        if weight is None:
            continue  # HEALTHY or an unrecognised status string: ignored, not counted
        count += weight
        flagged.append(f"{name} {label}")
    names = ", ".join(str(n) for n in top["name"])
    if not flagged:
        return None, (f"no player-status file row flags any of {team}'s five most-used OL ({names}); a player the "
                      f"file does not list is not proof of health")
    when = f"file refreshed {newest}, {age} days ago" if newest and age is not None else "file refresh time unknown"
    window = f"{season} wk<={int(through_week)}" if through_week is not None else f"{season} season"
    return count, (f"player-status file ({when}): {'; '.join(flagged)} among {team}'s five most-used OL by "
                   f"{window} snaps ({names}); OUT/DOUBTFUL/HOLDOUT/SUSPENDED = 1, QUESTIONABLE = 0.5; "
                   f"players not in the file count as not flagged")


def _status_age_days(stamp: str | None) -> int | None:
    if not stamp:
        return None
    try:
        day = _dt.date.fromisoformat(str(stamp)[:10])
    except ValueError:
        return None
    return max(0, (_today() - day).days)


# ---------------------------------------------------------------------------
# Public small API
# ---------------------------------------------------------------------------
def neutral_context() -> dict[str, Any]:
    """The empty-but-well-formed context: every section ``{}``, ``has_features`` False."""
    return {
        "team": None, "opponent": None, "season": None, "week": None, "through_week": None, "is_home": None,
        "offense": {}, "opponent_defense": {}, "qb": {}, "player": {}, "game": {},
        "sources": {}, "warnings": [], "has_features": False,
        "qb_player": None, "player_match": None,
    }


def context_for_player(player_row: Any, opponent: Any, **kwargs: Any) -> dict[str, Any]:
    """:func:`build_game_context` for a player pool row (``player_id``, ``name``, ``team``, ``position``).

    The row supplies the offence (``team``; an explicit ``team=`` keyword wins) and is passed as
    ``player``. Any other keyword goes straight to :func:`build_game_context`.
    """
    row = _mapping(player_row)
    team = kwargs.pop("team", None) or row.get("team") or row.get("team_abbr")
    if not _text(opponent):
        opponent = row.get("opponent") or row.get("opponent_team")
    kwargs.setdefault("player", row or player_row)
    return build_game_context(team, opponent, **kwargs)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
class _Builder:
    """Mutable state of one assembly: sections, per-key provenance, warnings."""

    def __init__(self) -> None:
        self.ctx = neutral_context()
        self.sections: dict[str, dict[str, float]] = {name: self.ctx[name] for name in SECTIONS}
        self.meta: dict[str, dict[str, dict[str, str]]] = {name: {} for name in SECTIONS}
        self.warnings: list[str] = self.ctx["warnings"]
        self.absent: dict[str, str] = {}

    def warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    def put(self, section: str, key: str, value: Any, kind: str, basis: str) -> bool:
        number = _finite(value)
        if number is None:
            return False
        self.sections[section][key] = number
        self.meta[section][key] = {"kind": kind if kind in _KINDS else "proxy", "basis": basis}
        return True

    def take(self, section: str, values: Any, kinds: Any, basis: Any, label: str) -> None:
        if not isinstance(values, Mapping):
            return
        kinds = kinds if isinstance(kinds, Mapping) else {}
        basis = basis if isinstance(basis, Mapping) else {}
        for key, value in values.items():
            kind = kinds.get(key)
            text = basis.get(key)
            if kind not in _MEASURED:
                kind = "proxy"  # an unlabelled loader value is never presented as measured
                text = f"{label} (the loader gave no kind; treated as proxy)" + (f": {text}" if text else "")
            self.put(section, str(key), value, kind, str(text) if text else f"{label} (no basis string supplied)")

    def drop(self, key: str, *, keep_section: str | None = None) -> None:
        for name in SECTIONS:
            if name != keep_section and key in self.sections[name]:
                del self.sections[name][key]
                self.meta[name].pop(key, None)


def _coerce_override(kind: str, low: float, high: float, raw: Any) -> tuple[float | None, str | None]:
    """``(value, note)``; value ``None`` when ``raw`` is not a usable number inside ``[low, high]``."""
    note = None
    if kind == "rate":
        value = to_rate(raw)
        if value is not None and isinstance(raw, (int, float)) and not isinstance(raw, bool) and 1.0 < float(raw) <= 100.0:
            note = f"read {raw} as a percentage"
        return value, note
    source = raw
    if isinstance(raw, str) and raw.strip().endswith("%") and kind in ("grade", "value"):
        source = raw.strip()[:-1]
    value = to_float(source)
    if value is None:
        return None, None
    if kind == "fraction" and not (isinstance(raw, str) and raw.strip().endswith("%")) and 1.0 < abs(value) <= 100.0:
        value /= 100.0
        note = f"read {raw} as percentage points"
    if not (low <= value <= high):
        return None, None
    return value, note


_ALIAS_MAP: dict[str, str] | None = None
_MODEL_KEYS: set[str] = set()
_MODEL_MODULES = ("qb_collapse_model", "pass_game_collapse", "run_game_collapse", "explosion_model",
                  "explosive_play_model", "coverage_mismatch_model", "red_zone_funnel_model", "game_script_model")


def _alias_map() -> dict[str, str]:
    """``{alias: canonical key}`` read from the eight models' ``FACTORS`` (lazily; a missing model is skipped).

    A string that is a canonical key anywhere (``wr1_separation`` is canonical in one model and an alias
    in another) is never treated as an alias.
    """
    global _ALIAS_MAP
    if _ALIAS_MAP is not None:
        return _ALIAS_MAP
    import importlib

    canonical: set[str] = set(OVERRIDE_SPECS)
    aliases: dict[str, str] = {}
    for name in _MODEL_MODULES:
        try:
            module = importlib.import_module(f"modules.{name}")
            factors = getattr(module, "FACTORS", ())
            if isinstance(factors, Mapping):
                factors = [f for group in factors.values() for f in group]
            for factor in factors:
                canonical.add(factor.key)
                _MODEL_KEYS.add(factor.key)
                for alias in factor.aliases:
                    aliases.setdefault(alias, factor.key)
            split = getattr(module, "_SPLIT_ALIASES", None)       # e.g. pass_game_collapse: yprr_vs_man
            if isinstance(split, Mapping):
                for key, names in split.items():
                    canonical.add(key)
                    _MODEL_KEYS.add(key)
                    for alias in names:
                        aliases.setdefault(alias, key)
            for attr in ("CANONICAL_INPUTS", "NO_FREE_FEED_INPUTS"):
                for key in getattr(module, attr, ()) or ():
                    canonical.add(key)
                    _MODEL_KEYS.add(key)
        except Exception as error:  # noqa: BLE001
            LOG.debug("model %s not scanned for aliases: %s", name, error)
    _ALIAS_MAP = {alias: key for alias, key in aliases.items() if alias not in canonical}
    return _ALIAS_MAP


def _section_for(key: str) -> str | None:
    spec = OVERRIDE_SPECS.get(key)
    if spec:
        return spec[0]
    if key in _MODEL_KEYS:
        for prefix, section in _PREFIX_SECTION:
            if key.startswith(prefix):
                return section
        return "offense"
    return None


def _apply_overrides(b: _Builder, overrides: Any) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        b.warn(f"overrides ignored: expected a mapping of canonical keys to numbers, got {type(overrides).__name__}")
        return
    aliases = _alias_map()
    for raw_key in sorted(overrides, key=lambda k: str(k)):
        raw = overrides[raw_key]
        if raw is None:
            continue  # None never erases or sets anything
        if not isinstance(raw_key, str) or not raw_key.strip():
            b.warn(f"override with a non-text key {raw_key!r} dropped")
            continue
        given = raw_key.strip().lower()
        key = aliases.get(given, given)
        if key in MODEL_OUTPUT_KEYS:
            b.warn(f"override {raw_key!r} dropped: it is a model output the pipeline computes itself; "
                   f"override its drivers instead")
            continue
        section = _section_for(key)
        if section is None:
            b.warn(f"override {raw_key!r} dropped: not a canonical input key")
            continue
        spec = OVERRIDE_SPECS.get(key)
        kind, low, high = (spec[1], spec[2], spec[3]) if spec else ("value", -math.inf, math.inf)
        value, note = _coerce_override(kind, low, high, raw)
        if value is None:
            bounds = f" within [{low:g}, {high:g}]" if spec else ""
            b.warn(f"override {raw_key}={raw!r} dropped: not a usable number{bounds}")
            continue
        replaced = next((f"{name} {b.meta[name][key]['kind']} value {b.sections[name][key]:.4g}"
                         for name in SECTIONS if key in b.sections[name]), None)
        b.drop(key)
        extras = [f"supplied as {raw_key!r}" if key != given else None, note,
                  f"replaces the {replaced}" if replaced else None]
        detail = "; ".join(x for x in extras if x)
        b.put(section, key, value, "override", f"user override {raw!r}" + (f" ({detail})" if detail else ""))


def _dedupe(b: _Builder) -> None:
    """Where a key sits in several sections with different numbers, the later section (merge order) wins."""
    for i, name in enumerate(SECTIONS):
        for key in list(b.sections[name]):
            for later in SECTIONS[i + 1:]:
                if key in b.sections[later] and not math.isclose(
                        b.sections[later][key], b.sections[name][key], rel_tol=1e-9, abs_tol=1e-12):
                    del b.sections[name][key]
                    b.meta[name].pop(key, None)
                    break


def _dropbacks(entry: Mapping[str, Any]) -> int:
    basis = entry.get("basis")
    if isinstance(basis, Mapping):
        for key in ("qb_efficiency", "qb_scramble_rate", "qb_sack_avoidance"):
            match = re.search(r"\bn=(\d+)", str(basis.get(key) or ""))
            if match:
                return int(match.group(1))
    return 0


def _lookup_player(table: Mapping[str, Any], *, player_id: str | None, name: str | None, team: str | None,
                   position: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """``(entry, note)`` through ``find_player_features``: strict first, then without the team filter.

    A name that matches more than one player never resolves (``find_player_features`` returns ``None``).
    """
    if not table or not (player_id or name):
        return None, None
    entry = npf.find_player_features(table, player_id=player_id, name=name, team=team, position=position)
    if entry is not None:
        return entry, None
    if team and name:
        entry = npf.find_player_features(table, player_id=player_id, name=name, team=None, position=position)
        if entry is not None:
            return entry, f"listed with {entry.get('team') or 'another team'} in the feature table, not {team}"
    return None, None


def _player_ref(value: Any) -> dict[str, str | None]:
    if isinstance(value, str):
        text = _text(value)
        return {"player_id": text if text and _ID_RE.match(text) else None,
                "name": None if (text and _ID_RE.match(text)) else text, "team": None, "position": None}
    row = _mapping(value)
    return {
        "player_id": _text(row.get("player_id") or row.get("id") or row.get("gsis_id")),
        "name": _text(row.get("name") or row.get("player_name") or row.get("display_name")),
        "team": _text(row.get("team") or row.get("team_abbr")),
        "position": _text(row.get("position")),
    }


def _describe(entry: Mapping[str, Any], how: str) -> dict[str, Any]:
    return {"player_id": entry.get("player_id"), "name": entry.get("name"), "team": entry.get("team"),
            "position": entry.get("position"), "games": entry.get("games"), "how": how}


def _put_player_features(b: _Builder, section: str, entry: Mapping[str, Any]) -> None:
    b.take(section, entry.get("features"), entry.get("kinds"), entry.get("basis"),
           f"player table {entry.get('name') or entry.get('player_id')}")


def _schedule_game(frame: Any, team: str, opponent: str, week: int | None, season_in_progress: bool
                   ) -> tuple[dict[str, Any] | None, str]:
    """``(row, reason)``: the schedule row for the two teams. Only ids, teams, week, date and the two
    game lines are read -- never a score or a result."""
    need = {"home_team", "away_team", "week"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not need.issubset(frame.columns):
        return None, "schedule unavailable"
    frame = frame.reset_index(drop=True)
    home = frame["home_team"].map(ntf.normalize_team)
    away = frame["away_team"].map(ntf.normalize_team)
    pair = ((home == team) & (away == opponent)) | ((home == opponent) & (away == team))
    weeks = pd.to_numeric(frame["week"], errors="coerce")
    if week is not None:
        pair &= weeks == week
    elif "game_type" in frame.columns:
        pair &= frame["game_type"].astype(str).str.upper() == "REG"
    rows = frame[pair]
    if rows.empty:
        return None, f"no schedule row for {team} vs {opponent}" + (f" in week {week}" if week is not None else "")
    if len(rows) == 1:
        index = rows.index[0]
    else:
        # a divisional pair meets twice: without a week, only the next meeting of the season in progress is unambiguous
        index = None
        if season_in_progress and "gameday" in rows.columns:
            days = pd.to_datetime(rows["gameday"], errors="coerce")
            upcoming = days[days >= pd.Timestamp(_today())].sort_values(kind="mergesort")
            if not upcoming.empty:
                index = upcoming.index[0]
        if index is None:
            return None, f"{team} and {opponent} meet more than once; pass week to choose the game"
    row = frame.loc[index]
    return {
        "home": home.loc[index],
        "away": away.loc[index],
        "spread_line": _finite(row.get("spread_line")),
        "total_line": _finite(row.get("total_line")),
        "week": _clean_int(row.get("week"), 0, 25),
        "game_id": _text(row.get("game_id")),
    }, ""


def _build(team: Any, opponent: Any, season: Any, week: Any, is_home: Any, vegas_spread: Any, vegas_total: Any,
           player: Any, qb: Any, overrides: Any, through_week: Any) -> dict[str, Any]:
    b = _Builder()
    ctx = b.ctx
    warn = b.warn

    # ---- identity -------------------------------------------------------------------------------
    t, o = ntf.normalize_team(team), ntf.normalize_team(opponent)
    ctx["team"] = t or _text(team)
    ctx["opponent"] = o or _text(opponent)
    usable = True
    for label, raw, norm in (("team", team, t), ("opponent", opponent, o)):
        if norm is None:
            warn(f"unrecognised {label} {raw!r}: team and opponent features skipped")
            usable = False
    if usable and t == o:
        warn(f"team and opponent are both {t}: team and opponent features skipped")
        usable = False

    current = _current_season()
    if season is None:
        year = current
    else:
        year = _clean_int(season, 1999, current + 1)
        if year is None:
            warn(f"season {season!r} is not usable; using {current}")
            year = current
    ctx["season"] = year

    wk = None
    if week is not None:
        wk = _clean_int(week, 1, 22)
        if wk is None:
            warn(f"week {week!r} is not usable and was ignored")
    ctx["week"] = wk
    tw = None
    if through_week is not None:
        tw = _clean_int(through_week, 0, 22)
        if tw is None:
            warn(f"through_week {through_week!r} is not usable and was ignored")
    if tw is None and wk is not None:
        tw = max(0, wk - 1)
    elif tw is not None and wk is not None and tw >= wk:
        warn(f"through_week {tw} is not before week {wk}: the features include games at or after the one "
             f"being projected (look-ahead)")
    ctx["through_week"] = tw
    ctx["is_home"] = _clean_bool(is_home)

    # ---- team table -------------------------------------------------------------------------------
    games: dict[str, int | None] = {}
    team_row: Mapping[str, Any] = {}
    lead_label: str | None = None
    if usable:
        table: Any = {}
        try:
            table = _team_table(year, tw)
        except Exception as error:  # noqa: BLE001
            warn(f"team feature loader failed ({type(error).__name__}: {error}); team features unavailable")
        table = table if isinstance(table, Mapping) else {}
        if not table:
            warn("team features unavailable (the loader returned nothing); offence and defence left neutral")
        else:
            team_row = table.get(t) if isinstance(table.get(t), Mapping) else {}
            opp_row = table.get(o) if isinstance(table.get(o), Mapping) else {}
            if team_row:
                b.take("offense", team_row.get("offense"), team_row.get("kinds"), team_row.get("basis"),
                       f"team table {t} offence")
            else:
                warn(f"no team features for {t}; offence left neutral")
            if opp_row:
                b.take("opponent_defense", opp_row.get("defense"), opp_row.get("kinds"), opp_row.get("basis"),
                       f"team table {o} defence")
                off = opp_row.get("offense") if isinstance(opp_row.get("offense"), Mapping) else {}
                if _finite(off.get("team_epa_per_play")) is not None:
                    kind = (opp_row.get("kinds") or {}).get("team_epa_per_play", "real")
                    text = (opp_row.get("basis") or {}).get("team_epa_per_play") or "team table"
                    b.put("game", "opp_epa_per_play", off["team_epa_per_play"], kind if kind in _MEASURED else "proxy",
                          f"opponent {o} offence team_epa_per_play (for the game-script model): {text}")
            else:
                warn(f"no team features for {o}; opposing defence left neutral")
            games = {t: team_row.get("games") if team_row else None, o: opp_row.get("games") if opp_row else None}
            low = [f"{k} {int(v)} g" for k, v in games.items() if isinstance(v, (int, float)) and v < 4]
            if low:
                warn("early season (" + ", ".join(low) + "): team features lean on last season's values (the "
                     "loader shrinks toward the prior season) and player features are heavily shrunk toward the "
                     "league mean")
            basis_map = team_row.get("basis") if team_row else None
            if isinstance(basis_map, Mapping):
                for key in ("qb_efficiency", "qb_deep_attempt_rate"):
                    match = re.search(r"lead=([^;]+);", str(basis_map.get(key) or ""))
                    if match:
                        lead_label = match.group(1).strip()
                        break

    # ---- player table: quarterback and projected player ---------------------------------------------
    ref = _player_ref(player) if player is not None else None
    qb_ref = _player_ref(qb) if qb is not None else None
    ptable: Any = {}
    if usable:
        try:
            ptable = _player_table(year, tw)
        except Exception as error:  # noqa: BLE001
            warn(f"player feature loader failed ({type(error).__name__}: {error}); player features unavailable")
        ptable = ptable if isinstance(ptable, Mapping) else {}
        if not ptable:
            warn("player features unavailable (the loader returned nothing"
                 + ("; nothing precedes week 1" if tw == 0 else "") + "); quarterback and player left neutral")

    player_entry: dict[str, Any] | None = None
    if ref is not None and (ref["player_id"] or ref["name"]):
        label = ref["name"] or ref["player_id"]
        if ptable:
            player_entry, note = _lookup_player(ptable, player_id=ref["player_id"], name=ref["name"],
                                                team=ref["team"] or t, position=ref["position"])
            if player_entry is None:
                warn(f"no player features for {label!r}: not in the feature table, below its volume gates, or the "
                     f"name matches more than one player")
            else:
                if note:
                    warn(f"player {label}: {note}")
                elif t and ntf.normalize_team(player_entry.get("team")) not in (None, t):
                    warn(f"player {label} is listed with {player_entry.get('team')}, not {t}")
                _put_player_features(b, "player", player_entry)
                ctx["player_match"] = _describe(player_entry, "matched by id" if ref["player_id"] and
                                                ref["player_id"] == player_entry.get("player_id") else "matched by name")
    elif player is not None:
        warn("player ignored: it carried neither a player_id nor a name")

    qb_entry: dict[str, Any] | None = None
    qb_named = False
    if player_entry is not None and str(player_entry.get("position", "")).upper() == "QB":
        qb_entry = player_entry
        qb_named = True
        if qb_ref and (qb_ref["player_id"] or qb_ref["name"]):
            same = (qb_ref["player_id"] and qb_ref["player_id"] == player_entry.get("player_id")) or (
                qb_ref["name"] and _same_person(qb_ref["name"], player_entry.get("name"), player_entry.get("player_id")))
            if not same:
                warn("qb argument ignored: the projected player is himself a quarterback")
        how = "the projected player"
    elif qb_ref and (qb_ref["player_id"] or qb_ref["name"]):
        qb_named = True
        if ptable:
            qb_entry, note = _lookup_player(ptable, player_id=qb_ref["player_id"], name=qb_ref["name"],
                                            team=qb_ref["team"] or t, position="QB")
            if qb_entry is None:
                warn(f"quarterback {qb_ref['name'] or qb_ref['player_id']!r} not found (not in the feature table, "
                     f"below its volume gates, or the name matches more than one player); quarterback features omitted")
            elif note:
                warn(f"quarterback {qb_ref['name'] or qb_ref['player_id']}: {note}")
        how = "named by the caller"
    else:
        how = "most dropbacks on the team"
        if ptable and t:
            candidates = [(pid, e) for pid, e in ptable.items() if isinstance(e, Mapping)
                          and str(e.get("position", "")).upper() == "QB" and ntf.normalize_team(e.get("team")) == t]
            if candidates:
                pid, e = min(candidates, key=lambda c: (-_dropbacks(c[1]), -int(c[1].get("games") or 0),
                                                         -len(c[1].get("features") or {}), str(c[0])))
                qb_entry = dict(copy.deepcopy(dict(e)), player_id=str(pid))
            else:
                warn(f"no {t} quarterback in the player feature table; quarterback features omitted")
    if qb_entry is not None:
        _put_player_features(b, "qb", qb_entry)
        ctx["qb_player"] = _describe(qb_entry, how)
        # the team table's QB-anchored keys describe the team's primary QB: not this one -> drop them
        if _same_person(lead_label, qb_entry.get("name"), qb_entry.get("player_id")) is False:
            for key in ("qb_efficiency", "qb_deep_attempt_rate"):
                if key in b.sections["offense"]:
                    del b.sections["offense"][key]
                    b.meta["offense"].pop(key, None)
            warn(f"team-level qb_efficiency / qb_deep_attempt_rate describe {lead_label}, not "
                 f"{qb_entry.get('name') or qb_entry.get('player_id')}; dropped")
    elif qb_named and usable:
        for key in ("qb_efficiency", "qb_deep_attempt_rate"):
            if key in b.sections["offense"]:
                del b.sections["offense"][key]
                b.meta["offense"].pop(key, None)

    # ---- market lines -------------------------------------------------------------------------------
    spread = _supplied_line(b, "vegas_spread", vegas_spread, OVERRIDE_SPECS["vegas_spread"])
    total = _supplied_line(b, "vegas_total", vegas_total, OVERRIDE_SPECS["vegas_total"])
    need_schedule = usable and (is_home is None or (USE_SCHEDULE_LINES and (spread is None or total is None)))
    if need_schedule:
        frame = pd.DataFrame()
        try:
            frame = _load_schedule(year)
        except Exception as error:  # noqa: BLE001
            warn(f"schedule loader failed ({type(error).__name__}: {error})")
        game, reason = _schedule_game(frame, t, o, wk, year == current)
        if game is None:
            if USE_SCHEDULE_LINES and (spread is None or total is None):
                warn(f"market lines not set: {reason}")
        else:
            home = game["home"]
            if home in (t, o):
                schedule_home = home == t
                given_home = _clean_bool(is_home)
                if given_home is not None and given_home != schedule_home:
                    warn(f"is_home={given_home} disagrees with the schedule ({game['away']} at {home}); schedule used")
                ctx["is_home"] = schedule_home
            where = f"nflverse schedule {year} wk{game['week']} {game['away']}@{game['home']}"
            if USE_SCHEDULE_LINES:
                line = game["spread_line"]
                if spread is None and line is not None and home in (t, o):
                    value = (-line if home == t else line) + 0.0
                    if b.put("game", "vegas_spread", value, "real",
                             f"{where}: spread_line {line:+g} (positive = home favoured) -> {t} {value:+g} "
                             f"(sportsbook sign: negative = {t} favoured)"):
                        spread = value
                elif spread is None:
                    warn(f"market lines not set: the schedule row for {t} vs {o} has no spread_line yet")
                tline = game["total_line"]
                if total is None and tline is not None:
                    if b.put("game", "vegas_total", tline, "real", f"{where}: total_line {tline:g}"):
                        total = tline
                elif total is None:
                    warn(f"market lines not set: the schedule row for {t} vs {o} has no total_line yet")

    # ---- offensive-line injuries -----------------------------------------------------------------------
    if usable:
        count: float | None = None
        basis = "not looked up"
        try:
            count, basis = _ol_injury_count(t, year, tw)
        except Exception as error:  # noqa: BLE001
            basis = f"no player-status file result ({type(error).__name__}: {error})"
        count = _finite(count)
        if count is not None and 0.0 <= count <= 5.0:
            b.put("game", "ol_injuries", count, "real", str(basis))
        else:
            b.absent["ol_injuries"] = str(basis)

    # ---- overrides, derived values, precedence --------------------------------------------------------
    _apply_overrides(b, overrides)
    if "ol_injuries" in b.absent and "ol_injuries" not in b.sections["game"]:
        warn(f"ol_injuries left neutral: {b.absent['ol_injuries']}")
    injuries = b.sections["game"].get("ol_injuries")
    if "ol_health" not in b.sections["game"]:
        if injuries is not None:
            health = max(0.0, min(1.0, 1.0 - injuries / 5.0))
            b.put("game", "ol_health", health, "derived",
                  f"1 - ol_injuries/5 (ol_injuries {injuries:g}: {b.meta['game']['ol_injuries']['basis']})")
        else:
            b.absent["ol_health"] = "no ol_injuries (no player-status file data for it), so ol_health is not derived"
    _dedupe(b)

    sources: dict[str, dict[str, str]] = {}
    for name in SECTIONS:
        for key in b.sections[name]:
            sources[key] = dict(b.meta[name][key])
    for key, reason in b.absent.items():
        if key not in sources:
            sources[key] = {"kind": "default", "basis": f"neutral: {reason}"}
    if usable:
        why = "no {} supplied" + (" or found in the schedule" if USE_SCHEDULE_LINES else
                                  " (schedule lines are switched off: USE_SCHEDULE_LINES)")
        for key, label in (("vegas_spread", "spread"), ("vegas_total", "total")):
            if key not in sources:
                sources[key] = {"kind": "default", "basis": "neutral: " + why.format(label)}
    ctx["sources"] = sources
    ctx["has_features"] = any(src["kind"] in _MEASURED for src in sources.values())
    return ctx


def _supplied_line(b: _Builder, key: str, raw: Any, spec: OverrideSpec) -> float | None:
    """A caller-supplied ``vegas_*`` argument, range-checked; recorded as an override-labelled input."""
    if raw is None:
        return None
    value, _ = _coerce_override(spec[1], spec[2], spec[3], raw)
    if value is None:
        b.warn(f"{key}={raw!r} is not a usable number within [{spec[2]:g}, {spec[3]:g}]; ignored")
        return None
    sign = "; sportsbook sign: negative = this team favoured" if key == "vegas_spread" else ""
    b.put("game", key, value, "override", f"supplied by the caller ({key} argument{sign})")
    return value


def build_game_context(
    team: Any,
    opponent: Any,
    *,
    season: int | None = None,
    week: int | None = None,
    is_home: bool | None = None,
    vegas_spread: float | None = None,
    vegas_total: float | None = None,
    player: Any = None,
    qb: Any = None,
    overrides: Mapping[str, Any] | None = None,
    through_week: int | None = None,
) -> dict[str, Any]:
    """The canonical-feature context for ``team``'s offence against ``opponent``'s defence (see module docstring).

    ``team`` is the offence being projected and ``opponent`` the defence it faces (any common spelling;
    ``LAR`` folds to ``LA``). ``season`` defaults to the season in progress from September on, else the
    latest complete one. ``through_week`` defaults to ``week - 1`` when ``week`` is given. ``vegas_*`` use
    the sportsbook sign (-7 = this team favoured by 7) and are looked up in the schedule when absent.
    ``overrides`` (canonical keys -> numbers; routed by :data:`OVERRIDE_SPECS`) always win and are
    labelled ``kind="override"``. Never raises; inputs are never mutated.
    """
    try:
        return _build(team, opponent, season, week, is_home, vegas_spread, vegas_total, player, qb,
                      copy.deepcopy(dict(overrides)) if isinstance(overrides, Mapping) else overrides, through_week)
    except Exception as error:  # noqa: BLE001 - fail soft by contract
        LOG.warning("build_game_context failed: %s", error, exc_info=LOG.isEnabledFor(logging.DEBUG))
        ctx = neutral_context()
        ctx["team"], ctx["opponent"] = _text(team), _text(opponent)
        ctx["warnings"].append(f"context could not be assembled ({type(error).__name__}: {error}); all features neutral")
        return ctx


__all__ = [
    "MODEL_OUTPUT_KEYS",
    "OVERRIDE_SPECS",
    "SECTIONS",
    "STATUS_MAX_AGE_DAYS",
    "USE_SCHEDULE_LINES",
    "build_game_context",
    "clear_cache",
    "context_for_player",
    "neutral_context",
]
