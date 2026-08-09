"""Fast, outage-safe NFL player statistics and percentile normalization.

The module is the NFL equivalent of ``graph_data``: provider access is kept at
one boundary, expensive season loads are cached, and every consumer receives a
canonical table.  Live data comes from nflverse through ``nflreadpy``.  A small
last-known snapshot keeps the UI useful when the public dataset is unavailable;
the ``warnings`` attribute always makes that fallback explicit.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd

from modules.data_quality import fuzzy_name_match, normalize_text, safe_number
from modules.nfl import NFL_TEAMS, latest_completed_nfl_season


NFL_TEAM_COLORS = {
    "ARI":"#97233F","ATL":"#A71930","BAL":"#241773","BUF":"#00338D",
    "CAR":"#0085CA","CHI":"#0B162A","CIN":"#FB4F14","CLE":"#311D00",
    "DAL":"#003594","DEN":"#FB4F14","DET":"#0076B6","GB":"#203731",
    "HOU":"#03202F","IND":"#002C5F","JAX":"#006778","KC":"#E31837",
    "LA":"#003594","LAC":"#0080C6","LV":"#000000","MIA":"#008E97",
    "MIN":"#4F2683","NE":"#002244","NO":"#D3BC8D","NYG":"#0B2265",
    "NYJ":"#125740","PHI":"#004C54","PIT":"#FFB612","SEA":"#002244",
    "SF":"#AA0000","TB":"#D50A0A","TEN":"#0C2340","WAS":"#5A1414",
}
NFL_TEAM_SECONDARY = {
    "ARI":"#FFB612","ATL":"#000000","BAL":"#9E7C0C","BUF":"#C60C30",
    "CAR":"#101820","CHI":"#C83803","CIN":"#000000","CLE":"#FF3C00",
    "DAL":"#869397","DEN":"#002244","DET":"#B0B7BC","GB":"#FFB612",
    "HOU":"#A71930","IND":"#A2AAAD","JAX":"#D7A22A","KC":"#FFB81C",
    "LA":"#FFA300","LAC":"#FFC20E","LV":"#A5ACAF","MIA":"#FC4C02",
    "MIN":"#FFC62F","NE":"#C60C30","NO":"#101820","NYG":"#A71930",
    "NYJ":"#000000","PHI":"#A5ACAF","PIT":"#101820","SEA":"#69BE28",
    "SF":"#B3995D","TB":"#FF7900","TEN":"#4B92DB","WAS":"#FFB612",
}

POSITION_METRICS = {
    "QB": ["epa_per_play","cpoe","pressure_to_sack_rate","deep_accuracy","red_zone_efficiency","scramble_rate","time_to_throw","turnover_worthy_play_rate","explosive_pass_rate"],
    "RB": ["yards_after_contact","missed_tackles_forced","explosive_run_rate","target_share","red_zone_share","snap_share","rush_success_rate","yards_per_route_run","pass_block_win_rate"],
    "WR": ["target_share","air_yards_share","yards_per_route_run","separation","contested_catch_rate","explosive_play_rate","red_zone_target_rate","route_win_rate","average_depth_of_target"],
    "TE": ["target_share","air_yards_share","yards_per_route_run","separation","contested_catch_rate","explosive_play_rate","red_zone_target_rate","route_win_rate","average_depth_of_target"],
    "DEF": ["pressure_rate","blitz_rate","sack_probability","coverage_grade","run_stop_win_rate","explosive_plays_allowed","epa_allowed","pass_rush_win_rate"],
}
LOWER_IS_BETTER = {"pressure_to_sack_rate", "time_to_throw", "turnover_worthy_play_rate", "explosive_plays_allowed", "epa_allowed"}

PBP_PLAYER_COLUMNS = [
    "game_id","season","season_type","week","posteam","defteam","play_type",
    "passer_player_id","passer_player_name","rusher_player_id","rusher_player_name",
    "receiver_player_id","receiver_player_name","complete_pass","incomplete_pass",
    "pass_attempt","rush_attempt","sack","qb_scramble","interception","fumble_lost",
    "yards_gained","air_yards","yards_after_catch","epa","success","touchdown",
    "pass_touchdown","rush_touchdown","first_down","yardline_100","shotgun","no_huddle",
    "qb_hit","was_pressure","time_to_throw","cp","cpoe","home_team","away_team",
    "roof","surface","temp","wind","vegas_total","spread_line",
]

VALID_PLAYER_POSITIONS = {"QB", "RB", "WR", "TE", "DEF"}
POSITION_ALIASES = {
    "FB": "RB", "HB": "RB", "TB": "RB",
    "DL": "DEF", "DE": "DEF", "DT": "DEF", "NT": "DEF",
    "LB": "DEF", "ILB": "DEF", "OLB": "DEF",
    "CB": "DEF", "DB": "DEF", "FS": "DEF", "SS": "DEF", "S": "DEF",
    "DST": "DEF", "D/ST": "DEF",
}
KNOWN_WIDE_RECEIVERS = {
    "puka nacua", "deebo samuel", "tyreek hill", "justin jefferson",
    "ja marr chase", "ceedee lamb",
}
KNOWN_TIGHT_ENDS = {"travis kelce", "george kittle"}

# Projection code consumes these names only. The aliases keep nflverse schema
# changes and the bundled legacy snapshot at the ingestion boundary.
PROJECTION_FIELDS_BY_POSITION = {
    "QB": ("pass_attempts", "pass_yards", "pass_tds", "ints", "rush_attempts", "rush_yards", "rush_tds"),
    "RB": ("rush_attempts", "rush_yards", "rush_tds", "targets", "receptions", "rec_yards", "rec_tds"),
    "WR": ("targets", "receptions", "rec_yards", "rec_tds", "air_yards", "routes_run", "red_zone_targets"),
    "TE": ("targets", "receptions", "rec_yards", "rec_tds", "air_yards", "routes_run", "red_zone_targets"),
    "DEF": (),
}
PROJECTION_FIELD_ALIASES = {
    "pass_attempts": ("pass_attempts", "attempts"),
    "pass_yards": ("pass_yards", "passing_yards"),
    "pass_tds": ("pass_tds", "passing_tds"),
    "ints": ("ints", "interceptions"),
    "rush_attempts": ("rush_attempts", "carries"),
    "rush_yards": ("rush_yards", "rushing_yards"),
    "rush_tds": ("rush_tds", "rushing_tds"),
    "targets": ("targets",),
    "receptions": ("receptions",),
    "rec_yards": ("rec_yards", "receiving_yards"),
    "rec_tds": ("rec_tds", "receiving_tds"),
    "air_yards": ("air_yards", "receiving_air_yards"),
    "routes_run": ("routes_run", "routes"),
    "red_zone_targets": ("red_zone_targets",),
}


def normalize_player_position(value: Any) -> str | None:
    """Normalize an nflverse ``player_position`` value to the public schema."""
    raw = str(value or "").strip().upper()
    if raw in VALID_PLAYER_POSITIONS:
        return raw
    return POSITION_ALIASES.get(raw)


def detect_player_position(player: dict[str, Any] | pd.Series) -> str:
    """Resolve position from canonical nflverse data, then limited stat fallback.

    No touch/depth-chart heuristic runs when ``player_position`` is present.
    When it is absent, pass attempts, target involvement, and carry involvement
    provide the documented fallback.
    """
    row = player.to_dict() if isinstance(player, pd.Series) else dict(player)
    primary = normalize_player_position(row.get("player_position"))
    if primary:
        return primary
    name = normalize_text(row.get("player", row.get("player_name", "")))
    if name in KNOWN_WIDE_RECEIVERS:
        return "WR"
    if name in KNOWN_TIGHT_ENDS:
        return "TE"
    if str(row.get("player_id", "")).upper().startswith("DEF-") or " defense" in f" {name}":
        return "DEF"
    pass_attempts = safe_number(row.get("attempts", row.get("pass_attempts")))
    target_share = safe_number(row.get("target_share"))
    targets = safe_number(row.get("targets"))
    carry_share = safe_number(row.get("carry_share", row.get("rush_share")))
    carries = safe_number(row.get("carries", row.get("rush_attempts")))
    if pass_attempts > 0:
        return "QB"
    if target_share > 0 or targets > carries:
        return "WR"
    if carry_share > 0 or carries > 0:
        return "RB"
    if targets > 0:
        return "WR"
    return "WR"



def _usable_stat(row: dict[str, Any], aliases: tuple[str, ...]) -> tuple[float, bool]:
    """Return the first finite numeric alias and whether it truly existed."""
    for alias in aliases:
        value = row.get(alias)
        try:
            missing = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = True
        if not missing:
            return safe_number(value), True
    return 0.0, False


def canonical_player_stats(player: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Expose one position-correct, non-null projection stats dictionary.

    Missing route and red-zone volume can be estimated from stable season rates,
    but derivation is reported as low confidence instead of being hidden.
    """
    row = player.to_dict() if isinstance(player, pd.Series) else dict(player)
    position = detect_player_position(row)
    clean = dict(row)
    present: dict[str, bool] = {}
    for field, aliases in PROJECTION_FIELD_ALIASES.items():
        clean[field], present[field] = _usable_stat(row, aliases)

    previous_derived = row.get("projection_derived_fields", [])
    derived: list[str] = list(previous_derived) if isinstance(previous_derived, list) else []
    if not present["air_yards"] and clean["targets"] > 0:
        clean["air_yards"] = clean["targets"] * safe_number(row.get("average_depth_of_target"))
        derived.append("air_yards")
    if not present["routes_run"] and clean["targets"] > 0:
        yprr = safe_number(row.get("yards_per_route_run"))
        clean["routes_run"] = clean["rec_yards"] / yprr if yprr > 0 else clean["targets"] / .22
        derived.append("routes_run")
    if not present["red_zone_targets"] and clean["targets"] > 0:
        clean["red_zone_targets"] = clean["targets"] * safe_number(row.get("red_zone_target_rate")) / 100.0
        derived.append("red_zone_targets")

    required = PROJECTION_FIELDS_BY_POSITION.get(position, ())
    previous_missing = row.get("projection_missing_fields", [])
    missing = list(previous_missing) if isinstance(previous_missing, list) else []
    missing.extend(field for field in required if not present.get(field, False) and field not in derived)
    missing = list(dict.fromkeys(missing))
    clean.update({
        "position": position,
        "player_position": position,
        "projection_missing_fields": missing,
        "projection_derived_fields": derived,
        "projection_low_confidence": bool(missing or derived),
        "projection_data_confidence": round(max(0.0, 1.0 - (len(missing) + .5 * len(derived)) / max(len(required), 1)), 3),
    })
    return clean


def _add_projection_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical projection schema to every ingested player row."""
    attrs = dict(frame.attrs)
    table = pd.DataFrame([canonical_player_stats(row) for _, row in frame.iterrows()])
    for field in PROJECTION_FIELD_ALIASES:
        if field not in table:
            table[field] = 0.0
        table[field] = pd.to_numeric(table[field], errors="coerce").fillna(0.0)
    table.attrs.update(attrs)
    return table

def canonicalize_player_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee canonical position and identity columns at ingestion."""
    table = frame.copy()
    if "player_position" not in table:
        # Compatibility conversion happens only at the ingestion boundary.
        table["player_position"] = table["position"] if "position" in table else None
    table["player_position"] = [detect_player_position(row) for _, row in table.iterrows()]
    table["position"] = table["player_position"]  # read-only compatibility alias
    return _add_projection_fields(table)


def team_palette(team: str) -> tuple[str, str]:
    key = str(team).upper()
    return NFL_TEAM_COLORS.get(key, "#334155"), NFL_TEAM_SECONDARY.get(key, "#94A3B8")


def _fallback_rows() -> list[dict[str, Any]]:
    """Last-known representative profiles used only during provider outages."""
    return [
        {"player":"Patrick Mahomes","player_id":"00-0033873","team":"KC","position":"QB","games":16,"plays":610,"epa_per_play":.19,"cpoe":2.8,"pressure_to_sack_rate":13.0,"deep_accuracy":42.0,"red_zone_efficiency":62.0,"scramble_rate":6.2,"time_to_throw":2.86,"turnover_worthy_play_rate":2.8,"explosive_pass_rate":11.8,"attempts":565,"completions":378,"passing_yards":3928,"passing_tds":26,"interceptions":11,"carries":58,"rushing_yards":307,"rushing_tds":2,"fantasy_points_per_game":20.1,"snap_share":96.0},
        {"player":"Josh Allen","player_id":"00-0034857","team":"BUF","position":"QB","games":17,"plays":640,"epa_per_play":.24,"cpoe":3.5,"pressure_to_sack_rate":10.0,"deep_accuracy":45.0,"red_zone_efficiency":69.0,"scramble_rate":8.4,"time_to_throw":2.91,"turnover_worthy_play_rate":3.0,"explosive_pass_rate":12.5,"attempts":483,"completions":307,"passing_yards":3731,"passing_tds":28,"interceptions":6,"carries":102,"rushing_yards":531,"rushing_tds":12,"fantasy_points_per_game":23.1,"snap_share":96.5},
        {"player":"Lamar Jackson","player_id":"00-0034796","team":"BAL","position":"QB","games":17,"plays":625,"epa_per_play":.27,"cpoe":4.2,"pressure_to_sack_rate":16.0,"deep_accuracy":48.0,"red_zone_efficiency":67.0,"scramble_rate":10.8,"time_to_throw":3.02,"turnover_worthy_play_rate":2.4,"explosive_pass_rate":13.7,"attempts":474,"completions":316,"passing_yards":4172,"passing_tds":41,"interceptions":4,"carries":139,"rushing_yards":915,"rushing_tds":4,"fantasy_points_per_game":25.2,"snap_share":95.0},
        {"player":"Saquon Barkley","player_id":"00-0034844","team":"PHI","position":"RB","games":16,"plays":520,"yards_after_contact":3.4,"missed_tackles_forced":62,"explosive_run_rate":14.0,"target_share":9.0,"red_zone_share":55.0,"snap_share":76.0,"rush_success_rate":47.0,"yards_per_route_run":1.15,"pass_block_win_rate":82.0,"carries":345,"targets":43,"receptions":33,"rushing_yards":2005,"receiving_yards":278,"total_tds":15,"fantasy_points_per_game":22.2},
        {"player":"Christian McCaffrey","player_id":"00-0033280","team":"SF","position":"RB","games":16,"plays":505,"yards_after_contact":3.1,"missed_tackles_forced":56,"explosive_run_rate":12.5,"target_share":20.0,"red_zone_share":61.0,"snap_share":82.0,"rush_success_rate":49.0,"yards_per_route_run":1.95,"pass_block_win_rate":84.0,"carries":272,"targets":83,"receptions":67,"rushing_yards":1459,"receiving_yards":564,"total_tds":21,"fantasy_points_per_game":24.5},
        {"player":"Bijan Robinson","player_id":"00-0038569","team":"ATL","position":"RB","games":17,"plays":500,"yards_after_contact":3.0,"missed_tackles_forced":68,"explosive_run_rate":11.7,"target_share":16.0,"red_zone_share":48.0,"snap_share":75.0,"rush_success_rate":46.0,"yards_per_route_run":1.45,"pass_block_win_rate":80.0,"carries":304,"targets":72,"receptions":61,"rushing_yards":1456,"receiving_yards":431,"total_tds":15,"fantasy_points_per_game":20.0},
        {"player":"Justin Jefferson","player_id":"00-0036322","team":"MIN","position":"WR","games":17,"plays":570,"target_share":29.0,"air_yards_share":42.0,"yards_per_route_run":2.65,"separation":3.1,"contested_catch_rate":55.0,"explosive_play_rate":18.0,"red_zone_target_rate":23.0,"route_win_rate":51.0,"average_depth_of_target":12.8,"targets":154,"receptions":103,"receiving_yards":1533,"receiving_tds":10,"fantasy_points_per_game":18.7,"snap_share":88.0},
        {"player":"Ja'Marr Chase","player_id":"00-0036900","team":"CIN","position":"WR","games":17,"plays":575,"target_share":31.0,"air_yards_share":40.0,"yards_per_route_run":2.75,"separation":3.2,"contested_catch_rate":58.0,"explosive_play_rate":19.0,"red_zone_target_rate":25.0,"route_win_rate":53.0,"average_depth_of_target":11.7,"targets":175,"receptions":127,"receiving_yards":1708,"receiving_tds":17,"fantasy_points_per_game":23.0,"snap_share":90.0},
        {"player":"CeeDee Lamb","player_id":"00-0036358","team":"DAL","position":"WR","games":17,"plays":560,"target_share":30.0,"air_yards_share":38.0,"yards_per_route_run":2.45,"separation":3.3,"contested_catch_rate":52.0,"explosive_play_rate":16.5,"red_zone_target_rate":22.0,"route_win_rate":50.0,"average_depth_of_target":10.9,"targets":152,"receptions":101,"receiving_yards":1194,"receiving_tds":6,"fantasy_points_per_game":17.1,"snap_share":89.0},
        {"player":"Travis Kelce","player_id":"00-0030506","team":"KC","position":"TE","games":16,"plays":500,"target_share":22.0,"air_yards_share":24.0,"yards_per_route_run":1.75,"separation":3.0,"contested_catch_rate":56.0,"explosive_play_rate":13.0,"red_zone_target_rate":24.0,"route_win_rate":47.0,"average_depth_of_target":8.4,"targets":133,"receptions":97,"receiving_yards":823,"receiving_tds":3,"fantasy_points_per_game":12.2,"snap_share":82.0},
        {"player":"George Kittle","player_id":"00-0033881","team":"SF","position":"TE","games":15,"plays":480,"target_share":21.0,"air_yards_share":25.0,"yards_per_route_run":2.25,"separation":3.4,"contested_catch_rate":61.0,"explosive_play_rate":16.0,"red_zone_target_rate":26.0,"route_win_rate":50.0,"average_depth_of_target":9.1,"targets":94,"receptions":78,"receiving_yards":1106,"receiving_tds":8,"fantasy_points_per_game":15.2,"snap_share":86.0},
        {"player":"Puka Nacua","player_id":"00-0039075","team":"LA","position":"WR","games":17,"plays":560,"target_share":29.0,"air_yards_share":37.0,"yards_per_route_run":2.45,"separation":3.0,"contested_catch_rate":58.0,"explosive_play_rate":17.0,"red_zone_target_rate":19.0,"route_win_rate":50.0,"average_depth_of_target":10.8,"targets":160,"receptions":105,"receiving_yards":1486,"receiving_tds":7,"receiving_air_yards":1728,"carries":0,"rushing_yards":0,"rushing_tds":0,"snap_share":89.0},
        {"player":"Deebo Samuel","player_id":"00-0035719","team":"WAS","position":"WR","games":15,"plays":500,"target_share":21.0,"air_yards_share":21.0,"yards_per_route_run":1.78,"separation":3.4,"contested_catch_rate":50.0,"explosive_play_rate":18.0,"red_zone_target_rate":20.0,"route_win_rate":47.0,"average_depth_of_target":8.2,"targets":87,"receptions":51,"receiving_yards":670,"receiving_tds":3,"receiving_air_yards":713,"carries":42,"rushing_yards":136,"rushing_tds":1,"snap_share":78.0},
        {"player":"Tyreek Hill","player_id":"00-0033040","team":"MIA","position":"WR","games":17,"plays":545,"target_share":27.0,"air_yards_share":39.0,"yards_per_route_run":2.20,"separation":3.5,"contested_catch_rate":47.0,"explosive_play_rate":18.0,"red_zone_target_rate":21.0,"route_win_rate":52.0,"average_depth_of_target":11.9,"targets":150,"receptions":98,"receiving_yards":1365,"receiving_tds":8,"receiving_air_yards":1785,"carries":6,"rushing_yards":35,"rushing_tds":0,"snap_share":87.0},
        {"player":"Baltimore Ravens Defense","player_id":"DEF-BAL","team":"BAL","position":"DEF","games":17,"plays":1050,"pressure_rate":34.0,"blitz_rate":26.0,"sack_probability":8.2,"coverage_grade":76.0,"run_stop_win_rate":34.0,"explosive_plays_allowed":9.5,"epa_allowed":-.06,"pass_rush_win_rate":46.0,"snap_share":100.0},
        {"player":"Kansas City Chiefs Defense","player_id":"DEF-KC","team":"KC","position":"DEF","games":17,"plays":1035,"pressure_rate":35.0,"blitz_rate":31.0,"sack_probability":7.8,"coverage_grade":72.0,"run_stop_win_rate":32.0,"explosive_plays_allowed":10.2,"epa_allowed":-.04,"pass_rush_win_rate":48.0,"snap_share":100.0},
        {"player":"San Francisco 49ers Defense","player_id":"DEF-SF","team":"SF","position":"DEF","games":17,"plays":1040,"pressure_rate":32.0,"blitz_rate":22.0,"sack_probability":7.1,"coverage_grade":70.0,"run_stop_win_rate":33.0,"explosive_plays_allowed":11.0,"epa_allowed":.01,"pass_rush_win_rate":45.0,"snap_share":100.0},
    ]


def fallback_player_table(season: int | None = None) -> pd.DataFrame:
    table = pd.DataFrame(_fallback_rows())
    table["season"] = season or latest_completed_nfl_season()
    # The bundled snapshot is normalized to the same canonical nflverse field.
    table["player_position"] = table["position"]
    touchdown_splits = {
        "Saquon Barkley": (13, 2),
        "Christian McCaffrey": (14, 7),
        "Bijan Robinson": (14, 1),
    }
    for name, (rushing, receiving) in touchdown_splits.items():
        mask = table["player"] == name
        table.loc[mask, "rushing_tds"] = rushing
        table.loc[mask, "receiving_tds"] = receiving
    table = canonicalize_player_table(table)
    table.attrs["warnings"] = ["NFL provider unavailable; using the bundled last-known player snapshot."]
    table.attrs["source"] = "fallback"
    return table


def _pct(numerator: Any, denominator: Any, scale: float = 100.0) -> float:
    bottom = safe_number(denominator)
    return safe_number(numerator) / bottom * scale if bottom else 0.0


def _lookup_player_position(
    lookup: dict[str, str] | None,
    player_id: Any,
    player_name: Any,
) -> str | None:
    if not lookup:
        return None
    return normalize_player_position(
        lookup.get(str(player_id), lookup.get(normalize_text(player_name)))
    )


def _collapse_role_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Merge passer/rusher/receiver rows without adding the same stat twice."""
    if table.empty:
        return table
    data = table.copy()
    ids = data.get("player_id", pd.Series("", index=data.index)).astype(str)
    names = data.get("player", pd.Series("", index=data.index)).map(normalize_text)
    data["_identity"] = [pid if pid and pid.lower() not in {"nan", "none"} else name for pid, name in zip(ids, names)]
    rows: list[dict[str, Any]] = []
    identity_columns = {"player", "player_id", "team", "season", "player_position"}
    for _, group in data.groupby("_identity", sort=False):
        merged: dict[str, Any] = {}
        for column in data.columns:
            if column == "_identity":
                continue
            values = group[column]
            if column in identity_columns:
                present = values.dropna()
                merged[column] = present.iloc[0] if not present.empty else None
            else:
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().any():
                    # Role rows describe overlapping season totals, so max avoids
                    # double counting QB scrambles or gadget carries.
                    merged[column] = numeric.max()
                else:
                    present = values.dropna()
                    merged[column] = present.iloc[0] if not present.empty else None
        merged["player_position"] = detect_player_position(merged)
        rows.append(merged)
    return pd.DataFrame(rows)


def _aggregate_live(
    pbp: pd.DataFrame,
    season: int,
    position_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate player involvement from normalized nflverse play-by-play."""
    data = pbp.copy()
    if "season_type" in data:
        data = data[data["season_type"] == "REG"]
    rows: list[dict[str, Any]] = []
    team_plays = data.groupby("posteam").size().to_dict() if "posteam" in data else {}
    team_rushes = data[data.get("rusher_player_name").notna()].groupby("posteam").size().to_dict() if "rusher_player_name" in data else {}
    red_zone_team = data[pd.to_numeric(data.get("yardline_100"), errors="coerce") <= 20].groupby("posteam").size().to_dict() if "yardline_100" in data else {}

    def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
        return pd.to_numeric(frame[name], errors="coerce").fillna(0) if name in frame else pd.Series(0.0, index=frame.index)

    if "passer_player_name" in data:
        for name, group in data[data["passer_player_name"].notna()].groupby("passer_player_name"):
            attempts = max(safe_number(numeric(group, "pass_attempt").sum()), len(group))
            pressures = numeric(group, "was_pressure") if "was_pressure" in group else numeric(group, "qb_hit")
            sacks = numeric(group, "sack")
            deep = group[pd.to_numeric(group.get("air_yards"), errors="coerce") >= 20] if "air_yards" in group else group.iloc[0:0]
            rz = group[pd.to_numeric(group.get("yardline_100"), errors="coerce") <= 20] if "yardline_100" in group else group.iloc[0:0]
            rows.append({
                "player":name,"player_id":str(group.get("passer_player_id", pd.Series([""])).iloc[0]),"team":str(group.get("posteam", pd.Series([""])).mode().iloc[0]),"player_position":_lookup_player_position(position_lookup, group.get("passer_player_id", pd.Series([""])).iloc[0], name),"season":season,
                "games":group.get("game_id", pd.Series(index=group.index)).nunique(),"plays":len(group),"epa_per_play":safe_number(numeric(group,"epa").mean()),"cpoe":safe_number(numeric(group,"cpoe").mean()),
                "pressure_to_sack_rate":_pct(sacks.sum(), pressures.sum()),"deep_accuracy":_pct(numeric(deep,"complete_pass").sum(), len(deep)),"red_zone_efficiency":_pct(numeric(rz,"pass_touchdown").sum(), len(rz)),
                "scramble_rate":_pct(numeric(group,"qb_scramble").sum(), attempts),"time_to_throw":safe_number(numeric(group,"time_to_throw").replace(0,np.nan).mean(),2.8),"turnover_worthy_play_rate":_pct(numeric(group,"interception").sum()+numeric(group,"fumble_lost").sum(), attempts),
                "explosive_pass_rate":_pct((numeric(group,"yards_gained")>=20).sum(), attempts),"attempts":attempts,"completions":numeric(group,"complete_pass").sum(),"passing_yards":numeric(group,"yards_gained").sum(),"passing_tds":numeric(group,"pass_touchdown").sum(),"interceptions":numeric(group,"interception").sum(),
                "carries":numeric(group,"qb_scramble").sum(),"rushing_yards":numeric(group.loc[numeric(group,"qb_scramble")>0],"yards_gained").sum(),"rushing_tds":numeric(group.loc[numeric(group,"qb_scramble")>0],"rush_touchdown").sum(),"snap_share":95.0,
            })
    for role, name_col, id_col, position in [("rusher","rusher_player_name","rusher_player_id","RB"),("receiver","receiver_player_name","receiver_player_id","WR")]:
        if name_col not in data:
            continue
        for name, group in data[data[name_col].notna()].groupby(name_col):
            team = str(group["posteam"].mode().iloc[0]); team_total=max(safe_number(team_plays.get(team)),1)
            games = max(group["game_id"].nunique(),1) if "game_id" in group else 1
            if role == "rusher":
                carries=len(group); rz=group[pd.to_numeric(group.get("yardline_100"),errors="coerce")<=20] if "yardline_100" in group else group.iloc[0:0]
                rows.append({"player":name,"player_id":str(group.get(id_col,pd.Series([""])).iloc[0]),"team":team,"player_position":_lookup_player_position(position_lookup, group.get(id_col,pd.Series([""])).iloc[0], name),"season":season,"games":games,"plays":carries,
                    "carry_share":_pct(carries,team_rushes.get(team,1)),"yards_after_contact":safe_number(numeric(group,"yards_gained").mean())*.58,"missed_tackles_forced":round(carries*.14),"explosive_run_rate":_pct((numeric(group,"yards_gained")>=10).sum(),carries),"target_share":0.0,"red_zone_share":_pct(len(rz),red_zone_team.get(team,1)),"snap_share":min(95.0,_pct(carries,team_total)*2.4),"rush_success_rate":_pct(numeric(group,"success").sum(),carries),"yards_per_route_run":0.0,"pass_block_win_rate":80.0,
                    "carries":carries,"targets":0,"receptions":0,"rushing_yards":numeric(group,"yards_gained").sum(),"receiving_yards":0,"rushing_tds":numeric(group,"rush_touchdown").sum(),"total_tds":numeric(group,"rush_touchdown").sum()})
            else:
                targets=len(group); team_targets=max(data[(data.get("posteam")==team)&data.get("receiver_player_name").notna()].shape[0],1); air=numeric(group,"air_yards"); rz=group[pd.to_numeric(group.get("yardline_100"),errors="coerce")<=20] if "yardline_100" in group else group.iloc[0:0]
                pos="TE" if any(token in normalize_text(name) for token in ()) else "WR"
                rows.append({"player":name,"player_id":str(group.get(id_col,pd.Series([""])).iloc[0]),"team":team,"player_position":_lookup_player_position(position_lookup, group.get(id_col,pd.Series([""])).iloc[0], name),"season":season,"games":games,"plays":targets,
                    "target_share":_pct(targets,team_targets),"air_yards_share":_pct(air.sum(),pd.to_numeric(data.loc[data.get("posteam")==team,"air_yards"],errors="coerce").clip(lower=0).sum()),"yards_per_route_run":safe_number(numeric(group,"yards_gained").sum()/max(targets/.22,1)),"separation":3.0,"contested_catch_rate":50.0,"explosive_play_rate":_pct((numeric(group,"yards_gained")>=20).sum(),targets),"red_zone_target_rate":_pct(len(rz),targets),"route_win_rate":_pct(numeric(group,"complete_pass").sum(),targets),"average_depth_of_target":safe_number(air.mean()),
                    "targets":targets,"receptions":numeric(group,"complete_pass").sum(),"receiving_air_yards":air.clip(lower=0).sum(),"receiving_yards":numeric(group,"yards_gained").sum(),"receiving_tds":numeric(group,"pass_touchdown").sum(),"fantasy_points_per_game":0.0,"snap_share":min(95.0,_pct(targets,team_total)*5.0)})
    if "defteam" in data:
        for team, group in data[data["defteam"].notna()].groupby("defteam"):
            plays=max(len(group),1)
            pass_plays=group[group.get("play_type", pd.Series(index=group.index)).astype(str)=="pass"]
            pressure=numeric(group,"was_pressure") if "was_pressure" in group else numeric(group,"qb_hit")
            sacks=numeric(group,"sack")
            epa_allowed=safe_number(numeric(group,"epa").mean())
            rows.append({
                "player":f"{NFL_TEAMS.get(str(team), (team, team, []))[0]} Defense",
                "player_id":f"DEF-{team}","team":str(team),"player_position":"DEF","season":season,
                "games":group.get("game_id",pd.Series(index=group.index)).nunique(),"plays":plays,
                "pressure_rate":_pct(pressure.sum(),len(pass_plays)),"blitz_rate":safe_number(group.get("blitz",pd.Series(dtype=float)).mean())*100 if "blitz" in group else 25.0,
                "sack_probability":_pct(sacks.sum(),len(pass_plays)),"coverage_grade":max(0,min(100,65-epa_allowed*120)),
                "run_stop_win_rate":_pct((numeric(group.loc[group.get("play_type")=="run"],"success")==0).sum(),max((group.get("play_type")=="run").sum(),1)),
                "explosive_plays_allowed":_pct((numeric(group,"yards_gained")>=20).sum(),plays),"epa_allowed":epa_allowed,
                "pass_rush_win_rate":_pct(pressure.sum()+sacks.sum(),len(pass_plays)),"snap_share":100.0,
            })
    table=_collapse_role_rows(pd.DataFrame(rows))
    if table.empty:
        raise ValueError("NFL play-by-play contained no identifiable players.")
    for column in set().union(*POSITION_METRICS.values(), {"games","plays","snap_share","carries","targets","receptions","rushing_yards","receiving_air_yards","receiving_yards","passing_yards","passing_tds","rushing_tds","receiving_tds","total_tds","fantasy_points_per_game"}):
        if column not in table: table[column]=0.0
        table[column]=pd.to_numeric(table[column],errors="coerce")
    for column in table.select_dtypes(include="number").columns:
        table[column]=pd.to_numeric(table[column],errors="coerce").fillna(0.0)
    table=canonicalize_player_table(table)
    table.attrs["source"]="nflverse"; table.attrs["warnings"]=[]
    return table


def _load_position_lookup(provider: Any, season: int) -> dict[str, str]:
    """Map roster ids/names to canonical ``player_position`` values."""
    try:
        roster_raw = provider.load_rosters([season])
        roster = roster_raw.to_pandas() if hasattr(roster_raw, "to_pandas") else pd.DataFrame(roster_raw)
    except Exception:
        return {}
    lookup: dict[str, str] = {}
    for _, row in roster.iterrows():
        # nflverse roster releases may call the source column ``position``;
        # it is converted here, once, into our canonical player_position field.
        position = normalize_player_position(row.get("player_position", row.get("position")))
        if not position:
            continue
        for field in ("gsis_id", "player_id"):
            value = row.get(field)
            if value not in (None, "") and not pd.isna(value):
                lookup[str(value)] = position
        for field in ("full_name", "player_name", "football_name"):
            value = row.get(field)
            if value not in (None, "") and not pd.isna(value):
                lookup[normalize_text(value)] = position
    return lookup


@lru_cache(maxsize=3)
def load_player_stats(season: int | None = None) -> pd.DataFrame:
    """Load one canonical season table once per process."""
    season=season or latest_completed_nfl_season()
    try:
        import nflreadpy as nfl
        try: raw=nfl.load_pbp([season],columns=PBP_PLAYER_COLUMNS)
        except TypeError: raw=nfl.load_pbp([season])
        frame=raw.to_pandas() if hasattr(raw,"to_pandas") else pd.DataFrame(raw)
        positions=_load_position_lookup(nfl,season)
        return _aggregate_live(frame,season,positions)
    except Exception:
        return fallback_player_table(season)


def percentile(series: pd.Series, value: Any, higher_is_better: bool=True) -> float:
    values=pd.to_numeric(series,errors="coerce").dropna(); target=safe_number(value,float("nan"))
    if values.empty or pd.isna(target): return 50.0
    return round(float((values<=target).mean() if higher_is_better else (values>=target).mean())*100,1)


def blended_percentile(table: pd.DataFrame,row: pd.Series,metric: str) -> dict[str,float]:
    league=percentile(table[metric],row.get(metric),metric not in LOWER_IS_BETTER) if metric in table else 50.0
    peers=table[table.get("position",pd.Series(index=table.index,dtype=str)).astype(str)==str(row.get("position"))]
    position=percentile(peers[metric],row.get(metric),metric not in LOWER_IS_BETTER) if metric in peers and not peers.empty else league
    return {"league_percentile":league,"position_percentile":position,"blended_percentile":round(.6*league+.4*position,1)}


def resolve_player(name_or_id: str|int,season: int|None=None,table: pd.DataFrame|None=None) -> tuple[pd.Series,pd.DataFrame,list[str]]:
    data=canonicalize_player_table(table.copy() if table is not None else load_player_stats(season).copy())
    query=str(name_or_id).strip(); match=None
    if query and "player_id" in data:
        exact=data[data["player_id"].astype(str)==query]
        if not exact.empty: match=exact.iloc[0]
    if match is None and query:
        candidate=fuzzy_name_match(query,data.to_dict("records"),key=lambda item:str(item.get("player","")),cutoff=.62)
        if candidate: match=pd.Series(candidate)
    if match is None: raise ValueError(f"Could not identify NFL player {name_or_id!r}.")
    warnings=list(data.attrs.get("warnings",[])); return match,data,warnings


def normalized_profile(name_or_id: str|int,season: int|None=None,comparison_mode: str="League",display_mode: str="Adjusted") -> dict[str,Any]:
    row,table,warnings=resolve_player(name_or_id,season); position=detect_player_position(row); metrics=POSITION_METRICS.get(position,POSITION_METRICS["WR"])
    sample_field={"QB":"pass_attempts","RB":"rush_attempts","WR":"targets","TE":"targets","DEF":"plays"}.get(position,"plays")
    default_min={"QB":150.0,"RB":100.0,"WR":50.0,"TE":40.0,"DEF":300.0}.get(position,50.0)
    attempts=safe_number(row.get(sample_field,row.get("plays"))); total=max(safe_number(row.get("plays")),attempts)
    dynamic_min=min(default_min,max(total*.35,1.0)); sample_confidence=min(1.0,attempts/dynamic_min) if dynamic_min else 1.0
    attributes=[]
    for metric in metrics:
        score=blended_percentile(table,row,metric); selected=score["league_percentile"] if comparison_mode=="League" else score["position_percentile"]
        adjusted=round(score["blended_percentile"]*sample_confidence,1); visible=score["blended_percentile"] if display_mode=="Raw" else adjusted
        attributes.append({"metric":metric,"value":round(safe_number(row.get(metric)),2),**score,"selected_percentile":selected,"attempts":round(attempts,1),"dynamic_minimum":round(dynamic_min,1),"sample_confidence":round(sample_confidence,3),"raw_value":score["blended_percentile"],"adjusted_value":adjusted,"badge_value":visible,"display_value":visible})
    if sample_confidence<1: warnings.append("Low sample size: NFL ratings are sample-confidence weighted.")
    return {"player":str(row.get("player")),"player_id":str(row.get("player_id","")),"team":str(row.get("team","")),"position":position,"season":int(row.get("season",season or latest_completed_nfl_season())),"source":table.attrs.get("source","unknown"),"display_mode":display_mode,"comparison_mode":comparison_mode,"attributes":attributes,"raw":row.to_dict(),"warnings":list(dict.fromkeys(warnings))}


def get_player_stats(player: Any, season: int | None = None) -> dict[str, Any]:
    """Unified, non-null stats accessor used by ``nfl_projections``.

    ``player`` may be a player name/id, a mapping with a ``player``/``player_id``
    key, or an object exposing a ``player_id``/``player_name`` attribute. Every
    lookup resolves through :func:`resolve_player` so the returned fields are
    always the canonical, position-correct, non-null projection inputs.
    """
    if isinstance(player, (str, int)):
        name_or_id = player
    elif isinstance(player, dict):
        name_or_id = player.get("player_id") or player.get("player") or player.get("player_name")
    else:
        name_or_id = getattr(player, "player_id", None) or getattr(player, "player_name", None) or getattr(player, "player", None)
    row, _table, _warnings = resolve_player(name_or_id, season)
    return canonical_player_stats(row)