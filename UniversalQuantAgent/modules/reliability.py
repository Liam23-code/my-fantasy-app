"""Quant Reliability Score for NBA player projections."""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import safe_dict, safe_get, safe_list, safe_number
from modules.fusion_model import fuse_projection, normalize_model_output
from modules.model_performance import DATA_FILE, get_model_performance_summary


def _historical_score(fusion: dict[str, Any]) -> tuple[float, str]:
    """Prefer MAE from similar contexts when optional context columns exist."""
    fusion = safe_dict(fusion)
    if DATA_FILE.exists():
        try:
            data = pd.read_csv(DATA_FILE)
            required = {"projection", "actual"}
            if required.issubset(data.columns):
                comparable = data.copy()
                context = safe_dict(safe_get(fusion, "context"))
                payload = safe_dict(safe_get(context, "components"))
                targets = {
                    "matchup_difficulty": safe_get(
                        normalize_model_output(safe_get(payload, "matchup")),
                        "value",
                    ),
                    "pace": safe_get(
                        normalize_model_output(safe_get(payload, "pace")),
                        "value",
                    ),
                    "minutes": safe_get(
                        normalize_model_output(safe_get(payload, "minutes")),
                        "value",
                    ),
                }
                tolerances = {
                    "matchup_difficulty": 10.0,
                    "pace": 5.0,
                    "minutes": 5.0,
                }
                used = []
                for column, target in targets.items():
                    if column in comparable and target is not None:
                        numeric = pd.to_numeric(
                            comparable[column], errors="coerce"
                        )
                        comparable = comparable[
                            (numeric - safe_number(target)).abs()
                            <= tolerances[column]
                        ]
                        used.append(column)
                if len(comparable) >= 5:
                    errors = (
                        pd.to_numeric(
                            comparable["projection"], errors="coerce"
                        )
                        - pd.to_numeric(comparable["actual"], errors="coerce")
                    ).abs().dropna()
                    if not errors.empty:
                        mae = safe_number(errors.mean())
                        return (
                            max(0.0, min(100.0, 100 - mae * 8)),
                            f"Historical score uses {len(errors)} similar "
                            f"matchups ({', '.join(used)}).",
                        )
        except Exception:
            pass

    performance = safe_dict(get_model_performance_summary())
    categories = safe_list(safe_get(performance, "categories"))
    valid_rows = [
        safe_dict(row) for row in categories if isinstance(row, dict)
    ]
    if not valid_rows:
        return 55.0, (
            "No settled-history file; neutral historical reliability used."
        )
    weighted = sum(
        max(0.0, min(100.0, 100 - safe_number(row.get("mae")) * 8))
        * max(1, int(safe_number(row.get("sample_size"), 1)))
        for row in valid_rows
    )
    samples = sum(
        max(1, int(safe_number(row.get("sample_size"), 1)))
        for row in valid_rows
    )
    return (
        weighted / samples,
        f"Historical score uses {performance.get('sample_size', 0)} "
        "settled predictions.",
    )


def _similarity_confidence(
    player_name: str, season: str | None
) -> tuple[float, str]:
    """Measure how well the similarity model can support this projection."""
    try:
        from modules.similarity_engine import compute_player_similarity

        result = safe_dict(compute_player_similarity(player_name, season))
        matches = [
            safe_dict(row)
            for row in safe_list(result.get("similar_players"))
            if isinstance(row, dict)
        ]
        if not matches:
            return 25.0, "No usable similar-player profiles were available."
        coverage = min(len(matches) / 10.0, 1.0)
        top_scores = [
            safe_number(row.get("similarity_score")) for row in matches[:3]
        ]
        average_similarity = sum(top_scores) / len(top_scores)
        score = coverage * 70.0 + average_similarity * 0.30
        return max(0.0, min(100.0, score)), (
            f"Similarity confidence uses {len(matches)} comparable players."
        )
    except Exception as error:
        return 25.0, f"Similarity context unavailable: {error}"


def get_reliability_score(
    player_name: str,
    opponent_team: str,
    season: str | None = None,
    fusion_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a 0-100 score spanning data, fallbacks, and model context."""
    fusion = safe_dict(
        fusion_result or fuse_projection(player_name, opponent_team, season)
    )
    base = safe_dict(safe_get(fusion, "base_projection"))
    context = safe_dict(safe_get(fusion, "context"))
    component_payload = safe_dict(safe_get(context, "components"))
    minute_component = normalize_model_output(
        safe_get(component_payload, "minutes")
    )
    pace_component = normalize_model_output(
        safe_get(component_payload, "pace")
    )
    matchup_component = normalize_model_output(
        safe_get(component_payload, "matchup")
    )
    availability = safe_dict(safe_get(component_payload, "availability"))
    input_breakdown = safe_dict(safe_get(fusion, "input_breakdown"))

    completeness = safe_number(
        safe_get(
            safe_dict(safe_get(base, "data_quality")),
            "completeness_score",
        ),
        min(100, len(safe_list(safe_get(base, "features_used"))) / 10 * 100),
    )
    minutes_inputs = safe_dict(safe_get(input_breakdown, "minutes"))
    usage_inputs = safe_dict(safe_get(input_breakdown, "usage"))
    fallback_minutes_quality = safe_number(
        safe_get(
            minutes_inputs,
            "quality",
            safe_get(minute_component, "fallback_minutes_quality"),
        ),
        50.0,
    )
    fallback_usage_quality = safe_number(
        safe_get(usage_inputs, "quality"), 50.0
    )
    opponent_confidence = safe_number(
        safe_get(matchup_component, "confidence"), 50.0
    )
    pace_confidence = safe_number(
        safe_get(pace_component, "confidence"), 50.0
    )
    similarity_confidence, similarity_note = _similarity_confidence(
        str(safe_get(fusion, "player", player_name)), season
    )

    injury_context = safe_dict(safe_get(context, "injury_impact"))
    injury_status = str(
        safe_get(
            availability,
            "availability",
            safe_get(injury_context, "status", "UNKNOWN"),
        )
        or "UNKNOWN"
    ).upper()
    injury_certainty = {
        "ACTIVE": 92.0,
        "PROBABLE": 75.0,
        "QUESTIONABLE": 35.0,
        "OUT": 25.0,
        "UNKNOWN": 45.0,
    }.get(injury_status, 45.0)

    widths = []
    final_projection = safe_dict(safe_get(fusion, "final_projection"))
    confidence_ranges = safe_dict(safe_get(fusion, "confidence_range"))
    for stat, projected in final_projection.items():
        interval = safe_dict(safe_get(confidence_ranges, stat))
        projected_number = safe_number(projected)
        low = safe_number(safe_get(interval, "low"), projected_number)
        high = safe_number(safe_get(interval, "high"), projected_number)
        span = abs(high - low)
        denominator = abs(projected_number)
        widths.append(span / denominator if denominator else span)
    average_width = safe_number(
        sum(widths) / len(widths) if widths else 1.0
    )
    model_confidence = max(0.0, min(100.0, 100 - average_width * 65))

    historical, historical_note = _historical_score(fusion)
    components = {
        "data_completeness": completeness,
        "fallback_usage_quality": fallback_usage_quality,
        "fallback_minutes_quality": fallback_minutes_quality,
        "minutes_stability": safe_number(
            safe_get(minute_component, "coach_tendency_score"),
            safe_number(safe_get(context, "role_stability"), 50),
        ),
        "role_stability": safe_number(
            safe_get(context, "role_stability"), 50
        ),
        "opponent_difficulty_confidence": opponent_confidence,
        "pace_confidence": pace_confidence,
        "similarity_model_confidence": similarity_confidence,
        "injury_certainty": injury_certainty,
        "model_confidence": model_confidence,
        "historical_mae_score": historical,
    }
    weights = {
        "data_completeness": .14,
        "fallback_usage_quality": .10,
        "fallback_minutes_quality": .10,
        "minutes_stability": .08,
        "role_stability": .08,
        "opponent_difficulty_confidence": .10,
        "pace_confidence": .10,
        "similarity_model_confidence": .08,
        "injury_certainty": .08,
        "model_confidence": .10,
        "historical_mae_score": .04,
    }
    score = sum(
        safe_number(components[key]) * weight
        for key, weight in weights.items()
    )
    score = max(0.0, min(100.0, score))
    rating = "high" if score >= 75 else "medium" if score >= 55 else "low"
    warnings = list(dict.fromkeys(
        [str(item) for item in safe_list(safe_get(base, "warnings"))]
        + [str(item) for item in safe_list(safe_get(context, "warnings"))]
    ))
    return {
        "player": safe_get(fusion, "player", player_name),
        "opponent": opponent_team,
        "score": round(score, 1),
        "rating": rating,
        "components": {
            key: round(value, 1) for key, value in components.items()
        },
        "weights": weights,
        "explanation": (
            f"{rating.title()} reliability. {historical_note} "
            f"{similarity_note}"
        ),
        "warnings": warnings,
        "fusion": fusion,
    }