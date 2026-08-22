"""Advanced projection models used by the fantasy Quant Engine."""

from .projection_engine import (
    ProjectionEngine,
    breakout_bust_probability_model,
    compute_final_projection,
    configure_projection_data,
    injury_adjusted_projection_model,
    matchup_based_projection_model,
    regression_to_mean_model,
    usage_based_projection_model,
    volatility_adjusted_projection_model,
    weighted_historical_model,
)

__all__ = [
    "ProjectionEngine",
    "breakout_bust_probability_model",
    "compute_final_projection",
    "configure_projection_data",
    "injury_adjusted_projection_model",
    "matchup_based_projection_model",
    "regression_to_mean_model",
    "usage_based_projection_model",
    "volatility_adjusted_projection_model",
    "weighted_historical_model",
]
