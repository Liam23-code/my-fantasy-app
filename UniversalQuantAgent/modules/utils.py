"""Small, reusable helpers shared by every domain module.

Keeping formatting and cleaning here prevents finance, sports, and future
modules from becoming cluttered with presentation details.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import safe_number


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate rows removed and infinite values replaced.

    Missing values are intentionally not filled here: each analysis must decide
    whether forward-filling, dropping, or using zero makes sense for its data.
    """
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates()
    return cleaned.replace([float("inf"), float("-inf")], pd.NA)


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Limit a number to a range. Opportunity scores use the 0-100 default."""
    return max(minimum, min(maximum, value))


def format_number(value: Any, decimals: int = 2) -> str:
    """Format numbers consistently for terminal output."""
    return f"{safe_number(value):,.{decimals}f}"


def print_section(title: str) -> None:
    """Print a simple terminal section heading."""
    print(f"\n{title}\n{'-' * len(title)}")


def print_key_values(values: dict[str, Any]) -> None:
    """Print a dictionary in a readable key/value layout."""
    for key, value in values.items():
        friendly_key = key.replace("_", " ").title()
        print(f"  {friendly_key}: {value}")

