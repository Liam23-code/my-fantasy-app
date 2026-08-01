"""Unit tests for fantasy.utils."""

from __future__ import annotations

import math

import pytest

from fantasy.utils import clamp, safe_float, safe_int


def test_safe_float_none_returns_default():
    assert safe_float(None) == 0.0
    assert safe_float(None, default=5.0) == 5.0


def test_safe_float_bool_coerces_to_zero_or_one():
    assert safe_float(True) == 1.0
    assert safe_float(False) == 0.0


def test_safe_float_passes_through_finite_numbers():
    assert safe_float(42) == 42.0
    assert safe_float(3.14) == pytest.approx(3.14)
    assert safe_float(-7) == -7.0


def test_safe_float_rejects_nan_and_infinity():
    assert safe_float(float("nan")) == 0.0
    assert safe_float(float("inf")) == 0.0
    assert safe_float(float("-inf"), default=-1.0) == -1.0


def test_safe_float_coerces_numeric_strings():
    assert safe_float("245.0") == 245.0
    assert safe_float("  10  ") == 10.0
    assert safe_float("1,234.5") == 1234.5


def test_safe_float_blank_string_returns_default():
    assert safe_float("") == 0.0
    assert safe_float("   ") == 0.0


def test_safe_float_non_numeric_string_returns_default():
    assert safe_float("not-a-number") == 0.0


def test_safe_float_numeric_string_that_is_nan_or_infinite_returns_default():
    assert safe_float("nan") == 0.0
    assert safe_float("inf") == 0.0


def test_safe_float_unsupported_type_returns_default():
    assert safe_float(["not", "a", "number"]) == 0.0
    assert safe_float({"also": "not a number"}) == 0.0


def test_safe_int_rounds_to_nearest():
    assert safe_int(4.6) == 5
    assert safe_int(4.4) == 4
    assert safe_int("7") == 7


def test_safe_int_defaults_when_unparseable():
    assert safe_int(None) == 0
    assert safe_int("bogus", default=3) == 3


def test_clamp_within_bounds_returns_value():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_low_returns_low():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above_high_returns_high():
    assert clamp(50, 0, 10) == 10


def test_clamp_at_exact_bounds():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10


def test_safe_float_is_never_nan_or_inf_in_output():
    for value in [None, "abc", float("nan"), float("inf"), [], {}, "1e400"]:
        result = safe_float(value)
        assert math.isfinite(result)
