"""Contract for modules.odds_input -- the shared paste-a-price helper behind the
MoneyLines, Player Props, and Parlays pages."""

import math
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from betting.odds_math import expected_value, implied_probability

from modules.odds_input import (
    evaluate_price,
    format_american,
    format_money,
    format_pct,
    format_signed_pct,
    no_vig_probabilities,
    parse_number,
    parse_odds,
    value_verdict,
)


class ParseOddsTests(unittest.TestCase):
    def test_american_prices_round_trip(self):
        for text, expected in (("-150", -150.0), ("+120", 120.0), ("120", 120.0), ("-110", -110.0), ("+100", 100.0)):
            with self.subTest(text=text):
                parsed = parse_odds(text)
                self.assertEqual((parsed.price, parsed.error, parsed.kind), (expected, None, "american"))

    def test_typographic_minus_whitespace_and_symbols_are_cleaned(self):
        self.assertEqual(parse_odds("−150").price, -150.0)
        self.assertEqual(parse_odds("  - 150 ").price, -150.0)
        self.assertEqual(parse_odds("+1,200").price, 1200.0)

    def test_even_money_words(self):
        for text in ("even", "EVEN", "ev", "Evens"):
            with self.subTest(text=text):
                self.assertEqual(parse_odds(text).price, 100.0)

    def test_decimal_odds_convert_to_american(self):
        self.assertEqual(parse_odds("2.50"), (150.0, None, "decimal"))
        self.assertEqual(parse_odds("1.91").price, -110.0)

    def test_blank_is_empty_not_an_error(self):
        for text in ("", "   ", None):
            with self.subTest(text=text):
                self.assertEqual(parse_odds(text), (None, None, "empty"))

    def test_invalid_input_is_reported_not_guessed(self):
        for text in ("abc", "-50", "+99", "0", "1.0", "-1.5", "150.5", "12x", "--110"):
            with self.subTest(text=text):
                parsed = parse_odds(text)
                self.assertIsNone(parsed.price)
                self.assertEqual(parsed.kind, "invalid")
                self.assertTrue(parsed.error)


class ParseNumberTests(unittest.TestCase):
    def test_prop_lines_and_spreads(self):
        self.assertEqual(parse_number("249.5").value, 249.5)
        self.assertEqual(parse_number("-3.5").value, -3.5)
        self.assertEqual(parse_number("−3.5").value, -3.5)
        self.assertEqual(parse_number("0.5").value, 0.5)

    def test_leading_and_trailing_decimal_point(self):
        self.assertEqual(parse_number(".5").value, 0.5)
        self.assertEqual(parse_number("5.").value, 5.0)
        self.assertEqual(parse_number("-.5").value, -0.5)
        self.assertIsNone(parse_number(".").value)
        self.assertIsNone(parse_number("1.2.3").value)

    def test_blank_and_bad_text(self):
        self.assertEqual(parse_number(""), (None, None))
        value, error = parse_number("two hundred")
        self.assertIsNone(value)
        self.assertIn("number", error)

    def test_bounds_and_sign(self):
        self.assertIsNotNone(parse_number("-1", allow_negative=False).error)
        self.assertIsNotNone(parse_number("5", minimum=10).error)
        self.assertIsNotNone(parse_number("500", maximum=100).error)
        self.assertEqual(parse_number("50", minimum=10, maximum=100).value, 50.0)


class FormattingTests(unittest.TestCase):
    def test_formatters_never_emit_nan(self):
        for formatter in (format_american, format_pct, format_signed_pct, format_money):
            with self.subTest(formatter=formatter.__name__):
                self.assertEqual(formatter(None), "—")
                self.assertEqual(formatter(float("nan")), "—")
        self.assertEqual(format_american(float("inf")), "—")

    def test_values_that_round_to_zero_carry_no_sign(self):
        self.assertEqual(format_signed_pct(-0.00004), "0.0 pts")
        self.assertEqual(format_signed_pct(0.0004), "0.0 pts")
        self.assertEqual(format_signed_pct(-0.0006), "-0.1 pts")
        self.assertEqual(format_money(-0.004), "$0.00")
        self.assertEqual(format_money(0.004), "$0.00")
        self.assertEqual(format_money(-0.006), "-$0.01")

    def test_values(self):
        self.assertEqual(format_american(120), "+120")
        self.assertEqual(format_american(-150.0), "-150")
        self.assertEqual(format_pct(0.5434), "54.3%")
        self.assertEqual(format_signed_pct(0.032), "+3.2 pts")
        self.assertEqual(format_signed_pct(-0.018), "-1.8 pts")
        self.assertEqual(format_money(4.2), "+$4.20")
        self.assertEqual(format_money(-11), "-$11.00")
        self.assertEqual(format_money(0), "$0.00")


class EvaluatePriceTests(unittest.TestCase):
    def test_matches_engine_odds_math(self):
        result = evaluate_price(0.62, -150)
        self.assertAlmostEqual(result["implied_probability"], implied_probability(-150), places=4)
        self.assertAlmostEqual(result["expected_value"], expected_value(0.62, -150), places=2)
        self.assertAlmostEqual(result["value_gap"], 0.62 - implied_probability(-150), places=4)
        self.assertAlmostEqual(result["ev_pct"], result["expected_value"] / 100.0, places=3)

    def test_positive_value_when_model_beats_price(self):
        result = evaluate_price(0.50, 120)  # break-even is 45.45%
        self.assertGreater(result["value_gap"], 0.03)
        self.assertGreater(result["expected_value"], 0)
        self.assertEqual(result["verdict"], "Strong value")
        self.assertTrue(result["is_value"])

    def test_negative_value_when_price_is_too_short(self):
        result = evaluate_price(0.50, -150)  # break-even is 60%
        self.assertLess(result["value_gap"], 0)
        self.assertLess(result["expected_value"], 0)
        self.assertEqual(result["verdict"], "Overpriced")
        self.assertFalse(result["is_value"])

    def test_break_even_probability_has_zero_ev(self):
        result = evaluate_price(implied_probability(-110), -110)
        self.assertAlmostEqual(result["expected_value"], 0.0, places=1)
        self.assertEqual(result["verdict"], "Fair price")

    def test_stake_scales_ev_but_not_percentages(self):
        small, large = evaluate_price(0.6, -110, stake=100), evaluate_price(0.6, -110, stake=500)
        self.assertAlmostEqual(large["expected_value"], small["expected_value"] * 5, places=1)
        self.assertAlmostEqual(large["ev_pct"], small["ev_pct"], places=3)

    def test_rejects_out_of_range_probability(self):
        for bad in (-0.1, 1.5, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    evaluate_price(bad, -110)

    def test_rejects_prices_that_are_not_real_american_odds(self):
        for bad in (0, 50, -50, 99, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    evaluate_price(0.5, bad)

    def test_fair_price_is_consistent_with_probability(self):
        fair = evaluate_price(0.60, -110)["fair_price"]
        self.assertAlmostEqual(implied_probability(fair), 0.60, delta=0.01)


class VerdictAndVigTests(unittest.TestCase):
    def test_verdict_thresholds(self):
        self.assertEqual(value_verdict(0.05), ("Strong value", "success"))
        self.assertEqual(value_verdict(0.015), ("Slight value", "info"))
        self.assertEqual(value_verdict(0.0), ("Fair price", "neutral"))
        self.assertEqual(value_verdict(-0.02), ("Overpriced", "danger"))

    def test_nan_gap_is_unavailable_not_overpriced(self):
        for gap in (float("nan"), None, float("inf")):
            with self.subTest(gap=gap):
                self.assertEqual(value_verdict(gap), ("Unavailable", "neutral"))

    def test_no_vig_probabilities_sum_to_one(self):
        fair_a, fair_b = no_vig_probabilities(-150, 130)
        self.assertTrue(math.isclose(fair_a + fair_b, 1.0, abs_tol=1e-9))
        self.assertGreater(fair_a, fair_b)


if __name__ == "__main__":
    unittest.main()
