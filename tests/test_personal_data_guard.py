"""Tests for the personal data guard module.

Verifies PII detection in user input and wrapper registration
requirement lookups without requiring LLM API keys.
"""

from __future__ import annotations

import unittest


class TestPersonalDataGuardRegex(unittest.TestCase):
    """Test regex-based PII detection (no LLM needed)."""

    def test_clean_company_input(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("Apple Inc", "US")
        self.assertFalse(result.has_personal_data)
        self.assertEqual(len(result.warnings), 0)

    def test_clean_ticker_input(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("AAPL", "US")
        self.assertFalse(result.has_personal_data)

    def test_clean_japanese_input(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("7203", "Japan")
        self.assertFalse(result.has_personal_data)

    def test_email_detected(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("john@example.com", "US")
        self.assertTrue(result.has_personal_data)
        self.assertTrue(any("email" in w.lower() for w in result.warnings))

    def test_ssn_detected(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("123-45-6789", "US")
        self.assertTrue(result.has_personal_data)

    def test_credit_card_detected(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("4111-1111-1111-1111", "US")
        self.assertTrue(result.has_personal_data)

    def test_phone_in_country(self):
        from operator1.clients.personal_data_guard import check_user_input_for_pii

        result = check_user_input_for_pii("Samsung", "+82-10-1234-5678")
        self.assertTrue(result.has_personal_data)


class TestWrapperPersonalData(unittest.TestCase):
    """Test wrapper registration requirement lookups."""

    def test_sec_edgar_low(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("us_sec_edgar")
        self.assertFalse(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "low")

    def test_edinet_medium(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("jp_edinet")
        self.assertTrue(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "medium")
        self.assertTrue(any("phone" in w.lower() for w in result.warnings))

    def test_dart_medium(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("kr_dart")
        self.assertTrue(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "medium")

    def test_esef_none(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("eu_esef")
        self.assertFalse(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "none")

    def test_mops_none(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("tw_mops")
        self.assertFalse(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "none")

    def test_cvm_none(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("br_cvm")
        self.assertFalse(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "none")

    def test_unknown_market_defaults_to_none(self):
        from operator1.clients.personal_data_guard import check_wrapper_personal_data

        result = check_wrapper_personal_data("xx_nonexistent")
        self.assertFalse(result.has_personal_data)
        self.assertEqual(result.market_personal_data_level, "none")


class TestFormatWarning(unittest.TestCase):
    """Test warning message formatting."""

    def test_empty_result_no_warning(self):
        from operator1.clients.personal_data_guard import (
            PersonalDataCheckResult,
            format_pii_warning,
        )

        result = PersonalDataCheckResult()
        self.assertEqual(format_pii_warning(result), "")

    def test_warning_with_personal_data(self):
        from operator1.clients.personal_data_guard import (
            PersonalDataCheckResult,
            format_pii_warning,
        )

        result = PersonalDataCheckResult(
            has_personal_data=True,
            warnings=["Input contains an email address"],
            details="Regex-based detection.",
        )
        text = format_pii_warning(result)
        self.assertIn("WARNING", text)
        self.assertIn("email", text)


class TestMarketInfoPersonalDataFields(unittest.TestCase):
    """Test that pit_registry.MarketInfo has personal data fields."""

    def test_edinet_has_personal_data_fields(self):
        from operator1.clients.pit_registry import get_market

        market = get_market("jp_edinet")
        self.assertIsNotNone(market)
        self.assertEqual(market.personal_data_level, "medium")
        self.assertIn("phone", market.input_requirements.lower())

    def test_dart_has_personal_data_fields(self):
        from operator1.clients.pit_registry import get_market

        market = get_market("kr_dart")
        self.assertIsNotNone(market)
        self.assertEqual(market.personal_data_level, "medium")

    def test_sec_edgar_has_low_level(self):
        from operator1.clients.pit_registry import get_market

        market = get_market("us_sec_edgar")
        self.assertIsNotNone(market)
        self.assertEqual(market.personal_data_level, "low")

    def test_esef_has_none_level(self):
        from operator1.clients.pit_registry import get_market

        market = get_market("eu_esef")
        self.assertIsNotNone(market)
        self.assertEqual(market.personal_data_level, "none")


if __name__ == "__main__":
    unittest.main()
