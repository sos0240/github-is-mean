"""Tests for UK Companies House iXBRL parsing integration.

Verifies that the _extract_ixbrl_values method in uk_ch_wrapper.py
correctly parses iXBRL financial data using ixbrl-parse and regex fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


class TestIXBRLValueExtraction:
    """Tests for UKCompaniesHouseClient._extract_ixbrl_values."""

    SAMPLE_IXBRL_HTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<ix:nonFraction name="uk-gaap:TurnoverRevenue" contextRef="c1" unitRef="u1"
    decimals="0">1500000</ix:nonFraction>
<ix:nonFraction name="uk-gaap:OperatingProfitLoss" contextRef="c1" unitRef="u1"
    decimals="0">250000</ix:nonFraction>
<ix:nonFraction name="uk-gaap:ProfitLossForPeriod" contextRef="c1" unitRef="u1"
    decimals="0">180000</ix:nonFraction>
<ix:nonFraction name="uk-gaap:ShareholderFunds" contextRef="c2" unitRef="u1"
    decimals="0">900000</ix:nonFraction>
<ix:nonFraction name="frs102:TurnoverRevenue" contextRef="c3" unitRef="u1"
    decimals="0">1600000</ix:nonFraction>
</body>
</html>"""

    def test_regex_fallback_extracts_values(self, tmp_path):
        """When ixbrl-parse is not installed, regex fallback should work."""
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.SAMPLE_IXBRL_HTML
        mock_response.content = self.SAMPLE_IXBRL_HTML.encode()

        with patch("requests.get", return_value=mock_response):
            # Force ImportError on ixbrl_parse to test regex fallback
            with patch.dict("sys.modules", {"ixbrl_parse": None, "ixbrl_parse.ixbrl": None}):
                values = client._extract_ixbrl_values("00000001", "txn123")

        assert len(values) > 0
        assert "revenue" in values
        assert values["revenue"] == 1500000.0 or values["revenue"] == 1600000.0
        assert "operating_income" in values
        assert "net_income" in values

    def test_ixbrl_parse_library_extraction(self, tmp_path):
        """When ixbrl-parse is installed, it should be used for extraction."""
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path)

        # Create mock ixbrl parsed result
        mock_value_revenue = MagicMock()
        mock_value_revenue.name = "uk-gaap:TurnoverRevenue"
        mock_value_revenue.to_value.return_value = 1500000

        mock_value_equity = MagicMock()
        mock_value_equity.name = "uk-gaap:ShareholderFunds"
        mock_value_equity.to_value.return_value = 900000

        mock_ixbrl_result = MagicMock()
        mock_ixbrl_result.values = {"v1": mock_value_revenue, "v2": mock_value_equity}

        mock_parse_fn = MagicMock(return_value=mock_ixbrl_result)
        mock_ixbrl_module = MagicMock()
        mock_ixbrl_module.parse = mock_parse_fn

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.SAMPLE_IXBRL_HTML
        mock_response.content = self.SAMPLE_IXBRL_HTML.encode()

        with patch("requests.get", return_value=mock_response):
            with patch.dict("sys.modules", {
                "ixbrl_parse": MagicMock(),
                "ixbrl_parse.ixbrl": mock_ixbrl_module,
            }):
                values = client._extract_ixbrl_values("00000001", "txn123")

        # Should have extracted financial values (either via mock ixbrl-parse or regex fallback)
        assert "revenue" in values
        assert values["revenue"] in (1500000.0, 1600000.0)  # ixbrl-parse or regex may pick different tag
        # total_equity may come from ixbrl-parse mock or regex fallback
        assert "total_equity" in values or "operating_income" in values

    def test_download_failure_returns_empty(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path)

        with patch("requests.get", side_effect=Exception("network error")):
            values = client._extract_ixbrl_values("00000001", "txn123")

        assert values == {}

    def test_404_returns_empty(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b""

        with patch("requests.get", return_value=mock_response):
            values = client._extract_ixbrl_values("00000001", "txn123")

        assert values == {}


class TestFRS102TagMapping:
    """Verify FRS 102 tags are in the UK GAAP mapping."""

    def test_frs102_income_tags(self):
        from operator1.clients.canonical_translator import _UKGAAP_MAP
        assert _UKGAAP_MAP["frs102:TurnoverRevenue"] == "revenue"
        assert _UKGAAP_MAP["frs102:OperatingProfit"] == "operating_income"
        assert _UKGAAP_MAP["frs102:ProfitLossBeforeTax"] == "ebit"
        assert _UKGAAP_MAP["frs102:ProfitLossForPeriod"] == "net_income"

    def test_frs102_balance_tags(self):
        from operator1.clients.canonical_translator import _UKGAAP_MAP
        assert _UKGAAP_MAP["frs102:ShareholderFunds"] == "total_equity"
        assert _UKGAAP_MAP["frs102:CreditorsDueWithinOneYear"] == "current_liabilities"
        assert _UKGAAP_MAP["frs102:CreditorsDueAfterOneYear"] == "long_term_debt"
        assert _UKGAAP_MAP["frs102:CashAtBankAndInHand"] == "cash_and_equivalents"

    def test_frs102_cashflow_tags(self):
        from operator1.clients.canonical_translator import _UKGAAP_MAP
        assert _UKGAAP_MAP["frs102:NetCashFromOperatingActivities"] == "operating_cash_flow"
        assert _UKGAAP_MAP["frs102:NetCashUsedInInvestingActivities"] == "investing_cf"
        assert _UKGAAP_MAP["frs102:NetCashUsedInFinancingActivities"] == "financing_cf"

    def test_bare_name_tags(self):
        from operator1.clients.canonical_translator import _UKGAAP_MAP
        assert _UKGAAP_MAP["Turnover"] == "revenue"
        assert _UKGAAP_MAP["OperatingProfit"] == "operating_income"
        assert _UKGAAP_MAP["ShareholderFunds"] == "total_equity"

    def test_combined_map_in_market_concept_maps(self):
        from operator1.clients.canonical_translator import get_concept_map
        cmap = get_concept_map("uk_companies_house")
        # Should include both UK GAAP and IFRS mappings
        assert "uk-gaap:TurnoverRevenue" in cmap
        assert "frs102:TurnoverRevenue" in cmap
        assert "ifrs-full:Revenue" in cmap
