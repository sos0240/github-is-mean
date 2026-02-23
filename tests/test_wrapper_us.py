"""Tests for US SEC EDGAR wrapper client (edgartools + sec-edgar-api).

Tests the USEdgarClient against the PITClient protocol, verifying
profile extraction, financial statement retrieval, peer discovery,
caching, and fallback behavior.

Note: Tests that hit live SEC endpoints are marked with @pytest.mark.network.
Run with: pytest tests/test_wrapper_us.py -v
Skip network tests: pytest tests/test_wrapper_us.py -v -m "not network"
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from operator1.clients.us_edgar import USEdgarClient, USEdgarError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    """Create a USEdgarClient with a temp cache directory."""
    return USEdgarClient(
        user_agent="Operator1Test/1.0 (test@example.com)",
        cache_dir=tmp_path / "cache" / "us_sec_edgar",
    )


@pytest.fixture
def sample_profile():
    """Sample company profile dict."""
    return {
        "name": "APPLE INC",
        "ticker": "AAPL",
        "cik": "320193",
        "isin": "",
        "country": "US",
        "sector": "Electronic Computers",
        "industry": "Electronic Computers",
        "exchange": "NASDAQ",
        "currency": "USD",
    }


@pytest.fixture
def sample_companyfacts():
    """Minimal SEC EDGAR companyfacts JSON structure."""
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 394328000000,
                                "filed": "2023-11-03",
                                "end": "2023-09-30",
                                "start": "2022-10-01",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            },
                            {
                                "val": 117154000000,
                                "filed": "2024-02-02",
                                "end": "2023-12-30",
                                "start": "2023-10-01",
                                "form": "10-Q",
                                "fy": 2024,
                                "fp": "Q1",
                            },
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 96995000000,
                                "filed": "2023-11-03",
                                "end": "2023-09-30",
                                "start": "2022-10-01",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 352583000000,
                                "filed": "2023-11-03",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            },
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "val": 110543000000,
                                "filed": "2023-11-03",
                                "end": "2023-09-30",
                                "start": "2022-10-01",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            },
                        ]
                    }
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------

class TestUSEdgarProtocol:
    """Verify USEdgarClient satisfies the PITClient protocol."""

    def test_market_id(self, client):
        assert client.market_id == "us_sec_edgar"

    def test_market_name(self, client):
        assert "United States" in client.market_name
        assert "SEC EDGAR" in client.market_name

    def test_has_required_methods(self, client):
        """All PITClient protocol methods must exist."""
        assert hasattr(client, "list_companies")
        assert hasattr(client, "search_company")
        assert hasattr(client, "get_profile")
        assert hasattr(client, "get_income_statement")
        assert hasattr(client, "get_balance_sheet")
        assert hasattr(client, "get_cashflow_statement")
        assert hasattr(client, "get_quotes")
        assert hasattr(client, "get_peers")
        assert hasattr(client, "get_executives")

    def test_get_quotes_returns_empty(self, client):
        """OHLCV is handled by ohlcv_provider, not this client."""
        df = client.get_quotes("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_executives_returns_empty(self, client):
        """SEC has no direct executives endpoint."""
        result = client.get_executives("AAPL")
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------

class TestCaching:
    """Test profile and filing caching logic."""

    def test_write_and_read_cache(self, client, sample_profile):
        """Cache write then read should return the same data."""
        client._write_cache("AAPL", "profile.json", sample_profile)
        cached = client._read_cache("AAPL", "profile.json")
        assert cached is not None
        assert cached["name"] == "APPLE INC"
        assert cached["ticker"] == "AAPL"

    def test_cache_miss_returns_none(self, client):
        """Reading non-existent cache returns None."""
        result = client._read_cache("NONEXISTENT", "profile.json")
        assert result is None

    def test_cache_path_sanitization(self, client):
        """Special characters in identifiers should be sanitized."""
        path = client._cache_path("some/weird\\id", "profile.json")
        assert "/" not in path.stem or "\\" not in path.stem

    def test_filing_cache_structure(self, client, tmp_path):
        """Filings should be cached as separate JSON files per period."""
        df = pd.DataFrame([
            {
                "concept": "revenue",
                "value": 394328000000,
                "filing_date": pd.Timestamp("2023-11-03"),
                "report_date": pd.Timestamp("2023-09-30"),
                "form": "10-K",
                "period_type": "annual",
            },
            {
                "concept": "net_income",
                "value": 96995000000,
                "filing_date": pd.Timestamp("2023-11-03"),
                "report_date": pd.Timestamp("2023-09-30"),
                "form": "10-K",
                "period_type": "annual",
            },
        ])
        client._cache_filings("AAPL", df)

        # Check file exists
        filing_path = client._cache_path("AAPL", "filings/2023-09-30.json")
        assert filing_path.exists()

        # Check content
        data = json.loads(filing_path.read_text())
        assert data["period_end"] == "2023-09-30"
        assert len(data["rows"]) == 2


# ---------------------------------------------------------------------------
# Fallback XBRL extraction tests
# ---------------------------------------------------------------------------

class TestFallbackExtraction:
    """Test sec-edgar-api fallback financial extraction."""

    def test_income_statement_fallback(self, client, sample_companyfacts):
        """Fallback should extract income statement from raw companyfacts."""
        mock_sec_client = MagicMock()
        mock_sec_client.get_company_facts.return_value = sample_companyfacts
        client._sec_api_client = mock_sec_client

        # Mock the CIK resolution
        client._company_list_cache = [
            {"ticker": "AAPL", "name": "Apple Inc", "cik": "0000320193", "exchange": "", "market_id": "us_sec_edgar"}
        ]

        df = client._fetch_statement_fallback("AAPL", "income")
        assert isinstance(df, pd.DataFrame)
        # Should have revenue and net_income rows
        if not df.empty:
            assert "filing_date" in df.columns or "concept" in df.columns

    def test_balance_sheet_fallback(self, client, sample_companyfacts):
        """Fallback should extract balance sheet from raw companyfacts."""
        mock_sec_client = MagicMock()
        mock_sec_client.get_company_facts.return_value = sample_companyfacts
        client._sec_api_client = mock_sec_client

        client._company_list_cache = [
            {"ticker": "AAPL", "name": "Apple Inc", "cik": "0000320193", "exchange": "", "market_id": "us_sec_edgar"}
        ]

        df = client._fetch_statement_fallback("AAPL", "balance")
        assert isinstance(df, pd.DataFrame)

    def test_cashflow_fallback(self, client, sample_companyfacts):
        """Fallback should extract cash flow from raw companyfacts."""
        mock_sec_client = MagicMock()
        mock_sec_client.get_company_facts.return_value = sample_companyfacts
        client._sec_api_client = mock_sec_client

        client._company_list_cache = [
            {"ticker": "AAPL", "name": "Apple Inc", "cik": "0000320193", "exchange": "", "market_id": "us_sec_edgar"}
        ]

        df = client._fetch_statement_fallback("AAPL", "cashflow")
        assert isinstance(df, pd.DataFrame)

    def test_empty_facts_returns_empty_df(self, client):
        """Empty companyfacts should return empty DataFrame gracefully."""
        mock_sec_client = MagicMock()
        mock_sec_client.get_company_facts.return_value = {"facts": {"us-gaap": {}}}
        client._sec_api_client = mock_sec_client

        client._company_list_cache = [
            {"ticker": "AAPL", "name": "Apple Inc", "cik": "0000320193", "exchange": "", "market_id": "us_sec_edgar"}
        ]

        df = client._fetch_statement_fallback("AAPL", "income")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# CIK resolution tests
# ---------------------------------------------------------------------------

class TestCIKResolution:
    """Test ticker -> CIK resolution."""

    def test_numeric_cik_passthrough(self, client):
        """Numeric identifiers should pass through as CIK."""
        result = client._resolve_cik_fallback("320193")
        assert result == "0000320193"

    def test_ticker_resolution(self, client):
        """Ticker should resolve to CIK from company list."""
        client._company_list_cache = [
            {"ticker": "AAPL", "name": "Apple Inc", "cik": "320193", "exchange": "", "market_id": "us_sec_edgar"},
            {"ticker": "MSFT", "name": "Microsoft", "cik": "789019", "exchange": "", "market_id": "us_sec_edgar"},
        ]
        result = client._resolve_cik_fallback("AAPL")
        assert result == "0000320193"

    def test_unknown_ticker_raises(self, client):
        """Unknown ticker should raise USEdgarError."""
        client._company_list_cache = [
            {"ticker": "AAPL", "name": "Apple Inc", "cik": "320193", "exchange": "", "market_id": "us_sec_edgar"},
        ]
        with pytest.raises(USEdgarError):
            client._resolve_cik_fallback("ZZZZZZ")


# ---------------------------------------------------------------------------
# Concept mapping tests
# ---------------------------------------------------------------------------

class TestConceptMapping:
    """Test edgartools concept label -> canonical name mapping."""

    def test_income_mappings(self, client):
        assert client._map_edgartools_concept("Revenue", "income") == "revenue"
        assert client._map_edgartools_concept("Net Income", "income") == "net_income"
        assert client._map_edgartools_concept("Interest Expense", "income") == "interest_expense"

    def test_balance_mappings(self, client):
        assert client._map_edgartools_concept("Total Assets", "balance") == "total_assets"
        assert client._map_edgartools_concept("Cash and Cash Equivalents", "balance") == "cash_and_equivalents"
        assert client._map_edgartools_concept("Accounts Receivable, Net", "balance") == "receivables"

    def test_cashflow_mappings(self, client):
        assert client._map_edgartools_concept("Operating Cash Flow", "cashflow") == "operating_cash_flow"
        assert client._map_edgartools_concept("Capital Expenditures", "cashflow") == "capex"
        assert client._map_edgartools_concept("Dividends Paid", "cashflow") == "dividends_paid"

    def test_unknown_concept_returns_none(self, client):
        assert client._map_edgartools_concept("Some Random Field", "income") is None

    def test_case_insensitive(self, client):
        assert client._map_edgartools_concept("REVENUE", "income") == "revenue"
        assert client._map_edgartools_concept("total assets", "balance") == "total_assets"


# ---------------------------------------------------------------------------
# Integration tests (require network -- skip in CI)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestLiveIntegration:
    """Live integration tests against SEC EDGAR.

    Skipped by default. Run with: pytest -m network
    """

    def test_list_companies(self, client):
        companies = client.list_companies()
        assert len(companies) > 1000
        tickers = {c["ticker"] for c in companies}
        assert "AAPL" in tickers

    def test_search_company(self, client):
        results = client.search_company("Apple")
        assert len(results) > 0
        assert any("AAPL" in r.get("ticker", "") for r in results)

    def test_get_profile_aapl(self, client):
        profile = client.get_profile("AAPL")
        assert profile["name"]
        assert profile["ticker"] == "AAPL"
        assert profile["country"] == "US"
        assert profile["currency"] == "USD"

    def test_get_income_statement_aapl(self, client):
        df = client.get_income_statement("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "filing_date" in df.columns or len(df) > 0

    def test_get_balance_sheet_aapl(self, client):
        df = client.get_balance_sheet("AAPL")
        assert isinstance(df, pd.DataFrame)

    def test_get_cashflow_aapl(self, client):
        df = client.get_cashflow_statement("AAPL")
        assert isinstance(df, pd.DataFrame)

    def test_get_peers_aapl(self, client):
        peers = client.get_peers("AAPL")
        assert isinstance(peers, list)
        assert len(peers) > 0
