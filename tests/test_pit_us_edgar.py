"""Tests for operator1.clients.us_edgar.USEdgarClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestUSEdgarClient:
    """Tests for operator1.clients.us_edgar.USEdgarClient."""

    def test_import(self):
        from operator1.clients.us_edgar import USEdgarClient, USEdgarError
        assert USEdgarClient is not None
        assert USEdgarError is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(
            user_agent="Test/1.0 (test@example.com)",
            cache_dir=tmp_path / "cache",
        )
        assert client.market_id == "us_sec_edgar"
        assert "US" in client.market_name or "SEC" in client.market_name

    def test_market_id(self, tmp_path):
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(cache_dir=tmp_path / "cache")
        assert client.market_id == "us_sec_edgar"

    def test_get_profile_returns_dict(self, tmp_path):
        """Mock the internal method to avoid network calls."""
        from operator1.clients.us_edgar import USEdgarClient

        client = USEdgarClient(cache_dir=tmp_path / "cache")
        # Mock the internal _get_edgar_company to avoid edgartools dependency
        mock_company = MagicMock()
        mock_company.name = "APPLE INC"
        mock_company.tickers = ["AAPL"]
        mock_company.cik = 320193
        mock_company.sic = MagicMock()
        mock_company.sic.description = "Electronic Computers"
        mock_company.sic.office = "Technology"

        with patch.object(client, "_get_edgar_company", return_value=mock_company):
            profile = client.get_profile("AAPL")

        assert isinstance(profile, dict)

    def test_get_income_statement_returns_dataframe(self, tmp_path):
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(cache_dir=tmp_path / "cache")
        with patch.object(client, "_fetch_statement_edgartools", return_value=pd.DataFrame()):
            result = client.get_income_statement("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_get_balance_sheet_returns_dataframe(self, tmp_path):
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(cache_dir=tmp_path / "cache")
        with patch.object(client, "_fetch_statement_edgartools", return_value=pd.DataFrame()):
            result = client.get_balance_sheet("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_get_cashflow_statement_returns_dataframe(self, tmp_path):
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(cache_dir=tmp_path / "cache")
        with patch.object(client, "_fetch_statement_edgartools", return_value=pd.DataFrame()):
            result = client.get_cashflow_statement("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_get_quotes_returns_empty_dataframe(self, tmp_path):
        """SEC EDGAR doesn't provide OHLCV data."""
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(cache_dir=tmp_path / "cache")
        result = client.get_quotes("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_get_executives_returns_list(self, tmp_path):
        from operator1.clients.us_edgar import USEdgarClient
        client = USEdgarClient(cache_dir=tmp_path / "cache")
        result = client.get_executives("AAPL")
        assert isinstance(result, list)
