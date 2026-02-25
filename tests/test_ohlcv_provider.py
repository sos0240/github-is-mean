"""Tests for operator1.clients.ohlcv_provider dispatcher."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


class TestOHLCVProvider:
    """Tests for operator1.clients.ohlcv_provider dispatcher."""

    def test_import(self):
        from operator1.clients.ohlcv_provider import fetch_ohlcv
        assert callable(fetch_ohlcv)

    def test_us_market_goes_to_yfinance_directly(self):
        from operator1.clients.ohlcv_provider import fetch_ohlcv

        mock_df = pd.DataFrame({"date": ["2023-01-01"], "close": [150.0]})

        with patch("operator1.clients.ohlcv_yfinance.fetch_ohlcv_yfinance", return_value=mock_df):
            result = fetch_ohlcv("AAPL", "us_sec_edgar")
        assert not result.empty

    def test_kr_market_tries_pykrx_first(self):
        from operator1.clients.ohlcv_provider import fetch_ohlcv

        mock_df = pd.DataFrame({"date": ["2023-01-01"], "close": [60000]})

        with patch("operator1.clients.ohlcv_pykrx.fetch_ohlcv_pykrx", return_value=mock_df):
            result = fetch_ohlcv("005930", "kr_dart")
        assert not result.empty

    def test_fallback_to_yfinance_when_primary_empty(self):
        from operator1.clients.ohlcv_provider import fetch_ohlcv

        yf_df = pd.DataFrame({"date": ["2023-01-01"], "close": [60000]})

        with patch("operator1.clients.ohlcv_pykrx.fetch_ohlcv_pykrx", return_value=pd.DataFrame()):
            with patch("operator1.clients.ohlcv_yfinance.fetch_ohlcv_yfinance", return_value=yf_df):
                result = fetch_ohlcv("005930", "kr_dart")
        assert not result.empty

    def test_returns_empty_when_both_fail(self):
        from operator1.clients.ohlcv_provider import fetch_ohlcv

        with patch("operator1.clients.ohlcv_pykrx.fetch_ohlcv_pykrx", return_value=pd.DataFrame()):
            with patch("operator1.clients.ohlcv_yfinance.fetch_ohlcv_yfinance", return_value=pd.DataFrame()):
                result = fetch_ohlcv("005930", "kr_dart")
        assert result.empty
