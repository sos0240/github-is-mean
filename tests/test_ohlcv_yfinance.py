"""Tests for operator1.clients.ohlcv_yfinance wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestOHLCVYfinance:
    """Tests for operator1.clients.ohlcv_yfinance."""

    def test_import(self):
        from operator1.clients.ohlcv_yfinance import (
            fetch_ohlcv_yfinance,
            get_yfinance_suffix,
        )
        assert callable(fetch_ohlcv_yfinance)
        assert callable(get_yfinance_suffix)

    def test_suffix_mapping(self):
        from operator1.clients.ohlcv_yfinance import get_yfinance_suffix
        assert get_yfinance_suffix("us_sec_edgar") == ""
        assert get_yfinance_suffix("uk_companies_house") == ".L"
        assert get_yfinance_suffix("jp_jquants") == ".T"
        assert get_yfinance_suffix("kr_dart") == ".KS"
        assert get_yfinance_suffix("tw_mops") == ".TW"
        assert get_yfinance_suffix("br_cvm") == ".SA"
        assert get_yfinance_suffix("ca_sedar") == ".TO"
        assert get_yfinance_suffix("au_asx") == ".AX"
        assert get_yfinance_suffix("in_bse") == ".NS"
        assert get_yfinance_suffix("cn_sse") == ".SS"
        assert get_yfinance_suffix("hk_hkex") == ".HK"
        assert get_yfinance_suffix("sg_sgx") == ".SI"
        assert get_yfinance_suffix("mx_bmv") == ".MX"
        assert get_yfinance_suffix("za_jse") == ".JO"
        assert get_yfinance_suffix("ch_six") == ".SW"
        assert get_yfinance_suffix("sa_tadawul") == ".SR"
        assert get_yfinance_suffix("ae_dfm") == ".AE"
        assert get_yfinance_suffix("nl_esef") == ".AS"
        assert get_yfinance_suffix("es_esef") == ".MC"
        assert get_yfinance_suffix("it_esef") == ".MI"
        assert get_yfinance_suffix("se_esef") == ".ST"

    def test_unknown_market_returns_empty_suffix(self):
        from operator1.clients.ohlcv_yfinance import get_yfinance_suffix
        assert get_yfinance_suffix("unknown_market") == ""

    def test_fetch_with_mock_yfinance(self):
        """Mock yfinance.download to return a proper DataFrame."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance

        mock_df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Open": [100.0, 101, 102, 103, 104],
            "High": [105.0, 106, 107, 108, 109],
            "Low": [95.0, 96, 97, 98, 99],
            "Close": [102.0, 103, 104, 105, 106],
            "Volume": [1000, 2000, 3000, 4000, 5000],
            "Adj Close": [102.0, 103, 104, 105, 106],
        })
        mock_df = mock_df.set_index("Date")

        with patch("yfinance.download", return_value=mock_df):
            result = fetch_ohlcv_yfinance("AAPL", market_id="us_sec_edgar")

        assert not result.empty
        assert "close" in result.columns
        assert "volume" in result.columns
        assert len(result) == 5

    def test_fetch_empty_on_failure(self):
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance

        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = fetch_ohlcv_yfinance("INVALID", market_id="us_sec_edgar")
        assert result.empty

    def test_fetch_handles_multiindex_columns(self):
        """yfinance >= 1.2.0 returns MultiIndex columns for single ticker."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance

        dates = pd.date_range("2023-01-01", periods=3)
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        idx = pd.MultiIndex.from_tuples(tuples)
        data = [
            [100, 105, 95, 102, 1000],
            [101, 106, 96, 103, 2000],
            [102, 107, 97, 104, 3000],
        ]
        mock_df = pd.DataFrame(data, index=dates, columns=idx)

        with patch("yfinance.download", return_value=mock_df):
            result = fetch_ohlcv_yfinance("AAPL", market_id="us_sec_edgar")

        assert not result.empty
        assert "close" in result.columns
        assert len(result) == 3

    def test_fetch_exception_returns_empty(self):
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance

        with patch("yfinance.download", side_effect=Exception("network error")):
            result = fetch_ohlcv_yfinance("AAPL")
        assert result.empty

    def test_suffix_not_doubled(self):
        """If ticker already has suffix, don't add it again."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance

        with patch("yfinance.download", return_value=pd.DataFrame()) as mock_dl:
            fetch_ohlcv_yfinance("7203.T", market_id="jp_jquants")
            call_args = mock_dl.call_args
            assert call_args[0][0] == "7203.T"
