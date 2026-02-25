"""Tests for operator1.clients.ohlcv_akshare wrapper (China A-shares)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestOHLCVAkshare:
    """Tests for operator1.clients.ohlcv_akshare."""

    def test_import(self):
        from operator1.clients.ohlcv_akshare import fetch_ohlcv_akshare
        assert callable(fetch_ohlcv_akshare)

    def test_fetch_with_mock(self):
        from operator1.clients.ohlcv_akshare import fetch_ohlcv_akshare

        mock_df = pd.DataFrame({
            "\u65e5\u671f": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "\u5f00\u76d8": [100.0, 101.0, 102.0],
            "\u6700\u9ad8": [105.0, 106.0, 107.0],
            "\u6700\u4f4e": [95.0, 96.0, 97.0],
            "\u6536\u76d8": [102.0, 103.0, 104.0],
            "\u6210\u4ea4\u91cf": [10000, 20000, 30000],
        })

        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.return_value = mock_df

        with patch.dict("sys.modules", {"akshare": mock_ak}):
            result = fetch_ohlcv_akshare("600519", years=1)

        assert not result.empty
        assert "close" in result.columns
        assert len(result) == 3

    def test_fetch_empty_on_no_data(self):
        from operator1.clients.ohlcv_akshare import fetch_ohlcv_akshare

        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"akshare": mock_ak}):
            result = fetch_ohlcv_akshare("000000")
        assert result.empty

    def test_fetch_empty_on_exception(self):
        from operator1.clients.ohlcv_akshare import fetch_ohlcv_akshare

        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"akshare": mock_ak}):
            result = fetch_ohlcv_akshare("600519")
        assert result.empty
