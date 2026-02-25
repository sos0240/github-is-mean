"""Tests for operator1.clients.ohlcv_jugaad wrapper (India NSE)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestOHLCVJugaad:
    """Tests for operator1.clients.ohlcv_jugaad."""

    def test_import(self):
        from operator1.clients.ohlcv_jugaad import fetch_ohlcv_jugaad
        assert callable(fetch_ohlcv_jugaad)

    def test_fetch_with_mock(self):
        from operator1.clients.ohlcv_jugaad import fetch_ohlcv_jugaad

        mock_df = pd.DataFrame({
            "DATE": pd.date_range("2023-01-01", periods=3),
            "CH_OPENING_PRICE": [100.0, 101, 102],
            "CH_TRADE_HIGH_PRICE": [105.0, 106, 107],
            "CH_TRADE_LOW_PRICE": [95.0, 96, 97],
            "CH_CLOSING_PRICE": [102.0, 103, 104],
            "CH_TOT_TRADED_QTY": [10000, 20000, 30000],
        })

        mock_stock_df = MagicMock(return_value=mock_df)
        mock_module = MagicMock()
        mock_module.stock_df = mock_stock_df

        with patch.dict("sys.modules", {"jugaad_data": MagicMock(), "jugaad_data.nse": mock_module}):
            result = fetch_ohlcv_jugaad("TCS", years=1)

        assert not result.empty
        assert "close" in result.columns

    def test_fetch_empty_on_exception(self):
        from operator1.clients.ohlcv_jugaad import fetch_ohlcv_jugaad

        mock_module = MagicMock()
        mock_module.stock_df.side_effect = Exception("NSE error")

        with patch.dict("sys.modules", {"jugaad_data": MagicMock(), "jugaad_data.nse": mock_module}):
            result = fetch_ohlcv_jugaad("TCS")
        assert result.empty
