"""Tests for operator1.clients.ohlcv_pykrx wrapper (Korea KRX)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestOHLCVPykrx:
    """Tests for operator1.clients.ohlcv_pykrx."""

    def test_import(self):
        from operator1.clients.ohlcv_pykrx import fetch_ohlcv_pykrx
        assert callable(fetch_ohlcv_pykrx)

    def test_fetch_with_mock(self):
        """Mock pykrx.stock to return Korean OHLCV data."""
        # pykrx returns DataFrame with date as index (named 날짜)
        idx = pd.Index(["2023-01-02", "2023-01-03"], name="\ub0a0\uc9dc")
        mock_df = pd.DataFrame({
            "\uc2dc\uac00": [60000, 61000],
            "\uace0\uac00": [62000, 63000],
            "\uc800\uac00": [59000, 60000],
            "\uc885\uac00": [61000, 62000],
            "\uac70\ub798\ub7c9": [100000, 200000],
        }, index=idx)

        mock_stock = MagicMock()
        mock_stock.get_market_ohlcv.return_value = mock_df

        mock_pykrx = MagicMock()
        mock_pykrx.stock = mock_stock

        # Need to reload the module so the `from pykrx import stock` picks up our mock
        import operator1.clients.ohlcv_pykrx as mod
        with patch.dict("sys.modules", {"pykrx": mock_pykrx, "pykrx.stock": mock_stock}):
            importlib.reload(mod)
            result = mod.fetch_ohlcv_pykrx("005930", years=1)

        assert not result.empty
        assert "close" in result.columns

    def test_fetch_empty_on_exception(self):
        from operator1.clients.ohlcv_pykrx import fetch_ohlcv_pykrx

        mock_stock = MagicMock()
        mock_stock.get_market_ohlcv.side_effect = Exception("KRX error")

        mock_pykrx = MagicMock()
        mock_pykrx.stock = mock_stock

        import operator1.clients.ohlcv_pykrx as mod
        with patch.dict("sys.modules", {"pykrx": mock_pykrx, "pykrx.stock": mock_stock}):
            importlib.reload(mod)
            result = mod.fetch_ohlcv_pykrx("005930")
        assert result.empty
