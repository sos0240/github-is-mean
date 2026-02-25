"""Tests for operator1.clients.macro_ons wrapper (UK via FRED)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroONS:
    """Tests for operator1.clients.macro_ons."""

    def test_import(self):
        from operator1.clients.macro_ons import fetch_macro_ons
        assert callable(fetch_macro_ons)

    def test_series_ids_defined(self):
        from operator1.clients.macro_ons import _FRED_GB_SERIES
        assert "gdp_growth" in _FRED_GB_SERIES
        assert "inflation_rate_yoy" in _FRED_GB_SERIES
        assert "unemployment_rate" in _FRED_GB_SERIES
        assert "interest_rate" in _FRED_GB_SERIES
        assert "exchange_rate" in _FRED_GB_SERIES

    def test_returns_empty_without_fred_key(self):
        from operator1.clients.macro_ons import fetch_macro_ons

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_ons()
        assert result == {}

    def test_fetch_with_mock_fred(self):
        from operator1.clients.macro_ons import fetch_macro_ons

        mock_series = pd.Series([1.5, 1.6, 1.7], index=pd.date_range("2023-01-01", periods=3))

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_series
        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
                result = fetch_macro_ons()

        assert len(result) > 0
