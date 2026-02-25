"""Tests for operator1.clients.macro_estat wrapper (Japan via FRED)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroEstat:
    """Tests for operator1.clients.macro_estat."""

    def test_import(self):
        from operator1.clients.macro_estat import fetch_macro_estat
        assert callable(fetch_macro_estat)

    def test_fred_jp_series_defined(self):
        from operator1.clients.macro_estat import _FRED_JP_SERIES
        assert "gdp_growth" in _FRED_JP_SERIES
        assert "inflation_rate_yoy" in _FRED_JP_SERIES
        assert "unemployment_rate" in _FRED_JP_SERIES

    def test_returns_empty_without_keys(self):
        from operator1.clients.macro_estat import fetch_macro_estat

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_estat()
        assert result == {}

    def test_fetch_via_fred_with_mock(self):
        from operator1.clients.macro_estat import fetch_macro_estat

        mock_series = pd.Series([1.0, 1.1], index=pd.date_range("2023-01-01", periods=2))

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_series
        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
                result = fetch_macro_estat()

        assert len(result) > 0
