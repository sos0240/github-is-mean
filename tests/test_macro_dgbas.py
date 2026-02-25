"""Tests for operator1.clients.macro_dgbas wrapper (Taiwan via FRED)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroDGBAS:
    """Tests for operator1.clients.macro_dgbas."""

    def test_import(self):
        from operator1.clients.macro_dgbas import fetch_macro_dgbas
        assert callable(fetch_macro_dgbas)

    def test_fred_tw_series_defined(self):
        from operator1.clients.macro_dgbas import _FRED_TW_SERIES
        assert "inflation_rate_yoy" in _FRED_TW_SERIES
        assert "interest_rate" in _FRED_TW_SERIES

    def test_returns_empty_without_fred_key(self):
        from operator1.clients.macro_dgbas import fetch_macro_dgbas

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_dgbas()
        assert result == {}

    def test_fetch_with_mock_fred(self):
        from operator1.clients.macro_dgbas import fetch_macro_dgbas

        mock_series = pd.Series([2.0, 2.1], index=pd.date_range("2023-01-01", periods=2))

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_series
        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
                result = fetch_macro_dgbas()

        assert len(result) > 0
