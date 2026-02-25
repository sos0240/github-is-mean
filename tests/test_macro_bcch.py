"""Tests for operator1.clients.macro_bcch wrapper (Chile via FRED)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroBCCh:
    """Tests for operator1.clients.macro_bcch."""

    def test_import(self):
        from operator1.clients.macro_bcch import fetch_macro_bcch
        assert callable(fetch_macro_bcch)

    def test_fred_cl_series_defined(self):
        from operator1.clients.macro_bcch import _FRED_CL_SERIES
        assert "inflation_rate_yoy" in _FRED_CL_SERIES
        assert "unemployment_rate" in _FRED_CL_SERIES

    def test_returns_empty_without_fred_key(self):
        from operator1.clients.macro_bcch import fetch_macro_bcch

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_bcch()
        assert result == {}

    def test_fetch_with_mock_fred(self):
        from operator1.clients.macro_bcch import fetch_macro_bcch

        mock_series = pd.Series([4.0, 3.8], index=pd.date_range("2023-01-01", periods=2))

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_series
        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
                result = fetch_macro_bcch()

        assert len(result) > 0
