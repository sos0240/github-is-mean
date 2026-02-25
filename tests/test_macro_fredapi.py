"""Tests for operator1.clients.macro_fredapi wrapper (US FRED)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroFredapi:
    """Tests for operator1.clients.macro_fredapi."""

    def test_import(self):
        from operator1.clients.macro_fredapi import fetch_macro_fred
        assert callable(fetch_macro_fred)

    def test_returns_empty_without_api_key(self):
        from operator1.clients.macro_fredapi import fetch_macro_fred

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_fred(api_key="")
        assert result == {}

    def test_fetch_with_mock_fredapi(self):
        from operator1.clients.macro_fredapi import fetch_macro_fred

        mock_series = pd.Series([2.5, 2.7, 2.9], index=pd.date_range("2023-01-01", periods=3))

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_series

        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
            result = fetch_macro_fred(api_key="test_key_123")

        assert len(result) > 0
        for name, series in result.items():
            assert isinstance(series, pd.Series)
            assert len(series) == 3

    def test_fetch_handles_individual_series_failure(self):
        from operator1.clients.macro_fredapi import fetch_macro_fred

        call_count = 0

        def mock_get_series(series_id, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("series not found")
            return pd.Series([1.0, 2.0], index=pd.date_range("2023-01-01", periods=2))

        mock_fred = MagicMock()
        mock_fred.get_series = mock_get_series

        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
            result = fetch_macro_fred(api_key="test_key")

        assert len(result) >= 1
