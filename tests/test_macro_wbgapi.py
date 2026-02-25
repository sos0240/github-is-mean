"""Tests for operator1.clients.macro_wbgapi wrapper (World Bank)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroWbgapi:
    """Tests for operator1.clients.macro_wbgapi."""

    def test_import(self):
        from operator1.clients.macro_wbgapi import fetch_macro_wbgapi
        assert callable(fetch_macro_wbgapi)

    def test_country_mapping(self):
        from operator1.clients.macro_wbgapi import _COUNTRY_TO_WB
        assert _COUNTRY_TO_WB["US"] == "USA"
        assert _COUNTRY_TO_WB["BR"] == "BRA"
        assert _COUNTRY_TO_WB["JP"] == "JPN"
        assert _COUNTRY_TO_WB["GB"] == "GBR"

    def test_indicator_mapping(self):
        from operator1.clients.macro_wbgapi import _WB_INDICATORS
        assert "gdp_growth" in _WB_INDICATORS
        assert "inflation_rate_yoy" in _WB_INDICATORS
        assert "unemployment_rate" in _WB_INDICATORS

    def test_fetch_with_mock(self):
        from operator1.clients.macro_wbgapi import fetch_macro_wbgapi

        mock_row = pd.Series(
            [2.5, 2.8, 3.0, 2.7, 2.9],
            index=[2019, 2020, 2021, 2022, 2023],
            name="USA",
        )
        mock_df = pd.DataFrame([mock_row])
        mock_df.index = ["USA"]

        mock_wb_data = MagicMock()
        mock_wb_data.DataFrame.return_value = mock_df

        mock_wb = MagicMock()
        mock_wb.data = mock_wb_data

        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetch_macro_wbgapi("US", years=5)

        assert len(result) > 0
        for name, series in result.items():
            assert isinstance(series, pd.Series)
