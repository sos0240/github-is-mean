"""Tests for operator1.clients.macro_provider dispatcher."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


class TestMacroProvider:
    """Tests for operator1.clients.macro_provider dispatcher."""

    def test_import(self):
        from operator1.clients.macro_provider import fetch_macro
        assert callable(fetch_macro)

    def test_primary_fetcher_mapping(self):
        from operator1.clients.macro_provider import _PRIMARY_FETCHERS
        assert _PRIMARY_FETCHERS["US"] == "fred"
        assert _PRIMARY_FETCHERS["BR"] == "bcb"
        assert _PRIMARY_FETCHERS["MX"] == "banxico"
        assert _PRIMARY_FETCHERS["GB"] == "ons"
        assert _PRIMARY_FETCHERS["JP"] == "estat"
        assert _PRIMARY_FETCHERS["KR"] == "kosis"
        assert _PRIMARY_FETCHERS["TW"] == "dgbas"
        assert _PRIMARY_FETCHERS["CL"] == "bcch"
        assert _PRIMARY_FETCHERS["EU"] == "ecb"
        assert _PRIMARY_FETCHERS["DE"] == "ecb"
        assert _PRIMARY_FETCHERS["FR"] == "ecb"
        assert _PRIMARY_FETCHERS["NL"] == "ecb"
        assert _PRIMARY_FETCHERS["ES"] == "ecb"
        assert _PRIMARY_FETCHERS["IT"] == "ecb"

    def test_fetch_falls_back_to_wbgapi(self):
        from operator1.clients.macro_provider import fetch_macro

        wb_data = {
            "gdp_growth": pd.Series([2.5], index=pd.to_datetime(["2023"])),
            "inflation_rate_yoy": pd.Series([3.0], index=pd.to_datetime(["2023"])),
            "unemployment_rate": pd.Series([4.5], index=pd.to_datetime(["2023"])),
        }

        with patch("operator1.clients.macro_fredapi.fetch_macro_fred", return_value={}):
            with patch("operator1.clients.macro_wbgapi.fetch_macro_wbgapi", return_value=wb_data):
                result = fetch_macro("US", secrets={"FRED_API_KEY": "test"})

        assert len(result) >= 3
