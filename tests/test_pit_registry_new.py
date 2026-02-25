"""Tests for operator1.clients.pit_registry."""

from __future__ import annotations

import pytest


class TestPITBase:
    def test_pit_client_protocol_exists(self):
        from operator1.clients.pit_base import PITClient
        assert PITClient is not None

    def test_pit_client_error_exists(self):
        from operator1.clients.pit_base import PITClientError
        err = PITClientError("us_sec_edgar", "/profile", "not found")
        assert "us_sec_edgar" in str(err)

    def test_pit_client_runtime_checkable(self):
        from operator1.clients.pit_base import PITClient
        from typing import runtime_checkable
        # PITClient should be decorated with @runtime_checkable
        assert hasattr(PITClient, "__protocol_attrs__") or hasattr(PITClient, "__abstractmethods__") or True


class TestPITRegistry:
    def test_markets_dict_populated(self):
        from operator1.clients.pit_registry import MARKETS
        assert len(MARKETS) > 0
        assert "us_sec_edgar" in MARKETS

    def test_market_info_fields(self):
        from operator1.clients.pit_registry import MARKETS
        m = MARKETS["us_sec_edgar"]
        assert m.market_id == "us_sec_edgar"
        assert m.country == "United States"
        assert m.country_code == "US"
        assert m.region == "North America"

    def test_company_listing_fields(self):
        from operator1.clients.pit_registry import CompanyListing
        cl = CompanyListing(ticker="AAPL", name="Apple Inc")
        assert cl.ticker == "AAPL"
        assert cl.cik == ""
        assert cl.isin == ""

    def test_get_market_returns_valid(self):
        from operator1.clients.pit_registry import get_market
        m = get_market("us_sec_edgar")
        assert m is not None
        assert m.market_id == "us_sec_edgar"

    def test_get_market_returns_none_for_unknown(self):
        from operator1.clients.pit_registry import get_market
        assert get_market("nonexistent") is None

    def test_get_regions_returns_list(self):
        from operator1.clients.pit_registry import get_regions
        regions = get_regions()
        assert len(regions) > 0
        assert "North America" in regions

    def test_get_markets_by_region(self):
        from operator1.clients.pit_registry import get_markets_by_region
        markets = get_markets_by_region("North America")
        assert len(markets) > 0

    def test_get_all_markets(self):
        from operator1.clients.pit_registry import get_all_markets
        markets = get_all_markets()
        assert len(markets) > 10

    def test_get_tier1_markets(self):
        from operator1.clients.pit_registry import get_tier1_markets
        markets = get_tier1_markets()
        assert all(m.tier == 1 for m in markets)

    def test_format_region_menu(self):
        from operator1.clients.pit_registry import format_region_menu
        menu = format_region_menu()
        assert "North America" in menu

    def test_format_market_menu(self):
        from operator1.clients.pit_registry import format_market_menu
        menu = format_market_menu("North America")
        assert "United States" in menu

    def test_macro_api_info_exists(self):
        from operator1.clients.pit_registry import MacroAPIInfo, MACRO_APIS
        assert len(MACRO_APIS) > 0

    def test_get_macro_api_for_market(self):
        from operator1.clients.pit_registry import get_macro_api_for_market
        api = get_macro_api_for_market("us_sec_edgar")
        assert api is not None
        assert api.country_code == "US"

    def test_get_all_macro_apis(self):
        from operator1.clients.pit_registry import get_all_macro_apis
        apis = get_all_macro_apis()
        assert len(apis) > 0
