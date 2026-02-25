"""Tests for operator1.clients.uk_ch_wrapper.UKCompaniesHouseClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestUKCompaniesHouseClient:
    """Tests for operator1.clients.uk_ch_wrapper.UKCompaniesHouseClient."""

    def test_import(self):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient, UKCompaniesHouseError
        assert UKCompaniesHouseClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path / "cache")
        assert client.market_id == "uk_companies_house"

    def test_market_properties(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(cache_dir=tmp_path / "cache")
        assert client.market_id == "uk_companies_house"
        assert "UK" in client.market_name or "United Kingdom" in client.market_name

    def test_cache_path(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(cache_dir=tmp_path / "cache")
        path = client._cache_path("00000001", "profile.json")
        assert "00000001" in str(path).upper()

    def test_get_profile_from_cache(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(cache_dir=tmp_path / "cache")
        cache_data = {"name": "TEST PLC", "ticker": "TEST", "country": "GB"}
        client._write_cache("TEST", "profile.json", cache_data)
        result = client._read_cache("TEST", "profile.json")
        assert result is not None
        assert result["name"] == "TEST PLC"
