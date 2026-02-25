"""Tests for operator1.clients.jp_jquants_wrapper.JPJquantsClient."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestJPJquantsClient:
    """Tests for operator1.clients.jp_jquants_wrapper.JPJquantsClient."""

    def test_import(self):
        from operator1.clients.jp_jquants_wrapper import JPJquantsClient, JPJquantsError
        assert JPJquantsClient is not None

    def test_instantiation_without_key(self, tmp_path):
        from operator1.clients.jp_jquants_wrapper import JPJquantsClient

        with patch.dict(os.environ, {}, clear=True):
            client = JPJquantsClient(api_key="", cache_dir=tmp_path)
        assert client._client is None

    def test_client_class_exists(self, tmp_path):
        """JPJquantsClient is a raw client; market_id is on the adapter."""
        from operator1.clients.jp_jquants_wrapper import JPJquantsClient
        client = JPJquantsClient(cache_dir=tmp_path)
        # The raw client doesn't expose market_id -- the adapter does
        assert hasattr(client, "get_profile")
        assert hasattr(client, "get_financials")

    def test_v2_column_mappings(self):
        from operator1.clients.jp_jquants_wrapper import (
            _V2_INCOME_MAP,
            _V2_BALANCE_MAP,
            _V2_CASHFLOW_MAP,
        )
        assert "Sales" in _V2_INCOME_MAP
        assert _V2_INCOME_MAP["Sales"] == "revenue"
        assert "TA" in _V2_BALANCE_MAP
        assert _V2_BALANCE_MAP["TA"] == "total_assets"
        assert "CFO" in _V2_CASHFLOW_MAP

    def test_get_profile_without_client(self, tmp_path):
        from operator1.clients.jp_jquants_wrapper import JPJquantsClient

        with patch.dict(os.environ, {}, clear=True):
            client = JPJquantsClient(cache_dir=tmp_path)
            profile = client.get_profile("7203")
        assert isinstance(profile, dict)


class TestJPJquantsAdapter:
    """Tests for the _JPJquantsAdapter in equity_provider.py."""

    def test_adapter_wraps_client(self):
        from operator1.clients.equity_provider import _JPJquantsAdapter

        mock_client = MagicMock()
        mock_client.list_companies.return_value = [{"ticker": "7203", "name": "Toyota"}]
        mock_client.get_profile.return_value = {"name": "Toyota", "ticker": "7203"}
        mock_client.get_peers.return_value = ["7267", "7261"]
        mock_client.get_executives.return_value = []
        mock_client.get_financials.return_value = {
            "income": pd.DataFrame({"revenue": [1000]}),
            "balance": pd.DataFrame({"total_assets": [5000]}),
            "cashflow": pd.DataFrame({"operating_cashflow": [200]}),
        }

        adapter = _JPJquantsAdapter(mock_client)
        assert adapter.market_id == "jp_jquants"
        assert "Japan" in adapter.market_name

        companies = adapter.list_companies("toyota")
        assert len(companies) == 1

        profile = adapter.get_profile("7203")
        assert profile["ticker"] == "7203"

        income = adapter.get_income_statement("7203")
        assert not income.empty

        balance = adapter.get_balance_sheet("7203")
        assert not balance.empty

        cashflow = adapter.get_cashflow_statement("7203")
        assert not cashflow.empty

        quotes = adapter.get_quotes("7203")
        assert isinstance(quotes, pd.DataFrame)
