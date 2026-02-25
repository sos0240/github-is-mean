"""Tests for operator1.clients.eu_esef_wrapper.EUEsefClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestEUEsefClient:
    """Tests for operator1.clients.eu_esef_wrapper.EUEsefClient."""

    def test_import(self):
        from operator1.clients.eu_esef_wrapper import EUEsefClient, EUEsefError
        assert EUEsefClient is not None

    def test_instantiation_default(self, tmp_path):
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        client = EUEsefClient(cache_dir=tmp_path / "cache")
        assert client.market_id == "eu_esef"

    def test_instantiation_france(self, tmp_path):
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        client = EUEsefClient(country_code="FR", market_id="fr_esef", cache_dir=tmp_path)
        assert client._country_code == "FR"
        assert client.market_id == "fr_esef"

    def test_instantiation_germany(self, tmp_path):
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        client = EUEsefClient(country_code="DE", market_id="de_esef", cache_dir=tmp_path)
        assert client._country_code == "DE"

    def test_returns_empty_statements(self, tmp_path):
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        client = EUEsefClient(cache_dir=tmp_path)
        with patch("operator1.http_utils.cached_get", side_effect=Exception("no network")):
            result = client.get_income_statement("TEST_ISIN")
        assert isinstance(result, pd.DataFrame)
