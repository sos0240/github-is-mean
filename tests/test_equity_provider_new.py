"""Tests for operator1.clients.equity_provider (create_pit_client + registry)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestEquityProviderProtocol:
    def test_equity_provider_protocol(self):
        from operator1.clients.equity_provider import EquityProvider
        assert EquityProvider is not None

    def test_market_choices_populated(self):
        from operator1.clients.equity_provider import MARKET_CHOICES
        assert len(MARKET_CHOICES) > 0
        assert "us_sec_edgar" in MARKET_CHOICES


class TestCreatePitClient:
    def test_create_pit_client_us(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("us_sec_edgar")
        assert client.market_id == "us_sec_edgar"

    def test_create_pit_client_uk(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("uk_companies_house")
        assert client.market_id == "uk_companies_house"

    def test_create_pit_client_eu(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("eu_esef")
        assert client.market_id == "eu_esef"

    def test_create_pit_client_jp(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("jp_jquants")
        assert client.market_id == "jp_jquants"

    def test_create_pit_client_kr(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("kr_dart")
        assert client.market_id == "kr_dart"

    def test_create_pit_client_tw(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("tw_mops")
        assert client.market_id == "tw_mops"

    def test_create_pit_client_br(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("br_cvm")
        assert client.market_id == "br_cvm"

    def test_create_pit_client_cl(self):
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client("cl_cmf")
        assert client.market_id == "cl_cmf"

    def test_create_pit_client_phase2_markets(self):
        from operator1.clients.equity_provider import create_pit_client
        phase2_ids = [
            "ca_sedar", "au_asx", "in_bse", "cn_sse", "hk_hkex",
            "sg_sgx", "mx_bmv", "za_jse", "ch_six", "sa_tadawul", "ae_dfm",
        ]
        for mid in phase2_ids:
            client = create_pit_client(mid)
            assert client.market_id == mid, f"Failed for {mid}"

    def test_create_pit_client_eu_variants(self):
        from operator1.clients.equity_provider import create_pit_client
        for mid in ["nl_esef", "es_esef", "it_esef", "se_esef"]:
            client = create_pit_client(mid)
            assert client.market_id == mid

    def test_create_pit_client_unknown_raises_system_exit(self):
        from operator1.clients.equity_provider import create_pit_client
        with pytest.raises(SystemExit):
            create_pit_client("unknown_market_xyz")

    def test_create_equity_provider_backward_compat(self):
        from operator1.clients.equity_provider import create_equity_provider
        client = create_equity_provider(secrets={}, market_id="us_sec_edgar")
        assert client.market_id == "us_sec_edgar"
