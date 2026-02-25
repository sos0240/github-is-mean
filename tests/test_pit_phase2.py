"""Tests for Phase 2 PIT clients (ca/au/in/cn/hk/sg/mx/za/ch/sa/ae).

These all follow a similar stub pattern with market_id, market_name,
get_profile returning dict, and financial methods returning DataFrames.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


class TestCASedarClient:
    def test_import(self):
        from operator1.clients.ca_sedar import CASedarClient
        assert CASedarClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.ca_sedar import CASedarClient
        client = CASedarClient(cache_dir=tmp_path)
        assert client.market_id == "ca_sedar"
        assert "Canada" in client.market_name

    def test_get_profile_returns_dict(self, tmp_path):
        from operator1.clients.ca_sedar import CASedarClient
        client = CASedarClient(cache_dir=tmp_path)
        with patch("operator1.http_utils.cached_get", side_effect=Exception("no net")):
            profile = client.get_profile("RY")
        assert isinstance(profile, dict)
        assert profile.get("country") == "CA"

    def test_financial_methods_return_dataframe(self, tmp_path):
        from operator1.clients.ca_sedar import CASedarClient
        client = CASedarClient(cache_dir=tmp_path)
        assert isinstance(client.get_income_statement("RY"), pd.DataFrame)
        assert isinstance(client.get_balance_sheet("RY"), pd.DataFrame)
        assert isinstance(client.get_cashflow_statement("RY"), pd.DataFrame)

    def test_get_peers_returns_list(self, tmp_path):
        from operator1.clients.ca_sedar import CASedarClient
        client = CASedarClient(cache_dir=tmp_path)
        assert isinstance(client.get_peers("RY"), list)


class TestAUAsxClient:
    def test_import(self):
        from operator1.clients.au_asx import AUAsxClient
        assert AUAsxClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.au_asx import AUAsxClient
        client = AUAsxClient(cache_dir=tmp_path)
        assert client.market_id == "au_asx"
        assert "Australia" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.au_asx import AUAsxClient
        client = AUAsxClient(cache_dir=tmp_path)
        with patch("operator1.http_utils.cached_get", side_effect=Exception("no net")):
            profile = client.get_profile("BHP")
        assert isinstance(profile, dict)
        assert profile.get("country") == "AU"


class TestINBseClient:
    def test_import(self):
        from operator1.clients.in_bse import INBseClient
        assert INBseClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.in_bse import INBseClient
        client = INBseClient(cache_dir=tmp_path)
        assert client.market_id == "in_bse"
        assert "India" in client.market_name or "BSE" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.in_bse import INBseClient
        client = INBseClient(cache_dir=tmp_path)
        with patch("operator1.http_utils.cached_get", side_effect=Exception("no net")):
            profile = client.get_profile("TCS")
        assert isinstance(profile, dict)


class TestCNSseClient:
    def test_import(self):
        from operator1.clients.cn_sse import CNSseClient
        assert CNSseClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.cn_sse import CNSseClient
        client = CNSseClient(cache_dir=tmp_path)
        assert client.market_id == "cn_sse"
        assert "China" in client.market_name or "SSE" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.cn_sse import CNSseClient
        client = CNSseClient(cache_dir=tmp_path)
        with patch("operator1.http_utils.cached_get", side_effect=Exception("no net")):
            profile = client.get_profile("600519")
        assert isinstance(profile, dict)


class TestHKHkexClient:
    def test_import(self):
        from operator1.clients.hk_hkex import HKHkexClient
        assert HKHkexClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.hk_hkex import HKHkexClient
        client = HKHkexClient(cache_dir=tmp_path)
        assert client.market_id == "hk_hkex"
        assert "Hong Kong" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.hk_hkex import HKHkexClient
        client = HKHkexClient(cache_dir=tmp_path)
        profile = client.get_profile("0005")
        assert isinstance(profile, dict)
        assert profile.get("country") == "HK"

    def test_stub_methods(self, tmp_path):
        from operator1.clients.hk_hkex import HKHkexClient
        client = HKHkexClient(cache_dir=tmp_path)
        assert isinstance(client.get_income_statement("0005"), pd.DataFrame)
        assert isinstance(client.get_balance_sheet("0005"), pd.DataFrame)
        assert isinstance(client.get_cashflow_statement("0005"), pd.DataFrame)
        assert isinstance(client.get_quotes("0005"), pd.DataFrame)
        assert isinstance(client.get_peers("0005"), list)
        assert isinstance(client.get_executives("0005"), list)


class TestSGSgxClient:
    def test_import(self):
        from operator1.clients.sg_sgx import SGSgxClient
        assert SGSgxClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.sg_sgx import SGSgxClient
        client = SGSgxClient(cache_dir=tmp_path)
        assert client.market_id == "sg_sgx"
        assert "Singapore" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.sg_sgx import SGSgxClient
        client = SGSgxClient(cache_dir=tmp_path)
        profile = client.get_profile("D05")
        assert isinstance(profile, dict)
        assert profile.get("country") == "SG"


class TestMXBmvClient:
    def test_import(self):
        from operator1.clients.mx_bmv import MXBmvClient
        assert MXBmvClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.mx_bmv import MXBmvClient
        client = MXBmvClient(cache_dir=tmp_path)
        assert client.market_id == "mx_bmv"
        assert "Mexico" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.mx_bmv import MXBmvClient
        client = MXBmvClient(cache_dir=tmp_path)
        profile = client.get_profile("AMXL")
        assert isinstance(profile, dict)
        assert profile.get("country") == "MX"


class TestZAJseClient:
    def test_import(self):
        from operator1.clients.za_jse import ZAJseClient
        assert ZAJseClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.za_jse import ZAJseClient
        client = ZAJseClient(cache_dir=tmp_path)
        assert client.market_id == "za_jse"
        assert "South Africa" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.za_jse import ZAJseClient
        client = ZAJseClient(cache_dir=tmp_path)
        profile = client.get_profile("NPN")
        assert isinstance(profile, dict)
        assert profile.get("country") == "ZA"


class TestCHSixClient:
    def test_import(self):
        from operator1.clients.ch_six import CHSixClient
        assert CHSixClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.ch_six import CHSixClient
        client = CHSixClient(cache_dir=tmp_path)
        assert client.market_id == "ch_six"
        assert "Switzerland" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.ch_six import CHSixClient
        client = CHSixClient(cache_dir=tmp_path)
        profile = client.get_profile("NESN")
        assert isinstance(profile, dict)
        assert profile.get("country") == "CH"


class TestSATadawulClient:
    def test_import(self):
        from operator1.clients.sa_tadawul import SATadawulClient
        assert SATadawulClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.sa_tadawul import SATadawulClient
        client = SATadawulClient(cache_dir=tmp_path)
        assert client.market_id == "sa_tadawul"
        assert "Saudi" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.sa_tadawul import SATadawulClient
        client = SATadawulClient(cache_dir=tmp_path)
        profile = client.get_profile("2222")
        assert isinstance(profile, dict)
        assert profile.get("country") == "SA"


class TestAEDfmClient:
    def test_import(self):
        from operator1.clients.ae_dfm import AEDfmClient
        assert AEDfmClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.ae_dfm import AEDfmClient
        client = AEDfmClient(cache_dir=tmp_path)
        assert client.market_id == "ae_dfm"
        assert "UAE" in client.market_name or "DFM" in client.market_name

    def test_get_profile(self, tmp_path):
        from operator1.clients.ae_dfm import AEDfmClient
        client = AEDfmClient(cache_dir=tmp_path)
        profile = client.get_profile("EMAAR")
        assert isinstance(profile, dict)
        assert profile.get("country") == "AE"
