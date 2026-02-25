"""Tests for operator1.clients.br_cvm_wrapper.BRCvmClient."""

from __future__ import annotations

import pandas as pd
import pytest


class TestBRCvmClient:
    def test_import(self):
        from operator1.clients.br_cvm_wrapper import BRCvmClient, BRCvmError
        assert BRCvmClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.br_cvm_wrapper import BRCvmClient
        client = BRCvmClient(cache_dir=tmp_path)
        assert client.market_id == "br_cvm"

    def test_market_properties(self, tmp_path):
        from operator1.clients.br_cvm_wrapper import BRCvmClient
        client = BRCvmClient(cache_dir=tmp_path)
        assert "Brazil" in client.market_name or "CVM" in client.market_name

    def test_cache_operations(self, tmp_path):
        from operator1.clients.br_cvm_wrapper import BRCvmClient
        client = BRCvmClient(cache_dir=tmp_path)
        test_data = {"name": "Petrobras", "ticker": "PETR4"}
        client._write_cache("PETR4", "profile.json", test_data)
        result = client._read_cache("PETR4", "profile.json")
        assert result is not None
        assert result["name"] == "Petrobras"

    def test_error_class(self):
        from operator1.clients.br_cvm_wrapper import BRCvmError
        err = BRCvmError("profile", "not found")
        assert "profile" in str(err)
        assert "not found" in str(err)
