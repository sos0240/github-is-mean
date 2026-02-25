"""Tests for operator1.clients.cl_cmf_wrapper.CLCmfClient."""

from __future__ import annotations

import pytest


class TestCLCmfClient:
    def test_import(self):
        from operator1.clients.cl_cmf_wrapper import CLCmfClient, CLCmfError
        assert CLCmfClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.cl_cmf_wrapper import CLCmfClient
        client = CLCmfClient(cache_dir=tmp_path)
        assert client.market_id == "cl_cmf"

    def test_market_properties(self, tmp_path):
        from operator1.clients.cl_cmf_wrapper import CLCmfClient
        client = CLCmfClient(cache_dir=tmp_path)
        assert "Chile" in client.market_name or "CMF" in client.market_name

    def test_cache_operations(self, tmp_path):
        from operator1.clients.cl_cmf_wrapper import CLCmfClient
        client = CLCmfClient(cache_dir=tmp_path)
        test_data = {"name": "SQM", "ticker": "SQM-B"}
        client._write_cache("SQM-B", "profile.json", test_data)
        result = client._read_cache("SQM-B", "profile.json")
        assert result is not None
