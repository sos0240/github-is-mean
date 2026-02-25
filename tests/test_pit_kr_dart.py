"""Tests for operator1.clients.kr_dart_wrapper.KRDartClient."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestKRDartClient:
    def test_import(self):
        from operator1.clients.kr_dart_wrapper import KRDartClient, KRDartError
        assert KRDartClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.kr_dart_wrapper import KRDartClient
        with patch.dict(os.environ, {}, clear=True):
            client = KRDartClient(api_key="", cache_dir=tmp_path)
        assert client.market_id == "kr_dart"

    def test_market_properties(self, tmp_path):
        from operator1.clients.kr_dart_wrapper import KRDartClient
        client = KRDartClient(cache_dir=tmp_path)
        assert "Korea" in client.market_name

    def test_cache_operations(self, tmp_path):
        from operator1.clients.kr_dart_wrapper import KRDartClient
        client = KRDartClient(cache_dir=tmp_path)
        test_data = {"name": "Samsung", "ticker": "005930"}
        client._write_cache("005930", "profile.json", test_data)
        result = client._read_cache("005930", "profile.json")
        assert result is not None
        assert result["name"] == "Samsung"
