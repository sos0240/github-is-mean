"""Tests for operator1.clients.tw_mops_wrapper.TWMopsClient."""

from __future__ import annotations

import pytest


class TestTWMopsClient:
    def test_import(self):
        from operator1.clients.tw_mops_wrapper import TWMopsClient, TWMopsError
        assert TWMopsClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.tw_mops_wrapper import TWMopsClient
        client = TWMopsClient(cache_dir=tmp_path)
        assert client.market_id == "tw_mops"

    def test_roc_date_conversion(self):
        from operator1.clients.tw_mops_wrapper import _roc_to_ce, _ce_to_roc
        assert _roc_to_ce(113) == 2024
        assert _roc_to_ce(112) == 2023
        assert _ce_to_roc(2024) == 113
        assert _ce_to_roc(2023) == 112

    def test_market_name(self, tmp_path):
        from operator1.clients.tw_mops_wrapper import TWMopsClient
        client = TWMopsClient(cache_dir=tmp_path)
        assert "Taiwan" in client.market_name or "MOPS" in client.market_name
