"""Tests for operator1.clients.macro_sdmx wrapper (ECB/EU)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroSDMX:
    """Tests for operator1.clients.macro_sdmx."""

    def test_import(self):
        from operator1.clients.macro_sdmx import fetch_macro_ecb
        assert callable(fetch_macro_ecb)

    def test_ecb_rest_base_url(self):
        from operator1.clients.macro_sdmx import _ECB_REST_BASE
        assert "ecb.europa.eu" in _ECB_REST_BASE

    def test_fetch_returns_dict(self):
        from operator1.clients.macro_sdmx import fetch_macro_ecb

        with patch.dict("sys.modules", {"sdmx": None}):
            with patch("requests.get", side_effect=Exception("no network")):
                result = fetch_macro_ecb()
        assert isinstance(result, dict)
