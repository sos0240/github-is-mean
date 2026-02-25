"""Tests for operator1.clients.macro_bcb wrapper (Brazil BCB)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroBCB:
    """Tests for operator1.clients.macro_bcb."""

    def test_import(self):
        from operator1.clients.macro_bcb import fetch_macro_bcb
        assert callable(fetch_macro_bcb)

    def test_fetch_with_mock(self):
        from operator1.clients.macro_bcb import fetch_macro_bcb

        mock_df = pd.DataFrame(
            {"interest_rate": [13.75, 13.75, 13.25]},
            index=pd.date_range("2023-01-01", periods=3),
        )

        mock_sgs = MagicMock()
        mock_sgs.get.return_value = mock_df

        mock_bcb = MagicMock()
        mock_bcb.sgs = mock_sgs

        with patch.dict("sys.modules", {"bcb": mock_bcb, "bcb.sgs": mock_sgs}):
            result = fetch_macro_bcb(years=1)

        assert len(result) > 0

    def test_fetch_empty_on_import_error(self):
        from operator1.clients.macro_bcb import fetch_macro_bcb

        with patch.dict("sys.modules", {"bcb": None}):
            result = fetch_macro_bcb()
        assert isinstance(result, dict)
