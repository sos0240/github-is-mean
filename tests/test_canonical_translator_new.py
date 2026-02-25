"""Tests for operator1.clients.canonical_translator."""

from __future__ import annotations

import pandas as pd
import pytest


class TestCanonicalTranslator:
    def test_canonical_sets_defined(self):
        from operator1.clients.canonical_translator import (
            CANONICAL_INCOME, CANONICAL_BALANCE, CANONICAL_CASHFLOW, CANONICAL_PROFILE,
        )
        assert "revenue" in CANONICAL_INCOME
        assert "total_assets" in CANONICAL_BALANCE
        assert "operating_cash_flow" in CANONICAL_CASHFLOW
        assert "name" in CANONICAL_PROFILE

    def test_ifrs_map_exists(self):
        from operator1.clients.canonical_translator import _IFRS_MAP
        assert len(_IFRS_MAP) > 0
        assert "ifrs-full:Revenue" in _IFRS_MAP

    def test_normalize_roc_date(self):
        from operator1.clients.canonical_translator import _normalize_roc_date
        assert _normalize_roc_date("113/01/15") == "2024-01-15"
        assert _normalize_roc_date("1130115") == "2024-01-15"
        assert _normalize_roc_date("2024-01-15") == "2024-01-15"

    def test_normalize_japanese_era_date(self):
        from operator1.clients.canonical_translator import _normalize_japanese_era_date
        assert _normalize_japanese_era_date("R5.12.31") == "2023-12-31"
        assert _normalize_japanese_era_date("2023-12-31") == "2023-12-31"

    def test_normalize_numeric(self):
        from operator1.clients.canonical_translator import _normalize_numeric
        assert _normalize_numeric("1,234,567") == 1234567.0
        assert _normalize_numeric("(500)") == -500.0
        assert _normalize_numeric(None) is None
        assert _normalize_numeric(42) == 42.0
        assert _normalize_numeric("1.234,56") == 1234.56

    def test_translate_financials_basic(self):
        from operator1.clients.canonical_translator import translate_financials
        df = pd.DataFrame({
            "concept": ["ifrs-full:Revenue", "ifrs-full:Assets"],
            "value": ["1,000,000", "5,000,000"],
            "filing_date": ["2023-06-15", "2023-06-15"],
            "report_date": ["2023-03-31", "2023-03-31"],
        })
        result = translate_financials(df, market_id="eu_esef")
        assert not result.empty
        assert "canonical_name" in result.columns
        assert "market_id" in result.columns

    def test_translate_empty_df(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(pd.DataFrame(), market_id="us_sec_edgar")
        assert result.empty

    def test_translate_profile(self):
        from operator1.clients.canonical_translator import translate_profile
        raw = {"company_name": "Test Corp", "ticker": "TEST", "country": "US"}
        result = translate_profile(raw, market_id="us_sec_edgar")
        assert result.get("name") == "Test Corp"

    def test_get_concept_map_returns_dict(self):
        from operator1.clients.canonical_translator import get_concept_map
        cmap = get_concept_map("us_sec_edgar")
        assert isinstance(cmap, dict)
        assert len(cmap) > 0

    def test_map_concept(self):
        from operator1.clients.canonical_translator import _map_concept, _IFRS_MAP
        result = _map_concept("ifrs-full:Revenue", _IFRS_MAP)
        assert result == "revenue"
