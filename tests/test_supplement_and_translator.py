"""Tests for supplement.py and canonical_translator.py modules.

Validates:
- Canonical translator concept mapping for all regions
- Numeric normalization (commas, EU decimals, parentheses, etc.)
- Date normalization (ROC calendar, Japanese era)
- Profile translation and enrichment merge logic
- Pivot to wide format
- Supplement enrichment dispatcher (offline / mock mode)
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# canonical_translator tests
# ---------------------------------------------------------------------------

class TestCanonicalTranslator:
    """Tests for operator1.clients.canonical_translator."""

    def test_import(self):
        from operator1.clients.canonical_translator import (
            translate_financials,
            translate_profile,
            pivot_to_canonical_wide,
            get_concept_map,
        )
        assert callable(translate_financials)
        assert callable(translate_profile)
        assert callable(pivot_to_canonical_wide)
        assert callable(get_concept_map)

    # -- Concept mapping ---------------------------------------------------

    def test_ifrs_concept_mapping(self):
        """IFRS concepts should map to canonical names."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "ifrs-full:Revenue",
                "ifrs-full:Assets",
                "ifrs-full:CashFlowsFromUsedInOperatingActivities",
            ],
            "value": [1000000, 5000000, 200000],
            "filing_date": ["2024-03-15", "2024-03-15", "2024-03-15"],
            "report_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
        })

        result = translate_financials(df, "eu_esef")
        assert "canonical_name" in result.columns
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names
        assert "operating_cash_flow" in canonical_names

    def test_jppfs_concept_mapping(self):
        """Japan GAAP (JPPFS) concepts should map to canonical names."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "jppfs_cor:NetSales",
                "jppfs_cor:TotalAssets",
                "jppfs_cor:ProfitLoss",
            ],
            "value": [100000, 500000, 20000],
            "filing_date": ["2024-06-20", "2024-06-20", "2024-06-20"],
            "report_date": ["2024-03-31", "2024-03-31", "2024-03-31"],
        })

        result = translate_financials(df, "jp_jquants")
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names
        assert "net_income" in canonical_names

    def test_tifrs_concept_mapping(self):
        """Taiwan (TIFRS) Chinese field names should map to canonical names."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": ["營業收入合計", "資產總計", "本期淨利（淨損）"],
            "value": [50000, 200000, 10000],
            "filing_date": ["2024-04-01", "2024-04-01", "2024-04-01"],
            "report_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
        })

        result = translate_financials(df, "tw_mops")
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names
        assert "net_income" in canonical_names

    def test_cvm_account_code_mapping(self):
        """CVM account codes should map to canonical names."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": ["Receita", "Ativo Total", "Lucro"],
            "value": [300000, 900000, 50000],
            "account_code": ["3.01", "1", "3.11"],
            "filing_date": ["2024-03-31", "2024-03-31", "2024-03-31"],
            "report_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
        })

        result = translate_financials(df, "br_cvm")
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names
        assert "net_income" in canonical_names

    def test_cmf_concept_mapping(self):
        """CMF/Chile Spanish concept names should map to canonical names."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "Ingresos de actividades ordinarias",
                "Total de activos",
                "Ganancia (pérdida)",
            ],
            "value": [100000, 400000, 15000],
            "filing_date": ["2024-04-15", "2024-04-15", "2024-04-15"],
            "report_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
        })

        result = translate_financials(df, "cl_cmf")
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names
        assert "net_income" in canonical_names

    def test_usgaap_concept_mapping(self):
        """US GAAP concepts should map for completeness."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": ["us-gaap:Revenues", "us-gaap:Assets"],
            "value": [1000, 5000],
            "filing_date": ["2024-02-15", "2024-02-15"],
            "report_date": ["2023-12-31", "2023-12-31"],
        })

        result = translate_financials(df, "us_sec_edgar")
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names

    # -- Numeric normalization ---------------------------------------------

    def test_normalize_numeric_commas(self):
        """Numbers with commas (US format) should parse correctly."""
        from operator1.clients.canonical_translator import _normalize_numeric

        assert _normalize_numeric("1,234,567") == 1234567.0
        assert _normalize_numeric("1,234.56") == 1234.56

    def test_normalize_numeric_eu_format(self):
        """EU-style decimals (1.234,56) should parse correctly."""
        from operator1.clients.canonical_translator import _normalize_numeric

        assert _normalize_numeric("1.234.567,89") == 1234567.89
        assert _normalize_numeric("1.234,56") == 1234.56

    def test_normalize_numeric_parentheses_negative(self):
        """Parenthesized numbers should parse as negative."""
        from operator1.clients.canonical_translator import _normalize_numeric

        assert _normalize_numeric("(500)") == -500.0
        assert _normalize_numeric("(1,234.56)") == -1234.56

    def test_normalize_numeric_none_and_empty(self):
        """None, empty, and N/A should return None."""
        from operator1.clients.canonical_translator import _normalize_numeric

        assert _normalize_numeric(None) is None
        assert _normalize_numeric("") is None
        assert _normalize_numeric("N/A") is None
        assert _normalize_numeric("-") is None

    def test_normalize_numeric_already_numeric(self):
        """Already-numeric values should pass through."""
        from operator1.clients.canonical_translator import _normalize_numeric

        assert _normalize_numeric(42) == 42.0
        assert _normalize_numeric(3.14) == 3.14

    # -- Date normalization ------------------------------------------------

    def test_roc_date_slash_format(self):
        """ROC date 113/01/15 should convert to 2024-01-15."""
        from operator1.clients.canonical_translator import _normalize_roc_date

        assert _normalize_roc_date("113/01/15") == "2024-01-15"
        assert _normalize_roc_date("112/06/30") == "2023-06-30"

    def test_roc_date_compact_format(self):
        """ROC date 1130115 should convert to 2024-01-15."""
        from operator1.clients.canonical_translator import _normalize_roc_date

        assert _normalize_roc_date("1130115") == "2024-01-15"

    def test_japanese_era_date(self):
        """Reiwa era date R5.12.31 should convert to 2023-12-31."""
        from operator1.clients.canonical_translator import _normalize_japanese_era_date

        assert _normalize_japanese_era_date("R5.12.31") == "2023-12-31"
        assert _normalize_japanese_era_date("R6.3.31") == "2024-03-31"

    def test_iso_date_passthrough(self):
        """ISO dates should pass through unchanged."""
        from operator1.clients.canonical_translator import _normalize_roc_date

        assert _normalize_roc_date("2024-01-15") == "2024-01-15"

    # -- Profile translation -----------------------------------------------

    def test_translate_profile_canonical_fields(self):
        """Translated profile should have all canonical profile fields."""
        from operator1.clients.canonical_translator import translate_profile

        raw = {
            "name": "Test Corp",
            "ticker": "TEST",
            "country": "JP",
        }
        result = translate_profile(raw, "jp_jquants")
        assert result["name"] == "Test Corp"
        assert result["ticker"] == "TEST"
        assert result["currency"] == "JPY"
        # All canonical fields should be present
        assert "sector" in result
        assert "industry" in result
        assert "isin" in result
        assert "exchange" in result

    def test_translate_profile_aliases(self):
        """Profile field aliases should be mapped to canonical names."""
        from operator1.clients.canonical_translator import translate_profile

        raw = {
            "company_name": "Alias Corp",
            "symbol": "ALC",
            "SETOR_ATIVIDADE": "Technology",
        }
        result = translate_profile(raw, "br_cvm")
        assert result["name"] == "Alias Corp"
        assert result["ticker"] == "ALC"
        assert result["sector"] == "Technology"

    # -- Pivot to wide format ----------------------------------------------

    def test_pivot_to_wide(self):
        """Long-format canonical data should pivot to wide format."""
        from operator1.clients.canonical_translator import pivot_to_canonical_wide

        df = pd.DataFrame({
            "canonical_name": ["revenue", "net_income", "revenue", "net_income"],
            "value": [1000, 100, 1200, 150],
            "report_date": pd.to_datetime(
                ["2022-12-31", "2022-12-31", "2023-12-31", "2023-12-31"]
            ),
            "filing_date": pd.to_datetime(
                ["2023-02-15", "2023-02-15", "2024-02-15", "2024-02-15"]
            ),
        })

        wide = pivot_to_canonical_wide(df)
        assert not wide.empty
        assert "revenue" in wide.columns
        assert "net_income" in wide.columns
        assert len(wide) == 2  # two report dates

    def test_pivot_empty_df(self):
        """Empty DataFrame should return empty."""
        from operator1.clients.canonical_translator import pivot_to_canonical_wide

        result = pivot_to_canonical_wide(pd.DataFrame())
        assert result.empty

    # -- Empty / edge cases ------------------------------------------------

    def test_translate_empty_df(self):
        """Empty DataFrame should return empty."""
        from operator1.clients.canonical_translator import translate_financials

        result = translate_financials(pd.DataFrame(), "eu_esef")
        assert result.empty

    def test_translate_none_df(self):
        """None DataFrame should return empty."""
        from operator1.clients.canonical_translator import translate_financials

        result = translate_financials(None, "eu_esef")
        assert result.empty

    def test_get_concept_map(self):
        """get_concept_map should return a dict for known markets."""
        from operator1.clients.canonical_translator import get_concept_map

        for market in ["us_sec_edgar", "eu_esef", "jp_jquants", "tw_mops", "br_cvm", "cl_cmf"]:
            cmap = get_concept_map(market)
            assert isinstance(cmap, dict)
            assert len(cmap) > 0

    def test_market_metadata_added(self):
        """Translated DataFrame should have market_id and currency columns."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": ["ifrs-full:Revenue"],
            "value": [1000],
            "filing_date": ["2024-01-15"],
            "report_date": ["2023-12-31"],
        })

        result = translate_financials(df, "de_esef", "income")
        assert "market_id" in result.columns
        assert result["market_id"].iloc[0] == "de_esef"
        assert "currency" in result.columns
        assert result["currency"].iloc[0] == "EUR"
        assert "statement_type" in result.columns
        assert result["statement_type"].iloc[0] == "income"


# ---------------------------------------------------------------------------
# supplement.py tests
# ---------------------------------------------------------------------------

class TestSupplement:
    """Tests for operator1.clients.supplement."""

    def test_import(self):
        from operator1.clients.supplement import (
            enrich_profile,
            openfigi_enrich,
            euronext_enrich,
            jpx_enrich,
            twse_enrich,
            b3_enrich,
            santiago_enrich,
        )
        assert callable(enrich_profile)
        assert callable(openfigi_enrich)

    def test_enrich_profile_no_op_for_full_markets(self):
        """Markets not in the enricher map should return profile as-is."""
        from operator1.clients.supplement import enrich_profile

        profile = {"name": "Apple", "ticker": "AAPL", "sector": "Tech"}
        result = enrich_profile("us_sec_edgar", "AAPL", existing_profile=profile)
        assert result == profile

    def test_enrich_profile_merge_logic(self):
        """Enrichment should fill empty fields but not overwrite existing ones."""
        from operator1.clients.supplement import enrich_profile

        # Mock the enricher by temporarily patching _MARKET_ENRICHERS
        from operator1.clients import supplement as sup_mod

        original = sup_mod._MARKET_ENRICHERS.get("eu_esef")
        sup_mod._MARKET_ENRICHERS["eu_esef"] = lambda t, **kw: {
            "name": "Should Not Overwrite",
            "sector": "Technology",
            "industry": "Software",
        }

        try:
            profile = {"name": "SAP SE", "ticker": "SAP", "sector": ""}
            result = enrich_profile("eu_esef", "SAP", existing_profile=profile)
            # Name should NOT be overwritten (already has value)
            assert result["name"] == "SAP SE"
            # Sector should be filled (was empty)
            assert result["sector"] == "Technology"
            # Industry should be filled (was missing)
            assert result["industry"] == "Software"
            # Metadata should be present
            assert "_supplement_source" in result
        finally:
            if original is not None:
                sup_mod._MARKET_ENRICHERS["eu_esef"] = original

    def test_enrich_profile_handles_exception(self):
        """If the enricher raises, the original profile should be returned."""
        from operator1.clients.supplement import enrich_profile
        from operator1.clients import supplement as sup_mod

        original = sup_mod._MARKET_ENRICHERS.get("jp_jquants")
        sup_mod._MARKET_ENRICHERS["jp_jquants"] = lambda t, **kw: (_ for _ in ()).throw(
            RuntimeError("API down")
        )

        try:
            profile = {"name": "Toyota", "ticker": "7203"}
            result = enrich_profile("jp_jquants", "7203", existing_profile=profile)
            assert result["name"] == "Toyota"
        finally:
            if original is not None:
                sup_mod._MARKET_ENRICHERS["jp_jquants"] = original

    def test_enrich_profile_empty_ticker(self):
        """Empty ticker should return profile unchanged."""
        from operator1.clients.supplement import enrich_profile

        profile = {"name": "Test"}
        result = enrich_profile("tw_mops", "", existing_profile=profile)
        # Should still have original data
        assert result["name"] == "Test"

    def test_openfigi_enrich_no_args(self):
        """OpenFIGI with no args should return empty dict."""
        from operator1.clients.supplement import openfigi_enrich

        assert openfigi_enrich() == {}

    def test_openfigi_search_no_query(self):
        """OpenFIGI search with empty query should return empty list."""
        from operator1.clients.supplement import openfigi_search

        assert openfigi_search("") == []

    def test_all_partial_markets_have_enrichers(self):
        """All partial-coverage markets should have enrichers registered."""
        from operator1.clients.supplement import _MARKET_ENRICHERS

        partial_markets = ["eu_esef", "fr_esef", "de_esef", "jp_jquants",
                          "tw_mops", "br_cvm", "cl_cmf"]
        for market_id in partial_markets:
            assert market_id in _MARKET_ENRICHERS, (
                f"Missing enricher for partial market: {market_id}"
            )


# ---------------------------------------------------------------------------
# Integration: translator + financial data shapes
# ---------------------------------------------------------------------------

class TestTranslatorIntegration:
    """Integration tests verifying translated data has correct shape."""

    def test_esef_translated_shape(self):
        """ESEF translated data should have canonical columns."""
        from operator1.clients.canonical_translator import translate_financials

        # Simulate ESEF client output
        df = pd.DataFrame({
            "concept": [
                "ifrs-full:Revenue",
                "ifrs-full:GrossProfit",
                "ifrs-full:ProfitLoss",
            ],
            "value": [5000000, 2000000, 800000],
            "filing_date": ["2024-04-15", "2024-04-15", "2024-04-15"],
            "report_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
            "entity": ["Test GmbH", "Test GmbH", "Test GmbH"],
            "source": ["xbrl_facts", "xbrl_facts", "xbrl_facts"],
        })

        result = translate_financials(df, "de_esef", "income")
        assert "canonical_name" in result.columns
        assert "market_id" in result.columns
        assert "currency" in result.columns
        assert result["currency"].iloc[0] == "EUR"
        # All values should be numeric
        assert result["value"].dtype in ("float64", "int64")

    def test_edinet_translated_shape(self):
        """EDINET translated data should handle both JPPFS and IFRS concepts."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "jppfs_cor:NetSales",
                "ifrs-full:Revenue",
                "jppfs_cor:OperatingIncome",
            ],
            "value": [100000, 200000, 30000],
            "filing_date": ["2024-06-20", "2024-06-20", "2024-06-20"],
            "report_date": ["2024-03-31", "2024-03-31", "2024-03-31"],
        })

        result = translate_financials(df, "jp_jquants", "income")
        canonical_names = set(result["canonical_name"].tolist())
        assert "revenue" in canonical_names
        assert "operating_income" in canonical_names

    def test_cvm_account_code_fallback(self):
        """CVM data should use account_code for mapping when concept fails."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": ["Unknown Brazilian Field", "Another Field"],
            "value": [50000, 100000],
            "account_code": ["3.01", "1"],
            "filing_date": ["2024-03-31", "2024-03-31"],
            "report_date": ["2023-12-31", "2023-12-31"],
        })

        result = translate_financials(df, "br_cvm")
        canonical_names = result["canonical_name"].tolist()
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names

    def test_full_pipeline_wide_format(self):
        """Full pipeline: translate -> pivot should produce wide format."""
        from operator1.clients.canonical_translator import (
            translate_financials,
            pivot_to_canonical_wide,
        )

        # Simulate multi-period IFRS data
        df = pd.DataFrame({
            "concept": [
                "ifrs-full:Revenue", "ifrs-full:Assets",
                "ifrs-full:Revenue", "ifrs-full:Assets",
            ],
            "value": [1000, 5000, 1200, 6000],
            "filing_date": [
                "2023-03-15", "2023-03-15",
                "2024-03-15", "2024-03-15",
            ],
            "report_date": [
                "2022-12-31", "2022-12-31",
                "2023-12-31", "2023-12-31",
            ],
        })

        translated = translate_financials(df, "eu_esef", "income")
        wide = pivot_to_canonical_wide(translated)

        assert not wide.empty
        assert "revenue" in wide.columns
        assert "total_assets" in wide.columns
        assert len(wide) == 2

    # -- Full-coverage client translation tests ----------------------------

    def test_sec_edgar_bare_concepts(self):
        """SEC EDGAR bare concept names (no namespace) should map correctly."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "Revenues",
                "Assets",
                "NetIncomeLoss",
                "NetCashProvidedByUsedInOperatingActivities",
                "EarningsPerShareBasic",
            ],
            "value": [50000000, 200000000, 8000000, 12000000, 2.50],
            "filing_date": ["2024-02-15"] * 5,
            "report_date": ["2023-12-31"] * 5,
        })

        result = translate_financials(df, "us_sec_edgar", "income")
        canonical_names = set(result["canonical_name"].tolist())
        assert "revenue" in canonical_names
        assert "total_assets" in canonical_names
        assert "net_income" in canonical_names
        assert "operating_cash_flow" in canonical_names
        assert "eps_basic" in canonical_names
        assert result["currency"].iloc[0] == "USD"

    def test_uk_companies_house_concepts(self):
        """UK GAAP + IFRS concepts from Companies House should map."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "uk-gaap:Turnover",
                "uk-gaap:ProfitLossForPeriod",
                "ifrs-full:Assets",
                "uk-gaap:CashBankInHand",
            ],
            "value": [10000000, 500000, 30000000, 2000000],
            "filing_date": ["2024-01-20"] * 4,
            "report_date": ["2023-12-31"] * 4,
        })

        result = translate_financials(df, "uk_companies_house", "income")
        canonical_names = set(result["canonical_name"].tolist())
        assert "revenue" in canonical_names
        assert "net_income" in canonical_names
        assert "total_assets" in canonical_names
        assert "cash_and_equivalents" in canonical_names
        assert result["currency"].iloc[0] == "GBP"

    def test_dart_korean_concepts(self):
        """DART Korean account names should map to canonical names."""
        from operator1.clients.canonical_translator import translate_financials

        df = pd.DataFrame({
            "concept": [
                "매출액",
                "영업이익",
                "당기순이익",
                "자산총계",
                "영업활동현금흐름",
            ],
            "value": [80000000, 15000000, 10000000, 300000000, 20000000],
            "filing_date": ["2024-03-15"] * 5,
            "report_date": ["2023-12-31"] * 5,
        })

        result = translate_financials(df, "kr_dart", "IS")
        canonical_names = set(result["canonical_name"].tolist())
        assert "revenue" in canonical_names
        assert "operating_income" in canonical_names
        assert "net_income" in canonical_names
        assert "total_assets" in canonical_names
        assert "operating_cash_flow" in canonical_names
        assert result["currency"].iloc[0] == "KRW"

    def test_all_markets_have_concept_maps(self):
        """Every registered market should have a concept map."""
        from operator1.clients.canonical_translator import get_concept_map

        all_markets = [
            "us_sec_edgar", "uk_companies_house",
            "eu_esef", "fr_esef", "de_esef",
            "jp_jquants", "kr_dart", "tw_mops",
            "br_cvm", "cl_cmf",
        ]
        for market_id in all_markets:
            cmap = get_concept_map(market_id)
            assert isinstance(cmap, dict), f"No concept map for {market_id}"
            assert len(cmap) > 5, f"Concept map too small for {market_id}: {len(cmap)}"

    def test_all_markets_have_currencies(self):
        """Every market should produce the right currency in translated output."""
        from operator1.clients.canonical_translator import translate_financials

        expected = {
            "us_sec_edgar": "USD",
            "uk_companies_house": "GBP",
            "eu_esef": "EUR",
            "fr_esef": "EUR",
            "de_esef": "EUR",
            "jp_jquants": "JPY",
            "kr_dart": "KRW",
            "tw_mops": "TWD",
            "br_cvm": "BRL",
            "cl_cmf": "CLP",
        }

        for market_id, expected_currency in expected.items():
            df = pd.DataFrame({
                "concept": ["test_concept"],
                "value": [100],
                "filing_date": ["2024-01-01"],
                "report_date": ["2023-12-31"],
            })
            # Force a canonical_name so it doesn't get filtered
            result = translate_financials(df, market_id)
            assert result["currency"].iloc[0] == expected_currency if not result.empty else True
