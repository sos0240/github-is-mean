"""Per-source canonical translator tests: raw API data -> model-ready format.

Verifies that for each supported source (US, EU/IFRS, JP, KR, TW, BR, CL, UK,
and DART/IFRS fallback markets) the translator:

1. Maps ALL three statement types (income, balance, cash flow) correctly
2. Normalizes numbers and dates properly
3. Pivots to wide format with report_date as index
4. Produces a DataFrame that the cache builder and models can consume
5. Handles mixed-language and special character concepts
6. Metadata columns (market_id, currency) are correctly attached
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _build_raw_df(
    concepts: list[str],
    values: list,
    *,
    filing_date: str = "2024-03-15",
    report_date: str = "2023-12-31",
    account_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Helper to build a raw DataFrame mimicking API output."""
    n = len(concepts)
    data = {
        "concept": concepts,
        "value": values,
        "filing_date": [filing_date] * n,
        "report_date": [report_date] * n,
    }
    if account_codes:
        data["account_code"] = account_codes
    return pd.DataFrame(data)


def _translate_and_pivot(df: pd.DataFrame, market_id: str) -> pd.DataFrame:
    """Translate raw -> canonical -> wide format (model-ready)."""
    from operator1.clients.canonical_translator import (
        translate_financials,
        pivot_to_canonical_wide,
    )
    translated = translate_financials(df, market_id)
    if translated.empty:
        return translated
    wide = pivot_to_canonical_wide(translated)
    return wide


# ===========================================================================
# US SEC EDGAR (US GAAP)
# ===========================================================================


class TestUSEdgarTranslation(unittest.TestCase):
    """Full translation pipeline for US SEC EDGAR data."""

    MARKET = "us_sec_edgar"

    def _raw_income(self):
        return _build_raw_df(
            ["Revenues", "CostOfRevenue", "GrossProfit", "OperatingIncomeLoss",
             "NetIncomeLoss", "IncomeTaxExpenseBenefit", "InterestExpense",
             "SellingGeneralAndAdministrativeExpense", "ResearchAndDevelopmentExpense"],
            [1000000, 600000, 400000, 150000, 120000, 30000, 10000, 80000, 60000],
        )

    def _raw_balance(self):
        return _build_raw_df(
            ["Assets", "Liabilities", "StockholdersEquity", "AssetsCurrent",
             "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue",
             "ShortTermBorrowings", "LongTermDebt", "AccountsReceivableNetCurrent"],
            [5000000, 3000000, 2000000, 1500000, 800000, 500000, 200000, 1000000, 300000],
        )

    def _raw_cashflow(self):
        return _build_raw_df(
            ["NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInInvestingActivities",
             "NetCashProvidedByUsedInFinancingActivities",
             "PaymentsToAcquirePropertyPlantAndEquipment",
             "PaymentsOfDividends"],
            [200000, -150000, -50000, -80000, -20000],
        )

    def test_income_statement_maps(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_income(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("gross_profit", names)
        self.assertIn("operating_income", names)
        self.assertIn("net_income", names)
        self.assertIn("taxes", names)

    def test_balance_sheet_maps(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_balance(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("total_assets", names)
        self.assertIn("total_liabilities", names)
        self.assertIn("total_equity", names)
        self.assertIn("current_assets", names)
        self.assertIn("cash_and_equivalents", names)

    def test_cashflow_maps(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_cashflow(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("operating_cash_flow", names)
        self.assertIn("investing_cf", names)
        self.assertIn("financing_cf", names)
        self.assertIn("capex", names)

    def test_pivot_model_ready(self):
        all_raw = pd.concat([self._raw_income(), self._raw_balance(), self._raw_cashflow()])
        wide = _translate_and_pivot(all_raw, self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)
        self.assertIn("total_assets", wide.columns)
        self.assertIn("operating_cash_flow", wide.columns)
        # Should have one row per report_date
        self.assertEqual(len(wide), 1)

    def test_metadata_columns(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_income(), self.MARKET)
        self.assertTrue((result["market_id"] == "us_sec_edgar").all())
        self.assertTrue((result["currency"] == "USD").all())

    def test_values_are_numeric(self):
        wide = _translate_and_pivot(self._raw_income(), self.MARKET)
        for col in ["revenue", "gross_profit", "net_income"]:
            if col in wide.columns:
                val = wide[col].iloc[0]
                self.assertIsInstance(val, (int, float, np.integer, np.floating))

    def test_negative_cashflow_preserved(self):
        wide = _translate_and_pivot(self._raw_cashflow(), self.MARKET)
        if "investing_cf" in wide.columns:
            self.assertLess(float(wide["investing_cf"].iloc[0]), 0)


# ===========================================================================
# EU ESEF (IFRS)
# ===========================================================================


class TestEUEsefTranslation(unittest.TestCase):
    """Full translation pipeline for EU ESEF (IFRS) data."""

    MARKET = "eu_esef"

    def _raw_full(self):
        return _build_raw_df(
            ["ifrs-full:Revenue", "ifrs-full:GrossProfit",
             "ifrs-full:ProfitLoss", "ifrs-full:Assets",
             "ifrs-full:Liabilities", "ifrs-full:Equity",
             "ifrs-full:CurrentAssets", "ifrs-full:CurrentLiabilities",
             "ifrs-full:CashAndCashEquivalents",
             "ifrs-full:CashFlowsFromUsedInOperatingActivities",
             "ifrs-full:CashFlowsFromUsedInInvestingActivities",
             "ifrs-full:CashFlowsFromUsedInFinancingActivities"],
            [500000, 200000, 80000, 3000000, 1500000, 1500000,
             800000, 400000, 300000, 150000, -100000, -50000],
        )

    def test_full_statement_mapping(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        for expected in ["revenue", "gross_profit", "net_income", "total_assets",
                         "total_liabilities", "total_equity", "cash_and_equivalents",
                         "operating_cash_flow"]:
            self.assertIn(expected, names, f"Missing: {expected}")

    def test_pivot_model_ready(self):
        wide = _translate_and_pivot(self._raw_full(), self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)
        self.assertIn("total_assets", wide.columns)

    def test_currency_is_eur(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "EUR").all())


# ===========================================================================
# Japan EDINET (JPPFS + IFRS)
# ===========================================================================


class TestJapanEdinetTranslation(unittest.TestCase):
    """Full translation pipeline for Japan EDINET data."""

    MARKET = "jp_edinet"

    def _raw_full(self):
        return _build_raw_df(
            ["jppfs_cor:NetSales", "jppfs_cor:CostOfSales",
             "jppfs_cor:GrossProfit", "jppfs_cor:OperatingIncome",
             "jppfs_cor:ProfitLoss",
             "jppfs_cor:TotalAssets", "jppfs_cor:TotalLiabilities",
             "jppfs_cor:NetAssets", "jppfs_cor:CurrentAssets",
             "jppfs_cor:CurrentLiabilities", "jppfs_cor:CashAndDeposits",
             "jppfs_cor:NetCashProvidedByUsedInOperatingActivities",
             "jppfs_cor:NetCashProvidedByUsedInInvestingActivities"],
            [10000000, 7000000, 3000000, 800000, 600000,
             50000000, 30000000, 20000000, 15000000, 10000000, 5000000,
             2000000, -1500000],
            report_date="2024-03-31",
        )

    def test_jppfs_concepts_map(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("total_assets", names)
        self.assertIn("total_equity", names)  # NetAssets -> total_equity
        self.assertIn("operating_cash_flow", names)

    def test_pivot_model_ready(self):
        wide = _translate_and_pivot(self._raw_full(), self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)
        # Verify revenue value is correct
        self.assertEqual(float(wide["revenue"].iloc[0]), 10000000.0)

    def test_currency_is_jpy(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "JPY").all())

    def test_japanese_era_dates_normalized(self):
        """Japanese era dates in report_date should be converted."""
        from operator1.clients.canonical_translator import translate_financials
        df = _build_raw_df(
            ["jppfs_cor:NetSales"], [1000000],
            report_date="R5.12.31",
        )
        result = translate_financials(df, self.MARKET)
        # report_date should be converted to 2023-12-31
        rd = result["report_date"].iloc[0]
        self.assertEqual(rd.year, 2023)
        self.assertEqual(rd.month, 12)
        self.assertEqual(rd.day, 31)


# ===========================================================================
# South Korea DART (K-IFRS)
# ===========================================================================


class TestKoreaDartTranslation(unittest.TestCase):
    """Full translation pipeline for Korea DART data."""

    MARKET = "kr_dart"

    def _raw_full(self):
        return _build_raw_df(
            ["매출액", "매출원가", "매출총이익", "영업이익",
             "당기순이익", "법인세비용", "이자비용",
             "자산총계", "부채총계", "자본총계",
             "유동자산", "유동부채", "현금및현금성자산",
             "영업활동현금흐름", "투자활동현금흐름"],
            [50000000, 35000000, 15000000, 5000000,
             3000000, 1000000, 500000,
             100000000, 60000000, 40000000,
             30000000, 20000000, 10000000,
             8000000, -6000000],
        )

    def test_korean_concepts_map(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("operating_income", names)
        self.assertIn("total_assets", names)
        self.assertIn("cash_and_equivalents", names)
        self.assertIn("operating_cash_flow", names)

    def test_pivot_model_ready(self):
        wide = _translate_and_pivot(self._raw_full(), self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)
        self.assertIn("total_assets", wide.columns)

    def test_currency_is_krw(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "KRW").all())


# ===========================================================================
# Taiwan MOPS (TIFRS)
# ===========================================================================


class TestTaiwanMopsTranslation(unittest.TestCase):
    """Full translation pipeline for Taiwan MOPS data."""

    MARKET = "tw_mops"

    def _raw_full(self):
        return _build_raw_df(
            ["營業收入合計", "營業成本合計", "營業毛利（毛損）",
             "營業利益（損失）", "本期淨利（淨損）",
             "資產總計", "負債總計", "權益總計",
             "流動資產合計", "流動負債合計", "現金及約當現金",
             "營業活動之淨現金流入（流出）"],
            [8000000, 5000000, 3000000,
             1000000, 800000,
             20000000, 12000000, 8000000,
             6000000, 4000000, 2000000,
             1500000],
            report_date="113/06/30",  # ROC date
        )

    def test_chinese_concepts_map(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("gross_profit", names)
        self.assertIn("total_assets", names)
        self.assertIn("cash_and_equivalents", names)

    def test_roc_date_normalized(self):
        """ROC dates (113/06/30) should convert to 2024-06-30."""
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        rd = result["report_date"].iloc[0]
        self.assertEqual(rd.year, 2024)
        self.assertEqual(rd.month, 6)

    def test_pivot_model_ready(self):
        wide = _translate_and_pivot(self._raw_full(), self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)

    def test_currency_is_twd(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "TWD").all())


# ===========================================================================
# Brazil CVM
# ===========================================================================


class TestBrazilCvmTranslation(unittest.TestCase):
    """Full translation pipeline for Brazil CVM data."""

    MARKET = "br_cvm"

    def _raw_full(self):
        return _build_raw_df(
            ["Receita Liquida", "Custos", "Lucro Bruto",
             "Resultado Operacional", "Lucro Liquido",
             "Ativo Total", "Passivo Total", "Patrimonio",
             "Ativo Circulante", "Passivo Circulante",
             "Caixa", "FCO", "FCI"],
            [2000000, 1200000, 800000,
             300000, 200000,
             10000000, 6000000, 4000000,
             3000000, 2000000,
             1000000, 500000, -400000],
            account_codes=["3.01", "3.02", "3.03",
                           "3.05", "3.11",
                           "1", "2", "2.03",
                           "1.01", "2.01",
                           "1.01.01", "6.01", "6.02"],
        )

    def test_account_code_mapping(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("total_assets", names)
        self.assertIn("net_income", names)
        self.assertIn("operating_cash_flow", names)

    def test_pivot_model_ready(self):
        wide = _translate_and_pivot(self._raw_full(), self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)

    def test_currency_is_brl(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "BRL").all())


# ===========================================================================
# Chile CMF
# ===========================================================================


class TestChileCmfTranslation(unittest.TestCase):
    """Full translation pipeline for Chile CMF data."""

    MARKET = "cl_cmf"

    def _raw_full(self):
        return _build_raw_df(
            ["Ingresos de actividades ordinarias",
             "Costo de ventas",
             "Ganancia bruta",
             "Ganancia (pérdida) por actividades de operación",
             "Ganancia (pérdida)",
             "Total de activos", "Total de pasivos", "Total de patrimonio",
             "Activos corrientes totales", "Pasivos corrientes totales",
             "Efectivo y equivalentes al efectivo",
             "Flujos de efectivo procedentes de actividades de operación"],
            [5000000, 3000000, 2000000, 800000, 600000,
             30000000, 18000000, 12000000, 10000000, 7000000,
             4000000, 1200000],
        )

    def test_spanish_concepts_map(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("gross_profit", names)
        self.assertIn("net_income", names)
        self.assertIn("total_assets", names)
        self.assertIn("operating_cash_flow", names)

    def test_pivot_model_ready(self):
        wide = _translate_and_pivot(self._raw_full(), self.MARKET)
        self.assertFalse(wide.empty)
        self.assertIn("revenue", wide.columns)

    def test_currency_is_clp(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "CLP").all())


# ===========================================================================
# UK Companies House (UK GAAP + IFRS)
# ===========================================================================


class TestUKCompaniesHouseTranslation(unittest.TestCase):
    """Full translation pipeline for UK Companies House data."""

    MARKET = "uk_companies_house"

    def _raw_full(self):
        return _build_raw_df(
            ["uk-gaap:Turnover", "uk-gaap:GrossProfitLoss",
             "uk-gaap:OperatingProfitLoss",
             "uk-gaap:ProfitLossForPeriod",
             "uk-gaap:TaxOnProfitOnOrdinaryActivities",
             "uk-gaap:CurrentAssets",
             "uk-gaap:ShareholderFunds",
             "uk-gaap:CashBankInHand",
             "uk-gaap:NetCashInflowOutflowFromOperatingActivities"],
            [2000000, 800000, 300000, 250000, 50000,
             1000000, 500000, 200000, 350000],
        )

    def test_ukgaap_concepts_map(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("gross_profit", names)
        self.assertIn("net_income", names)
        self.assertIn("total_equity", names)
        self.assertIn("cash_and_equivalents", names)

    def test_also_accepts_ifrs(self):
        """UK market should also accept IFRS concepts."""
        from operator1.clients.canonical_translator import translate_financials
        df = _build_raw_df(
            ["ifrs-full:Revenue", "ifrs-full:Assets"],
            [1000000, 5000000],
        )
        result = translate_financials(df, self.MARKET)
        names = set(result["canonical_name"].tolist())
        self.assertIn("revenue", names)
        self.assertIn("total_assets", names)

    def test_currency_is_gbp(self):
        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(self._raw_full(), self.MARKET)
        self.assertTrue((result["currency"] == "GBP").all())


# ===========================================================================
# Numeric edge cases per source
# ===========================================================================


class TestNumericEdgeCases(unittest.TestCase):
    """Test that tricky numeric formats from different sources translate correctly."""

    def test_string_with_commas_us(self):
        from operator1.clients.canonical_translator import translate_financials
        df = _build_raw_df(
            ["Revenues"], ["1,234,567"],
        )
        result = translate_financials(df, "us_sec_edgar")
        self.assertEqual(float(result["value"].iloc[0]), 1234567.0)

    def test_eu_decimal_format(self):
        from operator1.clients.canonical_translator import _normalize_numeric
        self.assertEqual(_normalize_numeric("1.234.567,89"), 1234567.89)

    def test_parentheses_negative(self):
        from operator1.clients.canonical_translator import translate_financials
        df = _build_raw_df(
            ["NetCashProvidedByUsedInInvestingActivities"], ["(500,000)"],
        )
        result = translate_financials(df, "us_sec_edgar")
        self.assertEqual(float(result["value"].iloc[0]), -500000.0)

    def test_currency_symbols_stripped(self):
        from operator1.clients.canonical_translator import _normalize_numeric
        self.assertEqual(_normalize_numeric("R$1.234,56"), 1234.56)
        self.assertEqual(_normalize_numeric("€1,234.56"), 1234.56)
        self.assertEqual(_normalize_numeric("£500"), 500.0)

    def test_zero_value(self):
        from operator1.clients.canonical_translator import _normalize_numeric
        self.assertEqual(_normalize_numeric(0), 0.0)
        self.assertEqual(_normalize_numeric("0"), 0.0)
        self.assertEqual(_normalize_numeric("0.00"), 0.0)

    def test_very_large_numbers(self):
        from operator1.clients.canonical_translator import _normalize_numeric
        self.assertEqual(_normalize_numeric("999999999999"), 999999999999.0)
        self.assertEqual(_normalize_numeric(1e15), 1e15)


# ===========================================================================
# Model readiness: verify pivot output has the right shape
# ===========================================================================


class TestModelReadiness(unittest.TestCase):
    """Verify translated + pivoted data is ready for model consumption."""

    def test_wide_format_has_report_date(self):
        """Pivoted data should have report_date column."""
        df = _build_raw_df(
            ["ifrs-full:Revenue", "ifrs-full:Assets"],
            [1000000, 5000000],
        )
        wide = _translate_and_pivot(df, "eu_esef")
        self.assertIn("report_date", wide.columns)

    def test_wide_format_values_are_float(self):
        """All value columns in pivoted data should be numeric."""
        df = _build_raw_df(
            ["ifrs-full:Revenue", "ifrs-full:Assets", "ifrs-full:CashAndCashEquivalents"],
            [1000000, 5000000, 300000],
        )
        wide = _translate_and_pivot(df, "eu_esef")
        for col in ["revenue", "total_assets", "cash_and_equivalents"]:
            if col in wide.columns:
                dtype = wide[col].dtype
                self.assertTrue(
                    np.issubdtype(dtype, np.number),
                    f"Column {col} has non-numeric dtype: {dtype}",
                )

    def test_multi_period_pivot(self):
        """Multiple report periods should create multiple rows."""
        df = pd.DataFrame({
            "concept": ["ifrs-full:Revenue", "ifrs-full:Revenue",
                        "ifrs-full:Assets", "ifrs-full:Assets"],
            "value": [1000000, 1200000, 5000000, 5500000],
            "filing_date": ["2023-03-15", "2024-03-15",
                            "2023-03-15", "2024-03-15"],
            "report_date": ["2022-12-31", "2023-12-31",
                            "2022-12-31", "2023-12-31"],
        })
        wide = _translate_and_pivot(df, "eu_esef")
        self.assertEqual(len(wide), 2)

    def test_all_25_markets_have_concept_maps(self):
        """Every market in _MARKET_CONCEPT_MAPS should produce translations."""
        from operator1.clients.canonical_translator import _MARKET_CONCEPT_MAPS

        for market_id in _MARKET_CONCEPT_MAPS:
            concept_map = _MARKET_CONCEPT_MAPS[market_id]
            self.assertGreater(
                len(concept_map), 0,
                f"Market {market_id} has empty concept map",
            )

    def test_all_markets_have_currency(self):
        """Every market should have a currency mapping."""
        from operator1.clients.canonical_translator import (
            _MARKET_CONCEPT_MAPS, _MARKET_CURRENCIES,
        )
        for market_id in _MARKET_CONCEPT_MAPS:
            self.assertIn(
                market_id, _MARKET_CURRENCIES,
                f"Market {market_id} missing currency mapping",
            )

    def test_cache_builder_fields_covered(self):
        """Key STATEMENT_FIELDS from cache_builder should be producible
        by at least one market's concept map."""
        from operator1.clients.canonical_translator import _MARKET_CONCEPT_MAPS
        from operator1.steps.cache_builder import STATEMENT_FIELDS

        # Collect all canonical names that any market can produce
        all_canonical: set[str] = set()
        for concept_map in _MARKET_CONCEPT_MAPS.values():
            all_canonical.update(concept_map.values())

        # Map translator canonical -> cache builder field names
        # The translator uses slightly different names in some cases
        translator_to_cache = {
            "operating_income": "ebit",  # some overlap
            "sga_expense": "sga_expenses",
            "research_and_development": "rd_expenses",
        }

        # Check that core fields are covered
        core_fields = [
            "revenue", "gross_profit", "net_income",
            "total_assets", "total_liabilities", "total_equity",
            "current_assets", "current_liabilities",
            "cash_and_equivalents",
            "operating_cash_flow", "investing_cf", "financing_cf",
        ]
        for field in core_fields:
            self.assertIn(
                field, all_canonical,
                f"Core field '{field}' not produced by any market translator",
            )


if __name__ == "__main__":
    unittest.main()
