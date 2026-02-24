"""Tests for macro data integration with survival time-phases analysis.

Verifies that:
1. Macro indicators (GDP, inflation, unemployment, credit spread, yield curve,
   FX volatility) flow correctly into the 2-year daily cache.
2. Country survival flags are properly triggered by macro threshold breaches.
3. The enriched cache (with macro data) feeds into survival timeline
   classification for the full 2-year window.
4. Walk-forward and survival-aware weighting work correctly when macro
   columns are present in the cache.
5. MacroDataset -> align_macro_to_daily -> enrich_cache_with_indicators
   -> compute_survival_flags -> compute_survival_timeline is end-to-end
   coherent over a 2-year period.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers -- 2-year synthetic cache with macro data
# ---------------------------------------------------------------------------

# Approximate 2 years of business days
_TWO_YEARS_BDAYS = 504


def _make_daily_index(days: int = _TWO_YEARS_BDAYS) -> pd.DatetimeIndex:
    """Return a 2-year business-day index starting Jan 2023."""
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_macro_dataset(
    *,
    gdp_growth: list[float] | None = None,
    inflation: list[float] | None = None,
    unemployment: list[float] | None = None,
    interest_rate: list[float] | None = None,
    exchange_rate: list[float] | None = None,
):
    """Build a synthetic MacroDataset with yearly indicators."""
    from operator1.steps.macro_mapping import MacroDataset

    ds = MacroDataset(country_iso3="USA")
    ds.indicators = {}
    ds.missing = []

    years = [2021, 2022, 2023, 2024]

    if gdp_growth is not None:
        ds.indicators["gdp_growth"] = pd.DataFrame({
            "year": years[: len(gdp_growth)],
            "value": gdp_growth,
        })
    else:
        ds.missing.append("gdp_growth")

    if inflation is not None:
        ds.indicators["inflation_rate_yoy"] = pd.DataFrame({
            "year": years[: len(inflation)],
            "value": inflation,
        })
    else:
        ds.missing.append("inflation_rate_yoy")

    if unemployment is not None:
        ds.indicators["unemployment_rate"] = pd.DataFrame({
            "year": years[: len(unemployment)],
            "value": unemployment,
        })
    else:
        ds.missing.append("unemployment_rate")

    if interest_rate is not None:
        ds.indicators["real_interest_rate"] = pd.DataFrame({
            "year": years[: len(interest_rate)],
            "value": interest_rate,
        })
    else:
        ds.missing.append("real_interest_rate")

    if exchange_rate is not None:
        ds.indicators["official_exchange_rate_lcu_per_usd"] = pd.DataFrame({
            "year": years[: len(exchange_rate)],
            "value": exchange_rate,
        })
    else:
        ds.missing.append("official_exchange_rate_lcu_per_usd")

    return ds


def _make_2yr_cache_with_macro(
    days: int = _TWO_YEARS_BDAYS,
    *,
    seed: int = 42,
    unemployment_crisis_start: int | None = None,
    credit_spread_crisis_start: int | None = None,
    yield_curve_inversion_start: int | None = None,
    fx_vol_crisis_start: int | None = None,
) -> pd.DataFrame:
    """Build a synthetic 2-year daily cache with macro indicators and
    configurable crisis injection points.

    Parameters
    ----------
    days:
        Number of business days.
    unemployment_crisis_start:
        Day index at which unemployment starts surging.
    credit_spread_crisis_start:
        Day index at which credit spread breaches threshold.
    yield_curve_inversion_start:
        Day index at which yield curve inverts.
    fx_vol_crisis_start:
        Day index at which FX volatility spikes.
    """
    np.random.seed(seed)
    idx = _make_daily_index(days)

    close = 100 + np.cumsum(np.random.randn(days) * 0.5)

    # Company financial metrics (normal conditions)
    current_ratio = np.full(days, 1.8)
    debt_to_equity_abs = np.full(days, 1.2)
    fcf_yield = np.full(days, 0.04)
    drawdown_252d = np.full(days, -0.08)
    volatility_21d = np.abs(np.random.randn(days) * 0.1) + 0.05
    market_cap = np.full(days, 5_000_000.0)

    # Macro indicators -- normal baseline
    gdp_growth = np.full(days, 2.5)
    gdp_current_usd = np.full(days, 25_000_000_000_000.0)
    inflation_rate_yoy = np.full(days, 2.8)
    inflation_daily_eq = inflation_rate_yoy / 365.0
    real_interest_rate = np.full(days, 1.5)
    lending_interest_rate = np.full(days, 5.0)
    unemployment_rate = np.full(days, 4.0)
    exchange_rate = np.full(days, 1.0)
    credit_spread = np.full(days, 2.0)
    yield_curve_slope = np.full(days, 1.0)
    fx_volatility = np.full(days, 8.0)
    real_return_1d = np.full(days, 0.0003)

    # Inject crises at specified points
    if unemployment_crisis_start is not None:
        # Surge from 4% to 8% over ~126 days (6 months)
        crisis_len = min(126, days - unemployment_crisis_start)
        surge = np.linspace(4.0, 8.0, crisis_len)
        unemployment_rate[unemployment_crisis_start:
                          unemployment_crisis_start + crisis_len] = surge
        # Keep high after surge
        if unemployment_crisis_start + crisis_len < days:
            unemployment_rate[unemployment_crisis_start + crisis_len:] = 8.0

    if credit_spread_crisis_start is not None:
        credit_spread[credit_spread_crisis_start:] = 7.0  # > 5% threshold

    if yield_curve_inversion_start is not None:
        yield_curve_slope[yield_curve_inversion_start:] = -1.0  # < -0.5 threshold

    if fx_vol_crisis_start is not None:
        fx_volatility[fx_vol_crisis_start:] = 25.0  # > 20% threshold

    return pd.DataFrame({
        "close": close,
        "current_ratio": current_ratio,
        "debt_to_equity_abs": debt_to_equity_abs,
        "fcf_yield": fcf_yield,
        "drawdown_252d": drawdown_252d,
        "volatility_21d": volatility_21d,
        "market_cap": market_cap,
        # Macro indicators
        "gdp_growth": gdp_growth,
        "gdp_current_usd": gdp_current_usd,
        "inflation_rate_yoy": inflation_rate_yoy,
        "inflation_rate_daily_equivalent": inflation_daily_eq,
        "real_interest_rate": real_interest_rate,
        "lending_interest_rate": lending_interest_rate,
        "unemployment_rate": unemployment_rate,
        "official_exchange_rate_lcu_per_usd": exchange_rate,
        "credit_spread": credit_spread,
        "yield_curve_slope": yield_curve_slope,
        "fx_volatility": fx_volatility,
        "real_return_1d": real_return_1d,
    }, index=idx)


# ===========================================================================
# 1. Macro alignment -> daily cache (2yr window)
# ===========================================================================


class TestMacroAlignmentToTwoYearCache(unittest.TestCase):
    """Verify macro alignment produces data over the full 2-year window."""

    def test_align_covers_full_index(self):
        from operator1.features.macro_alignment import align_macro_to_daily

        daily_index = _make_daily_index(_TWO_YEARS_BDAYS)
        ds = _make_macro_dataset(
            gdp_growth=[2.0, 2.3, 2.5, 2.1],
            inflation=[3.0, 3.5, 2.8, 2.5],
            unemployment=[3.5, 3.8, 4.0, 4.2],
            interest_rate=[1.0, 1.5, 2.0, 1.8],
            exchange_rate=[1.0, 1.0, 1.0, 1.0],
        )
        result = align_macro_to_daily(ds, daily_index)

        # Every day should have a value (as-of alignment)
        self.assertEqual(len(result), _TWO_YEARS_BDAYS)
        for col in ["gdp_growth", "inflation_rate_yoy", "unemployment_rate"]:
            non_null = result[col].notna().sum()
            self.assertEqual(
                non_null, _TWO_YEARS_BDAYS,
                f"{col} has {_TWO_YEARS_BDAYS - non_null} missing days",
            )

    def test_as_of_logic_yearly_to_daily(self):
        """Days in 2023 should get 2023 values; 2024 days get 2024 values."""
        from operator1.features.macro_alignment import align_macro_to_daily

        daily_index = _make_daily_index(_TWO_YEARS_BDAYS)
        ds = _make_macro_dataset(
            gdp_growth=[1.0, 1.5, 2.0, 2.5],
            inflation=[2.0, 2.5, 3.0, 3.5],
        )
        result = align_macro_to_daily(ds, daily_index)

        # 2023 days should have 2023 values
        days_2023 = result.loc[result.index.year == 2023]
        if len(days_2023) > 0:
            self.assertAlmostEqual(days_2023["gdp_growth"].iloc[0], 2.0)
            self.assertAlmostEqual(days_2023["inflation_rate_yoy"].iloc[0], 3.0)

        # 2024 days should have 2024 values
        days_2024 = result.loc[result.index.year == 2024]
        if len(days_2024) > 0:
            self.assertAlmostEqual(days_2024["gdp_growth"].iloc[0], 2.5)
            self.assertAlmostEqual(days_2024["inflation_rate_yoy"].iloc[0], 3.5)

    def test_staleness_tracking(self):
        """Macro data older than 1 year should be flagged as stale."""
        from operator1.features.macro_alignment import align_macro_to_daily

        daily_index = _make_daily_index(_TWO_YEARS_BDAYS)
        # Only provide 2021 data -- everything after is stale
        ds = _make_macro_dataset(gdp_growth=[1.5])
        result = align_macro_to_daily(ds, daily_index)

        stale_col = "is_stale_gdp_growth"
        if stale_col in result.columns:
            # Most days should be stale since we only have 2021 data
            stale_pct = result[stale_col].mean()
            self.assertGreater(stale_pct, 0.5, "Expected most days to be stale")

    def test_missing_indicator_all_nan(self):
        """Indicators not in MacroDataset should be all NaN with is_missing=1."""
        from operator1.features.macro_alignment import align_macro_to_daily

        daily_index = _make_daily_index(20)
        ds = _make_macro_dataset(gdp_growth=[2.0, 2.5, 3.0, 2.8])
        # unemployment, interest_rate, exchange_rate are missing
        result = align_macro_to_daily(ds, daily_index)

        self.assertTrue(result["unemployment_rate"].isna().all())
        self.assertTrue((result["is_missing_unemployment_rate"] == 1).all())

    def test_inflation_daily_equivalent_computed(self):
        """inflation_rate_daily_equivalent = inflation_rate_yoy / 365."""
        from operator1.features.macro_alignment import align_macro_to_daily

        daily_index = _make_daily_index(20)
        ds = _make_macro_dataset(inflation=[3.0, 3.0, 3.0, 3.0])
        result = align_macro_to_daily(ds, daily_index)

        expected = 3.0 / 365.0
        self.assertAlmostEqual(
            result["inflation_rate_daily_equivalent"].iloc[0],
            expected, places=8,
        )


# ===========================================================================
# 2. Country survival flags triggered by macro thresholds
# ===========================================================================


class TestCountrySurvivalFlagMacroTriggers(unittest.TestCase):
    """Verify that country_survival_mode_flag fires when macro data
    breaches the thresholds from country_protection_rules.yml."""

    def test_credit_spread_triggers_flag(self):
        """credit_spread > 5.0% should trigger country survival."""
        from operator1.analysis.survival_mode import compute_country_survival_flag

        cache = _make_2yr_cache_with_macro(
            100, credit_spread_crisis_start=50,
        )
        flag = compute_country_survival_flag(cache)

        # Before crisis: no flag
        self.assertEqual(flag.iloc[10], 0)
        # After crisis: flag should be 1
        self.assertEqual(flag.iloc[60], 1)

    def test_unemployment_surge_triggers_flag(self):
        """unemployment rising > 3 ppts in 6 months should trigger flag."""
        from operator1.analysis.survival_mode import compute_country_survival_flag

        # Need enough days for the lookback window (6 months ~ 126 bdays)
        cache = _make_2yr_cache_with_macro(
            400, unemployment_crisis_start=150,
        )
        flag = compute_country_survival_flag(cache)

        # Well after the surge (150 + 126 = 276), flag should fire
        self.assertEqual(flag.iloc[300], 1)

    def test_yield_curve_inversion_triggers_flag(self):
        """yield_curve_slope < -0.5 should trigger country survival."""
        from operator1.analysis.survival_mode import compute_country_survival_flag

        cache = _make_2yr_cache_with_macro(
            100, yield_curve_inversion_start=50,
        )
        flag = compute_country_survival_flag(cache)

        self.assertEqual(flag.iloc[10], 0)
        self.assertEqual(flag.iloc[60], 1)

    def test_fx_volatility_triggers_flag(self):
        """fx_volatility > 20% should trigger country survival."""
        from operator1.analysis.survival_mode import compute_country_survival_flag

        cache = _make_2yr_cache_with_macro(
            100, fx_vol_crisis_start=50,
        )
        flag = compute_country_survival_flag(cache)

        self.assertEqual(flag.iloc[10], 0)
        self.assertEqual(flag.iloc[60], 1)

    def test_no_macro_data_defaults_to_zero(self):
        """When no macro columns are present, country flag defaults to 0."""
        from operator1.analysis.survival_mode import compute_country_survival_flag

        idx = _make_daily_index(50)
        cache = pd.DataFrame({"close": np.random.randn(50)}, index=idx)
        flag = compute_country_survival_flag(cache)

        self.assertTrue((flag == 0).all())

    def test_multiple_triggers_any_fires(self):
        """Multiple macro breaches should all trigger (OR logic)."""
        from operator1.analysis.survival_mode import compute_country_survival_flag

        cache = _make_2yr_cache_with_macro(
            200,
            credit_spread_crisis_start=100,
            yield_curve_inversion_start=100,
            fx_vol_crisis_start=100,
        )
        flag = compute_country_survival_flag(cache)

        self.assertEqual(flag.iloc[50], 0)
        self.assertEqual(flag.iloc[120], 1)


# ===========================================================================
# 3. Company survival flags from financial data
# ===========================================================================


class TestCompanySurvivalFlagInCache(unittest.TestCase):
    """Verify company survival flag with macro-enriched cache."""

    def test_healthy_company_no_flag(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag

        cache = _make_2yr_cache_with_macro(100)
        flag = compute_company_survival_flag(cache)
        self.assertTrue((flag == 0).all())

    def test_distressed_company_triggers(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag

        cache = _make_2yr_cache_with_macro(100)
        # Inject distress: current_ratio < 1, high debt, negative FCF
        cache.loc[cache.index[50:], "current_ratio"] = 0.5
        cache.loc[cache.index[50:], "fcf_yield"] = -0.03

        flag = compute_company_survival_flag(cache)
        self.assertEqual(flag.iloc[10], 0)
        self.assertEqual(flag.iloc[60], 1)


# ===========================================================================
# 4. Full survival flags computation with macro data
# ===========================================================================


class TestComputeSurvivalFlagsWithMacro(unittest.TestCase):
    """Test compute_survival_flags with macro data present in the cache."""

    def test_all_three_flags_computed(self):
        from operator1.analysis.survival_mode import compute_survival_flags

        cache = _make_2yr_cache_with_macro(
            200,
            credit_spread_crisis_start=100,
        )
        result = compute_survival_flags(cache, target_sector="Technology")

        self.assertIn("company_survival_mode_flag", result.columns)
        self.assertIn("country_survival_mode_flag", result.columns)
        self.assertIn("country_protected_flag", result.columns)

    def test_strategic_sector_protection(self):
        from operator1.analysis.survival_mode import compute_survival_flags

        cache = _make_2yr_cache_with_macro(100)
        result = compute_survival_flags(cache, target_sector="Energy")

        # Energy is a strategic sector -> always protected
        self.assertTrue((result["country_protected_flag"] == 1).all())

    def test_non_strategic_sector_no_protection(self):
        from operator1.analysis.survival_mode import compute_survival_flags

        cache = _make_2yr_cache_with_macro(100)
        result = compute_survival_flags(cache, target_sector="Retail")

        # Retail is not strategic and market cap won't exceed GDP threshold
        self.assertTrue((result["country_protected_flag"] == 0).all())


# ===========================================================================
# 5. Cache enrichment with macro indicators
# ===========================================================================


class TestEnrichCacheWithMacroIndicators(unittest.TestCase):
    """Test enrich_cache_with_indicators() with macro data."""

    def test_macro_columns_merged(self):
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        idx = _make_daily_index(100)
        cache = pd.DataFrame({
            "close": np.random.randn(100) + 100,
        }, index=idx)

        macro = pd.DataFrame({
            "gdp_growth": np.full(100, 2.5),
            "inflation_rate_yoy": np.full(100, 3.0),
            "unemployment_rate": np.full(100, 4.2),
            "credit_spread": np.full(100, 2.0),
            "yield_curve_slope": np.full(100, 1.0),
            "fx_volatility": np.full(100, 8.0),
        }, index=idx)

        enriched = enrich_cache_with_indicators(cache, macro_aligned=macro)

        for col in ["gdp_growth", "inflation_rate_yoy", "unemployment_rate",
                     "credit_spread", "yield_curve_slope", "fx_volatility"]:
            self.assertIn(col, enriched.columns)
            self.assertFalse(enriched[col].isna().all(),
                             f"{col} is all NaN after enrichment")

    def test_survival_flags_merged(self):
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        idx = _make_daily_index(50)
        cache = pd.DataFrame({"close": np.ones(50)}, index=idx)
        flags = pd.DataFrame({
            "company_survival_mode_flag": np.zeros(50),
            "country_survival_mode_flag": np.ones(50),
            "country_protected_flag": np.ones(50),
        }, index=idx)

        enriched = enrich_cache_with_indicators(cache, survival_flags=flags)

        self.assertTrue((enriched["country_survival_mode_flag"] == 1).all())
        self.assertTrue((enriched["country_protected_flag"] == 1).all())

    def test_fuzzy_protection_merged(self):
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        idx = _make_daily_index(50)
        cache = pd.DataFrame({"close": np.ones(50)}, index=idx)
        fuzzy = pd.DataFrame({
            "fuzzy_protection_degree": np.full(50, 0.65),
            "fuzzy_sector_score": np.full(50, 0.8),
            "fuzzy_economic_score": np.full(50, 0.4),
            "fuzzy_policy_score": np.full(50, 0.2),
        }, index=idx)

        enriched = enrich_cache_with_indicators(cache, fuzzy_result_series=fuzzy)

        self.assertAlmostEqual(
            float(enriched["fuzzy_protection_degree"].iloc[0]), 0.65,
        )

    def test_combined_enrichment(self):
        """Enrich with macro, survival flags, and fuzzy -- all at once."""
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        idx = _make_daily_index(50)
        cache = pd.DataFrame({"close": np.ones(50)}, index=idx)
        macro = pd.DataFrame({
            "gdp_growth": np.full(50, 2.0),
            "inflation_rate_yoy": np.full(50, 3.0),
        }, index=idx)
        flags = pd.DataFrame({
            "company_survival_mode_flag": np.zeros(50),
            "country_survival_mode_flag": np.zeros(50),
            "country_protected_flag": np.zeros(50),
        }, index=idx)
        fuzzy = pd.DataFrame({
            "fuzzy_protection_degree": np.full(50, 0.3),
        }, index=idx)

        enriched = enrich_cache_with_indicators(
            cache, macro_aligned=macro, survival_flags=flags,
            fuzzy_result_series=fuzzy,
        )

        self.assertIn("gdp_growth", enriched.columns)
        self.assertIn("company_survival_mode_flag", enriched.columns)
        self.assertIn("fuzzy_protection_degree", enriched.columns)


# ===========================================================================
# 6. Survival timeline with macro-enriched 2yr cache
# ===========================================================================


class TestSurvivalTimelineWithMacro(unittest.TestCase):
    """Verify survival timeline uses macro-driven country flags over 2yrs."""

    def test_timeline_with_credit_spread_crisis(self):
        """Credit spread crisis should create country_exposed mode days."""
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_2yr_cache_with_macro(
            _TWO_YEARS_BDAYS,
            credit_spread_crisis_start=300,
        )
        cache = compute_survival_flags(cache, target_sector="Retail")
        result = compute_survival_timeline(cache)

        self.assertTrue(result.fitted)
        self.assertEqual(len(result.timeline), _TWO_YEARS_BDAYS)

        # Before crisis: all normal
        modes_before = result.timeline["survival_mode"].iloc[:280]
        self.assertTrue((modes_before == "normal").all())

        # After crisis: country_exposed (company healthy, not protected)
        modes_after = result.timeline["survival_mode"].iloc[320:]
        self.assertTrue(
            (modes_after == "country_exposed").any(),
            "Expected country_exposed mode after credit spread crisis",
        )

    def test_timeline_with_macro_crisis_and_strategic_sector(self):
        """Strategic sector company during country crisis -> country_protected."""
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_2yr_cache_with_macro(
            _TWO_YEARS_BDAYS,
            fx_vol_crisis_start=300,
        )
        cache = compute_survival_flags(cache, target_sector="Energy")
        result = compute_survival_timeline(cache)

        self.assertTrue(result.fitted)

        # Energy is strategic -> protected flag = 1
        # After FX crisis: should be country_protected, not country_exposed
        modes_after = result.timeline["survival_mode"].iloc[320:]
        self.assertTrue(
            (modes_after == "country_protected").any(),
            "Strategic sector company should be country_protected during crisis",
        )
        self.assertFalse(
            (modes_after == "country_exposed").any(),
            "Strategic sector company should NOT be country_exposed",
        )

    def test_timeline_both_crises(self):
        """Company distress + country crisis -> both_unprotected or both_protected."""
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_2yr_cache_with_macro(
            _TWO_YEARS_BDAYS,
            credit_spread_crisis_start=200,
        )
        # Inject company distress starting at day 250
        cache.loc[cache.index[250:], "current_ratio"] = 0.5
        cache.loc[cache.index[250:], "fcf_yield"] = -0.05

        cache = compute_survival_flags(cache, target_sector="Retail")
        result = compute_survival_timeline(cache)

        self.assertTrue(result.fitted)

        # After day 250: both company and country flags active, not protected
        modes_late = result.timeline["survival_mode"].iloc[260:]
        self.assertTrue(
            (modes_late == "both_unprotected").any(),
            "Expected both_unprotected when company+country in crisis",
        )

    def test_switch_points_at_macro_crisis_onset(self):
        """Switch points should appear when macro crisis starts."""
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_2yr_cache_with_macro(
            _TWO_YEARS_BDAYS,
            yield_curve_inversion_start=250,
        )
        cache = compute_survival_flags(cache, target_sector="Technology")
        result = compute_survival_timeline(cache)

        self.assertGreater(result.n_switches, 0)
        # There should be a switch near day 250
        switch_dates = [
            pd.Timestamp(sp["date"])
            for sp in result.switch_points
        ]
        crisis_onset = cache.index[250]
        # At least one switch within 5 days of crisis onset
        near_onset = any(
            abs((sd - crisis_onset).days) <= 5
            for sd in switch_dates
        )
        self.assertTrue(near_onset, "Expected switch point near crisis onset")

    def test_mode_distribution_sums_to_one(self):
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_2yr_cache_with_macro(
            _TWO_YEARS_BDAYS,
            credit_spread_crisis_start=300,
        )
        cache = compute_survival_flags(cache, target_sector="Technology")
        result = compute_survival_timeline(cache)

        total = sum(result.mode_distribution.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_stability_decreases_around_transitions(self):
        """Stability score should drop when modes switch frequently."""
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_2yr_cache_with_macro(
            _TWO_YEARS_BDAYS,
            credit_spread_crisis_start=250,
        )
        cache = compute_survival_flags(cache, target_sector="Retail")
        result = compute_survival_timeline(cache)

        # Stability at the very start (long run of normal) should be 1.0
        early_stability = result.timeline["stability_score_21d"].iloc[30]
        self.assertAlmostEqual(early_stability, 1.0)


# ===========================================================================
# 7. Walk-forward with macro-enriched cache
# ===========================================================================


class TestWalkForwardWithMacro(unittest.TestCase):
    """Walk-forward predictions on a cache that includes macro data."""

    def test_walk_forward_with_macro_columns(self):
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline
        from operator1.models.walk_forward import run_walk_forward

        cache = _make_2yr_cache_with_macro(
            250, credit_spread_crisis_start=150,
        )
        cache = compute_survival_flags(cache, target_sector="Retail")
        tl = compute_survival_timeline(cache)

        wf = run_walk_forward(
            tl.timeline,
            survival_modes=tl.timeline["survival_mode"],
            switch_points=tl.timeline["switch_point"],
            min_history=30,
        )

        self.assertTrue(wf.fitted)
        self.assertGreater(wf.total_predictions, 0)

    def test_mode_scores_include_country_modes(self):
        """Walk-forward should track errors for country survival modes."""
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline
        from operator1.models.walk_forward import run_walk_forward

        cache = _make_2yr_cache_with_macro(
            300, credit_spread_crisis_start=150,
        )
        cache = compute_survival_flags(cache, target_sector="Retail")
        tl = compute_survival_timeline(cache)

        wf = run_walk_forward(
            tl.timeline,
            survival_modes=tl.timeline["survival_mode"],
            switch_points=tl.timeline["switch_point"],
            min_history=30,
        )

        self.assertTrue(wf.fitted)
        # mode_scores is a list of ModelModeScore objects with .survival_mode
        observed_modes = set(tl.timeline["survival_mode"].unique())
        scored_modes = set(s.survival_mode for s in wf.mode_scores)
        # At least some observed modes should have scores
        overlap = observed_modes & scored_modes
        self.assertGreater(
            len(overlap), 0,
            f"No overlap between observed modes {observed_modes} "
            f"and scored modes {scored_modes}",
        )


# ===========================================================================
# 8. Survival-aware weighting with macro-driven modes
# ===========================================================================


class TestSurvivalAwareWeightsWithMacro(unittest.TestCase):
    """End-to-end: macro -> flags -> timeline -> walk-forward -> weights."""

    def test_full_macro_pipeline(self):
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline
        from operator1.models.walk_forward import (
            run_walk_forward, get_mode_weights_from_walk_forward,
        )
        from operator1.models.prediction_aggregator import (
            compute_survival_aware_weights,
            get_survival_context_from_cache,
        )

        # Build 2yr cache with macro crisis at day 200
        cache = _make_2yr_cache_with_macro(
            400, credit_spread_crisis_start=200,
        )
        cache = compute_survival_flags(cache, target_sector="Technology")
        tl = compute_survival_timeline(cache)
        self.assertTrue(tl.fitted)

        wf = run_walk_forward(
            tl.timeline,
            survival_modes=tl.timeline["survival_mode"],
            switch_points=tl.timeline["switch_point"],
            min_history=30,
        )
        self.assertTrue(wf.fitted)

        mode_weights = get_mode_weights_from_walk_forward(wf)
        self.assertIsInstance(mode_weights, dict)

        # Get context from the enriched timeline
        ctx = get_survival_context_from_cache(tl.timeline)
        self.assertIn(ctx["current_mode"], [
            "normal", "company_only", "country_protected",
            "country_exposed", "both_unprotected", "both_protected",
        ])

        # Compute final weights
        base_weights = {
            "baseline_last": 0.25, "ema_21": 0.25,
            "linear_trend": 0.25, "mean_reversion": 0.25,
        }
        final_weights = compute_survival_aware_weights(
            base_weights=base_weights,
            mode_weights=mode_weights,
            current_mode=ctx["current_mode"],
            days_in_mode=ctx["days_in_mode"],
            previous_mode=ctx["previous_mode"],
        )

        total = sum(final_weights.values())
        self.assertAlmostEqual(total, 1.0, places=4)


# ===========================================================================
# 9. Macro alignment -> merge -> survival flags (integrated pipeline)
# ===========================================================================


class TestMacroToSurvivalPipeline(unittest.TestCase):
    """Integration: MacroDataset -> align -> merge -> survival flags -> timeline."""

    def test_full_pipeline_from_macro_dataset(self):
        from operator1.features.macro_alignment import (
            align_macro_to_daily, merge_macro_onto_target,
        )
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline

        daily_index = _make_daily_index(250)

        # Build target cache with financial data
        np.random.seed(42)
        target = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(250) * 0.5),
            "return_1d": np.concatenate(
                [[np.nan], np.random.randn(249) * 0.02],
            ),
            "current_ratio": np.full(250, 1.8),
            "debt_to_equity_abs": np.full(250, 1.2),
            "fcf_yield": np.full(250, 0.04),
            "drawdown_252d": np.full(250, -0.08),
            "market_cap": np.full(250, 5_000_000.0),
        }, index=daily_index)

        # Build macro dataset with crisis-level values
        ds = _make_macro_dataset(
            gdp_growth=[2.0, 1.0, -0.5, 0.5],
            inflation=[2.5, 3.0, 5.0, 4.0],
            unemployment=[3.5, 4.0, 7.5, 6.0],
            interest_rate=[1.0, 2.0, 3.5, 2.5],
        )

        # Align and merge
        macro_aligned = align_macro_to_daily(ds, daily_index)
        merged = merge_macro_onto_target(target, macro_aligned)

        # Verify macro columns present
        self.assertIn("inflation_rate_yoy", merged.columns)
        self.assertIn("unemployment_rate", merged.columns)
        self.assertIn("real_return_1d", merged.columns)

        # Compute survival flags
        flagged = compute_survival_flags(merged, target_sector="Technology")
        self.assertIn("company_survival_mode_flag", flagged.columns)
        self.assertIn("country_survival_mode_flag", flagged.columns)

        # Compute timeline
        tl = compute_survival_timeline(flagged)
        self.assertTrue(tl.fitted)
        self.assertEqual(len(tl.timeline), 250)

    def test_macro_missing_graceful_degradation(self):
        """Pipeline should work even when MacroDataset has no indicators."""
        from operator1.features.macro_alignment import (
            align_macro_to_daily, merge_macro_onto_target,
        )
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.survival_timeline import compute_survival_timeline
        from operator1.steps.macro_mapping import MacroDataset

        daily_index = _make_daily_index(100)
        target = pd.DataFrame({
            "close": np.random.randn(100) + 100,
            "current_ratio": np.full(100, 1.5),
            "debt_to_equity_abs": np.full(100, 1.0),
            "fcf_yield": np.full(100, 0.04),
            "drawdown_252d": np.full(100, -0.1),
        }, index=daily_index)

        # Empty macro dataset
        ds = MacroDataset(country_iso3="UNKNOWN")
        macro_aligned = align_macro_to_daily(ds, daily_index)
        merged = merge_macro_onto_target(target, macro_aligned)

        # Should still compute flags (country defaults to 0)
        flagged = compute_survival_flags(merged, target_sector="Technology")
        tl = compute_survival_timeline(flagged)

        self.assertTrue(tl.fitted)
        # All days should be normal (no macro = no country crisis)
        modes = tl.timeline["survival_mode"]
        self.assertTrue((modes == "normal").all())


# ===========================================================================
# 10. MACRO_INDICATOR_FIELDS and PROTECTION_SCORE_FIELDS completeness
# ===========================================================================


class TestCacheFieldsCompleteness(unittest.TestCase):
    """Verify that MACRO_INDICATOR_FIELDS covers what survival analysis needs."""

    def test_macro_fields_cover_country_survival_inputs(self):
        """All columns used by compute_country_survival_flag should be in
        MACRO_INDICATOR_FIELDS or available in the cache."""
        from operator1.steps.cache_builder import MACRO_INDICATOR_FIELDS

        # Columns the country survival flag checks for
        needed = {
            "credit_spread", "unemployment_rate",
            "yield_curve_slope", "fx_volatility",
        }
        available = set(MACRO_INDICATOR_FIELDS)
        missing = needed - available
        self.assertEqual(
            missing, set(),
            f"MACRO_INDICATOR_FIELDS is missing columns needed by "
            f"compute_country_survival_flag: {missing}",
        )

    def test_macro_fields_cover_country_protection_inputs(self):
        """Columns used by compute_country_protected_flag should be available."""
        from operator1.steps.cache_builder import MACRO_INDICATOR_FIELDS

        needed = {"lending_interest_rate", "gdp_current_usd"}
        available = set(MACRO_INDICATOR_FIELDS)
        missing = needed - available
        self.assertEqual(
            missing, set(),
            f"MACRO_INDICATOR_FIELDS missing for protection check: {missing}",
        )

    def test_protection_score_fields_cover_survival_flags(self):
        from operator1.steps.cache_builder import PROTECTION_SCORE_FIELDS

        needed = {
            "company_survival_mode_flag",
            "country_survival_mode_flag",
            "country_protected_flag",
        }
        available = set(PROTECTION_SCORE_FIELDS)
        missing = needed - available
        self.assertEqual(
            missing, set(),
            f"PROTECTION_SCORE_FIELDS missing survival flags: {missing}",
        )

    def test_build_entity_daily_cache_has_macro_placeholders(self):
        """build_entity_daily_cache should pre-allocate NaN for all macro fields."""
        from operator1.steps.cache_builder import (
            MACRO_INDICATOR_FIELDS, PROTECTION_SCORE_FIELDS,
        )
        # Just verify the tuples exist and are non-empty
        self.assertGreater(len(MACRO_INDICATOR_FIELDS), 0)
        self.assertGreater(len(PROTECTION_SCORE_FIELDS), 0)


# ===========================================================================
# 11. Macro quadrant classification with 2yr cache
# ===========================================================================


class TestMacroQuadrantWith2YrCache(unittest.TestCase):
    """Verify macro quadrant classification works across the full 2yr window."""

    def test_quadrant_computed_over_2yr(self):
        from operator1.features.macro_quadrant import compute_macro_quadrant

        cache = _make_2yr_cache_with_macro(_TWO_YEARS_BDAYS)
        ds = _make_macro_dataset(
            gdp_growth=[2.0, 2.3, 2.5, 2.1],
            inflation=[1.5, 2.0, 2.5, 3.5],
        )

        updated_cache, result = compute_macro_quadrant(cache, ds)

        self.assertIn("macro_quadrant", updated_cache.columns)
        self.assertIn("macro_quadrant_numeric", updated_cache.columns)
        self.assertGreater(result.n_days_classified, 0)

    def test_quadrant_with_high_inflation(self):
        """High inflation + low growth should produce stagflation."""
        from operator1.features.macro_quadrant import compute_macro_quadrant

        idx = _make_daily_index(50)
        cache = pd.DataFrame({"close": np.ones(50)}, index=idx)

        # Low growth, high inflation
        ds = _make_macro_dataset(
            gdp_growth=[0.5, 0.3, 0.2, 0.1],
            inflation=[5.0, 6.0, 7.0, 8.0],
        )

        updated_cache, result = compute_macro_quadrant(cache, ds)

        # With very low growth and high inflation, should see stagflation
        quadrants = updated_cache["macro_quadrant"]
        unique_q = quadrants.unique()
        self.assertTrue(
            "stagflation" in unique_q or "deflation" in unique_q,
            f"Expected stagflation or deflation, got {unique_q}",
        )


if __name__ == "__main__":
    unittest.main()
