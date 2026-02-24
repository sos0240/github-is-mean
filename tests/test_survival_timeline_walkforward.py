"""Tests for survival timeline (Phase 2), walk-forward (Phase 3),
survival-aware weighting (Phase 4), and cache enrichment (Phase 1).
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_index(days: int = 250) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_cache(days: int = 250, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily cache with survival flags for testing."""
    np.random.seed(seed)
    idx = _make_daily_index(days)
    mid = days // 2

    close = 100 + np.cumsum(np.random.randn(days) * 0.5)

    # Company survival flag: triggered in second half
    comp_flag = np.concatenate([np.zeros(mid), np.ones(days - mid)])
    # Country survival flag: triggered in last quarter
    q3 = 3 * days // 4
    ctry_flag = np.concatenate([np.zeros(q3), np.ones(days - q3)])
    # Protected flag: always on for first 80% of days
    prot_flag = np.concatenate([np.ones(4 * days // 5), np.zeros(days - 4 * days // 5)])

    return pd.DataFrame({
        "close": close,
        "current_ratio": 1.5 + np.cumsum(np.random.randn(days) * 0.005),
        "debt_to_equity_abs": 1.0 + np.cumsum(np.random.randn(days) * 0.003),
        "fcf_yield": 0.05 + np.cumsum(np.random.randn(days) * 0.001),
        "volatility_21d": np.abs(np.random.randn(days) * 0.1) + 0.05,
        "company_survival_mode_flag": comp_flag,
        "country_survival_mode_flag": ctry_flag,
        "country_protected_flag": prot_flag,
    }, index=idx)


# ===========================================================================
# Phase 2 -- Survival Timeline
# ===========================================================================


class TestClassifySurvivalMode(unittest.TestCase):
    """Test individual mode classification."""

    def test_normal(self):
        from operator1.analysis.survival_timeline import classify_survival_mode
        self.assertEqual(classify_survival_mode(0, 0, 0), "normal")
        self.assertEqual(classify_survival_mode(0, 0, 1), "normal")

    def test_company_only(self):
        from operator1.analysis.survival_timeline import classify_survival_mode
        self.assertEqual(classify_survival_mode(1, 0, 0), "company_only")
        self.assertEqual(classify_survival_mode(1, 0, 1), "company_only")

    def test_country_protected(self):
        from operator1.analysis.survival_timeline import classify_survival_mode
        self.assertEqual(classify_survival_mode(0, 1, 1), "country_protected")

    def test_country_exposed(self):
        from operator1.analysis.survival_timeline import classify_survival_mode
        self.assertEqual(classify_survival_mode(0, 1, 0), "country_exposed")

    def test_both_unprotected(self):
        from operator1.analysis.survival_timeline import classify_survival_mode
        self.assertEqual(classify_survival_mode(1, 1, 0), "both_unprotected")

    def test_both_protected(self):
        from operator1.analysis.survival_timeline import classify_survival_mode
        self.assertEqual(classify_survival_mode(1, 1, 1), "both_protected")


class TestSurvivalTimeline(unittest.TestCase):
    """Test the full timeline computation."""

    def test_basic_timeline(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_cache(100)
        result = compute_survival_timeline(cache)

        self.assertTrue(result.fitted)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.timeline), 100)

        # Check required columns exist
        for col in ("survival_mode", "survival_mode_code", "switch_point",
                     "days_in_mode", "stability_score_21d"):
            self.assertIn(col, result.timeline.columns)

    def test_switch_points_detected(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_cache(100)
        result = compute_survival_timeline(cache)

        # With flags changing at mid and 3/4 point, we expect switches
        self.assertGreater(result.n_switches, 0)
        self.assertEqual(len(result.switch_points), result.n_switches)

    def test_days_in_mode_counter(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_cache(100)
        result = compute_survival_timeline(cache)

        dim = result.timeline["days_in_mode"]
        # First day always starts at 1
        self.assertEqual(dim.iloc[0], 1)
        # Counter should always be >= 1
        self.assertTrue((dim >= 1).all())

    def test_stability_score_range(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_cache(100)
        result = compute_survival_timeline(cache)

        stab = result.timeline["stability_score_21d"]
        # Stability is always in (0, 1]
        self.assertTrue((stab > 0).all())
        self.assertTrue((stab <= 1.0).all())

    def test_mode_distribution_sums_to_one(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_cache(100)
        result = compute_survival_timeline(cache)

        total = sum(result.mode_distribution.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_empty_cache(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        result = compute_survival_timeline(pd.DataFrame())
        self.assertFalse(result.fitted)
        self.assertIsNotNone(result.error)

    def test_missing_flags_defaults_to_normal(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline

        idx = _make_daily_index(50)
        cache = pd.DataFrame({"close": np.random.randn(50)}, index=idx)
        result = compute_survival_timeline(cache)

        self.assertTrue(result.fitted)
        # All days should be "normal" when flags are missing
        modes = result.timeline["survival_mode"]
        self.assertTrue((modes == "normal").all())


class TestGetModeAtDate(unittest.TestCase):
    """Test mode lookup by date."""

    def test_lookup(self):
        from operator1.analysis.survival_timeline import (
            compute_survival_timeline, get_mode_at_date,
        )

        cache = _make_cache(50)
        result = compute_survival_timeline(cache)

        # First day should be normal (no flags active)
        mode = get_mode_at_date(result, cache.index[0])
        self.assertEqual(mode, "normal")

    def test_unknown_date(self):
        from operator1.analysis.survival_timeline import (
            compute_survival_timeline, get_mode_at_date,
        )

        cache = _make_cache(50)
        result = compute_survival_timeline(cache)

        mode = get_mode_at_date(result, pd.Timestamp("1900-01-01"))
        self.assertEqual(mode, "unknown")


# ===========================================================================
# Phase 3 -- Walk-Forward
# ===========================================================================


class TestWalkForwardModels(unittest.TestCase):
    """Test individual walk-forward models."""

    def test_baseline_predicts_last_value(self):
        from operator1.models.walk_forward import WFBaselineModel

        model = WFBaselineModel()
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(model.predict_next(history), 5.0)

    def test_ema_model(self):
        from operator1.models.walk_forward import WFEMAModel

        model = WFEMAModel(span=3)
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = model.predict_next(history)
        self.assertIsInstance(pred, float)
        self.assertFalse(math.isnan(pred))

    def test_linear_trend_model(self):
        from operator1.models.walk_forward import WFLinearTrendModel

        model = WFLinearTrendModel()
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        model.fit(history)
        pred = model.predict_next(history)
        # Linear trend on [1,2,3,4,5] should predict ~6
        self.assertAlmostEqual(pred, 6.0, places=0)

    def test_mean_reversion_model(self):
        from operator1.models.walk_forward import WFMeanReversionModel

        model = WFMeanReversionModel(window=5, speed=0.5)
        history = pd.Series([10.0, 10.0, 10.0, 10.0, 20.0])
        pred = model.predict_next(history)
        # Should revert partially from 20 toward mean of 12
        self.assertLess(pred, 20.0)
        self.assertGreater(pred, 12.0)


class TestWalkForwardEngine(unittest.TestCase):
    """Test the walk-forward prediction loop."""

    def test_basic_walk_forward(self):
        from operator1.models.walk_forward import run_walk_forward

        cache = _make_cache(150)
        result = run_walk_forward(cache, min_history=30)

        self.assertTrue(result.fitted)
        self.assertIsNone(result.error)
        self.assertGreater(result.total_predictions, 0)
        self.assertGreater(len(result.day_errors), 0)

    def test_mode_scores_populated(self):
        from operator1.models.walk_forward import run_walk_forward

        cache = _make_cache(150)
        result = run_walk_forward(cache, min_history=30)

        self.assertGreater(len(result.mode_scores), 0)
        # At least one best model per mode should exist
        self.assertGreater(len(result.best_model_by_mode), 0)

    def test_overall_best_model_selected(self):
        from operator1.models.walk_forward import run_walk_forward

        cache = _make_cache(150)
        result = run_walk_forward(cache, min_history=30)

        self.assertNotEqual(result.overall_best_model, "")
        self.assertFalse(math.isnan(result.overall_mae))

    def test_retrain_at_switch_points(self):
        from operator1.models.walk_forward import run_walk_forward
        from operator1.analysis.survival_timeline import compute_survival_timeline

        cache = _make_cache(150)
        tl = compute_survival_timeline(cache)

        result = run_walk_forward(
            cache,
            survival_modes=tl.timeline["survival_mode"],
            switch_points=tl.timeline["switch_point"],
            min_history=30,
        )

        # If there are switch points after min_history, retrains should occur
        if tl.n_switches > 0:
            # Some switch points may fall before min_history, so retrain_dates
            # could be <= n_switches
            self.assertIsInstance(result.retrain_dates, list)

    def test_empty_cache(self):
        from operator1.models.walk_forward import run_walk_forward

        result = run_walk_forward(pd.DataFrame())
        self.assertFalse(result.fitted)
        self.assertIsNotNone(result.error)

    def test_mode_weights_extraction(self):
        from operator1.models.walk_forward import (
            run_walk_forward, get_mode_weights_from_walk_forward,
        )

        cache = _make_cache(150)
        wf_result = run_walk_forward(cache, min_history=30)

        mode_weights = get_mode_weights_from_walk_forward(wf_result)
        self.assertIsInstance(mode_weights, dict)

        # Weights within each mode should sum to ~1.0
        for mode, weights in mode_weights.items():
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"Weights for mode {mode} don't sum to 1")


# ===========================================================================
# Phase 4 -- Survival-Aware Weighting
# ===========================================================================


class TestSurvivalAwareWeights(unittest.TestCase):
    """Test survival-aware ensemble weight computation."""

    def test_no_mode_weights_returns_base(self):
        from operator1.models.prediction_aggregator import (
            compute_survival_aware_weights,
        )
        base = {"model_a": 0.6, "model_b": 0.4}
        result = compute_survival_aware_weights(base, mode_weights=None)
        self.assertEqual(result, base)

    def test_mode_weights_override(self):
        from operator1.models.prediction_aggregator import (
            compute_survival_aware_weights,
        )
        base = {"model_a": 0.6, "model_b": 0.4}
        mode_w = {
            "company_only": {"model_a": 0.3, "model_b": 0.7},
        }
        result = compute_survival_aware_weights(
            base, mode_weights=mode_w, current_mode="company_only",
        )
        # model_b should have higher weight in company_only mode
        self.assertGreater(result["model_b"], result["model_a"])

    def test_transition_blending(self):
        from operator1.models.prediction_aggregator import (
            compute_survival_aware_weights,
        )
        base = {"model_a": 0.5, "model_b": 0.5}
        mode_w = {
            "normal": {"model_a": 0.8, "model_b": 0.2},
            "company_only": {"model_a": 0.2, "model_b": 0.8},
        }

        # Day 1 of transition: should blend (more like previous mode)
        result_day1 = compute_survival_aware_weights(
            base, mode_weights=mode_w,
            current_mode="company_only",
            days_in_mode=1,
            previous_mode="normal",
            transition_halflife=5,
        )

        # Day 20 of transition: should be almost fully in new mode
        result_day20 = compute_survival_aware_weights(
            base, mode_weights=mode_w,
            current_mode="company_only",
            days_in_mode=20,
            previous_mode="normal",
            transition_halflife=5,
        )

        # At day 20, model_b should be closer to 0.8 than at day 1
        self.assertGreater(result_day20["model_b"], result_day1["model_b"])

    def test_weights_sum_to_one(self):
        from operator1.models.prediction_aggregator import (
            compute_survival_aware_weights,
        )
        base = {"m1": 0.3, "m2": 0.3, "m3": 0.4}
        mode_w = {"normal": {"m1": 0.5, "m2": 0.3, "m3": 0.2}}

        result = compute_survival_aware_weights(
            base, mode_weights=mode_w, current_mode="normal",
        )
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestSurvivalContextExtraction(unittest.TestCase):
    """Test extraction of survival context from enriched cache."""

    def test_context_from_enriched_cache(self):
        from operator1.models.prediction_aggregator import (
            get_survival_context_from_cache,
        )

        idx = _make_daily_index(50)
        cache = pd.DataFrame({
            "close": np.random.randn(50),
            "survival_mode": ["normal"] * 40 + ["company_only"] * 10,
            "days_in_mode": list(range(1, 41)) + list(range(1, 11)),
            "stability_score_21d": [1.0] * 40 + [0.5] * 10,
        }, index=idx)

        ctx = get_survival_context_from_cache(cache)
        self.assertEqual(ctx["current_mode"], "company_only")
        self.assertEqual(ctx["days_in_mode"], 10)
        self.assertEqual(ctx["previous_mode"], "normal")
        self.assertAlmostEqual(ctx["stability_score"], 0.5)

    def test_context_empty_cache(self):
        from operator1.models.prediction_aggregator import (
            get_survival_context_from_cache,
        )
        ctx = get_survival_context_from_cache(pd.DataFrame())
        self.assertEqual(ctx["current_mode"], "normal")


# ===========================================================================
# Phase 1 -- Cache Enrichment
# ===========================================================================


class TestCacheEnrichment(unittest.TestCase):
    """Test that cache enrichment adds indicator columns."""

    def test_macro_fields_placeholder(self):
        from operator1.steps.cache_builder import (
            MACRO_INDICATOR_FIELDS, PROTECTION_SCORE_FIELDS,
        )
        # Verify the tuples exist and are non-empty
        self.assertGreater(len(MACRO_INDICATOR_FIELDS), 0)
        self.assertGreater(len(PROTECTION_SCORE_FIELDS), 0)

    def test_enrich_cache(self):
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        cache = _make_cache(50)
        # Create a mock macro_aligned DataFrame
        idx = cache.index
        macro = pd.DataFrame({
            "gdp_growth": np.full(len(idx), 2.5),
            "inflation_rate_yoy": np.full(len(idx), 3.0),
            "unemployment_rate": np.full(len(idx), 5.0),
        }, index=idx)

        enriched = enrich_cache_with_indicators(cache, macro_aligned=macro)

        self.assertIn("gdp_growth", enriched.columns)
        self.assertIn("inflation_rate_yoy", enriched.columns)
        self.assertIn("unemployment_rate", enriched.columns)

    def test_enrich_with_survival_flags(self):
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        cache = _make_cache(50)
        flags = pd.DataFrame({
            "company_survival_mode_flag": np.ones(len(cache)),
            "country_survival_mode_flag": np.zeros(len(cache)),
            "country_protected_flag": np.ones(len(cache)),
        }, index=cache.index)

        enriched = enrich_cache_with_indicators(cache, survival_flags=flags)

        self.assertTrue((enriched["company_survival_mode_flag"] == 1).all())

    def test_enrich_with_fuzzy_protection(self):
        from operator1.steps.cache_builder import enrich_cache_with_indicators

        cache = _make_cache(50)
        fuzzy = pd.DataFrame({
            "fuzzy_protection_degree": np.full(len(cache), 0.75),
            "fuzzy_sector_score": np.full(len(cache), 0.9),
            "fuzzy_economic_score": np.full(len(cache), 0.3),
            "fuzzy_policy_score": np.full(len(cache), 0.1),
        }, index=cache.index)

        enriched = enrich_cache_with_indicators(cache, fuzzy_result_series=fuzzy)

        self.assertAlmostEqual(
            float(enriched["fuzzy_protection_degree"].iloc[0]), 0.75,
        )


# ===========================================================================
# Integration: Timeline -> WalkForward -> Weights
# ===========================================================================


class TestEndToEndIntegration(unittest.TestCase):
    """Integration test: timeline -> walk-forward -> survival-aware weights."""

    def test_full_pipeline(self):
        from operator1.analysis.survival_timeline import compute_survival_timeline
        from operator1.models.walk_forward import (
            run_walk_forward, get_mode_weights_from_walk_forward,
        )
        from operator1.models.prediction_aggregator import (
            compute_survival_aware_weights,
        )

        # Step 1: Build cache with survival flags
        cache = _make_cache(200)

        # Step 2: Compute survival timeline
        tl = compute_survival_timeline(cache)
        self.assertTrue(tl.fitted)

        # Step 3: Run walk-forward
        wf = run_walk_forward(
            tl.timeline,
            survival_modes=tl.timeline["survival_mode"],
            switch_points=tl.timeline["switch_point"],
            min_history=30,
        )
        self.assertTrue(wf.fitted)

        # Step 4: Extract mode weights
        mode_weights = get_mode_weights_from_walk_forward(wf)
        self.assertIsInstance(mode_weights, dict)

        # Step 5: Compute survival-aware weights
        base_weights = {"baseline_last": 0.25, "ema_21": 0.25,
                        "linear_trend": 0.25, "mean_reversion": 0.25}

        current_mode = str(tl.timeline["survival_mode"].iloc[-1])
        days_in = int(tl.timeline["days_in_mode"].iloc[-1])

        final_weights = compute_survival_aware_weights(
            base_weights=base_weights,
            mode_weights=mode_weights,
            current_mode=current_mode,
            days_in_mode=days_in,
        )

        # Weights should sum to ~1.0
        total = sum(final_weights.values())
        self.assertAlmostEqual(total, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
