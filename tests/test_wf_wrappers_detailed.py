"""Detailed individual tests for each walk-forward model wrapper.

Tests cover edge cases, numerical correctness, and fit/predict behaviour.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd


# ===========================================================================
# WFBaselineModel
# ===========================================================================


class TestWFBaselineModel(unittest.TestCase):
    """Thorough tests for the last-value carry-forward baseline."""

    def _make_model(self):
        from operator1.models.walk_forward import WFBaselineModel
        return WFBaselineModel()

    def test_predict_returns_last_value(self):
        model = self._make_model()
        history = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(model.predict_next(history), 50.0)

    def test_predict_single_value(self):
        model = self._make_model()
        history = pd.Series([42.0])
        self.assertEqual(model.predict_next(history), 42.0)

    def test_predict_with_trailing_nans(self):
        """NaNs at the end should be skipped -- last clean value returned."""
        model = self._make_model()
        history = pd.Series([1.0, 2.0, 3.0, float("nan"), float("nan")])
        self.assertEqual(model.predict_next(history), 3.0)

    def test_predict_all_nans(self):
        model = self._make_model()
        history = pd.Series([float("nan"), float("nan")])
        self.assertTrue(math.isnan(model.predict_next(history)))

    def test_predict_empty_series(self):
        model = self._make_model()
        history = pd.Series([], dtype=float)
        self.assertTrue(math.isnan(model.predict_next(history)))

    def test_predict_negative_values(self):
        model = self._make_model()
        history = pd.Series([-5.0, -3.0, -1.0])
        self.assertEqual(model.predict_next(history), -1.0)

    def test_predict_constant_series(self):
        model = self._make_model()
        history = pd.Series([7.7] * 100)
        self.assertEqual(model.predict_next(history), 7.7)

    def test_fit_is_noop(self):
        """Calling fit should not raise and should not change predictions."""
        model = self._make_model()
        history = pd.Series([1.0, 2.0, 3.0])
        model.fit(history)
        self.assertEqual(model.predict_next(history), 3.0)

    def test_name_attribute(self):
        model = self._make_model()
        self.assertEqual(model.name, "baseline_last")


# ===========================================================================
# WFEMAModel
# ===========================================================================


class TestWFEMAModel(unittest.TestCase):
    """Thorough tests for the exponential moving average model."""

    def _make_model(self, span=21):
        from operator1.models.walk_forward import WFEMAModel
        return WFEMAModel(span=span)

    def test_predict_trending_up(self):
        """EMA on a rising series should be below the last value."""
        model = self._make_model(span=3)
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        pred = model.predict_next(history)
        # EMA lags behind in a trending series
        self.assertLess(pred, 10.0)
        self.assertGreater(pred, 5.0)

    def test_predict_constant_series(self):
        """EMA on a constant series should equal that constant."""
        model = self._make_model(span=5)
        history = pd.Series([3.0] * 50)
        pred = model.predict_next(history)
        self.assertAlmostEqual(pred, 3.0, places=10)

    def test_predict_single_value(self):
        model = self._make_model(span=5)
        history = pd.Series([42.0])
        self.assertEqual(model.predict_next(history), 42.0)

    def test_predict_two_values(self):
        model = self._make_model(span=3)
        history = pd.Series([10.0, 20.0])
        pred = model.predict_next(history)
        self.assertIsInstance(pred, float)
        self.assertFalse(math.isnan(pred))
        # Should be between 10 and 20
        self.assertGreaterEqual(pred, 10.0)
        self.assertLessEqual(pred, 20.0)

    def test_predict_with_nans(self):
        """NaN values should be dropped before EMA computation."""
        model = self._make_model(span=3)
        history = pd.Series([1.0, float("nan"), 2.0, float("nan"), 3.0])
        pred = model.predict_next(history)
        self.assertFalse(math.isnan(pred))

    def test_predict_all_nans(self):
        model = self._make_model(span=3)
        history = pd.Series([float("nan")] * 5)
        self.assertTrue(math.isnan(model.predict_next(history)))

    def test_predict_empty(self):
        model = self._make_model(span=3)
        history = pd.Series([], dtype=float)
        self.assertTrue(math.isnan(model.predict_next(history)))

    def test_different_spans(self):
        """Shorter span should react faster to recent values."""
        history = pd.Series([1.0] * 50 + [100.0])

        short_model = self._make_model(span=3)
        long_model = self._make_model(span=50)

        short_pred = short_model.predict_next(history)
        long_pred = long_model.predict_next(history)

        # Short span should be closer to the spike
        self.assertGreater(short_pred, long_pred)

    def test_fit_is_noop(self):
        model = self._make_model(span=5)
        history = pd.Series([1.0, 2.0, 3.0])
        model.fit(history)  # should not raise

    def test_name_attribute(self):
        model = self._make_model()
        self.assertEqual(model.name, "ema_21")


# ===========================================================================
# WFLinearTrendModel
# ===========================================================================


class TestWFLinearTrendModel(unittest.TestCase):
    """Thorough tests for the linear trend extrapolation model."""

    def _make_model(self):
        from operator1.models.walk_forward import WFLinearTrendModel
        return WFLinearTrendModel()

    def test_perfect_linear_series(self):
        """On a perfect line y=x, should predict the next value exactly."""
        model = self._make_model()
        history = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        model.fit(history)
        pred = model.predict_next(history)
        # Next value should be 10.0
        self.assertAlmostEqual(pred, 10.0, places=1)

    def test_decreasing_series(self):
        """On a decreasing line, prediction should continue the decline."""
        model = self._make_model()
        history = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        model.fit(history)
        pred = model.predict_next(history)
        self.assertAlmostEqual(pred, 4.0, places=1)

    def test_constant_series(self):
        """Constant series should predict the same constant."""
        model = self._make_model()
        history = pd.Series([5.0] * 20)
        model.fit(history)
        pred = model.predict_next(history)
        self.assertAlmostEqual(pred, 5.0, places=1)

    def test_short_series_unfitted(self):
        """With fewer than 5 points, fit should fail gracefully."""
        model = self._make_model()
        history = pd.Series([1.0, 2.0])
        model.fit(history)
        # Should still return something (fallback to last value)
        pred = model.predict_next(history)
        self.assertIsInstance(pred, float)
        self.assertFalse(math.isnan(pred))

    def test_predict_without_explicit_fit(self):
        """predict_next should auto-fit if not already fitted."""
        model = self._make_model()
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pred = model.predict_next(history)
        self.assertIsInstance(pred, float)
        self.assertFalse(math.isnan(pred))

    def test_refit_changes_prediction(self):
        """Refitting with different data should change the prediction."""
        model = self._make_model()

        history1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        model.fit(history1)
        pred1 = model.predict_next(history1)

        history2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        model.fit(history2)
        pred2 = model.predict_next(history2)

        # Predictions should differ (one trends up, other trends down)
        self.assertNotAlmostEqual(pred1, pred2, places=1)

    def test_with_nans(self):
        """NaN values should be cleaned before fitting."""
        model = self._make_model()
        history = pd.Series([1.0, float("nan"), 3.0, float("nan"), 5.0, 6.0, 7.0])
        model.fit(history)
        pred = model.predict_next(history)
        self.assertIsInstance(pred, float)

    def test_negative_slope(self):
        model = self._make_model()
        history = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
        model.fit(history)
        pred = model.predict_next(history)
        self.assertLess(pred, 50.0)

    def test_name_attribute(self):
        model = self._make_model()
        self.assertEqual(model.name, "linear_trend")


# ===========================================================================
# WFMeanReversionModel
# ===========================================================================


class TestWFMeanReversionModel(unittest.TestCase):
    """Thorough tests for the mean reversion model."""

    def _make_model(self, window=63, speed=0.1):
        from operator1.models.walk_forward import WFMeanReversionModel
        return WFMeanReversionModel(window=window, speed=speed)

    def test_at_mean_stays_at_mean(self):
        """When last value equals the mean, prediction should stay."""
        model = self._make_model(window=5, speed=0.5)
        history = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
        pred = model.predict_next(history)
        self.assertAlmostEqual(pred, 10.0)

    def test_above_mean_reverts_down(self):
        """When last value is above the mean, prediction should pull down."""
        model = self._make_model(window=5, speed=0.5)
        history = pd.Series([10.0, 10.0, 10.0, 10.0, 20.0])
        pred = model.predict_next(history)
        # Mean is 12.0, last is 20.0, should pull toward 12
        self.assertLess(pred, 20.0)
        self.assertGreater(pred, 12.0)

    def test_below_mean_reverts_up(self):
        """When last value is below the mean, prediction should pull up."""
        model = self._make_model(window=5, speed=0.5)
        history = pd.Series([10.0, 10.0, 10.0, 10.0, 2.0])
        pred = model.predict_next(history)
        # Mean is ~6.4, last is 2.0, should pull toward mean
        self.assertGreater(pred, 2.0)

    def test_speed_zero_returns_last(self):
        """With speed=0, prediction should equal last value (no reversion)."""
        model = self._make_model(window=5, speed=0.0)
        history = pd.Series([10.0, 10.0, 10.0, 10.0, 50.0])
        pred = model.predict_next(history)
        self.assertAlmostEqual(pred, 50.0)

    def test_speed_one_full_reversion(self):
        """With speed=1.0, prediction should jump to the window mean."""
        model = self._make_model(window=4, speed=1.0)
        history = pd.Series([10.0, 10.0, 10.0, 10.0, 50.0])
        pred = model.predict_next(history)
        # Window is last 4 values: [10, 10, 10, 50], mean = 20
        # Full reversion: 50 + 1.0 * (20 - 50) = 20
        # But window includes the current value (50), so mean = 20
        expected_mean = np.mean([10.0, 10.0, 10.0, 50.0])
        expected_pred = 50.0 + 1.0 * (expected_mean - 50.0)
        self.assertAlmostEqual(pred, expected_pred, places=5)

    def test_single_value(self):
        model = self._make_model(window=5, speed=0.5)
        history = pd.Series([42.0])
        pred = model.predict_next(history)
        # Mean of window = 42, last = 42, so prediction = 42
        self.assertAlmostEqual(pred, 42.0)

    def test_empty_series(self):
        model = self._make_model(window=5, speed=0.5)
        history = pd.Series([], dtype=float)
        pred = model.predict_next(history)
        self.assertTrue(math.isnan(pred))

    def test_all_nans(self):
        model = self._make_model(window=5, speed=0.5)
        history = pd.Series([float("nan")] * 10)
        pred = model.predict_next(history)
        self.assertTrue(math.isnan(pred))

    def test_large_window_with_short_series(self):
        """Window larger than series should use all available data."""
        model = self._make_model(window=100, speed=0.5)
        history = pd.Series([1.0, 2.0, 3.0])
        pred = model.predict_next(history)
        # mean of [1,2,3] = 2, last = 3, pred = 3 + 0.5*(2-3) = 2.5
        self.assertAlmostEqual(pred, 2.5)

    def test_negative_values(self):
        model = self._make_model(window=5, speed=0.3)
        history = pd.Series([-10.0, -8.0, -6.0, -4.0, -20.0])
        pred = model.predict_next(history)
        # Mean = -9.6, last = -20, should pull toward mean (up)
        self.assertGreater(pred, -20.0)

    def test_fit_is_noop(self):
        model = self._make_model()
        history = pd.Series([1.0, 2.0, 3.0])
        model.fit(history)  # should not raise

    def test_name_attribute(self):
        model = self._make_model()
        self.assertEqual(model.name, "mean_reversion")


# ===========================================================================
# Cross-wrapper consistency
# ===========================================================================


class TestCrossWrapperConsistency(unittest.TestCase):
    """Test that all wrappers follow the same interface contract."""

    def _get_all_models(self):
        from operator1.models.walk_forward import get_default_wf_models
        return get_default_wf_models()

    def test_all_have_name(self):
        for model in self._get_all_models():
            self.assertIsInstance(model.name, str)
            self.assertGreater(len(model.name), 0)

    def test_all_have_predict_next(self):
        for model in self._get_all_models():
            self.assertTrue(hasattr(model, "predict_next"))
            self.assertTrue(callable(model.predict_next))

    def test_all_have_fit(self):
        for model in self._get_all_models():
            self.assertTrue(hasattr(model, "fit"))
            self.assertTrue(callable(model.fit))

    def test_all_return_float(self):
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        for model in self._get_all_models():
            pred = model.predict_next(history)
            self.assertIsInstance(pred, float,
                                 msg=f"{model.name} did not return float")

    def test_all_handle_constant_series(self):
        history = pd.Series([5.0] * 100)
        for model in self._get_all_models():
            pred = model.predict_next(history)
            self.assertFalse(math.isnan(pred),
                             msg=f"{model.name} returned NaN on constant series")
            # On a constant series, all models should predict ~5.0
            self.assertAlmostEqual(pred, 5.0, places=1,
                                   msg=f"{model.name} deviated on constant series")

    def test_all_handle_empty(self):
        history = pd.Series([], dtype=float)
        for model in self._get_all_models():
            pred = model.predict_next(history)
            # Should return NaN, not crash
            self.assertTrue(math.isnan(pred),
                            msg=f"{model.name} did not return NaN on empty")

    def test_all_handle_single_value(self):
        history = pd.Series([99.0])
        for model in self._get_all_models():
            pred = model.predict_next(history)
            self.assertIsInstance(pred, float,
                                 msg=f"{model.name} failed on single value")

    def test_fit_then_predict(self):
        """fit + predict should not crash for any model."""
        history = pd.Series(np.random.randn(100).cumsum() + 100)
        for model in self._get_all_models():
            model.fit(history)
            pred = model.predict_next(history)
            self.assertIsInstance(pred, float,
                                 msg=f"{model.name} failed fit+predict")
            self.assertFalse(math.isnan(pred),
                             msg=f"{model.name} returned NaN after fit")

    def test_unique_names(self):
        models = self._get_all_models()
        names = [m.name for m in models]
        self.assertEqual(len(names), len(set(names)),
                         msg="Model names are not unique")


if __name__ == "__main__":
    unittest.main()
