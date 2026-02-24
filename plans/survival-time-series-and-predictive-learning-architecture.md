# Survival Time-Series and Predictive Learning Architecture

## Overview

This plan introduces a 4-phase enhancement to Operator 1's prediction pipeline
that makes the forecasting system *survival-mode aware*.  Instead of treating
all market conditions identically, the system now classifies each day into one
of six survival modes, tracks model performance per mode via walk-forward
validation, and dynamically selects the best ensemble weights for the current
regime.

---

## Phase 1: Cache All Indicators Daily

**Goal:** Ensure all country macro indicators (GDP, inflation, unemployment,
rates, FX vol) and company protection scores (fuzzy protection, sector
strategicness) are persisted in the 2-year daily Parquet cache so models can
consume them as features.

**Implementation:**

- Added `MACRO_INDICATOR_FIELDS` and `PROTECTION_SCORE_FIELDS` tuples to
  `cache_builder.py` defining the full set of model-consumable columns.
- `build_entity_daily_cache()` now pre-allocates NaN placeholders for all
  macro and protection columns.
- New `enrich_cache_with_indicators()` function merges macro-aligned data,
  survival flags, and fuzzy protection scores into the daily cache.

**Files modified:** `operator1/steps/cache_builder.py`

---

## Phase 2: Pre-Analysis Survival Timeline

**Goal:** Classify every day in the cache into one of six survival modes and
compute transition metadata.

**Survival modes:**

| Code | Mode               | Company Flag | Country Flag | Protected |
|------|--------------------|:----------:|:-----------:|:---------:|
| 0    | normal             | 0          | 0           | -         |
| 1    | company_only       | 1          | 0           | -         |
| 2    | country_protected  | 0          | 1           | 1         |
| 3    | country_exposed    | 0          | 1           | 0         |
| 4    | both_unprotected   | 1          | 1           | 0         |
| 5    | both_protected     | 1          | 1           | 1         |

**Computed columns:**

- `survival_mode` -- string label for the current mode
- `survival_mode_code` -- integer encoding (0-5)
- `switch_point` -- binary flag: 1 on days where the mode changes
- `days_in_mode` -- running counter of consecutive days in current mode
- `stability_score_21d` -- fraction of the last 21 days sharing the same mode

**Files created:** `operator1/analysis/survival_timeline.py`

---

## Phase 3: Walk-Forward Prediction Loop

**Goal:** Iterate day-by-day through the 2-year cache, predict day t+1 from
day t, compare against actuals, and track per-model errors by survival mode.

**Key behaviours:**

1. **Per-model error tracking** -- each model's MAE/RMSE are recorded per
   survival mode, building a mode-conditioned leaderboard.
2. **Retrain at switch points** -- when the survival timeline reports a mode
   transition, all models are re-fitted on history up to day t.
3. **Mode-conditioned leaderboard** -- after the pass, each model gets a
   score per mode, identifying which model works best under each condition.

**Walk-forward models (lightweight, for the WF loop):**

- `WFBaselineModel` -- last-value carry-forward
- `WFEMAModel` -- exponential moving average (span=21)
- `WFLinearTrendModel` -- simple linear trend extrapolation
- `WFMeanReversionModel` -- partial reversion toward rolling mean

**Output:** `WalkForwardResult` containing day-level errors, mode scores,
best-model-by-mode mapping, retrain dates, and overall statistics.

**Files created:** `operator1/models/walk_forward.py`

---

## Phase 4: Survival-Aware Model Weighting

**Goal:** The prediction aggregator uses walk-forward results to select
different ensemble weights depending on the current survival mode, with
soft blending during transitions.

**Key additions to `prediction_aggregator.py`:**

- `compute_survival_aware_weights()` -- given base weights and per-mode
  weights from walk-forward, computes the effective weights for the current
  survival mode with exponential transition blending.
- `get_survival_context_from_cache()` -- extracts the current mode,
  days-in-mode, previous mode, and stability score from the enriched cache.
- `run_prediction_aggregation()` now accepts an optional `mode_weights`
  parameter.  When provided, it overrides the base inverse-RMSE weights
  with survival-aware weights.

**Transition blending:** When the mode changes, the new mode's weights are
blended with the old mode's weights using an exponential ramp:

```
alpha = 1 - exp(-days_in_mode / halflife)
effective_weight = (1 - alpha) * prev_weight + alpha * target_weight
```

Default half-life is 5 business days.

**Files modified:** `operator1/models/prediction_aggregator.py`

---

## Data Flow

```
daily cache (Parquet)
   |
   v
enrich_cache_with_indicators()   <-- macro + fuzzy protection + survival flags
   |
   v
compute_survival_timeline()      <-- 6-mode classification + switch points
   |
   v
run_walk_forward()               <-- day-by-day prediction loop, mode-conditioned errors
   |
   v
get_mode_weights_from_walk_forward()  <-- inverse-MAE weights per mode
   |
   v
run_prediction_aggregation(mode_weights=...)  <-- survival-aware ensemble
```

---

## Testing

Tests are in `tests/test_survival_timeline_walkforward.py` and cover:

- Mode classification (all 6 modes)
- Switch point detection and days-in-mode counting
- Stability score calculation
- Full timeline computation
- Walk-forward engine (per-model errors, retraining, mode scores)
- Survival-aware weight computation and transition blending
- Cache enrichment with indicator columns
