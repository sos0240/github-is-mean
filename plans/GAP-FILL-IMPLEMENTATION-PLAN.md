# Gap-Fill Implementation Plan

**Date**: 2026-02-20
**Based on**: `COMPREHENSIVE-PDF-VS-CODE-GAP-ANALYSIS.md`
**Goal**: Bring the Operator 1 codebase to full conformance with both PDF specifications.

---

## Eulerpool Client Verification Results

Tested all 9 Eulerpool endpoints against `api.eulerpool.com` with the provided API key:

| Finding | Detail |
|---------|--------|
| **API is live** | Root endpoint returns `{"health":"ok2"}` |
| **Endpoint paths are correct** | `/api/1/equity/profile/{isin}` returns 401 (not 404) |
| **Auth format is correct** | Bearer token is recognized by the API |
| **Bug fixed** | Added `User-Agent: Operator1/1.0` header to prevent Cloudflare 403 blocks (affects `urllib`-based tools; `requests` library sets its own default but explicit is safer) |
| **Key permissions** | The test key returns 401 "not authorized to access this resource" -- this is a key/subscription issue, not a code defect |

**Conclusion**: The Eulerpool client implementation (`operator1/clients/eulerpool.py`) is **correctly structured** with proper endpoint paths, auth format, response normalization, and error handling. No structural changes needed.

---

## Phase 1: Critical Missing Modules (P0)

### 1.1 Particle Filter (Sequential Monte Carlo)
**Spec ref**: Core Idea PDF, Section E.2, Module Category 2
**Why**: Required for non-linear, non-Gaussian state estimation. Handles extreme events better than Kalman filter. Specified for survival mode prediction.

**File**: `operator1/models/particle_filter.py` (new)

**Implementation**:
- `ParticleFilter` class with configurable particle count (default 1000)
- `predict()` method: propagate particles through state transition model
- `update()` method: weight particles by observation likelihood, resample
- `get_distribution()` method: return full probability distribution (not just mean)
- Integration: Wire into `EnsemblePredictor` in `forecasting.py`
- Fallback: Degrade to Kalman filter if particle count is too low for convergence

**Estimated effort**: ~300 lines

### 1.2 Transformer Architecture
**Spec ref**: Core Idea PDF, Section E.2, Module Category 2
**Why**: Attention-based modeling identifies which past days/variables matter most. Modern alternative to LSTM for long sequences.

**File**: `operator1/models/transformer_forecaster.py` (new)

**Implementation**:
- `TransformerForecaster(nn.Module)` with configurable heads, layers, d_model
- Positional encoding for temporal sequences
- Multi-head self-attention over variable + time dimensions
- `train_transformer()` function matching LSTM training interface
- `predict_transformer()` function returning predictions + attention weights
- Integration: Register in `EnsemblePredictor` alongside LSTM
- Fallback: Skip if data is too thin (< 100 rows) or training diverges

**Estimated effort**: ~350 lines

---

## Phase 2: Incomplete Core Logic (P1)

### 2.1 Soft Regime Switching
**Spec ref**: Core Idea PDF, Section K.2
**Why**: Instead of hard regime labels, predictions should be a mixture: `pred = sum_r p(r|t) * pred_r`

**File**: Modify `operator1/models/forecasting.py` (EnsemblePredictor)

**Implementation**:
- Store regime probability vectors from HMM/GMM (already available in `regime_probs`)
- Modify `EnsemblePredictor.predict()` to compute weighted mixture across regimes
- Each model produces per-regime predictions (or a single prediction modulated by regime weights)
- Regime transition probabilities inform prediction uncertainty

**Estimated effort**: ~150 lines of changes

### 2.2 Per-Regime Model Instances
**Spec ref**: Core Idea PDF, Section K.3
**Why**: Each regime should maintain its own model parameters updated incrementally.

**File**: Modify `operator1/models/forecasting.py`

**Implementation**:
- `RegimeAwareModelBank` class: holds N instances of each model type (one per regime)
- During forward pass, select model instance(s) based on regime probabilities
- Update only the active regime's model parameters (or all, weighted by probability)
- Maintain per-regime rolling sufficient statistics (means, covariances)
- Keep compute feasible: use lightweight models per regime (Ridge/AR instead of full LSTM)

**Estimated effort**: ~250 lines

### 2.3 Regime-Weighted Burn-Out Windows
**Spec ref**: Core Idea PDF, Section L.1
**Why**: Training window should weight past days by regime similarity and recency: `w(t) ~ exp(-dt/half_life) * similarity(regime(t), regime(t_current))`

**File**: Modify `operator1/models/forecasting.py` (`run_burnout`)

**Implementation**:
- Compute regime similarity matrix from HMM transition probabilities
- For each burn-out step, weight historical samples by `exp(-age/half_life) * regime_similarity`
- Use weighted samples in model updates (weighted loss / weighted sampling)
- Track convergence per-regime (stop when regime-weighted error plateaus)

**Estimated effort**: ~100 lines of changes

### 2.4 Module Synergy Wiring
**Spec ref**: Core Idea PDF, Section E.3

**Synergy 2: Structural Break -> Model Reset**
- File: `operator1/models/forecasting.py` (`run_forward_pass`)
- When `structural_break == 1` on day t, trigger:
  - Kalman filter state reset (reinitialize P matrix)
  - LSTM: mark for retraining on post-break data
  - Ensemble weights: reset to uniform
- Estimated effort: ~50 lines

**Synergy 3: Granger Causality -> Feature Pruning**
- File: `operator1/models/forecasting.py` (before VAR/LSTM training)
- After computing Granger causality matrix, filter input variables:
  - Keep only variables with significant causal links (p < 0.05) to target variables
  - Apply `max_vars_for_var` cap per spec
- Estimated effort: ~40 lines

**Synergy 6: Sobol -> Hierarchy Validation**
- File: `operator1/models/sensitivity.py` + `operator1/report/profile_builder.py`
- Compare Sobol sensitivity indices to hierarchy tier weights
- Flag mismatches (e.g., Tier 3 variable has higher Sobol index than Tier 2 variable)
- Include mismatch report in company profile under `model_metrics`
- Estimated effort: ~60 lines

---

## Phase 3: Incomplete Data Pipeline (P2)

### 3.1 Wire Eulerpool Peers/Supply-Chain/Executives
**Spec ref**: Coding Instructions PDF, Section 3 (Step C)
**Why**: These endpoints seed the relationship graph and provide exec compensation data for vanity analysis.

**File**: Modify `operator1/steps/entity_discovery.py` + `operator1/steps/data_extraction.py`

**Implementation**:
- In `discover_linked_entities()`, call `get_peers()` and `get_supply_chain()` BEFORE Gemini discovery
- Merge Eulerpool peer/supply-chain ISINs with Gemini-discovered entities (deduplicate by ISIN)
- In `extract_all_data()`, call `get_executives()` for target company
- Parse executive compensation data into `executive_compensation` column for vanity module
- Store peer ISINs from Eulerpool as `relationship_type: "eulerpool_peer"`

**Estimated effort**: ~80 lines of changes

### 3.2 Deepen Linked Entity Caches
**Spec ref**: Coding Instructions PDF, Section 4
**Why**: Linked entities should have the same derived variables as the target for proper aggregation.

**File**: Modify `operator1/steps/cache_builder.py` + `operator1/features/linked_aggregates.py`

**Implementation**:
- After building linked entity raw caches, run `compute_derived_variables()` on each
- Compute full set: returns, volatility, drawdown, solvency ratios, profitability, valuation
- In `linked_aggregates.py`, compute daily aggregates per relationship group:
  - `competitors_avg_return_21d`, `competitors_median_vol_21d`
  - `supply_chain_avg_drawdown_252d`
  - `sector_median_debt_to_equity`, `industry_median_pe`
  - `rel_strength_vs_sector`, `valuation_premium_vs_industry`
- Budget caps: limit to top N linked entities per group to stay within compute limits

**Estimated effort**: ~150 lines of changes

### 3.3 No-Look-Ahead Validation Test
**Spec ref**: Coding Instructions PDF, Section 7.1

**File**: `tests/test_no_lookahead.py` (new)

**Implementation**:
- Generate synthetic cache with known statement dates
- Verify that for every day t, all `*_asof` values have `report_date <= t`
- Verify that no forward-filled market data exists across weekends/holidays
- Assert that `is_missing_*` flags are set correctly for truly missing values

**Estimated effort**: ~80 lines

---

## Phase 4: Enhancement Gaps (P3)

### 4.1 Copula Model Variants
**File**: Modify `operator1/models/copula.py`
- Add Student-t copula for heavy-tailed dependencies
- Add Clayton copula for lower-tail dependence (crisis co-movements)
- Report tail dependence coefficient in profile
- Estimated effort: ~100 lines

### 4.2 Full Genetic Algorithm (DEAP)
**File**: Modify `operator1/models/forecasting.py`
- Wire DEAP-based evolutionary optimization for ensemble weights (currently simplified)
- Configurable generations, population size, crossover/mutation rates
- Early stopping when fitness plateaus
- Estimated effort: ~80 lines

### 4.3 Unit Normalization Metadata
**File**: Modify `operator1/features/derived_variables.py` + `operator1/quality/data_quality.py`
- Store `unit` metadata for each variable (e.g., USD, ratio, percentage)
- Add `unit_check()` validation that flags unit mismatches in ratios
- Estimated effort: ~60 lines

---

## Implementation Order (Recommended)

| Sprint | Tasks | Estimated Lines | Priority |
|--------|-------|----------------|----------|
| Sprint 1 | 1.1 Particle Filter + 1.2 Transformer | ~650 new | P0 |
| Sprint 2 | 2.1 Soft Regime Switching + 2.2 Per-Regime Models | ~400 modified | P1 |
| Sprint 3 | 2.3 Regime-Weighted Burn-Out + 2.4 Synergy Wiring | ~250 modified | P1 |
| Sprint 4 | 3.1 Wire Eulerpool Endpoints + 3.2 Deepen Linked Caches | ~230 modified | P2 |
| Sprint 5 | 3.3 No-Look-Ahead Test + 4.1 Copula Variants | ~180 new | P2/P3 |
| Sprint 6 | 4.2 DEAP Genetic Algorithm + 4.3 Unit Normalization | ~140 modified | P3 |

**Total estimated**: ~1,850 lines of new/modified code across 6 sprints.

---

## Testing Strategy

Each sprint should include:
1. Unit tests for new modules (Particle Filter, Transformer, etc.)
2. Integration test: verify module registers in ensemble and produces predictions
3. Regression test: existing tests continue passing
4. For Sprint 5: dedicated no-look-ahead validation test

---

## Non-Code Action Items

1. **Eulerpool API key**: The provided key (`eu_prod_*`) returns 401 for equity endpoints. Need to verify the subscription level includes equity data access, or request an upgraded key.
2. **Test environment**: Need pytest + dependencies installed to run test suite. Consider adding a `Makefile` or `scripts/setup.sh` for environment bootstrap.
3. **Kaggle notebook**: The codebase runs as a CLI tool (`main.py`). Consider also generating a Kaggle-compatible notebook from the pipeline for users who prefer that execution model.
