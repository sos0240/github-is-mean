# Operator 1: Full Architectural Review

## Part 1: What's Strong

### The core data flow is sound
The pipeline follows a clean linear progression: verify -> extract -> cache -> derive -> estimate -> model -> predict -> report. Each stage receives a DataFrame, adds columns, and passes it forward. The as-of merge for financial statements is correctly implemented with look-ahead validation. The point-in-time guarantee is the backbone of the whole system's integrity.

### The try/except resilience pattern works
Every model is wrapped in try/except. If LSTM fails, the tree ensemble takes over. If the transformer diverges, forecasting continues without it. The baseline (last-value/EMA) always succeeds. The pipeline never crashes from a single model failure. This matches the spec's explicit requirement.

### The 5-tier survival hierarchy is faithful to the spec
The weights in survival_hierarchy.yml match the PDF exactly (20/20/20/20/20 normal, 50/30/15/4/1 survival, etc.). The 4 survival triggers match. The vanity adjustment matches. The 4+1 regime cases match. This is one of the most spec-compliant parts of the codebase.

### The report generator is remarkably complete
22 sections, 25 builder functions, Bloomberg-style dark theme charts, Gemini prompt with validation, fallback template when Gemini is unavailable. Every model name is mapped to a human-readable label. The report reads like something a finance professional would write, not a programmer.

---

## Part 2: What's Weak or Fragile

### 2.1 TTM Aggregation Is An Approximation
**Problem**: The spec says `free_cash_flow_ttm_asof` should be "sum of last 4 quarterly filings, point-in-time." The code currently uses the single as-of-aligned value (the latest quarter's value repeated daily), not a rolling 4-quarter sum.

**Impact**: Revenue TTM, net income TTM, EBITDA TTM, and FCF TTM are all approximations. This means profitability margins (gross, operating, net) and valuation ratios (P/E, EV/EBITDA) are based on single-quarter data, not trailing twelve months. In practice, this overstates seasonal effects and understates annual trends.

**Fix priority**: HIGH. The cache_builder already has the quarterly statements with report_dates. The fix is to compute a rolling 4-period sum during the as-of merge, keeping only periods where all 4 quarters are available.

### 2.2 The Forward Pass and Burn-Out Are Simplified
**Problem**: The spec describes a day-by-day forward pass where each model predicts t+1, compares to cache, and updates parameters online. The actual implementation in `forecasting.py` fits models on the full training window once, then generates predictions. The burn-out loop runs but is a simplified retrain, not the spec's "intensive re-training with higher learning rates and regime-specific model instances."

**Impact**: The models don't truly learn online. They fit once and extrapolate. This means they can't adapt to mid-window regime changes as effectively as the spec envisions.

**Fix priority**: MEDIUM. This is a compute-vs-accuracy tradeoff. The current approach works for Kaggle's constraints. A true online learning loop would require per-step model updates, which is expensive.

### 2.3 Linked Entity Caches Are Often Thin
**Problem**: When `--skip-linked` is used (which is common for faster runs), there are zero linked entities. Even without that flag, Gemini may return few usable entities, and the equity provider may not have data for all of them.

**Impact**: The linked aggregates (competitors_avg_return, supply_chain_avg_drawdown, sector_median_vol, etc.) are null. The peer ranking module produces no ranks. The game theory module has no competitors to analyze. Roughly 30% of the report sections become "not available for this run."

**Fix priority**: MEDIUM. The fallback to sector peers helps but is limited. Consider caching a pre-built peer universe from FMP's sector list endpoint.

### 2.4 Country Survival Mode Is Limited by World Bank Frequency
**Problem**: World Bank macro indicators are annual (sometimes monthly). Country survival triggers (credit spread > 5%, unemployment surge, yield curve inversion) require higher-frequency data that World Bank doesn't provide.

**Impact**: Country survival mode is almost never triggered because annual data can't capture intra-year spikes. The country_survival_mode_flag is effectively always 0 for most runs.

**Fix priority**: LOW for now. The spec acknowledges this: "World Bank is mostly low-frequency macro. Use best-effort proxies." A future enhancement would add a secondary rates provider (e.g., FRED for US, ECB for Europe) for higher-frequency data.

### 2.5 Data Source Reconciliation (FMP vs Eulerpool vs EOD)
**Problem**: Different providers may return different values for the same company's fundamentals. FMP might report revenue of $95B while Eulerpool says $97B (different filing interpretations, currency conversions, or reporting periods).

**Impact**: When switching between providers, the pipeline produces slightly different results. A company analyzed with Eulerpool then re-analyzed with FMP will have different derived variables, different survival flags, and potentially different recommendations.

**What should happen**: The system should detect discrepancies and flag them. Key reconciliation points:
- Revenue, net income, total debt should be within 5% between providers
- If they diverge more, log a warning and use the primary provider's value
- Store a `data_source_confidence` score in the profile

**Fix priority**: LOW. This is an accuracy refinement, not a pipeline blocker.

### 2.6 The Estimation Engine Doesn't Use Regime Context
**Problem**: The spec says (Section J.2): "Use a regime-aware window: select historical samples from the same regime label as day t." The current estimator uses BayesianRidge/IterativeImputer without regime filtering.

**Impact**: The Sudoku inference treats all historical days equally when imputing missing values. In a bear market day, it might impute values based on bull market patterns, which would be wrong.

**Fix priority**: MEDIUM. Add a regime-weighted sample selection before the imputer runs.

---

## Part 3: How the 5-Plane Model Should Actually Affect Predictions

The plane classification we added is currently decorative -- it labels the company and appears in the report, but doesn't change how models weight linked variables.

### What should change mathematically:

1. **Granger causality pruning should be plane-aware**: When pruning weak variables before VAR/LSTM training, adjacent-plane linked variables (e.g., a supplier's revenue for a manufacturing company) should get a lower pruning threshold than distant-plane variables (e.g., a bank's capital ratio for a retailer). Currently all linked variables are treated equally by the causal pruning.

2. **The survival hierarchy should shift by plane**: A company in the Financial Services plane should weight Tier 2 (solvency) even higher than Tier 1 (liquidity) because banks die from insolvency, not illiquidity. A company in the Supply plane should weight Tier 3 (market stability) higher because commodity price volatility is their primary survival risk.

3. **Linked aggregate weights should reflect plane adjacency**: When computing `competitors_avg_return`, entities in the same plane should count more. When computing `supply_chain_stress`, upstream-plane entities should count more than same-plane entities.

4. **The Monte Carlo simulation should use plane-specific correlation structures**: Supply plane companies are correlated through commodity prices. Financial Services companies are correlated through interest rates. Manufacturing companies are correlated through capex cycles. The copula model should use different dependency structures per plane.

### Implementation path:
These changes require modifying `model_synergies.py` and `linked_aggregates.py` to accept plane classification as a parameter and adjust weights accordingly. This is a meaningful enhancement but not a blocking issue.

---

## Part 4: What to Build Next (Prioritized)

### Priority 1: Fix TTM aggregation
- Compute rolling 4-quarter sums for revenue, net_income, ebitda, FCF in `cache_builder.py`
- This fixes profitability margins and valuation ratios across the board
- Low risk, high impact on accuracy

### Priority 2: Add FMP sector peers as a pre-built fallback
- When no linked entities are discovered, fetch the top 10 companies in the same sector from FMP
- FMP endpoint: `/stock-screener?sector={sector}&limit=10`
- This ensures linked aggregates and peer ranking always have data

### Priority 3: Wire temporal masking into linked_aggregates.py
- Use the `relationship_start`/`relationship_end` fields from Gemini discovery
- Apply null masking on linked variables outside validity windows
- Low complexity, already designed

### Priority 4: Plane-aware model weighting
- Modify `model_synergies.py` to accept plane classification
- Adjust Granger pruning thresholds by plane adjacency
- Add plane-specific correlation to Monte Carlo copula

### Priority 5: Data source reconciliation layer
- When using FMP-full, compare a few key metrics against the profile values
- Flag discrepancies > 5% as warnings in the profile metadata
- Display in the LIMITATIONS section of the report

### Priority 6: Online learning in the forward pass
- Convert the forecasting loop to true online learning (update model parameters per step)
- This is the most complex change and the spec's core innovation
- Requires careful compute budgeting for Kaggle
