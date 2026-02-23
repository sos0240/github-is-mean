# Full Pipeline Audit -- Data Flow, Model Integration, and Report Quality

## Critical Issue: Long-to-Wide Pivot Missing

### Problem

The canonical translator outputs financial DataFrames in **long format**:

```
| canonical_name | value    | filing_date | report_date | market_id | currency |
|---------------|----------|-------------|-------------|-----------|----------|
| revenue       | 5000000  | 2024-03-15  | 2023-12-31  | eu_esef   | EUR      |
| total_assets  | 20000000 | 2024-03-15  | 2023-12-31  | eu_esef   | EUR      |
| net_income    | 800000   | 2024-03-15  | 2023-12-31  | eu_esef   | EUR      |
```

But `main.py` Step 4 merger (lines 706-742) takes `select_dtypes(include=["number"])` columns and forward-fills them. With long format, only the `value` column is numeric -- so ALL financial concepts get merged as a single `value` column. The cache ends up with one `value` column instead of `revenue`, `total_assets`, `net_income`, etc.

Downstream modules (`derived_variables.py`, `financial_health.py`, `estimator.py`) all expect wide-format columns:
- `df.get("revenue", ...)` 
- `df.get("total_assets", ...)`
- `df.get("cash_and_equivalents", ...)`

These all return NaN because the columns don't exist.

### Fix

Add a pivot step in `main.py` after canonical translation and before cache merge. Use `pivot_to_canonical_wide()` from `canonical_translator.py` to convert:

```
| report_date | revenue | total_assets | net_income | ... |
|-------------|---------|-------------|------------|-----|
| 2023-12-31  | 5000000 | 20000000    | 800000     | ... |
| 2022-12-31  | 4500000 | 18000000    | 700000     | ... |
```

This wide format is what `cache_builder.py`'s `_asof_merge_statement()` and `main.py`'s ad-hoc merger expect.

## Pipeline Data Flow Verification

### Step-by-Step Flow (with fixes)

```
Step 1: Region/Market/Company selection
  OK -- works correctly

Step 2: Profile fetch
  OK -- canonical translation applied, profile has all canonical fields

Step 3: Financial data fetch  
  NOW: PIT clients return long-format canonical DataFrames
  FIX NEEDED: Pivot to wide format before Step 4

Step 3b: Reconciliation
  OK -- normalize_field_names + filing date validation + dedup
  NOTE: reconciliation also does field name aliasing (camelCase->snake_case)
  but our canonical_name column already has correct names

Step 4: Cache build (as-of merge)
  AFTER FIX: Wide-format DataFrames will have individual columns
  that merge correctly into the daily cache

Step 4a: Macro data
  OK -- macro indicators merge correctly as macro_* columns

Step 4b: Estimation (Sudoku inference)
  OK -- expects columns like revenue, total_assets, operates on cache
  Will work once wide-format columns exist from Step 4

Step 5: Feature engineering
  OK -- derived_variables.py uses df.get("revenue", ...) pattern
  Safe: returns NaN series when column missing, computes ratios safely

Step 5b-5d: Survival, fuzzy, financial health
  OK -- all operate on cache columns, handle missing gracefully

Step 5e: Linked entity discovery
  OK -- Gemini proposes names, cross-region resolution works

Step 6: Temporal models
  OK -- all models wrap in try/except, handle missing data
  
  Synergy checks:
  - Granger causality prunes extra_variables before forecasting: OK
  - Cycle features injected pre-forecasting: OK
  - Transformer forecasts injected into forecast_result: OK
  - Copula tail adjustment widens CI on pred_result: OK
  - Pattern drift adjustment applied to OHLC predictor: OK
  - GA optimizer runs post-forecasting: OK

Step 7: Profile builder
  OK -- aggregates all results into company_profile.json
  All model results passed via _to_dict() / _available_dict()

Step 8: Report generation
  OK -- three tiers (Basic/Pro/Premium)
  Uses Gemini or fallback template
  All 22 sections populated from profile JSON
```

### Model Synergy Matrix (verified)

| Synergy | Source Model | Target Model | Integration Point | Status |
|---------|-------------|-------------|-------------------|--------|
| A | Particle Filter + Kalman | State estimation | model_synergies.fuse_kalman_particle_estimates | OK |
| B | Cycle decomposition | Forecasting | apply_pre_forecasting_synergies | OK |
| C | Pattern detector | OHLC predictor | compute_pattern_drift_adjustment | OK |
| D | Granger + Transfer Entropy | Feature pruning | unified causal network | OK |
| E | Copula | Monte Carlo | tail_dependence widening | OK |
| F | DTW analogs | Training windows | analog-boosted learning | OK |
| G | Peer ranking | Survival thresholds | dynamic sector triggers | Linked entities needed |
| H | GA optimizer | Burn-out loop | evolving ensemble weights | OK |

## Action Items

### Must Fix (Critical)

- [ ] **Pivot financial DataFrames from long to wide format in main.py** before cache merge. Add `pivot_to_canonical_wide()` calls after Step 3/3b for income_df, balance_df, cashflow_df.

### Verify After Fix

- [ ] Run full test suite to confirm no regressions
- [ ] Verify derived_variables receives actual financial columns
- [ ] Verify financial_health can compute Altman Z, Beneish M
- [ ] Verify estimation Pass 1 accounting identities can resolve
