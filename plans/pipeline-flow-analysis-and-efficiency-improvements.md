# Pipeline Flow Analysis and Efficiency Improvements

## Current Pipeline Flow (main.py)

```
Step 1: Market/Company Selection
  -> pit_registry.py (market lookup)
  -> _create_pit_client() (client instantiation)
  -> pit_client.search_company() / list_companies()

Step 2: Company Profile
  -> pit_client.get_profile()
  -> canonical_translator.translate_profile()
  -> supplement.enrich_profile() (for non-US markets)

Step 3: PIT Financial Data
  -> pit_client.get_income_statement()    -> canonical_translator.translate_financials()
  -> pit_client.get_balance_sheet()       -> canonical_translator.translate_financials()
  -> pit_client.get_cashflow_statement()  -> canonical_translator.translate_financials()
  -> pit_client.get_quotes()              -> ohlcv_provider.fetch_ohlcv() (fallback)
  -> data_reconciliation.reconcile_financial_data()
  -> canonical_translator.pivot_to_canonical_wide()  (long -> wide format)

Step 4: Daily Cache Build
  -> OHLCV as spine (or empty date range)
  -> Forward-fill financial statements onto daily index (as-of join)
  -> Macro data fetch (macro_client.fetch_macro_indicators())
  -> MacroDataset construction (macro_mapping.fetch_macro_data())
  -> Macro indicators merged into cache
  -> Macro quadrant classification

Step 4b: Estimation
  -> estimator.run_estimation() (Pass 1: accounting identities, Pass 2: imputation)

Step 5: Feature Engineering
  -> derived_variables.compute_derived_variables() (~25 financial ratios)
  -> survival_mode.compute_company_survival_flag()
  -> hierarchy_weights.compute_hierarchy_weights()
  -> fuzzy_protection.compute_fuzzy_protection()
  -> financial_health.compute_financial_health()
  -> Entity discovery via Gemini (optional)
  -> Graph risk + game theory (optional)

Step 6: Temporal Modeling (25+ models, all optional)
  -> regime_detector.detect_regimes_and_breaks()
  -> regime_mixer.compute_dual_regimes()
  -> granger_causality.compute_granger_causality()
  -> causality.compute_transfer_entropy()
  -> cycle_decomposition.run_cycle_decomposition()
  -> pattern_detector.detect_patterns()
  -> model_synergies.apply_pre_forecasting_synergies()
  -> forecasting.run_forecasting()
  -> forecasting.run_forward_pass()
  -> forecasting.run_burnout()
  -> monte_carlo.run_monte_carlo()
  -> copula.run_copula_analysis()
  -> transformer_forecaster.train_transformer()
  -> particle_filter.run_particle_filter()
  -> prediction_aggregator.run_prediction_aggregation()
  -> conformal.build_conformal_result()
  -> dtw_analogs.find_historical_analogs()
  -> explainability.compute_shap_explanations()
  -> sensitivity.run_sensitivity_analysis()
  -> genetic_optimizer.run_genetic_optimization()
  -> ohlc_predictor.predict_ohlc_series()

Step 7: Profile Build
  -> profile_builder.build_company_profile()
  -> Inject all model results, macro data, metadata

Step 8: Report Generation
  -> report_generator.generate_all_reports() (Basic + Pro + Premium tiers)
```

## Flow Correctness Assessment

### What Works Well
1. **PIT data integrity**: Filing dates are correctly preserved for as-of joins (no look-ahead bias)
2. **Canonical translation**: After our fix, all 10 PIT sources produce uniform field names
3. **Graceful degradation**: Every step is wrapped in try/except, so partial failures don't crash the pipeline
4. **Multi-market support**: The same pipeline works for US, EU, UK, Japan, Korea, Taiwan, Brazil, Chile

### Flow Issues Still Present

1. **Duplicate client creation logic**: `main.py:_create_pit_client()` and `equity_provider.py:create_pit_client()` are identical. `entity_discovery.py` uses the equity_provider version while main.py uses its own copy.

2. **Reconciliation runs BEFORE pivot**: At main.py:654, `reconcile_financial_data()` runs on long-format DataFrames, but at main.py:674, the data gets pivoted to wide format. The reconciliation should run AFTER pivot (or on both) since column renaming happens in the reconciliation step but columns change shape in the pivot.

3. **OHLCV counter is process-global**: The Alpha Vantage daily counter in `ohlcv_provider.py` uses module-level globals. In a notebook where cells are re-run, the counter persists but the actual API limit resets server-side. Could cause premature throttling.

4. **Macro data fetched but not validated**: The macro indicators are merged into the cache without checking that time ranges overlap. A 5-year macro series merged with a 2-year OHLCV spine could have gaps at the edges.

---

## Efficiency Bottlenecks (Ranked by Impact)

### 1. EDINET Scanning (HIGH IMPACT)
**Current**: 13 API calls for `list_companies()`, 52 API calls per statement type for `_fetch_filings()`. A single company analysis = ~170 EDINET API calls.
**Fix**: Download EDINET code list CSV once (published monthly), use it as bulk lookup. For filings, download the document list for a specific company using EDINET code directly instead of scanning all dates.
**Savings**: 170 API calls -> ~5 API calls (97% reduction)

### 2. No Parallel API Calls (MEDIUM IMPACT)
**Current**: All 3 financial statements + OHLCV are fetched sequentially in main.py Steps 3-4.
**Fix**: Use `concurrent.futures.ThreadPoolExecutor` to fetch income, balance, cashflow, and quotes in parallel since they hit different endpoints.
**Savings**: 4x faster data fetching (especially for slow APIs like EDINET, CVM)

### 3. CVM CSV Downloads Not Cached (MEDIUM IMPACT)
**Current**: CVM fetches full annual CSV datasets (~10MB each) via raw `requests.get()` bypassing the disk cache. Each run re-downloads the same CSVs.
**Fix**: Route through `cached_get()` or add a dedicated CSV cache.
**Savings**: Eliminates ~40MB of redundant downloads per run

### 4. Unnecessary Re-computation (LOW-MEDIUM IMPACT)
**Current**: Every model in Step 6 runs on every pipeline execution, even when cached results exist.
**Fix**: Add a result cache keyed on (company_id, cache_hash). If the cache DataFrame hasn't changed, skip model re-execution.
**Savings**: 60-80 seconds saved on re-runs of the same company

### 5. OpenFIGI Rate Limit Exposure (LOW IMPACT)
**Current**: OpenFIGI calls (25/min without key) bypass the per-host rate limiter.
**Fix**: Route through a `cached_post()` function that respects rate limits.
**Savings**: Prevents 429 errors during multi-company enrichment

### 6. Redundant Canonical Translation (LOW IMPACT)
**Current**: Each PIT client does its own concept mapping in `_CONCEPT_MAP`, then calls `canonical_translator.translate_financials()` which does the mapping again via `_MARKET_CONCEPT_MAPS`. The client-level maps are subsets of the translator's maps.
**Fix**: Remove client-level `_CONCEPT_MAP` dicts and let the canonical translator handle all mapping centrally.
**Savings**: ~200 lines of duplicated mapping code removed, single source of truth

---

## Recommended Implementation Priority

| Priority | Change | Impact | Effort | Risk |
|----------|--------|--------|--------|------|
| 1 | EDINET bulk download | -97% API calls for JP market | Medium | Low |
| 2 | Parallel financial data fetch | -75% fetch time | Low | Low |
| 3 | CVM CSV caching | -40MB/run for BR market | Low | Low |
| 4 | Deduplicate create_pit_client | Code quality | Low | None |
| 5 | Remove redundant client concept maps | Code quality | Low | Low |
| 6 | Add cached_post() to http_utils | Prevents rate limits | Medium | Low |
| 7 | Model result caching | -60s on re-runs | Medium | Medium |

---

## Summary

The pipeline flow is architecturally sound. The main inefficiencies are:
- **EDINET scanning** (biggest offender -- 170 API calls per company)
- **Sequential data fetching** (easy parallelization opportunity)
- **Missing POST request caching** (CVM, MOPS, OpenFIGI, Gemini)

The bugs we already fixed (canonical naming, load_config, Z-Score, EBITDA) were the most critical correctness issues. The remaining improvements are performance/efficiency optimizations.
