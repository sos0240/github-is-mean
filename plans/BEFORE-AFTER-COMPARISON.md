# Before vs After Gap-Filling: What Changed and What Holds More Value

**Date**: 2026-02-20
**Branch**: `feature/comprehensive-gap-analysis` vs `main`
**Diff**: +2,404 lines / -108 lines across 16 files

---

## Quantitative Summary

| Metric | Before (main) | After (this branch) | Delta |
|--------|:---:|:---:|:---:|
| Total Python lines | 22,558 | 24,785 | +2,227 |
| Module count | 47 | 50 | +3 new modules |
| Files changed | -- | 16 | -- |
| Data providers | 2 (Eulerpool, EOD) | 3 (+ FMP-full) | +1 |
| Math model modules | 17 | 19 | +2 (Particle Filter, Transformer) |
| CLI flags | 8 | 9 | +1 (--provider) |
| Config parameters | 8 | 13 | +5 rate limit/retry settings |

---

## What Changed: Category by Category

### 1. Data Provider Architecture (HIGH VALUE)

**Before**: Pipeline required Eulerpool or EOD API key + FMP for OHLCV. If Eulerpool was down or the key lacked permissions, the entire pipeline was dead.

**After**: Three-tier provider system with FMP as a complete standalone alternative.

| Aspect | Before | After | Value |
|--------|--------|-------|-------|
| Minimum API keys needed | 3 (Eulerpool + FMP + Gemini) | 2 (FMP + Gemini) | Removes Eulerpool dependency |
| Provider options | 2 (Eulerpool, EOD) | 3 (+ FMP-full) | Resilience |
| User choice | Automatic (hardcoded priority) | `--provider` flag + interactive UI | Flexibility |
| FMP coverage | OHLCV only | Profile, 3 statements, peers, executives, OHLCV | Complete |
| Supply chain from FMP | No | No (uses Gemini) | Same |

**Why this holds the most value**: Eulerpool has been unreliable (403/401 errors in testing). EOD free tier only covers 1 year. FMP-full lets users run the complete pipeline with a single, widely-available API key. This is the difference between "can run" and "cannot run."

### 2. Missing P0 Math Modules (HIGH VALUE)

**Before**: The spec requires 20+ mathematical modules. Two were completely missing:
- Particle Filter (Sequential Monte Carlo) -- spec says "handles extreme events better than Kalman"
- Transformer Architecture -- spec says "modern alternative to LSTM, better at long sequences"

**After**: Both implemented and wired into the pipeline.

| Module | Lines | What it adds |
|--------|-------|-------------|
| Particle Filter | 379 | Full probability distribution (not just point estimate), systematic resampling, handles non-linear state transitions. Critical for survival mode where Kalman assumptions break down. |
| Transformer | 431 | Self-attention identifies which past days/variables matter most. Gradient-based feature importance. Multi-horizon forecasting. Handles long sequences better than LSTM. |

**Why this holds high value**: These complete the spec's required model suite. The Particle Filter is specifically called out for "survival mode prediction" -- without it, the system's non-linear crisis modeling was a gap. The Transformer adds attention-based interpretability that the existing LSTM lacks.

### 3. Rate Limiting and Retry Logic (MEDIUM-HIGH VALUE)

**Before**: Retried ALL non-200 HTTP status codes (including 401, 403, 404). No rate limiting. No configurable retry codes.

**After**: Smart retries + rate limiting.

| Aspect | Before | After |
|--------|--------|-------|
| Retry on 401 (bad key) | Yes (wasteful, 3 attempts) | No (fail fast) |
| Retry on 403 (forbidden) | Yes (wasteful) | No (fail fast) |
| Retry on 429 (rate limit) | Yes | Yes |
| Retry on 5xx (server error) | Yes | Yes |
| Rate limiter | None | Token-bucket, 2 calls/sec default |
| Max retries | 3 | 5 (configurable) |
| Retry-After header | Respected | Respected |

**Why this matters**: On a free FMP tier (250 calls/day), the old code would waste retries on non-retryable errors and had no rate control. The pipeline could exhaust its daily quota before finishing. Now it fails fast on auth errors and throttles to stay within limits.

### 4. API Key Safety (MEDIUM VALUE)

**Before**: Keys loaded as-is from env vars, .env, and Kaggle secrets. Copy-pasting keys with trailing newlines/spaces caused silent 401 failures.

**After**: All key values `.strip()`-ed at every loading point.

| Source | Before | After |
|--------|--------|-------|
| `.env` file | `strip()` on value | `strip()` on value (was already done) |
| Environment variables | No stripping | `.strip()` applied |
| Kaggle secrets | No stripping | `.strip()` applied |
| Eulerpool client | No User-Agent | `User-Agent: Operator1/1.0` (prevents Cloudflare 403) |

**Why this matters**: A common user failure mode. "My key works in curl but not in the pipeline" is almost always trailing whitespace.

### 5. Cache Builder Field Mappings (MEDIUM VALUE)

**Before**: Missing rename mappings for 8 FMP field names. Missing `sga_expenses`, `stock_buybacks`, `total_debt` from STATEMENT_FIELDS.

**After**: Full field coverage.

| Field | Before | After | Used by |
|-------|--------|-------|---------|
| `sgaExpenses` -> `sga_expenses` | Missing | Mapped | Vanity analysis |
| `freeCashFlow` -> `free_cash_flow` | Missing | Mapped | Cash is King filter |
| `stockBuybacks`/`commonStockRepurchased` -> `stock_buybacks` | Missing | Mapped | Vanity analysis |
| `totalDebt` -> `total_debt` | Missing | Mapped | Solvency filter |
| `rdExpenses` -> `rd_expenses` | Missing | Mapped | Future features |
| `eps`, `epsDiluted` | Missing | Mapped | Valuation |

**Why this matters**: Without `sga_expenses` mapping, the vanity analysis module couldn't compute SGA bloat -- a core spec requirement. Without `stock_buybacks`, it couldn't detect vanity buybacks during cash burn.

### 6. Interactive UI (run.py) (MEDIUM VALUE)

**Before**: Fixed flow assuming Eulerpool + FMP. No provider choice. Error if no Eulerpool/EOD key.

**After**: Clear provider selection with rate limit info.

```
Choose your data provider setup:

  1. FMP for everything (fundamentals + OHLCV)
     Simplest setup. FMP handles profile, statements, peers, executives, and prices.

  2. Eulerpool (fundamentals) + FMP (OHLCV only)
     Eulerpool for financials, FMP only for daily price candles.

  3. EOD Historical Data (fundamentals) + FMP (OHLCV only)
     Note: EOD free tier only covers ~1 year of history.

Rate limit info:
  - FMP free tier: 250 calls/day. Paid: 300/min.
  - Disk caching: API responses cached for 24h to avoid repeat calls.
```

### 7. Error Handling (LOW-MEDIUM VALUE)

**Before**: `data_extraction.py` and `verify_identifiers.py` only caught `EulerportAPIError` and `EODAPIError`. Using FMP-full as provider would cause unhandled exceptions.

**After**: `FMPFullAPIError` added to all exception handlers across the pipeline.

### 8. Documentation (REFERENCE VALUE)

Two new plan documents:
- `COMPREHENSIVE-PDF-VS-CODE-GAP-ANALYSIS.md` (508 lines) -- detailed section-by-section comparison
- `GAP-FILL-IMPLEMENTATION-PLAN.md` (237 lines) -- 6-sprint plan with effort estimates

---

## Value Ranking: What Matters Most

### Tier 1: Makes the app usable (without these, the app may not run)
1. **FMP-full provider** -- Users can now run the entire pipeline with just an FMP key. Eliminates the Eulerpool single-point-of-failure.
2. **API key whitespace stripping** -- Prevents the most common "works in curl, fails in pipeline" user issue.
3. **Error handling for FMPFullAPIError** -- Without this, selecting `--provider fmp-full` would crash on any API error.

### Tier 2: Makes the app correct (addresses spec conformance gaps)
4. **Particle Filter** -- Completes the spec's non-linear estimation requirement. Critical for survival mode modeling.
5. **Transformer Architecture** -- Completes the spec's attention-based forecasting requirement. Adds interpretable feature importance.
6. **Cache builder field mappings** -- Without `sga_expenses`/`stock_buybacks` mappings, vanity analysis was silently broken.

### Tier 3: Makes the app robust (production quality improvements)
7. **Smart retry logic** -- Fail fast on 401/403/404, only retry on 429/5xx. Saves API quota.
8. **Rate limiting** -- Prevents hitting free-tier daily limits during a single pipeline run.
9. **Provider selection UI** -- Clear user experience with rate limit awareness.

### Tier 4: Reference (helps future development)
10. **Gap analysis document** -- Maps every PDF section to implementation status with coverage percentages.
11. **Implementation plan** -- 6-sprint roadmap for remaining gaps (Sprints 2-6 still pending).

---

## What's Still Missing (Sprints 2-6)

| Gap | Impact | Status |
|-----|--------|--------|
| Soft regime switching (mixture predictions) | Predictions use hard regime labels instead of soft mixtures | Not started |
| Per-regime model instances | All models share one param set instead of regime-specific | Not started |
| Regime-weighted burn-out | Training doesn't weight by regime similarity | Not started |
| Synergy wiring (break->reset, Granger->pruning) | Detected breaks don't reset models; Granger doesn't prune inputs | Not started |
| Eulerpool peers/supply-chain/executives wiring | Endpoints exist but not actively populating linked entities | Not started |
| Linked entity cache depth | Linked entities don't get full derived variable computation | Not started |
| No-look-ahead validation test | No automated test verifying temporal integrity | Not started |
| Copula variants (t-copula, Clayton) | Only Gaussian copula, no tail dependencies | Not started |

These represent ~1,200 additional lines across Sprints 2-6.
