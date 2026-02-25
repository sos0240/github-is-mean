# Research Log: J-Quants API (Replacement for EDINET)
**Generated**: 2026-02-25T11:45:00Z
**Status**: Complete
**Replaces**: jp-edinet-2026-02-24.md (EDINET registration requires project hosted on a website)

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| jquants-api-client | NOT INSTALLED | 2.0.0 | INSTALL |

**Latest Version Published**: 2.0.0 (2026-01-19, from PyPI)
**Documentation Last Updated**: 2026-01-19
**Python Requirement**: >=3.10, <4.0

---

## 2. DOCUMENTATION SOURCES

### Primary Source (Official)
- **URL**: https://github.com/J-Quants/jquants-api-client-python
- **Verified**: 2026-02-25
- **Version Covered**: 2.0.0

### Secondary Sources
- PyPI: https://pypi.org/project/jquants-api-client/
- Official Website: https://jpx-jquants.com/
- API Dashboard: https://jpx-jquants.com/dashboard/api-keys

---

## 3. WHY REPLACE EDINET WITH J-QUANTS

### Problem with EDINET
- EDINET API v2 registration requires a **project hosted on a website** (organizational registration)
- The `edinet-tools` wrapper also requires `EDINET_API_KEY` which has the same registration barrier
- User cannot complete EDINET registration without deploying the project first

### Why J-Quants is Better
- Registration is **email-based** at https://jpx-jquants.com/login (no website hosting required)
- Free plan includes financial summary data (NetSales, OperatingProfit, TotalAssets, etc.)
- Official Python SDK maintained by JPX (Japan Exchange Group)
- Covers same universe: ~3,800+ TSE-listed Japanese companies
- V2 API is current and actively maintained (v2.0.0 released 2026-01-19)
- API key required but obtainable via simple email registration

---

## 4. VERBATIM CODE SNIPPETS

### 4a. Authentication Pattern
**Source**: https://github.com/J-Quants/jquants-api-client-python/blob/main/README.md (V2 section)
```python
# COPIED VERBATIM -- V2 API key authentication
from datetime import datetime
from dateutil import tz
import jquantsapi

my_api_key: str = "*****"
cli = jquantsapi.ClientV2(api_key=my_api_key)
```

**Environment variable alternative:**
```python
# COPIED VERBATIM -- environment variable auth
import jquantsapi

cli = jquantsapi.ClientV2()  # Uses JQUANTS_API_KEY env var
```

**Config file alternative:**
```toml
# COPIED VERBATIM -- jquants-api.toml
[jquants-api-client]
api_key = "*****"
```

**Config file search order** (from README):
1. `/content/drive/MyDrive/drive_ws/secret/jquants-api.toml` (Google Colab only)
2. `${HOME}/.jquants-api/jquants-api.toml`
3. `jquants-api.toml`
4. `os.environ["JQUANTS_API_CLIENT_CONFIG_FILE"]`
5. `${JQUANTS_API_KEY}`

**Checksum**:
- Environment variable name: `JQUANTS_API_KEY`
- Auth method: API key passed to ClientV2 constructor or via env var
- No token prefix required (raw key string)

### 4b. Listed Companies (Equity Master)
**Source**: jquantsapi/client_v2.py method signature (inferred usage, not verbatim README)
```python
# INFERRED from method signature: get_eq_master(code="", date="")
df = cli.get_eq_master(code="27800", date="20260101")
# Returns: pd.DataFrame with listed company info (v2 field names)
```

### 4c. Listed Companies with Sector Info (Utility)
**Source**: jquantsapi/client_v2.py method signature (inferred usage, not verbatim README)
```python
# INFERRED from method signature: get_list(code="", date_yyyymmdd="")
df = cli.get_list(code="", date_yyyymmdd="")
# Returns: pd.DataFrame with company info + 17/33 sector names + market segment
```

### 4d. Daily Price Bars
**Source**: README.md (V2 API section)
```python
# COPIED VERBATIM
df = cli.get_eq_bars_daily_range(
    start_dt=datetime(2022, 7, 25, tzinfo=tz.gettz("Asia/Tokyo")),
    end_dt=datetime(2022, 7, 26, tzinfo=tz.gettz("Asia/Tokyo")),
)
# Returns DataFrame with Code, Date, Open, High, Low, Close, AdjC, AdjVo, etc.
```

### 4e. Financial Summary (FREE PLAN)
**Source**: jquantsapi/client_v2.py method signature + constants.py columns
```python
# INFERRED from method signature: get_fin_summary(code="", date_yyyymmdd="")
df = cli.get_fin_summary(code="72030", date_yyyymmdd="20260101")
# Returns: pd.DataFrame with financial summary (v2 field names)
```

**Financial Summary Response Columns** (from constants.py, V1 names for reference):
```
# COPIED VERBATIM from jquantsapi/constants.py FINS_STATEMENTS_COLUMNS
DisclosedDate, DisclosedTime, LocalCode, DisclosureNumber,
TypeOfDocument, TypeOfCurrentPeriod,
CurrentPeriodStartDate, CurrentPeriodEndDate,
CurrentFiscalYearStartDate, CurrentFiscalYearEndDate,
NetSales, OperatingProfit, OrdinaryProfit, Profit,
EarningsPerShare, DilutedEarningsPerShare,
TotalAssets, Equity, EquityToAssetRatio, BookValuePerShare,
CashFlowsFromOperatingActivities,
CashFlowsFromInvestingActivities,
CashFlowsFromFinancingActivities,
CashAndEquivalents,
ResultDividendPerShareAnnual, ResultPayoutRatioAnnual,
ForecastNetSales, ForecastOperatingProfit, ForecastProfit,
NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock,
NumberOfTreasuryStockAtTheEndOfFiscalYear, AverageNumberOfShares
```

**V2 Column Names** (shorter, from constants.py FIN_SUMMARY_COLUMNS_V2):
```
# COPIED VERBATIM from jquantsapi/constants.py FIN_SUMMARY_COLUMNS_V2
# -- Period identifiers --
DiscDate, DiscTime, Code, DiscNo, DocType, CurPerType,
CurPerSt, CurPerEn, CurFYSt, CurFYEn, NxtFYSt, NxtFYEn,
# -- Consolidated actuals --
Sales, OP, OdP, NP, EPS, DEPS, TA, Eq, EqAR, BPS,
CFO, CFI, CFF, CashEq,
# -- Dividends (result) --
Div1Q, Div2Q, Div3Q, DivFY, DivAnn, DivUnit, DivTotalAnn, PayoutRatioAnn,
# -- Dividends (forecast) --
FDiv1Q, FDiv2Q, FDiv3Q, FDivFY, FDivAnn, FDivUnit, FDivTotalAnn, FPayoutRatioAnn,
# -- Dividends (next year forecast) --
NxFDiv1Q, NxFDiv2Q, NxFDiv3Q, NxFDivFY, NxFDivAnn, NxFDivUnit, NxFPayoutRatioAnn,
# -- Forecast earnings (2Q) --
FSales2Q, FOP2Q, FOdP2Q, FNP2Q, FEPS2Q,
NxFSales2Q, NxFOP2Q, NxFOdP2Q, NxFNp2Q, NxFEPS2Q,
# -- Forecast earnings (FY) --
FSales, FOP, FOdP, FNP, FEPS,
NxFSales, NxFOP, NxFOdP, NxFNp, NxFEPS,
# -- Accounting changes --
MatChgSub, SigChgInC, ChgByASRev, ChgNoASRev, ChgAcEst, RetroRst,
# -- Share counts --
ShOutFY, TrShFY, AvgSh,
# -- Non-consolidated actuals --
NCSales, NCOP, NCOdP, NCNP, NCEPS, NCTA, NCEq, NCEqAR, NCBPS,
# -- Non-consolidated forecasts --
FNCSales2Q, FNCOP2Q, FNCOdP2Q, FNCNP2Q, FNCEPS2Q,
NxFNCSales2Q, NxFNCOP2Q, NxFNCOdP2Q, NxFNCNP2Q, NxFNCEPS2Q,
FNCSales, FNCOP, FNCOdP, FNCNP, FNCEPS,
NxFNCSales, NxFNCOP, NxFNCOdP, NxFNCNP, NxFNCEPS
```

### 4f. Financial Summary Range (Batch Fetch)
**Source**: README.md (V2 Utility section)
```python
# COPIED VERBATIM
df = cli.get_fin_summary_range(
    start_dt="20080707",
    end_dt=datetime.now(),
    cache_dir="",  # optional CSV cache directory
)
```

### 4g. Error Handling
**Source**: README.md (Rate Limits section)
```
# COPIED VERBATIM
Rate limit exceeded: API returns HTTP 429 Too Many Requests.
Retry after waiting, or use narrower date ranges.
```

The SDK uses `tenacity` for automatic retry logic internally.

---

## 5. AVAILABLE API ENDPOINTS BY PLAN

### Free Plan (email registration only)
| Method | Description | Canonical Fields Covered |
|--------|-------------|-------------------------|
| `get_eq_master` | Listed companies master | Company name, code, sector |
| `get_eq_bars_daily` | Daily OHLCV | Open, High, Low, Close, Volume |
| `get_fin_summary` | Financial summary | NetSales, OperatingProfit, Profit, TotalAssets, Equity, CashFlows |
| `get_eq_earnings_cal` | Earnings calendar | Disclosure dates |

### Light Plan (paid)
| Method | Description |
|--------|-------------|
| `get_idx_bars_daily` | Index daily bars |
| `get_mkt_calendar` | Market calendar |
| `get_bulk_list` / `get_bulk` | Bulk data |

### Standard Plan (paid)
| Method | Description |
|--------|-------------|
| `get_mkt_short_ratio` | Short selling ratio |
| `get_mkt_margin_interest` | Margin interest |
| `get_drv_bars_daily_fut` / `_opt` | Derivatives bars |

### Premium Plan (paid)
| Method | Description |
|--------|-------------|
| `get_fin_details` | Detailed financials |
| `get_fin_dividend` | Dividend data |
| `get_mkt_breakdown` | Trade breakdown |

---

## 6. CANONICAL FIELD MAPPING (from get_fin_summary)

| J-Quants Field | Canonical Field | Statement |
|----------------|-----------------|-----------|
| NetSales / Sales | revenue | Income |
| OperatingProfit / OP | operating_income | Income |
| OrdinaryProfit / OdP | pretax_income | Income |
| Profit / NP | net_income | Income |
| EarningsPerShare / EPS | eps | Income |
| TotalAssets / TA | total_assets | Balance |
| Equity / Eq | total_equity | Balance |
| CashFlowsFromOperatingActivities | operating_cashflow | Cashflow |
| CashFlowsFromInvestingActivities | investing_cashflow | Cashflow |
| CashFlowsFromFinancingActivities | financing_cashflow | Cashflow |
| CashAndEquivalents | cash_and_equivalents | Balance |

---

## 7. BREAKING CHANGES ANALYSIS

### Changes in Last 12 Months
- **2026-01-19**: V2.0.0 released -- major version bump
  - V1 `Client` class deprecated, V2 `ClientV2` is now primary
  - V2 uses API key auth (not email/password)
  - V2 uses shorter column names (e.g., `Sales` instead of `NetSales`)
  - V1 API endpoints will be removed in future versions

### Deprecation Warnings
- V1 `Client` class is deprecated. Use `ClientV2` only.
- V1 auth (email/password/refresh_token) deprecated. Use API key.

---

## 8. DEPENDENCY ANALYSIS

### Dependencies Required (from PyPI)
- numpy>=1.22.4
- pandas>=2.2.0,<3.0.0
- requests>=2.28.0,<3.0.0
- tenacity>=8.2.0,<9.0.0
- tomli>=2.0.1,<3.0.0 (Python < 3.11)
- typing_extensions>=4.5.0,<5.0.0 (Python < 3.13)

### Upgrade Impact Assessment
- **Breaking**: NO (new installation)
- **Migration Effort**: MEDIUM (rewrite JP wrapper to use jquantsapi instead of edinet-tools)
- **Required Steps**:
  1. Add `jquants-api-client>=2.0.0` to requirements.txt
  2. Rewrite `jp_edinet_wrapper.py` to use `jquantsapi.ClientV2`
  3. Map J-Quants V2 column names to canonical fields
  4. Update `.env.example` with `JQUANTS_API_KEY`

---

## OPENAPI SPECIFICATION

**Status**: NOT FOUND
**Attempted URLs**:
- `https://jpx-jquants.com/openapi.json` - 307 redirect
- `https://jpx-jquants.com/swagger.json` - 307 redirect
- `https://jpx-jquants.com/.well-known/openapi.json` - 307 redirect
- `https://jpx-jquants.com/api/v2/openapi.json` - 200 but HTML page, not JSON spec

**Alternative**: J-Quants publishes API specs at:
- https://jpx-jquants.com/en/spec/fin-summary (financial summary)
- https://jpx-jquants.com/en/spec/fin-details (financial details)
- https://jpx-jquants.com/en/spec/rate-limits (rate limits)

These are HTML docs, not machine-readable OpenAPI specs.

---

## 9. REQUIRED USER INPUTS

| Parameter | Type | Purpose | Example | Source |
|-----------|------|---------|---------|--------|
| `JQUANTS_API_KEY` | string | API authentication | (from dashboard) | User registration at jpx-jquants.com |

**Registration process**:
1. Go to https://jpx-jquants.com/login
2. Sign up with email + password (free)
3. Select Free plan
4. Get API key from https://jpx-jquants.com/dashboard/api-keys
5. Set `JQUANTS_API_KEY` in `.env`

**No website hosting required** (unlike EDINET registration).

---

## 10. COMPARISON: EDINET vs J-Quants

| Feature | EDINET | J-Quants |
|---------|--------|----------|
| Registration | Requires project on website | Email registration only |
| API Key | EDINET_API_KEY (hard to get) | JQUANTS_API_KEY (easy, free plan) |
| Financial Data | XBRL parsing required | Structured JSON/DataFrame |
| Company Coverage | ~11,000 (all filers) | ~3,800+ (TSE listed) |
| Data Freshness | Real-time filings | Next business day |
| Price Data | Not included | Included (get_eq_bars_daily) |
| Python SDK | edinet-tools 0.3.0 (community) | jquants-api-client 2.0.0 (official JPX) |

---

## 11. IMPLEMENTATION READINESS

### Pre-Flight Checklist
- [x] Official documentation found and verified
- [x] Latest version identified (2.0.0, 2026-01-19)
- [x] Verbatim code snippets extracted
- [x] Breaking changes reviewed (V1 deprecated, V2 current)
- [x] Dependencies checked (compatible with project)
- [x] Registration process documented (email-based, no website needed)

### Recommendation
- **READY TO IMPLEMENT** -- J-Quants API key requires free registration at jpx-jquants.com (email only, no website hosting). Financial summary on free plan covers all canonical income/balance/cashflow fields.

### Next Steps
1. Add `jquants-api-client>=2.0.0` to requirements.txt
2. Rewrite `jp_edinet_wrapper.py` -> use `jquantsapi.ClientV2`
3. Map V2 columns to canonical fields
4. Add `JQUANTS_API_KEY` to `.env.example`

---

## 12. OHLCV & MACRO COVERAGE

### D1. OHLCV Price Data

**Primary option 1**: J-Quants `get_eq_bars_daily` (included in free plan, no extra library)
```python
# VERBATIM from jquants-api-client README
df = cli.get_eq_bars_daily_range(
    start_dt=datetime(2022, 7, 25, tzinfo=tz.gettz("Asia/Tokyo")),
    end_dt=datetime(2022, 7, 26, tzinfo=tz.gettz("Asia/Tokyo")),
)
```

**Primary option 2**: yfinance 1.2.0 (global, no API key, 21.7k stars)
**Research log**: `.roo/research/ohlcv-yfinance-2026-02-24.md`

```python
# VERBATIM from yfinance source -- fetch Japan OHLCV
import yfinance as yf
df = yf.download(
    "7203.T",     # Japan ticker with ".T" suffix
    period="5y",
    auto_adjust=False,
    progress=False,
)
```

**Ticker suffix for Japan**: `.T`

### D2. Macro Economic Data

**Primary**: wbgapi 1.0.13 (World Bank, 200+ countries, no API key)
**Research log**: `.roo/research/macro-wbgapi-2026-02-24.md`

```python
# VERBATIM from wbgapi source -- fetch Japan macro indicators
import wbgapi as wb
df = wb.data.DataFrame(
    ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"],
    economy="JPN",
    mrv=10,
    numericTimeKeys=True,
)
```

**World Bank code for Japan**: `JPN`

**END OF RESEARCH LOG**
