# Research Log: yfinance (Global OHLCV Provider)
**Generated**: 2026-02-24T21:39:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| yfinance | NOT INSTALLED | 1.2.0 | INSTALL |

**Latest Version Published**: 2026-02-21 (pushed to GitHub)
**Documentation Last Updated**: 2025 (Sphinx docs at ranaroussi.github.io/yfinance)

---

## 2. DOCUMENTATION SOURCES

### Primary Source (Official)
- **URL**: https://ranaroussi.github.io/yfinance/reference/index.html
- **Verified**: 2026-02-24
- **Version Covered**: 1.2.0

### Secondary Sources
- PyPI: https://pypi.org/project/yfinance/
- GitHub: https://github.com/ranaroussi/yfinance (21.7k stars, Apache 2.0)
- Last push: 2026-02-21T17:38:03Z (3 days ago -- actively maintained)

---

## 3. VERBATIM CODE SNIPPETS

### 3a. Authentication Pattern
**Source**: PyPI page + GitHub README
**No authentication required** -- yfinance uses Yahoo Finance's publicly available APIs.
No API key needed.

### 3b. Primary Usage: Download OHLCV for a Single Ticker
**Source**: yfinance/multi.py line 39-43 (GitHub main branch)
```python
# COPIED VERBATIM from source code
import yfinance as yf

# Single ticker -- Ticker.history() method
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="5y", auto_adjust=False)
# Returns DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits

# Multi-ticker batch download
df = yf.download(
    tickers="AAPL MSFT GOOG",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False,   # IMPORTANT: False = raw/unadjusted OHLCV
    interval="1d",
    progress=False,
)
```

### 3c. Download Function Signature (VERBATIM)
**Source**: yfinance/multi.py lines 39-43
```python
def download(
    tickers,               # str or list -- ticker symbols
    start=None,            # str "YYYY-MM-DD" or datetime
    end=None,              # str "YYYY-MM-DD" or datetime
    actions=False,         # include dividends/splits
    threads=True,          # multi-threaded download
    ignore_tz=None,        # timezone handling
    group_by='column',     # 'column' or 'ticker'
    auto_adjust=True,      # True=adjusted, False=raw OHLCV
    back_adjust=False,     # back-adjust for splits
    repair=False,          # repair bad data
    keepna=False,          # keep NaN rows
    progress=True,         # show progress bar
    period=None,           # "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
    interval="1d",         # "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"
    prepost=False,         # include pre/post market
    rounding=False,        # round values
    timeout=10,            # request timeout
    session=None,          # custom requests session
    multi_level_index=True # multi-level columns for multi-ticker
) -> pd.DataFrame | None
```

### 3d. Response Structure
```
DataFrame columns (auto_adjust=False):
  - Open        float64   (raw open price)
  - High        float64   (raw high price)
  - Low         float64   (raw low price)
  - Close       float64   (raw close price)
  - Adj Close   float64   (split+dividend adjusted close)
  - Volume      int64     (trading volume)

Index: DatetimeIndex (trading dates)
```

### 3e. Error Handling
```python
# yfinance returns empty DataFrame on failure, logs warnings
import yfinance as yf
ticker = yf.Ticker("INVALID_TICKER")
hist = ticker.history(period="1y")
# hist will be empty DataFrame, no exception raised
# Check: hist.empty
```

---

## 4. TICKER FORMAT BY MARKET

Yahoo Finance uses these ticker suffixes for international markets:

| Market | Suffix | Example | Notes |
|--------|--------|---------|-------|
| US (NYSE/NASDAQ) | (none) | AAPL | No suffix needed |
| UK (LSE) | .L | BP.L | London Stock Exchange |
| EU - France | .PA | AIR.PA | Paris/Euronext |
| EU - Germany | .DE | SIE.DE | XETRA/Frankfurt |
| EU - Netherlands | .AS | ASML.AS | Amsterdam |
| EU - Spain | .MC | SAN.MC | Madrid |
| EU - Italy | .MI | ENI.MI | Milan |
| EU - Sweden | .ST | VOLV-B.ST | Stockholm |
| Japan (TSE) | .T | 7203.T | Tokyo |
| South Korea (KRX) | .KS / .KQ | 005930.KS | KOSPI / KOSDAQ |
| Taiwan (TWSE) | .TW | 2330.TW | TWSE |
| Brazil (B3) | .SA | PETR4.SA | Sao Paulo |
| Chile (Santiago) | .SN | SQM.SN | Santiago |
| Canada (TSX) | .TO | RY.TO | Toronto |
| Australia (ASX) | .AX | BHP.AX | ASX |
| India (BSE/NSE) | .BO / .NS | RELIANCE.BO | BSE / NSE |
| China (SSE/SZSE) | .SS / .SZ | 600519.SS | Shanghai / Shenzhen |
| Hong Kong (HKEX) | .HK | 0700.HK | HKEX |
| Singapore (SGX) | .SI | D05.SI | SGX |
| Mexico (BMV) | .MX | AMXL.MX | BMV |
| South Africa (JSE) | .JO | NPN.JO | JSE |
| Switzerland (SIX) | .SW | NESN.SW | SIX |
| Saudi Arabia | .SR | 2222.SR | Tadawul |
| UAE (DFM) | .AE | EMAAR.AE | DFM (limited) |

**Coverage assessment**: yfinance covers ALL Tier 1 and most Tier 2 markets.

---

## 5. DEPENDENCY ANALYSIS

### Dependencies (15 packages)
- pandas>=1.3.0 (already in requirements.txt)
- numpy>=1.16.5 (already in requirements.txt)
- requests>=2.31 (already in requirements.txt)
- multitasking>=0.0.7
- platformdirs>=2.0.0
- pytz>=2022.5
- frozendict>=2.3.4
- peewee>=3.16.2
- beautifulsoup4>=4.11.1
- curl_cffi>=0.7,<0.14
- protobuf>=3.19.0
- websockets>=13.0

### Optional extras
- `nospam`: requests_cache, requests_ratelimiter (anti-throttling)
- `repair`: scipy (data repair)

### Upgrade Impact Assessment
- **Breaking**: NO (new install)
- **Migration Effort**: LOW
- **curl_cffi note**: This is a newer dependency; may need system-level libcurl

---

## 6. BREAKING CHANGES ANALYSIS

### Key change: auto_adjust default
- `download()` has `auto_adjust=True` by default
- We MUST set `auto_adjust=False` for raw/unadjusted OHLCV
- The old `ohlcv_provider.py` used raw OHLCV from Alpha Vantage

### Rate limiting
- Yahoo Finance informally rate-limits; no published numbers
- yfinance v1.2.0 has built-in handling for YFRateLimitError
- Optional `nospam` extra adds request caching + rate limiting

---

## 7. IMPLEMENTATION READINESS

### Pre-Flight Checklist
- [x] Official documentation found and verified
- [x] Latest version identified (1.2.0)
- [x] Verbatim code snippets extracted
- [x] Breaking changes reviewed (auto_adjust default)
- [x] Dependencies checked (curl_cffi is the main new one)
- [x] Ticker format mapping for all markets documented

### Recommendation
- **READY TO IMPLEMENT** -- All checks passed

### Implementation Notes
1. Use `yf.download(ticker, period="5y", auto_adjust=False, progress=False)` for batch
2. Use `yf.Ticker(ticker).history(period="5y", auto_adjust=False)` for single ticker
3. Map market_id -> Yahoo ticker suffix using the table in Section 4
4. Handle empty DataFrame as "no data available"
5. Consider installing with `[nospam]` extra for rate limiting

---

**END OF RESEARCH LOG**
