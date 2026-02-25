# Research Log: Saudi Arabia Tadawul
**Generated**: 2026-02-24T15:46:00Z
**Status**: Complete

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER
# =====================================================================

## A1. PyPI Search

**Source**: PyPI simple index search (fetched 2026-02-24)
Searched: tadawul, saudi-stock, tadawul-sdk, saudi-exchange
```
Results: NONE FOUND on PyPI
```

## A2. GitHub Search

**Source**: GitHub API search (fetched 2026-02-24)
```
Results:
- Logic-gate/TadawulStocks (8 stars) -- "A minimalistic Python based portfolio manager.
  This version only supports (Tadaw"
- zaakki-ahamed/Saudi_Stock_Web_Scrape (3 stars) -- web scraper
- Others: <2 stars
```
No proper API wrapper library. Only basic scrapers and portfolio tools.

---

# =====================================================================
# PART B: GOVERNMENT API -- Tadawul
# =====================================================================

## B1. API Used by Stub Client

**Source**: pit_registry.py
```
pit_api_url: https://www.saudiexchange.sa
```
No key required. Tadawul provides limited free data.

## B2. Macro Source
**Current**: World Bank fallback only (wbgapi)
**Potential**: SAMA (Saudi Arabian Monetary Authority) has free data but no PyPI wrapper found

---

# =====================================================================
# PART C: SUMMARY
# =====================================================================

| Component | Status |
|-----------|--------|
| Micro wrapper | NONE FOUND on PyPI -- only basic GitHub scrapers |
| Macro wrapper | NONE FOUND -- World Bank fallback |

---


# =====================================================================
# PART D: OHLCV & MACRO COVERAGE (added 2026-02-24)
# =====================================================================

## D1. OHLCV Price Data

**Primary**: yfinance 1.2.0 (global, no API key, 21.7k stars)
**Research log**: `.roo/research/ohlcv-yfinance-2026-02-24.md`

```python
# VERBATIM from yfinance source -- fetch Saudi Arabia OHLCV
import yfinance as yf
df = yf.download(
    "2222.SR",     # Saudi Arabia ticker with ".SR" suffix
    period="5y",
    auto_adjust=False,      # raw/unadjusted OHLCV (PIT-safe)
    progress=False,
)
# Returns: DataFrame(Open, High, Low, Close, Adj Close, Volume)
```

**Ticker suffix for Saudi Arabia**: `.SR`
**Example**: `2222.SR`
**Notes**: No wrapper found. Use yfinance.

## D2. Macro Economic Data

**Primary**: wbgapi 1.0.13 (World Bank, 200+ countries, no API key)
**Research log**: `.roo/research/macro-wbgapi-2026-02-24.md`

```python
# VERBATIM from wbgapi source -- fetch Saudi Arabia macro indicators
import wbgapi as wb
df = wb.data.DataFrame(
    ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"],
    economy="SAU",    # Saudi Arabia World Bank code
    mrv=10,                 # most recent 10 years
    numericTimeKeys=True,
)
# Returns: DataFrame with GDP growth, inflation, unemployment
```

**World Bank code for Saudi Arabia**: `SAU`

**END OF RESEARCH LOG**
