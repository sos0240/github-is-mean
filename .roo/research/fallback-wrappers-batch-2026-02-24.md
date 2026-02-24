# Research Log: Fallback Wrappers (Batch Research)
**Generated**: 2026-02-24T21:42:00Z
**Status**: Complete

These are fallback/complement libraries for specific markets where yfinance or wbgapi may have gaps.

---

## B2: pandas-datareader (Macro complement)

| Field | Value |
|-------|-------|
| PyPI | pandas-datareader 0.10.0 |
| GitHub | pydata/pandas-datareader (3,160 stars) |
| License | BSD |
| Last Push | 2025-04-03 |
| Status | Stable but less actively maintained |

### Key API
```python
import pandas_datareader as pdr

# World Bank macro data
df = pdr.wb.download(indicator='NY.GDP.MKTP.CD', country='US', start=2015, end=2025)

# OECD data
df = pdr.DataReader('GDP', 'oecd')

# Eurostat
df = pdr.DataReader('tps00001', 'eurostat')
```

### Assessment
- Overlaps with wbgapi for World Bank data
- Adds OECD and Eurostat access
- **RECOMMENDATION**: Optional complement -- wbgapi is preferred for World Bank

---

## C5: pykrx (Korea OHLCV)

| Field | Value |
|-------|-------|
| PyPI | pykrx 1.2.4 |
| GitHub | sharebook-kr/pykrx (905 stars) |
| License | MIT |
| Last Push | 2026-02-20 (4 days ago) |
| Status | Actively maintained |

### Key API
```python
from pykrx import stock

# OHLCV for Samsung (005930) 
df = stock.get_market_ohlcv_by_date("20200101", "20251231", "005930")
# Returns: date, open, high, low, close, volume, trade_value

# Company list
tickers = stock.get_market_ticker_list("20250101", market="KOSPI")

# Fundamental data
df = stock.get_market_fundamental_by_date("20200101", "20251231", "005930")
# Returns: BPS, PER, PBR, EPS, DIV, DPS
```

### Assessment
- Direct scraping of KRX (Korean Exchange)
- No API key needed
- yfinance also covers Korea (.KS/.KQ suffix)
- **RECOMMENDATION**: Optional fallback for Korea if yfinance fails

---

## C7: twstock (Taiwan OHLCV)

| Field | Value |
|-------|-------|
| PyPI | twstock 1.4.0 |
| GitHub | mlouielu/twstock (1,291 stars) |
| License | Unlicensed |
| Last Push | 2025-05-09 |
| Status | Moderately maintained |

### Key API
```python
import twstock

# TSMC (2330) historical data
stock = twstock.Stock('2330')
# stock.price     -- list of close prices
# stock.high      -- list of high prices
# stock.low       -- list of low prices
# stock.open      -- list of open prices
# stock.capacity  -- list of volumes
# stock.date      -- list of dates

# Fetch specific month
stock.fetch_from(2020, 1)  # from 2020 January
```

### Assessment
- Direct TWSE data, no API key
- yfinance also covers Taiwan (.TW suffix)
- **RECOMMENDATION**: Optional fallback for Taiwan if yfinance fails

---

## D1: akshare (China OHLCV + Macro)

| Field | Value |
|-------|-------|
| PyPI | akshare 1.18.28 |
| GitHub | akfamily/akshare (16,497 stars) |
| License | MIT |
| Last Push | 2026-02-24 (today!) |
| Status | Very actively maintained |
| Dependencies | 28 (heavy) |

### Key API
```python
import akshare as ak

# China A-shares OHLCV (daily)
df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20200101", end_date="20251231")
# Returns: date, open, close, high, low, volume, turnover, amplitude, change_pct, change_amount, turnover_rate

# Macro data -- China GDP
df = ak.macro_china_gdp()

# Macro data -- China CPI
df = ak.macro_china_cpi()
```

### Assessment
- Comprehensive coverage of Chinese markets (SSE/SZSE)
- Also provides Chinese macro data
- Heavy dependency tree (28 packages)
- yfinance covers China (.SS/.SZ) but may have gaps
- **RECOMMENDATION**: Required for China market -- yfinance China coverage is inconsistent

---

## D3: jugaad-data (India OHLCV)

| Field | Value |
|-------|-------|
| PyPI | jugaad-data 0.29 |
| GitHub | jugaad-py/jugaad-data (489 stars) |
| License | YOLO |
| Last Push | 2025-11-25 |
| Status | Moderately maintained |

### Key API
```python
from jugaad_data.nse import stock_df
from datetime import date

# Reliance Industries OHLCV
df = stock_df(
    symbol="RELIANCE",
    from_date=date(2020, 1, 1),
    to_date=date(2025, 12, 31),
    series="EQ",
)
# Returns DataFrame with: DATE, OPEN, HIGH, LOW, CLOSE, LTP, VOLUME, etc.
```

### Assessment
- Direct NSE data scraping
- yfinance covers India (.NS/.BO suffix)
- **RECOMMENDATION**: Optional fallback for India if yfinance fails

---

## E6: python-bcb (Brazil Macro)

| Field | Value |
|-------|-------|
| PyPI | python-bcb 0.3.3 |
| GitHub | wilsonfreitas/python-bcb (107 stars) |
| License | Not specified |
| Last Push | 2025-04-21 |
| Status | Moderately maintained |

### Key API
```python
from bcb import sgs

# SELIC rate (series 432)
df = sgs.get({"selic": 432}, start="2020-01-01")

# IPCA inflation (series 433)
df = sgs.get({"ipca": 433}, start="2020-01-01")

# GDP (series 4380)
df = sgs.get({"gdp": 4380}, start="2020-01-01")

# Multiple series at once
df = sgs.get({
    "selic": 432,
    "ipca": 433,
    "gdp": 4380,
    "usd_brl": 1,
}, start="2020-01-01")
```

### Assessment
- Direct BCB (Brazilian Central Bank) data
- Higher frequency than World Bank (monthly/daily vs annual)
- wbgapi covers Brazil for annual macro
- **RECOMMENDATION**: Optional complement for Brazil if higher-frequency macro data needed

---

## FINAL RECOMMENDATIONS

### Must-have (install immediately):
1. **yfinance>=1.2.0** -- Global OHLCV for all markets
2. **wbgapi>=1.0.13** -- Global macro for all countries

### Recommended (install for better coverage):
3. **akshare>=1.18** -- China market (yfinance China coverage unreliable)
4. **pykrx>=1.2** -- Korea fallback (Korean exchange direct data)

### Optional (install only if gaps found):
5. **twstock>=1.4** -- Taiwan fallback
6. **jugaad-data>=0.29** -- India fallback
7. **python-bcb>=0.3** -- Brazil higher-frequency macro
8. **pandas-datareader>=0.10** -- OECD/Eurostat complement

---

**END OF RESEARCH LOG**
