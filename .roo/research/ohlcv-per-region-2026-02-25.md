# Research Log: Per-Region OHLCV Wrappers
**Generated**: 2026-02-25T12:29:00Z
**Status**: Complete
**Strategy**: Per-region primary wrappers + yfinance as global fallback

---

## Architecture

```
OHLCV Request for {market_id}
    |
    v
[Per-region primary wrapper] -- try first
    |
    | (if fails or not installed)
    v
[yfinance fallback] -- global coverage, no key needed
```

---

## Per-Region OHLCV Wrappers

### 1. Korea (KR) -- pykrx
| Field | Value |
|-------|-------|
| Package | pykrx 1.2.4 |
| PyPI | https://pypi.org/project/pykrx/ |
| GitHub | sharebook-kr/pykrx (905 stars) |
| Key Required | **NO** |
| Python | >=3.10 |
| Verified | 2026-02-25 (PyPI fetch) |

**Verbatim usage** (from README):
```python
from pykrx import stock
df = stock.get_market_ohlcv("20220720", "20220810", "005930")
# Returns: DataFrame with date, open, high, low, close, volume, trade_value
```

**Fallback**: yfinance with `.KS` suffix (KOSPI) or `.KQ` (KOSDAQ)

---

### 2. Taiwan (TW) -- twstock
| Field | Value |
|-------|-------|
| Package | twstock 1.4.0 |
| PyPI | https://pypi.org/project/twstock/ |
| GitHub | mlouielu/twstock (1,291 stars) |
| Key Required | **NO** |
| Python | >=3.5 |
| Verified | 2026-02-25 (PyPI fetch) |

**Verbatim usage** (from README):
```python
import twstock
s = twstock.Stock('2330')
# s.price, s.high, s.low, s.open, s.capacity, s.date
s.fetch_from(2020, 1)
```

**Fallback**: yfinance with `.TW` suffix

---

### 3. China (CN) -- akshare
| Field | Value |
|-------|-------|
| Package | akshare 1.18.29 |
| PyPI | https://pypi.org/project/akshare/ |
| GitHub | akfamily/akshare (16,497 stars) |
| Key Required | **NO** |
| Python | >=3.8 |
| Verified | 2026-02-25 (PyPI fetch) |
| Note | Heavy dependency tree (28 packages) |

**Verbatim usage** (from fallback-wrappers-batch research):
```python
import akshare as ak
df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20200101", end_date="20251231")
# Returns: date, open, close, high, low, volume, turnover, amplitude, change_pct, change_amount, turnover_rate
```

**Fallback**: yfinance with `.SS` (Shanghai) or `.SZ` (Shenzhen) suffix

---

### 4. India (IN) -- jugaad-data
| Field | Value |
|-------|-------|
| Package | jugaad-data 0.29 |
| PyPI | https://pypi.org/project/jugaad-data/ |
| GitHub | jugaad-py/jugaad-data (489 stars) |
| Key Required | **NO** |
| Python | >=3.6 |
| Verified | 2026-02-25 (PyPI fetch) |

**Verbatim usage** (from fallback-wrappers-batch research):
```python
from jugaad_data.nse import stock_df
from datetime import date
df = stock_df(symbol="TCS", from_date=date(2020,1,1), to_date=date(2025,12,31), series="EQ")
# Returns: DATE, OPEN, HIGH, LOW, CLOSE, LTP, VOLUME, etc.
```

**Fallback**: yfinance with `.NS` (NSE) or `.BO` (BSE) suffix

---

### 5. Japan (JP) -- J-Quants (already implemented)
| Field | Value |
|-------|-------|
| Package | jquants-api-client 2.0.0 |
| Key Required | **YES** (JQUANTS_API_KEY) |
| Note | Already implemented in jp_jquants_wrapper.py |

**Usage**: `cli.get_eq_bars_daily_range(start_dt, end_dt)`
**Fallback**: yfinance with `.T` suffix

---

### 6-20. All Other Markets -- yfinance (primary AND only)

These markets don't have specific free wrappers. yfinance is both primary and fallback:

| Market | yfinance Suffix | Example |
|--------|----------------|---------|
| US | (none) | AAPL |
| UK | .L | BP.L |
| EU (generic) | .PA/.DE/.AS | TTE.PA |
| France | .PA | TTE.PA |
| Germany | .DE | SAP.DE |
| Brazil | .SA | VALE3.SA |
| Chile | .SN | SQM-B.SN |
| Canada | .TO | RY.TO |
| Australia | .AX | BHP.AX |
| Hong Kong | .HK | 0005.HK |
| Singapore | .SI | D05.SI |
| Mexico | .MX | AMXL.MX |
| South Africa | .JO | AGL.JO |
| Switzerland | .SW | NESN.SW |
| Saudi Arabia | .SR | 2222.SR |
| UAE | .AE | EMAAR.AE |

---

## Global Fallback: yfinance

| Field | Value |
|-------|-------|
| Package | yfinance 1.2.0 |
| Key Required | **NO** |
| Research | .roo/research/ohlcv-yfinance-2026-02-24.md |

```python
import yfinance as yf
df = yf.download(ticker, period="5y", auto_adjust=False, progress=False)
```

**END OF RESEARCH LOG**
