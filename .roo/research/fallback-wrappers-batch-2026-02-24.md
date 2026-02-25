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

---

## COMPLETE INPUT REQUIREMENTS SUMMARY (ALL LIBRARIES)

### 1. yfinance -- Global OHLCV
```python
# yf.download() -- VERBATIM from source
yf.download(
    tickers,                # REQUIRED: str or list -- "AAPL" or ["AAPL", "MSFT"]
    start=None,             # optional: str "YYYY-MM-DD" or datetime
    end=None,               # optional: str "YYYY-MM-DD" or datetime
    period=None,            # optional: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
    interval="1d",          # optional: "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"
    auto_adjust=True,       # IMPORTANT: set False for raw OHLCV
    progress=True,          # set False for pipeline use
    threads=True,           # multi-threaded
    timeout=10,             # request timeout seconds
)
# Returns: pd.DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
# Index: DatetimeIndex
# NO API key needed. NO authentication.

# Ticker suffixes by market:
#   US: AAPL       UK: BP.L        France: AIR.PA    Germany: SIE.DE
#   Japan: 7203.T  Korea: 005930.KS  Taiwan: 2330.TW  Brazil: PETR4.SA
#   Chile: SQM.SN  Canada: RY.TO   Australia: BHP.AX  India: RELIANCE.NS
#   China: 600519.SS  HK: 0700.HK  Singapore: D05.SI  Mexico: AMXL.MX
#   S.Africa: NPN.JO  Switzerland: NESN.SW  Saudi: 2222.SR  UAE: EMAAR.AE
```

### 2. wbgapi -- Global Macro
```python
import wbgapi as wb

# wb.data.DataFrame() -- VERBATIM from source
wb.data.DataFrame(
    series,                 # REQUIRED: str or list -- "NY.GDP.MKTP.CD" or ["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG"]
    economy='all',          # optional: str or list -- "USA" or ["USA", "GBR"] (ISO-3 codes)
    time='all',             # optional: str, int, or range -- "YR2020" or range(2015, 2025)
    mrv=None,               # optional: int -- most recent N values (e.g., mrv=10)
    mrnev=None,             # optional: int -- most recent N non-empty values
    skipBlanks=False,       # optional: skip empty observations
    labels=False,           # optional: include dimension names
    skipAggs=False,         # optional: skip aggregate economies
    numericTimeKeys=False,  # optional: use 2020 instead of "YR2020"
    db=None,                # optional: database ID (default: 2 = WDI)
)
# Returns: pd.DataFrame
# Index: economy code
# Columns: time periods (YR2020, YR2021, etc.)
# NO API key needed. NO authentication.

# Key indicator IDs:
#   GDP growth: NY.GDP.MKTP.KD.ZG    Inflation: FP.CPI.TOTL.ZG
#   Interest rate: FR.INR.RINR        Unemployment: SL.UEM.TOTL.ZS
#   Exchange rate: PA.NUS.FCRF        GDP current USD: NY.GDP.MKTP.CD

# Country codes (ISO-2 -> ISO-3 for WB):
#   US->USA  GB->GBR  EU->EMU  FR->FRA  DE->DEU  JP->JPN
#   KR->KOR  TW->(not covered)  BR->BRA  CL->CHL  CA->CAN
#   AU->AUS  IN->IND  CN->CHN  HK->HKG  SG->SGP  MX->MEX
#   ZA->ZAF  CH->CHE  SA->SAU  AE->ARE
```

### 3. pykrx -- Korea OHLCV
```python
from pykrx import stock

# VERBATIM from source
stock.get_market_ohlcv_by_date(
    fromdate,               # REQUIRED: str "YYYYMMDD" -- e.g., "20200101"
    todate,                 # REQUIRED: str "YYYYMMDD" -- e.g., "20251231"
    ticker,                 # REQUIRED: str -- Korean stock code, e.g., "005930" (Samsung)
    freq="d",               # optional: "d" (daily), "m" (monthly), "y" (yearly)
    adjusted=True,          # optional: adjusted prices (set False for raw)
    name_display=False,     # optional: show column names in Korean
)
# Returns: pd.DataFrame with Korean column headers (시가/고가/저가/종가/거래량)
#   meaning: open/high/low/close/volume
# Index: DatetimeIndex
# NO API key needed.
```

### 4. akshare -- China OHLCV + Macro
```python
import akshare as ak

# OHLCV -- VERBATIM from source
ak.stock_zh_a_hist(
    symbol="000001",        # REQUIRED: str -- 6-digit stock code (no suffix)
    period="daily",         # optional: "daily", "weekly", "monthly"
    start_date="19700101",  # optional: str "YYYYMMDD"
    end_date="20500101",    # optional: str "YYYYMMDD"
    adjust="",              # optional: "" (raw/no adjust), "qfq" (forward), "hfq" (backward)
    timeout=None,           # optional: request timeout
)
# Returns: pd.DataFrame with columns (Chinese):
#   日期/开盘/收盘/最高/最低/成交量/成交额/振幅/涨跌幅/涨跌额/换手率
#   meaning: date/open/close/high/low/volume/turnover/amplitude/change_pct/change_amount/turnover_rate
# NO API key needed.

# Macro -- China GDP
ak.macro_china_gdp()  # No arguments needed, returns full history
# Macro -- China CPI
ak.macro_china_cpi()  # No arguments needed
```

### 5. jugaad-data -- India OHLCV
```python
from jugaad_data.nse import stock_df
from datetime import date

# VERBATIM from source
stock_df(
    symbol,                 # REQUIRED: str -- NSE symbol, e.g., "RELIANCE"
    from_date,              # REQUIRED: datetime.date -- e.g., date(2020, 1, 1)
    to_date,                # REQUIRED: datetime.date -- e.g., date(2025, 12, 31)
    series="EQ",            # optional: "EQ" (equity), "BE" (T+2), etc.
)
# Returns: pd.DataFrame with columns: DATE, OPEN, HIGH, LOW, CLOSE, LTP, VOLUME, etc.
# NO API key needed.
```

### 6. python-bcb -- Brazil Macro
```python
from bcb import sgs

# VERBATIM from README
sgs.get(
    codes,                  # REQUIRED: dict -- {"name": series_id} e.g., {"selic": 432}
    start=None,             # optional: str "YYYY-MM-DD" -- e.g., "2020-01-01"
    end=None,               # optional: str "YYYY-MM-DD"
)
# Returns: pd.DataFrame with DatetimeIndex
# NO API key needed.

# Key series IDs:
#   SELIC rate: 432          IPCA inflation: 433
#   GDP: 4380                USD/BRL exchange: 1
#   Unemployment: 24369
```

### 7. twstock -- Taiwan OHLCV
```python
import twstock

# VERBATIM from README
stock = twstock.Stock(code)  # REQUIRED: str -- TWSE stock code, e.g., "2330" (TSMC)
stock.fetch_from(year, month)  # REQUIRED: int, int -- e.g., (2020, 1)

# Properties after fetch:
#   stock.date       -- list of datetime.date
#   stock.open       -- list of float (open prices)
#   stock.high       -- list of float
#   stock.low        -- list of float
#   stock.close      -- list of float
#   stock.capacity   -- list of int (volume)
#   stock.price      -- alias for close
# NO API key needed.
```

### 8. pandas-datareader -- OECD/Eurostat Macro
```python
import pandas_datareader as pdr

# World Bank
pdr.wb.download(
    indicator,              # REQUIRED: str -- e.g., "NY.GDP.MKTP.CD"
    country,                # REQUIRED: str or list -- ISO-2 codes, e.g., "US" or ["US", "GB"]
    start,                  # REQUIRED: int -- start year, e.g., 2015
    end,                    # REQUIRED: int -- end year, e.g., 2025
)
# Returns: pd.DataFrame

# OECD
pdr.DataReader(name, 'oecd')  # name: str -- dataset name, e.g., "GDP"
# NO API key needed for World Bank/OECD/Eurostat.
```

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
