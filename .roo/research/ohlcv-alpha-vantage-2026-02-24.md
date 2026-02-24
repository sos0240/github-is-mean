# Research Log: Alpha Vantage (OHLCV + News Sentiment)
**Generated**: 2026-02-24T15:40:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

No library -- direct HTTP to Alpha Vantage API.

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER
# =====================================================================

**No unofficial wrapper library used.** Direct HTTP calls to Alpha Vantage REST API.

---

# =====================================================================
# PART B: GOVERNMENT/PUBLIC API -- Alpha Vantage
# =====================================================================

## B1. Documentation Source

**Source**: https://www.alphavantage.co/documentation/ (fetched 2026-02-24)
**Key required**: Yes (free tier: register at alphavantage.co)

## B2. Verbatim -- TIME_SERIES_DAILY Endpoint (OHLCV)

**Source**: Alpha Vantage docs, used by `ohlcv_provider.py`
```
GET https://www.alphavantage.co/query
Params:
  function=TIME_SERIES_DAILY
  symbol={ticker}
  outputsize=full
  apikey={key}
```

Response keys (from our wrapper parsing):
```python
# From ohlcv_provider.py
"Time Series (Daily)": {
    "2026-02-21": {
        "1. open": "...",
        "2. high": "...",
        "3. low": "...",
        "4. close": "...",
        "5. volume": "..."
    }
}
```

## B3. Verbatim -- NEWS_SENTIMENT Endpoint

**Source**: https://www.alphavantage.co/documentation/ (NEWS_SENTIMENT section, fetched 2026-02-24)
```
# COPIED VERBATIM FROM AV DOCS
API Parameters
  Required: function
    The function of your choice. In this case, function=NEWS_SENTIMENT
  Optional: tickers
    The stock/crypto/forex symbols of your choice. For example:
    tickers=IBM will filter for articles that mention the IBM ticker;
    tickers=COIN,CRYPTO:BTC,FOREX:USD will filter for articles that
    simultaneously mention Coinbase (COIN), Bitcoin (CRYPTO:BTC), and
    US Dollar (FOREX:USD) in their content.
  Optional: topics
    The news topics of your choice. For example:
    topics=technology will filter for articles that write about the
    technology sector
```

**Used by**: `features/news_sentiment.py`
```python
# VERBATIM from news_sentiment.py _fetch_news_alpha_vantage()
cached_get(
    "https://www.alphavantage.co/query",
    params={
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "limit": 200,
        "apikey": api_key,
    },
)
```

**Response parsing** (verbatim from news_sentiment.py):
```python
feed = data.get("feed", [])
for item in feed:
    title = item.get("title", "")
    pub_date = item.get("time_published", "")   # AV format: YYYYMMDDTHHMMSS
    summary = item.get("summary", "")
```

## B4. Rate Limiting

**Source**: Alpha Vantage free tier documentation
- Free tier: **25 requests per day** (shared across ALL endpoints)
- Our wrapper codes this as `_AV_FREE_TIER_LIMIT = 25` in `ohlcv_provider.py`
- **FIX APPLIED**: `news_sentiment.py` now shares the daily counter via
  `_check_av_daily_limit()` and `_increment_av_counter()`

## B5. Ticker Suffix Mappings

**Source**: ohlcv_provider.py (verified against Alpha Vantage symbol format)
```python
# VERBATIM from ohlcv_provider.py
_AV_SUFFIX = {
    "us_sec_edgar": "",           # AAPL
    "uk_companies_house": ".LON", # BP.LON
    "jp_edinet": ".TYO",          # 7203.TYO
    "kr_dart": ".KRX",            # 005930.KRX
    "tw_mops": ".TPE",            # 2330.TPE
    "br_cvm": ".SAO",             # PETR4.SAO
    "cl_cmf": ".SNX",             # SQM.SNX
}
```
**Note**: NEWS_SENTIMENT uses bare tickers without exchange suffixes.

## B6. Canonical Field Coverage

- **OHLCV**: date, open, high, low, close, volume -- ALL COVERED
- **News Sentiment**: title, publishedDate, summary -- ALL COVERED

---

# =====================================================================
# PART C: SUMMARY
# =====================================================================

| Endpoint | Used By | Status |
|----------|---------|--------|
| TIME_SERIES_DAILY | ohlcv_provider.py | OK -- endpoint verified |
| NEWS_SENTIMENT | features/news_sentiment.py | OK -- endpoint verified from AV docs |

### Fixes Applied
1. Shared rate limit counter between OHLCV and NEWS_SENTIMENT

---

**END OF RESEARCH LOG**
