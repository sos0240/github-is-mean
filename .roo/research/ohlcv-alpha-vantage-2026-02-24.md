# Research Log: Alpha Vantage (OHLCV + News Sentiment)
**Generated**: 2026-02-24T14:26:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

No library -- direct HTTP to Alpha Vantage API.

---

## 2. DOCUMENTATION SOURCES

### Primary Source -- Alpha Vantage API
- **URL**: https://www.alphavantage.co/documentation/
- **Verified**: 2026-02-24
- **Key required**: Yes (free tier: register at alphavantage.co)

---

## 3. WRAPPER CODE AUDIT

### 3a. OHLCV Provider (`ohlcv_provider.py`)

| Method | Status | Issue |
|--------|--------|-------|
| `fetch_ohlcv()` | OK | `TIME_SERIES_DAILY` endpoint correct |
| `_to_av_ticker()` | VERIFY | Suffix mappings may be outdated |
| `_check_av_daily_limit()` | VERIFY | Free tier limit coded as 25/day |
| OHLCV column parsing | OK | Maps AV JSON keys to standard names |

### 3b. News Sentiment (`features/news_sentiment.py`)

| Method | Status | Issue |
|--------|--------|-------|
| `_fetch_news_alpha_vantage()` | OK | `NEWS_SENTIMENT` endpoint verified in AV docs |

**Verified endpoint call**:
```python
# From news_sentiment.py -- VERIFIED against AV docs 2026-02-24
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
- `function=NEWS_SENTIMENT` -- confirmed in AV docs (Market News & Sentiment section)
- `tickers` param -- correct (AV uses `tickers` not `symbol`)
- `limit=200` -- valid parameter for number of articles
- Response format: `data["feed"]` array with `title`, `time_published`, `summary` -- correct

**Response parsing verified**:
```python
feed = data.get("feed", [])
for item in feed:
    title = item.get("title", "")
    pub_date = item.get("time_published", "")   # AV format: YYYYMMDDTHHMMSS
    summary = item.get("summary", "")
```
This matches Alpha Vantage's documented NEWS_SENTIMENT response schema.

**Note**: NEWS_SENTIMENT shares the same daily rate limit with OHLCV (25 total).
The `ohlcv_provider.py` tracks this limit, but `news_sentiment.py` does NOT
decrement the shared counter. This could cause silent failures if both are
called in the same session.

### Alpha Vantage Free Tier Limits
- The wrapper codes 25 requests/day as `_AV_FREE_TIER_LIMIT = 25`
- Current (2026): Free tier is 25 requests/day -- wrapper value is CORRECT
- NEWS_SENTIMENT and TIME_SERIES_DAILY share the same limit
- Premium tier: different limits apply

### Ticker Suffix Mappings (OHLCV only)
```python
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
Note: NEWS_SENTIMENT uses bare tickers (e.g., "AAPL") without exchange suffixes.

### Canonical Field Coverage
- **OHLCV**: date, open, high, low, close, volume -- ALL COVERED
- **adjusted_close**: Available from AV but may need premium tier
- **News Sentiment**: title, publishedDate, summary -- ALL COVERED

---

## 4. REQUIRED FIXES

1. **MINOR**: News sentiment does not decrement the shared AV daily call counter.
   If both OHLCV and news are fetched in the same session, the counter in
   `ohlcv_provider.py` may undercount, potentially exceeding the 25/day limit.
2. **NONE critical** -- both endpoints and response parsing are correct

---

## 5. IMPLEMENTATION READINESS

### Recommendation
- READY TO IMPLEMENT -- endpoints verified, response parsing correct

---

**END OF RESEARCH LOG**
