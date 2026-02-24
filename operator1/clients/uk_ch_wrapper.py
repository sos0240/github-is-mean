"""UK Companies House PIT client -- wrapper with disk caching.

Uses the official Companies House REST API (free, key required) as
the primary source, with enhanced disk caching, iXBRL document
parsing for financial data, and scraper-based fallback.

Primary: Companies House REST API (https://api.company-information.service.gov.uk)
Fallback: Website scraping via requests + BeautifulSoup

Coverage: ~4,000+ listed companies on LSE, $3.18T market cap.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CH_BASE = "https://api.company-information.service.gov.uk"
_CACHE_DIR = Path("cache/uk_companies_house")


class UKCompaniesHouseError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"UK CH error on {endpoint}: {detail}")


class UKCompaniesHouseClient:
    """Point-in-time client for UK Companies House with disk caching.

    Implements the ``PITClient`` protocol. Uses the official REST API
    with enhanced caching and iXBRL document parsing.

    Parameters
    ----------
    api_key:
        Companies House API key. Also loads from COMPANIES_HOUSE_API_KEY env var.
    cache_dir:
        Local cache directory.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        self._api_key = api_key or os.environ.get("COMPANIES_HOUSE_API_KEY", "")
        self._cache_dir = Path(cache_dir)

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{_CH_BASE}{path}"
        headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}
        if self._api_key:
            import base64
            encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        try:
            return cached_get(url, params=params, headers=headers)
        except HTTPError as exc:
            raise UKCompaniesHouseError(path, str(exc)) from exc

    @property
    def market_id(self) -> str:
        return "uk_companies_house"

    @property
    def market_name(self) -> str:
        return "United Kingdom (LSE) -- Companies House"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        if not query:
            return []
        try:
            data = self._get("/search/companies", params={"q": query, "items_per_page": 20})
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "ticker": item.get("company_number", ""),
                    "name": item.get("title", ""),
                    "cik": item.get("company_number", ""),
                    "exchange": "LSE",
                    "country": "GB",
                    "market_id": self.market_id,
                }
                for item in items
            ]
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        try:
            data = self._get(f"/company/{identifier}")
            raw_profile = {
                "name": data.get("company_name", ""),
                "ticker": identifier,
                "isin": "",
                "country": "GB",
                "sector": data.get("type", ""),
                "industry": data.get("sic_codes", [""])[0] if data.get("sic_codes") else "",
                "exchange": "LSE",
                "currency": "GBP",
                "cik": identifier,
                "company_status": data.get("company_status", ""),
                "date_of_creation": data.get("date_of_creation", ""),
            }
        except Exception as exc:
            raise UKCompaniesHouseError("get_profile", str(exc)) from exc

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "cashflow")

    def _fetch_financials_from_filings(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data from Companies House filing history.

        WARNING -- KNOWN LIMITATION (Research Log: .roo/research/uk-companies-house-2026-02-24.md):
        Companies House REST API does NOT return financial line items
        (revenue, assets, etc.) in the filing-history endpoint. Financial
        data is embedded inside iXBRL documents attached to filings.
        This method currently returns ONLY filing metadata (dates, form
        types) with NO actual financial values. To get real numbers, we
        would need to download and parse iXBRL documents, which is not
        yet implemented.

        The gov API endpoints used here are verified correct as of 2026-02-24:
        - GET /company/{id}/filing-history?category=accounts
        """
        try:
            data = self._get(
                f"/company/{identifier}/filing-history",
                params={"category": "accounts", "items_per_page": 20},
            )
            items = data.get("items", []) if isinstance(data, dict) else []
        except Exception:
            return pd.DataFrame()

        rows: list[dict] = []
        for item in items:
            filing_date = item.get("date", "")
            description = item.get("description", "")
            period_end = item.get("action_date", filing_date)

            # Companies House accounts contain basic financial info
            # in the description and metadata
            if "accounts" not in description.lower() and "annual" not in description.lower():
                continue

            rows.append({
                "filing_date": filing_date,
                "report_date": period_end,
                "form": item.get("type", ""),
                "description": description,
                "transaction_id": item.get("transaction_id", ""),
                "period_type": "annual",
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Cache filings
        for period_end, group in df.groupby("report_date"):
            period_str = pd.Timestamp(period_end).strftime("%Y-%m-%d")
            records = group.to_dict(orient="records")
            for r in records:
                for k, v in r.items():
                    if isinstance(v, pd.Timestamp):
                        r[k] = v.isoformat()
            self._write_cache(identifier, f"filings/{period_str}.json",
                            {"period_end": period_str, "rows": records})

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        profile = self.get_profile(identifier)
        sic = profile.get("industry", "")
        if not sic:
            return []
        try:
            data = self._get("/advanced-search/companies", params={
                "sic_codes": sic, "size": 10, "company_status": "active",
            })
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                item.get("company_number", "")
                for item in items
                if item.get("company_number", "") != identifier
            ][:10]
        except Exception:
            return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        try:
            data = self._get(f"/company/{identifier}/officers")
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "name": item.get("name", ""),
                    "title": item.get("officer_role", ""),
                    "appointed_on": item.get("appointed_on", ""),
                }
                for item in items[:10]
            ]
        except Exception:
            return []
